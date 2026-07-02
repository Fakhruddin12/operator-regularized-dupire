"""Operator-regularized inverse calibration for real option-market data.

This module uses the forward-normalized Dupire PDE. It therefore supports an
expiry-dependent forward and discount factor without forcing constant rates or
dividends into the pricing map.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csc_matrix

from src.inverse.linearized_inverse import quote_weights, weighted_rmse
from src.inverse.nonlinear_inverse import (
    nonlinear_objective,
    solve_gauss_newton_step,
)
from src.pricing.forward_normalized_pde import (
    interpolate_normalized_call_prices,
    solve_forward_normalized_dupire,
)


def make_x_local_volatility_function(
    log_variance_surface: np.ndarray,
    maturities: np.ndarray,
    log_moneyness: np.ndarray,
) -> Callable[[np.ndarray, float], np.ndarray]:
    """Create ``sigma(x,T)`` from a gridded log-variance surface.

    Outside the calibration grid, coordinates are clipped to the nearest edge,
    giving flat extrapolation consistent with the regularizer's natural
    boundary behavior.
    """
    surface = np.asarray(log_variance_surface, dtype=float)
    maturity_values = np.asarray(maturities, dtype=float)
    x_values = np.asarray(log_moneyness, dtype=float)

    expected_shape = (maturity_values.size, x_values.size)
    if surface.shape != expected_shape:
        raise ValueError(
            f"log_variance_surface has shape {surface.shape}; "
            f"expected {expected_shape}."
        )
    if np.any(np.diff(maturity_values) <= 0):
        raise ValueError("maturities must be strictly increasing.")
    if np.any(np.diff(x_values) <= 0):
        raise ValueError("log_moneyness must be strictly increasing.")
    if np.any(~np.isfinite(surface)):
        raise ValueError("log_variance_surface contains non-finite values.")

    interpolator = RegularGridInterpolator(
        (maturity_values, x_values),
        surface,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    minimum_maturity = float(maturity_values[0])
    maximum_maturity = float(maturity_values[-1])
    minimum_x = float(x_values[0])
    maximum_x = float(x_values[-1])

    def local_volatility(
        x_query: np.ndarray,
        maturity: float,
    ) -> np.ndarray:
        x_array = np.asarray(x_query, dtype=float)
        clipped_x = np.clip(x_array, minimum_x, maximum_x)
        clipped_maturity = float(
            np.clip(maturity, minimum_maturity, maximum_maturity)
        )
        points = np.column_stack(
            [
                np.full(x_array.size, clipped_maturity),
                clipped_x.reshape(-1),
            ]
        )
        interpolated = interpolator(points).reshape(x_array.shape)
        return np.exp(0.5 * interpolated)

    return local_volatility


def _validate_market_quote_data(
    quote_data: pd.DataFrame,
    price_column: str | None = None,
) -> None:
    """Validate the columns required by the normalized real-market pricer."""
    required = {
        "maturity",
        "log_moneyness",
        "discount_factor",
        "forward",
    }
    if price_column is not None:
        required.add(price_column)

    missing = required.difference(quote_data.columns)
    if missing:
        raise ValueError(
            f"quote_data is missing required columns: {sorted(missing)}"
        )
    if quote_data.empty:
        raise ValueError("quote_data must contain at least one quote.")
    if np.any(quote_data["maturity"].to_numpy(dtype=float) <= 0):
        raise ValueError("all maturities must be positive.")
    if np.any(quote_data["discount_factor"].to_numpy(dtype=float) <= 0):
        raise ValueError("all discount factors must be positive.")
    if np.any(quote_data["forward"].to_numpy(dtype=float) <= 0):
        raise ValueError("all forwards must be positive.")


def price_market_quotes_from_log_variance(
    log_variance_surface: np.ndarray,
    calibration_maturities: np.ndarray,
    calibration_log_moneyness: np.ndarray,
    quote_data: pd.DataFrame,
    pde_x_min: float = -1.0,
    pde_x_max: float = 1.0,
    number_of_pde_x_points: int = 201,
    number_of_time_steps: int = 150,
    theta: float = 0.5,
) -> np.ndarray:
    """Price real-market quotes for one gridded local-volatility surface."""
    _validate_market_quote_data(quote_data)

    local_volatility = make_x_local_volatility_function(
        log_variance_surface=log_variance_surface,
        maturities=calibration_maturities,
        log_moneyness=calibration_log_moneyness,
    )

    maximum_maturity = float(
        quote_data["maturity"].max()
    )
    x_grid, time_grid, normalized_surface = (
        solve_forward_normalized_dupire(
            local_volatility=local_volatility,
            max_maturity=maximum_maturity,
            x_min=pde_x_min,
            x_max=pde_x_max,
            number_of_x_points=number_of_pde_x_points,
            number_of_time_steps=number_of_time_steps,
            theta=theta,
        )
    )

    normalized_prices = interpolate_normalized_call_prices(
        x_grid=x_grid,
        time_grid=time_grid,
        normalized_call_surface=normalized_surface,
        log_moneyness=quote_data["log_moneyness"].to_numpy(dtype=float),
        maturities=quote_data["maturity"].to_numpy(dtype=float),
    )

    scale = (
        quote_data["discount_factor"].to_numpy(dtype=float)
        * quote_data["forward"].to_numpy(dtype=float)
    )
    return np.asarray(normalized_prices, dtype=float) * scale


def finite_difference_market_jacobian(
    reference_log_variance: np.ndarray,
    calibration_maturities: np.ndarray,
    calibration_log_moneyness: np.ndarray,
    quote_data: pd.DataFrame,
    finite_difference_step: float = 1e-3,
    scheme: str = "forward",
    pde_x_min: float = -1.0,
    pde_x_max: float = 1.0,
    number_of_pde_x_points: int = 201,
    number_of_time_steps: int = 150,
    theta: float = 0.5,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the quote-price Jacobian with respect to log variance."""
    reference = np.asarray(reference_log_variance, dtype=float)
    if finite_difference_step <= 0:
        raise ValueError("finite_difference_step must be positive.")
    if scheme not in {"forward", "central"}:
        raise ValueError("scheme must be 'forward' or 'central'.")

    pricing_arguments = dict(
        calibration_maturities=calibration_maturities,
        calibration_log_moneyness=calibration_log_moneyness,
        quote_data=quote_data,
        pde_x_min=pde_x_min,
        pde_x_max=pde_x_max,
        number_of_pde_x_points=number_of_pde_x_points,
        number_of_time_steps=number_of_time_steps,
        theta=theta,
    )

    base_prices = price_market_quotes_from_log_variance(
        log_variance_surface=reference,
        **pricing_arguments,
    )

    number_of_quotes = len(quote_data)
    number_of_parameters = reference.size
    jacobian = np.empty(
        (number_of_quotes, number_of_parameters),
        dtype=float,
    )

    for parameter_index in range(number_of_parameters):
        if verbose and (
            parameter_index == 0
            or (parameter_index + 1) % 20 == 0
            or parameter_index + 1 == number_of_parameters
        ):
            print(
                "Jacobian column "
                f"{parameter_index + 1}/{number_of_parameters}"
            )

        perturbation = np.zeros(number_of_parameters, dtype=float)
        perturbation[parameter_index] = finite_difference_step
        perturbation_surface = perturbation.reshape(
            reference.shape,
            order="C",
        )

        plus_prices = price_market_quotes_from_log_variance(
            log_variance_surface=reference + perturbation_surface,
            **pricing_arguments,
        )

        if scheme == "forward":
            jacobian[:, parameter_index] = (
                plus_prices - base_prices
            ) / finite_difference_step
        else:
            minus_prices = price_market_quotes_from_log_variance(
                log_variance_surface=reference - perturbation_surface,
                **pricing_arguments,
            )
            jacobian[:, parameter_index] = (
                plus_prices - minus_prices
            ) / (2.0 * finite_difference_step)

    return base_prices, jacobian


def run_market_gauss_newton_calibration(
    reference_log_variance: np.ndarray,
    initial_log_variance: np.ndarray,
    calibration_maturities: np.ndarray,
    calibration_log_moneyness: np.ndarray,
    quote_data: pd.DataFrame,
    regularization_matrix: csc_matrix,
    regularization_strength: float,
    observed_price_column: str = "observed_call_price",
    noise_column: str = "noise_standard_deviation",
    finite_difference_step: float = 1e-3,
    jacobian_scheme: str = "forward",
    pde_x_min: float = -1.0,
    pde_x_max: float = 1.0,
    number_of_pde_x_points: int = 201,
    number_of_time_steps: int = 150,
    theta: float = 0.5,
    maximum_iterations: int = 3,
    initial_damping: float = 1e-2,
    damping_increase: float = 10.0,
    damping_decrease: float = 0.3,
    maximum_damping_attempts: int = 6,
    line_search_contraction: float = 0.5,
    minimum_step_scale: float = 1.0 / 64.0,
    maximum_absolute_step: float | None = 1.0,
    relative_objective_tolerance: float = 1e-4,
    relative_step_tolerance: float = 1e-3,
    verbose: bool = False,
) -> dict[str, object]:
    """Run damped Gauss--Newton calibration on real-market training quotes."""
    reference = np.asarray(reference_log_variance, dtype=float)
    current = np.asarray(initial_log_variance, dtype=float).copy()

    if current.shape != reference.shape:
        raise ValueError("initial and reference surfaces must have equal shape.")
    if regularization_matrix.shape != (reference.size, reference.size):
        raise ValueError("regularization_matrix has the wrong shape.")
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be at least 1.")

    _validate_market_quote_data(
        quote_data,
        price_column=observed_price_column,
    )
    observed = quote_data[observed_price_column].to_numpy(dtype=float)
    weights = quote_weights(quote_data, noise_column=noise_column)

    pricing_arguments = dict(
        calibration_maturities=calibration_maturities,
        calibration_log_moneyness=calibration_log_moneyness,
        quote_data=quote_data,
        pde_x_min=pde_x_min,
        pde_x_max=pde_x_max,
        number_of_pde_x_points=number_of_pde_x_points,
        number_of_time_steps=number_of_time_steps,
        theta=theta,
    )

    current_prices = price_market_quotes_from_log_variance(
        log_variance_surface=current,
        **pricing_arguments,
    )
    current_correction = (current - reference).reshape(-1, order="C")
    current_metrics = nonlinear_objective(
        predicted_prices=current_prices,
        observed_prices=observed,
        weights=weights,
        total_correction=current_correction,
        regularization_matrix=regularization_matrix,
        regularization_strength=regularization_strength,
    )

    history = [
        {
            "iteration": 0,
            "objective": current_metrics["objective"],
            "weighted_rmse": current_metrics["weighted_rmse"],
            "weighted_data_misfit": current_metrics["weighted_data_misfit"],
            "regularization_penalty": current_metrics["regularization_penalty"],
            "damping": initial_damping,
            "step_scale": 0.0,
            "step_norm": 0.0,
            "accepted": True,
        }
    ]

    damping = initial_damping
    stop_reason = "maximum_iterations_reached"
    latest_jacobian = None

    for iteration in range(1, maximum_iterations + 1):
        current_prices, jacobian = finite_difference_market_jacobian(
            reference_log_variance=current,
            finite_difference_step=finite_difference_step,
            scheme=jacobian_scheme,
            verbose=verbose,
            **pricing_arguments,
        )
        latest_jacobian = jacobian
        current_correction = (current - reference).reshape(-1, order="C")
        current_metrics = nonlinear_objective(
            predicted_prices=current_prices,
            observed_prices=observed,
            weights=weights,
            total_correction=current_correction,
            regularization_matrix=regularization_matrix,
            regularization_strength=regularization_strength,
        )
        residual = observed - current_prices

        accepted = False
        damping_used = damping

        for _ in range(maximum_damping_attempts):
            step_result = solve_gauss_newton_step(
                jacobian=jacobian,
                residual=residual,
                weights=weights,
                current_total_correction=current_correction,
                regularization_matrix=regularization_matrix,
                regularization_strength=regularization_strength,
                damping=damping_used,
            )
            step = np.asarray(step_result["step"], dtype=float)

            if maximum_absolute_step is not None:
                largest = float(np.max(np.abs(step)))
                if largest > maximum_absolute_step:
                    step *= maximum_absolute_step / largest

            step_scale = 1.0
            while step_scale >= minimum_step_scale:
                candidate = (
                    current.reshape(-1, order="C")
                    + step_scale * step
                ).reshape(current.shape, order="C")
                try:
                    candidate_prices = price_market_quotes_from_log_variance(
                        log_variance_surface=candidate,
                        **pricing_arguments,
                    )
                    candidate_correction = (
                        candidate - reference
                    ).reshape(-1, order="C")
                    candidate_metrics = nonlinear_objective(
                        predicted_prices=candidate_prices,
                        observed_prices=observed,
                        weights=weights,
                        total_correction=candidate_correction,
                        regularization_matrix=regularization_matrix,
                        regularization_strength=regularization_strength,
                    )
                except (ValueError, FloatingPointError, RuntimeError):
                    step_scale *= line_search_contraction
                    continue

                if candidate_metrics["objective"] < current_metrics["objective"]:
                    accepted = True
                    previous_objective = current_metrics["objective"]
                    current = candidate
                    current_prices = candidate_prices
                    current_metrics = candidate_metrics
                    accepted_scale = step_scale
                    accepted_step = step
                    break

                step_scale *= line_search_contraction

            if accepted:
                break
            damping_used = max(1e-12, damping_used * damping_increase)

        if not accepted:
            stop_reason = "no_objective_decrease"
            break

        scaled_step = accepted_scale * accepted_step
        relative_step = float(np.linalg.norm(scaled_step)) / (
            1.0 + float(np.linalg.norm(current.reshape(-1, order="C")))
        )
        relative_improvement = float(
            (previous_objective - current_metrics["objective"])
            / max(1.0, abs(previous_objective))
        )

        history.append(
            {
                "iteration": iteration,
                "objective": current_metrics["objective"],
                "weighted_rmse": current_metrics["weighted_rmse"],
                "weighted_data_misfit": current_metrics["weighted_data_misfit"],
                "regularization_penalty": current_metrics["regularization_penalty"],
                "damping": damping_used,
                "step_scale": accepted_scale,
                "step_norm": float(np.linalg.norm(scaled_step)),
                "accepted": True,
            }
        )

        if verbose:
            print(
                f"Iteration {iteration}: objective={current_metrics['objective']:.6f}, "
                f"weighted RMSE={current_metrics['weighted_rmse']:.6f}, "
                f"step scale={accepted_scale:.4f}"
            )

        damping = max(1e-12, damping_used * damping_decrease)

        if relative_improvement <= relative_objective_tolerance:
            stop_reason = "relative_objective_tolerance"
            break
        if relative_step <= relative_step_tolerance:
            stop_reason = "relative_step_tolerance"
            break

    final_prices = price_market_quotes_from_log_variance(
        log_variance_surface=current,
        **pricing_arguments,
    )

    return {
        "reference_log_variance": reference,
        "initial_log_variance": np.asarray(initial_log_variance, dtype=float),
        "estimated_log_variance": current,
        "correction_surface": current - reference,
        "observed_prices": observed,
        "fitted_prices": final_prices,
        "weights": weights,
        "history": pd.DataFrame(history),
        "stop_reason": stop_reason,
        "final_weighted_rmse": weighted_rmse(
            final_prices,
            observed,
            weights,
        ),
        "latest_jacobian": latest_jacobian,
    }
