"""Weighted linearised inverse calibration for local volatility.

Around a reference log-variance surface ``u0``, the forward pricing map is
approximated by

    F(u0 + h) approximately F(u0) + L h.

The correction is estimated from

    min_h ||W(Lh - d)||^2 + lambda * h.T R h,

where ``d = observed_prices - F(u0)``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

from src.inverse.parameterization import make_local_volatility_function
from src.pricing.local_vol_pde import (
    interpolate_call_prices,
    solve_forward_dupire,
)


def _validate_quote_data(
    quote_data: pd.DataFrame,
    price_column: str | None = None,
) -> None:
    """Check the quote columns required by the forward or inverse calculation."""
    required = {"strike", "maturity"}
    if price_column is not None:
        required.add(price_column)

    missing = required.difference(quote_data.columns)
    if missing:
        raise ValueError(
            f"quote_data is missing required columns: {sorted(missing)}"
        )
    if len(quote_data) == 0:
        raise ValueError("quote_data must contain at least one quote.")
    if np.any(quote_data["strike"].to_numpy(dtype=float) <= 0):
        raise ValueError("all quote strikes must be positive.")
    if np.any(quote_data["maturity"].to_numpy(dtype=float) <= 0):
        raise ValueError("all quote maturities must be positive.")


def price_quotes_from_log_variance(
    log_variance_surface: np.ndarray,
    calibration_maturities: np.ndarray,
    calibration_log_moneyness: np.ndarray,
    quote_data: pd.DataFrame,
    spot: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    strike_max: float | None = None,
    number_of_strike_points: int = 181,
    number_of_time_steps: int = 160,
    theta: float = 0.5,
) -> np.ndarray:
    """Price all quotes for one gridded log-variance surface."""
    _validate_quote_data(quote_data)

    local_volatility = make_local_volatility_function(
        log_variance_surface=log_variance_surface,
        maturities=calibration_maturities,
        log_moneyness=calibration_log_moneyness,
        spot=spot,
        rate=rate,
        dividend_yield=dividend_yield,
    )

    quote_strikes = quote_data["strike"].to_numpy(dtype=float)
    quote_maturities = quote_data["maturity"].to_numpy(dtype=float)

    if strike_max is None:
        strike_max = max(
            3.0 * spot,
            1.25 * float(np.max(quote_strikes)),
        )

    strike_grid, time_grid, call_surface = solve_forward_dupire(
        spot=spot,
        local_volatility=local_volatility,
        max_maturity=float(np.max(quote_maturities)),
        rate=rate,
        dividend_yield=dividend_yield,
        strike_max=strike_max,
        number_of_strike_points=number_of_strike_points,
        number_of_time_steps=number_of_time_steps,
        theta=theta,
    )

    return np.asarray(
        interpolate_call_prices(
            strike_grid=strike_grid,
            time_grid=time_grid,
            call_surface=call_surface,
            strikes=quote_strikes,
            maturities=quote_maturities,
        ),
        dtype=float,
    )


def finite_difference_jacobian(
    reference_log_variance: np.ndarray,
    calibration_maturities: np.ndarray,
    calibration_log_moneyness: np.ndarray,
    quote_data: pd.DataFrame,
    spot: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    finite_difference_step: float = 1e-3,
    scheme: str = "forward",
    strike_max: float | None = None,
    number_of_strike_points: int = 181,
    number_of_time_steps: int = 160,
    theta: float = 0.5,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate the pricing Jacobian with respect to gridded log variance.

    Column ``j`` contains the change in all quote prices caused by changing
    the ``j``-th element of the C-order-flattened log-variance surface.
    """
    reference = np.asarray(reference_log_variance, dtype=float)
    calibration_maturities = np.asarray(
        calibration_maturities,
        dtype=float,
    )
    calibration_log_moneyness = np.asarray(
        calibration_log_moneyness,
        dtype=float,
    )

    expected_shape = (
        calibration_maturities.size,
        calibration_log_moneyness.size,
    )
    if reference.shape != expected_shape:
        raise ValueError(
            f"reference_log_variance has shape {reference.shape}; "
            f"expected {expected_shape}."
        )
    if finite_difference_step <= 0:
        raise ValueError("finite_difference_step must be positive.")
    if scheme not in {"forward", "central"}:
        raise ValueError("scheme must be 'forward' or 'central'.")

    pricing_arguments = dict(
        calibration_maturities=calibration_maturities,
        calibration_log_moneyness=calibration_log_moneyness,
        quote_data=quote_data,
        spot=spot,
        rate=rate,
        dividend_yield=dividend_yield,
        strike_max=strike_max,
        number_of_strike_points=number_of_strike_points,
        number_of_time_steps=number_of_time_steps,
        theta=theta,
    )

    base_prices = price_quotes_from_log_variance(
        log_variance_surface=reference,
        **pricing_arguments,
    )

    reference_vector = reference.reshape(-1, order="C")
    jacobian = np.empty(
        (len(quote_data), reference_vector.size),
        dtype=float,
    )

    report_every = max(1, reference_vector.size // 10)

    for parameter_index in range(reference_vector.size):
        positive_vector = reference_vector.copy()
        positive_vector[parameter_index] += finite_difference_step
        positive_surface = positive_vector.reshape(
            reference.shape,
            order="C",
        )
        positive_prices = price_quotes_from_log_variance(
            log_variance_surface=positive_surface,
            **pricing_arguments,
        )

        if scheme == "forward":
            jacobian[:, parameter_index] = (
                positive_prices - base_prices
            ) / finite_difference_step
        else:
            negative_vector = reference_vector.copy()
            negative_vector[parameter_index] -= finite_difference_step
            negative_surface = negative_vector.reshape(
                reference.shape,
                order="C",
            )
            negative_prices = price_quotes_from_log_variance(
                log_variance_surface=negative_surface,
                **pricing_arguments,
            )
            jacobian[:, parameter_index] = (
                positive_prices - negative_prices
            ) / (2.0 * finite_difference_step)

        if verbose and (
            (parameter_index + 1) % report_every == 0
            or parameter_index + 1 == reference_vector.size
        ):
            print(
                "Jacobian columns completed:",
                f"{parameter_index + 1}/{reference_vector.size}",
            )

    return base_prices, jacobian


def quote_weights(
    quote_data: pd.DataFrame,
    noise_column: str = "noise_standard_deviation",
    minimum_standard_deviation: float = 1e-6,
) -> np.ndarray:
    """Return diagonal-weight entries ``1 / noise_standard_deviation``."""
    if noise_column not in quote_data.columns:
        raise ValueError(f"quote_data does not contain '{noise_column}'.")
    if minimum_standard_deviation <= 0:
        raise ValueError("minimum_standard_deviation must be positive.")

    standard_deviation = quote_data[noise_column].to_numpy(dtype=float)
    if np.any(~np.isfinite(standard_deviation)):
        raise ValueError("quote noise contains non-finite values.")
    if np.any(standard_deviation <= 0):
        raise ValueError("quote noise standard deviations must be positive.")

    return 1.0 / np.maximum(
        standard_deviation,
        minimum_standard_deviation,
    )


def solve_weighted_linearized_problem(
    jacobian: np.ndarray,
    residual: np.ndarray,
    weights: np.ndarray,
    regularization_matrix: csc_matrix,
    regularization_strength: float,
) -> dict[str, np.ndarray | float]:
    """Solve the weighted Tikhonov normal equations for the correction ``h``."""
    L = np.asarray(jacobian, dtype=float)
    d = np.asarray(residual, dtype=float).reshape(-1)
    weight_vector = np.asarray(weights, dtype=float).reshape(-1)

    if L.ndim != 2:
        raise ValueError("jacobian must be two-dimensional.")
    if L.shape[0] != d.size or d.size != weight_vector.size:
        raise ValueError("jacobian, residual, and weights are incompatible.")
    if regularization_matrix.shape != (L.shape[1], L.shape[1]):
        raise ValueError("regularization_matrix has the wrong shape.")
    if np.any(weight_vector <= 0):
        raise ValueError("weights must be positive.")
    if regularization_strength < 0:
        raise ValueError("regularization_strength must be non-negative.")

    weighted_jacobian = weight_vector[:, None] * L
    weighted_residual = weight_vector * d

    if regularization_strength == 0:
        correction, *_ = np.linalg.lstsq(
            weighted_jacobian,
            weighted_residual,
            rcond=None,
        )
    else:
        normal_matrix = csc_matrix(
            weighted_jacobian.T @ weighted_jacobian
        ) + regularization_strength * regularization_matrix
        right_hand_side = weighted_jacobian.T @ weighted_residual
        correction = spsolve(normal_matrix, right_hand_side)

    fitted_residual = L @ correction - d
    weighted_data_misfit = float(
        np.sum((weight_vector * fitted_residual) ** 2)
    )
    penalty = float(
        correction
        @ (regularization_matrix @ correction)
    )
    objective = (
        weighted_data_misfit
        + regularization_strength * penalty
    )

    return {
        "correction": np.asarray(correction),
        "linearized_price_change": L @ correction,
        "weighted_data_misfit": weighted_data_misfit,
        "regularization_penalty": penalty,
        "objective": objective,
    }


def weighted_rmse(
    predicted: np.ndarray,
    observed: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Return root mean squared residual in noise-standard-deviation units."""
    predicted_values = np.asarray(predicted, dtype=float).reshape(-1)
    observed_values = np.asarray(observed, dtype=float).reshape(-1)
    weight_values = np.asarray(weights, dtype=float).reshape(-1)

    if not (
        predicted_values.size
        == observed_values.size
        == weight_values.size
    ):
        raise ValueError("predicted, observed, and weights are incompatible.")

    return float(
        np.sqrt(
            np.mean(
                (
                    weight_values
                    * (predicted_values - observed_values)
                )
                ** 2
            )
        )
    )


def run_linearized_calibration(
    reference_log_variance: np.ndarray,
    calibration_maturities: np.ndarray,
    calibration_log_moneyness: np.ndarray,
    quote_data: pd.DataFrame,
    regularization_matrix: csc_matrix,
    regularization_strength: float,
    spot: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    observed_price_column: str = "observed_call_price",
    noise_column: str = "noise_standard_deviation",
    finite_difference_step: float = 1e-3,
    jacobian_scheme: str = "forward",
    strike_max: float | None = None,
    number_of_strike_points: int = 181,
    number_of_time_steps: int = 160,
    theta: float = 0.5,
    verbose: bool = False,
) -> dict[str, object]:
    """Build the Jacobian, solve once for ``h``, and reprice nonlinearly."""
    _validate_quote_data(
        quote_data,
        price_column=observed_price_column,
    )

    observed_prices = quote_data[observed_price_column].to_numpy(
        dtype=float
    )
    weights = quote_weights(
        quote_data=quote_data,
        noise_column=noise_column,
    )

    common_pricing_arguments = dict(
        calibration_maturities=calibration_maturities,
        calibration_log_moneyness=calibration_log_moneyness,
        quote_data=quote_data,
        spot=spot,
        rate=rate,
        dividend_yield=dividend_yield,
        strike_max=strike_max,
        number_of_strike_points=number_of_strike_points,
        number_of_time_steps=number_of_time_steps,
        theta=theta,
    )

    reference_prices, jacobian = finite_difference_jacobian(
        reference_log_variance=reference_log_variance,
        finite_difference_step=finite_difference_step,
        scheme=jacobian_scheme,
        verbose=verbose,
        **common_pricing_arguments,
    )

    residual = observed_prices - reference_prices
    linear_solution = solve_weighted_linearized_problem(
        jacobian=jacobian,
        residual=residual,
        weights=weights,
        regularization_matrix=regularization_matrix,
        regularization_strength=regularization_strength,
    )

    correction_vector = linear_solution["correction"]
    correction_surface = correction_vector.reshape(
        np.asarray(reference_log_variance).shape,
        order="C",
    )
    estimated_log_variance = (
        np.asarray(reference_log_variance, dtype=float)
        + correction_surface
    )

    linearized_prices = (
        reference_prices
        + linear_solution["linearized_price_change"]
    )
    nonlinear_prices = price_quotes_from_log_variance(
        log_variance_surface=estimated_log_variance,
        **common_pricing_arguments,
    )

    singular_values = np.linalg.svd(
        weights[:, None] * jacobian,
        compute_uv=False,
    )

    return {
        "reference_prices": reference_prices,
        "observed_prices": observed_prices,
        "weights": weights,
        "residual": residual,
        "jacobian": jacobian,
        "singular_values": singular_values,
        "correction_surface": correction_surface,
        "estimated_log_variance": estimated_log_variance,
        "linearized_prices": linearized_prices,
        "nonlinear_prices": nonlinear_prices,
        "reference_weighted_rmse": weighted_rmse(
            reference_prices,
            observed_prices,
            weights,
        ),
        "linearized_weighted_rmse": weighted_rmse(
            linearized_prices,
            observed_prices,
            weights,
        ),
        "nonlinear_weighted_rmse": weighted_rmse(
            nonlinear_prices,
            observed_prices,
            weights,
        ),
        **linear_solution,
    }
