"""Damped Gauss-Newton calibration for the nonlinear local-volatility inverse problem.

The fixed reference log-variance surface is ``u_ref`` and the total correction is

    h = u - u_ref.

The nonlinear objective is

    Phi(u)
    = ||W(F(u) - y)||^2
      + lambda * h.T R h.

At the current iterate ``u_k``, the pricing map is linearised as

    F(u_k + delta) approximately F(u_k) + J_k delta.

The Gauss-Newton step therefore solves

    (J_k.T W.T W J_k + lambda R + mu I) delta
    = J_k.T W.T W (y - F(u_k)) - lambda R h_k,

where ``mu`` is a Levenberg-Marquardt damping parameter. A backtracking line
search accepts only steps that reduce the full nonlinear objective.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import spsolve

from src.inverse.linearized_inverse import (
    finite_difference_jacobian,
    price_quotes_from_log_variance,
    quote_weights,
    weighted_rmse,
)


def nonlinear_objective(
    predicted_prices: np.ndarray,
    observed_prices: np.ndarray,
    weights: np.ndarray,
    total_correction: np.ndarray,
    regularization_matrix: csc_matrix,
    regularization_strength: float,
) -> dict[str, float]:
    """Evaluate the full weighted nonlinear objective.

    Parameters
    ----------
    predicted_prices:
        Prices produced by the nonlinear PDE map.
    observed_prices:
        Market or synthetic observed prices.
    weights:
        Diagonal entries of ``W``.
    total_correction:
        C-order-flattened ``u - u_ref``.
    regularization_matrix:
        Matrix ``R``.
    regularization_strength:
        Non-negative value ``lambda``.

    Returns
    -------
    dict
        Weighted data misfit, regularisation penalty, total objective, and
        weighted RMSE.
    """
    predicted = np.asarray(predicted_prices, dtype=float).reshape(-1)
    observed = np.asarray(observed_prices, dtype=float).reshape(-1)
    weight_values = np.asarray(weights, dtype=float).reshape(-1)
    correction = np.asarray(total_correction, dtype=float).reshape(-1)

    if not (
        predicted.size == observed.size == weight_values.size
    ):
        raise ValueError(
            "predicted_prices, observed_prices, and weights are incompatible."
        )
    if regularization_matrix.shape != (
        correction.size,
        correction.size,
    ):
        raise ValueError(
            "regularization_matrix is incompatible with total_correction."
        )
    if np.any(weight_values <= 0):
        raise ValueError("weights must be positive.")
    if regularization_strength < 0:
        raise ValueError(
            "regularization_strength must be non-negative."
        )

    weighted_residual = weight_values * (
        predicted - observed
    )
    data_misfit = float(weighted_residual @ weighted_residual)
    penalty = float(
        correction
        @ (regularization_matrix @ correction)
    )
    objective = (
        data_misfit
        + regularization_strength * penalty
    )

    return {
        "weighted_data_misfit": data_misfit,
        "regularization_penalty": penalty,
        "objective": objective,
        "weighted_rmse": float(
            np.sqrt(np.mean(weighted_residual**2))
        ),
    }


def solve_gauss_newton_step(
    jacobian: np.ndarray,
    residual: np.ndarray,
    weights: np.ndarray,
    current_total_correction: np.ndarray,
    regularization_matrix: csc_matrix,
    regularization_strength: float,
    damping: float = 0.0,
) -> dict[str, np.ndarray | float]:
    """Solve one damped Gauss-Newton normal equation.

    The residual is defined as

        observed_prices - current_model_prices.

    The right-hand side includes ``-lambda R h_k`` because regularisation acts
    on the total correction ``h_k + delta``, not only on the new step.
    """
    J = np.asarray(jacobian, dtype=float)
    residual_values = np.asarray(
        residual,
        dtype=float,
    ).reshape(-1)
    weight_values = np.asarray(
        weights,
        dtype=float,
    ).reshape(-1)
    current_correction = np.asarray(
        current_total_correction,
        dtype=float,
    ).reshape(-1)

    if J.ndim != 2:
        raise ValueError("jacobian must be two-dimensional.")
    if J.shape[0] != residual_values.size:
        raise ValueError("jacobian and residual are incompatible.")
    if residual_values.size != weight_values.size:
        raise ValueError("residual and weights are incompatible.")
    if J.shape[1] != current_correction.size:
        raise ValueError(
            "jacobian and current_total_correction are incompatible."
        )
    if regularization_matrix.shape != (
        current_correction.size,
        current_correction.size,
    ):
        raise ValueError(
            "regularization_matrix has the wrong shape."
        )
    if np.any(weight_values <= 0):
        raise ValueError("weights must be positive.")
    if regularization_strength < 0:
        raise ValueError(
            "regularization_strength must be non-negative."
        )
    if damping < 0:
        raise ValueError("damping must be non-negative.")

    weighted_jacobian = weight_values[:, None] * J
    weighted_residual = weight_values * residual_values

    normal_matrix = csc_matrix(
        weighted_jacobian.T @ weighted_jacobian
    )
    if regularization_strength > 0:
        normal_matrix = (
            normal_matrix
            + regularization_strength
            * regularization_matrix
        )
    if damping > 0:
        normal_matrix = (
            normal_matrix
            + damping
            * eye(J.shape[1], format="csc")
        )

    right_hand_side = (
        weighted_jacobian.T @ weighted_residual
        - regularization_strength
        * (
            regularization_matrix
            @ current_correction
        )
    )

    step = np.asarray(
        spsolve(normal_matrix, right_hand_side),
        dtype=float,
    )

    predicted_linear_price_change = J @ step
    linear_residual_after_step = (
        predicted_linear_price_change
        - residual_values
    )
    proposed_total_correction = (
        current_correction + step
    )

    linear_data_misfit = float(
        np.sum(
            (
                weight_values
                * linear_residual_after_step
            )
            ** 2
        )
    )
    proposed_penalty = float(
        proposed_total_correction
        @ (
            regularization_matrix
            @ proposed_total_correction
        )
    )

    return {
        "step": step,
        "linearized_price_change": predicted_linear_price_change,
        "linearized_weighted_data_misfit": linear_data_misfit,
        "proposed_regularization_penalty": proposed_penalty,
        "linearized_objective": (
            linear_data_misfit
            + regularization_strength
            * proposed_penalty
        ),
    }


def run_gauss_newton_calibration(
    reference_log_variance: np.ndarray,
    initial_log_variance: np.ndarray,
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
    number_of_strike_points: int = 151,
    number_of_time_steps: int = 120,
    theta: float = 0.5,
    maximum_iterations: int = 4,
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
    """Run damped Gauss-Newton iterations with nonlinear objective checks."""
    reference = np.asarray(
        reference_log_variance,
        dtype=float,
    )
    current = np.asarray(
        initial_log_variance,
        dtype=float,
    ).copy()
    maturity_values = np.asarray(
        calibration_maturities,
        dtype=float,
    )
    x_values = np.asarray(
        calibration_log_moneyness,
        dtype=float,
    )

    expected_shape = (
        maturity_values.size,
        x_values.size,
    )
    if reference.shape != expected_shape:
        raise ValueError(
            f"reference_log_variance has shape {reference.shape}; "
            f"expected {expected_shape}."
        )
    if current.shape != expected_shape:
        raise ValueError(
            f"initial_log_variance has shape {current.shape}; "
            f"expected {expected_shape}."
        )
    if regularization_matrix.shape != (
        reference.size,
        reference.size,
    ):
        raise ValueError(
            "regularization_matrix has the wrong shape."
        )
    if observed_price_column not in quote_data.columns:
        raise ValueError(
            f"quote_data does not contain '{observed_price_column}'."
        )
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be at least 1.")
    if initial_damping < 0:
        raise ValueError("initial_damping must be non-negative.")
    if damping_increase <= 1:
        raise ValueError("damping_increase must exceed 1.")
    if not 0 < damping_decrease <= 1:
        raise ValueError(
            "damping_decrease must lie in (0, 1]."
        )
    if maximum_damping_attempts < 1:
        raise ValueError(
            "maximum_damping_attempts must be at least 1."
        )
    if not 0 < line_search_contraction < 1:
        raise ValueError(
            "line_search_contraction must lie in (0, 1)."
        )
    if not 0 < minimum_step_scale <= 1:
        raise ValueError(
            "minimum_step_scale must lie in (0, 1]."
        )
    if maximum_absolute_step is not None:
        if maximum_absolute_step <= 0:
            raise ValueError(
                "maximum_absolute_step must be positive."
            )
    if relative_objective_tolerance < 0:
        raise ValueError(
            "relative_objective_tolerance must be non-negative."
        )
    if relative_step_tolerance < 0:
        raise ValueError(
            "relative_step_tolerance must be non-negative."
        )

    observed_prices = quote_data[
        observed_price_column
    ].to_numpy(dtype=float)
    weights = quote_weights(
        quote_data=quote_data,
        noise_column=noise_column,
    )

    pricing_arguments = dict(
        calibration_maturities=maturity_values,
        calibration_log_moneyness=x_values,
        quote_data=quote_data,
        spot=spot,
        rate=rate,
        dividend_yield=dividend_yield,
        strike_max=strike_max,
        number_of_strike_points=number_of_strike_points,
        number_of_time_steps=number_of_time_steps,
        theta=theta,
    )

    current_prices = price_quotes_from_log_variance(
        log_variance_surface=current,
        **pricing_arguments,
    )
    current_correction = (
        current - reference
    ).reshape(-1, order="C")
    current_metrics = nonlinear_objective(
        predicted_prices=current_prices,
        observed_prices=observed_prices,
        weights=weights,
        total_correction=current_correction,
        regularization_matrix=regularization_matrix,
        regularization_strength=regularization_strength,
    )

    history: list[dict[str, float | int | bool]] = [
        {
            "iteration": 0,
            "objective": current_metrics["objective"],
            "weighted_data_misfit": current_metrics[
                "weighted_data_misfit"
            ],
            "regularization_penalty": current_metrics[
                "regularization_penalty"
            ],
            "weighted_rmse": current_metrics["weighted_rmse"],
            "damping": initial_damping,
            "step_scale": 0.0,
            "step_norm": 0.0,
            "relative_objective_improvement": 0.0,
            "accepted": True,
        }
    ]

    damping = initial_damping
    stop_reason = "maximum_iterations_reached"
    latest_jacobian = None
    latest_singular_values = None

    for iteration in range(1, maximum_iterations + 1):
        base_prices, jacobian = finite_difference_jacobian(
            reference_log_variance=current,
            finite_difference_step=finite_difference_step,
            scheme=jacobian_scheme,
            verbose=verbose,
            **pricing_arguments,
        )
        current_prices = base_prices
        latest_jacobian = jacobian
        latest_singular_values = np.linalg.svd(
            weights[:, None] * jacobian,
            compute_uv=False,
        )

        current_correction = (
            current - reference
        ).reshape(-1, order="C")
        current_metrics = nonlinear_objective(
            predicted_prices=current_prices,
            observed_prices=observed_prices,
            weights=weights,
            total_correction=current_correction,
            regularization_matrix=regularization_matrix,
            regularization_strength=regularization_strength,
        )
        residual = observed_prices - current_prices

        accepted = False
        accepted_scale = 0.0
        accepted_step = None
        accepted_prices = None
        accepted_metrics = None
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
            step = np.asarray(
                step_result["step"],
                dtype=float,
            )

            if maximum_absolute_step is not None:
                largest_component = float(
                    np.max(np.abs(step))
                )
                if largest_component > maximum_absolute_step:
                    step = step * (
                        maximum_absolute_step
                        / largest_component
                    )

            step_scale = 1.0
            while step_scale >= minimum_step_scale:
                candidate_vector = (
                    current.reshape(-1, order="C")
                    + step_scale * step
                )
                candidate_surface = candidate_vector.reshape(
                    current.shape,
                    order="C",
                )
                candidate_prices = (
                    price_quotes_from_log_variance(
                        log_variance_surface=candidate_surface,
                        **pricing_arguments,
                    )
                )
                candidate_correction = (
                    candidate_surface - reference
                ).reshape(-1, order="C")
                candidate_metrics = nonlinear_objective(
                    predicted_prices=candidate_prices,
                    observed_prices=observed_prices,
                    weights=weights,
                    total_correction=candidate_correction,
                    regularization_matrix=regularization_matrix,
                    regularization_strength=regularization_strength,
                )

                if (
                    candidate_metrics["objective"]
                    < current_metrics["objective"]
                ):
                    accepted = True
                    accepted_scale = step_scale
                    accepted_step = step
                    accepted_prices = candidate_prices
                    accepted_metrics = candidate_metrics
                    current = candidate_surface
                    break

                step_scale *= line_search_contraction

            if accepted:
                break

            damping_used = max(
                1e-12,
                damping_used * damping_increase,
            )

        if not accepted:
            stop_reason = "no_objective_decrease"
            history.append(
                {
                    "iteration": iteration,
                    "objective": current_metrics["objective"],
                    "weighted_data_misfit": current_metrics[
                        "weighted_data_misfit"
                    ],
                    "regularization_penalty": current_metrics[
                        "regularization_penalty"
                    ],
                    "weighted_rmse": current_metrics[
                        "weighted_rmse"
                    ],
                    "damping": damping_used,
                    "step_scale": 0.0,
                    "step_norm": 0.0,
                    "relative_objective_improvement": 0.0,
                    "accepted": False,
                }
            )
            break

        previous_objective = current_metrics["objective"]
        current_prices = np.asarray(accepted_prices)
        current_metrics = accepted_metrics

        scaled_step = accepted_scale * accepted_step
        step_norm = float(np.linalg.norm(scaled_step))
        relative_step = step_norm / (
            1.0
            + float(
                np.linalg.norm(
                    current.reshape(-1, order="C")
                )
            )
        )
        relative_improvement = float(
            (
                previous_objective
                - current_metrics["objective"]
            )
            / max(1.0, abs(previous_objective))
        )

        history.append(
            {
                "iteration": iteration,
                "objective": current_metrics["objective"],
                "weighted_data_misfit": current_metrics[
                    "weighted_data_misfit"
                ],
                "regularization_penalty": current_metrics[
                    "regularization_penalty"
                ],
                "weighted_rmse": current_metrics[
                    "weighted_rmse"
                ],
                "damping": damping_used,
                "step_scale": accepted_scale,
                "step_norm": step_norm,
                "relative_objective_improvement": (
                    relative_improvement
                ),
                "accepted": True,
            }
        )

        damping = max(
            1e-12,
            damping_used * damping_decrease,
        )

        if verbose:
            print(
                f"Iteration {iteration}: "
                f"objective={current_metrics['objective']:.6f}, "
                f"weighted RMSE={current_metrics['weighted_rmse']:.6f}, "
                f"step scale={accepted_scale:.4f}, "
                f"damping={damping_used:.3e}"
            )

        if (
            relative_improvement
            <= relative_objective_tolerance
        ):
            stop_reason = "relative_objective_tolerance"
            break

        if relative_step <= relative_step_tolerance:
            stop_reason = "relative_step_tolerance"
            break

    final_correction_surface = current - reference
    final_prices = current_prices
    final_metrics = nonlinear_objective(
        predicted_prices=final_prices,
        observed_prices=observed_prices,
        weights=weights,
        total_correction=final_correction_surface.reshape(
            -1,
            order="C",
        ),
        regularization_matrix=regularization_matrix,
        regularization_strength=regularization_strength,
    )

    return {
        "reference_log_variance": reference,
        "initial_log_variance": np.asarray(
            initial_log_variance,
            dtype=float,
        ),
        "estimated_log_variance": current,
        "correction_surface": final_correction_surface,
        "observed_prices": observed_prices,
        "initial_prices": price_quotes_from_log_variance(
            log_variance_surface=np.asarray(
                initial_log_variance,
                dtype=float,
            ),
            **pricing_arguments,
        ),
        "fitted_prices": final_prices,
        "weights": weights,
        "history": pd.DataFrame(history),
        "stop_reason": stop_reason,
        "final_weighted_rmse": weighted_rmse(
            final_prices,
            observed_prices,
            weights,
        ),
        "final_objective": final_metrics["objective"],
        "final_weighted_data_misfit": final_metrics[
            "weighted_data_misfit"
        ],
        "final_regularization_penalty": final_metrics[
            "regularization_penalty"
        ],
        "latest_jacobian": latest_jacobian,
        "latest_singular_values": latest_singular_values,
    }
