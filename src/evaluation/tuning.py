"""Regularisation-parameter tuning for the linearised inverse problem."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.linalg import solve


def linearized_lambda_sweep(
    jacobian: np.ndarray,
    residual: np.ndarray,
    weights: np.ndarray,
    regularization_matrix,
    lambda_values: Iterable[float],
    current_total_correction: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[float, np.ndarray]]:
    """Solve the weighted linearised problem over a sequence of lambdas.

    Generalised cross-validation is computed as

    ``GCV = (RSS / m) / (1 - df / m)^2``,

    where ``df = trace(A (A.T A + lambda R)^(-1) A.T)`` and
    ``A = W J``.
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
    R = regularization_matrix.toarray()

    if J.ndim != 2:
        raise ValueError("jacobian must be two-dimensional.")
    if J.shape[0] != residual_values.size:
        raise ValueError(
            "jacobian and residual are incompatible."
        )
    if residual_values.size != weight_values.size:
        raise ValueError(
            "residual and weights are incompatible."
        )
    if R.shape != (J.shape[1], J.shape[1]):
        raise ValueError(
            "regularization_matrix has the wrong shape."
        )
    if np.any(weight_values <= 0):
        raise ValueError("weights must be positive.")

    lambdas = np.asarray(
        list(lambda_values),
        dtype=float,
    )

    if lambdas.ndim != 1 or lambdas.size == 0:
        raise ValueError(
            "lambda_values must contain at least one value."
        )
    if np.any(lambdas <= 0):
        raise ValueError(
            "all lambda values must be positive."
        )

    if current_total_correction is None:
        current_correction = np.zeros(J.shape[1])
    else:
        current_correction = np.asarray(
            current_total_correction,
            dtype=float,
        ).reshape(-1)
        if current_correction.size != J.shape[1]:
            raise ValueError(
                "current_total_correction has the wrong size."
            )

    A = weight_values[:, None] * J
    b = weight_values * residual_values
    gram_matrix = A.T @ A
    weighted_rhs = A.T @ b
    number_of_quotes = J.shape[0]

    rows = []
    solutions: dict[float, np.ndarray] = {}

    for lambda_value in lambdas:
        system_matrix = (
            gram_matrix
            + lambda_value * R
        )
        right_hand_side = (
            weighted_rhs
            - lambda_value
            * (R @ current_correction)
        )

        step = solve(
            system_matrix,
            right_hand_side,
            assume_a="sym",
        )
        total_correction = (
            current_correction + step
        )

        weighted_residual_after = (
            A @ step - b
        )
        weighted_data_misfit = float(
            weighted_residual_after
            @ weighted_residual_after
        )
        penalty = float(
            total_correction
            @ (R @ total_correction)
        )

        inverse_times_gram = solve(
            system_matrix,
            gram_matrix,
            assume_a="sym",
        )
        effective_df = float(
            np.trace(inverse_times_gram)
        )

        denominator = (
            1.0
            - effective_df / number_of_quotes
        )
        if denominator <= 0:
            gcv = np.inf
        else:
            gcv = (
                weighted_data_misfit
                / number_of_quotes
            ) / denominator**2

        rows.append(
            {
                "lambda": float(lambda_value),
                "weighted_data_misfit": (
                    weighted_data_misfit
                ),
                "weighted_rmse": float(
                    np.sqrt(
                        weighted_data_misfit
                        / number_of_quotes
                    )
                ),
                "regularization_penalty": penalty,
                "objective": (
                    weighted_data_misfit
                    + lambda_value * penalty
                ),
                "effective_degrees_of_freedom": (
                    effective_df
                ),
                "gcv": float(gcv),
                "correction_norm": float(
                    np.linalg.norm(total_correction)
                ),
            }
        )
        solutions[float(lambda_value)] = (
            total_correction
        )

    results = pd.DataFrame(rows).sort_values(
        "lambda"
    ).reset_index(drop=True)

    return results, solutions


def select_lambda_by_gcv(
    sweep_results: pd.DataFrame,
) -> float:
    """Return the lambda corresponding to the smallest finite GCV value."""
    required_columns = {"lambda", "gcv"}
    missing = required_columns.difference(
        sweep_results.columns
    )
    if missing:
        raise ValueError(
            f"sweep_results is missing columns: {sorted(missing)}"
        )

    finite_results = sweep_results[
        np.isfinite(sweep_results["gcv"])
    ]
    if finite_results.empty:
        raise ValueError(
            "sweep_results contains no finite GCV values."
        )

    best_index = finite_results["gcv"].idxmin()
    return float(
        finite_results.loc[best_index, "lambda"]
    )
