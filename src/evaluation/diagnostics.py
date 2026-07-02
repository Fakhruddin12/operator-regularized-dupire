"""Diagnostics for local-volatility calibration results."""

from __future__ import annotations

import numpy as np
import pandas as pd


def weighted_residual_table(
    quote_data: pd.DataFrame,
    predicted_prices: np.ndarray,
    observed_price_column: str = "observed_call_price",
    noise_column: str = "noise_standard_deviation",
) -> pd.DataFrame:
    """Return quote-level raw and noise-scaled residuals.

    The residual convention is

        model price - observed price.
    """
    if observed_price_column not in quote_data.columns:
        raise ValueError(
            f"quote_data does not contain '{observed_price_column}'."
        )
    if noise_column not in quote_data.columns:
        raise ValueError(
            f"quote_data does not contain '{noise_column}'."
        )

    predicted = np.asarray(
        predicted_prices,
        dtype=float,
    ).reshape(-1)

    if predicted.size != len(quote_data):
        raise ValueError(
            "predicted_prices must contain one value per quote."
        )

    observed = quote_data[
        observed_price_column
    ].to_numpy(dtype=float)
    noise = quote_data[
        noise_column
    ].to_numpy(dtype=float)

    if np.any(noise <= 0):
        raise ValueError(
            "noise standard deviations must be positive."
        )

    result = quote_data.copy()
    result["model_call_price"] = predicted
    result["price_residual"] = predicted - observed
    result["weighted_residual"] = (
        result["price_residual"] / noise
    )
    result["absolute_weighted_residual"] = np.abs(
        result["weighted_residual"]
    )

    return result


def residual_summary(
    residual_table: pd.DataFrame,
) -> dict[str, float]:
    """Summarise weighted calibration residuals."""
    if "weighted_residual" not in residual_table.columns:
        raise ValueError(
            "residual_table must contain 'weighted_residual'."
        )

    residuals = residual_table[
        "weighted_residual"
    ].to_numpy(dtype=float)

    if residuals.size == 0:
        raise ValueError("residual_table is empty.")

    return {
        "mean_weighted_residual": float(
            np.mean(residuals)
        ),
        "weighted_residual_sd": float(
            np.std(residuals, ddof=0)
        ),
        "weighted_rmse": float(
            np.sqrt(np.mean(residuals**2))
        ),
        "mean_absolute_weighted_residual": float(
            np.mean(np.abs(residuals))
        ),
        "maximum_absolute_weighted_residual": float(
            np.max(np.abs(residuals))
        ),
    }


def jacobian_spectrum(
    jacobian: np.ndarray,
    weights: np.ndarray,
    relative_tolerance: float = 1e-8,
) -> dict[str, np.ndarray | float | int]:
    """Compute singular-value diagnostics for the weighted Jacobian."""
    J = np.asarray(jacobian, dtype=float)
    weight_values = np.asarray(
        weights,
        dtype=float,
    ).reshape(-1)

    if J.ndim != 2:
        raise ValueError("jacobian must be two-dimensional.")
    if J.shape[0] != weight_values.size:
        raise ValueError(
            "jacobian and weights are incompatible."
        )
    if np.any(weight_values <= 0):
        raise ValueError("weights must be positive.")
    if relative_tolerance <= 0:
        raise ValueError(
            "relative_tolerance must be positive."
        )

    singular_values = np.linalg.svd(
        weight_values[:, None] * J,
        compute_uv=False,
    )

    if singular_values.size == 0:
        raise ValueError("jacobian has no singular values.")

    threshold = (
        relative_tolerance * singular_values[0]
    )
    numerical_rank = int(
        np.sum(singular_values > threshold)
    )

    positive = singular_values[
        singular_values > threshold
    ]
    condition_number = (
        float(positive[0] / positive[-1])
        if positive.size > 0
        else np.inf
    )

    cumulative_energy = np.cumsum(
        singular_values**2
    ) / np.sum(singular_values**2)

    return {
        "singular_values": singular_values,
        "numerical_rank": numerical_rank,
        "condition_number": condition_number,
        "relative_threshold": threshold,
        "cumulative_energy": cumulative_energy,
    }


def surface_rmse(
    estimated_surface: np.ndarray,
    true_surface: np.ndarray,
) -> float:
    """Return RMSE between two equally shaped surfaces."""
    estimated = np.asarray(
        estimated_surface,
        dtype=float,
    )
    truth = np.asarray(
        true_surface,
        dtype=float,
    )

    if estimated.shape != truth.shape:
        raise ValueError(
            "estimated_surface and true_surface must have equal shape."
        )

    return float(
        np.sqrt(np.mean((estimated - truth) ** 2))
    )
