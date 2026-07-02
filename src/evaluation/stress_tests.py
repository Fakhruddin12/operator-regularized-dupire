"""Diagnostics for localised-feature synthetic stress tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.benchmark import (
    reconstruction_metrics,
)


def localized_bump_mask(
    point_data: pd.DataFrame,
    x_centre: float = 0.0,
    maturity_centre: float = 0.60,
    x_width: float = 0.12,
    maturity_width: float = 0.25,
    radius: float = 2.0,
) -> np.ndarray:
    """Return an elliptical mask around the known synthetic bump.

    The default region contains points within two Gaussian width parameters
    from the bump centre.
    """
    required_columns = {
        "maturity",
        "log_moneyness",
    }
    missing = required_columns.difference(
        point_data.columns
    )
    if missing:
        raise ValueError(
            f"point_data is missing columns: {sorted(missing)}"
        )
    if x_width <= 0 or maturity_width <= 0:
        raise ValueError(
            "x_width and maturity_width must be positive."
        )
    if radius <= 0:
        raise ValueError("radius must be positive.")

    scaled_squared_distance = (
        (
            (
                point_data[
                    "log_moneyness"
                ].to_numpy(dtype=float)
                - x_centre
            )
            / x_width
        )
        ** 2
        + (
            (
                point_data[
                    "maturity"
                ].to_numpy(dtype=float)
                - maturity_centre
            )
            / maturity_width
        )
        ** 2
    )

    return scaled_squared_distance <= radius**2


def localized_feature_metrics(
    pointwise_results: pd.DataFrame,
    method_columns: list[str] | None = None,
    truth_column: str = "true_local_volatility",
    shared_mask_column: str = "shared_valid",
    baseline_volatility: float = 0.20,
    x_centre: float = 0.0,
    maturity_centre: float = 0.60,
    x_width: float = 0.12,
    maturity_width: float = 0.25,
    radius: float = 2.0,
    flat_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Measure overall error and recovery of a localised volatility feature.

    In addition to RMSE, this reports the estimated peak amplitude and location.
    Flat estimates have undefined peak locations and therefore receive NaN for
    the location diagnostics.
    """
    required_columns = {
        "maturity",
        "log_moneyness",
        truth_column,
        shared_mask_column,
    }
    missing = required_columns.difference(
        pointwise_results.columns
    )
    if missing:
        raise ValueError(
            "pointwise_results is missing columns: "
            f"{sorted(missing)}"
        )
    if baseline_volatility <= 0:
        raise ValueError(
            "baseline_volatility must be positive."
        )

    if method_columns is None:
        excluded = required_columns
        method_columns = [
            column
            for column in pointwise_results.columns
            if column not in excluded
        ]

    if not method_columns:
        raise ValueError(
            "at least one method column is required."
        )

    shared_mask = pointwise_results[
        shared_mask_column
    ].to_numpy(dtype=bool)
    evaluation_data = pointwise_results.loc[
        shared_mask
    ].reset_index(drop=True)

    if evaluation_data.empty:
        raise ValueError(
            "no shared valid points are available."
        )

    bump_mask = localized_bump_mask(
        evaluation_data,
        x_centre=x_centre,
        maturity_centre=maturity_centre,
        x_width=x_width,
        maturity_width=maturity_width,
        radius=radius,
    )
    outside_mask = ~bump_mask

    if not np.any(bump_mask):
        raise ValueError(
            "the chosen bump region contains no points."
        )
    if not np.any(outside_mask):
        raise ValueError(
            "the chosen bump region contains all points."
        )

    truth = evaluation_data[
        truth_column
    ].to_numpy(dtype=float)

    true_peak_index = int(np.nanargmax(truth))
    true_peak = float(truth[true_peak_index])
    true_peak_x = float(
        evaluation_data.loc[
            true_peak_index,
            "log_moneyness",
        ]
    )
    true_peak_maturity = float(
        evaluation_data.loc[
            true_peak_index,
            "maturity",
        ]
    )
    true_excess = true_peak - baseline_volatility

    rows = []

    for method in method_columns:
        if method not in evaluation_data.columns:
            raise ValueError(
                f"pointwise_results does not contain '{method}'."
            )

        estimate = evaluation_data[
            method
        ].to_numpy(dtype=float)

        overall = reconstruction_metrics(
            estimate=estimate,
            truth=truth,
        )
        bump = reconstruction_metrics(
            estimate=estimate,
            truth=truth,
            valid_mask=bump_mask,
        )
        outside = reconstruction_metrics(
            estimate=estimate,
            truth=truth,
            valid_mask=outside_mask,
        )

        estimate_range = float(
            np.nanmax(estimate)
            - np.nanmin(estimate)
        )

        if estimate_range <= flat_tolerance:
            estimated_peak = float(
                np.nanmax(estimate)
            )
            estimated_peak_x = np.nan
            estimated_peak_maturity = np.nan
            scaled_peak_location_error = np.nan
        else:
            estimated_peak_index = int(
                np.nanargmax(estimate)
            )
            estimated_peak = float(
                estimate[estimated_peak_index]
            )
            estimated_peak_x = float(
                evaluation_data.loc[
                    estimated_peak_index,
                    "log_moneyness",
                ]
            )
            estimated_peak_maturity = float(
                evaluation_data.loc[
                    estimated_peak_index,
                    "maturity",
                ]
            )
            scaled_peak_location_error = float(
                np.sqrt(
                    (
                        (
                            estimated_peak_x
                            - true_peak_x
                        )
                        / x_width
                    )
                    ** 2
                    + (
                        (
                            estimated_peak_maturity
                            - true_peak_maturity
                        )
                        / maturity_width
                    )
                    ** 2
                )
            )

        recovered_excess = (
            estimated_peak - baseline_volatility
        )
        peak_excess_ratio = (
            recovered_excess / true_excess
            if true_excess > 0
            else np.nan
        )

        rows.append(
            {
                "method": method,
                "overall_rmse": overall["rmse"],
                "overall_mae": overall["mae"],
                "bump_region_rmse": bump["rmse"],
                "outside_region_rmse": outside["rmse"],
                "true_peak_volatility": true_peak,
                "estimated_peak_volatility": (
                    estimated_peak
                ),
                "peak_amplitude_error": (
                    estimated_peak - true_peak
                ),
                "peak_excess_ratio": (
                    peak_excess_ratio
                ),
                "true_peak_x": true_peak_x,
                "estimated_peak_x": (
                    estimated_peak_x
                ),
                "true_peak_maturity": (
                    true_peak_maturity
                ),
                "estimated_peak_maturity": (
                    estimated_peak_maturity
                ),
                "scaled_peak_location_error": (
                    scaled_peak_location_error
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "overall_rmse"
    ).reset_index(drop=True)


def repeated_noise_summary(
    repeated_results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise reconstruction performance over independent noise draws."""
    required_columns = {
        "method",
        "rmse",
        "mae",
        "shared_valid_fraction",
    }
    missing = required_columns.difference(
        repeated_results.columns
    )
    if missing:
        raise ValueError(
            "repeated_results is missing columns: "
            f"{sorted(missing)}"
        )

    return (
        repeated_results.groupby(
            "method",
            as_index=False,
        )
        .agg(
            mean_rmse=("rmse", "mean"),
            sd_rmse=("rmse", "std"),
            minimum_rmse=("rmse", "min"),
            maximum_rmse=("rmse", "max"),
            mean_mae=("mae", "mean"),
            mean_shared_valid_fraction=(
                "shared_valid_fraction",
                "mean",
            ),
        )
        .sort_values("mean_rmse")
        .reset_index(drop=True)
    )
