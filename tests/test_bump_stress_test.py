"""Tests for the Stage 11 localised-bump stress-test diagnostics."""

import numpy as np
import pandas as pd

from src.evaluation.stress_tests import (
    localized_bump_mask,
    localized_feature_metrics,
    repeated_noise_summary,
)


def _pointwise_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "maturity": [0.5, 0.5, 1.5],
            "log_moneyness": [0.0, 0.3, 0.0],
            "true_local_volatility": [
                0.25,
                0.20,
                0.20,
            ],
            "perfect": [0.25, 0.20, 0.20],
            "flat": [0.20, 0.20, 0.20],
            "shared_valid": [True, True, True],
        }
    )


def test_localized_bump_mask_identifies_central_point() -> None:
    data = _pointwise_data()

    mask = localized_bump_mask(data)

    assert mask.tolist() == [True, False, False]


def test_perfect_feature_recovery_has_zero_error() -> None:
    data = _pointwise_data()

    metrics = localized_feature_metrics(
        pointwise_results=data,
        method_columns=["perfect"],
    )

    row = metrics.iloc[0]

    assert np.isclose(row["overall_rmse"], 0.0)
    assert np.isclose(row["bump_region_rmse"], 0.0)
    assert np.isclose(row["peak_excess_ratio"], 1.0)
    assert np.isclose(
        row["scaled_peak_location_error"],
        0.0,
    )


def test_flat_surface_has_no_defined_peak_location() -> None:
    data = _pointwise_data()

    metrics = localized_feature_metrics(
        pointwise_results=data,
        method_columns=["flat"],
    )

    row = metrics.iloc[0]

    assert np.isclose(row["peak_excess_ratio"], 0.0)
    assert np.isnan(row["estimated_peak_x"])
    assert np.isnan(
        row["scaled_peak_location_error"]
    )


def test_repeated_noise_summary_orders_by_mean_rmse() -> None:
    results = pd.DataFrame(
        {
            "method": ["a", "a", "b", "b"],
            "rmse": [1.0, 2.0, 0.2, 0.4],
            "mae": [0.8, 1.5, 0.1, 0.3],
            "shared_valid_fraction": [
                1.0,
                0.9,
                1.0,
                1.0,
            ],
        }
    )

    summary = repeated_noise_summary(results)

    assert summary.iloc[0]["method"] == "b"
    assert np.isclose(
        summary.iloc[0]["mean_rmse"],
        0.3,
    )
