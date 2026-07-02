"""Tests for the final realistic-bumpy local-volatility benchmark."""

import numpy as np
import pandas as pd

from src.evaluation.realistic_bumpy_benchmark import (
    default_quote_log_moneyness,
    default_quote_maturities,
    multi_feature_recovery_metrics,
    realistic_bumpy_local_volatility,
    realistic_equity_base_surface,
    realistic_noise_standard_deviation,
)


def test_realistic_surface_is_positive_and_plausible() -> None:
    x = np.linspace(-0.40, 0.40, 81)
    T = np.linspace(0.03, 1.50, 61)
    x_mesh, T_mesh = np.meshgrid(x, T)
    surface = realistic_bumpy_local_volatility(x_mesh, T_mesh)

    assert np.all(np.isfinite(surface))
    assert float(np.min(surface)) > 0.14
    assert float(np.max(surface)) < 0.32


def test_base_surface_has_equity_style_downside_skew() -> None:
    downside = realistic_equity_base_surface(-0.20, 0.50)
    upside = realistic_equity_base_surface(0.20, 0.50)
    assert downside > upside


def test_added_features_have_expected_signs() -> None:
    primary = realistic_bumpy_local_volatility(-0.105, 0.30)
    primary_base = realistic_equity_base_surface(-0.105, 0.30)
    secondary = realistic_bumpy_local_volatility(0.075, 0.78)
    secondary_base = realistic_equity_base_surface(0.075, 0.78)

    assert primary - primary_base > 0.035
    assert secondary - secondary_base < -0.012


def test_noise_is_higher_in_wings_and_at_short_maturity() -> None:
    prices = np.array([5.0, 5.0, 5.0])
    x = np.array([0.0, 0.32, 0.0])
    T = np.array([0.75, 0.75, 0.07])
    noise = realistic_noise_standard_deviation(prices, x, T)

    assert noise[1] > noise[0]
    assert noise[2] > noise[0]


def test_perfect_reconstruction_has_zero_feature_error() -> None:
    maturities = default_quote_maturities()
    x_values = default_quote_log_moneyness()
    x_mesh, T_mesh = np.meshgrid(x_values, maturities)
    truth = realistic_bumpy_local_volatility(x_mesh, T_mesh).reshape(-1)
    pointwise = pd.DataFrame(
        {
            "maturity": T_mesh.reshape(-1),
            "log_moneyness": x_mesh.reshape(-1),
            "true_local_volatility": truth,
            "shared_valid": True,
            "perfect": truth,
        }
    )

    result = multi_feature_recovery_metrics(
        pointwise,
        method_columns=["perfect"],
    ).iloc[0]

    assert np.isclose(result["overall_rmse"], 0.0)
    assert np.isclose(result["primary_event_rmse"], 0.0)
    assert np.isclose(result["secondary_dip_rmse"], 0.0)
    assert np.isclose(result["feature_correlation"], 1.0)
