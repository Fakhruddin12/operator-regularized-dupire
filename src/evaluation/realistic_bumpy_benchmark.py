"""Final synthetic benchmark with a realistic multi-feature local-volatility surface.

The truth is deliberately smoother and more market-like than the isolated
Gaussian bump used in Stage 11. It combines an equity-index term structure and
downside skew with a broad event-related elevation and a weaker secondary dip.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.data.synthetic_data import generate_synthetic_option_data
from src.evaluation.benchmark import (
    compare_methods_on_quote_grid,
    repeated_noise_benchmark,
    reconstruction_metrics,
)


def realistic_equity_base_surface(
    log_moneyness: np.ndarray | float,
    maturity: np.ndarray | float,
) -> np.ndarray | float:
    """Return a smooth equity-index-style base local-volatility surface.

    It has higher short-dated volatility, persistent downside skew, mild smile
    curvature, and a gently increasing long-run term component.
    """
    x, T = np.broadcast_arrays(
        np.asarray(log_moneyness, dtype=float),
        np.asarray(maturity, dtype=float),
    )
    if np.any(T < 0):
        raise ValueError("maturity must be non-negative.")

    result = (
        0.175
        + 0.040 * np.exp(-2.4 * T)
        - 0.060 * x
        + 0.085 * x**2
        + 0.010 * np.sqrt(T / (1.0 + T))
    )
    if np.any(result <= 0):
        raise ValueError("the base surface produced non-positive volatility.")
    return float(result) if result.ndim == 0 else result


def realistic_bumpy_local_volatility(
    log_moneyness: np.ndarray | float,
    maturity: np.ndarray | float,
) -> np.ndarray | float:
    """Return the final realistic, smooth, non-parametric test surface.

    The surface adds three economically plausible smooth departures to the
    equity-style base:

    1. a broad downside event elevation near three months;
    2. a weaker upside/medium-term volatility depression;
    3. a short-dated downside-wing ridge.

    All components are smooth Gaussians, so the surface contains no artificial
    discontinuities or high-frequency oscillations.
    """
    x, T = np.broadcast_arrays(
        np.asarray(log_moneyness, dtype=float),
        np.asarray(maturity, dtype=float),
    )
    base = np.asarray(realistic_equity_base_surface(x, T), dtype=float)

    event_elevation = 0.045 * np.exp(
        -0.5 * ((x + 0.105) / 0.080) ** 2
        -0.5 * ((T - 0.30) / 0.16) ** 2
    )
    medium_term_dip = -0.018 * np.exp(
        -0.5 * ((x - 0.075) / 0.095) ** 2
        -0.5 * ((T - 0.78) / 0.24) ** 2
    )
    downside_ridge = 0.020 * np.exp(
        -0.5 * ((x + 0.235) / 0.070) ** 2
        -0.5 * ((T - 0.12) / 0.085) ** 2
    )

    result = base + event_elevation + medium_term_dip + downside_ridge
    if np.any(result <= 0):
        raise ValueError("the realistic surface produced non-positive volatility.")
    return float(result) if result.ndim == 0 else result


def default_quote_maturities() -> np.ndarray:
    """Return a non-uniform maturity grid resembling listed expiries."""
    return np.array([0.07, 0.12, 0.20, 0.32, 0.50, 0.75, 1.00, 1.35])


def default_quote_log_moneyness() -> np.ndarray:
    """Return a denser central strike grid with sparser wings."""
    return np.array(
        [
            -0.32,
            -0.27,
            -0.22,
            -0.18,
            -0.14,
            -0.10,
            -0.06,
            -0.03,
            0.00,
            0.03,
            0.06,
            0.10,
            0.14,
            0.18,
            0.23,
            0.28,
            0.34,
        ]
    )


def realistic_noise_standard_deviation(
    true_call_price: np.ndarray,
    log_moneyness: np.ndarray,
    maturity: np.ndarray,
) -> np.ndarray:
    """Construct heteroskedastic quote noise resembling bid-ask uncertainty.

    Noise is smallest around the money and larger in the wings and at short
    maturities. A small price-proportional component prevents unrealistically
    precise deep in-the-money quotes.
    """
    prices, x, T = np.broadcast_arrays(
        np.asarray(true_call_price, dtype=float),
        np.asarray(log_moneyness, dtype=float),
        np.asarray(maturity, dtype=float),
    )
    if np.any(prices < 0):
        raise ValueError("true_call_price must be non-negative.")
    if np.any(T <= 0):
        raise ValueError("maturity must be positive.")

    scaled_wing = np.abs(x) / max(float(np.max(np.abs(x))), 1e-12)
    return (
        0.008
        + 0.0010 * prices
        + 0.020 * scaled_wing**1.6
        + 0.010 * np.exp(-T / 0.15)
    )


def generate_realistic_bumpy_quotes(
    random_seed: int = 314159,
    spot: float = 100.0,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    number_of_strike_points: int = 241,
    number_of_time_steps: int = 220,
) -> pd.DataFrame:
    """Generate a noisy rectangular option market from the final truth."""
    data = generate_synthetic_option_data(
        surface_function=realistic_bumpy_local_volatility,
        spot=spot,
        maturities=default_quote_maturities(),
        log_moneyness_values=default_quote_log_moneyness(),
        rate=rate,
        dividend_yield=dividend_yield,
        relative_noise=0.0,
        minimum_noise=0.0,
        random_seed=random_seed,
        number_of_strike_points=number_of_strike_points,
        number_of_time_steps=number_of_time_steps,
    )

    noise_std = realistic_noise_standard_deviation(
        true_call_price=data["true_call_price"].to_numpy(dtype=float),
        log_moneyness=data["log_moneyness"].to_numpy(dtype=float),
        maturity=data["maturity"].to_numpy(dtype=float),
    )
    rng = np.random.default_rng(random_seed)
    noisy = data["true_call_price"].to_numpy(dtype=float) + rng.normal(
        0.0,
        noise_std,
    )
    observed = np.clip(
        noisy,
        data["call_lower_bound"].to_numpy(dtype=float),
        data["call_upper_bound"].to_numpy(dtype=float),
    )
    data["noise_standard_deviation"] = noise_std
    data["observed_call_price"] = observed
    data["noise"] = observed - data["true_call_price"].to_numpy(dtype=float)
    return data.sort_values(["maturity", "log_moneyness"]).reset_index(drop=True)


def elliptical_region_mask(
    point_data: pd.DataFrame,
    x_centre: float,
    maturity_centre: float,
    x_width: float,
    maturity_width: float,
    radius: float = 1.75,
) -> np.ndarray:
    """Return an elliptical feature-region mask."""
    if x_width <= 0 or maturity_width <= 0 or radius <= 0:
        raise ValueError("widths and radius must be positive.")
    required = {"log_moneyness", "maturity"}
    missing = required.difference(point_data.columns)
    if missing:
        raise ValueError(f"point_data is missing columns: {sorted(missing)}")
    distance = (
        ((point_data["log_moneyness"].to_numpy(dtype=float) - x_centre) / x_width) ** 2
        + ((point_data["maturity"].to_numpy(dtype=float) - maturity_centre) / maturity_width) ** 2
    )
    return distance <= radius**2


def multi_feature_recovery_metrics(
    pointwise_results: pd.DataFrame,
    method_columns: Iterable[str] = (
        "raw_dupire",
        "smoothed_dupire",
        "ssvi_dupire",
        "operator_regularized",
    ),
) -> pd.DataFrame:
    """Measure overall, event-region, dip-region, and background accuracy."""
    required = {
        "maturity",
        "log_moneyness",
        "true_local_volatility",
        "shared_valid",
    }
    missing = required.difference(pointwise_results.columns)
    if missing:
        raise ValueError(
            f"pointwise_results is missing columns: {sorted(missing)}"
        )

    data = pointwise_results.loc[
        pointwise_results["shared_valid"].to_numpy(dtype=bool)
    ].reset_index(drop=True)
    if data.empty:
        raise ValueError("no shared valid points are available.")

    primary = elliptical_region_mask(
        data,
        x_centre=-0.105,
        maturity_centre=0.30,
        x_width=0.080,
        maturity_width=0.16,
    )
    secondary = elliptical_region_mask(
        data,
        x_centre=0.075,
        maturity_centre=0.78,
        x_width=0.095,
        maturity_width=0.24,
    )
    background = ~(primary | secondary)
    truth = data["true_local_volatility"].to_numpy(dtype=float)
    base = np.asarray(
        realistic_equity_base_surface(
            data["log_moneyness"].to_numpy(dtype=float),
            data["maturity"].to_numpy(dtype=float),
        ),
        dtype=float,
    )
    true_feature = truth - base

    rows = []
    for method in method_columns:
        if method not in data.columns:
            raise ValueError(f"pointwise_results does not contain '{method}'.")
        estimate = data[method].to_numpy(dtype=float)
        estimated_feature = estimate - base

        overall = reconstruction_metrics(estimate, truth)
        primary_metrics = reconstruction_metrics(estimate, truth, primary)
        secondary_metrics = reconstruction_metrics(estimate, truth, secondary)
        background_metrics = reconstruction_metrics(estimate, truth, background)

        primary_truth_index = np.flatnonzero(primary)[
            int(np.argmax(true_feature[primary]))
        ]
        primary_estimate_index = np.flatnonzero(primary)[
            int(np.argmax(estimated_feature[primary]))
        ]
        secondary_truth_index = np.flatnonzero(secondary)[
            int(np.argmin(true_feature[secondary]))
        ]
        secondary_estimate_index = np.flatnonzero(secondary)[
            int(np.argmin(estimated_feature[secondary]))
        ]

        rows.append(
            {
                "method": method,
                "overall_rmse": overall["rmse"],
                "overall_mae": overall["mae"],
                "primary_event_rmse": primary_metrics["rmse"],
                "secondary_dip_rmse": secondary_metrics["rmse"],
                "background_rmse": background_metrics["rmse"],
                "true_primary_excess": float(true_feature[primary_truth_index]),
                "estimated_primary_excess": float(
                    estimated_feature[primary_estimate_index]
                ),
                "primary_amplitude_error": float(
                    estimated_feature[primary_estimate_index]
                    - true_feature[primary_truth_index]
                ),
                "true_secondary_deviation": float(
                    true_feature[secondary_truth_index]
                ),
                "estimated_secondary_deviation": float(
                    estimated_feature[secondary_estimate_index]
                ),
                "secondary_amplitude_error": float(
                    estimated_feature[secondary_estimate_index]
                    - true_feature[secondary_truth_index]
                ),
                "feature_correlation": float(
                    np.corrcoef(true_feature, estimated_feature)[0, 1]
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("overall_rmse").reset_index(drop=True)


def compare_final_methods(
    quote_data: pd.DataFrame,
    calibration_maturities: np.ndarray,
    calibration_log_moneyness: np.ndarray,
    operator_volatility_surface: np.ndarray,
    spot: float = 100.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Compare only the four requested methods on one common validity mask."""
    table, pointwise, metadata = compare_methods_on_quote_grid(
        quote_data=quote_data,
        regularized_maturities=calibration_maturities,
        regularized_log_moneyness=calibration_log_moneyness,
        regularized_volatility_surface=operator_volatility_surface,
        spot=spot,
        reference_volatility=0.20,
    )
    requested = {
        "raw_dupire",
        "smoothed_dupire",
        "ssvi_dupire",
        "operator_regularized",
    }
    table = table[table["method"].isin(requested)].reset_index(drop=True)
    return table, pointwise, metadata


def run_repeated_noise_comparison(
    clean_quote_data: pd.DataFrame,
    random_seeds: Iterable[int],
    reference_prices: np.ndarray,
    jacobian: np.ndarray,
    weights: np.ndarray,
    reference_log_variance: np.ndarray,
    calibration_maturities: np.ndarray,
    calibration_log_moneyness: np.ndarray,
    regularization_matrix,
    regularization_strength: float,
    spot: float = 100.0,
) -> pd.DataFrame:
    """Run the existing efficient repeated-noise benchmark and keep four methods."""
    results = repeated_noise_benchmark(
        clean_quote_data=clean_quote_data,
        random_seeds=random_seeds,
        reference_prices=reference_prices,
        jacobian=jacobian,
        weights=weights,
        reference_log_variance=reference_log_variance,
        calibration_maturities=calibration_maturities,
        calibration_log_moneyness=calibration_log_moneyness,
        regularization_matrix=regularization_matrix,
        regularization_strength=regularization_strength,
        spot=spot,
    )
    requested = {
        "raw_dupire",
        "smoothed_dupire",
        "ssvi_dupire",
        "operator_regularized_linearized",
    }
    return results[results["method"].isin(requested)].reset_index(drop=True)


def repeated_noise_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Summarise four-method accuracy across independent noise draws."""
    required = {"method", "rmse", "mae", "shared_valid_fraction"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"results is missing columns: {sorted(missing)}")
    return (
        results.groupby("method", as_index=False)
        .agg(
            mean_rmse=("rmse", "mean"),
            sd_rmse=("rmse", "std"),
            minimum_rmse=("rmse", "min"),
            maximum_rmse=("rmse", "max"),
            mean_mae=("mae", "mean"),
            mean_shared_valid_fraction=("shared_valid_fraction", "mean"),
        )
        .sort_values("mean_rmse")
        .reset_index(drop=True)
    )
