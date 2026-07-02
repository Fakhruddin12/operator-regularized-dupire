"""Common-grid synthetic benchmarks for local-volatility reconstruction."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from src.dupire.direct_dupire import (
    raw_dupire_from_quotes,
    smoothed_dupire_from_quotes,
)
from src.evaluation.ssvi_dupire import (
    ssvi_dupire_from_quotes,
)
from src.inverse.linearized_inverse import (
    solve_weighted_linearized_problem,
)


def interpolate_surface_to_quotes(
    maturities: np.ndarray,
    log_moneyness: np.ndarray,
    surface: np.ndarray,
    quote_data: pd.DataFrame,
) -> np.ndarray:
    """Interpolate a rectangular surface to quote locations."""
    maturity_values = np.asarray(maturities, dtype=float)
    x_values = np.asarray(log_moneyness, dtype=float)
    surface_values = np.asarray(surface, dtype=float)
    expected_shape = (maturity_values.size, x_values.size)
    if surface_values.shape != expected_shape:
        raise ValueError(
            f"surface has shape {surface_values.shape}; expected {expected_shape}."
        )
    interpolator = RegularGridInterpolator(
        (maturity_values, x_values),
        surface_values,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    points = np.column_stack(
        [
            quote_data["maturity"].to_numpy(dtype=float),
            quote_data["log_moneyness"].to_numpy(dtype=float),
        ]
    )
    return np.asarray(interpolator(points), dtype=float)


def reconstruction_metrics(
    estimate: np.ndarray,
    truth: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Calculate reconstruction error metrics on a specified common mask."""
    estimated = np.asarray(estimate, dtype=float).reshape(-1)
    true_values = np.asarray(truth, dtype=float).reshape(-1)
    if estimated.size != true_values.size:
        raise ValueError("estimate and truth must have equal size.")
    finite_mask = np.isfinite(estimated) & np.isfinite(true_values)
    if valid_mask is None:
        mask = finite_mask
    else:
        supplied_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if supplied_mask.size != estimated.size:
            raise ValueError("valid_mask has the wrong size.")
        mask = finite_mask & supplied_mask
    if not np.any(mask):
        raise ValueError("no valid points remain for evaluation.")
    error = estimated[mask] - true_values[mask]
    return {
        "number_of_evaluation_points": int(np.sum(mask)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "maximum_absolute_error": float(np.max(np.abs(error))),
        "bias": float(np.mean(error)),
    }


def compare_methods_on_quote_grid(
    quote_data: pd.DataFrame,
    regularized_maturities: np.ndarray,
    regularized_log_moneyness: np.ndarray,
    regularized_volatility_surface: np.ndarray,
    spot: float,
    reference_volatility: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Compare all methods on one shared quote-grid mask."""
    if "true_local_volatility" not in quote_data.columns:
        raise ValueError("quote_data must contain true_local_volatility.")

    sorted_quotes = quote_data.sort_values(
        ["maturity", "log_moneyness"]
    ).reset_index(drop=True)
    raw = raw_dupire_from_quotes(sorted_quotes).sort_values(
        ["maturity", "log_moneyness"]
    ).reset_index(drop=True)
    smoothed = smoothed_dupire_from_quotes(sorted_quotes).sort_values(
        ["maturity", "log_moneyness"]
    ).reset_index(drop=True)
    ssvi, ssvi_fit, implied_data = ssvi_dupire_from_quotes(
        quote_data=sorted_quotes,
        spot=spot,
    )
    ssvi = ssvi.sort_values(
        ["maturity", "log_moneyness"]
    ).reset_index(drop=True)

    regularized_values = interpolate_surface_to_quotes(
        maturities=regularized_maturities,
        log_moneyness=regularized_log_moneyness,
        surface=regularized_volatility_surface,
        quote_data=sorted_quotes,
    )
    truth = sorted_quotes["true_local_volatility"].to_numpy(dtype=float)
    values = {
        "constant_reference": np.full_like(truth, reference_volatility),
        "raw_dupire": raw["local_volatility"].to_numpy(dtype=float),
        "smoothed_dupire": smoothed["local_volatility"].to_numpy(dtype=float),
        "ssvi_dupire": ssvi["local_volatility"].to_numpy(dtype=float),
        "operator_regularized": regularized_values,
    }
    validity = {
        "constant_reference": np.ones(truth.size, dtype=bool),
        "raw_dupire": raw["valid_dupire"].to_numpy(dtype=bool),
        "smoothed_dupire": smoothed["valid_dupire"].to_numpy(dtype=bool),
        "ssvi_dupire": ssvi["valid_dupire"].to_numpy(dtype=bool),
        "operator_regularized": np.isfinite(regularized_values),
    }
    shared_mask = np.logical_and.reduce(list(validity.values()))

    rows = []
    for method, estimate in values.items():
        metrics = reconstruction_metrics(estimate, truth, shared_mask)
        metrics["method"] = method
        metrics["method_valid_fraction"] = float(np.mean(validity[method]))
        rows.append(metrics)

    pointwise = sorted_quotes[
        ["maturity", "log_moneyness", "true_local_volatility"]
    ].copy()
    for method, estimate in values.items():
        pointwise[method] = estimate
    pointwise["shared_valid"] = shared_mask

    metadata = {
        "ssvi_fit": ssvi_fit,
        "ssvi_implied_data": implied_data,
    }
    return (
        pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True),
        pointwise,
        metadata,
    )


def resample_observed_quotes(
    clean_quote_data: pd.DataFrame,
    random_seed: int,
) -> pd.DataFrame:
    """Create another noisy market from the same clean synthetic prices."""
    data = clean_quote_data.copy()
    rng = np.random.default_rng(random_seed)
    raw_observed = (
        data["true_call_price"].to_numpy(dtype=float)
        + rng.normal(
            0.0,
            data["noise_standard_deviation"].to_numpy(dtype=float),
        )
    )
    observed = np.clip(
        raw_observed,
        data["call_lower_bound"].to_numpy(dtype=float),
        data["call_upper_bound"].to_numpy(dtype=float),
    )
    data["observed_call_price"] = observed
    data["noise"] = observed - data["true_call_price"].to_numpy(dtype=float)
    return data


def repeated_noise_benchmark(
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
    spot: float,
) -> pd.DataFrame:
    """Repeat the comparison over independent noise realisations."""
    reference_price_values = np.asarray(reference_prices, dtype=float).reshape(-1)
    J = np.asarray(jacobian, dtype=float)
    weight_values = np.asarray(weights, dtype=float).reshape(-1)
    reference_surface = np.asarray(reference_log_variance, dtype=float)
    rows = []

    for seed in random_seeds:
        noisy_quotes = resample_observed_quotes(clean_quote_data, int(seed))
        sorted_quotes = noisy_quotes.sort_values(
            ["maturity", "log_moneyness"]
        ).reset_index(drop=True)
        raw = raw_dupire_from_quotes(sorted_quotes).sort_values(
            ["maturity", "log_moneyness"]
        ).reset_index(drop=True)
        smoothed = smoothed_dupire_from_quotes(sorted_quotes).sort_values(
            ["maturity", "log_moneyness"]
        ).reset_index(drop=True)
        ssvi, _, _ = ssvi_dupire_from_quotes(sorted_quotes, spot=spot)
        ssvi = ssvi.sort_values(
            ["maturity", "log_moneyness"]
        ).reset_index(drop=True)

        observed = sorted_quotes["observed_call_price"].to_numpy(dtype=float)
        solution = solve_weighted_linearized_problem(
            jacobian=J,
            residual=observed - reference_price_values,
            weights=weight_values,
            regularization_matrix=regularization_matrix,
            regularization_strength=regularization_strength,
        )
        estimated_log_variance = reference_surface + solution["correction"].reshape(
            reference_surface.shape, order="C"
        )
        operator_values = interpolate_surface_to_quotes(
            calibration_maturities,
            calibration_log_moneyness,
            np.exp(0.5 * estimated_log_variance),
            sorted_quotes,
        )
        truth = sorted_quotes["true_local_volatility"].to_numpy(dtype=float)
        method_values = {
            "constant_reference": np.full_like(truth, 0.20),
            "raw_dupire": raw["local_volatility"].to_numpy(dtype=float),
            "smoothed_dupire": smoothed["local_volatility"].to_numpy(dtype=float),
            "ssvi_dupire": ssvi["local_volatility"].to_numpy(dtype=float),
            "operator_regularized_linearized": operator_values,
        }
        method_valid = {
            "constant_reference": np.ones(truth.size, dtype=bool),
            "raw_dupire": raw["valid_dupire"].to_numpy(dtype=bool),
            "smoothed_dupire": smoothed["valid_dupire"].to_numpy(dtype=bool),
            "ssvi_dupire": ssvi["valid_dupire"].to_numpy(dtype=bool),
            "operator_regularized_linearized": np.isfinite(operator_values),
        }
        shared_mask = np.logical_and.reduce(list(method_valid.values()))
        for method, estimate in method_values.items():
            metrics = reconstruction_metrics(estimate, truth, shared_mask)
            metrics["seed"] = int(seed)
            metrics["method"] = method
            metrics["shared_valid_fraction"] = float(np.mean(shared_mask))
            rows.append(metrics)
    return pd.DataFrame(rows)
