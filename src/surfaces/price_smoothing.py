"""Simple price-surface smoothing for the smoothed-Dupire baseline.

This module uses sequential weighted smoothing splines:

1. smooth across log-moneyness at each maturity;
2. smooth across maturity at each log-moneyness location.

This is a transparent baseline rather than the final regularised inverse method.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import UnivariateSpline


def smooth_price_surface(
    price_surface: np.ndarray,
    maturities: np.ndarray,
    log_moneyness: np.ndarray,
    noise_standard_deviation: np.ndarray | None = None,
    x_smoothing_multiplier: float = 0.5,
    time_smoothing_multiplier: float = 1.0,
    spline_degree: int = 3,
) -> np.ndarray:
    """Smooth a rectangular call-price surface with sequential splines.

    Parameters
    ----------
    price_surface:
        Matrix with rows corresponding to maturities and columns corresponding
        to log-moneyness.
    maturities:
        Strictly increasing one-dimensional maturity grid.
    log_moneyness:
        Strictly increasing one-dimensional log-moneyness grid.
    noise_standard_deviation:
        Optional matrix of positive quote-noise standard deviations. When
        supplied, its reciprocals are used as spline weights.
    x_smoothing_multiplier:
        Controls smoothing across log-moneyness. Larger values permit more
        smoothing.
    time_smoothing_multiplier:
        Controls smoothing across maturity. Larger values permit more
        smoothing.
    spline_degree:
        Spline degree, reduced automatically when a grid is small.

    Returns
    -------
    numpy.ndarray
        Smoothed price matrix with the same shape as ``price_surface``.
    """
    prices = np.asarray(price_surface, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    x_values = np.asarray(log_moneyness, dtype=float)

    expected_shape = (maturities.size, x_values.size)
    if prices.shape != expected_shape:
        raise ValueError(
            f"price_surface has shape {prices.shape}; expected {expected_shape}."
        )
    if maturities.ndim != 1 or x_values.ndim != 1:
        raise ValueError("maturities and log_moneyness must be one-dimensional.")
    if np.any(np.diff(maturities) <= 0):
        raise ValueError("maturities must be strictly increasing.")
    if np.any(np.diff(x_values) <= 0):
        raise ValueError("log_moneyness must be strictly increasing.")
    if np.any(~np.isfinite(prices)):
        raise ValueError("price_surface contains non-finite values.")
    if x_smoothing_multiplier < 0 or time_smoothing_multiplier < 0:
        raise ValueError("smoothing multipliers must be non-negative.")

    if noise_standard_deviation is None:
        noise = np.ones_like(prices)
    else:
        noise = np.asarray(noise_standard_deviation, dtype=float)
        if noise.shape != expected_shape:
            raise ValueError(
                "noise_standard_deviation must have the same shape as "
                "price_surface."
            )
        if np.any(~np.isfinite(noise)) or np.any(noise <= 0):
            raise ValueError(
                "noise_standard_deviation must contain finite positive values."
            )

    x_degree = min(spline_degree, x_values.size - 1)
    time_degree = min(spline_degree, maturities.size - 1)

    # First pass: smooth each maturity slice across log-moneyness.
    x_smoothed = np.empty_like(prices)
    x_smoothing_target = x_smoothing_multiplier * x_values.size

    for maturity_index in range(maturities.size):
        spline = UnivariateSpline(
            x=x_values,
            y=prices[maturity_index],
            w=1.0 / noise[maturity_index],
            k=x_degree,
            s=x_smoothing_target,
        )
        x_smoothed[maturity_index] = spline(x_values)

    # Second pass: smooth each log-moneyness column across maturity.
    fully_smoothed = np.empty_like(prices)
    time_smoothing_target = time_smoothing_multiplier * maturities.size

    for x_index in range(x_values.size):
        spline = UnivariateSpline(
            x=maturities,
            y=x_smoothed[:, x_index],
            w=1.0 / noise[:, x_index],
            k=time_degree,
            s=time_smoothing_target,
        )
        fully_smoothed[:, x_index] = spline(maturities)

    return fully_smoothed
