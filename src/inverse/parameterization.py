"""Parameterisation of local variance on a log-moneyness/maturity grid.

The inverse problem works with

    u(x,T) = log(sigma_loc(x,T)^2).

Using log variance guarantees positive local volatility because

    sigma_loc(x,T) = exp(u(x,T) / 2).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from src.pricing.black_scholes import forward_price


def reference_log_variance_surface(
    maturities: np.ndarray,
    log_moneyness: np.ndarray,
    reference_volatility: float = 0.20,
) -> np.ndarray:
    """Return a constant reference surface ``log(reference_volatility**2)``."""
    maturity_values = np.asarray(maturities, dtype=float)
    x_values = np.asarray(log_moneyness, dtype=float)

    if maturity_values.ndim != 1 or x_values.ndim != 1:
        raise ValueError("maturities and log_moneyness must be one-dimensional.")
    if maturity_values.size < 2 or x_values.size < 2:
        raise ValueError("both grids must contain at least two points.")
    if np.any(np.diff(maturity_values) <= 0):
        raise ValueError("maturities must be strictly increasing.")
    if np.any(np.diff(x_values) <= 0):
        raise ValueError("log_moneyness must be strictly increasing.")
    if reference_volatility <= 0:
        raise ValueError("reference_volatility must be positive.")

    return np.full(
        (maturity_values.size, x_values.size),
        np.log(reference_volatility**2),
        dtype=float,
    )


def local_volatility_from_log_variance(
    log_variance_surface: np.ndarray,
) -> np.ndarray:
    """Convert ``u = log(sigma^2)`` into positive local volatility."""
    log_variance = np.asarray(log_variance_surface, dtype=float)

    if np.any(~np.isfinite(log_variance)):
        raise ValueError("log_variance_surface contains non-finite values.")

    return np.exp(0.5 * log_variance)


def make_local_volatility_function(
    log_variance_surface: np.ndarray,
    maturities: np.ndarray,
    log_moneyness: np.ndarray,
    spot: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> Callable[[np.ndarray, float], np.ndarray]:
    """Create the ``sigma(K,T)`` callback required by the forward PDE.

    The unknown surface is stored on an ``(x,T)`` grid. The PDE instead asks
    for local volatility at strikes and time. This function performs

        K -> x = log(K/F(T)) -> interpolate u(x,T) -> exp(u/2).

    Outside the calibration grid, coordinates are clipped to the nearest grid
    boundary. This gives flat extrapolation, consistent with the natural
    boundary behaviour used by the regulariser.
    """
    log_variance = np.asarray(log_variance_surface, dtype=float)
    maturity_values = np.asarray(maturities, dtype=float)
    x_values = np.asarray(log_moneyness, dtype=float)

    expected_shape = (maturity_values.size, x_values.size)
    if log_variance.shape != expected_shape:
        raise ValueError(
            f"log_variance_surface has shape {log_variance.shape}; "
            f"expected {expected_shape}."
        )
    if spot <= 0:
        raise ValueError("spot must be positive.")
    if np.any(np.diff(maturity_values) <= 0):
        raise ValueError("maturities must be strictly increasing.")
    if np.any(np.diff(x_values) <= 0):
        raise ValueError("log_moneyness must be strictly increasing.")
    if np.any(~np.isfinite(log_variance)):
        raise ValueError("log_variance_surface contains non-finite values.")

    interpolator = RegularGridInterpolator(
        (maturity_values, x_values),
        log_variance,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    minimum_maturity = float(maturity_values[0])
    maximum_maturity = float(maturity_values[-1])
    minimum_x = float(x_values[0])
    maximum_x = float(x_values[-1])

    def local_volatility(strikes: np.ndarray, maturity: float) -> np.ndarray:
        strike_values = np.asarray(strikes, dtype=float)

        if np.any(strike_values <= 0):
            raise ValueError("PDE interior strikes must be positive.")
        if maturity < 0:
            raise ValueError("maturity must be non-negative.")

        forward = forward_price(
            spot=spot,
            maturity=maturity,
            rate=rate,
            dividend_yield=dividend_yield,
        )
        x_query = np.log(strike_values / forward)

        clipped_maturity = np.clip(
            maturity,
            minimum_maturity,
            maximum_maturity,
        )
        clipped_x = np.clip(x_query, minimum_x, maximum_x)

        query_points = np.column_stack(
            [
                np.full(strike_values.size, clipped_maturity),
                clipped_x.reshape(-1),
            ]
        )
        interpolated_log_variance = interpolator(query_points).reshape(
            strike_values.shape
        )

        return np.exp(0.5 * interpolated_log_variance)

    return local_volatility
