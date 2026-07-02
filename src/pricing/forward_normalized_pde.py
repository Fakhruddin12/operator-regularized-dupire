"""Forward-normalized Dupire PDE for real option-market calibration.

For deterministic discounting and forwards, define

    c(x,T) = C(K,T) / (D(T) F(T)),
    x      = log(K / F(T)).

The European call-price Dupire equation becomes

    dc/dT = 0.5 * sigma_loc(x,T)^2 * (d2c/dx2 - dc/dx).

This removes the interest-rate and dividend terms from the PDE. Market-specific
forwards and discount factors are applied only when normalized prices are
converted back into currency prices.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve


LocalVolatilityXInput = float | int | Callable[[np.ndarray, float], np.ndarray | float]


def normalized_black_call_price(
    log_moneyness: np.ndarray | float,
    maturity: np.ndarray | float,
    volatility: np.ndarray | float,
) -> np.ndarray | float:
    """Return Black call price divided by ``D(T)F(T)``.

    The formula is useful for validating the normalized PDE and for converting
    model prices into implied volatility without separately specifying spot,
    rates, or dividends.
    """
    from scipy.stats import norm

    x, T, sigma = np.broadcast_arrays(
        np.asarray(log_moneyness, dtype=float),
        np.asarray(maturity, dtype=float),
        np.asarray(volatility, dtype=float),
    )

    if np.any(T <= 0):
        raise ValueError("maturity must be positive.")
    if np.any(sigma <= 0):
        raise ValueError("volatility must be positive.")

    total_standard_deviation = sigma * np.sqrt(T)
    d1 = (-x + 0.5 * sigma**2 * T) / total_standard_deviation
    d2 = d1 - total_standard_deviation
    result = norm.cdf(d1) - np.exp(x) * norm.cdf(d2)

    return float(result) if result.ndim == 0 else result


def _evaluate_local_volatility(
    local_volatility: LocalVolatilityXInput,
    log_moneyness: np.ndarray,
    maturity: float,
) -> np.ndarray:
    """Evaluate positive local volatility on the PDE interior grid."""
    if callable(local_volatility):
        values = np.asarray(
            local_volatility(log_moneyness, maturity),
            dtype=float,
        )
        values = np.broadcast_to(values, log_moneyness.shape).copy()
    else:
        values = np.full_like(
            log_moneyness,
            float(local_volatility),
            dtype=float,
        )

    if np.any(~np.isfinite(values)):
        raise ValueError("local volatility contains non-finite values.")
    if np.any(values <= 0):
        raise ValueError("local volatility must be strictly positive.")

    return values


def solve_forward_normalized_dupire(
    local_volatility: LocalVolatilityXInput,
    max_maturity: float,
    x_min: float = -1.0,
    x_max: float = 1.0,
    number_of_x_points: int = 241,
    number_of_time_steps: int = 180,
    theta: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the forward-normalized Dupire PDE on a uniform ``(x,T)`` grid.

    Initial and boundary conditions are

    ``c(x,0) = max(1-exp(x), 0)``,

    ``c(x_min,T) = 1-exp(x_min)``, and ``c(x_max,T)=0``.
    """
    if max_maturity <= 0:
        raise ValueError("max_maturity must be positive.")
    if x_min >= 0:
        raise ValueError("x_min must be negative.")
    if x_max <= 0:
        raise ValueError("x_max must be positive.")
    if x_max <= x_min:
        raise ValueError("x_max must exceed x_min.")
    if number_of_x_points < 5:
        raise ValueError("number_of_x_points must be at least 5.")
    if number_of_time_steps < 1:
        raise ValueError("number_of_time_steps must be at least 1.")
    if not 0.5 <= theta <= 1.0:
        raise ValueError("theta must lie between 0.5 and 1.0.")

    x_grid = np.linspace(
        x_min,
        x_max,
        number_of_x_points,
        dtype=float,
    )
    time_grid = np.linspace(
        0.0,
        max_maturity,
        number_of_time_steps + 1,
        dtype=float,
    )

    dx = float(x_grid[1] - x_grid[0])
    dt = float(time_grid[1] - time_grid[0])

    normalized_calls = np.empty(
        (time_grid.size, x_grid.size),
        dtype=float,
    )
    normalized_calls[0] = np.maximum(
        1.0 - np.exp(x_grid),
        0.0,
    )

    interior_x = x_grid[1:-1]
    identity = eye(interior_x.size, format="csc")

    left_boundary = float(1.0 - np.exp(x_min))
    right_boundary = 0.0

    for time_index in range(number_of_time_steps):
        old_time = float(time_grid[time_index])
        new_time = float(time_grid[time_index + 1])
        midpoint_time = 0.5 * (old_time + new_time)

        sigma = _evaluate_local_volatility(
            local_volatility,
            interior_x,
            midpoint_time,
        )
        half_variance = 0.5 * sigma**2

        lower = half_variance * (
            1.0 / dx**2 + 1.0 / (2.0 * dx)
        )
        centre = -2.0 * half_variance / dx**2
        upper = half_variance * (
            1.0 / dx**2 - 1.0 / (2.0 * dx)
        )

        operator = diags(
            diagonals=[lower[1:], centre, upper[:-1]],
            offsets=[-1, 0, 1],
            format="csc",
        )

        left_matrix = identity - theta * dt * operator
        right_matrix = identity + (1.0 - theta) * dt * operator

        right_hand_side = (
            right_matrix @ normalized_calls[time_index, 1:-1]
        )
        right_hand_side[0] += dt * lower[0] * left_boundary
        right_hand_side[-1] += dt * upper[-1] * right_boundary

        normalized_calls[time_index + 1, 0] = left_boundary
        normalized_calls[time_index + 1, -1] = right_boundary
        normalized_calls[time_index + 1, 1:-1] = spsolve(
            left_matrix,
            right_hand_side,
        )

    return x_grid, time_grid, normalized_calls


def interpolate_normalized_call_prices(
    x_grid: np.ndarray,
    time_grid: np.ndarray,
    normalized_call_surface: np.ndarray,
    log_moneyness: np.ndarray | float,
    maturities: np.ndarray | float,
) -> np.ndarray | float:
    """Interpolate normalized call prices at requested ``(x,T)`` points."""
    x_values = np.asarray(x_grid, dtype=float)
    time_values = np.asarray(time_grid, dtype=float)
    surface = np.asarray(normalized_call_surface, dtype=float)

    expected_shape = (time_values.size, x_values.size)
    if surface.shape != expected_shape:
        raise ValueError(
            f"normalized_call_surface has shape {surface.shape}; "
            f"expected {expected_shape}."
        )

    x_query, maturity_query = np.broadcast_arrays(
        np.asarray(log_moneyness, dtype=float),
        np.asarray(maturities, dtype=float),
    )

    interpolator = RegularGridInterpolator(
        (time_values, x_values),
        surface,
        method="linear",
        bounds_error=True,
    )
    points = np.column_stack(
        [
            maturity_query.reshape(-1),
            x_query.reshape(-1),
        ]
    )
    result = interpolator(points).reshape(x_query.shape)

    return float(result) if result.ndim == 0 else result
