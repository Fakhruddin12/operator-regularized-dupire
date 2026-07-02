"""Finite-difference pricing under a local-volatility model.

This module solves the forward Dupire PDE for European call prices:

    dC/dT
    = 0.5 * sigma_loc(K, T)^2 * K^2 * d2C/dK2
      - (r - q) * K * dC/dK
      - q * C.

The output is a call-price surface across strike and maturity.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Union

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve

LocalVolatilityInput = Union[
    float,
    int,
    Callable[[np.ndarray, float], np.ndarray | float],
]


def _evaluate_local_volatility(
    local_volatility: LocalVolatilityInput,
    strikes: np.ndarray,
    maturity: float,
) -> np.ndarray:
    """Evaluate local volatility on the supplied strike points.

    Parameters
    ----------
    local_volatility:
        Either one positive constant or a function with inputs
        ``(strikes, maturity)``.
    strikes:
        One-dimensional strike array.
    maturity:
        Time at which local volatility is required.

    Returns
    -------
    numpy.ndarray
        Positive volatility values with the same shape as ``strikes``.
    """
    if callable(local_volatility):
        values = np.asarray(local_volatility(strikes, maturity), dtype=float)
        values = np.broadcast_to(values, strikes.shape).copy()
    else:
        values = np.full_like(strikes, float(local_volatility), dtype=float)

    if np.any(~np.isfinite(values)):
        raise ValueError("local volatility contains non-finite values.")
    if np.any(values <= 0):
        raise ValueError("local volatility must be strictly positive.")

    return values


def solve_forward_dupire(
    spot: float,
    local_volatility: LocalVolatilityInput,
    max_maturity: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    strike_max: float | None = None,
    number_of_strike_points: int = 401,
    number_of_time_steps: int = 400,
    theta: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the forward Dupire PDE on a uniform strike-time grid.

    Parameters
    ----------
    spot:
        Current underlying price.
    local_volatility:
        Constant volatility or a function ``sigma(strikes, maturity)``.
    max_maturity:
        Largest maturity on the returned grid.
    rate:
        Continuously compounded risk-free rate.
    dividend_yield:
        Continuously compounded dividend yield.
    strike_max:
        Upper strike boundary. The default is ``3 * spot``.
    number_of_strike_points:
        Number of strike grid points, including both boundaries.
    number_of_time_steps:
        Number of time steps between zero and ``max_maturity``.
    theta:
        Time-stepping parameter. ``0.5`` is Crank-Nicolson and ``1.0`` is
        fully implicit Euler.

    Returns
    -------
    strike_grid:
        One-dimensional strike grid.
    time_grid:
        One-dimensional maturity grid, starting at zero.
    call_surface:
        Matrix with shape ``(number_of_time_steps + 1,
        number_of_strike_points)``. Rows are maturities and columns are strikes.

    Notes
    -----
    The initial and boundary conditions are:

    ``C(K, 0) = max(spot - K, 0)``

    ``C(0, T) = spot * exp(-qT)``

    ``C(strike_max, T) = 0``

    The last condition is an approximation, so ``strike_max`` should be far
    enough above the spot price.
    """
    if spot <= 0:
        raise ValueError("spot must be positive.")
    if max_maturity <= 0:
        raise ValueError("max_maturity must be positive.")
    if number_of_strike_points < 3:
        raise ValueError("number_of_strike_points must be at least 3.")
    if number_of_time_steps < 1:
        raise ValueError("number_of_time_steps must be at least 1.")
    if not 0.5 <= theta <= 1.0:
        raise ValueError("theta must lie between 0.5 and 1.0.")

    if strike_max is None:
        strike_max = 3.0 * spot
    if strike_max <= spot:
        raise ValueError("strike_max must be greater than spot.")

    strike_grid = np.linspace(
        0.0,
        strike_max,
        number_of_strike_points,
        dtype=float,
    )
    time_grid = np.linspace(
        0.0,
        max_maturity,
        number_of_time_steps + 1,
        dtype=float,
    )

    strike_step = strike_grid[1] - strike_grid[0]
    time_step = time_grid[1] - time_grid[0]

    call_surface = np.empty(
        (number_of_time_steps + 1, number_of_strike_points),
        dtype=float,
    )

    # At T = 0, a call is worth its payoff.
    call_surface[0, :] = np.maximum(spot - strike_grid, 0.0)

    interior_strikes = strike_grid[1:-1]
    number_of_interior_points = interior_strikes.size
    identity = eye(number_of_interior_points, format="csc")

    for time_index in range(number_of_time_steps):
        old_time = time_grid[time_index]
        new_time = time_grid[time_index + 1]
        midpoint_time = 0.5 * (old_time + new_time)

        sigma = _evaluate_local_volatility(
            local_volatility,
            interior_strikes,
            midpoint_time,
        )

        # Finite-difference coefficients for the Dupire differential operator.
        lower = (
            0.5 * sigma**2 * interior_strikes**2 / strike_step**2
            + (rate - dividend_yield) * interior_strikes / (2.0 * strike_step)
        )
        centre = (
            -sigma**2 * interior_strikes**2 / strike_step**2
            - dividend_yield
        )
        upper = (
            0.5 * sigma**2 * interior_strikes**2 / strike_step**2
            - (rate - dividend_yield) * interior_strikes / (2.0 * strike_step)
        )

        operator = diags(
            diagonals=[lower[1:], centre, upper[:-1]],
            offsets=[-1, 0, 1],
            format="csc",
        )

        left_matrix = identity - theta * time_step * operator
        right_matrix = identity + (1.0 - theta) * time_step * operator

        right_hand_side = right_matrix @ call_surface[time_index, 1:-1]

        old_left_boundary = spot * np.exp(-dividend_yield * old_time)
        new_left_boundary = spot * np.exp(-dividend_yield * new_time)

        old_right_boundary = 0.0
        new_right_boundary = 0.0

        # Add the omitted boundary contributions to the right-hand side.
        right_hand_side[0] += time_step * (
            (1.0 - theta) * lower[0] * old_left_boundary
            + theta * lower[0] * new_left_boundary
        )
        right_hand_side[-1] += time_step * (
            (1.0 - theta) * upper[-1] * old_right_boundary
            + theta * upper[-1] * new_right_boundary
        )

        call_surface[time_index + 1, 0] = new_left_boundary
        call_surface[time_index + 1, -1] = new_right_boundary
        call_surface[time_index + 1, 1:-1] = spsolve(
            left_matrix,
            right_hand_side,
        )

    return strike_grid, time_grid, call_surface


def interpolate_call_prices(
    strike_grid: np.ndarray,
    time_grid: np.ndarray,
    call_surface: np.ndarray,
    strikes: np.ndarray | float,
    maturities: np.ndarray | float,
) -> np.ndarray | float:
    """Interpolate PDE prices at requested strike-maturity pairs."""
    strike_grid = np.asarray(strike_grid, dtype=float)
    time_grid = np.asarray(time_grid, dtype=float)
    call_surface = np.asarray(call_surface, dtype=float)

    expected_shape = (time_grid.size, strike_grid.size)
    if call_surface.shape != expected_shape:
        raise ValueError(
            f"call_surface has shape {call_surface.shape}; "
            f"expected {expected_shape}."
        )

    strikes_array, maturities_array = np.broadcast_arrays(
        np.asarray(strikes, dtype=float),
        np.asarray(maturities, dtype=float),
    )

    interpolator = RegularGridInterpolator(
        (time_grid, strike_grid),
        call_surface,
        method="linear",
        bounds_error=True,
    )

    query_points = np.column_stack(
        [
            maturities_array.reshape(-1),
            strikes_array.reshape(-1),
        ]
    )
    prices = interpolator(query_points).reshape(strikes_array.shape)

    return float(prices) if prices.ndim == 0 else prices
