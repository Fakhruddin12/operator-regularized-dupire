"""Synthetic local-volatility surfaces used in controlled experiments.

Each surface is defined in log-moneyness and maturity coordinates:

    x = log(K / F(T)).

The functions accept scalars or NumPy-broadcastable arrays.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from src.pricing.black_scholes import forward_price


def constant_surface(
    log_moneyness: np.ndarray | float,
    maturity: np.ndarray | float,
    level: float = 0.20,
) -> np.ndarray | float:
    """Return a constant local-volatility surface."""
    if level <= 0:
        raise ValueError("level must be positive.")

    x_array, maturity_array = np.broadcast_arrays(
        np.asarray(log_moneyness, dtype=float),
        np.asarray(maturity, dtype=float),
    )

    if np.any(maturity_array < 0):
        raise ValueError("maturity must be non-negative.")

    result = np.full_like(x_array, level, dtype=float)
    return float(result) if result.ndim == 0 else result


def smile_surface(
    log_moneyness: np.ndarray | float,
    maturity: np.ndarray | float,
    base_level: float = 0.20,
    smile_strength: float = 0.08,
    short_maturity_strength: float = 0.03,
    maturity_decay: float = 3.0,
) -> np.ndarray | float:
    """Return a smooth smile-shaped local-volatility surface.

    The formula is

        sigma(x, T)
        = base_level
          + smile_strength * x^2
          + short_maturity_strength * exp(-maturity_decay * T).
    """
    x_array, maturity_array = np.broadcast_arrays(
        np.asarray(log_moneyness, dtype=float),
        np.asarray(maturity, dtype=float),
    )

    if np.any(maturity_array < 0):
        raise ValueError("maturity must be non-negative.")
    if base_level <= 0:
        raise ValueError("base_level must be positive.")
    if smile_strength < 0 or short_maturity_strength < 0:
        raise ValueError("surface strengths must be non-negative.")
    if maturity_decay < 0:
        raise ValueError("maturity_decay must be non-negative.")

    result = (
        base_level
        + smile_strength * x_array**2
        + short_maturity_strength * np.exp(-maturity_decay * maturity_array)
    )

    if np.any(result <= 0):
        raise ValueError("the smile surface produced non-positive volatility.")

    return float(result) if result.ndim == 0 else result


def bump_surface(
    log_moneyness: np.ndarray | float,
    maturity: np.ndarray | float,
    base_level: float = 0.20,
    bump_height: float = 0.05,
    bump_x_centre: float = 0.0,
    bump_maturity_centre: float = 0.60,
    bump_x_width: float = 0.12,
    bump_maturity_width: float = 0.25,
) -> np.ndarray | float:
    """Return a smooth localised bump on top of a constant base surface."""
    x_array, maturity_array = np.broadcast_arrays(
        np.asarray(log_moneyness, dtype=float),
        np.asarray(maturity, dtype=float),
    )

    if np.any(maturity_array < 0):
        raise ValueError("maturity must be non-negative.")
    if base_level <= 0:
        raise ValueError("base_level must be positive.")
    if bump_height < 0:
        raise ValueError("bump_height must be non-negative.")
    if bump_x_width <= 0 or bump_maturity_width <= 0:
        raise ValueError("bump widths must be positive.")

    exponent = (
        -0.5 * ((x_array - bump_x_centre) / bump_x_width) ** 2
        -0.5
        * ((maturity_array - bump_maturity_centre) / bump_maturity_width) ** 2
    )

    result = base_level + bump_height * np.exp(exponent)

    if np.any(result <= 0):
        raise ValueError("the bump surface produced non-positive volatility.")

    return float(result) if result.ndim == 0 else result


def make_strike_time_local_volatility(
    surface_function: Callable[
        [np.ndarray | float, np.ndarray | float],
        np.ndarray | float,
    ],
    spot: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> Callable[[np.ndarray, float], np.ndarray]:
    """Convert a surface sigma(x,T) into the sigma(K,T) callback used by the PDE.

    The PDE solver asks for volatility at strike and maturity. The project
    stores surfaces in log-moneyness, so this wrapper performs

        K -> x = log(K / F(T)) -> sigma(x, T).
    """
    if spot <= 0:
        raise ValueError("spot must be positive.")

    def strike_time_surface(
        strikes: np.ndarray,
        maturity: float,
    ) -> np.ndarray:
        strikes_array = np.asarray(strikes, dtype=float)

        if np.any(strikes_array <= 0):
            raise ValueError("PDE interior strikes must be positive.")
        if maturity < 0:
            raise ValueError("maturity must be non-negative.")

        forward = forward_price(
            spot=spot,
            maturity=maturity,
            rate=rate,
            dividend_yield=dividend_yield,
        )
        log_moneyness = np.log(strikes_array / forward)

        values = np.asarray(
            surface_function(log_moneyness, maturity),
            dtype=float,
        )
        return np.broadcast_to(values, strikes_array.shape).copy()

    return strike_time_surface
