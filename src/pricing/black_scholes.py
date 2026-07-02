"""Black-Scholes pricing and implied-volatility utilities."""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

ArrayLike = Union[float, int, np.ndarray]


def _validate_positive(name: str, value: ArrayLike) -> None:
    """Raise a clear error when a scalar or array contains non-positive values."""
    array = np.asarray(value, dtype=float)
    if np.any(array <= 0):
        raise ValueError(f"{name} must contain only positive values.")


def _return_scalar_when_scalar(array: np.ndarray) -> np.ndarray | float:
    """Return a Python float for zero-dimensional output, otherwise an array."""
    return float(array) if array.ndim == 0 else array


def forward_price(
    spot: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike = 0.0,
    dividend_yield: ArrayLike = 0.0,
) -> np.ndarray | float:
    """Return the continuously compounded forward price S exp((r-q)T)."""
    _validate_positive("spot", spot)
    maturity_array = np.asarray(maturity, dtype=float)
    if np.any(maturity_array < 0):
        raise ValueError("maturity must be non-negative.")

    result = np.asarray(spot, dtype=float) * np.exp(
        (np.asarray(rate, dtype=float) - np.asarray(dividend_yield, dtype=float))
        * maturity_array
    )
    result = np.asarray(result)
    return _return_scalar_when_scalar(result)


def black_scholes_call(
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    volatility: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
) -> np.ndarray | float:
    """Price a European call under the Black-Scholes model.

    Parameters may be scalars or NumPy-broadcastable arrays. At maturity zero,
    the function returns the intrinsic value.
    """
    _validate_positive("spot", spot)
    _validate_positive("strike", strike)
    _validate_positive("volatility", volatility)

    spot_b, strike_b, maturity_b, rate_b, vol_b, div_b = np.broadcast_arrays(
        np.asarray(spot, dtype=float),
        np.asarray(strike, dtype=float),
        np.asarray(maturity, dtype=float),
        np.asarray(rate, dtype=float),
        np.asarray(volatility, dtype=float),
        np.asarray(dividend_yield, dtype=float),
    )

    if np.any(maturity_b < 0):
        raise ValueError("maturity must be non-negative.")

    intrinsic = np.maximum(spot_b - strike_b, 0.0)
    positive_time = maturity_b > 0

    # Use a safe maturity only to avoid division by zero while evaluating arrays.
    safe_maturity = np.where(positive_time, maturity_b, 1.0)
    sqrt_t = np.sqrt(safe_maturity)

    d1 = (
        np.log(spot_b / strike_b)
        + (rate_b - div_b + 0.5 * vol_b**2) * safe_maturity
    ) / (vol_b * sqrt_t)
    d2 = d1 - vol_b * sqrt_t

    positive_time_price = (
        spot_b * np.exp(-div_b * maturity_b) * norm.cdf(d1)
        - strike_b * np.exp(-rate_b * maturity_b) * norm.cdf(d2)
    )

    result = np.where(positive_time, positive_time_price, intrinsic)
    return _return_scalar_when_scalar(np.asarray(result))


def black_scholes_put(
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    volatility: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
) -> np.ndarray | float:
    """Price a European put using put-call parity."""
    call = black_scholes_call(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )

    result = (
        np.asarray(call, dtype=float)
        - np.asarray(spot, dtype=float)
        * np.exp(-np.asarray(dividend_yield, dtype=float) * np.asarray(maturity, dtype=float))
        + np.asarray(strike, dtype=float)
        * np.exp(-np.asarray(rate, dtype=float) * np.asarray(maturity, dtype=float))
    )
    return _return_scalar_when_scalar(np.asarray(result))


def call_price_bounds(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float = 0.0,
) -> tuple[float, float]:
    """Return no-arbitrage lower and upper bounds for a European call."""
    _validate_positive("spot", spot)
    _validate_positive("strike", strike)
    if maturity < 0:
        raise ValueError("maturity must be non-negative.")

    discounted_spot = spot * np.exp(-dividend_yield * maturity)
    discounted_strike = strike * np.exp(-rate * maturity)
    lower = max(discounted_spot - discounted_strike, 0.0)
    upper = discounted_spot
    return float(lower), float(upper)


def implied_volatility_call(
    market_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float = 0.0,
    volatility_lower: float = 1e-8,
    volatility_upper: float = 5.0,
) -> float:
    """Recover Black-Scholes implied volatility for one European call.

    Brent's method is used because it is robust and does not require a starting
    derivative. The market price must lie inside the no-arbitrage call bounds.
    """
    if maturity <= 0:
        raise ValueError("Implied volatility requires strictly positive maturity.")

    lower_bound, upper_bound = call_price_bounds(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        dividend_yield=dividend_yield,
    )

    tolerance = 1e-12
    if market_price < lower_bound - tolerance or market_price > upper_bound + tolerance:
        raise ValueError(
            "market_price is outside the European call no-arbitrage bounds "
            f"[{lower_bound:.8f}, {upper_bound:.8f}]."
        )

    def pricing_error(volatility: float) -> float:
        return float(
            black_scholes_call(
                spot=spot,
                strike=strike,
                maturity=maturity,
                rate=rate,
                volatility=volatility,
                dividend_yield=dividend_yield,
            )
            - market_price
        )

    low_error = pricing_error(volatility_lower)
    high_error = pricing_error(volatility_upper)

    if abs(low_error) < tolerance:
        return volatility_lower
    if abs(high_error) < tolerance:
        return volatility_upper
    if low_error * high_error > 0:
        raise RuntimeError(
            "Could not bracket the implied volatility. Increase volatility_upper."
        )

    return float(
        brentq(
            pricing_error,
            volatility_lower,
            volatility_upper,
            xtol=1e-12,
            rtol=1e-12,
            maxiter=200,
        )
    )


def black_scholes_vega(
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    volatility: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
) -> np.ndarray | float:
    """Return call/put vega: derivative of price with respect to volatility."""
    _validate_positive("spot", spot)
    _validate_positive("strike", strike)
    _validate_positive("volatility", volatility)

    spot_b, strike_b, maturity_b, rate_b, vol_b, div_b = np.broadcast_arrays(
        np.asarray(spot, dtype=float),
        np.asarray(strike, dtype=float),
        np.asarray(maturity, dtype=float),
        np.asarray(rate, dtype=float),
        np.asarray(volatility, dtype=float),
        np.asarray(dividend_yield, dtype=float),
    )

    if np.any(maturity_b < 0):
        raise ValueError("maturity must be non-negative.")

    positive_time = maturity_b > 0
    safe_maturity = np.where(positive_time, maturity_b, 1.0)
    sqrt_t = np.sqrt(safe_maturity)

    d1 = (
        np.log(spot_b / strike_b)
        + (rate_b - div_b + 0.5 * vol_b**2) * safe_maturity
    ) / (vol_b * sqrt_t)

    positive_time_vega = (
        spot_b
        * np.exp(-div_b * maturity_b)
        * norm.pdf(d1)
        * np.sqrt(np.where(positive_time, maturity_b, 0.0))
    )

    result = np.where(positive_time, positive_time_vega, 0.0)
    return _return_scalar_when_scalar(np.asarray(result))
