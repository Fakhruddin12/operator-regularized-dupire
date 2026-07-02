"""Basic tests for the first project stage."""

import numpy as np

from src.pricing.black_scholes import (
    black_scholes_call,
    black_scholes_put,
    implied_volatility_call,
)
from src.surfaces.grids import surface_to_vector, vector_to_surface


def test_black_scholes_reference_value() -> None:
    price = black_scholes_call(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
    )
    assert np.isclose(price, 10.450583572185565, atol=1e-10)


def test_put_call_parity() -> None:
    spot = 100.0
    strike = 105.0
    maturity = 1.2
    rate = 0.03
    dividend_yield = 0.01
    volatility = 0.24

    call = black_scholes_call(
        spot, strike, maturity, rate, volatility, dividend_yield
    )
    put = black_scholes_put(
        spot, strike, maturity, rate, volatility, dividend_yield
    )

    parity_error = (
        call
        - put
        - spot * np.exp(-dividend_yield * maturity)
        + strike * np.exp(-rate * maturity)
    )
    assert abs(parity_error) < 1e-10


def test_implied_volatility_recovery() -> None:
    true_volatility = 0.31
    price = black_scholes_call(
        spot=100.0,
        strike=110.0,
        maturity=0.75,
        rate=0.02,
        volatility=true_volatility,
        dividend_yield=0.01,
    )
    recovered = implied_volatility_call(
        market_price=price,
        spot=100.0,
        strike=110.0,
        maturity=0.75,
        rate=0.02,
        dividend_yield=0.01,
    )
    assert np.isclose(recovered, true_volatility, atol=1e-10)


def test_surface_round_trip() -> None:
    surface = np.arange(12, dtype=float).reshape(3, 4)
    vector = surface_to_vector(surface)
    restored = vector_to_surface(vector, 3, 4)
    assert np.array_equal(surface, restored)
