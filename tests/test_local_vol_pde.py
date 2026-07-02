"""Tests for the forward local-volatility PDE pricer."""

import numpy as np

from src.pricing.black_scholes import black_scholes_call
from src.pricing.local_vol_pde import (
    interpolate_call_prices,
    solve_forward_dupire,
)


def test_initial_and_boundary_conditions() -> None:
    spot = 100.0

    strike_grid, time_grid, call_surface = solve_forward_dupire(
        spot=spot,
        local_volatility=0.20,
        max_maturity=0.5,
        strike_max=300.0,
        number_of_strike_points=101,
        number_of_time_steps=50,
    )

    assert np.allclose(
        call_surface[0],
        np.maximum(spot - strike_grid, 0.0),
    )
    assert np.allclose(call_surface[:, -1], 0.0)
    assert np.allclose(call_surface[:, 0], spot)


def test_constant_local_volatility_matches_black_scholes() -> None:
    spot = 100.0
    rate = 0.05
    dividend_yield = 0.0
    volatility = 0.20

    strike_grid, time_grid, call_surface = solve_forward_dupire(
        spot=spot,
        local_volatility=volatility,
        max_maturity=1.0,
        rate=rate,
        dividend_yield=dividend_yield,
        strike_max=300.0,
        number_of_strike_points=301,
        number_of_time_steps=300,
    )

    test_strikes = np.array([80.0, 100.0, 120.0])
    test_maturities = np.array([0.25, 0.50, 1.00])

    strike_mesh, maturity_mesh = np.meshgrid(
        test_strikes,
        test_maturities,
    )

    pde_prices = interpolate_call_prices(
        strike_grid,
        time_grid,
        call_surface,
        strike_mesh,
        maturity_mesh,
    )

    black_scholes_prices = black_scholes_call(
        spot=spot,
        strike=strike_mesh,
        maturity=maturity_mesh,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )

    maximum_error = np.max(np.abs(pde_prices - black_scholes_prices))
    assert maximum_error < 0.01
