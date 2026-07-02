"""Tests for raw and smoothed direct Dupire estimators."""

import numpy as np

from src.data.synthetic_data import generate_synthetic_option_data
from src.dupire.direct_dupire import (
    dupire_from_price_surface,
    raw_dupire_from_quotes,
    smoothed_dupire_from_quotes,
)
from src.pricing.black_scholes import black_scholes_call, forward_price
from src.surfaces.synthetic_surfaces import smile_surface


def test_dupire_recovers_constant_black_scholes_volatility() -> None:
    spot = 100.0
    rate = 0.03
    dividend_yield = 0.01
    volatility = 0.20

    maturities = np.linspace(0.05, 2.0, 50)
    log_moneyness = np.linspace(-0.40, 0.40, 81)

    maturity_mesh, x_mesh = np.meshgrid(
        maturities,
        log_moneyness,
        indexing="ij",
    )

    forwards = forward_price(
        spot=spot,
        maturity=maturity_mesh,
        rate=rate,
        dividend_yield=dividend_yield,
    )
    strikes = forwards * np.exp(x_mesh)

    call_prices = black_scholes_call(
        spot=spot,
        strike=strikes,
        maturity=maturity_mesh,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )

    result = dupire_from_price_surface(
        price_surface=call_prices,
        maturities=maturities,
        log_moneyness=log_moneyness,
        dividend_yield=dividend_yield,
    )

    recovered = result["local_volatility"]

    central_region = (
        (maturity_mesh >= 0.15)
        & (maturity_mesh <= 1.90)
        & (x_mesh >= -0.25)
        & (x_mesh <= 0.25)
        & np.isfinite(recovered)
    )

    rmse = np.sqrt(
        np.mean((recovered[central_region] - volatility) ** 2)
    )

    assert rmse < 0.002


def test_smoothing_improves_noisy_synthetic_recovery() -> None:
    quote_data = generate_synthetic_option_data(
        surface_function=smile_surface,
        relative_noise=0.005,
        minimum_noise=0.01,
        random_seed=1234,
        number_of_strike_points=201,
        number_of_time_steps=150,
    )

    raw_result = raw_dupire_from_quotes(quote_data)
    smoothed_result = smoothed_dupire_from_quotes(quote_data)

    truth = quote_data[
        ["maturity", "log_moneyness", "true_local_volatility"]
    ]

    raw_result = raw_result.merge(
        truth,
        on=["maturity", "log_moneyness"],
        how="left",
    )
    smoothed_result = smoothed_result.merge(
        truth,
        on=["maturity", "log_moneyness"],
        how="left",
    )

    shared_valid = (
        raw_result["valid_dupire"].to_numpy()
        & smoothed_result["valid_dupire"].to_numpy()
    )

    raw_rmse = np.sqrt(
        np.mean(
            (
                raw_result.loc[shared_valid, "local_volatility"]
                - raw_result.loc[shared_valid, "true_local_volatility"]
            ) ** 2
        )
    )

    smoothed_rmse = np.sqrt(
        np.mean(
            (
                smoothed_result.loc[shared_valid, "local_volatility"]
                - smoothed_result.loc[shared_valid, "true_local_volatility"]
            ) ** 2
        )
    )

    assert shared_valid.mean() > 0.85
    assert smoothed_rmse < raw_rmse
