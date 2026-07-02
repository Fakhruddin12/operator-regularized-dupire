"""Generate controlled synthetic option-market data."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from src.pricing.black_scholes import forward_price
from src.pricing.local_vol_pde import (
    interpolate_call_prices,
    solve_forward_dupire,
)
from src.surfaces.synthetic_surfaces import (
    make_strike_time_local_volatility,
)


def build_quote_grid(
    spot: float,
    maturities: np.ndarray,
    log_moneyness_values: np.ndarray,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> pd.DataFrame:
    """Create every requested maturity and log-moneyness combination."""
    if spot <= 0:
        raise ValueError("spot must be positive.")

    maturities = np.asarray(maturities, dtype=float)
    log_moneyness_values = np.asarray(log_moneyness_values, dtype=float)

    if maturities.ndim != 1 or log_moneyness_values.ndim != 1:
        raise ValueError("maturities and log_moneyness_values must be 1D arrays.")
    if np.any(maturities <= 0):
        raise ValueError("all quote maturities must be positive.")

    x_mesh, maturity_mesh = np.meshgrid(
        log_moneyness_values,
        maturities,
    )

    forwards = forward_price(
        spot=spot,
        maturity=maturity_mesh,
        rate=rate,
        dividend_yield=dividend_yield,
    )
    strikes = forwards * np.exp(x_mesh)

    return pd.DataFrame(
        {
            "maturity": maturity_mesh.reshape(-1),
            "log_moneyness": x_mesh.reshape(-1),
            "forward": np.asarray(forwards).reshape(-1),
            "strike": strikes.reshape(-1),
        }
    )


def generate_synthetic_option_data(
    surface_function: Callable[
        [np.ndarray | float, np.ndarray | float],
        np.ndarray | float,
    ],
    spot: float = 100.0,
    maturities: np.ndarray | None = None,
    log_moneyness_values: np.ndarray | None = None,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    relative_noise: float = 0.005,
    minimum_noise: float = 0.01,
    random_seed: int = 1234,
    strike_max: float | None = None,
    number_of_strike_points: int = 301,
    number_of_time_steps: int = 300,
) -> pd.DataFrame:
    """Generate clean and noisy synthetic European call quotes.

    The procedure is:

    1. create a sparse quote grid in (x,T);
    2. convert it to strikes;
    3. price the true local-volatility surface with the forward PDE;
    4. add reproducible Gaussian quote noise;
    5. clip noisy prices to European call no-arbitrage bounds.
    """
    if maturities is None:
        maturities = np.array([0.10, 0.25, 0.50, 1.00, 1.50, 2.00])

    if log_moneyness_values is None:
        log_moneyness_values = np.linspace(-0.30, 0.30, 13)

    if relative_noise < 0:
        raise ValueError("relative_noise must be non-negative.")
    if minimum_noise < 0:
        raise ValueError("minimum_noise must be non-negative.")

    quote_data = build_quote_grid(
        spot=spot,
        maturities=np.asarray(maturities, dtype=float),
        log_moneyness_values=np.asarray(log_moneyness_values, dtype=float),
        rate=rate,
        dividend_yield=dividend_yield,
    )

    pde_local_volatility = make_strike_time_local_volatility(
        surface_function=surface_function,
        spot=spot,
        rate=rate,
        dividend_yield=dividend_yield,
    )

    maximum_maturity = float(quote_data["maturity"].max())

    if strike_max is None:
        strike_max = max(
            3.0 * spot,
            1.25 * float(quote_data["strike"].max()),
        )

    strike_grid, time_grid, call_surface = solve_forward_dupire(
        spot=spot,
        local_volatility=pde_local_volatility,
        max_maturity=maximum_maturity,
        rate=rate,
        dividend_yield=dividend_yield,
        strike_max=strike_max,
        number_of_strike_points=number_of_strike_points,
        number_of_time_steps=number_of_time_steps,
    )

    true_prices = interpolate_call_prices(
        strike_grid=strike_grid,
        time_grid=time_grid,
        call_surface=call_surface,
        strikes=quote_data["strike"].to_numpy(),
        maturities=quote_data["maturity"].to_numpy(),
    )

    true_local_volatility = np.asarray(
        surface_function(
            quote_data["log_moneyness"].to_numpy(),
            quote_data["maturity"].to_numpy(),
        ),
        dtype=float,
    )

    noise_standard_deviation = np.maximum(
        relative_noise * np.asarray(true_prices),
        minimum_noise,
    )

    random_generator = np.random.default_rng(random_seed)
    noise = random_generator.normal(
        loc=0.0,
        scale=noise_standard_deviation,
    )

    raw_observed_prices = np.asarray(true_prices) + noise

    discounted_spot = spot * np.exp(
        -dividend_yield * quote_data["maturity"].to_numpy()
    )
    discounted_strike = quote_data["strike"].to_numpy() * np.exp(
        -rate * quote_data["maturity"].to_numpy()
    )

    lower_bound = np.maximum(
        discounted_spot - discounted_strike,
        0.0,
    )
    upper_bound = discounted_spot

    observed_prices = np.clip(
        raw_observed_prices,
        lower_bound,
        upper_bound,
    )

    quote_data["true_local_volatility"] = true_local_volatility
    quote_data["true_call_price"] = np.asarray(true_prices)
    quote_data["noise_standard_deviation"] = noise_standard_deviation
    quote_data["noise"] = observed_prices - np.asarray(true_prices)
    quote_data["observed_call_price"] = observed_prices
    quote_data["call_lower_bound"] = lower_bound
    quote_data["call_upper_bound"] = upper_bound

    return quote_data.sort_values(
        ["maturity", "log_moneyness"]
    ).reset_index(drop=True)


def save_synthetic_option_data(
    data: pd.DataFrame,
    file_path: str | Path,
) -> Path:
    """Save synthetic option data to CSV and return the resulting path."""
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return output_path
