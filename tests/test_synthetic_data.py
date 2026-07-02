"""Tests for synthetic local-volatility surfaces and quote generation."""

import numpy as np

from src.data.synthetic_data import generate_synthetic_option_data
from src.surfaces.synthetic_surfaces import (
    bump_surface,
    constant_surface,
    smile_surface,
)


def test_synthetic_surfaces_are_positive_and_preserve_shape() -> None:
    x_grid = np.linspace(-0.4, 0.4, 9)
    maturity_grid = np.linspace(0.1, 2.0, 5)
    x_mesh, maturity_mesh = np.meshgrid(x_grid, maturity_grid)

    for surface_function in [
        constant_surface,
        smile_surface,
        bump_surface,
    ]:
        values = surface_function(x_mesh, maturity_mesh)
        assert values.shape == x_mesh.shape
        assert np.all(values > 0)


def test_synthetic_option_data_is_reproducible_and_valid() -> None:
    maturities = np.array([0.25, 0.50])
    log_moneyness_values = np.array([-0.10, 0.0, 0.10])

    first = generate_synthetic_option_data(
        surface_function=smile_surface,
        maturities=maturities,
        log_moneyness_values=log_moneyness_values,
        relative_noise=0.005,
        minimum_noise=0.01,
        random_seed=7,
        number_of_strike_points=121,
        number_of_time_steps=100,
    )

    second = generate_synthetic_option_data(
        surface_function=smile_surface,
        maturities=maturities,
        log_moneyness_values=log_moneyness_values,
        relative_noise=0.005,
        minimum_noise=0.01,
        random_seed=7,
        number_of_strike_points=121,
        number_of_time_steps=100,
    )

    assert len(first) == 6
    assert np.allclose(
        first["observed_call_price"],
        second["observed_call_price"],
    )
    assert np.all(
        first["observed_call_price"]
        >= first["call_lower_bound"] - 1e-12
    )
    assert np.all(
        first["observed_call_price"]
        <= first["call_upper_bound"] + 1e-12
    )
    assert np.all(first["true_local_volatility"] > 0)
