"""Tests for the revised Stage 10 SSVI benchmark."""

import numpy as np
import pandas as pd

from src.evaluation.benchmark import (
    interpolate_surface_to_quotes,
    reconstruction_metrics,
    resample_observed_quotes,
)
from src.evaluation.ssvi_dupire import (
    dupire_local_variance_from_derivatives,
    ssvi_strike_derivatives,
    ssvi_total_variance,
)


def test_ssvi_atm_total_variance_equals_theta() -> None:
    theta = np.array([0.02, 0.05, 0.10])
    result = ssvi_total_variance(
        log_moneyness=np.zeros(3),
        theta=theta,
        rho=-0.3,
        eta=0.5,
        gamma=0.5,
    )
    assert np.allclose(result, theta)


def test_ssvi_analytic_strike_derivatives() -> None:
    k = np.array([-0.15, 0.0, 0.20])
    theta = 0.06
    parameters = dict(rho=-0.25, eta=0.6, gamma=0.4)
    w, first, second = ssvi_strike_derivatives(
        k, theta, **parameters
    )
    step = 1e-5
    plus = ssvi_total_variance(k + step, theta, **parameters)
    minus = ssvi_total_variance(k - step, theta, **parameters)
    numerical_first = (plus - minus) / (2.0 * step)
    numerical_second = (plus - 2.0 * w + minus) / step**2
    assert np.allclose(first, numerical_first, rtol=1e-5, atol=1e-7)
    assert np.allclose(second, numerical_second, rtol=2e-4, atol=2e-6)


def test_total_variance_dupire_recovers_constant_volatility() -> None:
    k = np.linspace(-0.3, 0.3, 7)
    volatility = 0.20
    result = dupire_local_variance_from_derivatives(
        log_moneyness=k,
        total_variance=np.full_like(k, volatility**2),
        time_derivative=np.full_like(k, volatility**2),
        first_strike_derivative=np.zeros_like(k),
        second_strike_derivative=np.zeros_like(k),
    )
    assert np.all(result["valid_mask"])
    assert np.allclose(result["local_volatility"], volatility)


def test_benchmark_helpers() -> None:
    estimate = np.array([1.0, 3.0, 5.0])
    truth = np.array([1.0, 2.0, 4.0])
    mask = np.array([True, True, False])
    result = reconstruction_metrics(estimate, truth, mask)
    assert result["number_of_evaluation_points"] == 2
    assert np.isclose(result["rmse"], np.sqrt(0.5))

    maturities = np.array([0.5, 1.0])
    x_values = np.array([-0.1, 0.1])
    surface = np.array([[0.20, 0.21], [0.22, 0.23]])
    quote_data = pd.DataFrame(
        {"maturity": [0.5, 1.0], "log_moneyness": [-0.1, 0.1]}
    )
    interpolated = interpolate_surface_to_quotes(
        maturities, x_values, surface, quote_data
    )
    assert np.allclose(interpolated, [0.20, 0.23])

    data = pd.DataFrame(
        {
            "true_call_price": [1.0, 2.0],
            "noise_standard_deviation": [0.1, 0.2],
            "call_lower_bound": [0.0, 0.0],
            "call_upper_bound": [1.5, 2.5],
        }
    )
    first = resample_observed_quotes(data, 10)
    second = resample_observed_quotes(data, 10)
    assert np.allclose(first["observed_call_price"], second["observed_call_price"])
