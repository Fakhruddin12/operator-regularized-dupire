"""Tests for Stage 9 Gaussian/Laplace uncertainty."""

import numpy as np
from scipy.sparse import csc_matrix, eye

from src.bayesian.laplace import (
    build_laplace_posterior,
    predictive_price_uncertainty,
    surface_uncertainty_summary,
)


def test_laplace_covariance_matches_diagonal_example() -> None:
    map_surface = np.zeros((1, 2))
    jacobian = np.diag([2.0, 3.0])
    weights = np.array([0.5, 2.0])
    regularization_matrix = csc_matrix(
        np.diag([4.0, 5.0])
    )
    lambda_value = 2.0

    result = build_laplace_posterior(
        map_log_variance=map_surface,
        jacobian=jacobian,
        weights=weights,
        regularization_matrix=regularization_matrix,
        regularization_strength=lambda_value,
    )

    expected_precision = np.diag(
        [
            (0.5 * 2.0) ** 2
            + lambda_value * 4.0,
            (2.0 * 3.0) ** 2
            + lambda_value * 5.0,
        ]
    )
    expected_covariance = np.diag(
        1.0 / np.diag(expected_precision)
    )

    assert np.allclose(
        result["precision"],
        expected_precision,
    )
    assert np.allclose(
        result["covariance"],
        expected_covariance,
    )


def test_stronger_regularization_reduces_uncertainty() -> None:
    map_surface = np.zeros((1, 3))
    jacobian = np.eye(3)
    weights = np.ones(3)
    regularization_matrix = eye(
        3,
        format="csc",
    )

    weak = build_laplace_posterior(
        map_log_variance=map_surface,
        jacobian=jacobian,
        weights=weights,
        regularization_matrix=regularization_matrix,
        regularization_strength=1.0,
    )
    strong = build_laplace_posterior(
        map_log_variance=map_surface,
        jacobian=jacobian,
        weights=weights,
        regularization_matrix=regularization_matrix,
        regularization_strength=10.0,
    )

    assert np.all(
        np.diag(strong["covariance"])
        < np.diag(weak["covariance"])
    )


def test_surface_summary_transforms_log_variance_correctly() -> None:
    map_surface = np.log(
        np.array([[0.20**2, 0.30**2]])
    )
    covariance = np.diag([0.04, 0.09])

    summary = surface_uncertainty_summary(
        map_log_variance=map_surface,
        posterior_covariance=covariance,
        maturities=np.array([1.0]),
        log_moneyness=np.array([-0.1, 0.1]),
        credibility=0.95,
    )

    assert np.allclose(
        summary["volatility_median"],
        [0.20, 0.30],
    )
    assert np.all(
        summary["volatility_lower"]
        < summary["volatility_median"]
    )
    assert np.all(
        summary["volatility_median"]
        < summary["volatility_upper"]
    )
    assert np.all(
        summary["volatility_mean"]
        > summary["volatility_median"]
    )


def test_predictive_variance_includes_observation_noise() -> None:
    map_prices = np.array([10.0, 5.0])
    jacobian = np.eye(2)
    covariance = np.diag([0.25, 1.00])
    noise = np.array([0.5, 0.25])

    result = predictive_price_uncertainty(
        map_prices=map_prices,
        jacobian=jacobian,
        posterior_covariance=covariance,
        noise_standard_deviation=noise,
    )

    assert np.allclose(
        result["latent_variance"],
        [0.25, 1.00],
    )
    assert np.allclose(
        result["observed_variance"],
        [
            0.25 + 0.5**2,
            1.00 + 0.25**2,
        ],
    )
    assert np.all(
        result["observed_standard_deviation"]
        > result["latent_standard_deviation"]
    )
