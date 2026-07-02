"""Tests for the Stage 6 weighted linearised inverse solver."""

import numpy as np
from scipy.sparse import eye

from src.data.synthetic_data import (
    build_quote_grid,
    generate_synthetic_option_data,
)
from src.inverse.linearized_inverse import (
    finite_difference_jacobian,
    price_quotes_from_log_variance,
    run_linearized_calibration,
    solve_weighted_linearized_problem,
)
from src.inverse.parameterization import (
    local_volatility_from_log_variance,
    make_local_volatility_function,
    reference_log_variance_surface,
)
from src.regularization.operators import build_regularization_matrix
from src.surfaces.synthetic_surfaces import smile_surface


def test_log_variance_parameterization_preserves_constant_volatility() -> None:
    maturities = np.array([0.10, 0.50, 1.00])
    x_values = np.array([-0.30, 0.0, 0.30])

    log_variance = reference_log_variance_surface(
        maturities=maturities,
        log_moneyness=x_values,
        reference_volatility=0.20,
    )

    assert np.allclose(
        local_volatility_from_log_variance(log_variance),
        0.20,
    )

    local_volatility = make_local_volatility_function(
        log_variance_surface=log_variance,
        maturities=maturities,
        log_moneyness=x_values,
        spot=100.0,
    )

    assert np.allclose(
        local_volatility(
            np.array([70.0, 100.0, 140.0]),
            0.25,
        ),
        0.20,
    )


def test_weighted_linearized_solver_reduces_data_misfit() -> None:
    jacobian = np.array(
        [
            [1.0, 0.2, 0.0],
            [0.1, 1.0, 0.3],
            [0.0, 0.2, 1.0],
            [0.7, 0.0, 0.4],
        ]
    )
    true_correction = np.array([0.10, -0.05, 0.08])
    residual = jacobian @ true_correction
    weights = np.array([1.0, 2.0, 1.5, 0.8])

    result = solve_weighted_linearized_problem(
        jacobian=jacobian,
        residual=residual,
        weights=weights,
        regularization_matrix=eye(3, format="csc"),
        regularization_strength=1e-6,
    )

    zero_misfit = np.sum((weights * residual) ** 2)

    assert result["weighted_data_misfit"] < zero_misfit
    assert np.allclose(
        result["correction"],
        true_correction,
        atol=1e-5,
    )


def test_finite_difference_jacobian_predicts_small_price_change() -> None:
    calibration_maturities = np.array([0.10, 0.60, 1.20])
    calibration_x = np.array([-0.30, -0.10, 0.10, 0.30])
    reference = reference_log_variance_surface(
        calibration_maturities,
        calibration_x,
        reference_volatility=0.20,
    )

    quote_data = build_quote_grid(
        spot=100.0,
        maturities=np.array([0.25, 0.75, 1.20]),
        log_moneyness_values=np.array([-0.20, 0.0, 0.20]),
    )

    base_prices, jacobian = finite_difference_jacobian(
        reference_log_variance=reference,
        calibration_maturities=calibration_maturities,
        calibration_log_moneyness=calibration_x,
        quote_data=quote_data,
        spot=100.0,
        finite_difference_step=1e-3,
        number_of_strike_points=101,
        number_of_time_steps=80,
    )

    direction = np.linspace(-1.0, 1.0, reference.size)
    direction /= np.linalg.norm(direction)
    step_size = 2e-3

    perturbed = (
        reference.reshape(-1, order="C")
        + step_size * direction
    ).reshape(reference.shape, order="C")

    actual_prices = price_quotes_from_log_variance(
        log_variance_surface=perturbed,
        calibration_maturities=calibration_maturities,
        calibration_log_moneyness=calibration_x,
        quote_data=quote_data,
        spot=100.0,
        number_of_strike_points=101,
        number_of_time_steps=80,
    )
    linear_prediction = (
        base_prices + jacobian @ (step_size * direction)
    )

    error = np.linalg.norm(actual_prices - linear_prediction)
    actual_change = np.linalg.norm(actual_prices - base_prices)

    assert error < 0.05 * actual_change + 1e-8


def test_end_to_end_linearized_calibration_improves_repricing() -> None:
    quote_data = generate_synthetic_option_data(
        surface_function=smile_surface,
        spot=100.0,
        maturities=np.array([0.20, 0.60, 1.20]),
        log_moneyness_values=np.array([-0.20, 0.0, 0.20]),
        relative_noise=0.001,
        minimum_noise=0.01,
        random_seed=9,
        number_of_strike_points=121,
        number_of_time_steps=100,
    )

    calibration_maturities = np.linspace(0.20, 1.20, 3)
    calibration_x = np.linspace(-0.25, 0.25, 5)
    reference = reference_log_variance_surface(
        calibration_maturities,
        calibration_x,
        reference_volatility=0.20,
    )

    regularization_matrix, _, _ = build_regularization_matrix(
        number_of_maturities=calibration_maturities.size,
        number_of_log_moneyness_points=calibration_x.size,
        maturity_spacing=calibration_maturities[1]
        - calibration_maturities[0],
        log_moneyness_spacing=calibration_x[1] - calibration_x[0],
        alpha_x=0.01,
        alpha_T=0.005,
        beta=1e-3,
    )

    result = run_linearized_calibration(
        reference_log_variance=reference,
        calibration_maturities=calibration_maturities,
        calibration_log_moneyness=calibration_x,
        quote_data=quote_data,
        regularization_matrix=regularization_matrix,
        regularization_strength=300.0,
        spot=100.0,
        finite_difference_step=1e-3,
        number_of_strike_points=121,
        number_of_time_steps=100,
    )

    assert (
        result["linearized_weighted_rmse"]
        < result["reference_weighted_rmse"]
    )
    assert (
        result["nonlinear_weighted_rmse"]
        < result["reference_weighted_rmse"]
    )
