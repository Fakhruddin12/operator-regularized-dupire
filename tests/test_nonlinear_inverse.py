"""Tests for the damped nonlinear Gauss-Newton calibration."""

import numpy as np
from scipy.sparse import eye

from src.data.synthetic_data import (
    generate_synthetic_option_data,
)
from src.inverse.linearized_inverse import (
    run_linearized_calibration,
)
from src.inverse.nonlinear_inverse import (
    nonlinear_objective,
    run_gauss_newton_calibration,
    solve_gauss_newton_step,
)
from src.inverse.parameterization import (
    reference_log_variance_surface,
)
from src.regularization.operators import (
    build_regularization_matrix,
)
from src.surfaces.synthetic_surfaces import (
    smile_surface,
)


def test_nonlinear_objective_uses_total_correction() -> None:
    predicted = np.array([1.0, 2.0])
    observed = np.array([1.5, 1.5])
    weights = np.array([2.0, 1.0])
    correction = np.array([1.0, -2.0])
    regularization_matrix = eye(2, format="csc")

    result = nonlinear_objective(
        predicted_prices=predicted,
        observed_prices=observed,
        weights=weights,
        total_correction=correction,
        regularization_matrix=regularization_matrix,
        regularization_strength=3.0,
    )

    expected_data_misfit = (
        (2.0 * -0.5) ** 2
        + (1.0 * 0.5) ** 2
    )
    expected_penalty = 1.0**2 + (-2.0) ** 2

    assert np.isclose(
        result["weighted_data_misfit"],
        expected_data_misfit,
    )
    assert np.isclose(
        result["regularization_penalty"],
        expected_penalty,
    )
    assert np.isclose(
        result["objective"],
        expected_data_misfit
        + 3.0 * expected_penalty,
    )


def test_gauss_newton_step_accounts_for_existing_correction() -> None:
    jacobian = np.eye(2)
    residual = np.array([1.0, -1.0])
    weights = np.ones(2)
    current_correction = np.array([0.5, 0.5])
    regularization_matrix = eye(2, format="csc")
    regularization_strength = 2.0

    result = solve_gauss_newton_step(
        jacobian=jacobian,
        residual=residual,
        weights=weights,
        current_total_correction=current_correction,
        regularization_matrix=regularization_matrix,
        regularization_strength=regularization_strength,
        damping=0.0,
    )

    expected_step = (
        residual
        - regularization_strength * current_correction
    ) / (1.0 + regularization_strength)

    assert np.allclose(result["step"], expected_step)


def _small_problem():
    quote_data = generate_synthetic_option_data(
        surface_function=smile_surface,
        spot=100.0,
        maturities=np.array([0.25, 0.75, 1.50]),
        log_moneyness_values=np.array(
            [-0.20, -0.10, 0.0, 0.10, 0.20]
        ),
        relative_noise=0.002,
        minimum_noise=0.01,
        random_seed=9,
        number_of_strike_points=121,
        number_of_time_steps=90,
    )

    maturities = np.array([0.25, 0.75, 1.50])
    x_values = np.array(
        [-0.25, -0.125, 0.0, 0.125, 0.25]
    )
    reference = reference_log_variance_surface(
        maturities=maturities,
        log_moneyness=x_values,
        reference_volatility=0.20,
    )

    regularization_matrix, _, _ = (
        build_regularization_matrix(
            number_of_maturities=maturities.size,
            number_of_log_moneyness_points=x_values.size,
            maturity_spacing=0.625,
            log_moneyness_spacing=0.125,
            alpha_x=0.003,
            alpha_T=0.001,
            beta=1e-4,
        )
    )

    return (
        quote_data,
        maturities,
        x_values,
        reference,
        regularization_matrix,
    )


def test_gauss_newton_history_is_monotone() -> None:
    (
        quote_data,
        maturities,
        x_values,
        reference,
        regularization_matrix,
    ) = _small_problem()

    result = run_gauss_newton_calibration(
        reference_log_variance=reference,
        initial_log_variance=reference,
        calibration_maturities=maturities,
        calibration_log_moneyness=x_values,
        quote_data=quote_data,
        regularization_matrix=regularization_matrix,
        regularization_strength=300.0,
        spot=100.0,
        finite_difference_step=1e-3,
        number_of_strike_points=101,
        number_of_time_steps=75,
        maximum_iterations=2,
        initial_damping=1e-2,
        verbose=False,
    )

    accepted_history = result["history"][
        result["history"]["accepted"]
    ]
    objectives = accepted_history["objective"].to_numpy()

    assert np.all(np.diff(objectives) < 0)
    assert result["final_weighted_rmse"] < (
        accepted_history.iloc[0]["weighted_rmse"]
    )


def test_gauss_newton_improves_linearized_start() -> None:
    (
        quote_data,
        maturities,
        x_values,
        reference,
        regularization_matrix,
    ) = _small_problem()

    linearized = run_linearized_calibration(
        reference_log_variance=reference,
        calibration_maturities=maturities,
        calibration_log_moneyness=x_values,
        quote_data=quote_data,
        regularization_matrix=regularization_matrix,
        regularization_strength=300.0,
        spot=100.0,
        finite_difference_step=1e-3,
        number_of_strike_points=101,
        number_of_time_steps=75,
        verbose=False,
    )

    nonlinear = run_gauss_newton_calibration(
        reference_log_variance=reference,
        initial_log_variance=linearized[
            "estimated_log_variance"
        ],
        calibration_maturities=maturities,
        calibration_log_moneyness=x_values,
        quote_data=quote_data,
        regularization_matrix=regularization_matrix,
        regularization_strength=300.0,
        spot=100.0,
        finite_difference_step=1e-3,
        number_of_strike_points=101,
        number_of_time_steps=75,
        maximum_iterations=2,
        initial_damping=1e-2,
        verbose=False,
    )

    assert nonlinear["final_objective"] <= (
        nonlinear["history"].iloc[0]["objective"]
    )
    assert nonlinear["final_weighted_rmse"] <= (
        linearized["nonlinear_weighted_rmse"] + 1e-8
    )
