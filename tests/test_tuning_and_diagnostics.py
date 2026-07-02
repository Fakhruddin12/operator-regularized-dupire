"""Tests for Stage 8 scaling, tuning, and diagnostics."""

import numpy as np
import pandas as pd
from scipy.sparse import eye

from src.evaluation.diagnostics import (
    jacobian_spectrum,
    residual_summary,
    surface_rmse,
    weighted_residual_table,
)
from src.evaluation.tuning import (
    linearized_lambda_sweep,
    select_lambda_by_gcv,
)
from src.regularization.operators import (
    build_regularization_matrix,
)
from src.regularization.scaling import (
    build_nondimensional_regularization_matrix,
    scaled_quadratic_penalty,
)


def _smooth_surface(maturities, x_values):
    x_mesh, maturity_mesh = np.meshgrid(
        x_values,
        maturities,
    )
    return (
        0.04
        * np.cos(np.pi * x_mesh / 0.8)
        * np.exp(-0.5 * maturity_mesh)
    )


def test_nondimensional_scaling_improves_grid_comparability() -> None:
    coarse_T = np.linspace(0.1, 2.0, 5)
    coarse_x = np.linspace(-0.4, 0.4, 9)
    fine_T = np.linspace(0.1, 2.0, 9)
    fine_x = np.linspace(-0.4, 0.4, 17)

    coarse_surface = _smooth_surface(
        coarse_T,
        coarse_x,
    )
    fine_surface = _smooth_surface(
        fine_T,
        fine_x,
    )

    coarse_raw, _, _ = build_regularization_matrix(
        number_of_maturities=coarse_T.size,
        number_of_log_moneyness_points=coarse_x.size,
        maturity_spacing=coarse_T[1] - coarse_T[0],
        log_moneyness_spacing=coarse_x[1] - coarse_x[0],
        alpha_x=1.0,
        alpha_T=1.0,
        beta=0.1,
    )
    fine_raw, _, _ = build_regularization_matrix(
        number_of_maturities=fine_T.size,
        number_of_log_moneyness_points=fine_x.size,
        maturity_spacing=fine_T[1] - fine_T[0],
        log_moneyness_spacing=fine_x[1] - fine_x[0],
        alpha_x=1.0,
        alpha_T=1.0,
        beta=0.1,
    )

    coarse_scaled, _, _, _ = (
        build_nondimensional_regularization_matrix(
            maturities=coarse_T,
            log_moneyness=coarse_x,
            alpha_x=1.0,
            alpha_T=1.0,
            beta=0.1,
        )
    )
    fine_scaled, _, _, _ = (
        build_nondimensional_regularization_matrix(
            maturities=fine_T,
            log_moneyness=fine_x,
            alpha_x=1.0,
            alpha_T=1.0,
            beta=0.1,
        )
    )

    raw_penalties = np.array(
        [
            scaled_quadratic_penalty(
                coarse_surface,
                coarse_raw,
            ),
            scaled_quadratic_penalty(
                fine_surface,
                fine_raw,
            ),
        ]
    )
    scaled_penalties = np.array(
        [
            scaled_quadratic_penalty(
                coarse_surface,
                coarse_scaled,
            ),
            scaled_quadratic_penalty(
                fine_surface,
                fine_scaled,
            ),
        ]
    )

    raw_relative_difference = (
        np.ptp(raw_penalties)
        / np.mean(raw_penalties)
    )
    scaled_relative_difference = (
        np.ptp(scaled_penalties)
        / np.mean(scaled_penalties)
    )

    assert (
        scaled_relative_difference
        < raw_relative_difference
    )
    assert scaled_relative_difference < 0.35


def test_lambda_sweep_matches_identity_solution() -> None:
    jacobian = np.eye(3)
    residual = np.array([1.0, -2.0, 0.5])
    weights = np.ones(3)
    regularization_matrix = eye(3, format="csc")

    results, solutions = linearized_lambda_sweep(
        jacobian=jacobian,
        residual=residual,
        weights=weights,
        regularization_matrix=regularization_matrix,
        lambda_values=[1.0, 4.0],
    )

    assert np.allclose(
        solutions[1.0],
        residual / 2.0,
    )
    assert np.allclose(
        solutions[4.0],
        residual / 5.0,
    )
    assert (
        results.loc[
            results["lambda"] == 4.0,
            "effective_degrees_of_freedom",
        ].iloc[0]
        <
        results.loc[
            results["lambda"] == 1.0,
            "effective_degrees_of_freedom",
        ].iloc[0]
    )
    assert select_lambda_by_gcv(results) in {
        1.0,
        4.0,
    }


def test_residual_diagnostics() -> None:
    quote_data = pd.DataFrame(
        {
            "observed_call_price": [1.0, 2.0],
            "noise_standard_deviation": [0.5, 0.25],
            "maturity": [0.5, 1.0],
            "log_moneyness": [0.0, 0.1],
        }
    )
    predicted = np.array([1.5, 1.5])

    table = weighted_residual_table(
        quote_data,
        predicted,
    )
    summary = residual_summary(table)

    assert np.allclose(
        table["weighted_residual"],
        [1.0, -2.0],
    )
    assert np.isclose(
        summary["weighted_rmse"],
        np.sqrt(2.5),
    )


def test_jacobian_and_surface_diagnostics() -> None:
    jacobian = np.diag([4.0, 2.0, 0.5])
    weights = np.ones(3)

    spectrum = jacobian_spectrum(
        jacobian,
        weights,
    )

    assert spectrum["numerical_rank"] == 3
    assert np.isclose(
        spectrum["condition_number"],
        8.0,
    )
    assert np.isclose(
        spectrum["cumulative_energy"][-1],
        1.0,
    )

    estimate = np.array([[1.0, 2.0]])
    truth = np.array([[1.0, 4.0]])
    assert np.isclose(
        surface_rmse(estimate, truth),
        np.sqrt(2.0),
    )
