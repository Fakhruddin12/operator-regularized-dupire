"""Tests for Stage 5 regularisation operators and potentials."""

import numpy as np
from scipy.sparse.linalg import eigsh

from src.regularization.operators import (
    build_2d_difference_operators,
    build_regularization_matrix,
    regularization_components,
)
from src.regularization.potentials import (
    combine_potentials,
    confidence_potential,
    quote_confidence_surface,
    wing_potential,
)


def test_difference_operator_shapes_and_constant_nullspace() -> None:
    number_of_maturities = 4
    number_of_x_points = 5

    D_x, D_T = build_2d_difference_operators(
        number_of_maturities=number_of_maturities,
        number_of_log_moneyness_points=number_of_x_points,
        maturity_spacing=0.25,
        log_moneyness_spacing=0.10,
    )

    number_of_unknowns = number_of_maturities * number_of_x_points

    assert D_x.shape == (
        number_of_maturities * (number_of_x_points - 1),
        number_of_unknowns,
    )
    assert D_T.shape == (
        (number_of_maturities - 1) * number_of_x_points,
        number_of_unknowns,
    )

    constant_surface = np.ones(
        (number_of_maturities, number_of_x_points)
    )
    constant_vector = constant_surface.reshape(-1, order="C")

    assert np.allclose(D_x @ constant_vector, 0.0)
    assert np.allclose(D_T @ constant_vector, 0.0)


def test_difference_operators_follow_c_order_surface_layout() -> None:
    maturities = np.array([0.0, 0.5, 1.0])
    x_values = np.array([-0.2, 0.0, 0.2, 0.4])

    x_mesh, maturity_mesh = np.meshgrid(x_values, maturities)

    D_x, D_T = build_2d_difference_operators(
        number_of_maturities=maturities.size,
        number_of_log_moneyness_points=x_values.size,
        maturity_spacing=maturities[1] - maturities[0],
        log_moneyness_spacing=x_values[1] - x_values[0],
    )

    x_linear_surface = x_mesh
    time_linear_surface = maturity_mesh

    assert np.allclose(
        D_x @ x_linear_surface.reshape(-1, order="C"),
        1.0,
    )
    assert np.allclose(
        D_T @ x_linear_surface.reshape(-1, order="C"),
        0.0,
    )
    assert np.allclose(
        D_x @ time_linear_surface.reshape(-1, order="C"),
        0.0,
    )
    assert np.allclose(
        D_T @ time_linear_surface.reshape(-1, order="C"),
        1.0,
    )


def test_regularization_matrix_is_symmetric_positive_definite() -> None:
    potential = np.full((4, 5), 0.2)

    regularization_matrix, D_x, D_T = build_regularization_matrix(
        number_of_maturities=4,
        number_of_log_moneyness_points=5,
        maturity_spacing=0.25,
        log_moneyness_spacing=0.10,
        alpha_x=0.8,
        alpha_T=0.6,
        beta=1e-3,
        potential=potential,
    )

    difference = (
        regularization_matrix - regularization_matrix.T
    ).toarray()

    assert np.allclose(difference, 0.0)

    smallest_eigenvalue = eigsh(
        regularization_matrix,
        k=1,
        which="SA",
        return_eigenvectors=False,
    )[0]

    assert smallest_eigenvalue > 0

    smooth = np.zeros((4, 5))
    rough = np.indices((4, 5)).sum(axis=0) % 2

    smooth_components = regularization_components(
        smooth,
        D_x,
        D_T,
        alpha_x=0.8,
        alpha_T=0.6,
        beta=1e-3,
        potential=potential,
    )
    rough_components = regularization_components(
        rough,
        D_x,
        D_T,
        alpha_x=0.8,
        alpha_T=0.6,
        beta=1e-3,
        potential=potential,
    )

    assert rough_components["total"] > smooth_components["total"]


def test_potential_surfaces_have_expected_behaviour() -> None:
    x_values = np.linspace(-0.4, 0.4, 9)
    maturities = np.linspace(0.1, 2.0, 5)

    wing = wing_potential(
        log_moneyness=x_values,
        maturities=maturities,
        strength=2.0,
        power=2.0,
        start=0.1,
    )

    assert wing.shape == (maturities.size, x_values.size)
    centre_index = np.argmin(np.abs(x_values))
    assert np.allclose(wing[:, centre_index], 0.0)
    assert wing[0, 0] > wing[0, centre_index]
    assert wing[0, -1] > wing[0, centre_index]

    confidence = quote_confidence_surface(
        log_moneyness=x_values,
        maturities=maturities,
        quote_log_moneyness=np.array([0.0]),
        quote_maturities=np.array([1.0]),
        log_moneyness_bandwidth=0.1,
        maturity_bandwidth=0.2,
    )
    confidence_penalty = confidence_potential(
        confidence,
        strength=3.0,
    )
    combined = combine_potentials(wing, confidence_penalty)

    nearest_maturity = np.argmin(np.abs(maturities - 1.0))
    assert confidence[
        nearest_maturity,
        centre_index,
    ] > confidence[0, 0]
    assert confidence_penalty[
        nearest_maturity,
        centre_index,
    ] < confidence_penalty[0, 0]
    assert np.allclose(combined, wing + confidence_penalty)
