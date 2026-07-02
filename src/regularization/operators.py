"""Sparse regularisation operators for local-volatility surfaces.

A surface has shape

    (number_of_maturities, number_of_log_moneyness_points)

and is flattened in C order, so log-moneyness varies fastest.

The regularisation matrix is

    R = beta I
        + alpha_x D_x.T D_x
        + alpha_T D_T.T D_T
        + diag(V).

The first-difference construction leaves boundary values free rather than
forcing them to zero. The associated normal equations therefore have the
natural discrete analogue of Neumann boundary conditions.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix, diags, eye, kron


def first_difference_matrix(
    number_of_points: int,
    spacing: float,
) -> csc_matrix:
    """Return a first-difference matrix divided by the grid spacing.

    For a vector ``z``,

        D @ z = [(z[1]-z[0])/spacing, ..., (z[-1]-z[-2])/spacing].

    The matrix has shape ``(number_of_points - 1, number_of_points)``.
    """
    if number_of_points < 2:
        raise ValueError("number_of_points must be at least 2.")
    if spacing <= 0:
        raise ValueError("spacing must be positive.")

    lower = -np.ones(number_of_points - 1) / spacing
    upper = np.ones(number_of_points - 1) / spacing

    return diags(
        diagonals=[lower, upper],
        offsets=[0, 1],
        shape=(number_of_points - 1, number_of_points),
        format="csc",
    )


def build_2d_difference_operators(
    number_of_maturities: int,
    number_of_log_moneyness_points: int,
    maturity_spacing: float,
    log_moneyness_spacing: float,
) -> tuple[csc_matrix, csc_matrix]:
    """Build first-difference operators for a flattened two-dimensional surface.

    Returns
    -------
    D_x:
        Differences across log-moneyness. Its shape is
        ``(n_T * (n_x - 1), n_T * n_x)``.
    D_T:
        Differences across maturity. Its shape is
        ``((n_T - 1) * n_x, n_T * n_x)``.
    """
    if number_of_maturities < 2:
        raise ValueError("number_of_maturities must be at least 2.")
    if number_of_log_moneyness_points < 2:
        raise ValueError(
            "number_of_log_moneyness_points must be at least 2."
        )

    one_dimensional_x = first_difference_matrix(
        number_of_points=number_of_log_moneyness_points,
        spacing=log_moneyness_spacing,
    )
    one_dimensional_time = first_difference_matrix(
        number_of_points=number_of_maturities,
        spacing=maturity_spacing,
    )

    identity_time = eye(number_of_maturities, format="csc")
    identity_x = eye(number_of_log_moneyness_points, format="csc")

    # C-order flattening means x varies fastest within each maturity row.
    D_x = kron(
        identity_time,
        one_dimensional_x,
        format="csc",
    )
    D_T = kron(
        one_dimensional_time,
        identity_x,
        format="csc",
    )

    return D_x, D_T


def _potential_vector(
    potential: np.ndarray | None,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    """Validate and flatten a non-negative potential surface."""
    if potential is None:
        return np.zeros(expected_shape[0] * expected_shape[1])

    potential_array = np.asarray(potential, dtype=float)

    if potential_array.shape == expected_shape:
        vector = potential_array.reshape(-1, order="C")
    elif potential_array.shape == (
        expected_shape[0] * expected_shape[1],
    ):
        vector = potential_array.copy()
    else:
        raise ValueError(
            "potential must have shape "
            f"{expected_shape} or {(expected_shape[0] * expected_shape[1],)}."
        )

    if np.any(~np.isfinite(vector)):
        raise ValueError("potential contains non-finite values.")
    if np.any(vector < 0):
        raise ValueError("potential must be non-negative.")

    return vector


def build_regularization_matrix(
    number_of_maturities: int,
    number_of_log_moneyness_points: int,
    maturity_spacing: float,
    log_moneyness_spacing: float,
    alpha_x: float = 1.0,
    alpha_T: float = 1.0,
    beta: float = 1e-6,
    potential: np.ndarray | None = None,
) -> tuple[csc_matrix, csc_matrix, csc_matrix]:
    """Build ``R`` together with the difference operators used to form it."""
    if alpha_x < 0 or alpha_T < 0 or beta < 0:
        raise ValueError("alpha_x, alpha_T, and beta must be non-negative.")

    D_x, D_T = build_2d_difference_operators(
        number_of_maturities=number_of_maturities,
        number_of_log_moneyness_points=number_of_log_moneyness_points,
        maturity_spacing=maturity_spacing,
        log_moneyness_spacing=log_moneyness_spacing,
    )

    number_of_unknowns = (
        number_of_maturities * number_of_log_moneyness_points
    )
    identity = eye(number_of_unknowns, format="csc")

    potential_vector = _potential_vector(
        potential=potential,
        expected_shape=(
            number_of_maturities,
            number_of_log_moneyness_points,
        ),
    )
    potential_matrix = diags(
        potential_vector,
        offsets=0,
        format="csc",
    )

    regularization_matrix = (
        beta * identity
        + alpha_x * (D_x.T @ D_x)
        + alpha_T * (D_T.T @ D_T)
        + potential_matrix
    ).tocsc()

    return regularization_matrix, D_x, D_T


def regularization_components(
    correction: np.ndarray,
    D_x: csc_matrix,
    D_T: csc_matrix,
    alpha_x: float = 1.0,
    alpha_T: float = 1.0,
    beta: float = 0.0,
    potential: np.ndarray | None = None,
) -> dict[str, float]:
    """Return the separate contributions to the quadratic penalty."""
    correction_vector = np.asarray(correction, dtype=float).reshape(
        -1,
        order="C",
    )

    if D_x.shape[1] != correction_vector.size:
        raise ValueError("D_x is incompatible with correction.")
    if D_T.shape[1] != correction_vector.size:
        raise ValueError("D_T is incompatible with correction.")
    if alpha_x < 0 or alpha_T < 0 or beta < 0:
        raise ValueError("penalty weights must be non-negative.")

    if potential is None:
        potential_vector = np.zeros(correction_vector.size)
    else:
        potential_vector = np.asarray(
            potential,
            dtype=float,
        ).reshape(-1, order="C")

        if potential_vector.size != correction_vector.size:
            raise ValueError("potential is incompatible with correction.")
        if np.any(potential_vector < 0):
            raise ValueError("potential must be non-negative.")

    magnitude = beta * float(correction_vector @ correction_vector)
    x_roughness = alpha_x * float(
        (D_x @ correction_vector) @ (D_x @ correction_vector)
    )
    time_roughness = alpha_T * float(
        (D_T @ correction_vector) @ (D_T @ correction_vector)
    )
    potential_penalty = float(
        correction_vector @ (potential_vector * correction_vector)
    )

    return {
        "magnitude": magnitude,
        "x_roughness": x_roughness,
        "time_roughness": time_roughness,
        "potential": potential_penalty,
        "total": (
            magnitude
            + x_roughness
            + time_roughness
            + potential_penalty
        ),
    }


def quadratic_penalty(
    correction: np.ndarray,
    regularization_matrix: csc_matrix,
) -> float:
    """Return ``h.T @ R @ h``."""
    correction_vector = np.asarray(correction, dtype=float).reshape(
        -1,
        order="C",
    )

    if regularization_matrix.shape != (
        correction_vector.size,
        correction_vector.size,
    ):
        raise ValueError(
            "regularization_matrix is incompatible with correction."
        )

    return float(
        correction_vector
        @ (regularization_matrix @ correction_vector)
    )
