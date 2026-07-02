"""Grid and coordinate scaling for the regularisation matrix.

The earlier difference operators divide by the physical grid spacing, which
gives correct derivative units. To make penalty magnitudes more comparable
across grids, this module additionally:

1. maps log-moneyness and maturity to the unit square;
2. multiplies the discrete squared-derivative sums by the unit-cell area.

The resulting matrix approximates an integral over nondimensional coordinates.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix

from src.regularization.operators import (
    build_regularization_matrix,
)


def _validate_uniform_grid(
    values: np.ndarray,
    name: str,
) -> tuple[np.ndarray, float]:
    """Validate a strictly increasing uniform one-dimensional grid."""
    grid = np.asarray(values, dtype=float)

    if grid.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if grid.size < 2:
        raise ValueError(f"{name} must contain at least two values.")
    if np.any(~np.isfinite(grid)):
        raise ValueError(f"{name} contains non-finite values.")

    differences = np.diff(grid)
    if np.any(differences <= 0):
        raise ValueError(f"{name} must be strictly increasing.")
    if not np.allclose(
        differences,
        differences[0],
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(
            f"{name} must be uniformly spaced for this implementation."
        )

    return grid, float(differences[0])


def nondimensional_grid(
    values: np.ndarray,
    name: str = "grid",
) -> np.ndarray:
    """Map a one-dimensional grid linearly onto the interval ``[0, 1]``."""
    grid, _ = _validate_uniform_grid(values, name)

    grid_range = grid[-1] - grid[0]
    if grid_range <= 0:
        raise ValueError(f"{name} must span a positive range.")

    return (grid - grid[0]) / grid_range


def build_nondimensional_regularization_matrix(
    maturities: np.ndarray,
    log_moneyness: np.ndarray,
    alpha_x: float = 1.0,
    alpha_T: float = 1.0,
    beta: float = 1e-6,
    potential: np.ndarray | None = None,
) -> tuple[
    csc_matrix,
    csc_matrix,
    csc_matrix,
    dict[str, float],
]:
    """Build an integral-scaled regulariser on the unit square.

    The penalty approximates

    ``integral [beta*h^2 + alpha_x*h_xi^2 + alpha_T*h_tau^2 + V*h^2] dxi dtau``,

    where ``xi`` and ``tau`` are nondimensional log-moneyness and maturity.
    """
    maturity_grid, _ = _validate_uniform_grid(
        maturities,
        "maturities",
    )
    x_grid, _ = _validate_uniform_grid(
        log_moneyness,
        "log_moneyness",
    )

    scaled_maturities = nondimensional_grid(
        maturity_grid,
        "maturities",
    )
    scaled_x = nondimensional_grid(
        x_grid,
        "log_moneyness",
    )

    delta_tau = float(
        scaled_maturities[1] - scaled_maturities[0]
    )
    delta_xi = float(
        scaled_x[1] - scaled_x[0]
    )
    cell_area = delta_tau * delta_xi

    unscaled_matrix, D_xi, D_tau = (
        build_regularization_matrix(
            number_of_maturities=maturity_grid.size,
            number_of_log_moneyness_points=x_grid.size,
            maturity_spacing=delta_tau,
            log_moneyness_spacing=delta_xi,
            alpha_x=alpha_x,
            alpha_T=alpha_T,
            beta=beta,
            potential=potential,
        )
    )

    square_root_area = np.sqrt(cell_area)

    metadata = {
        "log_moneyness_range": float(
            x_grid[-1] - x_grid[0]
        ),
        "maturity_range": float(
            maturity_grid[-1] - maturity_grid[0]
        ),
        "delta_xi": delta_xi,
        "delta_tau": delta_tau,
        "cell_area": cell_area,
    }

    return (
        (cell_area * unscaled_matrix).tocsc(),
        (square_root_area * D_xi).tocsc(),
        (square_root_area * D_tau).tocsc(),
        metadata,
    )


def scaled_quadratic_penalty(
    correction: np.ndarray,
    regularization_matrix: csc_matrix,
) -> float:
    """Return the grid-scaled quadratic penalty ``h.T R h``."""
    correction_vector = np.asarray(
        correction,
        dtype=float,
    ).reshape(-1, order="C")

    if regularization_matrix.shape != (
        correction_vector.size,
        correction_vector.size,
    ):
        raise ValueError(
            "regularization_matrix is incompatible with correction."
        )

    return float(
        correction_vector
        @ (
            regularization_matrix
            @ correction_vector
        )
    )
