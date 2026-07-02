"""Grid, log-moneyness, and surface reshaping utilities."""

from __future__ import annotations

import numpy as np


def make_uniform_grid(
    lower: float,
    upper: float,
    number_of_points: int,
) -> np.ndarray:
    """Create an increasing one-dimensional uniform grid."""
    if upper <= lower:
        raise ValueError("upper must be greater than lower.")
    if number_of_points < 2:
        raise ValueError("number_of_points must be at least 2.")

    return np.linspace(lower, upper, number_of_points, dtype=float)


def strike_to_log_moneyness(
    strike: np.ndarray | float,
    forward: np.ndarray | float,
) -> np.ndarray | float:
    """Convert strike and forward price to x = log(K/F)."""
    strike_array = np.asarray(strike, dtype=float)
    forward_array = np.asarray(forward, dtype=float)

    if np.any(strike_array <= 0) or np.any(forward_array <= 0):
        raise ValueError("strike and forward must be positive.")

    result = np.log(strike_array / forward_array)
    return float(result) if result.ndim == 0 else result


def log_moneyness_to_strike(
    log_moneyness: np.ndarray | float,
    forward: np.ndarray | float,
) -> np.ndarray | float:
    """Convert x = log(K/F) back to strike K = F exp(x)."""
    forward_array = np.asarray(forward, dtype=float)
    if np.any(forward_array <= 0):
        raise ValueError("forward must be positive.")

    result = forward_array * np.exp(np.asarray(log_moneyness, dtype=float))
    return float(result) if result.ndim == 0 else result


def make_calibration_grid(
    x_lower: float = -0.4,
    x_upper: float = 0.4,
    number_of_x_points: int = 25,
    maturity_lower: float = 0.05,
    maturity_upper: float = 2.0,
    number_of_maturity_points: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create the log-moneyness and maturity grids used for calibration.

    Returns
    -------
    x_grid:
        One-dimensional log-moneyness coordinates.
    maturity_grid:
        One-dimensional maturity coordinates.
    x_mesh, maturity_mesh:
        Arrays of shape (n_maturities, n_x). Rows correspond to maturities and
        columns correspond to log-moneyness points.
    """
    x_grid = make_uniform_grid(x_lower, x_upper, number_of_x_points)
    maturity_grid = make_uniform_grid(
        maturity_lower,
        maturity_upper,
        number_of_maturity_points,
    )
    x_mesh, maturity_mesh = np.meshgrid(x_grid, maturity_grid, indexing="xy")
    return x_grid, maturity_grid, x_mesh, maturity_mesh


def surface_to_vector(surface: np.ndarray) -> np.ndarray:
    """Flatten a (n_maturities, n_x) surface with x varying fastest."""
    surface_array = np.asarray(surface, dtype=float)
    if surface_array.ndim != 2:
        raise ValueError("surface must be a two-dimensional array.")

    return surface_array.reshape(-1, order="C")


def vector_to_surface(
    vector: np.ndarray,
    number_of_maturity_points: int,
    number_of_x_points: int,
) -> np.ndarray:
    """Restore a flattened vector to shape (n_maturities, n_x)."""
    vector_array = np.asarray(vector, dtype=float)

    expected_size = number_of_maturity_points * number_of_x_points
    if vector_array.size != expected_size:
        raise ValueError(
            f"vector has size {vector_array.size}, expected {expected_size}."
        )

    return vector_array.reshape(
        number_of_maturity_points,
        number_of_x_points,
        order="C",
    )
