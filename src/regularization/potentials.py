"""Potential surfaces used inside the Schrödinger-type regulariser."""

from __future__ import annotations

import numpy as np


def wing_potential(
    log_moneyness: np.ndarray,
    maturities: np.ndarray,
    strength: float = 1.0,
    power: float = 2.0,
    start: float = 0.0,
) -> np.ndarray:
    """Penalise corrections increasingly strongly in the log-moneyness wings.

    The potential is zero for ``abs(x) <= start`` and rises smoothly outside
    that region. It is normalised so its maximum equals ``strength``.
    """
    x_values = np.asarray(log_moneyness, dtype=float)
    maturity_values = np.asarray(maturities, dtype=float)

    if x_values.ndim != 1 or maturity_values.ndim != 1:
        raise ValueError(
            "log_moneyness and maturities must be one-dimensional."
        )
    if strength < 0:
        raise ValueError("strength must be non-negative.")
    if power <= 0:
        raise ValueError("power must be positive.")
    if start < 0:
        raise ValueError("start must be non-negative.")

    maximum_absolute_x = float(np.max(np.abs(x_values)))
    if maximum_absolute_x == 0:
        one_dimensional = np.zeros_like(x_values)
    elif start >= maximum_absolute_x:
        one_dimensional = np.zeros_like(x_values)
    else:
        scaled_distance = np.maximum(
            np.abs(x_values) - start,
            0.0,
        ) / (maximum_absolute_x - start)
        one_dimensional = strength * scaled_distance**power

    return np.tile(
        one_dimensional,
        (maturity_values.size, 1),
    )


def quote_confidence_surface(
    log_moneyness: np.ndarray,
    maturities: np.ndarray,
    quote_log_moneyness: np.ndarray,
    quote_maturities: np.ndarray,
    log_moneyness_bandwidth: float = 0.08,
    maturity_bandwidth: float = 0.25,
) -> np.ndarray:
    """Construct a confidence surface from nearby quote density.

    Each quote contributes a two-dimensional Gaussian kernel. The summed
    density is normalised to lie between zero and one.
    """
    x_values = np.asarray(log_moneyness, dtype=float)
    maturity_values = np.asarray(maturities, dtype=float)
    quote_x = np.asarray(quote_log_moneyness, dtype=float)
    quote_T = np.asarray(quote_maturities, dtype=float)

    if x_values.ndim != 1 or maturity_values.ndim != 1:
        raise ValueError(
            "log_moneyness and maturities must be one-dimensional."
        )
    if quote_x.ndim != 1 or quote_T.ndim != 1:
        raise ValueError(
            "quote_log_moneyness and quote_maturities must be one-dimensional."
        )
    if quote_x.size == 0:
        raise ValueError("at least one quote location is required.")
    if quote_x.size != quote_T.size:
        raise ValueError(
            "quote_log_moneyness and quote_maturities must have equal length."
        )
    if log_moneyness_bandwidth <= 0 or maturity_bandwidth <= 0:
        raise ValueError("bandwidths must be positive.")

    x_mesh, maturity_mesh = np.meshgrid(
        x_values,
        maturity_values,
    )

    density = np.zeros_like(x_mesh, dtype=float)

    for quote_x_value, quote_maturity in zip(quote_x, quote_T):
        squared_distance = (
            ((x_mesh - quote_x_value) / log_moneyness_bandwidth) ** 2
            + ((maturity_mesh - quote_maturity) / maturity_bandwidth) ** 2
        )
        density += np.exp(-0.5 * squared_distance)

    maximum_density = float(np.max(density))
    if maximum_density == 0:
        return density

    return density / maximum_density


def confidence_potential(
    confidence: np.ndarray,
    strength: float = 1.0,
    power: float = 1.0,
) -> np.ndarray:
    """Convert confidence in ``[0,1]`` into a non-negative penalty surface."""
    confidence_array = np.asarray(confidence, dtype=float)

    if np.any(~np.isfinite(confidence_array)):
        raise ValueError("confidence contains non-finite values.")
    if strength < 0:
        raise ValueError("strength must be non-negative.")
    if power <= 0:
        raise ValueError("power must be positive.")

    clipped_confidence = np.clip(confidence_array, 0.0, 1.0)

    return strength * (1.0 - clipped_confidence) ** power


def combine_potentials(
    *potentials: np.ndarray,
) -> np.ndarray:
    """Add potential surfaces after checking that their shapes agree."""
    if not potentials:
        raise ValueError("at least one potential is required.")

    arrays = [np.asarray(item, dtype=float) for item in potentials]
    reference_shape = arrays[0].shape

    for array in arrays:
        if array.shape != reference_shape:
            raise ValueError("all potentials must have the same shape.")
        if np.any(~np.isfinite(array)):
            raise ValueError("potentials must contain finite values.")
        if np.any(array < 0):
            raise ValueError("potentials must be non-negative.")

    return np.sum(arrays, axis=0)
