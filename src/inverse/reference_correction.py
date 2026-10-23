"""Compact, confidence-aware corrections to a full-domain reference surface."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def _smoothstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def make_reference_plus_correction_local_volatility(
    reference_local_volatility: Callable[[np.ndarray, float], np.ndarray],
    correction_surface: np.ndarray,
    correction_maturities: np.ndarray,
    correction_log_moneyness: np.ndarray,
    maturity_taper: float = 0.08,
    log_moneyness_taper: float = 0.10,
) -> Callable[[np.ndarray, float], np.ndarray]:
    """Add a smooth, compactly supported log-variance correction to a prior.

    The prior callback supplies the full-domain local volatility, including
    short maturities and wings. The gridded correction is interpolated only
    inside its support and smoothly tapered to zero at the support boundary.
    Thus unquoted regions inherit the reference model instead of flat-clipping
    the entire volatility surface at the edge of the observed quote panel.
    """
    correction = np.asarray(correction_surface, dtype=float)
    T = np.asarray(correction_maturities, dtype=float)
    x = np.asarray(correction_log_moneyness, dtype=float)
    if T.ndim != 1 or x.ndim != 1 or T.size < 2 or x.size < 2:
        raise ValueError("correction grids must be vectors with at least two nodes.")
    if correction.shape != (T.size, x.size):
        raise ValueError("correction_surface shape must match the correction grids.")
    if np.any(~np.isfinite(correction)):
        raise ValueError("correction_surface must be finite.")
    if np.any(np.diff(T) <= 0) or np.any(np.diff(x) <= 0):
        raise ValueError("correction grids must be strictly increasing.")
    if maturity_taper <= 0 or log_moneyness_taper <= 0:
        raise ValueError("taper widths must be positive.")

    interpolator = RegularGridInterpolator(
        (T, x), correction,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    def local_volatility(query_x: np.ndarray, maturity: float) -> np.ndarray:
        xq = np.asarray(query_x, dtype=float)
        Tq = float(maturity)
        base = np.asarray(reference_local_volatility(xq, Tq), dtype=float)
        base = np.broadcast_to(base, xq.shape)
        if np.any(~np.isfinite(base)) or np.any(base <= 0):
            raise ValueError("reference local volatility must be finite and positive.")

        points = np.column_stack((np.full(xq.size, Tq), xq.reshape(-1)))
        h = interpolator(points).reshape(xq.shape)
        x_weight = (
            _smoothstep((xq - x[0]) / log_moneyness_taper)
            * _smoothstep((x[-1] - xq) / log_moneyness_taper)
        )
        T_weight = float(
            _smoothstep(np.asarray((Tq - T[0]) / maturity_taper))
            * _smoothstep(np.asarray((T[-1] - Tq) / maturity_taper))
        )
        return base * np.exp(0.5 * h * x_weight * T_weight)

    return local_volatility
