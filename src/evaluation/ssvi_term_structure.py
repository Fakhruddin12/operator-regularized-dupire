"""Shared ATM total-variance interpolation with positive linear tails.

PCHIP preserves monotonicity inside its knots, not when its last cubic is
extrapolated. A PDE must query the reference beyond the last fitted expiry.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator


def theta_and_derivative(maturities, theta, query):
    times = np.asarray(maturities, float)
    values = np.asarray(theta, float)
    t = np.asarray(query, float)
    if (times.ndim != 1 or len(times) < 1 or values.shape != times.shape
            or np.any(~np.isfinite(times)) or np.any(~np.isfinite(values))
            or np.any(times <= 0) or np.any(np.diff(times) <= 0)
            or np.any(values <= 0) or np.any(np.diff(values) <= 0)):
        raise ValueError("SSVI ATM knots must be finite, positive and strictly increasing.")
    if np.any(~np.isfinite(t)) or np.any(t <= 0):
        raise ValueError("Query maturities must be finite and positive.")
    spline = PchipInterpolator(np.r_[0., times], np.r_[0., values], extrapolate=False)
    # Positive secant tails avoid negative cubic extrapolation and the singular
    # zero-time limit. The reference price and Dupire callbacks share this rule.
    early = t < times[0]
    late = t > times[-1]
    inside = np.clip(t, times[0], times[-1])
    result = np.asarray(spline(inside))
    derivative = np.asarray(spline.derivative()(inside))
    first_slope = values[0] / times[0]
    last_slope = ((values[-1] - values[-2]) / (times[-1] - times[-2])
                  if len(times) > 1 else first_slope)
    result = np.where(early, first_slope * t, result)
    derivative = np.where(early, first_slope, derivative)
    result = np.where(late, values[-1] + last_slope * (t - times[-1]), result)
    derivative = np.where(late, last_slope, derivative)
    # A PCHIP endpoint can have exactly zero derivative even for increasing knots.
    # Use the one-sided tail derivative at that endpoint, without clipping a
    # negative local variance anywhere in the surface.
    derivative = np.where(t == times[0], first_slope, derivative)
    derivative = np.where(t == times[-1], last_slope, derivative)
    return result, derivative
