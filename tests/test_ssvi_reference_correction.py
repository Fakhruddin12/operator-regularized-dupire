import numpy as np

from src.evaluation.market_ssvi import (
    SSVIFit,
    make_market_ssvi_local_volatility_function,
)
from src.evaluation.ssvi_dupire import evaluate_ssvi_dupire
from src.inverse.reference_correction import (
    make_reference_plus_correction_local_volatility,
)


def _fit():
    return SSVIFit(
        maturities=np.array([0.1, 0.3, 0.6, 1.0]),
        theta=np.array([0.004, 0.013, 0.028, 0.050]),
        rho=-0.55,
        eta=1.2,
        gamma=0.45,
        objective=0.0,
        success=True,
        message="test fixture",
        number_of_usable_quotes=100,
        fit_seconds=0.0,
    )


def test_vectorized_ssvi_callback_matches_surface_evaluator():
    fit = _fit()
    callback = make_market_ssvi_local_volatility_function(fit)
    x = np.linspace(-0.5, 0.5, 21)
    for maturity in (0.02, 0.1, 0.45, 1.0):
        expected = evaluate_ssvi_dupire(
            fit, np.full_like(x, maturity), x
        )["local_volatility"].to_numpy()
        np.testing.assert_allclose(callback(x, maturity), expected, rtol=1e-11, atol=1e-12)


def test_compact_correction_is_zero_at_and_outside_support():
    base = lambda x, T: np.full_like(np.asarray(x, dtype=float), 0.2)
    T = np.linspace(0.0, 1.0, 6)
    x = np.linspace(-0.5, 0.5, 9)
    correction = np.full((T.size, x.size), 0.4)
    callback = make_reference_plus_correction_local_volatility(
        base, correction, T, x,
        maturity_taper=0.1,
        log_moneyness_taper=0.1,
    )
    np.testing.assert_allclose(callback(np.array([-0.5, 0.5]), 0.5), [0.2, 0.2])
    np.testing.assert_allclose(callback(np.array([0.0]), 0.0), [0.2])
    np.testing.assert_allclose(callback(np.array([0.0]), 0.5), [0.2 * np.exp(0.2)])
    np.testing.assert_allclose(callback(np.array([-0.7, 0.7]), 0.5), [0.2, 0.2])
