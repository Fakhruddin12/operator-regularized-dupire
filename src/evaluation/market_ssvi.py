"""SSVI calibration and pricing for forward-normalized real option data."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, least_squares
from scipy.special import expit
from scipy.stats import norm

from src.evaluation.ssvi_dupire import (
    SSVIFit,
    evaluate_ssvi_dupire,
    ssvi_strike_derivatives,
    ssvi_total_variance,
)
from src.pricing.forward_normalized_pde import normalized_black_call_price


def normalized_black_vega(
    log_moneyness: np.ndarray | float,
    maturity: np.ndarray | float,
    volatility: np.ndarray | float,
) -> np.ndarray | float:
    """Derivative of normalized Black call price with respect to volatility."""
    x, T, sigma = np.broadcast_arrays(
        np.asarray(log_moneyness, dtype=float),
        np.asarray(maturity, dtype=float),
        np.asarray(volatility, dtype=float),
    )
    if np.any(T <= 0):
        raise ValueError("maturity must be positive.")
    if np.any(sigma <= 0):
        raise ValueError("volatility must be positive.")

    root_time = np.sqrt(T)
    d1 = (-x + 0.5 * sigma**2 * T) / (sigma * root_time)
    result = norm.pdf(d1) * root_time
    return float(result) if result.ndim == 0 else result


def implied_volatility_from_normalized_call(
    normalized_call_price: float,
    log_moneyness: float,
    maturity: float,
    volatility_lower: float = 1e-6,
    volatility_upper: float = 5.0,
) -> float:
    """Invert the normalized Black call formula."""
    if maturity <= 0:
        raise ValueError("maturity must be positive.")

    lower_bound = max(1.0 - np.exp(log_moneyness), 0.0)
    upper_bound = 1.0
    tolerance = 1e-12

    if normalized_call_price < lower_bound - tolerance:
        raise ValueError("normalized call price is below intrinsic value.")
    if normalized_call_price > upper_bound + tolerance:
        raise ValueError("normalized call price exceeds its upper bound.")

    price = float(
        np.clip(normalized_call_price, lower_bound, upper_bound)
    )

    def objective(volatility: float) -> float:
        return float(
            normalized_black_call_price(
                log_moneyness=log_moneyness,
                maturity=maturity,
                volatility=volatility,
            )
            - price
        )

    lower_value = objective(volatility_lower)
    upper_value = objective(volatility_upper)

    if abs(lower_value) < 1e-12:
        return volatility_lower
    if lower_value * upper_value > 0:
        raise ValueError("implied volatility is not bracketed.")

    return float(
        brentq(
            objective,
            volatility_lower,
            volatility_upper,
            xtol=1e-12,
            rtol=1e-12,
        )
    )


def forward_implied_variance_quotes(
    quote_data: pd.DataFrame,
    price_column: str = "observed_call_price",
    noise_column: str = "noise_standard_deviation",
    minimum_implied_volatility: float = 0.02,
    maximum_total_variance_standard_deviation: float = 0.50,
) -> pd.DataFrame:
    """Convert real option prices into forward implied-total-variance data."""
    required = {
        "maturity",
        "log_moneyness",
        "discount_factor",
        "forward",
        price_column,
        noise_column,
    }
    missing = required.difference(quote_data.columns)
    if missing:
        raise ValueError(
            f"quote_data is missing columns: {sorted(missing)}"
        )

    rows = []

    for row in quote_data.itertuples(index=False):
        maturity = float(getattr(row, "maturity"))
        x = float(getattr(row, "log_moneyness"))
        discount = float(getattr(row, "discount_factor"))
        forward = float(getattr(row, "forward"))
        scale = discount * forward
        normalized_price = float(getattr(row, price_column)) / scale
        normalized_noise = float(getattr(row, noise_column)) / scale

        try:
            implied_volatility = implied_volatility_from_normalized_call(
                normalized_call_price=normalized_price,
                log_moneyness=x,
                maturity=maturity,
            )
            effective_volatility = max(implied_volatility, 1e-4)
            vega = float(
                normalized_black_vega(
                    log_moneyness=x,
                    maturity=maturity,
                    volatility=effective_volatility,
                )
            )
            total_variance = implied_volatility**2 * maturity
            total_variance_std = (
                normalized_noise
                * 2.0
                * effective_volatility
                * maturity
                / max(vega, 1e-12)
            )
        except (ValueError, RuntimeError, FloatingPointError):
            implied_volatility = np.nan
            vega = np.nan
            total_variance = np.nan
            total_variance_std = np.inf

        usable = bool(
            np.isfinite(total_variance)
            and np.isfinite(total_variance_std)
            and implied_volatility > minimum_implied_volatility
            and total_variance_std < maximum_total_variance_standard_deviation
            and vega > 1e-10
        )

        rows.append(
            {
                "maturity": maturity,
                "log_moneyness": x,
                "normalized_call_price": normalized_price,
                "normalized_noise_standard_deviation": normalized_noise,
                "implied_volatility": implied_volatility,
                "total_variance": total_variance,
                "normalized_vega": vega,
                "total_variance_standard_deviation": total_variance_std,
                "usable_for_ssvi": usable,
            }
        )

    return pd.DataFrame(rows)


def _decode_parameters(
    raw_parameters: np.ndarray,
    maturities: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    """Map unconstrained parameters to a monotone SSVI term structure."""
    number_of_maturities = maturities.size
    time_increments = np.diff(np.concatenate(([0.0], maturities)))
    instantaneous_variances = np.exp(raw_parameters[:number_of_maturities])
    theta = np.cumsum(instantaneous_variances * time_increments)
    rho = 0.995 * np.tanh(raw_parameters[number_of_maturities])
    eta = np.exp(raw_parameters[number_of_maturities + 1])
    gamma = 0.05 + 0.90 * expit(raw_parameters[number_of_maturities + 2])
    return theta, float(rho), float(eta), float(gamma)


def fit_market_ssvi_surface(
    quote_data: pd.DataFrame,
    price_column: str = "observed_call_price",
    noise_column: str = "noise_standard_deviation",
    diagnostic_log_moneyness_limit: float = 0.60,
    standard_deviation_floor: float = 2e-5,
    standard_deviation_cap: float = 2e-2,
    maximum_function_evaluations: int = 10000,
) -> tuple[SSVIFit, pd.DataFrame]:
    """Fit an arbitrage-aware SSVI surface to training quotes."""
    start_time = perf_counter()
    observations = forward_implied_variance_quotes(
        quote_data=quote_data,
        price_column=price_column,
        noise_column=noise_column,
    )
    usable = observations[observations["usable_for_ssvi"]].copy()
    maturities = np.sort(quote_data["maturity"].unique().astype(float))

    if len(usable) < maturities.size + 3:
        raise ValueError("Too few usable observations for SSVI fitting.")

    maturity_index = {
        float(maturity): index
        for index, maturity in enumerate(maturities)
    }
    observation_indices = np.array(
        [maturity_index[float(value)] for value in usable["maturity"]],
        dtype=int,
    )
    observed_k = usable["log_moneyness"].to_numpy(dtype=float)
    observed_w = usable["total_variance"].to_numpy(dtype=float)
    observed_std = np.clip(
        usable["total_variance_standard_deviation"].to_numpy(dtype=float),
        standard_deviation_floor,
        standard_deviation_cap,
    )

    atm_theta = []
    for maturity in maturities:
        maturity_data = usable[
            np.isclose(usable["maturity"], maturity)
        ].sort_values("log_moneyness")
        if maturity_data.empty:
            atm_theta.append(0.04 * maturity)
        else:
            atm_theta.append(
                float(
                    np.interp(
                        0.0,
                        maturity_data["log_moneyness"],
                        maturity_data["total_variance"],
                    )
                )
            )

    atm_theta = np.maximum.accumulate(
        np.maximum(np.asarray(atm_theta), 1e-7)
    )
    time_increments = np.diff(np.concatenate(([0.0], maturities)))
    theta_increments = np.diff(np.concatenate(([0.0], atm_theta)))
    initial_instantaneous_variance = np.clip(
        theta_increments / time_increments,
        1e-4,
        2.0,
    )

    diagnostic_k = np.linspace(
        -diagnostic_log_moneyness_limit,
        diagnostic_log_moneyness_limit,
        121,
    )

    def residuals(raw_parameters: np.ndarray) -> np.ndarray:
        theta_values, rho, eta, gamma = _decode_parameters(
            raw_parameters,
            maturities,
        )
        fitted = ssvi_total_variance(
            observed_k,
            theta_values[observation_indices],
            rho,
            eta,
            gamma,
        )
        data_residual = (fitted - observed_w) / observed_std

        penalties = []
        diagnostic_slices = []
        for theta_value in theta_values:
            w, w_k, w_kk = ssvi_strike_derivatives(
                diagnostic_k,
                theta_value,
                rho,
                eta,
                gamma,
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                density_factor = (
                    (1.0 - diagnostic_k * w_k / (2.0 * w)) ** 2
                    - 0.25 * (0.25 + 1.0 / w) * w_k**2
                    + 0.5 * w_kk
                )
            penalties.extend(
                np.sqrt(1000.0) * np.minimum(density_factor, 0.0)
            )
            penalties.extend(
                np.sqrt(1000.0) * np.minimum(w - 1e-10, 0.0)
            )
            diagnostic_slices.append(w)

        calendar_increments = np.diff(
            np.asarray(diagnostic_slices),
            axis=0,
        )
        penalties.extend(
            (
                np.sqrt(1000.0)
                * np.minimum(calendar_increments, 0.0)
            ).reshape(-1)
        )

        return np.concatenate(
            [np.asarray(data_residual), np.asarray(penalties)]
        )

    base_logs = np.log(initial_instantaneous_variance)
    initialisations = []
    for rho_start, eta_start, gamma_start in [
        (-0.50, 0.50, 0.50),
        (-0.25, 0.30, 0.50),
        (0.0, 0.50, 0.50),
        (-0.70, 0.80, 0.70),
    ]:
        raw_rho = np.arctanh(np.clip(rho_start / 0.995, -0.99, 0.99))
        gamma_unit = (gamma_start - 0.05) / 0.90
        raw_gamma = np.log(gamma_unit / (1.0 - gamma_unit))
        initialisations.append(
            np.concatenate(
                [
                    base_logs,
                    [raw_rho, np.log(eta_start), raw_gamma],
                ]
            )
        )

    best_solution = None
    for initial in initialisations:
        solution = least_squares(
            residuals,
            initial,
            loss="soft_l1",
            max_nfev=maximum_function_evaluations,
        )
        if best_solution is None or solution.cost < best_solution.cost:
            best_solution = solution

    theta_values, rho, eta, gamma = _decode_parameters(
        best_solution.x,
        maturities,
    )
    fit = SSVIFit(
        maturities=maturities,
        theta=theta_values,
        rho=rho,
        eta=eta,
        gamma=gamma,
        objective=float(best_solution.cost),
        success=bool(best_solution.success),
        message=str(best_solution.message),
        number_of_usable_quotes=int(len(usable)),
        fit_seconds=float(perf_counter() - start_time),
    )
    return fit, observations


def evaluate_market_ssvi_quotes(
    fit: SSVIFit,
    quote_data: pd.DataFrame,
) -> pd.DataFrame:
    """Predict currency prices and implied volatilities for market quotes."""
    required = {
        "maturity",
        "log_moneyness",
        "discount_factor",
        "forward",
    }
    missing = required.difference(quote_data.columns)
    if missing:
        raise ValueError(
            f"quote_data is missing columns: {sorted(missing)}"
        )

    maturities = quote_data["maturity"].to_numpy(dtype=float)
    x = quote_data["log_moneyness"].to_numpy(dtype=float)

    theta_spline = PchipInterpolator(
        np.concatenate(([0.0], fit.maturities)),
        np.concatenate(([0.0], fit.theta)),
        extrapolate=True,
    )
    theta_values = theta_spline(maturities)
    total_variance = ssvi_total_variance(
        x,
        theta_values,
        fit.rho,
        fit.eta,
        fit.gamma,
    )
    implied_volatility = np.sqrt(
        np.maximum(total_variance, 0.0) / maturities
    )
    normalized_price = normalized_black_call_price(
        log_moneyness=x,
        maturity=maturities,
        volatility=implied_volatility,
    )
    scale = (
        quote_data["discount_factor"].to_numpy(dtype=float)
        * quote_data["forward"].to_numpy(dtype=float)
    )

    return pd.DataFrame(
        {
            "ssvi_total_variance": total_variance,
            "ssvi_implied_volatility": implied_volatility,
            "ssvi_normalized_call_price": normalized_price,
            "ssvi_call_price": scale * normalized_price,
        },
        index=quote_data.index,
    )


def evaluate_market_ssvi_local_volatility(
    fit: SSVIFit,
    maturities: np.ndarray,
    log_moneyness: np.ndarray,
) -> pd.DataFrame:
    """Evaluate SSVI--Dupire local volatility on a requested grid."""
    maturity_mesh, x_mesh = np.meshgrid(
        np.asarray(maturities, dtype=float),
        np.asarray(log_moneyness, dtype=float),
        indexing="ij",
    )
    return evaluate_ssvi_dupire(
        fit=fit,
        maturities=maturity_mesh,
        log_moneyness=x_mesh,
    )
