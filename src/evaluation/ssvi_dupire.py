"""SSVI implied-variance smoothing followed by the Dupire formula.

This module implements the standard two-step local-volatility workflow:

1. fit a smooth implied total-variance surface;
2. apply Dupire's formula to that fitted surface.

The fitted surface uses the Surface SVI (SSVI) parameterisation of Gatheral and
Jacquier. Positive ATM total-variance increments enforce an increasing ATM term
structure, while explicit penalties discourage negative density and calendar
crossings on a dense diagnostic grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.special import expit

from src.pricing.black_scholes import (
    black_scholes_vega,
    implied_volatility_call,
)


@dataclass(frozen=True)
class SSVIFit:
    """Calibrated SSVI parameters and diagnostics."""

    maturities: np.ndarray
    theta: np.ndarray
    rho: float
    eta: float
    gamma: float
    objective: float
    success: bool
    message: str
    number_of_usable_quotes: int
    fit_seconds: float


def _phi(theta: np.ndarray | float, eta: float, gamma: float):
    """Power-law SSVI curvature function."""
    theta_array = np.asarray(theta, dtype=float)
    return eta / (
        theta_array**gamma
        * (1.0 + theta_array) ** (1.0 - gamma)
    )


def ssvi_total_variance(
    log_moneyness: np.ndarray | float,
    theta: np.ndarray | float,
    rho: float,
    eta: float,
    gamma: float,
) -> np.ndarray | float:
    """Evaluate SSVI total implied variance ``w(k, theta)``."""
    k, theta_array = np.broadcast_arrays(
        np.asarray(log_moneyness, dtype=float),
        np.asarray(theta, dtype=float),
    )
    if np.any(theta_array <= 0):
        raise ValueError("theta must be positive.")
    if not -1.0 < rho < 1.0:
        raise ValueError("rho must lie strictly between -1 and 1.")
    if eta <= 0:
        raise ValueError("eta must be positive.")
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must lie strictly between zero and one.")

    phi = _phi(theta_array, eta, gamma)
    z = phi * k + rho
    root = np.sqrt(z**2 + 1.0 - rho**2)
    result = 0.5 * theta_array * (
        1.0 + rho * phi * k + root
    )
    return float(result) if result.ndim == 0 else result


def ssvi_strike_derivatives(
    log_moneyness: np.ndarray | float,
    theta: np.ndarray | float,
    rho: float,
    eta: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``w``, ``w_k`` and ``w_kk`` analytically."""
    k, theta_array = np.broadcast_arrays(
        np.asarray(log_moneyness, dtype=float),
        np.asarray(theta, dtype=float),
    )
    phi = _phi(theta_array, eta, gamma)
    z = phi * k + rho
    root = np.sqrt(z**2 + 1.0 - rho**2)

    w = 0.5 * theta_array * (
        1.0 + rho * phi * k + root
    )
    w_k = 0.5 * theta_array * phi * (
        rho + z / root
    )
    w_kk = (
        0.5
        * theta_array
        * phi**2
        * (1.0 - rho**2)
        / root**3
    )
    return w, w_k, w_kk


def dupire_local_variance_from_derivatives(
    log_moneyness: np.ndarray,
    total_variance: np.ndarray,
    time_derivative: np.ndarray,
    first_strike_derivative: np.ndarray,
    second_strike_derivative: np.ndarray,
    denominator_floor: float = 1e-10,
) -> dict[str, np.ndarray]:
    """Return local variance from total implied variance derivatives."""
    k, w, w_t, w_k, w_kk = np.broadcast_arrays(
        np.asarray(log_moneyness, dtype=float),
        np.asarray(total_variance, dtype=float),
        np.asarray(time_derivative, dtype=float),
        np.asarray(first_strike_derivative, dtype=float),
        np.asarray(second_strike_derivative, dtype=float),
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = (
            (1.0 - k * w_k / (2.0 * w)) ** 2
            - 0.25 * (0.25 + 1.0 / w) * w_k**2
            + 0.5 * w_kk
        )
        local_variance = w_t / denominator

    valid = (
        np.isfinite(w)
        & np.isfinite(w_t)
        & np.isfinite(denominator)
        & np.isfinite(local_variance)
        & (w > 0.0)
        & (w_t > 0.0)
        & (denominator > denominator_floor)
        & (local_variance > 0.0)
    )

    output_variance = np.where(valid, local_variance, np.nan)
    output_volatility = np.where(
        valid,
        np.sqrt(output_variance),
        np.nan,
    )
    return {
        "denominator": denominator,
        "local_variance": output_variance,
        "local_volatility": output_volatility,
        "valid_mask": valid,
    }


def implied_total_variance_quotes(
    quote_data: pd.DataFrame,
    spot: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    price_column: str = "observed_call_price",
    noise_column: str = "noise_standard_deviation",
    minimum_implied_volatility: float = 0.03,
    maximum_total_variance_standard_deviation: float = 0.50,
) -> pd.DataFrame:
    """Convert option prices into implied total-variance observations.

    Price noise is approximately transferred into total-variance noise using
    Black--Scholes vega. Degenerate bound-clipped or near-zero-vega quotes are
    marked unusable rather than allowed to dominate the SSVI fit.
    """
    required = {
        "maturity",
        "log_moneyness",
        "strike",
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
        strike = float(getattr(row, "strike"))
        price = float(getattr(row, price_column))
        price_noise = float(getattr(row, noise_column))

        try:
            implied_volatility = implied_volatility_call(
                market_price=price,
                spot=spot,
                strike=strike,
                maturity=maturity,
                rate=rate,
                dividend_yield=dividend_yield,
                volatility_upper=3.0,
            )
            effective_volatility = max(
                implied_volatility,
                1e-4,
            )
            vega = float(
                black_scholes_vega(
                    spot=spot,
                    strike=strike,
                    maturity=maturity,
                    rate=rate,
                    volatility=effective_volatility,
                    dividend_yield=dividend_yield,
                )
            )
            total_variance = (
                implied_volatility**2 * maturity
            )
            total_variance_std = (
                price_noise
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
            and implied_volatility
            > minimum_implied_volatility
            and total_variance_std
            < maximum_total_variance_standard_deviation
            and vega > 1e-8
        )

        rows.append(
            {
                "maturity": maturity,
                "log_moneyness": float(
                    getattr(row, "log_moneyness")
                ),
                "strike": strike,
                "implied_volatility": implied_volatility,
                "total_variance": total_variance,
                "vega": vega,
                "total_variance_standard_deviation": (
                    total_variance_std
                ),
                "usable_for_ssvi": usable,
            }
        )

    return pd.DataFrame(rows)


def _decode_parameters(
    raw_parameters: np.ndarray,
    maturities: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    """Map unconstrained optimisation parameters into valid SSVI values."""
    number_of_maturities = maturities.size
    time_increments = np.diff(
        np.concatenate(([0.0], maturities))
    )
    instantaneous_variances = np.exp(
        raw_parameters[:number_of_maturities]
    )
    theta = np.cumsum(
        instantaneous_variances * time_increments
    )
    rho = 0.995 * np.tanh(
        raw_parameters[number_of_maturities]
    )
    eta = np.exp(
        raw_parameters[number_of_maturities + 1]
    )
    gamma = 0.05 + 0.90 * expit(
        raw_parameters[number_of_maturities + 2]
    )
    return theta, float(rho), float(eta), float(gamma)


def fit_ssvi_surface(
    quote_data: pd.DataFrame,
    spot: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    price_column: str = "observed_call_price",
    noise_column: str = "noise_standard_deviation",
    diagnostic_log_moneyness_limit: float = 0.60,
    standard_deviation_floor: float = 2e-5,
    standard_deviation_cap: float = 2e-2,
    maximum_function_evaluations: int = 10000,
) -> tuple[SSVIFit, pd.DataFrame]:
    """Fit an arbitrage-aware SSVI total-variance surface."""
    start_time = perf_counter()
    observations = implied_total_variance_quotes(
        quote_data=quote_data,
        spot=spot,
        rate=rate,
        dividend_yield=dividend_yield,
        price_column=price_column,
        noise_column=noise_column,
    )
    usable = observations[
        observations["usable_for_ssvi"]
    ].copy()
    maturities = np.sort(
        quote_data["maturity"].unique().astype(float)
    )

    if len(usable) < maturities.size + 3:
        raise ValueError(
            "Too few usable implied-volatility observations for SSVI."
        )

    maturity_index = {
        float(maturity): index
        for index, maturity in enumerate(maturities)
    }
    observation_indices = np.array(
        [
            maturity_index[float(value)]
            for value in usable["maturity"]
        ],
        dtype=int,
    )
    observed_k = usable[
        "log_moneyness"
    ].to_numpy(dtype=float)
    observed_w = usable[
        "total_variance"
    ].to_numpy(dtype=float)
    observed_std = np.clip(
        usable[
            "total_variance_standard_deviation"
        ].to_numpy(dtype=float),
        standard_deviation_floor,
        standard_deviation_cap,
    )

    # ATM total variance provides a stable term-structure initialisation.
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
                        maturity_data[
                            "log_moneyness"
                        ],
                        maturity_data[
                            "total_variance"
                        ],
                    )
                )
            )
    atm_theta = np.maximum.accumulate(
        np.maximum(np.asarray(atm_theta), 1e-6)
    )
    time_increments = np.diff(
        np.concatenate(([0.0], maturities))
    )
    theta_increments = np.diff(
        np.concatenate(([0.0], atm_theta))
    )
    initial_instantaneous_variance = np.clip(
        theta_increments / time_increments,
        1e-4,
        1.0,
    )

    diagnostic_k = np.linspace(
        -diagnostic_log_moneyness_limit,
        diagnostic_log_moneyness_limit,
        121,
    )

    def residuals(raw_parameters: np.ndarray) -> np.ndarray:
        theta, rho, eta, gamma = _decode_parameters(
            raw_parameters,
            maturities,
        )
        fitted = ssvi_total_variance(
            observed_k,
            theta[observation_indices],
            rho,
            eta,
            gamma,
        )
        data_residual = (
            fitted - observed_w
        ) / observed_std

        penalty_terms = []
        diagnostic_slices = []
        for theta_value in theta:
            w, w_k, w_kk = ssvi_strike_derivatives(
                diagnostic_k,
                theta_value,
                rho,
                eta,
                gamma,
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                density_factor = (
                    (
                        1.0
                        - diagnostic_k
                        * w_k
                        / (2.0 * w)
                    )
                    ** 2
                    - 0.25
                    * (0.25 + 1.0 / w)
                    * w_k**2
                    + 0.5 * w_kk
                )
            penalty_terms.extend(
                np.sqrt(1000.0)
                * np.minimum(density_factor, 0.0)
            )
            penalty_terms.extend(
                np.sqrt(1000.0)
                * np.minimum(w - 1e-10, 0.0)
            )

            phi_value = float(
                _phi(theta_value, eta, gamma)
            )
            # Lee-wing and Gatheral--Jacquier style sufficient bounds.
            penalty_terms.append(
                np.sqrt(100.0)
                * max(
                    theta_value
                    * phi_value
                    * (1.0 + abs(rho))
                    - 3.99,
                    0.0,
                )
            )
            penalty_terms.append(
                np.sqrt(100.0)
                * max(
                    theta_value
                    * phi_value**2
                    * (1.0 + abs(rho))
                    - 3.99,
                    0.0,
                )
            )
            diagnostic_slices.append(w)

        diagnostic_slices = np.asarray(
            diagnostic_slices
        )
        calendar_increments = np.diff(
            diagnostic_slices,
            axis=0,
        )
        penalty_terms.extend(
            (
                np.sqrt(1000.0)
                * np.minimum(
                    calendar_increments,
                    0.0,
                )
            ).reshape(-1)
        )

        return np.concatenate(
            [
                np.asarray(data_residual),
                np.asarray(penalty_terms),
            ]
        )

    base_logs = np.log(
        initial_instantaneous_variance
    )
    initialisations = []
    for rho_start, eta_start, gamma_start in [
        (0.0, 0.50, 0.50),
        (-0.40, 0.50, 0.50),
        (0.40, 0.50, 0.50),
        (0.0, 0.25, 0.25),
        (0.0, 0.90, 0.75),
    ]:
        raw_rho = np.arctanh(
            np.clip(rho_start / 0.995, -0.99, 0.99)
        )
        gamma_unit = (
            gamma_start - 0.05
        ) / 0.90
        raw_gamma = np.log(
            gamma_unit / (1.0 - gamma_unit)
        )
        initialisations.append(
            np.concatenate(
                [
                    base_logs,
                    [
                        raw_rho,
                        np.log(eta_start),
                        raw_gamma,
                    ],
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
        if (
            best_solution is None
            or solution.cost < best_solution.cost
        ):
            best_solution = solution

    theta, rho, eta, gamma = _decode_parameters(
        best_solution.x,
        maturities,
    )
    fit = SSVIFit(
        maturities=maturities,
        theta=theta,
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


def evaluate_ssvi_dupire(
    fit: SSVIFit,
    maturities: np.ndarray,
    log_moneyness: np.ndarray,
    denominator_floor: float = 1e-10,
) -> pd.DataFrame:
    """Evaluate the fitted SSVI surface and its Dupire local volatility."""
    query_T, query_k = np.broadcast_arrays(
        np.asarray(maturities, dtype=float),
        np.asarray(log_moneyness, dtype=float),
    )
    if np.any(query_T <= 0):
        raise ValueError("maturities must be positive.")

    theta_spline = PchipInterpolator(
        np.concatenate(([0.0], fit.maturities)),
        np.concatenate(([0.0], fit.theta)),
        extrapolate=True,
    )
    theta = theta_spline(query_T)
    theta_t = theta_spline.derivative()(query_T)

    phi = _phi(theta, fit.eta, fit.gamma)
    z = phi * query_k + fit.rho
    root = np.sqrt(
        z**2 + 1.0 - fit.rho**2
    )
    total_variance = 0.5 * theta * (
        1.0
        + fit.rho * phi * query_k
        + root
    )
    first_k = 0.5 * theta * phi * (
        fit.rho + z / root
    )
    second_k = (
        0.5
        * theta
        * phi**2
        * (1.0 - fit.rho**2)
        / root**3
    )

    derivative_log_phi = (
        -fit.gamma / theta
        + (fit.gamma - 1.0)
        / (1.0 + theta)
    )
    phi_theta = phi * derivative_log_phi
    derivative_theta = (
        0.5
        * (
            1.0
            + fit.rho * phi * query_k
            + root
        )
        + 0.5
        * theta
        * query_k
        * phi_theta
        * (fit.rho + z / root)
    )
    time_derivative = (
        derivative_theta * theta_t
    )

    dupire = dupire_local_variance_from_derivatives(
        log_moneyness=query_k,
        total_variance=total_variance,
        time_derivative=time_derivative,
        first_strike_derivative=first_k,
        second_strike_derivative=second_k,
        denominator_floor=denominator_floor,
    )

    return pd.DataFrame(
        {
            "maturity": query_T.reshape(-1),
            "log_moneyness": query_k.reshape(-1),
            "ssvi_total_variance": total_variance.reshape(-1),
            "ssvi_implied_volatility": np.sqrt(
                total_variance / query_T
            ).reshape(-1),
            "ssvi_time_derivative": time_derivative.reshape(-1),
            "ssvi_dupire_denominator": dupire[
                "denominator"
            ].reshape(-1),
            "local_variance": dupire[
                "local_variance"
            ].reshape(-1),
            "local_volatility": dupire[
                "local_volatility"
            ].reshape(-1),
            "valid_dupire": dupire[
                "valid_mask"
            ].reshape(-1),
            "method": "ssvi_dupire",
        }
    )


def ssvi_dupire_from_quotes(
    quote_data: pd.DataFrame,
    spot: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    price_column: str = "observed_call_price",
    noise_column: str = "noise_standard_deviation",
) -> tuple[pd.DataFrame, SSVIFit, pd.DataFrame]:
    """Fit SSVI to quotes and evaluate local volatility at quote locations."""
    fit, implied_data = fit_ssvi_surface(
        quote_data=quote_data,
        spot=spot,
        rate=rate,
        dividend_yield=dividend_yield,
        price_column=price_column,
        noise_column=noise_column,
    )
    result = evaluate_ssvi_dupire(
        fit=fit,
        maturities=quote_data[
            "maturity"
        ].to_numpy(dtype=float),
        log_moneyness=quote_data[
            "log_moneyness"
        ].to_numpy(dtype=float),
    )
    return result, fit, implied_data
