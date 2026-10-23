"""Stage 18: full-domain SSVI prior with a compact operator correction.

This resolves a material issue found in Stages 16--17: a local-volatility grid
restricted to the observed quote panel was being flat-extrapolated into the
unobserved wings and back to time zero, even though the forward PDE needs the
whole state/time domain. Here the SSVI local-volatility prior is evaluated
continuously over the full PDE domain; only a smooth, tapered correction is
estimated on a broader finite grid. Lambda is selected in an inner split and
the original outer quote holdout remains untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix

from src.evaluation.market_comparison import market_model_metrics, market_prediction_table
from src.evaluation.market_ssvi import (
    evaluate_market_ssvi_quotes,
    fit_market_ssvi_surface,
    make_market_ssvi_local_volatility_function,
)
from src.evaluation.market_tuning import add_inner_validation_split
from src.evaluation.tuning import linearized_lambda_sweep
from src.inverse.market_inverse import (
    finite_difference_market_jacobian,
    nonlinear_objective,
    price_market_quotes_from_local_volatility_function,
)
from src.inverse.nonlinear_inverse import solve_gauss_newton_step
from src.inverse.parameterization import local_volatility_from_log_variance
from src.inverse.reference_correction import make_reference_plus_correction_local_volatility
from src.pricing.forward_normalized_pde import (
    interpolate_normalized_call_prices,
    solve_forward_normalized_dupire,
)
from src.regularization.potentials import (
    combine_potentials,
    confidence_potential,
    quote_confidence_surface,
    wing_potential,
)
from src.regularization.scaling import build_nondimensional_regularization_matrix


PDE = {"pde_x_min": -1.0, "pde_x_max": 1.0, "number_of_pde_x_points": 301,
       "number_of_time_steps": 600, "theta": 0.5}


def make_correction_grid() -> tuple[np.ndarray, np.ndarray]:
    # Broader than the quote panel so the smooth taper has room to reach zero.
    return np.linspace(0.0, 1.20, 10), np.linspace(-0.50, 0.50, 25)


def make_regularizer(T: np.ndarray, x: np.ndarray, quotes: pd.DataFrame) -> csc_matrix:
    wing = wing_potential(x, T, strength=0.50, power=2.0, start=0.22)
    confidence = quote_confidence_surface(
        x, T,
        quotes["log_moneyness"].to_numpy(dtype=float),
        quotes["maturity"].to_numpy(dtype=float),
        log_moneyness_bandwidth=0.06,
        maturity_bandwidth=0.12,
    )
    matrix, _, _, _ = build_nondimensional_regularization_matrix(
        T, x, alpha_x=0.005, alpha_T=0.002, beta=1e-4,
        potential=combine_potentials(
            wing, confidence_potential(confidence, strength=0.20)
        ),
    )
    return matrix


def price_with_correction(fit, h: np.ndarray, T: np.ndarray, x: np.ndarray,
                          quotes: pd.DataFrame) -> np.ndarray:
    sigma = make_reference_plus_correction_local_volatility(
        reference_local_volatility=make_market_ssvi_local_volatility_function(fit),
        correction_surface=h,
        correction_maturities=T,
        correction_log_moneyness=x,
        maturity_taper=0.08,
        log_moneyness_taper=0.10,
    )
    return price_market_quotes_from_local_volatility_function(
        sigma, quotes, **PDE
    )


def correction_jacobian(fit, h: np.ndarray, T: np.ndarray, x: np.ndarray,
                        quotes: pd.DataFrame, step: float = 1e-3,
                        verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    base = price_with_correction(fit, h, T, x, quotes)
    J = np.empty((len(quotes), h.size), dtype=float)
    for j in range(h.size):
        if verbose and (j == 0 or (j + 1) % 25 == 0 or j + 1 == h.size):
            print(f"Correction Jacobian column {j + 1}/{h.size}", flush=True)
        perturbed = h.copy().reshape(-1, order="C")
        perturbed[j] += step
        perturbed = perturbed.reshape(h.shape, order="C")
        J[:, j] = (price_with_correction(fit, perturbed, T, x, quotes) - base) / step
    return base, J


def select_lambda(fit, T: np.ndarray, x: np.ndarray, R: csc_matrix,
                  inner: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    zero = np.zeros((T.size, x.size), dtype=float)
    reference_prices, J = correction_jacobian(fit, zero, T, x, inner, verbose=True)
    fit_mask = inner["is_inner_fit"].to_numpy(dtype=bool)
    validation_mask = inner["is_inner_validation"].to_numpy(dtype=bool)
    observed = inner["observed_call_price"].to_numpy(dtype=float)
    noise = inner["noise_standard_deviation"].to_numpy(dtype=float)
    values = np.logspace(2, 8, 13)
    linear, corrections = linearized_lambda_sweep(
        J[fit_mask], observed[fit_mask] - reference_prices[fit_mask],
        1.0 / noise[fit_mask], R, values,
    )
    validation_quotes = inner.loc[validation_mask].reset_index(drop=True)
    rows = []
    for lam in values:
        candidate_h = corrections[float(lam)].reshape(zero.shape, order="C")
        predicted = price_with_correction(fit, candidate_h, T, x, validation_quotes)
        residual = (
            predicted
            - validation_quotes["observed_call_price"].to_numpy(dtype=float)
        ) / validation_quotes["noise_standard_deviation"].to_numpy(dtype=float)
        linear_row = linear[np.isclose(linear["lambda"], lam)].iloc[0]
        rows.append({
            "lambda": float(lam),
            "validation_weighted_price_rmse": float(np.sqrt(np.mean(residual**2))),
            "validation_price_rmse": float(np.sqrt(np.mean((predicted - validation_quotes["observed_call_price"].to_numpy(dtype=float))**2))),
            "linearized_inner_fit_weighted_rmse": float(linear_row["weighted_rmse"]),
            "effective_degrees_of_freedom": float(linear_row["effective_degrees_of_freedom"]),
            "correction_norm": float(linear_row["correction_norm"]),
        })
    results = pd.DataFrame(rows)
    best = int(results["validation_weighted_price_rmse"].idxmin())
    return results, float(results.loc[best, "lambda"])


def calibrate(fit, initial_h: np.ndarray, T: np.ndarray, x: np.ndarray,
              quotes: pd.DataFrame, R: csc_matrix, lam: float,
              max_iterations: int) -> dict:
    observed = quotes["observed_call_price"].to_numpy(dtype=float)
    noise = quotes["noise_standard_deviation"].to_numpy(dtype=float)
    weights = 1.0 / noise
    current = initial_h.copy()
    prices = price_with_correction(fit, current, T, x, quotes)
    metrics = nonlinear_objective(prices, observed, weights, current.ravel(), R, lam)
    history = [{"iteration": 0, **metrics, "step_scale": 0.0, "step_norm": 0.0}]
    damping = 1e-2
    stop_reason = "maximum_iterations_reached"
    for iteration in range(1, max_iterations + 1):
        prices, J = correction_jacobian(fit, current, T, x, quotes, verbose=True)
        residual = observed - prices
        accepted = False
        damping_used = damping
        for _ in range(7):
            step_result = solve_gauss_newton_step(
                J, residual, weights, current.ravel(), R, lam, damping=damping_used
            )
            step = np.asarray(step_result["step"], dtype=float)
            largest = float(np.max(np.abs(step)))
            if largest > 0.35:
                step *= 0.35 / largest
            scale = 1.0
            while scale >= 1.0 / 128.0:
                candidate = current + scale * step.reshape(current.shape, order="C")
                try:
                    candidate_prices = price_with_correction(fit, candidate, T, x, quotes)
                    candidate_metrics = nonlinear_objective(
                        candidate_prices, observed, weights, candidate.ravel(), R, lam
                    )
                except (ValueError, FloatingPointError, RuntimeError):
                    scale *= 0.5
                    continue
                if candidate_metrics["objective"] < metrics["objective"]:
                    accepted = True
                    previous_objective = metrics["objective"]
                    current, prices, metrics = candidate, candidate_prices, candidate_metrics
                    accepted_step = scale * step
                    accepted_scale = scale
                    break
                scale *= 0.5
            if accepted:
                break
            damping_used *= 10.0
        if not accepted:
            stop_reason = "no_objective_decrease"
            break
        step_norm = float(np.linalg.norm(accepted_step))
        rel_improvement = (previous_objective - metrics["objective"]) / max(1.0, abs(previous_objective))
        history.append({
            "iteration": iteration, **metrics,
            "step_scale": accepted_scale, "step_norm": step_norm,
            "damping": damping_used,
        })
        print(f"Iteration {iteration}: objective={metrics['objective']:.6f}, weighted RMSE={metrics['weighted_rmse']:.6f}", flush=True)
        damping = max(1e-12, damping_used * 0.3)
        if rel_improvement <= 1e-5:
            stop_reason = "relative_objective_tolerance"
            break
    final_prices = price_with_correction(fit, current, T, x, quotes)
    return {"correction": current, "prices": final_prices,
            "history": pd.DataFrame(history), "stop_reason": stop_reason}


def run(project_root: Path, max_iterations: int, output_name: str) -> None:
    stage13 = project_root / "outputs" / "stage_13"
    panel = pd.read_csv(
        stage13 / "checkpoints" / "market_panel.csv",
        parse_dates=["quote_date", "expiration"],
    )
    outer_train = panel[panel["is_train"]].reset_index(drop=True)
    outer_test = panel[panel["is_test"]].reset_index(drop=True)
    inner = add_inner_validation_split(outer_train, every_nth_quote=5, offset=2)
    inner_fit = inner[inner["is_inner_fit"]].reset_index(drop=True)
    T, x = make_correction_grid()
    output = project_root / "outputs" / output_name
    output.mkdir(parents=True, exist_ok=True)

    # Inner training fit -> held-out inner quotes -> frozen lambda choice.
    inner_ssvi, _ = fit_market_ssvi_surface(inner_fit, maximum_function_evaluations=5000)
    inner_R = make_regularizer(T, x, inner_fit)
    lambda_results, lam = select_lambda(inner_ssvi, T, x, inner_R, inner)
    lambda_results.to_csv(output / "lambda_inner_validation.csv", index=False)

    # Refit SSVI on all outer training quotes, then estimate only the residual.
    final_ssvi, _ = fit_market_ssvi_surface(outer_train, maximum_function_evaluations=5000)
    R = make_regularizer(T, x, outer_train)
    zero = np.zeros((T.size, x.size), dtype=float)
    base_prices, J = correction_jacobian(final_ssvi, zero, T, x, outer_train, verbose=True)
    linear, corrections = linearized_lambda_sweep(
        J,
        outer_train["observed_call_price"].to_numpy(dtype=float) - base_prices,
        1.0 / outer_train["noise_standard_deviation"].to_numpy(dtype=float),
        R,
        [lam],
    )
    initial_h = corrections[lam].reshape(zero.shape, order="C")
    result = calibrate(final_ssvi, initial_h, T, x, outer_train, R, lam, max_iterations)

    # Persist the expensive calibration before downstream metric/report work.
    np.savez_compressed(
        output / "surfaces.npz",
        correction_maturities=T,
        correction_log_moneyness=x,
        initial_log_variance_correction=initial_h,
        fitted_log_variance_correction=result["correction"],
        ssvi_theta=final_ssvi.theta,
        ssvi_maturities=final_ssvi.maturities,
        ssvi_rho=final_ssvi.rho,
        ssvi_eta=final_ssvi.eta,
        ssvi_gamma=final_ssvi.gamma,
    )
    result["history"].to_csv(output / "history.csv", index=False)

    operator_prices = price_with_correction(final_ssvi, result["correction"], T, x, panel)
    ssvi_prices = evaluate_market_ssvi_quotes(final_ssvi, panel)["ssvi_call_price"].to_numpy(dtype=float)
    pde_prior_prices = price_with_correction(final_ssvi, zero, T, x, panel)
    prediction_values = {
        "ssvi": ssvi_prices,
        "ssvi_global_pde": pde_prior_prices,
        "operator_ssvi_correction": operator_prices,
    }
    # Include the converged constant-reference model as the earlier baseline.
    stage15_file = project_root / "outputs" / "stage_15_convergence" / "predictions.csv"
    methods = ["ssvi", "ssvi_global_pde", "operator_ssvi_correction"]
    if stage15_file.exists():
        previous = pd.read_csv(stage15_file)
        prediction_values["operator_constant_reference"] = previous["operator_extended_price"].to_numpy(dtype=float)
        methods.append("operator_constant_reference")
    pred = market_prediction_table(panel, prediction_values)
    metrics = market_model_metrics(pred, methods)

    output.joinpath("predictions.csv").write_text(pred.to_csv(index=False), encoding="utf-8")
    metrics.to_csv(output / "metrics.csv", index=False)
    result["history"].to_csv(output / "history.csv", index=False)
    np.savez_compressed(
        output / "surfaces.npz",
        correction_maturities=T,
        correction_log_moneyness=x,
        initial_log_variance_correction=initial_h,
        fitted_log_variance_correction=result["correction"],
        ssvi_theta=final_ssvi.theta,
        ssvi_maturities=final_ssvi.maturities,
        ssvi_rho=final_ssvi.rho,
        ssvi_eta=final_ssvi.eta,
        ssvi_gamma=final_ssvi.gamma,
    )
    # Expiry-level paired diagnostics make quote dependence visible.
    d = pred.copy()
    d["operator_weighted_sq_error"] = ((d["operator_ssvi_correction_price"] - d["observed_call_price"]) / d["noise_standard_deviation"])**2
    d["ssvi_weighted_sq_error"] = ((d["ssvi_price"] - d["observed_call_price"]) / d["noise_standard_deviation"])**2
    expiry = d.groupby(["expiration", "is_test"], as_index=False).agg(
        n=("observed_call_price", "size"),
        operator_weighted_mse=("operator_weighted_sq_error", "mean"),
        ssvi_weighted_mse=("ssvi_weighted_sq_error", "mean"),
    )
    expiry.to_csv(output / "expiry_level_metrics.csv", index=False)
    summary = {
        "experiment": "SSVI continuous local-vol prior + compact tapered operator correction",
        "panel_quotes": int(len(panel)),
        "outer_train_quotes": int(len(outer_train)),
        "outer_test_quotes": int(len(outer_test)),
        "inner_fit_quotes": int(len(inner_fit)),
        "inner_validation_quotes": int(inner["is_inner_validation"].sum()),
        "selected_lambda": float(lam),
        "pde": PDE,
        "stop_reason": result["stop_reason"],
        "correction_min": float(result["correction"].min()),
        "correction_max": float(result["correction"].max()),
        "interpretation": "One-date nested cross-sectional calibration. It does not establish time-series robustness, execution quality, hedging performance, or trading profitability.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--maximum-iterations", type=int, default=10)
    parser.add_argument("--output-name", default="stage_18_full_domain_correction")
    args = parser.parse_args()
    run(args.project_root.resolve(), args.maximum_iterations, args.output_name)


if __name__ == "__main__":
    main()
