"""Leave-one-expiry-out cross-validation for the SSVI + operator correction.

All quotes at a target expiry are withheld together. The regularization level
is frozen at the value selected before this experiment (lambda=1000). Within
each fold, SSVI and the operator correction are fitted only on the other
expiries. By default, the correction receives up to four nonlinear
Gauss--Newton refinement steps; prices for the held-out expiry are generated
by the full nonlinear Dupire PDE. This is deliberately a grouped cross-sectional test on
one date, not chronological validation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.market_comparison import market_prediction_table
from src.evaluation.market_ssvi import (
    evaluate_market_ssvi_quotes,
    fit_market_ssvi_surface,
)
from src.inverse.market_inverse import price_market_quotes_from_local_volatility_function
from src.evaluation.tuning import linearized_lambda_sweep


def load_stage18():
    path = Path(__file__).with_name("18_ssvi_prior_full_domain_correction.py")
    spec = importlib.util.spec_from_file_location("stage18", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Stage 18 calibration functions")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize(predictions: pd.DataFrame, price_columns: dict[str, str]) -> pd.DataFrame:
    observed = predictions["observed_call_price"].to_numpy(float)
    noise = predictions["noise_standard_deviation"].to_numpy(float)
    bid = predictions["call_bid"].to_numpy(float)
    ask = predictions["call_ask"].to_numpy(float)
    rows = []
    for method, col in price_columns.items():
        err = predictions[col].to_numpy(float) - observed
        rows.append({
            "method": method,
            "number_of_quotes": int(len(predictions)),
            "price_rmse": float(np.sqrt(np.mean(err**2))),
            "weighted_price_rmse": float(np.sqrt(np.mean((err / noise)**2))),
            "inside_bid_ask_fraction": float(np.mean(
                (predictions[col].to_numpy(float) >= bid)
                & (predictions[col].to_numpy(float) <= ask)
            )),
        })
    return pd.DataFrame(rows)


def run(project_root: Path, max_folds: int | None = None,
        nonlinear_iterations: int = 4) -> None:
    stage18 = load_stage18()
    source = project_root / "outputs" / "stage_13" / "checkpoints" / "market_panel.csv"
    panel = pd.read_csv(source, parse_dates=["quote_date", "expiration"])
    expiries = sorted(panel["expiration"].dropna().unique())
    if max_folds is not None:
        expiries = expiries[:max_folds]

    output_name = (
        "stage_21_expiry_holdout_cv"
        if nonlinear_iterations == 0
        else "stage_22_expiry_holdout_nonlinear_cv"
    )
    output = project_root / "outputs" / output_name
    output.mkdir(parents=True, exist_ok=True)
    maturities, log_moneyness = stage18.make_correction_grid()
    lam = 1000.0  # frozen before this grouped validation
    pde = dict(pde_x_min=-1.0, pde_x_max=1.0,
               number_of_pde_x_points=301, number_of_time_steps=600, theta=0.5)
    fold_rows: list[pd.DataFrame] = []
    fold_meta: list[dict] = []

    for fold, expiry in enumerate(expiries, start=1):
        test = panel[panel["expiration"] == expiry].copy().reset_index(drop=True)
        train = panel[panel["expiration"] != expiry].copy().reset_index(drop=True)
        print(f"Fold {fold}/{len(expiries)} | expiry={pd.Timestamp(expiry).date()} | "
              f"train={len(train)} test={len(test)}", flush=True)

        fit, _ = fit_market_ssvi_surface(train, maximum_function_evaluations=5000)
        regularizer = stage18.make_regularizer(maturities, log_moneyness, train)
        zero = np.zeros((maturities.size, log_moneyness.size), dtype=float)
        base_train, jacobian = stage18.correction_jacobian(
            fit, zero, maturities, log_moneyness, train, verbose=True
        )
        observed_train = train["observed_call_price"].to_numpy(float)
        weights_train = 1.0 / train["noise_standard_deviation"].to_numpy(float)
        _, corrections = linearized_lambda_sweep(
            jacobian,
            observed_train - base_train,
            weights_train,
            regularizer,
            [lam],
        )
        initial_correction = corrections[lam].reshape(zero.shape, order="C")
        if nonlinear_iterations > 0:
            nonlinear_fit = stage18.calibrate(
                fit, initial_correction, maturities, log_moneyness,
                train, regularizer, lam, max_iterations=nonlinear_iterations,
            )
            correction = nonlinear_fit["correction"]
            nonlinear_stop_reason = nonlinear_fit["stop_reason"]
            nonlinear_steps = int(len(nonlinear_fit["history"]) - 1)
        else:
            correction = initial_correction
            nonlinear_stop_reason = "linearized_update_only"
            nonlinear_steps = 0

        ssvi_prices = evaluate_market_ssvi_quotes(fit, test)["ssvi_call_price"].to_numpy(float)
        corrected_prices = stage18.price_with_correction(
            fit, correction, maturities, log_moneyness, test
        )
        test["is_train"] = False
        test["is_test"] = True
        pred = market_prediction_table(test, {
            "ssvi": ssvi_prices,
            "operator_ssvi_correction": corrected_prices,
        })
        pred["fold_expiration"] = pd.Timestamp(expiry)
        fold_rows.append(pred)
        fold_meta.append({
            "fold": fold,
            "expiration": str(pd.Timestamp(expiry).date()),
            "maturity_years": float(test["maturity"].iloc[0]),
            "training_quotes": int(len(train)),
            "heldout_quotes": int(len(test)),
            "ssvi_fit_quotes": int(fit.number_of_usable_quotes),
            "frozen_lambda": lam,
            "nonlinear_steps": nonlinear_steps,
            "nonlinear_stop_reason": nonlinear_stop_reason,
            "correction_min": float(correction.min()),
            "correction_max": float(correction.max()),
            "fit_success": bool(fit.success),
        })
        pd.DataFrame(fold_meta).to_csv(output / "fold_summary.csv", index=False)
        print("  Completed grouped fold.", flush=True)

    all_predictions = pd.concat(fold_rows, ignore_index=True)
    all_predictions.to_csv(output / "out_of_fold_predictions.csv", index=False)
    metrics = summarize(all_predictions, {
        "ssvi": "ssvi_price",
        "ssvi_operator_correction_linearized": "operator_ssvi_correction_price",
    })
    metrics.to_csv(output / "metrics.csv", index=False)

    expiry_rows = []
    for expiry, group in all_predictions.groupby("fold_expiration"):
        m = summarize(group, {
            "ssvi": "ssvi_price",
            "ssvi_operator_correction_linearized": "operator_ssvi_correction_price",
        })
        m["expiration"] = str(pd.Timestamp(expiry).date())
        expiry_rows.append(m)
    by_expiry = pd.concat(expiry_rows, ignore_index=True)
    by_expiry.to_csv(output / "expiry_metrics.csv", index=False)

    pivot = by_expiry.pivot(index="expiration", columns="method", values="weighted_price_rmse")
    wins = (pivot["ssvi_operator_correction_linearized"] < pivot["ssvi"]).astype(int)
    summary = {
        "test": "leave-one-expiry-out grouped cross-validation",
        "valuation_dates": int(all_predictions["quote_date"].nunique()),
        "expiry_folds": int(len(pivot)),
        "quotes": int(len(all_predictions)),
        "frozen_lambda": lam,
        "operator_wins_by_expiry": int(wins.sum()),
        "expiry_win_fraction": float(wins.mean()),
        "pooled_metrics": metrics.to_dict(orient="records"),
        "nonlinear_iterations_requested": int(nonlinear_iterations),
        "interpretation": (
            "Whole expiries are held out; all quotes at the target expiry are excluded "
            "from both SSVI and correction fitting. The correction uses the regularized "
            "Stage 18 linearized initialization and, when requested, nonlinear Gauss-Newton "
            "refinement; evaluation uses the nonlinear PDE. Lambda is frozen from the "
            "earlier same-panel validation, so this is a grouped diagnostic but not a "
            "fully untouched nested test. One valuation "
            "date means this tests maturity transfer on one surface, not temporal or "
            "regime robustness."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--nonlinear-iterations", type=int, default=4,
                        help="Gauss-Newton iterations per fold; use 0 for the linearized diagnostic.")
    args = parser.parse_args()
    run(args.project_root.resolve(), args.max_folds, args.nonlinear_iterations)


if __name__ == "__main__":
    main()
