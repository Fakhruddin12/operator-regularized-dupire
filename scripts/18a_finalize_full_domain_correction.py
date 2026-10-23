"""Rebuild Stage 18 evaluation files from its saved calibration checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.market_comparison import market_model_metrics, market_prediction_table
from src.evaluation.market_ssvi import (
    SSVIFit,
    evaluate_market_ssvi_quotes,
    make_market_ssvi_local_volatility_function,
)
from src.inverse.market_inverse import price_market_quotes_from_local_volatility_function
from src.inverse.reference_correction import make_reference_plus_correction_local_volatility


def finalize(project_root: Path, source_name: str) -> None:
    root = project_root / "outputs"
    stage13 = root / "stage_13"
    output = root / source_name
    params = json.loads((stage13 / "ssvi_parameters.json").read_text())
    fit = SSVIFit(
        maturities=np.asarray(params["maturities"], dtype=float),
        theta=np.asarray(params["theta"], dtype=float),
        rho=float(params["rho"]), eta=float(params["eta"]),
        gamma=float(params["gamma"]), objective=float(params["objective"]),
        success=bool(params["success"]), message=str(params["message"]),
        number_of_usable_quotes=int(params["number_of_usable_quotes"]),
        fit_seconds=float(params["fit_seconds"]),
    )
    panel = pd.read_csv(
        stage13 / "checkpoints" / "market_panel.csv",
        parse_dates=["quote_date", "expiration"],
    )
    calibration = np.load(output / "surfaces.npz")
    T = calibration["correction_maturities"]
    x = calibration["correction_log_moneyness"]
    h = calibration["fitted_log_variance_correction"]
    zero = np.zeros_like(h)

    def price(correction: np.ndarray) -> np.ndarray:
        sigma = make_reference_plus_correction_local_volatility(
            make_market_ssvi_local_volatility_function(fit),
            correction, T, x,
            maturity_taper=0.08,
            log_moneyness_taper=0.10,
        )
        return price_market_quotes_from_local_volatility_function(
            sigma, panel,
            pde_x_min=-1.0,
            pde_x_max=1.0,
            number_of_pde_x_points=301,
            number_of_time_steps=600,
            theta=0.5,
        )

    pde_prior = price(zero)
    operator = price(h)
    ssvi = evaluate_market_ssvi_quotes(fit, panel)["ssvi_call_price"].to_numpy(dtype=float)
    prediction_values = {"ssvi": ssvi, "ssvi_global_pde": pde_prior,
                         "operator_ssvi_correction": operator}
    methods = list(prediction_values)
    stage15 = root / "stage_15_convergence" / "predictions.csv"
    if stage15.exists():
        prior = pd.read_csv(stage15)
        prediction_values["operator_constant_reference"] = prior["operator_extended_price"].to_numpy(dtype=float)
        methods.append("operator_constant_reference")

    predictions = market_prediction_table(panel, prediction_values)
    metrics = market_model_metrics(predictions, methods)
    predictions.to_csv(output / "predictions.csv", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)

    grouped = predictions.copy()
    observed = grouped["observed_call_price"].to_numpy(dtype=float)
    noise = grouped["noise_standard_deviation"].to_numpy(dtype=float)
    grouped["operator_sq_error"] = ((grouped["operator_ssvi_correction_price"] - observed) / noise) ** 2
    grouped["ssvi_sq_error"] = ((grouped["ssvi_price"] - observed) / noise) ** 2
    expiry = grouped.groupby(["expiration", "is_test"], as_index=False).agg(
        n=("observed_call_price", "size"),
        operator_weighted_mse=("operator_sq_error", "mean"),
        ssvi_weighted_mse=("ssvi_sq_error", "mean"),
    )
    expiry.to_csv(output / "expiry_level_metrics.csv", index=False)

    history = pd.read_csv(output / "history.csv")
    summary = {
        "experiment": "SSVI continuous local-vol prior + compact tapered operator correction",
        "panel_quotes": int(len(panel)),
        "outer_train_quotes": int(panel["is_train"].sum()),
        "outer_test_quotes": int(panel["is_test"].sum()),
        "selected_lambda": float(pd.read_csv(output / "lambda_inner_validation.csv").sort_values("validation_weighted_price_rmse").iloc[0]["lambda"]),
        "pde": {"pde_x_min": -1.0, "pde_x_max": 1.0, "number_of_pde_x_points": 301, "number_of_time_steps": 600, "theta": 0.5},
        "iterations": int(max(0, len(history) - 1)),
        "stop_reason": "relative_objective_tolerance",
        "correction_min": float(h.min()),
        "correction_max": float(h.max()),
        "interpretation": "One-date nested cross-sectional calibration. It does not establish time-series robustness, execution quality, hedging performance, or trading profitability.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", type=Path, default=ROOT)
    ap.add_argument("--source-name", default="stage_18_full_domain_correction")
    args = ap.parse_args()
    finalize(args.project_root.resolve(), args.source_name)


if __name__ == "__main__":
    main()
