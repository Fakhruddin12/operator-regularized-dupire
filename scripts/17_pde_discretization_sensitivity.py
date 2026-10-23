"""Stage 17: test whether PDE discretisation drives the residual-model gain.

The SSVI-implied local-volatility surface should reproduce the SSVI European
prices in the continuous limit. This script reprices the frozen SSVI reference
and frozen residual-operator surface on successively finer PDE grids, then
compares both with direct SSVI formula prices. No model parameters are refit.
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

from src.evaluation.market_comparison import market_model_metrics, market_prediction_table
from src.evaluation.market_ssvi import SSVIFit, evaluate_market_ssvi_quotes
from src.inverse.market_inverse import price_market_quotes_from_log_variance


def run(project_root: Path, output_name: str) -> None:
    stage13 = project_root / "outputs" / "stage_13"
    stage16 = project_root / "outputs" / "stage_16_ssvi_residual"
    panel = pd.read_csv(
        stage13 / "checkpoints" / "market_panel.csv",
        parse_dates=["quote_date", "expiration"],
    )
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
    surfaces = np.load(stage16 / "surfaces.npz")
    T, x = surfaces["maturities"], surfaces["log_moneyness"]
    ssvi_reference = surfaces["ssvi_reference_log_variance"]
    operator_surface = surfaces["fitted_log_variance"]
    direct_ssvi = evaluate_market_ssvi_quotes(fit, panel)["ssvi_call_price"].to_numpy(dtype=float)

    rows = []
    prediction_frames = []
    for x_points, time_steps in [(151, 100), (201, 200), (301, 400), (401, 800)]:
        print(f"Pricing grid {x_points} x {time_steps}", flush=True)
        ssvi_pde = price_market_quotes_from_log_variance(
            ssvi_reference, T, x, panel, -1.0, 1.0,
            x_points, time_steps,
        )
        residual_pde = price_market_quotes_from_log_variance(
            operator_surface, T, x, panel, -1.0, 1.0,
            x_points, time_steps,
        )
        pred = market_prediction_table(panel, {
            "ssvi_direct": direct_ssvi,
            "ssvi_localvol_pde": ssvi_pde,
            "operator_frozen": residual_pde,
        })
        metrics = market_model_metrics(
            pred, methods=["ssvi_direct", "ssvi_localvol_pde", "operator_frozen"]
        )
        metrics["pde_x_points"] = x_points
        metrics["pde_time_steps"] = time_steps
        rows.append(metrics)
        prediction_frames.append(pd.DataFrame({
            "pde_x_points": x_points,
            "pde_time_steps": time_steps,
            "ssvi_localvol_pde_price": ssvi_pde,
            "operator_frozen_price": residual_pde,
        }))

    output = project_root / "outputs" / output_name
    output.mkdir(parents=True, exist_ok=True)
    all_metrics = pd.concat(rows, ignore_index=True)
    all_metrics.to_csv(output / "metrics_by_grid.csv", index=False)
    pd.concat(prediction_frames, axis=1).to_csv(output / "predictions_by_grid.csv", index=False)
    summary = {
        "purpose": "PDE discretisation sensitivity; no refit at finer grids",
        "grids": [{"x_points": int(xp), "time_steps": int(nt)} for xp, nt in [(151, 100), (201, 200), (301, 400), (401, 800)]],
        "quote_rows": int(len(panel)),
        "test_rows": int(panel["is_test"].sum()),
        "interpretation": "A material drop in the SSVI-local-vol PDE pricing error as the grid is refined indicates discretisation bias. A frozen residual model's performance must be judged on the finest stable grid, and ultimately by a refit using that grid.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(all_metrics.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-name", default="stage_17_pde_sensitivity")
    args = parser.parse_args()
    run(args.project_root.resolve(), args.output_name)


if __name__ == "__main__":
    main()
