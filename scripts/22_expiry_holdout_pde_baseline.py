"""Add a PDE-matched SSVI comparator to Stage 22 grouped holdout results."""

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

from src.evaluation.market_comparison import market_model_metrics
from src.evaluation.market_ssvi import fit_market_ssvi_surface
from src.evaluation.market_comparison import implied_volatility_from_model_prices


def load_stage18():
    path = Path(__file__).with_name("18_ssvi_prior_full_domain_correction.py")
    spec = importlib.util.spec_from_file_location("stage18", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Stage 18 pricing functions")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(project_root: Path) -> None:
    stage18 = load_stage18()
    output = project_root / "outputs" / "stage_22_expiry_holdout_nonlinear_cv"
    source = project_root / "outputs" / "stage_13" / "checkpoints" / "market_panel.csv"
    panel = pd.read_csv(source, parse_dates=["quote_date", "expiration"])
    predictions = pd.read_csv(output / "out_of_fold_predictions.csv", parse_dates=["expiration"])
    maturities, x = stage18.make_correction_grid()
    zero = np.zeros((maturities.size, x.size), dtype=float)
    rows = []
    expiries = sorted(panel["expiration"].unique())
    for i, expiry in enumerate(expiries, start=1):
        train = panel[panel["expiration"] != expiry].reset_index(drop=True)
        test = panel[panel["expiration"] == expiry].reset_index(drop=True)
        fit, _ = fit_market_ssvi_surface(train, maximum_function_evaluations=5000)
        prices = stage18.price_with_correction(fit, zero, maturities, x, test)
        rows.append(pd.DataFrame({
            "expiration": pd.Timestamp(expiry),
            "strike": test["strike"].to_numpy(float),
            "ssvi_pde_price": prices,
        }))
        print(f"PDE-matched SSVI baseline {i}/{len(expiries)}", flush=True)
    baseline = pd.concat(rows, ignore_index=True)
    predictions = predictions.merge(baseline, on=["expiration", "strike"], how="left", validate="one_to_one")
    if predictions["ssvi_pde_price"].isna().any():
        raise RuntimeError("Failed to align all PDE baseline prices")
    predictions["ssvi_pde_implied_volatility"] = implied_volatility_from_model_prices(
        predictions, predictions["ssvi_pde_price"].to_numpy(float)
    )
    predictions.to_csv(output / "out_of_fold_predictions.csv", index=False)

    # Rename the correction to make clear this is the full nonlinear fit.
    predictions["ssvi_operator_correction_nonlinear_price"] = predictions[
        "operator_ssvi_correction_price"
    ]
    predictions["ssvi_operator_correction_nonlinear_implied_volatility"] = predictions[
        "operator_ssvi_correction_implied_volatility"
    ]
    methods = ["ssvi", "ssvi_pde", "ssvi_operator_correction_nonlinear"]
    metrics = market_model_metrics(predictions, methods)
    metrics.to_csv(output / "metrics.csv", index=False)

    expiry_parts = []
    for expiry, group in predictions.groupby("fold_expiration"):
        table = market_model_metrics(group, methods)
        table["expiration"] = str(pd.Timestamp(expiry).date())
        expiry_parts.append(table)
    by_expiry = pd.concat(expiry_parts, ignore_index=True)
    by_expiry.to_csv(output / "expiry_metrics.csv", index=False)
    weighted = by_expiry.pivot(index="expiration", columns="method", values="weighted_price_rmse")
    correction_wins = (weighted["ssvi_operator_correction_nonlinear"] < weighted["ssvi_pde"]).astype(int)
    summary = json.loads((output / "summary.json").read_text())
    summary["operator_wins_vs_pde_ssvi_by_expiry"] = int(correction_wins.sum())
    summary["expiry_win_fraction_vs_pde_ssvi"] = float(correction_wins.mean())
    summary["pooled_metrics"] = metrics.to_dict(orient="records")
    summary["interpretation"] = (
        "Leave-one-expiry-out on one valuation date. Each expiry is excluded in full "
        "from fitting both SSVI and the correction. The correction is nonlinearly refined; "
        "all predictions use the full PDE. Lambda=1000 was frozen from earlier validation "
        "on this same panel, so this is not a fully untouched nested test. Results test "
        "maturity transfer on one cross-section, not chronological/regime robustness."
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.project_root.resolve())


if __name__ == "__main__":
    main()
