"""Check that continuous SSVI-Dupire local vol reprices SSVI prices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.evaluation.market_ssvi import (
    SSVIFit,
    evaluate_market_ssvi_quotes,
    make_market_ssvi_local_volatility_function,
)
from src.inverse.market_inverse import price_market_quotes_from_local_volatility_function


def run(project_root: Path, output_name: str) -> None:
    outputs = project_root / "outputs"
    p = json.loads((outputs / "stage_13" / "ssvi_parameters.json").read_text())
    fit = SSVIFit(
        np.asarray(p["maturities"], dtype=float), np.asarray(p["theta"], dtype=float),
        float(p["rho"]), float(p["eta"]), float(p["gamma"]), float(p["objective"]),
        bool(p["success"]), str(p["message"]), int(p["number_of_usable_quotes"]),
        float(p["fit_seconds"]),
    )
    quotes = pd.read_csv(
        outputs / "stage_13" / "checkpoints" / "market_panel.csv",
        parse_dates=["quote_date", "expiration"],
    )
    direct = evaluate_market_ssvi_quotes(fit, quotes)["ssvi_call_price"].to_numpy(dtype=float)
    sigma = make_market_ssvi_local_volatility_function(fit)
    rows = []
    for nx, nt in [(201, 300), (301, 600), (401, 1200)]:
        pde = price_market_quotes_from_local_volatility_function(
            sigma, quotes,
            pde_x_min=-1.0, pde_x_max=1.0,
            number_of_pde_x_points=nx,
            number_of_time_steps=nt,
            theta=0.5,
        )
        error = pde - direct
        rows.append({
            "x_points": nx,
            "time_steps": nt,
            "price_rmse_vs_direct_ssvi": float(np.sqrt(np.mean(error**2))),
            "price_mae_vs_direct_ssvi": float(np.mean(np.abs(error))),
            "max_absolute_price_error": float(np.max(np.abs(error))),
            "weighted_rmse_vs_direct_ssvi": float(np.sqrt(np.mean((error / quotes["noise_standard_deviation"].to_numpy(dtype=float))**2))),
        })
    output = outputs / output_name
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "consistency.csv", index=False)
    summary = {
        "model": "SSVI implied total variance differentiated by Dupire, evaluated as a continuous local-volatility callback",
        "quote_rows": int(len(quotes)),
        "pde_domain": [-1.0, 1.0],
        "interpretation": "Numerical consistency diagnostic only. It compares a PDE approximation of the fitted SSVI model with its direct normalized Black pricing formula; it is not an empirical model-selection or trading result.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(frame.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-name", default="stage_20_ssvi_pde_consistency")
    args = parser.parse_args()
    run(args.project_root.resolve(), args.output_name)


if __name__ == "__main__":
    main()
