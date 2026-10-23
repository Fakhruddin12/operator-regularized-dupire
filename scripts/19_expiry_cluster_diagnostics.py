"""Paired test-loss diagnostics grouped by expiry for Stage 18."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def run(project_root: Path, source_name: str, seed: int, repetitions: int) -> None:
    source = project_root / "outputs" / source_name
    data = pd.read_csv(source / "predictions.csv", parse_dates=["expiration"])
    test = data[data["is_test"].astype(bool)].copy()
    observed = test["observed_call_price"].to_numpy(dtype=float)
    noise = test["noise_standard_deviation"].to_numpy(dtype=float)
    test["loss_ssvi"] = ((test["ssvi_price"].to_numpy(dtype=float) - observed) / noise) ** 2
    test["loss_residual_operator"] = (
        (test["operator_ssvi_correction_price"].to_numpy(dtype=float) - observed) / noise
    ) ** 2
    test["paired_loss_reduction"] = test["loss_ssvi"] - test["loss_residual_operator"]

    by_expiry = test.groupby("expiration", as_index=False).agg(
        quotes=("observed_call_price", "size"),
        ssvi_weighted_mse=("loss_ssvi", "mean"),
        operator_weighted_mse=("loss_residual_operator", "mean"),
        mean_paired_loss_reduction=("paired_loss_reduction", "mean"),
    )
    by_expiry["ssvi_weighted_rmse"] = np.sqrt(by_expiry["ssvi_weighted_mse"])
    by_expiry["operator_weighted_rmse"] = np.sqrt(by_expiry["operator_weighted_mse"])
    by_expiry["operator_wins"] = by_expiry["mean_paired_loss_reduction"] > 0
    by_expiry.to_csv(source / "expiry_cluster_diagnostics.csv", index=False)

    rng = np.random.default_rng(seed)
    numerator = by_expiry["mean_paired_loss_reduction"].to_numpy(dtype=float) * by_expiry["quotes"].to_numpy(dtype=float)
    denominator = by_expiry["quotes"].to_numpy(dtype=float)
    n = len(by_expiry)
    draws = np.empty(repetitions)
    for b in range(repetitions):
        ids = rng.integers(0, n, size=n)
        draws[b] = numerator[ids].sum() / denominator[ids].sum()

    ssvi_mse = float(np.average(test["loss_ssvi"], weights=np.ones(len(test))))
    operator_mse = float(np.average(test["loss_residual_operator"], weights=np.ones(len(test))))
    summary = {
        "paired_test_quotes": int(len(test)),
        "expiry_clusters": int(n),
        "operator_wins_by_expiry": int(by_expiry["operator_wins"].sum()),
        "expiry_win_fraction": float(by_expiry["operator_wins"].mean()),
        "pooled_ssvi_weighted_mse": ssvi_mse,
        "pooled_operator_weighted_mse": operator_mse,
        "pooled_ssvi_weighted_rmse": float(np.sqrt(ssvi_mse)),
        "pooled_operator_weighted_rmse": float(np.sqrt(operator_mse)),
        "relative_weighted_rmse_reduction": float(1.0 - np.sqrt(operator_mse / ssvi_mse)),
        "mean_paired_weighted_squared_error_reduction": float(np.mean(test["paired_loss_reduction"])),
        "expiry_cluster_bootstrap_95_interval": [float(v) for v in np.quantile(draws, [0.025, 0.975])],
        "bootstrap_repetitions": int(repetitions),
        "bootstrap_seed": int(seed),
        "interpretation": (
            "Descriptive uncertainty over eight expiry clusters on one cross-sectional date. "
            "The clusters are not independent market dates; the interval is not evidence of temporal robustness or alpha."
        ),
    }
    (source / "expiry_cluster_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(by_expiry.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--source-name", default="stage_18_full_domain_correction")
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--repetitions", type=int, default=20000)
    args = parser.parse_args()
    run(args.project_root.resolve(), args.source_name, args.seed, args.repetitions)


if __name__ == "__main__":
    main()
