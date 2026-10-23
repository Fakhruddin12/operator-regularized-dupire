"""Stage 15: test whether the market fit was stopped too early.

The original SPX fit was limited to three Gauss--Newton steps. This diagnostic
restarts from the exact saved Stage 13 initialization and selected lambda, then
continues the nonlinear calibration without changing the data split, objective,
regularizer, or PDE grid. It writes to a new output directory and never
overwrites the frozen Stage 13 results.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix

from src.evaluation.market_comparison import (
    market_model_metrics,
    market_prediction_table,
)
from src.inverse.market_inverse import (
    price_market_quotes_from_log_variance,
    run_market_gauss_newton_calibration,
)
from src.inverse.parameterization import local_volatility_from_log_variance
from src.evaluation.tuning import linearized_lambda_sweep


def run(project_root: Path, maximum_iterations: int, output_name: str) -> None:
    source = project_root / "outputs" / "stage_13"
    checkpoint_dir = source / "checkpoints"
    output = project_root / "outputs" / output_name
    output.mkdir(parents=True, exist_ok=True)

    checkpoint = np.load(checkpoint_dir / "tuning_checkpoint.npz")
    panel = pd.read_csv(
        checkpoint_dir / "market_panel.csv",
        parse_dates=["quote_date", "expiration"],
    )
    train = pd.read_csv(
        checkpoint_dir / "outer_training_with_inner_split.csv",
        parse_dates=["quote_date", "expiration"],
    )
    train = train[train["is_train"]].reset_index(drop=True) if "is_train" in train else train

    maturity_grid = checkpoint["calibration_maturities"]
    x_grid = checkpoint["calibration_log_moneyness"]
    reference = checkpoint["reference_log_variance"]
    regularizer = csc_matrix(checkpoint["regularization_matrix"])
    reference_prices = checkpoint["reference_prices"]
    jacobian = checkpoint["jacobian"]
    strength = float(checkpoint["selected_lambda"])

    observed = train["observed_call_price"].to_numpy(dtype=float)
    noise = train["noise_standard_deviation"].to_numpy(dtype=float)
    _, corrections = linearized_lambda_sweep(
        jacobian=jacobian,
        residual=observed - reference_prices,
        weights=1.0 / noise,
        regularization_matrix=regularizer,
        lambda_values=[strength],
    )
    initial = reference + corrections[strength].reshape(reference.shape, order="C")

    result = run_market_gauss_newton_calibration(
        reference_log_variance=reference,
        initial_log_variance=initial,
        calibration_maturities=maturity_grid,
        calibration_log_moneyness=x_grid,
        quote_data=train,
        regularization_matrix=regularizer,
        regularization_strength=strength,
        finite_difference_step=1e-3,
        jacobian_scheme="forward",
        pde_x_min=-1.0,
        pde_x_max=1.0,
        number_of_pde_x_points=151,
        number_of_time_steps=100,
        maximum_iterations=maximum_iterations,
        maximum_absolute_step=0.5,
        relative_objective_tolerance=1e-5,
        relative_step_tolerance=1e-4,
        verbose=True,
    )

    surface = np.asarray(result["estimated_log_variance"], dtype=float)
    predictions = price_market_quotes_from_log_variance(
        log_variance_surface=surface,
        calibration_maturities=maturity_grid,
        calibration_log_moneyness=x_grid,
        quote_data=panel,
        pde_x_min=-1.0,
        pde_x_max=1.0,
        number_of_pde_x_points=151,
        number_of_time_steps=100,
    )
    old = pd.read_csv(source / "real_market_predictions.csv")
    old_predictions = old["operator_price"].to_numpy(dtype=float)
    table = market_prediction_table(panel, {
        "operator_extended": predictions,
        "operator_three_step": old_predictions,
        "ssvi": old["ssvi_price"].to_numpy(dtype=float),
    })
    metrics = market_model_metrics(
        table,
        methods=["ssvi", "operator_three_step", "operator_extended"],
    )

    table.to_csv(output / "predictions.csv", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    result["history"].to_csv(output / "history.csv", index=False)
    np.savez_compressed(
        output / "calibration.npz",
        reference_log_variance=reference,
        initial_log_variance=initial,
        estimated_log_variance=surface,
        local_volatility=local_volatility_from_log_variance(surface),
        selected_lambda=strength,
        regularization_matrix=regularizer.toarray(),
    )

    summary = {
        "experiment": "continue frozen Stage 13 constant-reference calibration",
        "selection": "all data choices, lambda, reference and PDE grid held fixed",
        "maximum_iterations_requested": maximum_iterations,
        "stop_reason": result.get("stop_reason", "not_returned"),
        "selected_lambda": strength,
        "outer_train_quotes": int(len(train)),
        "outer_test_quotes": int(panel["is_test"].sum()),
        "surface_local_volatility_min": float(np.min(local_volatility_from_log_variance(surface))),
        "surface_local_volatility_max": float(np.max(local_volatility_from_log_variance(surface))),
        "interpretation": "This is a solver-convergence diagnostic on one date, not an independent test or evidence of trading profitability.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--maximum-iterations", type=int, default=12)
    parser.add_argument("--output-name", default="stage_15_convergence")
    args = parser.parse_args()
    run(args.project_root.resolve(), args.maximum_iterations, args.output_name)


if __name__ == "__main__":
    main()
