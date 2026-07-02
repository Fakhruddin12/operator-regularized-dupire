"""Run the final realistic-bumpy synthetic local-volatility benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PROJECT_ROOT))

from src.evaluation.realistic_bumpy_benchmark import (
    compare_final_methods,
    generate_realistic_bumpy_quotes,
    multi_feature_recovery_metrics,
    repeated_noise_summary,
    run_repeated_noise_comparison,
)
from src.evaluation.tuning import linearized_lambda_sweep, select_lambda_by_gcv
from src.inverse.linearized_inverse import finite_difference_jacobian, quote_weights
from src.inverse.nonlinear_inverse import run_gauss_newton_calibration
from src.inverse.parameterization import (
    local_volatility_from_log_variance,
    reference_log_variance_surface,
)
from src.regularization.potentials import (
    combine_potentials,
    confidence_potential,
    quote_confidence_surface,
    wing_potential,
)
from src.regularization.scaling import build_nondimensional_regularization_matrix


def main(project_root: Path, mode: str, overwrite: bool) -> None:
    output_directory = project_root / "outputs" / "stage_14_final"
    summary_path = output_directory / "benchmark_summary.json"
    if summary_path.exists() and not overwrite:
        print(f"Existing completed output found: {summary_path}")
        print("Use --overwrite to rerun the calibration.")
        return

    output_directory.mkdir(parents=True, exist_ok=True)
    data_directory = project_root / "data" / "synthetic"
    data_directory.mkdir(parents=True, exist_ok=True)

    if mode == "full":
        generation_strike_points = 241
        generation_time_steps = 220
        inverse_strike_points = 141
        inverse_time_steps = 110
        maximum_iterations = 3
        noise_seeds = range(10)
    else:
        generation_strike_points = 161
        generation_time_steps = 140
        inverse_strike_points = 101
        inverse_time_steps = 70
        maximum_iterations = 1
        noise_seeds = range(3)

    spot = 100.0
    quotes = generate_realistic_bumpy_quotes(
        random_seed=314159,
        spot=spot,
        number_of_strike_points=generation_strike_points,
        number_of_time_steps=generation_time_steps,
    )
    quote_path = data_directory / "realistic_bumpy_quotes.csv"
    quotes.to_csv(quote_path, index=False)

    calibration_T = np.linspace(
        float(quotes["maturity"].min()),
        float(quotes["maturity"].max()),
        6,
    )
    calibration_x = np.linspace(-0.34, 0.34, 11)
    reference_log_variance = reference_log_variance_surface(
        maturities=calibration_T,
        log_moneyness=calibration_x,
        reference_volatility=0.20,
    )

    wing = wing_potential(
        log_moneyness=calibration_x,
        maturities=calibration_T,
        strength=0.20,
        power=2.0,
        start=0.20,
    )
    confidence = quote_confidence_surface(
        log_moneyness=calibration_x,
        maturities=calibration_T,
        quote_log_moneyness=quotes["log_moneyness"].to_numpy(dtype=float),
        quote_maturities=quotes["maturity"].to_numpy(dtype=float),
        log_moneyness_bandwidth=0.08,
        maturity_bandwidth=0.22,
    )
    regularization_matrix, _, _, scaling = (
        build_nondimensional_regularization_matrix(
            maturities=calibration_T,
            log_moneyness=calibration_x,
            alpha_x=0.005,
            alpha_T=0.002,
            beta=1e-4,
            potential=combine_potentials(
                wing,
                confidence_potential(confidence, strength=0.20),
            ),
        )
    )

    print(f"Quotes: {len(quotes)}")
    print(f"Operator unknowns: {reference_log_variance.size}")

    reference_prices, jacobian = finite_difference_jacobian(
        reference_log_variance=reference_log_variance,
        calibration_maturities=calibration_T,
        calibration_log_moneyness=calibration_x,
        quote_data=quotes,
        spot=spot,
        finite_difference_step=1e-3,
        scheme="forward",
        number_of_strike_points=inverse_strike_points,
        number_of_time_steps=inverse_time_steps,
        verbose=True,
    )
    weights = quote_weights(quotes)
    observed = quotes["observed_call_price"].to_numpy(dtype=float)
    residual = observed - reference_prices

    lambda_values = np.logspace(-1, 5, 13)
    lambda_results, lambda_solutions = linearized_lambda_sweep(
        jacobian=jacobian,
        residual=residual,
        weights=weights,
        regularization_matrix=regularization_matrix,
        lambda_values=lambda_values,
    )
    selected_lambda = float(select_lambda_by_gcv(lambda_results))
    print("GCV-selected lambda:", selected_lambda)

    initial_log_variance = reference_log_variance + lambda_solutions[
        selected_lambda
    ].reshape(reference_log_variance.shape, order="C")

    operator_result = run_gauss_newton_calibration(
        reference_log_variance=reference_log_variance,
        initial_log_variance=initial_log_variance,
        calibration_maturities=calibration_T,
        calibration_log_moneyness=calibration_x,
        quote_data=quotes,
        regularization_matrix=regularization_matrix,
        regularization_strength=selected_lambda,
        spot=spot,
        finite_difference_step=1e-3,
        jacobian_scheme="forward",
        number_of_strike_points=inverse_strike_points,
        number_of_time_steps=inverse_time_steps,
        maximum_iterations=maximum_iterations,
        initial_damping=1e-2,
        maximum_absolute_step=1.0,
        relative_objective_tolerance=1e-4,
        relative_step_tolerance=1e-3,
        verbose=True,
    )
    operator_volatility = local_volatility_from_log_variance(
        operator_result["estimated_log_variance"]
    )

    comparison, pointwise, metadata = compare_final_methods(
        quote_data=quotes,
        calibration_maturities=calibration_T,
        calibration_log_moneyness=calibration_x,
        operator_volatility_surface=operator_volatility,
        spot=spot,
    )
    feature_metrics = multi_feature_recovery_metrics(pointwise)
    repeated = run_repeated_noise_comparison(
        clean_quote_data=quotes,
        random_seeds=noise_seeds,
        reference_prices=reference_prices,
        jacobian=jacobian,
        weights=weights,
        reference_log_variance=reference_log_variance,
        calibration_maturities=calibration_T,
        calibration_log_moneyness=calibration_x,
        regularization_matrix=regularization_matrix,
        regularization_strength=selected_lambda,
        spot=spot,
    )
    repeated_summary = repeated_noise_summary(repeated)

    comparison.to_csv(output_directory / "common_grid_comparison.csv", index=False)
    pointwise.to_csv(output_directory / "pointwise_results.csv", index=False)
    feature_metrics.to_csv(output_directory / "feature_recovery_metrics.csv", index=False)
    lambda_results.to_csv(output_directory / "lambda_sweep.csv", index=False)
    repeated.to_csv(output_directory / "repeated_noise_results.csv", index=False)
    repeated_summary.to_csv(
        output_directory / "repeated_noise_summary.csv", index=False
    )
    pd.DataFrame(operator_result["history"]).to_csv(
        output_directory / "operator_history.csv", index=False
    )
    np.savez_compressed(
        output_directory / "operator_calibration.npz",
        calibration_maturities=calibration_T,
        calibration_x=calibration_x,
        reference_log_variance=reference_log_variance,
        estimated_log_variance=operator_result["estimated_log_variance"],
        estimated_volatility=operator_volatility,
        selected_lambda=selected_lambda,
        reference_prices=reference_prices,
        reference_jacobian=jacobian,
        weights=weights,
    )

    winner = str(comparison.iloc[0]["method"])
    repeated_winner = str(repeated_summary.iloc[0]["method"])
    summary = {
        "mode": mode,
        "number_of_quotes": int(len(quotes)),
        "number_of_operator_unknowns": int(reference_log_variance.size),
        "selected_lambda": selected_lambda,
        "operator_stop_reason": str(operator_result["stop_reason"]),
        "operator_final_weighted_price_rmse": float(
            operator_result["final_weighted_rmse"]
        ),
        "single_noise_local_volatility_winner": winner,
        "repeated_noise_mean_rmse_winner": repeated_winner,
        "shared_valid_fraction": float(pointwise["shared_valid"].mean()),
        "ssvi_rho": float(metadata["ssvi_fit"].rho),
        "ssvi_eta": float(metadata["ssvi_fit"].eta),
        "ssvi_gamma": float(metadata["ssvi_fit"].gamma),
        "unit_square_cell_area": float(scaling["cell_area"]),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSingle-noise comparison:")
    print(comparison.to_string(index=False))
    print("\nFeature recovery:")
    print(feature_metrics.to_string(index=False))
    print("\nRepeated-noise summary:")
    print(repeated_summary.to_string(index=False))
    print("\nSaved outputs to:", output_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--mode", choices=["full", "quick"], default="full")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args.project_root.resolve(), args.mode, args.overwrite)
