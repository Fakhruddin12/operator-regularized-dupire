"""Stage 13B: nonlinear operator fit and held-out comparison with SSVI."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")



def _add_project_root_to_path(project_root: Path) -> None:
    root_text = str(project_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix


SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_add_project_root_to_path(SCRIPT_PROJECT_ROOT)

from src.evaluation.market_comparison import (
    market_model_metrics,
    market_prediction_table,
)
from src.evaluation.market_ssvi import (
    SSVIFit,
    evaluate_market_ssvi_local_volatility,
    evaluate_market_ssvi_quotes,
)
from src.evaluation.tuning import linearized_lambda_sweep
from src.inverse.market_inverse import (
    price_market_quotes_from_log_variance,
    run_market_gauss_newton_calibration,
)
from src.inverse.parameterization import local_volatility_from_log_variance


def main(project_root: Path) -> None:
    _add_project_root_to_path(project_root)
    output_directory = project_root / "outputs" / "stage_13"
    checkpoint_directory = output_directory / "checkpoints"

    checkpoint = np.load(checkpoint_directory / "tuning_checkpoint.npz")
    market_panel = pd.read_csv(
        checkpoint_directory / "market_panel.csv",
        parse_dates=["quote_date", "expiration"],
    )
    training_data = pd.read_csv(
        checkpoint_directory / "outer_training_with_inner_split.csv",
        parse_dates=["quote_date", "expiration"],
    )
    expiry_summary = pd.read_csv(
        checkpoint_directory / "selected_expiries.csv",
        parse_dates=["expiration"],
    )
    lambda_validation = pd.read_csv(
        checkpoint_directory / "lambda_inner_validation.csv"
    )

    calibration_T = checkpoint["calibration_maturities"]
    calibration_x = checkpoint["calibration_log_moneyness"]
    reference_log_variance = checkpoint["reference_log_variance"]
    regularization_matrix = csc_matrix(
        checkpoint["regularization_matrix"]
    )
    reference_prices = checkpoint["reference_prices"]
    reference_jacobian = checkpoint["jacobian"]
    selected_lambda = float(checkpoint["selected_lambda"])

    training_weights = (
        1.0
        / training_data["noise_standard_deviation"].to_numpy(dtype=float)
    )
    training_residual = (
        training_data["observed_call_price"].to_numpy(dtype=float)
        - reference_prices
    )
    _, corrections = linearized_lambda_sweep(
        jacobian=reference_jacobian,
        residual=training_residual,
        weights=training_weights,
        regularization_matrix=regularization_matrix,
        lambda_values=[selected_lambda],
    )
    initial_log_variance = (
        reference_log_variance
        + corrections[selected_lambda].reshape(
            reference_log_variance.shape,
            order="C",
        )
    )

    operator_result = run_market_gauss_newton_calibration(
        reference_log_variance=reference_log_variance,
        initial_log_variance=initial_log_variance,
        calibration_maturities=calibration_T,
        calibration_log_moneyness=calibration_x,
        quote_data=training_data,
        regularization_matrix=regularization_matrix,
        regularization_strength=selected_lambda,
        finite_difference_step=1e-3,
        jacobian_scheme="forward",
        pde_x_min=-1.0,
        pde_x_max=1.0,
        number_of_pde_x_points=151,
        number_of_time_steps=100,
        maximum_iterations=3,
        maximum_absolute_step=0.5,
        verbose=True,
    )
    operator_log_variance = operator_result["estimated_log_variance"]
    operator_volatility = local_volatility_from_log_variance(
        operator_log_variance
    )
    operator_prices = price_market_quotes_from_log_variance(
        log_variance_surface=operator_log_variance,
        calibration_maturities=calibration_T,
        calibration_log_moneyness=calibration_x,
        quote_data=market_panel,
        pde_x_min=-1.0,
        pde_x_max=1.0,
        number_of_pde_x_points=151,
        number_of_time_steps=100,
    )

    ssvi_values = json.loads(
        (checkpoint_directory / "ssvi_parameters.json").read_text(
            encoding="utf-8"
        )
    )
    ssvi_fit = SSVIFit(
        maturities=np.asarray(ssvi_values["maturities"], dtype=float),
        theta=np.asarray(ssvi_values["theta"], dtype=float),
        rho=float(ssvi_values["rho"]),
        eta=float(ssvi_values["eta"]),
        gamma=float(ssvi_values["gamma"]),
        objective=float(ssvi_values["objective"]),
        success=bool(ssvi_values["success"]),
        message=str(ssvi_values["message"]),
        number_of_usable_quotes=int(
            ssvi_values["number_of_usable_quotes"]
        ),
        fit_seconds=float(ssvi_values["fit_seconds"]),
    )
    ssvi_predictions = evaluate_market_ssvi_quotes(ssvi_fit, market_panel)
    ssvi_prices = ssvi_predictions["ssvi_call_price"].to_numpy(dtype=float)

    prediction_table = market_prediction_table(
        market_panel,
        predictions={
            "ssvi": ssvi_prices,
            "operator": operator_prices,
        },
    )
    metrics = market_model_metrics(
        prediction_table,
        methods=["ssvi", "operator"],
    )

    ssvi_local_table = evaluate_market_ssvi_local_volatility(
        ssvi_fit,
        calibration_T,
        calibration_x,
    )
    ssvi_local_volatility = ssvi_local_table[
        "local_volatility"
    ].to_numpy().reshape(calibration_T.size, calibration_x.size)

    output_directory.mkdir(parents=True, exist_ok=True)
    market_panel.to_csv(output_directory / "real_market_panel.csv", index=False)
    expiry_summary.to_csv(output_directory / "selected_expiries.csv", index=False)
    lambda_validation.to_csv(
        output_directory / "lambda_inner_validation.csv",
        index=False,
    )
    metrics.to_csv(
        output_directory / "held_out_model_metrics.csv",
        index=False,
    )
    prediction_table.to_csv(
        output_directory / "real_market_predictions.csv",
        index=False,
    )
    operator_result["history"].to_csv(
        output_directory / "operator_history.csv",
        index=False,
    )
    np.savez_compressed(
        output_directory / "local_volatility_surfaces.npz",
        calibration_maturities=calibration_T,
        calibration_log_moneyness=calibration_x,
        ssvi_local_volatility=ssvi_local_volatility,
        operator_local_volatility=operator_volatility,
    )
    np.savez_compressed(
        output_directory / "operator_calibration.npz",
        reference_log_variance=reference_log_variance,
        initial_log_variance=initial_log_variance,
        estimated_log_variance=operator_log_variance,
        selected_lambda=selected_lambda,
        regularization_matrix=regularization_matrix.toarray(),
        reference_jacobian=reference_jacobian,
        fitted_training_prices=operator_result["fitted_prices"],
    )
    (output_directory / "ssvi_parameters.json").write_text(
        json.dumps(ssvi_values, indent=2),
        encoding="utf-8",
    )

    test_metrics = metrics[metrics["split"] == "test"].set_index("method")
    print(
        "SSVI held-out weighted price RMSE: "
        f"{test_metrics.loc['ssvi', 'weighted_price_rmse']:.6f}"
    )
    print(
        "Operator held-out weighted price RMSE: "
        f"{test_metrics.loc['operator', 'weighted_price_rmse']:.6f}"
    )
    print(
        "SSVI held-out implied-volatility RMSE: "
        f"{test_metrics.loc['ssvi', 'implied_volatility_rmse']:.6f}"
    )
    print(
        "Operator held-out implied-volatility RMSE: "
        f"{test_metrics.loc['operator', 'implied_volatility_rmse']:.6f}"
    )
    print(
        "Operator local-volatility range: "
        f"{operator_volatility.min():.6f} to {operator_volatility.max():.6f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    arguments = parser.parse_args()
    main(arguments.project_root.resolve())
