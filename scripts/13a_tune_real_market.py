"""Stage 13A: select the real-data panel, fit SSVI, and tune lambda.

This script intentionally runs in its own process so the large finite-
difference Jacobian is released before the nonlinear calibration phase.
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



def _add_project_root_to_path(project_root: Path) -> None:
    root_text = str(project_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

import numpy as np
import pandas as pd


SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_add_project_root_to_path(SCRIPT_PROJECT_ROOT)

from src.evaluation.market_comparison import (
    estimate_reference_volatility,
    select_market_panel,
)
from src.evaluation.market_ssvi import fit_market_ssvi_surface
from src.evaluation.market_tuning import (
    add_inner_validation_split,
    market_lambda_validation_sweep,
    select_lambda_by_validation,
)
from src.inverse.market_inverse import finite_difference_market_jacobian
from src.inverse.parameterization import reference_log_variance_surface
from src.regularization.potentials import (
    combine_potentials,
    confidence_potential,
    quote_confidence_surface,
    wing_potential,
)
from src.regularization.scaling import (
    build_nondimensional_regularization_matrix,
)


def main(project_root: Path) -> None:
    _add_project_root_to_path(project_root)
    processed_directory = project_root / "data" / "market" / "processed"
    prepared_files = sorted(processed_directory.glob("*_prepared_quotes.csv"))
    if not prepared_files:
        raise FileNotFoundError(
            "No Stage 12 prepared quote file was found in data/market/processed."
        )

    output_directory = project_root / "outputs" / "stage_13"
    checkpoint_directory = output_directory / "checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    prepared_quotes = pd.read_csv(
        prepared_files[0],
        parse_dates=["quote_date", "expiration"],
    )
    market_panel, expiry_summary = select_market_panel(prepared_quotes)
    outer_training_quotes = market_panel[
        market_panel["is_train"]
    ].reset_index(drop=True)
    training_with_inner_split = add_inner_validation_split(
        outer_training_quotes,
        every_nth_quote=5,
        offset=2,
    )
    inner_fit_quotes = training_with_inner_split[
        training_with_inner_split["is_inner_fit"]
    ].reset_index(drop=True)

    ssvi_fit, _ = fit_market_ssvi_surface(
        outer_training_quotes,
        maximum_function_evaluations=5000,
    )

    calibration_T = np.linspace(
        market_panel["maturity"].min(),
        market_panel["maturity"].max(),
        7,
    )
    calibration_x = np.linspace(-0.25, 0.25, 13)

    reference_volatility = estimate_reference_volatility(inner_fit_quotes)
    reference_log_variance = reference_log_variance_surface(
        maturities=calibration_T,
        log_moneyness=calibration_x,
        reference_volatility=reference_volatility,
    )

    wing = wing_potential(
        log_moneyness=calibration_x,
        maturities=calibration_T,
        strength=0.20,
        power=2.0,
        start=0.18,
    )
    confidence = quote_confidence_surface(
        log_moneyness=calibration_x,
        maturities=calibration_T,
        quote_log_moneyness=inner_fit_quotes[
            "log_moneyness"
        ].to_numpy(),
        quote_maturities=inner_fit_quotes["maturity"].to_numpy(),
        log_moneyness_bandwidth=0.06,
        maturity_bandwidth=0.12,
    )
    regularization_matrix, _, _, _ = (
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

    reference_prices, jacobian = finite_difference_market_jacobian(
        reference_log_variance=reference_log_variance,
        calibration_maturities=calibration_T,
        calibration_log_moneyness=calibration_x,
        quote_data=training_with_inner_split,
        finite_difference_step=1e-3,
        scheme="forward",
        pde_x_min=-1.0,
        pde_x_max=1.0,
        number_of_pde_x_points=151,
        number_of_time_steps=100,
        verbose=True,
    )

    lambda_validation, _ = market_lambda_validation_sweep(
        reference_log_variance=reference_log_variance,
        reference_prices=reference_prices,
        jacobian=jacobian,
        training_quotes_with_inner_split=training_with_inner_split,
        regularization_matrix=regularization_matrix,
        lambda_values=np.logspace(2, 8, 13),
        calibration_maturities=calibration_T,
        calibration_log_moneyness=calibration_x,
        pde_x_min=-1.0,
        pde_x_max=1.0,
        number_of_pde_x_points=151,
        number_of_time_steps=100,
    )
    selected_lambda = select_lambda_by_validation(lambda_validation)

    market_panel.to_csv(
        checkpoint_directory / "market_panel.csv",
        index=False,
    )
    expiry_summary.to_csv(
        checkpoint_directory / "selected_expiries.csv",
        index=False,
    )
    training_with_inner_split.to_csv(
        checkpoint_directory / "outer_training_with_inner_split.csv",
        index=False,
    )
    lambda_validation.to_csv(
        checkpoint_directory / "lambda_inner_validation.csv",
        index=False,
    )
    np.savez_compressed(
        checkpoint_directory / "tuning_checkpoint.npz",
        calibration_maturities=calibration_T,
        calibration_log_moneyness=calibration_x,
        reference_log_variance=reference_log_variance,
        reference_volatility=reference_volatility,
        regularization_matrix=regularization_matrix.toarray(),
        reference_prices=reference_prices,
        jacobian=jacobian,
        selected_lambda=selected_lambda,
    )
    (checkpoint_directory / "ssvi_parameters.json").write_text(
        json.dumps(
            {
                "maturities": ssvi_fit.maturities.tolist(),
                "theta": ssvi_fit.theta.tolist(),
                "rho": ssvi_fit.rho,
                "eta": ssvi_fit.eta,
                "gamma": ssvi_fit.gamma,
                "objective": ssvi_fit.objective,
                "success": ssvi_fit.success,
                "message": ssvi_fit.message,
                "number_of_usable_quotes": ssvi_fit.number_of_usable_quotes,
                "fit_seconds": ssvi_fit.fit_seconds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Selected lambda: {selected_lambda}")
    print(f"Panel quotes: {len(market_panel)}")
    print(f"Outer training quotes: {len(outer_training_quotes)}")
    print(f"Outer test quotes: {int(market_panel['is_test'].sum())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    arguments = parser.parse_args()
    main(arguments.project_root.resolve())
