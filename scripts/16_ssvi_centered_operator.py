"""Stage 16: nested test of an SSVI-centred operator correction.

Hypothesis: the previous constant-volatility reference asks the regularizer to
learn the entire smile and term structure from noisy prices. Instead, use a
training-only arbitrage-aware SSVI surface as the prior centre, then let the
operator solve for a smooth residual correction. Lambda selection is nested
inside the existing outer training split; the outer quote holdout is untouched.

This remains a single-date cross-sectional experiment, not a trading test.
Results are written to a new directory and do not replace earlier outputs.
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
from src.evaluation.market_ssvi import (
    evaluate_market_ssvi_local_volatility,
    evaluate_market_ssvi_quotes,
    fit_market_ssvi_surface,
)
from src.evaluation.market_tuning import (
    add_inner_validation_split,
    market_lambda_validation_sweep,
    select_lambda_by_validation,
)
from src.evaluation.tuning import linearized_lambda_sweep
from src.inverse.market_inverse import (
    finite_difference_market_jacobian,
    price_market_quotes_from_log_variance,
    run_market_gauss_newton_calibration,
)
from src.inverse.parameterization import local_volatility_from_log_variance
from src.regularization.potentials import (
    combine_potentials,
    confidence_potential,
    quote_confidence_surface,
    wing_potential,
)
from src.regularization.scaling import build_nondimensional_regularization_matrix


def reference_from_ssvi(fit, maturities: np.ndarray, x: np.ndarray) -> np.ndarray:
    table = evaluate_market_ssvi_local_volatility(fit, maturities, x)
    sigma = table["local_volatility"].to_numpy(dtype=float).reshape(
        maturities.size, x.size, order="C"
    )
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("SSVI-Dupire produced an invalid reference surface.")
    return np.log(sigma**2)


def make_regularizer(
    maturities: np.ndarray,
    x: np.ndarray,
    quotes: pd.DataFrame,
) -> csc_matrix:
    wing = wing_potential(
        log_moneyness=x, maturities=maturities,
        strength=0.20, power=2.0, start=0.18,
    )
    confidence = quote_confidence_surface(
        log_moneyness=x, maturities=maturities,
        quote_log_moneyness=quotes["log_moneyness"].to_numpy(dtype=float),
        quote_maturities=quotes["maturity"].to_numpy(dtype=float),
        log_moneyness_bandwidth=0.06, maturity_bandwidth=0.12,
    )
    matrix, _, _, _ = build_nondimensional_regularization_matrix(
        maturities=maturities,
        log_moneyness=x,
        alpha_x=0.005,
        alpha_T=0.002,
        beta=1e-4,
        potential=combine_potentials(
            wing, confidence_potential(confidence, strength=0.20)
        ),
    )
    return matrix


def run(project_root: Path, max_iterations: int, output_name: str) -> None:
    stage13 = project_root / "outputs" / "stage_13"
    checkpoint = np.load(stage13 / "checkpoints" / "tuning_checkpoint.npz")
    panel = pd.read_csv(
        stage13 / "checkpoints" / "market_panel.csv",
        parse_dates=["quote_date", "expiration"],
    )
    outer_train = panel[panel["is_train"]].reset_index(drop=True)
    outer_test = panel[panel["is_test"]].reset_index(drop=True)
    inner = add_inner_validation_split(outer_train, every_nth_quote=5, offset=2)
    inner_fit = inner[inner["is_inner_fit"]].reset_index(drop=True)

    T = checkpoint["calibration_maturities"]
    x = checkpoint["calibration_log_moneyness"]
    output = project_root / "outputs" / output_name
    output.mkdir(parents=True, exist_ok=True)

    # Nested tuning: all reference parameters and quote-density information are
    # estimated using only inner-fit quotes, never the inner validation subset.
    inner_ssvi, _ = fit_market_ssvi_surface(
        inner_fit, maximum_function_evaluations=5000
    )
    inner_reference = reference_from_ssvi(inner_ssvi, T, x)
    inner_regularizer = make_regularizer(T, x, inner_fit)
    inner_prices, inner_jacobian = finite_difference_market_jacobian(
        reference_log_variance=inner_reference,
        calibration_maturities=T,
        calibration_log_moneyness=x,
        quote_data=inner,
        finite_difference_step=1e-3,
        scheme="forward",
        pde_x_min=-1.0,
        pde_x_max=1.0,
        number_of_pde_x_points=151,
        number_of_time_steps=100,
        verbose=True,
    )
    validation, _ = market_lambda_validation_sweep(
        reference_log_variance=inner_reference,
        reference_prices=inner_prices,
        jacobian=inner_jacobian,
        training_quotes_with_inner_split=inner,
        regularization_matrix=inner_regularizer,
        lambda_values=np.logspace(2, 8, 13),
        calibration_maturities=T,
        calibration_log_moneyness=x,
        pde_x_min=-1.0,
        pde_x_max=1.0,
        number_of_pde_x_points=151,
        number_of_time_steps=100,
    )
    selected_lambda = select_lambda_by_validation(validation)
    validation.to_csv(output / "lambda_inner_validation.csv", index=False)

    # Refit the prior and quote-confidence potential on all outer training data.
    final_ssvi, _ = fit_market_ssvi_surface(
        outer_train, maximum_function_evaluations=5000
    )
    reference = reference_from_ssvi(final_ssvi, T, x)
    regularizer = make_regularizer(T, x, outer_train)
    base_prices, jacobian = finite_difference_market_jacobian(
        reference_log_variance=reference,
        calibration_maturities=T,
        calibration_log_moneyness=x,
        quote_data=outer_train,
        finite_difference_step=1e-3,
        scheme="forward",
        pde_x_min=-1.0,
        pde_x_max=1.0,
        number_of_pde_x_points=151,
        number_of_time_steps=100,
        verbose=True,
    )
    _, corrections = linearized_lambda_sweep(
        jacobian=jacobian,
        residual=(outer_train["observed_call_price"].to_numpy(dtype=float)
                  - base_prices),
        weights=1.0 / outer_train["noise_standard_deviation"].to_numpy(dtype=float),
        regularization_matrix=regularizer,
        lambda_values=[selected_lambda],
    )
    initial = reference + corrections[selected_lambda].reshape(reference.shape, order="C")
    fit = run_market_gauss_newton_calibration(
        reference_log_variance=reference,
        initial_log_variance=initial,
        calibration_maturities=T,
        calibration_log_moneyness=x,
        quote_data=outer_train,
        regularization_matrix=regularizer,
        regularization_strength=selected_lambda,
        finite_difference_step=1e-3,
        jacobian_scheme="forward",
        pde_x_min=-1.0,
        pde_x_max=1.0,
        number_of_pde_x_points=151,
        number_of_time_steps=100,
        maximum_iterations=max_iterations,
        maximum_absolute_step=0.5,
        relative_objective_tolerance=1e-5,
        relative_step_tolerance=1e-4,
        verbose=True,
    )
    surface = np.asarray(fit["estimated_log_variance"], dtype=float)
    fitted_prices = price_market_quotes_from_log_variance(
        surface, T, x, panel, -1.0, 1.0, 151, 100
    )

    # The baseline SSVI has the same outer training data as this residual fit.
    ssvi_prices = evaluate_market_ssvi_quotes(final_ssvi, panel)[
        "ssvi_call_price"
    ].to_numpy(dtype=float)
    prior_pde_prices = price_market_quotes_from_log_variance(
        reference, T, x, panel, -1.0, 1.0, 151, 100
    )
    predictions = market_prediction_table(panel, {
        "ssvi": ssvi_prices,
        "ssvi_prior_pde": prior_pde_prices,
        "operator_ssvi_residual": fitted_prices,
    })
    metrics = market_model_metrics(
        predictions,
        methods=["ssvi", "ssvi_prior_pde", "operator_ssvi_residual"],
    )
    predictions.to_csv(output / "predictions.csv", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    fit["history"].to_csv(output / "history.csv", index=False)
    np.savez_compressed(
        output / "surfaces.npz",
        maturities=T,
        log_moneyness=x,
        ssvi_reference_log_variance=reference,
        initial_log_variance=initial,
        fitted_log_variance=surface,
        ssvi_reference_local_volatility=local_volatility_from_log_variance(reference),
        residual_operator_local_volatility=local_volatility_from_log_variance(surface),
        ssvi_rho=final_ssvi.rho,
        ssvi_eta=final_ssvi.eta,
        ssvi_gamma=final_ssvi.gamma,
        ssvi_theta=final_ssvi.theta,
    )
    summary = {
        "experiment": "SSVI-centred operator residual correction",
        "outer_train_quotes": int(len(outer_train)),
        "outer_test_quotes": int(len(outer_test)),
        "inner_fit_quotes": int(len(inner_fit)),
        "inner_validation_quotes": int(inner["is_inner_validation"].sum()),
        "selected_lambda": float(selected_lambda),
        "maximum_iterations_requested": int(max_iterations),
        "stop_reason": fit["stop_reason"],
        "operator_local_volatility_range": [
            float(np.min(local_volatility_from_log_variance(surface))),
            float(np.max(local_volatility_from_log_variance(surface))),
        ],
        "test_weighted_price_rmse": {
            row["method"]: float(row["weighted_price_rmse"])
            for row in metrics[metrics["split"] == "test"].to_dict("records")
        },
        "interpretation": (
            "One-date nested cross-sectional model comparison only. The outer test was not used for tuning. "
            "The candidate is an SSVI prior plus a regularized correction; any gain over SSVI must be "
            "confirmed on chronological market dates and against costs before use beyond calibration."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--maximum-iterations", type=int, default=12)
    parser.add_argument("--output-name", default="stage_16_ssvi_residual")
    args = parser.parse_args()
    run(args.project_root.resolve(), args.maximum_iterations, args.output_name)


if __name__ == "__main__":
    main()
