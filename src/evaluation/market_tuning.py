"""Nested validation for the real-market operator regularization strength."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.evaluation.tuning import linearized_lambda_sweep
from src.inverse.market_inverse import price_market_quotes_from_log_variance
from src.inverse.parameterization import local_volatility_from_log_variance


def add_inner_validation_split(
    training_quotes: pd.DataFrame,
    every_nth_quote: int = 5,
    offset: int = 2,
) -> pd.DataFrame:
    """Create an expiry-stratified inner validation set inside outer training.

    Quotes are ordered by log-moneyness within each expiry. Every ``n``-th quote
    is assigned to the inner validation set. The original held-out test quotes
    are never used for selecting lambda.
    """
    required = {
        "expiration",
        "log_moneyness",
    }
    missing = required.difference(training_quotes.columns)
    if missing:
        raise ValueError(
            f"training_quotes is missing columns: {sorted(missing)}"
        )
    if every_nth_quote < 3:
        raise ValueError("every_nth_quote must be at least 3.")
    if not 0 <= offset < every_nth_quote:
        raise ValueError("offset must lie between zero and every_nth_quote-1.")

    data = training_quotes.copy().reset_index(drop=True)
    data["is_inner_validation"] = False

    for _, group in data.groupby("expiration"):
        ordered_indices = (
            group.sort_values("log_moneyness").index.to_numpy()
        )
        positions = np.arange(offset, ordered_indices.size, every_nth_quote)
        data.loc[
            ordered_indices[positions],
            "is_inner_validation",
        ] = True

    data["is_inner_fit"] = ~data["is_inner_validation"]

    if not data["is_inner_fit"].any():
        raise ValueError("inner fit set is empty.")
    if not data["is_inner_validation"].any():
        raise ValueError("inner validation set is empty.")

    return data


def market_lambda_validation_sweep(
    reference_log_variance: np.ndarray,
    reference_prices: np.ndarray,
    jacobian: np.ndarray,
    training_quotes_with_inner_split: pd.DataFrame,
    regularization_matrix,
    lambda_values: Iterable[float],
    calibration_maturities: np.ndarray,
    calibration_log_moneyness: np.ndarray,
    pde_x_min: float = -1.0,
    pde_x_max: float = 1.0,
    number_of_pde_x_points: int = 201,
    number_of_time_steps: int = 150,
    theta: float = 0.5,
) -> tuple[pd.DataFrame, dict[float, np.ndarray]]:
    """Select lambda using nonlinear inner-validation quote repricing.

    Candidate corrections are estimated from the inner-fit quotes only. Each
    resulting surface is then repriced with the full nonlinear PDE on the
    inner-validation quotes. The outer held-out test set remains untouched.
    """
    data = training_quotes_with_inner_split.reset_index(drop=True)
    required = {
        "observed_call_price",
        "noise_standard_deviation",
        "is_inner_fit",
        "is_inner_validation",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            f"training data is missing columns: {sorted(missing)}"
        )

    base_prices = np.asarray(reference_prices, dtype=float).reshape(-1)
    J = np.asarray(jacobian, dtype=float)
    reference = np.asarray(reference_log_variance, dtype=float)

    if base_prices.size != len(data):
        raise ValueError("reference_prices must contain one value per quote.")
    if J.shape != (len(data), reference.size):
        raise ValueError("jacobian has the wrong shape.")

    fit_mask = data["is_inner_fit"].to_numpy(dtype=bool)
    validation_mask = data["is_inner_validation"].to_numpy(dtype=bool)
    observed = data["observed_call_price"].to_numpy(dtype=float)
    noise = data["noise_standard_deviation"].to_numpy(dtype=float)
    weights = 1.0 / noise
    residual = observed - base_prices

    linear_results, corrections = linearized_lambda_sweep(
        jacobian=J[fit_mask],
        residual=residual[fit_mask],
        weights=weights[fit_mask],
        regularization_matrix=regularization_matrix,
        lambda_values=lambda_values,
    )

    validation_quotes = data.loc[validation_mask].reset_index(drop=True)
    rows = []

    for lambda_value in linear_results["lambda"]:
        lambda_float = float(lambda_value)
        correction = corrections[lambda_float]
        candidate_surface = (
            reference
            + correction.reshape(reference.shape, order="C")
        )
        volatility = local_volatility_from_log_variance(candidate_surface)

        failure_message = ""
        try:
            predicted = price_market_quotes_from_log_variance(
                log_variance_surface=candidate_surface,
                calibration_maturities=calibration_maturities,
                calibration_log_moneyness=calibration_log_moneyness,
                quote_data=validation_quotes,
                pde_x_min=pde_x_min,
                pde_x_max=pde_x_max,
                number_of_pde_x_points=number_of_pde_x_points,
                number_of_time_steps=number_of_time_steps,
                theta=theta,
            )
            validation_observed = validation_quotes[
                "observed_call_price"
            ].to_numpy(dtype=float)
            validation_noise = validation_quotes[
                "noise_standard_deviation"
            ].to_numpy(dtype=float)
            residual_values = predicted - validation_observed
            standardized = residual_values / validation_noise
            validation_weighted_rmse = float(
                np.sqrt(np.mean(standardized**2))
            )
            validation_price_rmse = float(
                np.sqrt(np.mean(residual_values**2))
            )
            finite_candidate = bool(
                np.all(np.isfinite(predicted))
                and np.all(np.isfinite(volatility))
            )
        except (ValueError, FloatingPointError, RuntimeError) as error:
            validation_weighted_rmse = np.inf
            validation_price_rmse = np.inf
            finite_candidate = False
            failure_message = str(error)

        matching_linear_row = linear_results[
            np.isclose(linear_results["lambda"], lambda_float)
        ].iloc[0]

        rows.append(
            {
                "lambda": lambda_float,
                "validation_weighted_price_rmse": validation_weighted_rmse,
                "validation_price_rmse": validation_price_rmse,
                "linearized_inner_fit_weighted_rmse": float(
                    matching_linear_row["weighted_rmse"]
                ),
                "effective_degrees_of_freedom": float(
                    matching_linear_row["effective_degrees_of_freedom"]
                ),
                "gcv": float(matching_linear_row["gcv"]),
                "minimum_local_volatility": float(np.nanmin(volatility)),
                "maximum_local_volatility": float(np.nanmax(volatility)),
                "correction_norm": float(
                    matching_linear_row["correction_norm"]
                ),
                "finite_candidate": finite_candidate,
                "failure_message": failure_message,
            }
        )

    return (
        pd.DataFrame(rows).sort_values("lambda").reset_index(drop=True),
        corrections,
    )


def select_lambda_by_validation(
    validation_results: pd.DataFrame,
) -> float:
    """Return the finite lambda with the lowest inner-validation weighted RMSE."""
    required = {
        "lambda",
        "validation_weighted_price_rmse",
        "finite_candidate",
    }
    missing = required.difference(validation_results.columns)
    if missing:
        raise ValueError(
            f"validation_results is missing columns: {sorted(missing)}"
        )

    eligible = validation_results[
        validation_results["finite_candidate"]
        & np.isfinite(
            validation_results["validation_weighted_price_rmse"]
        )
    ]
    if eligible.empty:
        raise ValueError("No finite lambda candidate is available.")

    best_index = eligible["validation_weighted_price_rmse"].idxmin()
    return float(eligible.loc[best_index, "lambda"])
