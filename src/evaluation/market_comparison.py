"""Panel selection and held-out diagnostics for real option data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.market_ssvi import (
    forward_implied_variance_quotes,
)


def select_market_panel(
    prepared_quotes: pd.DataFrame,
    minimum_maturity: float = 0.08,
    maximum_maturity: float = 1.10,
    maximum_absolute_log_moneyness: float = 0.25,
    number_of_expiries: int = 8,
    minimum_training_quotes_per_expiry: int = 40,
    minimum_test_quotes_per_expiry: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select a deterministic, maturity-spanning real-data panel."""
    required = {
        "expiration",
        "maturity",
        "log_moneyness",
        "is_train",
        "is_test",
    }
    missing = required.difference(prepared_quotes.columns)
    if missing:
        raise ValueError(
            f"prepared_quotes is missing columns: {sorted(missing)}"
        )
    if number_of_expiries < 2:
        raise ValueError("number_of_expiries must be at least two.")

    data = prepared_quotes.copy()
    data["expiration"] = pd.to_datetime(data["expiration"])
    data = data[
        (data["maturity"] >= minimum_maturity)
        & (data["maturity"] <= maximum_maturity)
        & (
            np.abs(data["log_moneyness"])
            <= maximum_absolute_log_moneyness
        )
    ].copy()

    expiry_summary = (
        data.groupby(["expiration", "maturity"], as_index=False)
        .agg(
            number_of_quotes=("strike", "size"),
            number_of_training_quotes=("is_train", "sum"),
            number_of_test_quotes=("is_test", "sum"),
            minimum_log_moneyness=("log_moneyness", "min"),
            maximum_log_moneyness=("log_moneyness", "max"),
        )
    )
    eligible = expiry_summary[
        (
            expiry_summary["number_of_training_quotes"]
            >= minimum_training_quotes_per_expiry
        )
        & (
            expiry_summary["number_of_test_quotes"]
            >= minimum_test_quotes_per_expiry
        )
    ].sort_values("maturity").reset_index(drop=True)

    if len(eligible) < number_of_expiries:
        raise ValueError(
            "Too few expiries satisfy the panel-quality requirements."
        )

    selected_positions = np.unique(
        np.round(
            np.linspace(
                0,
                len(eligible) - 1,
                number_of_expiries,
            )
        ).astype(int)
    )
    if selected_positions.size != number_of_expiries:
        raise RuntimeError(
            "Could not choose the requested number of distinct expiries."
        )

    selected_summary = eligible.iloc[selected_positions].copy()
    selected_expirations = set(selected_summary["expiration"])
    panel = data[
        data["expiration"].isin(selected_expirations)
    ].sort_values(
        ["maturity", "log_moneyness"]
    ).reset_index(drop=True)

    selected_summary = selected_summary.reset_index(drop=True)
    return panel, selected_summary


def estimate_reference_volatility(
    training_quotes: pd.DataFrame,
    atm_log_moneyness_limit: float = 0.03,
) -> float:
    """Estimate a neutral constant reference volatility from near-ATM quotes."""
    implied = forward_implied_variance_quotes(training_quotes)
    usable = implied[
        implied["usable_for_ssvi"]
        & (
            np.abs(implied["log_moneyness"])
            <= atm_log_moneyness_limit
        )
    ]

    if usable.empty:
        usable = implied[implied["usable_for_ssvi"]]
    if usable.empty:
        raise ValueError("No usable implied volatilities for the reference.")

    return float(np.median(usable["implied_volatility"]))


def implied_volatility_from_model_prices(
    quote_data: pd.DataFrame,
    predicted_prices: np.ndarray,
) -> np.ndarray:
    """Convert model currency prices to forward implied volatilities."""
    data = quote_data.copy()
    predicted = np.asarray(predicted_prices, dtype=float).reshape(-1)
    if predicted.size != len(data):
        raise ValueError("predicted_prices must contain one value per quote.")

    data["predicted_model_price"] = predicted
    implied = forward_implied_variance_quotes(
        data,
        price_column="predicted_model_price",
        noise_column="noise_standard_deviation",
        minimum_implied_volatility=0.0,
        maximum_total_variance_standard_deviation=np.inf,
    )
    return implied["implied_volatility"].to_numpy(dtype=float)


def market_prediction_table(
    quote_data: pd.DataFrame,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Combine quote data with model price and implied-volatility predictions."""
    output = quote_data.copy().reset_index(drop=True)
    market_implied = forward_implied_variance_quotes(output)
    output["market_implied_volatility"] = market_implied[
        "implied_volatility"
    ].to_numpy(dtype=float)

    for method, prices in predictions.items():
        price_values = np.asarray(prices, dtype=float).reshape(-1)
        if price_values.size != len(output):
            raise ValueError(
                f"Prediction '{method}' does not contain one value per quote."
            )
        output[f"{method}_price"] = price_values
        output[f"{method}_implied_volatility"] = (
            implied_volatility_from_model_prices(output, price_values)
        )

    return output


def market_model_metrics(
    prediction_table: pd.DataFrame,
    methods: list[str],
) -> pd.DataFrame:
    """Calculate train and held-out pricing and implied-volatility metrics."""
    required = {
        "observed_call_price",
        "noise_standard_deviation",
        "call_bid",
        "call_ask",
        "market_implied_volatility",
        "is_train",
        "is_test",
    }
    missing = required.difference(prediction_table.columns)
    if missing:
        raise ValueError(
            f"prediction_table is missing columns: {sorted(missing)}"
        )

    rows = []
    split_masks = {
        "train": prediction_table["is_train"].to_numpy(dtype=bool),
        "test": prediction_table["is_test"].to_numpy(dtype=bool),
    }

    observed = prediction_table["observed_call_price"].to_numpy(dtype=float)
    noise = prediction_table["noise_standard_deviation"].to_numpy(dtype=float)
    market_iv = prediction_table["market_implied_volatility"].to_numpy(dtype=float)
    bid = prediction_table["call_bid"].to_numpy(dtype=float)
    ask = prediction_table["call_ask"].to_numpy(dtype=float)

    for method in methods:
        price_column = f"{method}_price"
        iv_column = f"{method}_implied_volatility"
        if price_column not in prediction_table or iv_column not in prediction_table:
            raise ValueError(f"Missing predictions for method '{method}'.")

        predicted = prediction_table[price_column].to_numpy(dtype=float)
        predicted_iv = prediction_table[iv_column].to_numpy(dtype=float)

        for split_name, split_mask in split_masks.items():
            finite = (
                split_mask
                & np.isfinite(predicted)
                & np.isfinite(observed)
                & np.isfinite(noise)
                & (noise > 0)
            )
            if not np.any(finite):
                continue

            residual = predicted[finite] - observed[finite]
            standardized = residual / noise[finite]
            iv_mask = finite & np.isfinite(predicted_iv) & np.isfinite(market_iv)
            iv_error = predicted_iv[iv_mask] - market_iv[iv_mask]

            inside_spread = (
                predicted[finite] >= bid[finite]
            ) & (
                predicted[finite] <= ask[finite]
            )

            rows.append(
                {
                    "method": method,
                    "split": split_name,
                    "number_of_quotes": int(np.sum(finite)),
                    "price_rmse": float(np.sqrt(np.mean(residual**2))),
                    "price_mae": float(np.mean(np.abs(residual))),
                    "weighted_price_rmse": float(
                        np.sqrt(np.mean(standardized**2))
                    ),
                    "mean_gaussian_nll": float(
                        np.mean(
                            0.5 * standardized**2
                            + np.log(noise[finite])
                            + 0.5 * np.log(2.0 * np.pi)
                        )
                    ),
                    "inside_bid_ask_fraction": float(np.mean(inside_spread)),
                    "implied_volatility_rmse": float(
                        np.sqrt(np.mean(iv_error**2))
                    ) if iv_error.size else np.nan,
                    "implied_volatility_mae": float(
                        np.mean(np.abs(iv_error))
                    ) if iv_error.size else np.nan,
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["split", "weighted_price_rmse"]
    ).reset_index(drop=True)
