"""Tests for Stage 13 real-market calibration utilities."""

import numpy as np
import pandas as pd

from src.evaluation.market_comparison import (
    market_model_metrics,
    select_market_panel,
)
from src.evaluation.market_ssvi import (
    implied_volatility_from_normalized_call,
)
from src.pricing.forward_normalized_pde import (
    interpolate_normalized_call_prices,
    normalized_black_call_price,
    solve_forward_normalized_dupire,
)


def test_normalized_pde_recovers_black_constant_volatility() -> None:
    volatility = 0.20
    x_grid, time_grid, surface = solve_forward_normalized_dupire(
        local_volatility=volatility,
        max_maturity=1.0,
        x_min=-1.0,
        x_max=1.0,
        number_of_x_points=301,
        number_of_time_steps=220,
        theta=0.5,
    )

    query_x = np.array([-0.20, 0.0, 0.20])
    query_T = np.array([0.25, 0.50, 1.00])
    pde_prices = interpolate_normalized_call_prices(
        x_grid,
        time_grid,
        surface,
        query_x,
        query_T,
    )
    black_prices = normalized_black_call_price(
        query_x,
        query_T,
        volatility,
    )

    assert np.allclose(
        pde_prices,
        black_prices,
        atol=2.5e-3,
    )


def test_normalized_implied_volatility_round_trip() -> None:
    x = -0.10
    maturity = 0.75
    volatility = 0.27
    price = normalized_black_call_price(
        x,
        maturity,
        volatility,
    )

    recovered = implied_volatility_from_normalized_call(
        normalized_call_price=price,
        log_moneyness=x,
        maturity=maturity,
    )

    assert np.isclose(recovered, volatility, atol=1e-10)


def test_market_panel_selection_is_deterministic() -> None:
    rows = []
    maturities = np.linspace(0.1, 1.0, 10)
    for expiry_index, maturity in enumerate(maturities):
        expiration = pd.Timestamp("2025-01-01") + pd.Timedelta(
            days=int(round(365.25 * maturity))
        )
        for quote_index, x in enumerate(np.linspace(-0.2, 0.2, 10)):
            rows.append(
                {
                    "expiration": expiration,
                    "maturity": maturity,
                    "log_moneyness": x,
                    "strike": 100.0 * np.exp(x),
                    "is_train": quote_index < 8,
                    "is_test": quote_index >= 8,
                }
            )
    data = pd.DataFrame(rows)

    first, first_summary = select_market_panel(
        data,
        number_of_expiries=5,
        minimum_training_quotes_per_expiry=8,
        minimum_test_quotes_per_expiry=2,
    )
    second, second_summary = select_market_panel(
        data,
        number_of_expiries=5,
        minimum_training_quotes_per_expiry=8,
        minimum_test_quotes_per_expiry=2,
    )

    assert first["expiration"].nunique() == 5
    assert first_summary["expiration"].equals(second_summary["expiration"])
    assert first.equals(second)


def test_market_metrics_reward_better_predictions() -> None:
    data = pd.DataFrame(
        {
            "observed_call_price": [10.0, 8.0, 6.0, 4.0],
            "noise_standard_deviation": [0.5, 0.5, 0.5, 0.5],
            "call_bid": [9.5, 7.5, 5.5, 3.5],
            "call_ask": [10.5, 8.5, 6.5, 4.5],
            "market_implied_volatility": [0.2, 0.21, 0.22, 0.23],
            "is_train": [True, True, False, False],
            "is_test": [False, False, True, True],
            "good_price": [10.0, 8.1, 6.1, 4.0],
            "bad_price": [12.0, 10.0, 8.0, 6.0],
            "good_implied_volatility": [0.2, 0.21, 0.221, 0.229],
            "bad_implied_volatility": [0.3, 0.3, 0.3, 0.3],
        }
    )

    metrics = market_model_metrics(
        data,
        methods=["good", "bad"],
    )
    test_metrics = metrics[metrics["split"] == "test"].set_index("method")

    assert (
        test_metrics.loc["good", "weighted_price_rmse"]
        < test_metrics.loc["bad", "weighted_price_rmse"]
    )
    assert (
        test_metrics.loc["good", "implied_volatility_rmse"]
        < test_metrics.loc["bad", "implied_volatility_rmse"]
    )

from src.evaluation.market_tuning import (
    add_inner_validation_split,
    select_lambda_by_validation,
)


def test_inner_validation_split_and_selection() -> None:
    rows = []
    for expiry in ["2025-06-01", "2025-12-01"]:
        for index, x in enumerate(np.linspace(-0.2, 0.2, 10)):
            rows.append(
                {
                    "expiration": pd.Timestamp(expiry),
                    "log_moneyness": x,
                    "quote_index": index,
                }
            )
    data = pd.DataFrame(rows)
    split = add_inner_validation_split(
        data,
        every_nth_quote=5,
        offset=2,
    )

    assert split["is_inner_validation"].sum() == 4
    assert split["is_inner_fit"].sum() == 16

    validation_results = pd.DataFrame(
        {
            "lambda": [1.0, 10.0, 100.0],
            "validation_weighted_price_rmse": [3.0, 1.0, np.inf],
            "finite_candidate": [True, True, False],
        }
    )
    assert select_lambda_by_validation(validation_results) == 10.0
