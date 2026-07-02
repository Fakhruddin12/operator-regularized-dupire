"""Tests for Stage 12 real-option data preparation."""

import numpy as np
import pandas as pd

from src.data.cboe_market_data import (
    PreparationSettings,
    infer_forward_curve_from_parity,
    prepare_calibration_quotes,
    standardise_cboe_columns,
    strike_arbitrage_diagnostics,
)


def _parity_consistent_fixture() -> pd.DataFrame:
    quote_date = pd.Timestamp("2025-01-02")
    expirations = [
        pd.Timestamp("2025-07-02"),
        pd.Timestamp("2026-01-02"),
    ]
    forwards = [103.0, 106.0]
    discounts = [0.98, 0.95]
    strikes = np.arange(80.0, 126.0, 5.0)

    rows = []

    for expiration, forward, discount in zip(
        expirations,
        forwards,
        discounts,
    ):
        maturity = (
            expiration - quote_date
        ).days / 365.25

        for strike in strikes:
            # A smooth positive put value; call then follows exact parity.
            put_mid = (
                1.0
                + 0.003
                * (strike - forward) ** 2
            )
            call_mid = (
                put_mid
                + discount
                * (forward - strike)
            )

            # Ensure all quotes are positive for the fixture.
            shift = max(
                0.0,
                0.2 - min(
                    call_mid,
                    put_mid,
                ),
            )
            call_mid += shift
            put_mid += shift

            for option_type, mid in [
                ("C", call_mid),
                ("P", put_mid),
            ]:
                rows.append(
                    {
                        "Underlying Symbol": "TEST",
                        "Quote Date": quote_date,
                        "Expiration": expiration,
                        "Strike": strike,
                        "Option Type": option_type,
                        "Bid 1545": mid - 0.05,
                        "Ask 1545": mid + 0.05,
                        "Trade Volume": 100,
                        "Open Interest": 500,
                    }
                )

    return pd.DataFrame(rows)


def test_column_standardisation() -> None:
    standardised = standardise_cboe_columns(
        _parity_consistent_fixture()
    )

    assert {
        "underlying_symbol",
        "quote_date",
        "expiration",
        "strike",
        "option_type",
        "bid_1545",
        "ask_1545",
    }.issubset(standardised.columns)


def test_put_call_parity_recovers_forward_and_discount() -> None:
    standardised = standardise_cboe_columns(
        _parity_consistent_fixture()
    )
    standardised["bid"] = standardised[
        "bid_1545"
    ]
    standardised["ask"] = standardised[
        "ask_1545"
    ]
    standardised["mid"] = 0.5 * (
        standardised["bid"]
        + standardised["ask"]
    )
    standardised["spread"] = (
        standardised["ask"]
        - standardised["bid"]
    )
    standardised["maturity"] = (
        standardised["expiration"]
        - standardised["quote_date"]
    ).dt.days / 365.25

    curve = infer_forward_curve_from_parity(
        standardised,
        minimum_pairs=5,
    )

    assert np.allclose(
        curve["forward"],
        [103.0, 106.0],
        atol=1e-6,
    )
    assert np.allclose(
        curve["discount_factor"],
        [0.98, 0.95],
        atol=1e-6,
    )


def test_prepared_dataset_has_required_fields_and_split() -> None:
    standardised = standardise_cboe_columns(
        _parity_consistent_fixture()
    )
    standardised["bid"] = standardised[
        "bid_1545"
    ]
    standardised["ask"] = standardised[
        "ask_1545"
    ]
    standardised["mid"] = 0.5 * (
        standardised["bid"]
        + standardised["ask"]
    )
    standardised["spread"] = (
        standardised["ask"]
        - standardised["bid"]
    )
    standardised["relative_spread"] = (
        standardised["spread"]
        / standardised["mid"]
    )
    standardised["maturity"] = (
        standardised["expiration"]
        - standardised["quote_date"]
    ).dt.days / 365.25

    curve = infer_forward_curve_from_parity(
        standardised,
        minimum_pairs=5,
    )

    prepared = prepare_calibration_quotes(
        filtered_quotes=standardised,
        forward_curve=curve,
        settings=PreparationSettings(
            maximum_absolute_log_moneyness=0.30,
        ),
    )

    assert {
        "observed_call_price",
        "noise_standard_deviation",
        "forward",
        "discount_factor",
        "log_moneyness",
        "is_train",
        "is_test",
    }.issubset(prepared.columns)
    assert prepared["is_train"].any()
    assert prepared["is_test"].any()
    assert np.all(
        prepared["noise_standard_deviation"]
        > 0
    )
    assert np.all(
        np.abs(prepared["log_moneyness"])
        <= 0.30 + 1e-12
    )


def test_clean_fixture_has_no_strike_arbitrage_violations() -> None:
    standardised = standardise_cboe_columns(
        _parity_consistent_fixture()
    )
    standardised["bid"] = standardised[
        "bid_1545"
    ]
    standardised["ask"] = standardised[
        "ask_1545"
    ]
    standardised["mid"] = 0.5 * (
        standardised["bid"]
        + standardised["ask"]
    )
    standardised["spread"] = (
        standardised["ask"]
        - standardised["bid"]
    )
    standardised["relative_spread"] = (
        standardised["spread"]
        / standardised["mid"]
    )
    standardised["maturity"] = (
        standardised["expiration"]
        - standardised["quote_date"]
    ).dt.days / 365.25

    curve = infer_forward_curve_from_parity(
        standardised,
        minimum_pairs=5,
    )
    prepared = prepare_calibration_quotes(
        filtered_quotes=standardised,
        forward_curve=curve,
        settings=PreparationSettings(
            maximum_absolute_log_moneyness=0.30,
        ),
    )

    diagnostics = strike_arbitrage_diagnostics(
        prepared,
        tolerance_multiplier=3.0,
    )

    assert (
        diagnostics[
            "monotonicity_violations"
        ].sum()
        == 0
    )
    assert (
        diagnostics[
            "convexity_violations"
        ].sum()
        == 0
    )
