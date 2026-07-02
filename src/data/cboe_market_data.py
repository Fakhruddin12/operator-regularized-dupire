"""Acquisition and preparation of real Cboe option-chain data.

The preferred source is Cboe DataShop's official Option EOD Summary sample.
It contains OPRA-derived NBBO snapshots at 15:45 U.S. Eastern and end of day.

The preparation pipeline:

1. standardises Cboe column names;
2. selects one symbol and quote date;
3. filters unusable or excessively wide quotes;
4. estimates forward and discount factor expiry by expiry from put-call parity;
5. uses the more liquid OTM call or OTM put at each strike;
6. converts OTM puts into European call-equivalent prices;
7. constructs forward log-moneyness and quote-noise weights;
8. creates a deterministic maturity-stratified train/test split.

For European index options such as SPX, put-call parity is exact up to market
microstructure noise. For American equity/ETF options it is only an
approximation, so the output records the exercise-style caveat.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd
import requests
from scipy.optimize import least_squares


CBOE_EOD_SAMPLE_URL = (
    "https://datashop.cboe.com/download/sample/217"
)


COLUMN_ALIASES = {
    "underlying": "underlying_symbol",
    "underlying_symbol": "underlying_symbol",
    "symbol": "underlying_symbol",
    "quote_date": "quote_date",
    "date": "quote_date",
    "root": "root",
    "expiration": "expiration",
    "expiration_date": "expiration",
    "expiry": "expiration",
    "strike": "strike",
    "option_type": "option_type",
    "type": "option_type",
    "bid_1545": "bid_1545",
    "ask_1545": "ask_1545",
    "bid_size_1545": "bid_size_1545",
    "ask_size_1545": "ask_size_1545",
    "underlying_bid_1545": "underlying_bid_1545",
    "underlying_ask_1545": "underlying_ask_1545",
    "active_underlying_price_1545": (
        "active_underlying_price_1545"
    ),
    "bid_eod": "bid_eod",
    "ask_eod": "ask_eod",
    "bid_size_eod": "bid_size_eod",
    "ask_size_eod": "ask_size_eod",
    "underlying_bid_eod": "underlying_bid_eod",
    "underlying_ask_eod": "underlying_ask_eod",
    "trade_volume": "trade_volume",
    "volume": "trade_volume",
    "open_interest": "open_interest",
}


@dataclass(frozen=True)
class PreparationSettings:
    """Quality filters used for the real-data calibration dataset."""

    snapshot: str = "1545"
    minimum_days_to_expiry: int = 14
    maximum_days_to_expiry: int = 730
    maximum_relative_spread: float = 0.50
    maximum_absolute_log_moneyness: float = 0.35
    minimum_noise_standard_deviation: float = 0.01
    minimum_parity_pairs: int = 5
    test_fraction: float = 0.20


def _normalise_column_name(name: str) -> str:
    """Convert a raw column label into a stable snake-case label."""
    normalised = str(name).strip().lower()
    normalised = re.sub(r"[^a-z0-9]+", "_", normalised)
    normalised = normalised.strip("_")

    # Cboe specifications sometimes show footnote digits next to labels.
    normalised = re.sub(
        r"_(1|2|3|4)$",
        "",
        normalised,
    )

    return COLUMN_ALIASES.get(
        normalised,
        normalised,
    )


def standardise_cboe_columns(
    raw_data: pd.DataFrame,
) -> pd.DataFrame:
    """Standardise common Cboe EOD Summary column names and data types."""
    if raw_data.empty:
        raise ValueError("raw_data is empty.")

    data = raw_data.copy()
    data.columns = [
        _normalise_column_name(column)
        for column in data.columns
    ]

    if data.columns.duplicated().any():
        duplicate_names = data.columns[
            data.columns.duplicated()
        ].tolist()
        raise ValueError(
            "Column normalisation produced duplicate columns: "
            f"{duplicate_names}"
        )

    required = {
        "underlying_symbol",
        "quote_date",
        "expiration",
        "strike",
        "option_type",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            f"Missing required Cboe columns: {sorted(missing)}"
        )

    data["underlying_symbol"] = (
        data["underlying_symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    data["option_type"] = (
        data["option_type"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace(
            {
                "CALL": "C",
                "PUT": "P",
            }
        )
    )
    data["quote_date"] = pd.to_datetime(
        data["quote_date"],
        errors="coerce",
    ).dt.normalize()
    data["expiration"] = pd.to_datetime(
        data["expiration"],
        errors="coerce",
    ).dt.normalize()

    numeric_columns = [
        "strike",
        "bid_1545",
        "ask_1545",
        "bid_size_1545",
        "ask_size_1545",
        "underlying_bid_1545",
        "underlying_ask_1545",
        "active_underlying_price_1545",
        "bid_eod",
        "ask_eod",
        "bid_size_eod",
        "ask_size_eod",
        "underlying_bid_eod",
        "underlying_ask_eod",
        "trade_volume",
        "open_interest",
    ]

    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

    data = data.dropna(
        subset=[
            "underlying_symbol",
            "quote_date",
            "expiration",
            "strike",
            "option_type",
        ]
    )

    data = data[
        data["option_type"].isin(["C", "P"])
        & (data["strike"] > 0)
    ].copy()

    return data.reset_index(drop=True)


def download_cboe_eod_sample(
    destination: str | Path,
    url: str = CBOE_EOD_SAMPLE_URL,
    timeout_seconds: int = 120,
) -> Path:
    """Download Cboe's official EOD Summary sample ZIP once.

    This accesses the DataShop sample download, not the delayed-quote webpage.
    Existing files are reused to avoid repeated downloads.
    """
    destination_path = Path(destination)
    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination_path.exists():
        return destination_path

    response = requests.get(
        url,
        timeout=timeout_seconds,
        headers={
            "User-Agent": (
                "Academic local-volatility research; "
                "single official sample download"
            )
        },
    )
    response.raise_for_status()

    content = response.content
    if not zipfile.is_zipfile(BytesIO(content)):
        raise ValueError(
            "The Cboe sample response was not a valid ZIP file."
        )

    destination_path.write_bytes(content)
    return destination_path


def load_cboe_sample_zip(
    zip_path: str | Path,
) -> pd.DataFrame:
    """Read and concatenate every CSV contained in a Cboe sample ZIP."""
    path = Path(zip_path)
    if not path.exists():
        raise FileNotFoundError(path)

    frames: list[pd.DataFrame] = []

    with zipfile.ZipFile(path) as archive:
        csv_names = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]

        if not csv_names:
            raise ValueError(
                "The ZIP archive contains no CSV files."
            )

        for name in csv_names:
            with archive.open(name) as file_object:
                frame = pd.read_csv(
                    file_object,
                    low_memory=False,
                )
                frame["_source_file"] = name
                frames.append(frame)

    return standardise_cboe_columns(
        pd.concat(
            frames,
            ignore_index=True,
        )
    )


def available_symbols(
    standardised_data: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise quote counts and matched call-put strikes by symbol/date."""
    required = {
        "underlying_symbol",
        "quote_date",
        "expiration",
        "strike",
        "option_type",
    }
    missing = required.difference(
        standardised_data.columns
    )
    if missing:
        raise ValueError(
            f"standardised_data is missing: {sorted(missing)}"
        )

    unique_series = (
        standardised_data[
            [
                "underlying_symbol",
                "quote_date",
                "expiration",
                "strike",
                "option_type",
            ]
        ]
        .drop_duplicates()
    )

    counts = (
        unique_series.groupby(
            [
                "underlying_symbol",
                "quote_date",
            ],
            as_index=False,
        )
        .agg(
            number_of_series=(
                "option_type",
                "size",
            ),
            number_of_expirations=(
                "expiration",
                "nunique",
            ),
        )
    )

    pair_counts = (
        unique_series.pivot_table(
            index=[
                "underlying_symbol",
                "quote_date",
                "expiration",
                "strike",
            ],
            columns="option_type",
            values="option_type",
            aggfunc="size",
            fill_value=0,
        )
        .reset_index()
    )

    if "C" not in pair_counts:
        pair_counts["C"] = 0
    if "P" not in pair_counts:
        pair_counts["P"] = 0

    pair_counts["matched_pair"] = (
        (pair_counts["C"] > 0)
        & (pair_counts["P"] > 0)
    )

    pair_summary = (
        pair_counts.groupby(
            [
                "underlying_symbol",
                "quote_date",
            ],
            as_index=False,
        )
        .agg(
            matched_call_put_strikes=(
                "matched_pair",
                "sum",
            )
        )
    )

    return (
        counts.merge(
            pair_summary,
            on=[
                "underlying_symbol",
                "quote_date",
            ],
            how="left",
        )
        .sort_values(
            [
                "matched_call_put_strikes",
                "number_of_series",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )


def choose_preferred_symbol(
    standardised_data: pd.DataFrame,
    preferred_symbols: tuple[str, ...] = (
        "^SPX",
        "SPX",
        "^XSP",
        "XSP",
        "SPY",
    ),
) -> tuple[str, pd.Timestamp]:
    """Choose a liquid symbol/date, prioritising European-style SPX/XSP."""
    summary = available_symbols(
        standardised_data
    )

    if summary.empty:
        raise ValueError(
            "No symbols are available in the sample."
        )

    for symbol in preferred_symbols:
        candidates = summary[
            summary["underlying_symbol"] == symbol
        ]
        if not candidates.empty:
            row = candidates.iloc[0]
            return (
                str(row["underlying_symbol"]),
                pd.Timestamp(row["quote_date"]),
            )

    row = summary.iloc[0]
    return (
        str(row["underlying_symbol"]),
        pd.Timestamp(row["quote_date"]),
    )


def _snapshot_columns(
    data: pd.DataFrame,
    snapshot: str,
) -> tuple[str, str]:
    """Resolve bid and ask column names for the chosen snapshot."""
    snapshot_value = snapshot.lower()

    if snapshot_value == "1545":
        bid_column = "bid_1545"
        ask_column = "ask_1545"
    elif snapshot_value == "eod":
        bid_column = "bid_eod"
        ask_column = "ask_eod"
    else:
        raise ValueError(
            "snapshot must be either '1545' or 'eod'."
        )

    missing = {
        bid_column,
        ask_column,
    }.difference(data.columns)
    if missing:
        raise ValueError(
            f"Data is missing snapshot columns: {sorted(missing)}"
        )

    return bid_column, ask_column


def filter_option_quotes(
    standardised_data: pd.DataFrame,
    symbol: str,
    quote_date: str | pd.Timestamp,
    settings: PreparationSettings,
) -> pd.DataFrame:
    """Select one chain snapshot and remove clearly unusable quotes."""
    bid_column, ask_column = _snapshot_columns(
        standardised_data,
        settings.snapshot,
    )

    date_value = pd.Timestamp(
        quote_date
    ).normalize()
    symbol_value = symbol.upper()

    data = standardised_data[
        (
            standardised_data[
                "underlying_symbol"
            ]
            == symbol_value
        )
        & (
            standardised_data[
                "quote_date"
            ]
            == date_value
        )
    ].copy()

    if data.empty:
        raise ValueError(
            f"No data found for {symbol_value} on {date_value.date()}."
        )

    data["bid"] = pd.to_numeric(
        data[bid_column],
        errors="coerce",
    )
    data["ask"] = pd.to_numeric(
        data[ask_column],
        errors="coerce",
    )
    data["mid"] = 0.5 * (
        data["bid"] + data["ask"]
    )
    data["spread"] = (
        data["ask"] - data["bid"]
    )
    data["relative_spread"] = (
        data["spread"]
        / np.maximum(data["mid"], 0.05)
    )
    data["days_to_expiry"] = (
        data["expiration"]
        - data["quote_date"]
    ).dt.days
    data["maturity"] = (
        data["days_to_expiry"] / 365.25
    )

    valid = (
        np.isfinite(data["bid"])
        & np.isfinite(data["ask"])
        & (data["bid"] >= 0)
        & (data["ask"] > 0)
        & (data["ask"] >= data["bid"])
        & (data["mid"] > 0)
        & (
            data["relative_spread"]
            <= settings.maximum_relative_spread
        )
        & (
            data["days_to_expiry"]
            >= settings.minimum_days_to_expiry
        )
        & (
            data["days_to_expiry"]
            <= settings.maximum_days_to_expiry
        )
    )

    filtered = data[valid].copy()

    if filtered.empty:
        raise ValueError(
            "All quotes were removed by the quality filters."
        )

    return filtered.sort_values(
        [
            "expiration",
            "strike",
            "option_type",
        ]
    ).reset_index(drop=True)


def infer_forward_curve_from_parity(
    filtered_quotes: pd.DataFrame,
    minimum_pairs: int = 5,
) -> pd.DataFrame:
    """Estimate discount factors and forwards from robust put-call parity.

    For each expiry, the fitted relation is

        C(K,T) - P(K,T) = A(T) - D(T) K,

    where ``D(T)`` is the discount factor and ``F(T)=A(T)/D(T)``.
    """
    required = {
        "expiration",
        "strike",
        "option_type",
        "mid",
        "spread",
        "maturity",
    }
    missing = required.difference(
        filtered_quotes.columns
    )
    if missing:
        raise ValueError(
            f"filtered_quotes is missing: {sorted(missing)}"
        )

    pivot = filtered_quotes.pivot_table(
        index=[
            "expiration",
            "strike",
            "maturity",
        ],
        columns="option_type",
        values=[
            "mid",
            "spread",
        ],
        aggfunc="median",
    )

    needed_columns = [
        ("mid", "C"),
        ("mid", "P"),
        ("spread", "C"),
        ("spread", "P"),
    ]
    if not all(
        column in pivot.columns
        for column in needed_columns
    ):
        raise ValueError(
            "Matched call and put quotes are required."
        )

    pairs = pivot.dropna(
        subset=needed_columns
    ).reset_index()

    results = []

    for expiration, group in pairs.groupby(
        "expiration"
    ):
        if len(group) < minimum_pairs:
            continue

        strikes = group["strike"].to_numpy(
            dtype=float
        )
        call_minus_put = (
            group[("mid", "C")].to_numpy(
                dtype=float
            )
            - group[("mid", "P")].to_numpy(
                dtype=float
            )
        )
        combined_half_spread = 0.5 * (
            group[("spread", "C")].to_numpy(
                dtype=float
            )
            + group[("spread", "P")].to_numpy(
                dtype=float
            )
        )
        scale = np.maximum(
            combined_half_spread,
            0.02,
        )
        maturity = float(
            group["maturity"].iloc[0]
        )

        initial_discount = np.exp(
            -0.04 * maturity
        )
        initial_prepaid_forward = float(
            np.median(
                call_minus_put
                + initial_discount * strikes
            )
        )
        initial_prepaid_forward = max(
            initial_prepaid_forward,
            1e-6,
        )

        def weighted_residual(
            parameters: np.ndarray,
        ) -> np.ndarray:
            prepaid_forward, discount_factor = (
                parameters
            )
            model = (
                prepaid_forward
                - discount_factor * strikes
            )
            return (
                call_minus_put - model
            ) / scale

        fit = least_squares(
            weighted_residual,
            x0=np.array(
                [
                    initial_prepaid_forward,
                    initial_discount,
                ]
            ),
            bounds=(
                np.array([1e-8, 0.50]),
                np.array([np.inf, 1.20]),
            ),
            loss="soft_l1",
            f_scale=1.0,
        )

        prepaid_forward = float(fit.x[0])
        discount_factor = float(fit.x[1])
        forward = (
            prepaid_forward
            / discount_factor
        )
        implied_rate = (
            -np.log(discount_factor)
            / maturity
        )

        raw_residual = (
            call_minus_put
            - (
                prepaid_forward
                - discount_factor * strikes
            )
        )

        results.append(
            {
                "expiration": pd.Timestamp(
                    expiration
                ),
                "maturity": maturity,
                "number_of_parity_pairs": int(
                    len(group)
                ),
                "prepaid_forward": (
                    prepaid_forward
                ),
                "discount_factor": (
                    discount_factor
                ),
                "forward": float(forward),
                "implied_rate": float(
                    implied_rate
                ),
                "parity_rmse": float(
                    np.sqrt(
                        np.mean(
                            raw_residual**2
                        )
                    )
                ),
                "parity_success": bool(
                    fit.success
                ),
            }
        )

    curve = pd.DataFrame(results)

    if curve.empty:
        raise ValueError(
            "No expiry had enough matched call-put pairs "
            "for forward inference."
        )

    return curve.sort_values(
        "expiration"
    ).reset_index(drop=True)


def _exercise_style_label(symbol: str) -> str:
    """Return the relevant model-compatibility label."""
    clean_symbol = symbol.upper().lstrip("^")
    if clean_symbol in {"SPX", "XSP"}:
        return "european_index"
    return "american_or_unknown"


def _deterministic_holdout(
    data: pd.DataFrame,
    test_fraction: float,
) -> pd.Series:
    """Create a maturity-stratified deterministic holdout mask."""
    if not 0 < test_fraction < 0.5:
        raise ValueError(
            "test_fraction must lie between 0 and 0.5."
        )

    mask = pd.Series(
        False,
        index=data.index,
    )

    for _, group in data.groupby(
        "expiration"
    ):
        ordered_indices = (
            group.sort_values(
                "log_moneyness"
            ).index.to_numpy()
        )
        group_size = ordered_indices.size

        if group_size < 5:
            continue

        number_test = max(
            1,
            int(round(test_fraction * group_size)),
        )
        positions = np.linspace(
            1,
            group_size - 2,
            number_test,
            dtype=int,
        )
        mask.loc[
            ordered_indices[
                np.unique(positions)
            ]
        ] = True

    return mask


def prepare_calibration_quotes(
    filtered_quotes: pd.DataFrame,
    forward_curve: pd.DataFrame,
    settings: PreparationSettings,
) -> pd.DataFrame:
    """Create one OTM-sourced European call-equivalent quote per strike."""
    required_quote_columns = {
        "underlying_symbol",
        "quote_date",
        "expiration",
        "strike",
        "option_type",
        "bid",
        "ask",
        "mid",
        "spread",
        "relative_spread",
        "maturity",
    }
    missing_quotes = required_quote_columns.difference(
        filtered_quotes.columns
    )
    if missing_quotes:
        raise ValueError(
            f"filtered_quotes is missing: {sorted(missing_quotes)}"
        )

    required_curve_columns = {
        "expiration",
        "discount_factor",
        "forward",
        "prepaid_forward",
        "implied_rate",
        "parity_rmse",
        "number_of_parity_pairs",
    }
    missing_curve = required_curve_columns.difference(
        forward_curve.columns
    )
    if missing_curve:
        raise ValueError(
            f"forward_curve is missing: {sorted(missing_curve)}"
        )

    data = filtered_quotes.merge(
        forward_curve,
        on=[
            "expiration",
            "maturity",
        ],
        how="inner",
        validate="many_to_one",
    )

    if data.empty:
        raise ValueError(
            "No quotes matched the inferred forward curve."
        )

    selected_rows = []

    for (
        expiration,
        strike,
    ), group in data.groupby(
        [
            "expiration",
            "strike",
        ]
    ):
        forward = float(
            group["forward"].iloc[0]
        )

        preferred_type = (
            "C"
            if float(strike) >= forward
            else "P"
        )

        preferred = group[
            group["option_type"]
            == preferred_type
        ]

        if preferred.empty:
            continue

        # If duplicates exist, keep the narrowest spread.
        row = preferred.sort_values(
            [
                "relative_spread",
                "spread",
            ]
        ).iloc[0].copy()

        parity_shift = float(
            row["discount_factor"]
            * (
                row["forward"]
                - row["strike"]
            )
        )

        if preferred_type == "C":
            call_bid = float(row["bid"])
            call_ask = float(row["ask"])
            source_label = "otm_call"
        else:
            call_bid = float(
                row["bid"] + parity_shift
            )
            call_ask = float(
                row["ask"] + parity_shift
            )
            source_label = "otm_put_parity"

        if call_ask < call_bid:
            continue

        call_mid = 0.5 * (
            call_bid + call_ask
        )
        call_spread = (
            call_ask - call_bid
        )
        lower_bound = float(
            row["discount_factor"]
            * max(
                row["forward"]
                - row["strike"],
                0.0,
            )
        )
        upper_bound = float(
            row["discount_factor"]
            * row["forward"]
        )
        tolerance = max(
            call_spread,
            settings.minimum_noise_standard_deviation,
        )

        if (
            call_mid < lower_bound - tolerance
            or call_mid > upper_bound + tolerance
        ):
            continue

        row["source_option"] = source_label
        row["source_option_type"] = (
            preferred_type
        )
        row["call_bid"] = call_bid
        row["call_ask"] = call_ask
        row["observed_call_price"] = (
            call_mid
        )
        row["call_spread"] = call_spread
        row[
            "noise_standard_deviation"
        ] = max(
            0.5 * call_spread,
            settings.minimum_noise_standard_deviation,
        )
        row["call_lower_bound"] = (
            lower_bound
        )
        row["call_upper_bound"] = (
            upper_bound
        )
        row["log_moneyness"] = float(
            np.log(
                row["strike"]
                / row["forward"]
            )
        )
        selected_rows.append(row)

    if not selected_rows:
        raise ValueError(
            "No OTM call-equivalent quotes could be constructed."
        )

    prepared = pd.DataFrame(
        selected_rows
    )

    prepared = prepared[
        np.abs(
            prepared["log_moneyness"]
        )
        <= settings.maximum_absolute_log_moneyness
    ].copy()

    if prepared.empty:
        raise ValueError(
            "No quotes remain inside the log-moneyness range."
        )

    prepared = prepared.sort_values(
        [
            "expiration",
            "log_moneyness",
        ]
    ).reset_index(drop=True)

    test_mask = _deterministic_holdout(
        prepared,
        settings.test_fraction,
    )
    prepared["is_test"] = (
        test_mask.to_numpy(dtype=bool)
    )
    prepared["is_train"] = (
        ~prepared["is_test"]
    )
    prepared["quote_weight"] = (
        1.0
        / prepared[
            "noise_standard_deviation"
        ]
    )
    prepared["exercise_style"] = (
        _exercise_style_label(
            str(
                prepared[
                    "underlying_symbol"
                ].iloc[0]
            )
        )
    )

    output_columns = [
        "underlying_symbol",
        "quote_date",
        "expiration",
        "maturity",
        "strike",
        "source_option",
        "source_option_type",
        "bid",
        "ask",
        "mid",
        "spread",
        "relative_spread",
        "call_bid",
        "call_ask",
        "observed_call_price",
        "call_spread",
        "noise_standard_deviation",
        "quote_weight",
        "discount_factor",
        "prepaid_forward",
        "forward",
        "implied_rate",
        "log_moneyness",
        "call_lower_bound",
        "call_upper_bound",
        "parity_rmse",
        "number_of_parity_pairs",
        "trade_volume",
        "open_interest",
        "is_train",
        "is_test",
        "exercise_style",
    ]

    for column in output_columns:
        if column not in prepared.columns:
            prepared[column] = np.nan

    return prepared[
        output_columns
    ].reset_index(drop=True)


def strike_arbitrage_diagnostics(
    prepared_quotes: pd.DataFrame,
    tolerance_multiplier: float = 2.0,
) -> pd.DataFrame:
    """Check monotonicity and convexity across strike for each expiry."""
    required = {
        "expiration",
        "strike",
        "observed_call_price",
        "noise_standard_deviation",
    }
    missing = required.difference(
        prepared_quotes.columns
    )
    if missing:
        raise ValueError(
            f"prepared_quotes is missing: {sorted(missing)}"
        )

    rows = []

    for expiration, group in prepared_quotes.groupby(
        "expiration"
    ):
        ordered = group.sort_values(
            "strike"
        )
        strikes = ordered[
            "strike"
        ].to_numpy(dtype=float)
        prices = ordered[
            "observed_call_price"
        ].to_numpy(dtype=float)
        noise = ordered[
            "noise_standard_deviation"
        ].to_numpy(dtype=float)

        if len(ordered) < 3:
            continue

        price_changes = np.diff(prices)
        monotonic_tolerance = (
            tolerance_multiplier
            * np.maximum(
                noise[:-1],
                noise[1:],
            )
        )
        monotonic_violations = int(
            np.sum(
                price_changes
                > monotonic_tolerance
            )
        )

        slopes = (
            np.diff(prices)
            / np.diff(strikes)
        )
        slope_changes = np.diff(slopes)
        typical_noise = float(
            np.median(noise)
        )
        typical_strike_step = float(
            np.median(
                np.diff(strikes)
            )
        )
        convexity_tolerance = (
            tolerance_multiplier
            * typical_noise
            / max(
                typical_strike_step,
                1e-8,
            )
        )
        convexity_violations = int(
            np.sum(
                slope_changes
                < -convexity_tolerance
            )
        )

        rows.append(
            {
                "expiration": pd.Timestamp(
                    expiration
                ),
                "number_of_quotes": int(
                    len(ordered)
                ),
                "monotonicity_violations": (
                    monotonic_violations
                ),
                "convexity_violations": (
                    convexity_violations
                ),
            }
        )

    return pd.DataFrame(rows)


def prepare_cboe_sample_dataset(
    raw_data: pd.DataFrame,
    symbol: str | None = None,
    quote_date: str | pd.Timestamp | None = None,
    settings: PreparationSettings | None = None,
) -> dict[str, object]:
    """Run the complete Stage 12 preparation pipeline."""
    settings_value = (
        settings
        if settings is not None
        else PreparationSettings()
    )

    standardised = standardise_cboe_columns(
        raw_data
    )

    if symbol is None or quote_date is None:
        selected_symbol, selected_date = (
            choose_preferred_symbol(
                standardised
            )
        )
        if symbol is None:
            symbol = selected_symbol
        if quote_date is None:
            quote_date = selected_date

    filtered = filter_option_quotes(
        standardised_data=standardised,
        symbol=str(symbol),
        quote_date=quote_date,
        settings=settings_value,
    )
    forward_curve = infer_forward_curve_from_parity(
        filtered,
        minimum_pairs=(
            settings_value.minimum_parity_pairs
        ),
    )
    prepared = prepare_calibration_quotes(
        filtered_quotes=filtered,
        forward_curve=forward_curve,
        settings=settings_value,
    )
    diagnostics = strike_arbitrage_diagnostics(
        prepared
    )

    metadata = {
        "symbol": str(symbol),
        "quote_date": str(
            pd.Timestamp(
                quote_date
            ).date()
        ),
        "snapshot": settings_value.snapshot,
        "exercise_style": str(
            prepared["exercise_style"].iloc[0]
        ),
        "number_of_raw_rows": int(
            len(raw_data)
        ),
        "number_of_filtered_option_rows": int(
            len(filtered)
        ),
        "number_of_prepared_quotes": int(
            len(prepared)
        ),
        "number_of_train_quotes": int(
            prepared["is_train"].sum()
        ),
        "number_of_test_quotes": int(
            prepared["is_test"].sum()
        ),
        "number_of_expirations": int(
            prepared["expiration"].nunique()
        ),
    }

    return {
        "standardised_data": standardised,
        "filtered_quotes": filtered,
        "forward_curve": forward_curve,
        "prepared_quotes": prepared,
        "arbitrage_diagnostics": diagnostics,
        "metadata": metadata,
    }
