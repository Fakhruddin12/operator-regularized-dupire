"""Raw and smoothed direct-Dupire local-volatility estimators.

The price surface is represented in log-moneyness and maturity coordinates:

    x = log(K / F(T)).

For the European call price C(x,T), the forward Dupire equation is

    sigma_loc(x,T)^2
    = 2 * (C_T + q C) / (C_xx - C_x).

The denominator contains a second derivative and is therefore highly sensitive
 to quote noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.surfaces.price_smoothing import smooth_price_surface


def rectangular_surface_from_quotes(
    quote_data: pd.DataFrame,
    value_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a complete rectangular quote grid into a matrix."""
    required_columns = {"maturity", "log_moneyness", value_column}
    missing_columns = required_columns.difference(quote_data.columns)
    if missing_columns:
        raise ValueError(
            f"quote_data is missing required columns: {sorted(missing_columns)}"
        )

    if quote_data.duplicated(["maturity", "log_moneyness"]).any():
        raise ValueError(
            "quote_data contains duplicate maturity/log-moneyness pairs."
        )

    maturities = np.unique(
        np.sort(quote_data["maturity"].to_numpy(dtype=float))
    )
    x_values = np.unique(
        np.sort(quote_data["log_moneyness"].to_numpy(dtype=float))
    )

    expected_number_of_rows = maturities.size * x_values.size
    if len(quote_data) != expected_number_of_rows:
        raise ValueError(
            "quote_data must contain every maturity/log-moneyness combination."
        )

    surface_frame = quote_data.pivot(
        index="maturity",
        columns="log_moneyness",
        values=value_column,
    ).reindex(index=maturities, columns=x_values)

    if surface_frame.isna().any().any():
        raise ValueError(
            "quote_data does not form a complete finite rectangular surface."
        )

    return maturities, x_values, surface_frame.to_numpy(dtype=float)


def finite_difference_price_derivatives(
    price_surface: np.ndarray,
    maturities: np.ndarray,
    log_moneyness: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate C_T, C_x, and C_xx using second-order finite differences."""
    prices = np.asarray(price_surface, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    x_values = np.asarray(log_moneyness, dtype=float)

    expected_shape = (maturities.size, x_values.size)
    if prices.shape != expected_shape:
        raise ValueError(
            f"price_surface has shape {prices.shape}; expected {expected_shape}."
        )
    if maturities.size < 3 or x_values.size < 3:
        raise ValueError(
            "At least three maturity and three log-moneyness values are needed."
        )
    if np.any(np.diff(maturities) <= 0):
        raise ValueError("maturities must be strictly increasing.")
    if np.any(np.diff(x_values) <= 0):
        raise ValueError("log_moneyness must be strictly increasing.")

    time_derivative = np.gradient(
        prices,
        maturities,
        axis=0,
        edge_order=2,
    )
    first_x_derivative = np.gradient(
        prices,
        x_values,
        axis=1,
        edge_order=2,
    )
    second_x_derivative = np.gradient(
        first_x_derivative,
        x_values,
        axis=1,
        edge_order=2,
    )

    return time_derivative, first_x_derivative, second_x_derivative


def dupire_from_price_surface(
    price_surface: np.ndarray,
    maturities: np.ndarray,
    log_moneyness: np.ndarray,
    dividend_yield: float = 0.0,
    denominator_floor: float = 1e-10,
) -> dict[str, np.ndarray]:
    """Apply the direct Dupire formula to a price surface.

    Invalid points are retained as NaN rather than clipped into apparently
    plausible volatility values.
    """
    prices = np.asarray(price_surface, dtype=float)

    time_derivative, first_x_derivative, second_x_derivative = (
        finite_difference_price_derivatives(
            price_surface=prices,
            maturities=maturities,
            log_moneyness=log_moneyness,
        )
    )

    numerator = 2.0 * (
        time_derivative + dividend_yield * prices
    )
    denominator = second_x_derivative - first_x_derivative

    with np.errstate(divide="ignore", invalid="ignore"):
        local_variance = numerator / denominator

    valid_mask = (
        np.isfinite(local_variance)
        & np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator > denominator_floor)
        & (local_variance > 0.0)
    )

    local_volatility = np.full_like(local_variance, np.nan)
    local_volatility[valid_mask] = np.sqrt(local_variance[valid_mask])

    local_variance = np.where(valid_mask, local_variance, np.nan)

    return {
        "time_derivative": time_derivative,
        "first_x_derivative": first_x_derivative,
        "second_x_derivative": second_x_derivative,
        "numerator": numerator,
        "denominator": denominator,
        "local_variance": local_variance,
        "local_volatility": local_volatility,
        "valid_mask": valid_mask,
    }


def _result_dataframe(
    maturities: np.ndarray,
    log_moneyness: np.ndarray,
    prices: np.ndarray,
    dupire_result: dict[str, np.ndarray],
    method: str,
) -> pd.DataFrame:
    """Convert matrix-valued Dupire outputs into a tidy DataFrame."""
    x_mesh, maturity_mesh = np.meshgrid(log_moneyness, maturities)

    return pd.DataFrame(
        {
            "maturity": maturity_mesh.reshape(-1),
            "log_moneyness": x_mesh.reshape(-1),
            "price_used": prices.reshape(-1),
            "time_derivative": dupire_result["time_derivative"].reshape(-1),
            "first_x_derivative": dupire_result["first_x_derivative"].reshape(-1),
            "second_x_derivative": dupire_result["second_x_derivative"].reshape(-1),
            "dupire_numerator": dupire_result["numerator"].reshape(-1),
            "dupire_denominator": dupire_result["denominator"].reshape(-1),
            "local_variance": dupire_result["local_variance"].reshape(-1),
            "local_volatility": dupire_result["local_volatility"].reshape(-1),
            "valid_dupire": dupire_result["valid_mask"].reshape(-1),
            "method": method,
        }
    )


def raw_dupire_from_quotes(
    quote_data: pd.DataFrame,
    price_column: str = "observed_call_price",
    dividend_yield: float = 0.0,
    denominator_floor: float = 1e-10,
) -> pd.DataFrame:
    """Apply direct finite-difference Dupire to raw option quotes."""
    maturities, x_values, prices = rectangular_surface_from_quotes(
        quote_data=quote_data,
        value_column=price_column,
    )

    result = dupire_from_price_surface(
        price_surface=prices,
        maturities=maturities,
        log_moneyness=x_values,
        dividend_yield=dividend_yield,
        denominator_floor=denominator_floor,
    )

    return _result_dataframe(
        maturities=maturities,
        log_moneyness=x_values,
        prices=prices,
        dupire_result=result,
        method="raw_dupire",
    )


def smoothed_dupire_from_quotes(
    quote_data: pd.DataFrame,
    price_column: str = "observed_call_price",
    noise_column: str | None = "noise_standard_deviation",
    dividend_yield: float = 0.0,
    denominator_floor: float = 1e-10,
    x_smoothing_multiplier: float = 0.5,
    time_smoothing_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Smooth the price surface first and then apply direct Dupire."""
    maturities, x_values, prices = rectangular_surface_from_quotes(
        quote_data=quote_data,
        value_column=price_column,
    )

    noise_surface = None
    if noise_column is not None:
        _, _, noise_surface = rectangular_surface_from_quotes(
            quote_data=quote_data,
            value_column=noise_column,
        )

    smoothed_prices = smooth_price_surface(
        price_surface=prices,
        maturities=maturities,
        log_moneyness=x_values,
        noise_standard_deviation=noise_surface,
        x_smoothing_multiplier=x_smoothing_multiplier,
        time_smoothing_multiplier=time_smoothing_multiplier,
    )

    result = dupire_from_price_surface(
        price_surface=smoothed_prices,
        maturities=maturities,
        log_moneyness=x_values,
        dividend_yield=dividend_yield,
        denominator_floor=denominator_floor,
    )

    return _result_dataframe(
        maturities=maturities,
        log_moneyness=x_values,
        prices=smoothed_prices,
        dupire_result=result,
        method="smoothed_dupire",
    )
