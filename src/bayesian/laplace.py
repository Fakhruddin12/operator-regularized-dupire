"""Gaussian/Laplace uncertainty for the local-volatility inverse problem.

Near the deterministic MAP estimate ``u_map``, the pricing map is linearised:

    F(u_map + delta) approximately F(u_map) + J delta.

With quote weights ``W`` and Gaussian prior precision ``lambda R``, the local
posterior precision is

    H = J.T W.T W J + lambda R.

The Laplace covariance is ``H^{-1}``.

Because quote residuals are already divided by their known noise standard
deviations, no additional observation-variance multiplier is required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csc_matrix
from scipy.stats import norm


def build_laplace_posterior(
    map_log_variance: np.ndarray,
    jacobian: np.ndarray,
    weights: np.ndarray,
    regularization_matrix: csc_matrix,
    regularization_strength: float,
    diagonal_jitter: float = 0.0,
) -> dict[str, np.ndarray | float]:
    """Construct the Gaussian/Laplace posterior around a MAP estimate.

    Parameters
    ----------
    map_log_variance:
        MAP log-variance surface, with shape ``(n_T, n_x)``.
    jacobian:
        Price Jacobian at the MAP estimate. Rows are quotes and columns are
        C-order-flattened log-variance parameters.
    weights:
        Diagonal entries of the quote-weight matrix ``W``.
    regularization_matrix:
        Scaled regularisation matrix ``R``.
    regularization_strength:
        Prior precision multiplier ``lambda``.
    diagonal_jitter:
        Optional small positive diagonal term used only for numerical
        stabilisation.

    Returns
    -------
    dict
        MAP vector, posterior precision, covariance, standard deviations, and
        correlation matrix.
    """
    map_surface = np.asarray(
        map_log_variance,
        dtype=float,
    )
    J = np.asarray(jacobian, dtype=float)
    weight_values = np.asarray(
        weights,
        dtype=float,
    ).reshape(-1)

    if map_surface.ndim != 2:
        raise ValueError(
            "map_log_variance must be a two-dimensional surface."
        )

    number_of_parameters = map_surface.size

    if J.ndim != 2:
        raise ValueError("jacobian must be two-dimensional.")
    if J.shape[1] != number_of_parameters:
        raise ValueError(
            "jacobian columns are incompatible with map_log_variance."
        )
    if J.shape[0] != weight_values.size:
        raise ValueError(
            "jacobian rows are incompatible with weights."
        )
    if np.any(~np.isfinite(J)):
        raise ValueError("jacobian contains non-finite values.")
    if np.any(~np.isfinite(weight_values)):
        raise ValueError("weights contain non-finite values.")
    if np.any(weight_values <= 0):
        raise ValueError("weights must be positive.")
    if regularization_matrix.shape != (
        number_of_parameters,
        number_of_parameters,
    ):
        raise ValueError(
            "regularization_matrix has the wrong shape."
        )
    if regularization_strength <= 0:
        raise ValueError(
            "regularization_strength must be positive."
        )
    if diagonal_jitter < 0:
        raise ValueError(
            "diagonal_jitter must be non-negative."
        )

    weighted_jacobian = weight_values[:, None] * J

    precision = (
        weighted_jacobian.T @ weighted_jacobian
        + regularization_strength
        * regularization_matrix.toarray()
    )

    if diagonal_jitter > 0:
        precision = (
            precision
            + diagonal_jitter
            * np.eye(number_of_parameters)
        )

    precision = 0.5 * (
        precision + precision.T
    )

    factor = cho_factor(
        precision,
        lower=True,
        check_finite=True,
    )
    covariance = cho_solve(
        factor,
        np.eye(number_of_parameters),
        check_finite=True,
    )
    covariance = 0.5 * (
        covariance + covariance.T
    )

    variances = np.diag(covariance)
    if np.any(variances <= 0):
        raise np.linalg.LinAlgError(
            "Laplace covariance has non-positive marginal variance."
        )

    standard_deviation = np.sqrt(variances)
    correlation = covariance / np.outer(
        standard_deviation,
        standard_deviation,
    )
    correlation = np.clip(
        correlation,
        -1.0,
        1.0,
    )

    return {
        "map_log_variance": map_surface,
        "map_vector": map_surface.reshape(
            -1,
            order="C",
        ),
        "precision": precision,
        "covariance": covariance,
        "standard_deviation": standard_deviation,
        "correlation": correlation,
        "precision_condition_number": float(
            np.linalg.cond(precision)
        ),
    }


def surface_uncertainty_summary(
    map_log_variance: np.ndarray,
    posterior_covariance: np.ndarray,
    maturities: np.ndarray,
    log_moneyness: np.ndarray,
    credibility: float = 0.95,
) -> pd.DataFrame:
    """Create pointwise Gaussian intervals for log variance and volatility.

    If a marginal log variance is Gaussian,

        u_j | y approximately Normal(m_j, v_j),

    then local volatility ``sigma_j = exp(u_j / 2)`` is lognormal. The
    volatility interval below is therefore obtained by transforming the
    Gaussian endpoints.
    """
    map_surface = np.asarray(
        map_log_variance,
        dtype=float,
    )
    covariance = np.asarray(
        posterior_covariance,
        dtype=float,
    )
    maturity_values = np.asarray(
        maturities,
        dtype=float,
    )
    x_values = np.asarray(
        log_moneyness,
        dtype=float,
    )

    expected_shape = (
        maturity_values.size,
        x_values.size,
    )
    if map_surface.shape != expected_shape:
        raise ValueError(
            f"map_log_variance has shape {map_surface.shape}; "
            f"expected {expected_shape}."
        )
    if covariance.shape != (
        map_surface.size,
        map_surface.size,
    ):
        raise ValueError(
            "posterior_covariance has the wrong shape."
        )
    if not 0 < credibility < 1:
        raise ValueError(
            "credibility must lie strictly between zero and one."
        )

    variances = np.diag(covariance)
    if np.any(variances < 0):
        raise ValueError(
            "posterior_covariance has negative marginal variance."
        )

    standard_deviation = np.sqrt(
        np.maximum(variances, 0.0)
    )
    z_value = float(
        norm.ppf(0.5 + credibility / 2.0)
    )

    map_vector = map_surface.reshape(
        -1,
        order="C",
    )
    lower_u = (
        map_vector
        - z_value * standard_deviation
    )
    upper_u = (
        map_vector
        + z_value * standard_deviation
    )

    volatility_median = np.exp(
        0.5 * map_vector
    )
    volatility_mean = np.exp(
        0.5 * map_vector
        + variances / 8.0
    )
    volatility_lower = np.exp(
        0.5 * lower_u
    )
    volatility_upper = np.exp(
        0.5 * upper_u
    )

    x_mesh, maturity_mesh = np.meshgrid(
        x_values,
        maturity_values,
    )

    return pd.DataFrame(
        {
            "maturity": maturity_mesh.reshape(-1),
            "log_moneyness": x_mesh.reshape(-1),
            "map_log_variance": map_vector,
            "sd_log_variance": standard_deviation,
            "lower_log_variance": lower_u,
            "upper_log_variance": upper_u,
            "volatility_median": volatility_median,
            "volatility_mean": volatility_mean,
            "volatility_lower": volatility_lower,
            "volatility_upper": volatility_upper,
            "volatility_interval_width": (
                volatility_upper
                - volatility_lower
            ),
        }
    )


def predictive_price_uncertainty(
    map_prices: np.ndarray,
    jacobian: np.ndarray,
    posterior_covariance: np.ndarray,
    noise_standard_deviation: np.ndarray,
    credibility: float = 0.95,
) -> dict[str, np.ndarray]:
    """Propagate posterior surface uncertainty into option-price uncertainty.

    The local linear approximation gives latent price covariance

        Sigma_price = J Sigma_u J.T.

    Observed-price predictive variance additionally includes quote noise:

        diag(Sigma_price) + noise_standard_deviation^2.
    """
    prices = np.asarray(
        map_prices,
        dtype=float,
    ).reshape(-1)
    J = np.asarray(jacobian, dtype=float)
    covariance = np.asarray(
        posterior_covariance,
        dtype=float,
    )
    noise = np.asarray(
        noise_standard_deviation,
        dtype=float,
    ).reshape(-1)

    if J.ndim != 2:
        raise ValueError("jacobian must be two-dimensional.")
    if J.shape[0] != prices.size:
        raise ValueError(
            "jacobian rows are incompatible with map_prices."
        )
    if noise.size != prices.size:
        raise ValueError(
            "noise_standard_deviation is incompatible with map_prices."
        )
    if covariance.shape != (
        J.shape[1],
        J.shape[1],
    ):
        raise ValueError(
            "posterior_covariance is incompatible with jacobian."
        )
    if np.any(noise <= 0):
        raise ValueError(
            "noise_standard_deviation must be positive."
        )
    if not 0 < credibility < 1:
        raise ValueError(
            "credibility must lie strictly between zero and one."
        )

    latent_covariance = (
        J @ covariance @ J.T
    )
    latent_covariance = 0.5 * (
        latent_covariance
        + latent_covariance.T
    )

    latent_variance = np.maximum(
        np.diag(latent_covariance),
        0.0,
    )
    observed_variance = (
        latent_variance + noise**2
    )

    latent_standard_deviation = np.sqrt(
        latent_variance
    )
    observed_standard_deviation = np.sqrt(
        observed_variance
    )

    z_value = float(
        norm.ppf(0.5 + credibility / 2.0)
    )

    return {
        "latent_covariance": latent_covariance,
        "latent_variance": latent_variance,
        "latent_standard_deviation": (
            latent_standard_deviation
        ),
        "observed_variance": observed_variance,
        "observed_standard_deviation": (
            observed_standard_deviation
        ),
        "latent_lower": (
            prices
            - z_value
            * latent_standard_deviation
        ),
        "latent_upper": (
            prices
            + z_value
            * latent_standard_deviation
        ),
        "observed_lower": (
            prices
            - z_value
            * observed_standard_deviation
        ),
        "observed_upper": (
            prices
            + z_value
            * observed_standard_deviation
        ),
    }
