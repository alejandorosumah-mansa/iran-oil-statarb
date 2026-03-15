"""Probability extraction for Brent price thresholds.

Converts predicted returns and residual uncertainty into threshold
exceedance probabilities under both Gaussian and Student-t assumptions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


def fit_student_t(residuals: np.ndarray) -> Tuple[float, float, float]:
    """Fit a Student-t distribution to OLS residuals via MLE.

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals (assumed zero-mean after OLS).

    Returns
    -------
    tuple[float, float, float]
        (df, loc, scale) - degrees of freedom, location, and scale
        parameters of the fitted t-distribution.
    """
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) < 5:
        raise ValueError("Need at least 5 finite residuals to fit t-distribution")
    df, loc, scale = stats.t.fit(residuals)
    return float(df), float(loc), float(scale)


def price_probabilities(
    current_price: float,
    predicted_ret: float,
    residual_std: float,
    thresholds: List[float],
    df_t: Optional[float] = None,
) -> Dict[float, Dict[str, float]]:
    """Compute P(price > threshold) under Gaussian and optionally Student-t.

    The predicted price is current_price * exp(predicted_ret). We model
    the log-return as N(predicted_ret, residual_std^2) or, when df_t is
    given, as t(df_t, predicted_ret, residual_std).

    Parameters
    ----------
    current_price : float
        Current Brent price level.
    predicted_ret : float
        Expected log return from the OLS model.
    residual_std : float
        Standard deviation of OLS residuals.
    thresholds : list[float]
        Price levels to compute exceedance probabilities for.
    df_t : float, optional
        Degrees of freedom for the Student-t model. If None, only
        Gaussian probabilities are returned.

    Returns
    -------
    dict[float, dict[str, float]]
        Mapping from threshold to {"gaussian": p, "student_t": p}.
    """
    result: Dict[float, Dict[str, float]] = {}
    for level in thresholds:
        # Required log return to reach the threshold
        required_ret = np.log(level / current_price)

        # Gaussian exceedance probability
        z = (required_ret - predicted_ret) / residual_std
        p_gauss = float(1.0 - stats.norm.cdf(z))

        probs: Dict[str, float] = {"gaussian": p_gauss}

        # Student-t exceedance probability
        if df_t is not None:
            t_val = (required_ret - predicted_ret) / residual_std
            p_t = float(1.0 - stats.t.cdf(t_val, df=df_t))
            probs["student_t"] = p_t

        result[level] = probs

    return result


def multi_horizon_probabilities(
    current_price: float,
    predicted_ret: float,
    residual_std: float,
    horizons: Optional[List[int]] = None,
    thresholds: Optional[List[float]] = None,
    df_t: Optional[float] = None,
) -> Dict[int, Dict[float, Dict[str, float]]]:
    """Compute threshold probabilities across multiple time horizons.

    Scales predicted return and volatility by horizon using the
    square-root-of-time rule:
        mu_h = predicted_ret * h
        sigma_h = residual_std * sqrt(h)

    Parameters
    ----------
    current_price : float
        Current Brent price level.
    predicted_ret : float
        1-day expected log return.
    residual_std : float
        1-day residual standard deviation.
    horizons : list[int], optional
        Forecast horizons in trading days (default: [1, 5, 10, 21]).
    thresholds : list[float], optional
        Price levels (default: [95, 100, 105, 110, 120]).
    df_t : float, optional
        Degrees of freedom for Student-t model.

    Returns
    -------
    dict[int, dict[float, dict[str, float]]]
        Nested mapping: horizon -> threshold -> {"gaussian": p, ...}.
    """
    if horizons is None:
        horizons = [1, 5, 10, 21]
    if thresholds is None:
        thresholds = [95.0, 100.0, 105.0, 110.0, 120.0]

    result: Dict[int, Dict[float, Dict[str, float]]] = {}
    for h in horizons:
        mu_h = predicted_ret * h
        sigma_h = residual_std * np.sqrt(h)
        result[h] = price_probabilities(
            current_price=current_price,
            predicted_ret=mu_h,
            residual_std=sigma_h,
            thresholds=thresholds,
            df_t=df_t,
        )
    return result
