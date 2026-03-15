"""Rolling OLS regression of Brent returns on lagged S3 returns.

Implements both full-sample and rolling-window OLS with diagnostic
statistics, signal generation, and strategy performance metrics.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_breuschpagan,
)
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from src.config import ROLLING_WINDOW, SIGNAL_THRESHOLD


def fit_static_ols(
    brent_ret: pd.Series,
    s3_ret_lag1: pd.Series,
) -> Dict:
    """Full-sample OLS: brent_ret ~ alpha + beta * s3_ret_lag1.

    Parameters
    ----------
    brent_ret : pd.Series
        Dependent variable (Brent log returns).
    s3_ret_lag1 : pd.Series
        Independent variable (S3 log returns, lagged 1 period).

    Returns
    -------
    dict
        Keys: beta, alpha, tstat, pvalue, r2, dw, jb, bp, lb.
        - dw: Durbin-Watson statistic for autocorrelation.
        - jb: Jarque-Bera p-value for residual normality.
        - bp: Breusch-Pagan p-value for heteroskedasticity.
        - lb: Ljung-Box p-value at lag 10.
    """
    X = sm.add_constant(s3_ret_lag1.values, prepend=True)
    y = brent_ret.values
    model = sm.OLS(y, X, missing="drop").fit()

    resid = model.resid
    dw = float(durbin_watson(resid))
    jb_stat, jb_pval, _, _ = jarque_bera(resid)
    bp_stat, bp_pval, _, _ = het_breuschpagan(resid, X)
    lb_result = acorr_ljungbox(resid, lags=[10], return_df=True)
    lb_pval = float(lb_result["lb_pvalue"].iloc[0])

    return {
        "alpha": float(model.params[0]),
        "beta": float(model.params[1]),
        "tstat": float(model.tvalues[1]),
        "pvalue": float(model.pvalues[1]),
        "r2": float(model.rsquared),
        "dw": dw,
        "jb": float(jb_pval),
        "bp": float(bp_pval),
        "lb": lb_pval,
    }


def fit_rolling_ols(
    brent_ret: pd.Series,
    s3_ret_lag1: pd.Series,
    window: int = ROLLING_WINDOW,
    threshold: float = SIGNAL_THRESHOLD,
) -> pd.DataFrame:
    """Rolling-window OLS with signal generation.

    For each window ending at time t, fit OLS on the trailing
    `window` observations and generate a directional signal for t+1.

    Parameters
    ----------
    brent_ret : pd.Series
        Brent log returns (DatetimeIndex).
    s3_ret_lag1 : pd.Series
        Lagged S3 log returns (DatetimeIndex).
    window : int
        Lookback window in trading days.
    threshold : float
        Minimum absolute predicted return to trigger a trade.

    Returns
    -------
    pd.DataFrame
        Columns: date, alpha, beta, predicted_ret, actual_ret,
        signal, strategy_ret.
    """
    dates = brent_ret.index
    n = len(dates)
    records = []

    for i in range(window, n):
        idx = slice(i - window, i)
        y_win = brent_ret.iloc[idx].values
        x_win = s3_ret_lag1.iloc[idx].values
        X_win = sm.add_constant(x_win, prepend=True)
        model = sm.OLS(y_win, X_win, missing="drop").fit()

        alpha = float(model.params[0])
        beta = float(model.params[1])
        pred = alpha + beta * s3_ret_lag1.iloc[i]
        actual = brent_ret.iloc[i]
        signal = generate_signal(
            s3_ret_lag1.iloc[i], alpha, beta, threshold
        )
        strat_ret = actual if signal == "LONG" else (
            -actual if signal == "SHORT" else 0.0
        )

        records.append(
            {
                "date": dates[i],
                "alpha": alpha,
                "beta": beta,
                "predicted_ret": float(pred),
                "actual_ret": float(actual),
                "signal": signal,
                "strategy_ret": float(strat_ret),
            }
        )

    return pd.DataFrame(records)


def generate_signal(
    s3_ret_today: float,
    alpha: float,
    beta: float,
    threshold: float = SIGNAL_THRESHOLD,
) -> str:
    """Generate a directional signal from the current model.

    Parameters
    ----------
    s3_ret_today : float
        Today's S3 return (to be used as tomorrow's lag-1 predictor).
    alpha : float
        OLS intercept.
    beta : float
        OLS slope on lagged S3 return.
    threshold : float
        Minimum absolute predicted return to generate a trade.

    Returns
    -------
    str
        One of "LONG", "SHORT", or "FLAT".
    """
    predicted = alpha + beta * s3_ret_today
    if predicted > threshold:
        return "LONG"
    elif predicted < -threshold:
        return "SHORT"
    return "FLAT"


def compute_strategy_metrics(
    strategy_rets: pd.Series,
    threshold: float = SIGNAL_THRESHOLD,
) -> Dict[str, float]:
    """Compute performance metrics for a return series.

    Parameters
    ----------
    strategy_rets : pd.Series
        Daily strategy returns.
    threshold : float
        Minimum return magnitude that counts as a trade.

    Returns
    -------
    dict
        Keys: sharpe, cumret, maxdd, calmar, directional_accuracy,
        n_trades.
    """
    rets = strategy_rets.dropna()
    if len(rets) == 0:
        return {
            "sharpe": 0.0,
            "cumret": 0.0,
            "maxdd": 0.0,
            "calmar": 0.0,
            "directional_accuracy": 0.0,
            "n_trades": 0,
        }

    # Annualized Sharpe (assuming 252 trading days)
    mean_ret = rets.mean()
    std_ret = rets.std()
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

    # Cumulative return
    cumret = float(np.expm1(rets.sum()))

    # Maximum drawdown
    cum = rets.cumsum()
    running_max = cum.cummax()
    drawdowns = cum - running_max
    maxdd = float(drawdowns.min())

    # Calmar ratio (annualized return / abs(max drawdown))
    ann_ret = mean_ret * 252
    calmar = (ann_ret / abs(maxdd)) if abs(maxdd) > 1e-10 else 0.0

    # Directional accuracy (fraction of positive-return days among trades)
    active = rets[rets.abs() > 1e-10]
    if len(active) > 0:
        directional_accuracy = float((active > 0).mean())
    else:
        directional_accuracy = 0.0

    n_trades = int((rets.abs() > 1e-10).sum())

    return {
        "sharpe": float(sharpe),
        "cumret": cumret,
        "maxdd": maxdd,
        "calmar": float(calmar),
        "directional_accuracy": directional_accuracy,
        "n_trades": n_trades,
    }


def model_selection_table(
    brent_ret: pd.Series,
    s3_ret: pd.Series,
    max_lag: int = 10,
) -> pd.DataFrame:
    """Compare OLS models across different S3 return lags using AIC/BIC.

    Parameters
    ----------
    brent_ret : pd.Series
        Brent log returns.
    s3_ret : pd.Series
        S3 log returns (unlagged).
    max_lag : int
        Maximum number of lags to test.

    Returns
    -------
    pd.DataFrame
        Columns: lag, aic, bic, r2, beta, pvalue.
    """
    records = []
    for lag in range(1, max_lag + 1):
        s3_lagged = s3_ret.shift(lag)
        valid = pd.DataFrame(
            {"y": brent_ret, "x": s3_lagged}
        ).dropna()
        if len(valid) < lag + 10:
            continue
        X = sm.add_constant(valid["x"].values, prepend=True)
        model = sm.OLS(valid["y"].values, X).fit()
        records.append(
            {
                "lag": lag,
                "aic": float(model.aic),
                "bic": float(model.bic),
                "r2": float(model.rsquared),
                "beta": float(model.params[1]),
                "pvalue": float(model.pvalues[1]),
            }
        )
    return pd.DataFrame(records)
