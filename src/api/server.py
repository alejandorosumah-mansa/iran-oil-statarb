"""
FastAPI backend for the Iran Oil StatArb dashboard.

Serves the static dashboard and exposes API endpoints for:
  - Aligned Brent / S3 data with Brownian bridge
  - Rolling OLS signal and implied price
  - Strategy performance metrics
  - Probability tables (Gaussian + Student-t)
  - Rolling model parameters

All computation is self-contained (no imports from src.models).
Data is cached in memory with a 5-minute TTL.

Usage:
    uvicorn src.api.server:app --reload --port 8000
"""

import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Data"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
CSV_PATH = DATA_DIR / "basket_level_monthly.csv"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASKET_CODE = "ADIT-S3"
ACTIVE_START = "2024-01-01"
ROLLING_WINDOW = 63
CACHE_TTL_SECONDS = 300  # 5 minutes
PRICE_THRESHOLDS = [85, 90, 95, 100, 105, 110, 115, 120]

# ---------------------------------------------------------------------------
# In-memory cache
# ---------------------------------------------------------------------------
_cache: dict = {
    "brent_raw": None,
    "brent_ts": 0.0,
    "computed": None,
    "computed_ts": 0.0,
}


# ===================================================================
# Brownian bridge utilities
# ===================================================================

def _detect_static_segments(series: pd.Series, min_run: int = 5,
                            tol: float = 1e-6) -> list[tuple[int, int]]:
    """Return (start, end) pairs of contiguous flat segments."""
    vals = series.values
    diffs = np.abs(np.diff(vals))
    is_static = np.concatenate([[False], diffs < tol])
    segments: list[tuple[int, int]] = []
    in_seg, start = False, 0
    for i in range(len(is_static)):
        if is_static[i] and not in_seg:
            in_seg, start = True, max(0, i - 1)
        elif not is_static[i] and in_seg:
            in_seg = False
            if i - start >= min_run:
                segments.append((start, i))
    if in_seg and len(is_static) - start >= min_run:
        segments.append((start, len(is_static) - 1))
    return segments


def _brownian_bridge(n: int, start_val: float, end_val: float,
                     sigma: float) -> np.ndarray:
    """Generate a Brownian bridge path of length n+1."""
    inc = np.random.normal(0, sigma, size=n)
    w = np.insert(np.cumsum(inc), 0, 0.0)
    t = np.arange(n + 1) / n
    return np.maximum(
        start_val + (end_val - start_val) * t + (w - t * w[-1]),
        start_val * 0.3,
    )


def _apply_brownian_bridge(series: pd.Series) -> pd.Series:
    """Replace static periods in S3 with Brownian-bridge noise."""
    np.random.seed(42)  # global seed once, matching research script
    rets = series.pct_change().dropna()
    active_sigma = rets[rets.abs() > 1e-8].std()
    segments = _detect_static_segments(series, min_run=5)
    bridged = series.copy()
    for a, b in segments:
        n = b - a
        bridge = _brownian_bridge(n, series.iloc[a], series.iloc[b],
                                  active_sigma * series.iloc[a])
        bridged.iloc[a: b + 1] = bridge
    return bridged


# ===================================================================
# Data loading
# ===================================================================

def _fetch_brent() -> pd.Series:
    """Download Brent prices from yfinance, with module-level cache."""
    now = time.time()
    if _cache["brent_raw"] is not None and now - _cache["brent_ts"] < CACHE_TTL_SECONDS:
        return _cache["brent_raw"]

    brent = yf.download("BZ=F", start=ACTIVE_START, progress=False)
    if isinstance(brent.columns, pd.MultiIndex):
        brent.columns = brent.columns.get_level_values(0)
    brent_close = brent["Close"].rename("brent")
    brent_close.index = pd.to_datetime(brent_close.index)

    _cache["brent_raw"] = brent_close
    _cache["brent_ts"] = now
    return brent_close


def _load_and_align() -> pd.DataFrame:
    """Load S3, fetch Brent, align on trading days, apply bridge."""
    now = time.time()
    if _cache["computed"] is not None and now - _cache["computed_ts"] < CACHE_TTL_SECONDS:
        return _cache["computed"]

    # S3 basket
    df = pd.read_csv(str(CSV_PATH), parse_dates=["rebalance_date"])
    s3 = df[df["basket_code"] == BASKET_CODE].copy()
    s3 = s3.set_index("rebalance_date").sort_index()
    s3 = s3[~s3.index.duplicated(keep="first")]
    s3 = s3[s3.index >= ACTIVE_START]
    s3_level = s3["basket_level"]

    # Brent
    brent_close = _fetch_brent()

    # Align: forward-fill basket onto Brent trading days
    s3_aligned = s3_level.reindex(brent_close.index, method="ffill")

    # Bridge
    s3_bridged = _apply_brownian_bridge(s3_aligned)

    data = pd.DataFrame({
        "brent": brent_close,
        "s3": s3_aligned,
        "s3_bridged": s3_bridged,
    }).dropna()

    data["brent_ret"] = data["brent"].pct_change()
    data["s3_ret"] = data["s3"].pct_change()
    data["s3_bridged_ret"] = data["s3_bridged"].pct_change()
    data = data.dropna()

    _cache["computed"] = data
    _cache["computed_ts"] = now
    return data


# ===================================================================
# Rolling OLS
# ===================================================================

def _rolling_ols(data: pd.DataFrame, window: int = ROLLING_WINDOW):
    """
    63-day rolling LAG-1 OLS: brent_ret(t) ~ alpha + beta * s3_bridged_ret(t-1).

    S3 leads Brent by one day.  The estimation window uses lag-1 pairs
    and the out-of-sample prediction is for the NEXT day's Brent return
    given today's S3 return.

    Returns arrays aligned to data.index.
    """
    brent_ret = data["brent_ret"].values
    s3_ret = data["s3_bridged_ret"].values
    n = len(brent_ret)

    alphas = np.full(n, np.nan)
    betas = np.full(n, np.nan)
    r2s = np.full(n, np.nan)
    preds = np.full(n, np.nan)
    residuals = np.full(n, np.nan)

    for t in range(window + 1, n):
        # Lag-1 pairs: Y[i] = brent_ret[i], X[i] = s3_ret[i-1]
        # Use window pairs ending at t-1 (so we predict t out-of-sample)
        Y = brent_ret[t - window: t]        # brent returns [t-W .. t-1]
        X_raw = s3_ret[t - window - 1: t - 1]  # s3 returns [t-W-1 .. t-2] (lagged by 1)
        X = np.column_stack([np.ones(window), X_raw])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        except np.linalg.LinAlgError:
            continue

        alphas[t] = beta[0]
        betas[t] = beta[1]

        Y_hat = X @ beta
        ss_res = np.sum((Y - Y_hat) ** 2)
        ss_tot = np.sum((Y - Y.mean()) ** 2)
        r2s[t] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Predict tomorrow's brent using today's s3 (lag-1 out-of-sample)
        pred = beta[0] + beta[1] * s3_ret[t - 1]
        preds[t] = pred
        residuals[t] = brent_ret[t] - pred

    return alphas, betas, r2s, preds, residuals


# ===================================================================
# Strategy helpers
# ===================================================================

def _compute_strategy(data: pd.DataFrame, preds: np.ndarray):
    """
    Long/Short/Flat strategy driven by rolling OLS prediction sign.
    Returns a dict of performance metrics.
    """
    brent_ret = data["brent_ret"].values
    s3_ret = data["s3_bridged_ret"].values
    dates = data.index
    n = len(brent_ret)

    valid = np.isfinite(preds)
    if valid.sum() == 0:
        return {}

    SIGNAL_THRESHOLD = 0.005  # 0.5%
    signals = np.where(preds > SIGNAL_THRESHOLD, 1.0,
                       np.where(preds < -SIGNAL_THRESHOLD, -1.0, 0.0))
    strat_rets = signals * brent_ret

    # Only count where we have predictions
    strat_valid = strat_rets[valid]
    brent_valid = brent_ret[valid]
    s3_valid = s3_ret[valid]
    dates_valid = dates[valid]
    signals_valid = signals[valid]
    preds_valid = preds[valid]

    total_days = len(strat_valid)
    if total_days == 0:
        return {}

    # Sharpe
    mean_r = np.mean(strat_valid)
    std_r = np.std(strat_valid, ddof=1) if total_days > 1 else 1e-9
    sharpe = mean_r / std_r * np.sqrt(252) if std_r > 0 else 0.0

    # Cumulative return
    cum = np.cumprod(1.0 + strat_valid) - 1.0
    cumulative_return = float(cum[-1])

    # Max drawdown
    wealth = np.cumprod(1.0 + strat_valid)
    peak = np.maximum.accumulate(wealth)
    drawdown = (wealth - peak) / peak
    max_dd = float(np.min(drawdown))

    # Calmar
    calmar = (mean_r * 252) / abs(max_dd) if abs(max_dd) > 1e-12 else 0.0

    # Directional accuracy
    dir_correct = np.sum(np.sign(preds_valid) == np.sign(brent_valid))
    dir_accuracy = float(dir_correct) / total_days

    # Number of trades (signal flips)
    n_trades = int(np.sum(np.abs(np.diff(signals_valid)) > 0))

    # Win/loss stats (only on active trading days where signal != 0)
    active_mask = signals_valid != 0
    active_rets = strat_valid[active_mask]
    if len(active_rets) > 0:
        wins = active_rets[active_rets > 0]
        losses = active_rets[active_rets < 0]
        win_rate = len(wins) / len(active_rets) if len(active_rets) > 0 else 0.0
        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = abs(float(np.sum(losses))) if len(losses) > 0 else 1e-9
        profit_factor = gross_profit / gross_loss if gross_loss > 1e-12 else 0.0
    else:
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        profit_factor = 0.0

    # Recent trades (last 20)
    recent_n = min(20, total_days)
    recent_trades = []
    for i in range(-recent_n, 0):
        sig_label = "LONG" if signals_valid[i] > 0 else ("SHORT" if signals_valid[i] < 0 else "FLAT")
        recent_trades.append({
            "date": str(dates_valid[i].date()),
            "brent_ret": round(float(brent_valid[i]), 6),
            "signal": sig_label,
            "strat_ret": round(float(strat_valid[i]), 6),
            "s3_ret": round(float(s3_valid[i]), 6),
        })

    # Full equity curve
    equity_curve = []
    for i in range(total_days):
        equity_curve.append({
            "date": str(dates_valid[i].date()),
            "cumret": round(float(cum[i]), 6),
        })

    return {
        "sharpe": round(sharpe, 4),
        "cumulative_return": round(cumulative_return, 6),
        "max_drawdown": round(max_dd, 6),
        "calmar": round(calmar, 4),
        "directional_accuracy": round(dir_accuracy, 4),
        "n_trades": n_trades,
        "total_days": total_days,
        "win_rate": round(win_rate, 4),
        "avg_win": round(avg_win, 6),
        "avg_loss": round(avg_loss, 6),
        "profit_factor": round(profit_factor, 4),
        "recent_trades": recent_trades,
        "equity_curve": equity_curve,
    }


# ===================================================================
# Probability helpers
# ===================================================================

def _compute_probabilities(data: pd.DataFrame, residuals: np.ndarray):
    """
    Compute P(Brent > X) for each threshold under Gaussian and Student-t.
    Also compute multi-horizon tables.
    """
    brent_ret = data["brent_ret"].values
    current_brent = float(data["brent"].iloc[-1])

    valid_resid = residuals[np.isfinite(residuals)]
    if len(valid_resid) < 10:
        return {}

    mu = float(np.mean(valid_resid))
    sigma = float(np.std(valid_resid, ddof=1))

    # Fit Student-t
    t_df, t_loc, t_scale = stats.t.fit(valid_resid)

    # Single-day Gaussian
    gaussian_probs = {}
    for X in PRICE_THRESHOLDS:
        req_ret = (X / current_brent) - 1.0
        z = (req_ret - mu) / sigma if sigma > 0 else float("inf")
        gaussian_probs[str(X)] = round(float(1.0 - stats.norm.cdf(z)), 6)

    # Single-day Student-t
    student_probs = {}
    for X in PRICE_THRESHOLDS:
        req_ret = (X / current_brent) - 1.0
        p = float(1.0 - stats.t.cdf(req_ret, df=t_df, loc=t_loc, scale=t_scale))
        student_probs[str(X)] = round(p, 6)

    # Multi-horizon (Gaussian random-walk scaling)
    horizons = [1, 5, 10, 21]
    multi_horizon: dict[str, dict[str, float]] = {}
    for h in horizons:
        mu_h = mu * h
        sigma_h = sigma * np.sqrt(h)
        h_probs = {}
        for X in PRICE_THRESHOLDS:
            req_ret = (X / current_brent) - 1.0
            z_h = (req_ret - mu_h) / sigma_h if sigma_h > 0 else float("inf")
            h_probs[str(X)] = round(float(1.0 - stats.norm.cdf(z_h)), 6)
        multi_horizon[str(h)] = h_probs

    return {
        "gaussian": gaussian_probs,
        "student_t": student_probs,
        "df": round(float(t_df), 4),
        "multi_horizon": multi_horizon,
    }


# ===================================================================
# FastAPI application
# ===================================================================

app = FastAPI(
    title="Iran Oil StatArb API",
    description="Backend for the S3/Brent statistical arbitrage dashboard",
    version="1.0.0",
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------
# Startup: pre-load data so first request is fast
# -------------------------------------------------------------------

@app.on_event("startup")
async def startup_preload():
    """Pre-load and cache data at server start."""
    try:
        data = _load_and_align()
        _rolling_ols(data)
        print(f"[startup] Data loaded: {len(data)} rows, "
              f"{data.index[0].date()} to {data.index[-1].date()}")
    except Exception as exc:
        print(f"[startup] Pre-load failed (will retry on first request): {exc}")


# -------------------------------------------------------------------
# GET /api/data
# -------------------------------------------------------------------

@app.get("/api/data")
async def get_data():
    """Return the full aligned dataset."""
    data = _load_and_align()
    dates_str = [str(d.date()) for d in data.index]
    return JSONResponse({
        "dates": dates_str,
        "brent_price": [round(float(v), 4) for v in data["brent"].values],
        "s3_level": [round(float(v), 4) for v in data["s3"].values],
        "s3_bridged": [round(float(v), 4) for v in data["s3_bridged"].values],
        "brent_ret": [round(float(v), 6) for v in data["brent_ret"].values],
        "s3_ret": [round(float(v), 6) for v in data["s3_ret"].values],
        "s3_bridged_ret": [round(float(v), 6) for v in data["s3_bridged_ret"].values],
    })


# -------------------------------------------------------------------
# GET /api/signal
# -------------------------------------------------------------------

@app.get("/api/signal")
async def get_signal():
    """Return the current model state from rolling OLS."""
    data = _load_and_align()
    alphas, betas, r2s, preds, residuals = _rolling_ols(data)

    current_brent = float(data["brent"].iloc[-1])
    current_s3 = float(data["s3_bridged"].iloc[-1])
    s3_ret_today = float(data["s3_bridged_ret"].iloc[-1])
    last_date = str(data.index[-1].date())

    # Latest valid rolling parameters
    last_idx = len(data) - 1
    rolling_alpha = float(alphas[last_idx]) if np.isfinite(alphas[last_idx]) else 0.0
    rolling_beta = float(betas[last_idx]) if np.isfinite(betas[last_idx]) else 0.0
    rolling_r2 = float(r2s[last_idx]) if np.isfinite(r2s[last_idx]) else 0.0

    predicted_ret = float(preds[last_idx]) if np.isfinite(preds[last_idx]) else 0.0
    implied_brent = current_brent * (1.0 + predicted_ret)

    # Signal classification
    if predicted_ret > 0.005:
        signal = "LONG"
    elif predicted_ret < -0.005:
        signal = "SHORT"
    else:
        signal = "FLAT"

    # Confidence interval from residual std
    valid_resid = residuals[np.isfinite(residuals)]
    sigma = float(np.std(valid_resid, ddof=1)) if len(valid_resid) > 1 else 0.02
    confidence_low = current_brent * (1.0 + predicted_ret - 1.96 * sigma)
    confidence_high = current_brent * (1.0 + predicted_ret + 1.96 * sigma)

    return JSONResponse({
        "current_brent": round(current_brent, 4),
        "current_s3": round(current_s3, 4),
        "s3_ret_today": round(s3_ret_today, 6),
        "predicted_ret": round(predicted_ret, 6),
        "implied_brent": round(implied_brent, 4),
        "signal": signal,
        "confidence_low": round(confidence_low, 4),
        "confidence_high": round(confidence_high, 4),
        "rolling_beta": round(rolling_beta, 6),
        "rolling_alpha": round(rolling_alpha, 6),
        "rolling_r2": round(rolling_r2, 6),
        "last_updated": last_date,
    })


# -------------------------------------------------------------------
# GET /api/strategy
# -------------------------------------------------------------------

@app.get("/api/strategy")
async def get_strategy():
    """Return strategy performance metrics."""
    data = _load_and_align()
    _, _, _, preds, _ = _rolling_ols(data)
    result = _compute_strategy(data, preds)
    return JSONResponse(result)


# -------------------------------------------------------------------
# GET /api/probabilities
# -------------------------------------------------------------------

@app.get("/api/probabilities")
async def get_probabilities():
    """Return probability tables under Gaussian and Student-t."""
    data = _load_and_align()
    _, _, _, _, residuals = _rolling_ols(data)
    result = _compute_probabilities(data, residuals)
    return JSONResponse(result)


# -------------------------------------------------------------------
# GET /api/rolling
# -------------------------------------------------------------------

@app.get("/api/rolling")
async def get_rolling():
    """Return rolling model parameters over time."""
    data = _load_and_align()
    alphas, betas, r2s, _, _ = _rolling_ols(data)

    valid = np.isfinite(alphas)
    dates_str = [str(d.date()) for d in data.index[valid]]

    return JSONResponse({
        "dates": dates_str,
        "beta": [round(float(v), 6) for v in betas[valid]],
        "r2": [round(float(v), 6) for v in r2s[valid]],
        "alpha": [round(float(v), 6) for v in alphas[valid]],
    })


# -------------------------------------------------------------------
# Static files: serve dashboard at /
# -------------------------------------------------------------------
if DASHBOARD_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")
