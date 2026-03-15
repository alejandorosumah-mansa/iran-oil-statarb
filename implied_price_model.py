"""
Lagged Predictive Model: S3 → Brent (Next-Day)
================================================
Exploits the lead-lag relationship where S3 basket moves
predict next-day Brent crude oil moves.

Key finding from historical_correlation.py:
  Cross-correlation at lag +1: corr=0.12, p=0.003
  S3 Granger-causes CL at borderline significance (p=0.051)

Sections:
  A. Single-lag optimization (returns, k=1..10)
  B. Multi-lag models (AIC/BIC selection, up to 5 lags)
  C. Level models + level-change hybrids
  D. Best model selection with full diagnostics
  E. Today's implied Brent price with prediction interval
  F. Rolling lagged model (63-day window)
  G. Multi-horizon predictions
  H. Probability extraction

Usage: python3 implied_price_model.py
"""

import pandas as pd
import numpy as np
from scipy import stats
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────

DATA_DIR = "Data"
BASKET_CODE = "ADIT-S3"
ACTIVE_START = "2024-01-01"
MAX_SINGLE_LAG = 10
MAX_MULTI_LAG = 5
ROLLING_WINDOW = 63
TRAIN_FRAC = 0.80
HORIZONS = [1, 2, 3, 5, 10]
PRICE_THRESHOLDS = [85, 90, 95, 100, 105, 110, 115, 120]

np.random.seed(42)


# ═════════════════════════════════════════════════════════════════════
# BROWNIAN BRIDGE (fill static S3 periods with synthetic noise)
# ═════════════════════════════════════════════════════════════════════

def detect_static_segments(series, min_run=5, tol=1e-6):
    """Detect contiguous runs where the series doesn't move."""
    vals = series.values
    diffs = np.abs(np.diff(vals))
    is_static = np.concatenate([[False], diffs < tol])
    segments, in_seg, start = [], False, 0
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


def brownian_bridge(n, start_val, end_val, sigma):
    """Generate a Brownian bridge from start_val to end_val with given volatility."""
    inc = np.random.normal(0, sigma, size=n)
    w = np.insert(np.cumsum(inc), 0, 0.0)
    t = np.arange(n + 1) / n
    return np.maximum(start_val + (end_val - start_val) * t + (w - t * w[-1]),
                      start_val * 0.3)


def apply_brownian_bridge(series):
    """Replace static periods in S3 with Brownian bridge noise."""
    rets = series.pct_change().dropna()
    active_sigma = rets[rets.abs() > 1e-8].std()
    segments = detect_static_segments(series, min_run=5)
    bridged = series.copy()
    for a, b in segments:
        n = b - a
        bridge = brownian_bridge(n, series.iloc[a], series.iloc[b],
                                 active_sigma * series.iloc[a])
        bridged.iloc[a:b + 1] = bridge
    return bridged, segments


# ═════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════

def load_data():
    """Load S3 basket and Brent, align, apply Brownian bridge, compute transforms."""
    # S3 basket
    df = pd.read_csv(f"{DATA_DIR}/basket_level_monthly.csv",
                     parse_dates=["rebalance_date"])
    s3 = df[df["basket_code"] == BASKET_CODE].copy()
    s3 = s3.set_index("rebalance_date").sort_index()
    s3 = s3[~s3.index.duplicated(keep="first")]
    s3 = s3[s3.index >= ACTIVE_START]
    s3_level = s3["basket_level"]

    # Brent crude (BZ=F)
    brent = yf.download("BZ=F", start=ACTIVE_START, progress=False)
    if isinstance(brent.columns, pd.MultiIndex):
        brent.columns = brent.columns.get_level_values(0)
    brent = brent["Close"].rename("brent")
    brent.index = pd.to_datetime(brent.index)

    # Align: forward-fill basket to Brent trading days
    s3_aligned = s3_level.reindex(brent.index, method="ffill")

    # Apply Brownian bridge to fill static periods
    s3_bridged, segments = apply_brownian_bridge(s3_aligned)
    n_static = sum(b - a for a, b in segments)
    print(f"  Brownian bridge: {len(segments)} static segments, {n_static} days filled")

    data = pd.DataFrame({
        "brent": brent,
        "s3": s3_bridged,
        "s3_raw": s3_aligned,  # keep raw for reference
    }).dropna()

    # Compute returns and differences (from bridged S3)
    data["brent_ret"] = data["brent"].pct_change()
    data["s3_ret"] = data["s3"].pct_change()
    data["brent_log_ret"] = np.log(data["brent"] / data["brent"].shift(1))
    data["s3_log_ret"] = np.log(data["s3"] / data["s3"].shift(1))
    data["s3_diff"] = data["s3"].diff()
    data["brent_diff"] = data["brent"].diff()

    return data.dropna()


def print_header(title):
    print(f"\n{'═' * 75}")
    print(f"  {title}")
    print(f"{'═' * 75}")


def print_sub(title):
    print(f"\n  ── {title} ──")


def ols_multi(X, Y):
    """
    Multivariate OLS: Y = X @ beta (X should include intercept column).
    Returns dict with coefficients, R², adjusted R², AIC, BIC, etc.
    """
    n, k = X.shape
    if n < k + 2:
        return None

    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    Y_hat = X @ beta
    resid = Y - Y_hat
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((Y - Y.mean()) ** 2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    adj_r_sq = 1 - (1 - r_sq) * (n - 1) / (n - k) if n > k else 0

    # Information criteria
    sigma2_mle = ss_res / n
    log_lik = -n / 2 * (np.log(2 * np.pi) + np.log(sigma2_mle) + 1)
    aic = -2 * log_lik + 2 * k
    bic = -2 * log_lik + k * np.log(n)

    # RMSE
    rmse = np.sqrt(ss_res / n)

    # Standard errors and t-stats for each coefficient
    if n > k:
        mse = ss_res / (n - k)
        try:
            cov_beta = mse * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(cov_beta))
        except np.linalg.LinAlgError:
            se = np.full(k, np.nan)
    else:
        se = np.full(k, np.nan)

    t_stats = beta / se
    p_values = np.array([2 * stats.t.sf(abs(t), df=max(n - k, 1)) for t in t_stats])

    # Durbin-Watson
    dw = np.sum(np.diff(resid) ** 2) / ss_res if ss_res > 0 else np.nan

    return {
        "beta": beta,
        "se": se,
        "t_stats": t_stats,
        "p_values": p_values,
        "r_sq": r_sq,
        "adj_r_sq": adj_r_sq,
        "aic": aic,
        "bic": bic,
        "rmse": rmse,
        "dw": dw,
        "resid": resid,
        "fitted": Y_hat,
        "n": n,
        "k": k,
        "ss_res": ss_res,
        "log_lik": log_lik,
    }


# ═════════════════════════════════════════════════════════════════════
# SECTION A: SINGLE-LAG OPTIMIZATION
# ═════════════════════════════════════════════════════════════════════

def run_single_lag_models(data):
    """Test CL_ret(t) = a + b * S3_ret(t-k) for k=1..MAX_SINGLE_LAG."""
    print_header("A. SINGLE-LAG RETURN MODELS")
    print(f"  Model: Brent_ret(t) = α + β · S3_ret(t−k)")
    print(f"  Testing lags k = 1 to {MAX_SINGLE_LAG}\n")

    results = {}
    brent_ret = data["brent_ret"].values
    s3_ret = data["s3_ret"].values
    n_total = len(brent_ret)

    print(f"  {'Lag':>3}  {'β':>10}  {'t-stat':>8}  {'p-value':>10}  {'R²':>8}  {'Adj R²':>8}  {'AIC':>10}  {'BIC':>10}  {'RMSE':>8}")
    print(f"  {'─' * 3}  {'─' * 10}  {'─' * 8}  {'─' * 10}  {'─' * 8}  {'─' * 8}  {'─' * 10}  {'─' * 10}  {'─' * 8}")

    for k in range(1, MAX_SINGLE_LAG + 1):
        Y = brent_ret[k:]
        X_lag = s3_ret[:-k] if k < n_total else s3_ret[:1]
        # Align: Y[i] corresponds to S3_ret at position i (which is t-k relative to Y)
        n = min(len(Y), len(X_lag))
        Y = Y[-n:]
        X_lag = X_lag[-n:]

        X_mat = np.column_stack([np.ones(n), X_lag])
        res = ols_multi(X_mat, Y)
        if res is None:
            continue

        results[k] = res
        sig = "***" if res["p_values"][1] < 0.001 else "**" if res["p_values"][1] < 0.01 else "*" if res["p_values"][1] < 0.05 else "." if res["p_values"][1] < 0.1 else ""
        print(f"  {k:>3}  {res['beta'][1]:>10.6f}  {res['t_stats'][1]:>8.3f}  {res['p_values'][1]:>10.4f}  "
              f"{res['r_sq']:>8.5f}  {res['adj_r_sq']:>8.5f}  {res['aic']:>10.1f}  {res['bic']:>10.1f}  {res['rmse']:>8.5f}  {sig}")

    if results:
        best_k = min(results, key=lambda k: results[k]["aic"])
        print(f"\n  ▸ Best single lag by AIC: k={best_k} (AIC={results[best_k]['aic']:.1f})")
        best_bic_k = min(results, key=lambda k: results[k]["bic"])
        print(f"  ▸ Best single lag by BIC: k={best_bic_k} (BIC={results[best_bic_k]['bic']:.1f})")

    return results


# ═════════════════════════════════════════════════════════════════════
# SECTION B: MULTI-LAG MODELS
# ═════════════════════════════════════════════════════════════════════

def run_multi_lag_models(data):
    """
    Test multi-lag models with AIC/BIC selection.
    1. Pure S3 lags: CL_ret(t) = a + Σ b_k * S3_ret(t-k)
    2. With own-lag (VAR-style): CL_ret(t) = a + c * CL_ret(t-1) + Σ b_k * S3_ret(t-k)
    """
    print_header("B. MULTI-LAG MODELS (AIC/BIC SELECTION)")

    brent_ret = data["brent_ret"].values
    s3_ret = data["s3_ret"].values

    all_results = {}

    # B1: Pure S3 multi-lag
    print_sub("B1: Pure S3 Lags")
    print(f"  Model: Brent_ret(t) = α + Σ β_k · S3_ret(t−k), k=1..K\n")
    print(f"  {'K':>3}  {'R²':>8}  {'Adj R²':>8}  {'AIC':>10}  {'BIC':>10}  {'RMSE':>8}  {'DW':>6}")
    print(f"  {'─' * 3}  {'─' * 8}  {'─' * 8}  {'─' * 10}  {'─' * 10}  {'─' * 8}  {'─' * 6}")

    for K in range(1, MAX_MULTI_LAG + 1):
        n = len(brent_ret) - K
        if n < K + 5:
            continue
        Y = brent_ret[K:]
        X_cols = [np.ones(n)]
        for k in range(1, K + 1):
            lag_col = s3_ret[K - k: K - k + n]
            X_cols.append(lag_col)
        X_mat = np.column_stack(X_cols)
        res = ols_multi(X_mat, Y)
        if res is None:
            continue
        res["model_type"] = f"S3_lags_1_to_{K}"
        res["K"] = K
        all_results[f"pure_s3_K{K}"] = res
        print(f"  {K:>3}  {res['r_sq']:>8.5f}  {res['adj_r_sq']:>8.5f}  {res['aic']:>10.1f}  {res['bic']:>10.1f}  {res['rmse']:>8.5f}  {res['dw']:>6.3f}")

    # B2: VAR-style (own lag + S3 lags)
    print_sub("B2: VAR-Style (Own Lag + S3 Lags)")
    print(f"  Model: Brent_ret(t) = α + γ · Brent_ret(t−1) + Σ β_k · S3_ret(t−k)\n")
    print(f"  {'K':>3}  {'γ':>10}  {'R²':>8}  {'Adj R²':>8}  {'AIC':>10}  {'BIC':>10}  {'RMSE':>8}")
    print(f"  {'─' * 3}  {'─' * 10}  {'─' * 8}  {'─' * 8}  {'─' * 10}  {'─' * 10}  {'─' * 8}")

    for K in range(1, MAX_MULTI_LAG + 1):
        start = max(K, 1)
        n = len(brent_ret) - start
        if n < K + 5:
            continue
        Y = brent_ret[start:]
        X_cols = [np.ones(n)]
        # Own lag
        own_lag = brent_ret[start - 1: start - 1 + n]
        X_cols.append(own_lag)
        # S3 lags
        for k in range(1, K + 1):
            lag_col = s3_ret[start - k: start - k + n]
            X_cols.append(lag_col)
        X_mat = np.column_stack(X_cols)
        res = ols_multi(X_mat, Y)
        if res is None:
            continue
        res["model_type"] = f"VAR_K{K}"
        res["K"] = K
        all_results[f"var_K{K}"] = res
        print(f"  {K:>3}  {res['beta'][1]:>10.6f}  {res['r_sq']:>8.5f}  {res['adj_r_sq']:>8.5f}  "
              f"{res['aic']:>10.1f}  {res['bic']:>10.1f}  {res['rmse']:>8.5f}")

    if all_results:
        best_key = min(all_results, key=lambda k: all_results[k]["aic"])
        print(f"\n  ▸ Best multi-lag model by AIC: {best_key} (AIC={all_results[best_key]['aic']:.1f})")
        best_bic_key = min(all_results, key=lambda k: all_results[k]["bic"])
        print(f"  ▸ Best multi-lag model by BIC: {best_bic_key} (BIC={all_results[best_bic_key]['bic']:.1f})")

    return all_results


# ═════════════════════════════════════════════════════════════════════
# SECTION C: LEVEL MODELS + HYBRIDS
# ═════════════════════════════════════════════════════════════════════

def run_level_models(data):
    """
    Test level-based lagged models:
    C1: CL(t) = a + b * S3(t-k)  for k=1..5
    C2: CL(t) = a + b * S3(t-1) + c * ΔS3(t-1)  (level + momentum)
    C3: ΔCL(t) = a + b * ΔS3(t-1) + c * [S3(t-1) - S3_mean]  (error-correction style)
    """
    print_header("C. LEVEL MODELS & HYBRIDS")

    brent = data["brent"].values
    s3 = data["s3"].values
    s3_diff = data["s3_diff"].values
    brent_diff = data["brent_diff"].values

    all_results = {}

    # C1: Lagged levels
    print_sub("C1: Lagged Level Regression")
    print(f"  Model: Brent(t) = α + β · S3(t−k)\n")
    print(f"  {'Lag':>3}  {'β':>10}  {'t-stat':>8}  {'p-value':>10}  {'R²':>8}  {'AIC':>10}  {'DW':>6}")
    print(f"  {'─' * 3}  {'─' * 10}  {'─' * 8}  {'─' * 10}  {'─' * 8}  {'─' * 10}  {'─' * 6}")

    for k in range(1, 6):
        n = len(brent) - k
        Y = brent[k:]
        X_lag = s3[:-k] if k < len(s3) else s3[:1]
        n = min(len(Y), len(X_lag))
        Y = Y[-n:]
        X_lag = X_lag[-n:]
        X_mat = np.column_stack([np.ones(n), X_lag])
        res = ols_multi(X_mat, Y)
        if res is None:
            continue
        res["model_type"] = f"level_lag{k}"
        all_results[f"level_lag{k}"] = res
        sig = "***" if res["p_values"][1] < 0.001 else "**" if res["p_values"][1] < 0.01 else "*" if res["p_values"][1] < 0.05 else ""
        print(f"  {k:>3}  {res['beta'][1]:>10.6f}  {res['t_stats'][1]:>8.3f}  {res['p_values'][1]:>10.4f}  "
              f"{res['r_sq']:>8.5f}  {res['aic']:>10.1f}  {res['dw']:>6.3f}  {sig}")

    print(f"\n  ⚠ Level regressions have high R² due to co-trending, but DW ≈ 0 = severe autocorrelation")
    print(f"    These are spurious — non-stationary series. Use with caution (Granger-Newbold problem).")

    # C2: Level + momentum hybrid
    print_sub("C2: Level + Momentum Hybrid")
    print(f"  Model: Brent(t) = α + β · S3(t−1) + γ · ΔS3(t−1)\n")

    n = len(brent) - 1
    Y = brent[1:]
    X_level = s3[:-1]
    X_mom = s3_diff[1:]  # ΔS3(t-1) = s3_diff at position t-1
    n = min(len(Y), len(X_level), len(X_mom))
    Y, X_level, X_mom = Y[-n:], X_level[-n:], X_mom[-n:]
    X_mat = np.column_stack([np.ones(n), X_level, X_mom])
    res = ols_multi(X_mat, Y)
    if res is not None:
        res["model_type"] = "level_momentum"
        all_results["level_momentum"] = res
        print(f"  α = {res['beta'][0]:.4f} (se={res['se'][0]:.4f})")
        print(f"  β (level)    = {res['beta'][1]:.6f} (t={res['t_stats'][1]:.3f}, p={res['p_values'][1]:.4f})")
        print(f"  γ (momentum) = {res['beta'][2]:.6f} (t={res['t_stats'][2]:.3f}, p={res['p_values'][2]:.4f})")
        print(f"  R² = {res['r_sq']:.5f}, Adj R² = {res['adj_r_sq']:.5f}, DW = {res['dw']:.3f}")

    # C3: Error-correction style
    print_sub("C3: Error-Correction Style")
    print(f"  Model: ΔBrent(t) = α + β · ΔS3(t−1) + γ · [S3(t−1) − S3_mean]\n")

    s3_mean = np.mean(s3)
    n = len(brent_diff) - 1
    Y = brent_diff[1:]
    X_ds3 = s3_diff[:-1]
    X_dev = s3[:-1] - s3_mean
    n = min(len(Y), len(X_ds3), len(X_dev))
    Y, X_ds3, X_dev = Y[-n:], X_ds3[-n:], X_dev[-n:]
    # Remove any NaNs
    valid = np.isfinite(Y) & np.isfinite(X_ds3) & np.isfinite(X_dev)
    Y, X_ds3, X_dev = Y[valid], X_ds3[valid], X_dev[valid]
    X_mat = np.column_stack([np.ones(len(Y)), X_ds3, X_dev])
    res = ols_multi(X_mat, Y)
    if res is not None:
        res["model_type"] = "error_correction"
        all_results["error_correction"] = res
        print(f"  α = {res['beta'][0]:.6f}")
        print(f"  β (ΔS3)       = {res['beta'][1]:.6f} (t={res['t_stats'][1]:.3f}, p={res['p_values'][1]:.4f})")
        print(f"  γ (deviation) = {res['beta'][2]:.6f} (t={res['t_stats'][2]:.3f}, p={res['p_values'][2]:.4f})")
        print(f"  R² = {res['r_sq']:.5f}, DW = {res['dw']:.3f}")

    return all_results


# ═════════════════════════════════════════════════════════════════════
# SECTION D: BEST MODEL SELECTION + DIAGNOSTICS
# ═════════════════════════════════════════════════════════════════════

def select_best_model(data, single_results, multi_results, level_results):
    """
    Compare all return-based models (not level — those are spurious).
    Pick by AIC, report full diagnostics, run out-of-sample test.
    """
    print_header("D. BEST MODEL SELECTION & DIAGNOSTICS")

    # Collect all return-based models
    candidates = {}
    for k, res in single_results.items():
        candidates[f"single_lag{k}"] = res
    for key, res in multi_results.items():
        candidates[key] = res
    # Include error-correction from level models (it uses differences, so stationary)
    if "error_correction" in level_results:
        candidates["error_correction"] = level_results["error_correction"]

    if not candidates:
        print("  No valid models to compare.")
        return None, None

    # Rank by AIC
    print_sub("D1: Model Ranking (by AIC)")
    ranked = sorted(candidates.items(), key=lambda x: x[1]["aic"])
    print(f"\n  {'Rank':>4}  {'Model':<25}  {'AIC':>10}  {'BIC':>10}  {'R²':>8}  {'Adj R²':>8}  {'RMSE':>8}")
    print(f"  {'─' * 4}  {'─' * 25}  {'─' * 10}  {'─' * 10}  {'─' * 8}  {'─' * 8}  {'─' * 8}")
    for i, (name, res) in enumerate(ranked[:10]):
        marker = " ◀ BEST" if i == 0 else ""
        print(f"  {i + 1:>4}  {name:<25}  {res['aic']:>10.1f}  {res['bic']:>10.1f}  "
              f"{res['r_sq']:>8.5f}  {res['adj_r_sq']:>8.5f}  {res['rmse']:>8.5f}{marker}")

    best_name, best_res = ranked[0]
    print(f"\n  ▸ Selected model: {best_name}")

    # Full diagnostics on best model
    print_sub("D2: Full Diagnostics — " + best_name)

    resid = best_res["resid"]
    n = best_res["n"]
    k = best_res["k"]

    # Coefficients
    labels = ["const"] + [f"x{i}" for i in range(1, k)]
    print(f"\n  {'Coeff':<10}  {'Value':>12}  {'Std Err':>10}  {'t-stat':>8}  {'p-value':>10}  {'Sig':>4}")
    print(f"  {'─' * 10}  {'─' * 12}  {'─' * 10}  {'─' * 8}  {'─' * 10}  {'─' * 4}")
    for j in range(k):
        sig = "***" if best_res["p_values"][j] < 0.001 else "**" if best_res["p_values"][j] < 0.01 else "*" if best_res["p_values"][j] < 0.05 else ""
        print(f"  {labels[j]:<10}  {best_res['beta'][j]:>12.8f}  {best_res['se'][j]:>10.6f}  "
              f"{best_res['t_stats'][j]:>8.3f}  {best_res['p_values'][j]:>10.4f}  {sig:>4}")

    print(f"\n  R² = {best_res['r_sq']:.6f}")
    print(f"  Adjusted R² = {best_res['adj_r_sq']:.6f}")
    print(f"  RMSE = {best_res['rmse']:.6f}")
    print(f"  Log-likelihood = {best_res['log_lik']:.1f}")
    print(f"  AIC = {best_res['aic']:.1f}")
    print(f"  BIC = {best_res['bic']:.1f}")

    # Durbin-Watson
    dw = best_res["dw"]
    dw_interp = "positive autocorrelation" if dw < 1.5 else "no autocorrelation" if dw < 2.5 else "negative autocorrelation"
    print(f"  Durbin-Watson = {dw:.4f} ({dw_interp})")

    # Jarque-Bera normality test
    jb_stat, jb_p = stats.jarque_bera(resid)
    print(f"  Jarque-Bera = {jb_stat:.2f} (p={jb_p:.4f}) — {'non-normal' if jb_p < 0.05 else 'normal'}")

    # Residual skewness and kurtosis
    print(f"  Residual skewness = {stats.skew(resid):.4f}")
    print(f"  Residual kurtosis = {stats.kurtosis(resid):.4f} (excess)")

    # Breusch-Pagan heteroskedasticity (manual)
    print_sub("D3: Breusch-Pagan Heteroskedasticity Test")
    resid_sq = resid ** 2
    fitted = best_res["fitted"]
    X_bp = np.column_stack([np.ones(len(fitted)), fitted])
    bp_res = ols_multi(X_bp, resid_sq)
    if bp_res is not None:
        bp_f = (bp_res["r_sq"] * n) / 2  # LM = n * R² / k (approx chi2 with k df)
        bp_p = 1 - stats.chi2.cdf(bp_f, df=1)
        print(f"  LM statistic = {bp_f:.3f}, p-value = {bp_p:.4f}")
        print(f"  {'Heteroskedastic' if bp_p < 0.05 else 'Homoskedastic'} residuals")

    # Ljung-Box test on residuals (autocorrelation at lags 1-10)
    print_sub("D4: Ljung-Box Autocorrelation Test (residuals)")
    acf_vals = []
    for lag in range(1, 11):
        if len(resid) > lag:
            r = np.corrcoef(resid[lag:], resid[:-lag])[0, 1]
            acf_vals.append(r)
        else:
            acf_vals.append(0)
    Q = n * (n + 2) * sum(r ** 2 / (n - lag) for lag, r in enumerate(acf_vals, 1))
    lb_p = 1 - stats.chi2.cdf(Q, df=10)
    print(f"  Q(10) = {Q:.2f}, p-value = {lb_p:.4f}")
    print(f"  {'Significant residual autocorrelation' if lb_p < 0.05 else 'No significant residual autocorrelation'}")

    # Out-of-sample test
    print_sub("D5: Out-of-Sample Validation")
    best_model_info = run_oos_test(data, best_name)

    return best_name, best_res, best_model_info


def run_oos_test(data, model_name):
    """
    Train on first TRAIN_FRAC, predict on remainder.
    Returns dict with OOS metrics.
    """
    brent_ret = data["brent_ret"].values
    s3_ret = data["s3_ret"].values
    brent = data["brent"].values
    s3 = data["s3"].values
    s3_diff = data["s3_diff"].values

    n = len(brent_ret)
    split = int(n * TRAIN_FRAC)

    print(f"  Train: first {split} obs ({TRAIN_FRAC * 100:.0f}%), Test: last {n - split} obs ({(1 - TRAIN_FRAC) * 100:.0f}%)")

    # Determine model structure from name
    if model_name.startswith("single_lag"):
        k = int(model_name.replace("single_lag", ""))
        # Train
        Y_train = brent_ret[k:split]
        X_train = np.column_stack([np.ones(split - k), s3_ret[:split - k]])
        beta = np.linalg.lstsq(X_train, Y_train, rcond=None)[0]
        # Test
        Y_test = brent_ret[split:]
        X_test = np.column_stack([np.ones(n - split), s3_ret[split - k:n - k]])
        Y_pred = X_test @ beta
    elif model_name.startswith("pure_s3_K"):
        K = int(model_name.replace("pure_s3_K", ""))
        Y_train = brent_ret[K:split]
        X_cols_train = [np.ones(split - K)]
        for kk in range(1, K + 1):
            X_cols_train.append(s3_ret[K - kk:split - kk])
        X_train = np.column_stack(X_cols_train)
        beta = np.linalg.lstsq(X_train, Y_train, rcond=None)[0]
        Y_test = brent_ret[split:]
        X_cols_test = [np.ones(n - split)]
        for kk in range(1, K + 1):
            X_cols_test.append(s3_ret[split - kk:n - kk])
        X_test = np.column_stack(X_cols_test)
        Y_pred = X_test @ beta
    elif model_name.startswith("var_K"):
        K = int(model_name.replace("var_K", ""))
        start = max(K, 1)
        Y_train = brent_ret[start:split]
        X_cols_train = [np.ones(split - start)]
        X_cols_train.append(brent_ret[start - 1:split - 1])
        for kk in range(1, K + 1):
            X_cols_train.append(s3_ret[start - kk:split - kk])
        X_train = np.column_stack(X_cols_train)
        beta = np.linalg.lstsq(X_train, Y_train, rcond=None)[0]
        Y_test = brent_ret[split:]
        X_cols_test = [np.ones(n - split)]
        X_cols_test.append(brent_ret[split - 1:n - 1])
        for kk in range(1, K + 1):
            X_cols_test.append(s3_ret[split - kk:n - kk])
        X_test = np.column_stack(X_cols_test)
        Y_pred = X_test @ beta
    elif model_name == "error_correction":
        s3_mean_train = np.mean(s3[:split])
        Y_train = data["brent_diff"].values[1:split]
        X_ds3_train = s3_diff[:split - 1]
        X_dev_train = s3[:split - 1] - s3_mean_train
        valid = np.isfinite(Y_train) & np.isfinite(X_ds3_train) & np.isfinite(X_dev_train)
        X_train = np.column_stack([np.ones(valid.sum()), X_ds3_train[valid], X_dev_train[valid]])
        beta = np.linalg.lstsq(X_train, Y_train[valid], rcond=None)[0]
        Y_test = data["brent_diff"].values[split:]
        X_ds3_test = s3_diff[split - 1:n - 1]
        X_dev_test = s3[split - 1:n - 1] - s3_mean_train
        nn = min(len(Y_test), len(X_ds3_test), len(X_dev_test))
        Y_test = Y_test[:nn]
        X_test = np.column_stack([np.ones(nn), X_ds3_test[:nn], X_dev_test[:nn]])
        Y_pred = X_test @ beta
    else:
        print(f"  (skipping OOS for {model_name})")
        return None

    # Metrics
    nn = min(len(Y_test), len(Y_pred))
    Y_test, Y_pred = Y_test[:nn], Y_pred[:nn]
    valid = np.isfinite(Y_test) & np.isfinite(Y_pred)
    Y_test, Y_pred = Y_test[valid], Y_pred[valid]

    oos_rmse = np.sqrt(np.mean((Y_test - Y_pred) ** 2))
    oos_mae = np.mean(np.abs(Y_test - Y_pred))

    # Directional accuracy
    if model_name == "error_correction":
        dir_correct = np.sum(np.sign(Y_pred) == np.sign(Y_test))
    else:
        dir_correct = np.sum(np.sign(Y_pred) == np.sign(Y_test))
    dir_accuracy = dir_correct / len(Y_test) * 100

    # Naive benchmark: predict 0 return (random walk)
    naive_rmse = np.sqrt(np.mean(Y_test ** 2))
    skill = 1 - oos_rmse / naive_rmse if naive_rmse > 0 else 0

    print(f"  OOS RMSE = {oos_rmse:.6f}")
    print(f"  OOS MAE  = {oos_mae:.6f}")
    print(f"  Naive (RW) RMSE = {naive_rmse:.6f}")
    print(f"  Forecast skill vs naive = {skill:.4f} ({'better' if skill > 0 else 'worse'} than random walk)")
    print(f"  Directional accuracy = {dir_accuracy:.1f}% ({dir_correct}/{len(Y_test)})")
    print(f"  {'▸ Model beats coin flip!' if dir_accuracy > 50 else '▸ Model does NOT beat coin flip'}")

    return {
        "oos_rmse": oos_rmse,
        "oos_mae": oos_mae,
        "naive_rmse": naive_rmse,
        "skill": skill,
        "dir_accuracy": dir_accuracy,
        "beta": beta,
    }


# ═════════════════════════════════════════════════════════════════════
# SECTION E: TODAY'S IMPLIED BRENT PRICE
# ═════════════════════════════════════════════════════════════════════

def compute_implied_price(data, best_name, best_res):
    """
    Using the best model, compute today's predicted Brent.
    """
    print_header("E. TODAY'S IMPLIED BRENT PRICE")

    brent_ret = data["brent_ret"].values
    s3_ret = data["s3_ret"].values
    brent = data["brent"].values
    s3 = data["s3"].values
    s3_diff = data["s3_diff"].values
    dates = data.index

    current_brent = brent[-1]
    current_s3 = s3[-1]
    last_date = dates[-1]

    print(f"  Latest data: {last_date.date()}")
    print(f"  Current Brent: ${current_brent:.2f}")
    print(f"  Current S3:    {current_s3:.2f}")

    sigma = np.std(best_res["resid"])

    # Compute prediction based on model type
    if best_name.startswith("single_lag"):
        k = int(best_name.replace("single_lag", ""))
        beta = best_res["beta"]
        # Use the most recent S3 return(s) as predictors
        # For single_lag k: we need S3_ret from k days ago to predict today
        # But since we're predicting NEXT day, use S3_ret from today (lag 1 from tomorrow)
        # Actually: model is Brent_ret(t) = a + b * S3_ret(t-k)
        # To predict Brent_ret(tomorrow), we need S3_ret(tomorrow - k) = S3_ret(today - k + 1)
        # To predict Brent_ret(today), we need S3_ret(today - k) which is s3_ret[-(k)]

        # Predict today's return using S3_ret from k days ago
        pred_ret = beta[0] + beta[1] * s3_ret[-k]
        s3_val_used = s3_ret[-k]
        s3_date_used = dates[-k]
        print(f"\n  Model: Brent_ret(t) = {beta[0]:.6f} + {beta[1]:.6f} · S3_ret(t−{k})")
        print(f"  S3_ret(t−{k}) = S3_ret({s3_date_used.date()}) = {s3_val_used:.6f}")
        print(f"  Predicted Brent return = {pred_ret:.6f} ({pred_ret * 100:.3f}%)")

        # For implied price: this predicts the return from yesterday's close to today's close
        # Since we HAVE today's close, the "next-day" prediction is for tomorrow
        # Implied tomorrow: current_brent * (1 + pred_ret)
        # But let's also show the CURRENT implied (what the model says today should be based on yesterday's inputs)
        pred_ret_today = beta[0] + beta[1] * s3_ret[-(k)]
        implied_today = brent[-2] * (1 + pred_ret_today) if len(brent) > 1 else current_brent

        # Next day prediction
        pred_ret_next = beta[0] + beta[1] * s3_ret[-(k - 1)] if k > 1 else beta[0] + beta[1] * s3_ret[-1]
        implied_next = current_brent * (1 + pred_ret_next)

    elif best_name.startswith("pure_s3_K"):
        K = int(best_name.replace("pure_s3_K", ""))
        beta = best_res["beta"]
        x_pred = [1.0]
        for kk in range(1, K + 1):
            x_pred.append(s3_ret[-kk])
        x_pred = np.array(x_pred)
        pred_ret = x_pred @ beta
        print(f"\n  Model: Brent_ret(t) = α + Σ β_k · S3_ret(t−k) for k=1..{K}")
        for kk in range(1, K + 1):
            print(f"    S3_ret(t−{kk}) = {s3_ret[-kk]:.6f} (β_{kk} = {beta[kk]:.6f})")
        print(f"  Predicted Brent return = {pred_ret:.6f} ({pred_ret * 100:.3f}%)")
        implied_today = brent[-2] * (1 + pred_ret) if len(brent) > 1 else current_brent
        implied_next = current_brent * (1 + pred_ret)

    elif best_name.startswith("var_K"):
        K = int(best_name.replace("var_K", ""))
        beta = best_res["beta"]
        x_pred = [1.0, brent_ret[-1]]
        for kk in range(1, K + 1):
            x_pred.append(s3_ret[-kk])
        x_pred = np.array(x_pred)
        pred_ret = x_pred @ beta
        print(f"\n  Model: Brent_ret(t) = α + γ·Brent_ret(t−1) + Σ β_k·S3_ret(t−k)")
        print(f"    Brent_ret(t−1) = {brent_ret[-1]:.6f} (γ = {beta[1]:.6f})")
        for kk in range(1, K + 1):
            print(f"    S3_ret(t−{kk})  = {s3_ret[-kk]:.6f} (β_{kk} = {beta[kk + 1]:.6f})")
        print(f"  Predicted Brent return = {pred_ret:.6f} ({pred_ret * 100:.3f}%)")
        implied_today = brent[-2] * (1 + pred_ret) if len(brent) > 1 else current_brent
        implied_next = current_brent * (1 + pred_ret)

    elif best_name == "error_correction":
        beta = best_res["beta"]
        s3_mean = np.mean(s3)
        pred_diff = beta[0] + beta[1] * s3_diff[-1] + beta[2] * (s3[-1] - s3_mean)
        print(f"\n  Model: ΔBrent(t) = {beta[0]:.6f} + {beta[1]:.6f}·ΔS3(t−1) + {beta[2]:.6f}·[S3(t−1) − {s3_mean:.1f}]")
        print(f"    ΔS3(t−1) = {s3_diff[-1]:.4f}")
        print(f"    S3(t−1) − S3_mean = {s3[-1] - s3_mean:.4f}")
        print(f"  Predicted ΔBrent = ${pred_diff:.4f}")
        implied_today = current_brent + pred_diff
        implied_next = implied_today
        pred_ret = pred_diff / current_brent
        sigma = np.std(best_res["resid"])  # in level terms

    else:
        print(f"  Cannot compute implied price for model: {best_name}")
        return None

    # Prediction interval
    if best_name == "error_correction":
        lower = implied_next - 1.96 * sigma
        upper = implied_next + 1.96 * sigma
    else:
        lower = current_brent * (1 + pred_ret - 1.96 * sigma)
        upper = current_brent * (1 + pred_ret + 1.96 * sigma)

    print(f"\n  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  PREDICTION FOR NEXT TRADING DAY                    │")
    print(f"  │                                                     │")
    print(f"  │  Current Brent:      ${current_brent:>8.2f}                   │")
    print(f"  │  Implied Brent:      ${implied_next:>8.2f}                   │")
    print(f"  │  Predicted return:   {pred_ret * 100:>+8.3f}%                   │")
    print(f"  │  95% interval:       ${lower:>7.2f} — ${upper:>7.2f}          │")
    print(f"  │  Residual σ:         {sigma:>8.6f}                   │")
    print(f"  │  Signal:             {'BUY (S3 predicts up)' if pred_ret > 0 else 'SELL (S3 predicts down)':>30}  │")
    print(f"  └─────────────────────────────────────────────────────┘")

    return {
        "implied_next": implied_next,
        "pred_ret": pred_ret,
        "lower": lower,
        "upper": upper,
        "sigma": sigma,
        "current_brent": current_brent,
        "current_s3": current_s3,
    }


# ═════════════════════════════════════════════════════════════════════
# SECTION F: ROLLING LAGGED MODEL
# ═════════════════════════════════════════════════════════════════════

def run_rolling_lagged(data, best_name):
    """
    63-day rolling window lagged regression.
    For each day, use trailing ROLLING_WINDOW days to fit, then predict next day.
    """
    print_header("F. ROLLING LAGGED MODEL (63-DAY WINDOW)")

    brent_ret = data["brent_ret"].values
    s3_ret = data["s3_ret"].values
    brent = data["brent"].values
    dates = data.index
    n = len(brent_ret)

    # Determine lag from best model
    if best_name.startswith("single_lag"):
        k = int(best_name.replace("single_lag", ""))
    elif best_name.startswith("pure_s3_K") or best_name.startswith("var_K"):
        k = 1  # Use lag 1 for rolling
    else:
        k = 1

    print(f"  Using lag k={k}, rolling window={ROLLING_WINDOW}")
    print(f"  Model: Brent_ret(t) = α_roll + β_roll · S3_ret(t−{k})")

    predictions = np.full(n, np.nan)
    alphas = np.full(n, np.nan)
    betas = np.full(n, np.nan)
    r_squareds = np.full(n, np.nan)

    start = ROLLING_WINDOW + k
    for t in range(start, n):
        # Training window: [t - ROLLING_WINDOW, t)
        Y_win = brent_ret[t - ROLLING_WINDOW:t]
        X_win = s3_ret[t - ROLLING_WINDOW - k:t - k]
        n_win = min(len(Y_win), len(X_win))
        Y_win = Y_win[-n_win:]
        X_win = X_win[-n_win:]

        if len(Y_win) < 10:
            continue

        X_mat = np.column_stack([np.ones(n_win), X_win])
        try:
            beta = np.linalg.lstsq(X_mat, Y_win, rcond=None)[0]
        except Exception:
            continue

        alphas[t] = beta[0]
        betas[t] = beta[1]

        # Predict today's return
        pred = beta[0] + beta[1] * s3_ret[t - k]
        predictions[t] = pred

        # R² of the window
        Y_hat = X_mat @ beta
        ss_res = np.sum((Y_win - Y_hat) ** 2)
        ss_tot = np.sum((Y_win - Y_win.mean()) ** 2)
        r_squareds[t] = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Convert predictions to implied price series
    implied_prices = np.full(n, np.nan)
    for t in range(start, n):
        if np.isfinite(predictions[t]):
            implied_prices[t] = brent[t - 1] * (1 + predictions[t])

    # Performance metrics
    valid = np.isfinite(predictions) & np.isfinite(brent_ret)
    valid_idx = np.where(valid)[0]
    actual_rets = brent_ret[valid]
    pred_rets = predictions[valid]

    if len(actual_rets) > 0:
        rmse = np.sqrt(np.mean((actual_rets - pred_rets) ** 2))
        naive_rmse = np.sqrt(np.mean(actual_rets ** 2))
        skill = 1 - rmse / naive_rmse if naive_rmse > 0 else 0
        dir_correct = np.sum(np.sign(pred_rets) == np.sign(actual_rets))
        dir_accuracy = dir_correct / len(actual_rets) * 100

        # Cumulative PnL (simple: go long if pred > 0, short if pred < 0)
        strategy_rets = np.sign(pred_rets) * actual_rets
        cum_strategy = np.cumsum(strategy_rets)
        cum_bnh = np.cumsum(actual_rets)
        sharpe = np.mean(strategy_rets) / np.std(strategy_rets) * np.sqrt(252) if np.std(strategy_rets) > 0 else 0

        print(f"\n  Rolling prediction results ({len(actual_rets)} predictions):")
        print(f"  RMSE = {rmse:.6f}")
        print(f"  Naive RMSE = {naive_rmse:.6f}")
        print(f"  Forecast skill = {skill:.4f}")
        print(f"  Directional accuracy = {dir_accuracy:.1f}% ({dir_correct}/{len(actual_rets)})")
        print(f"  Strategy Sharpe ratio (annualized) = {sharpe:.3f}")
        print(f"  Cumulative strategy return = {cum_strategy[-1] * 100:.2f}%")
        print(f"  Cumulative buy-and-hold return = {cum_bnh[-1] * 100:.2f}%")

        # Rolling beta stability
        valid_betas = betas[np.isfinite(betas)]
        if len(valid_betas) > 0:
            print(f"\n  Rolling β statistics:")
            print(f"    Mean = {np.mean(valid_betas):.6f}")
            print(f"    Std  = {np.std(valid_betas):.6f}")
            print(f"    Min  = {np.min(valid_betas):.6f}")
            print(f"    Max  = {np.max(valid_betas):.6f}")
            print(f"    % positive = {np.sum(valid_betas > 0) / len(valid_betas) * 100:.1f}%")

    # Store for chart
    rolling_data = pd.DataFrame({
        "date": dates,
        "implied_price": implied_prices,
        "alpha": alphas,
        "beta": betas,
        "r_squared": r_squareds,
        "pred_ret": predictions,
    }).set_index("date")

    return rolling_data


# ═════════════════════════════════════════════════════════════════════
# SECTION G: MULTI-HORIZON PREDICTIONS
# ═════════════════════════════════════════════════════════════════════

def run_multi_horizon(data):
    """
    Direct-horizon regressions: Brent cumulative return over h days
    predicted by S3_ret(t-1).
    """
    print_header("G. MULTI-HORIZON PREDICTIONS")
    print(f"  Model: Brent_cumret(t, t+h) = α_h + β_h · S3_ret(t−1)")
    print(f"  Horizons: {HORIZONS}\n")

    brent = data["brent"].values
    s3_ret = data["s3_ret"].values
    dates = data.index
    n = len(brent)

    current_brent = brent[-1]

    print(f"  {'h':>3}  {'β_h':>10}  {'t-stat':>8}  {'p-value':>10}  {'R²':>8}  {'Implied':>10}  {'95% CI':>20}")
    print(f"  {'─' * 3}  {'─' * 10}  {'─' * 8}  {'─' * 10}  {'─' * 8}  {'─' * 10}  {'─' * 20}")

    for h in HORIZONS:
        if n < h + 2:
            continue
        # Cumulative return from t to t+h
        Y = (brent[h:] - brent[:-h]) / brent[:-h]
        X = s3_ret[:-h]
        # Shift: we need S3_ret(t-1) to predict cumret(t, t+h)
        # So Y[i] = cumret(i+1, i+1+h), X[i] = S3_ret(i)
        nn = min(len(Y), len(X)) - 1
        Y = Y[:nn]
        X = X[:nn]

        valid = np.isfinite(Y) & np.isfinite(X)
        Y, X = Y[valid], X[valid]

        X_mat = np.column_stack([np.ones(len(Y)), X])
        res = ols_multi(X_mat, Y)
        if res is None:
            continue

        sigma = np.std(res["resid"])
        # Predict using most recent S3_ret
        pred_cumret = res["beta"][0] + res["beta"][1] * s3_ret[-1]
        implied = current_brent * (1 + pred_cumret)
        lower = current_brent * (1 + pred_cumret - 1.96 * sigma)
        upper = current_brent * (1 + pred_cumret + 1.96 * sigma)

        sig = "***" if res["p_values"][1] < 0.001 else "**" if res["p_values"][1] < 0.01 else "*" if res["p_values"][1] < 0.05 else ""
        print(f"  {h:>3}  {res['beta'][1]:>10.6f}  {res['t_stats'][1]:>8.3f}  {res['p_values'][1]:>10.4f}  "
              f"{res['r_sq']:>8.5f}  ${implied:>8.2f}  [${lower:.2f}, ${upper:.2f}]  {sig}")

    print(f"\n  Note: Wider CIs at longer horizons = signal decays with time")
    print(f"  S3 is primarily a 1-day leading indicator, not a long-term forecaster")


# ═════════════════════════════════════════════════════════════════════
# SECTION H: PROBABILITY EXTRACTION
# ═════════════════════════════════════════════════════════════════════

def extract_probabilities(implied_info):
    """
    From the model's predicted distribution, compute P(Brent > X)
    for various thresholds.
    """
    print_header("H. PROBABILITY EXTRACTION")

    if implied_info is None:
        print("  No implied price available.")
        return

    current = implied_info["current_brent"]
    pred_ret = implied_info["pred_ret"]
    sigma = implied_info["sigma"]

    print(f"  Assumption: Returns ~ N(μ_pred, σ²)")
    print(f"  μ_pred = {pred_ret:.6f} ({pred_ret * 100:.3f}%)")
    print(f"  σ_resid = {sigma:.6f} ({sigma * 100:.3f}%)")
    print(f"  Current Brent = ${current:.2f}")
    print(f"\n  P(Brent > X) = 1 − Φ((X/current − 1 − μ) / σ)\n")

    print(f"  {'Threshold':>12}  {'Req Return':>12}  {'z-score':>8}  {'P(Brent > X)':>14}  {'Odds':>10}")
    print(f"  {'─' * 12}  {'─' * 12}  {'─' * 8}  {'─' * 14}  {'─' * 10}")

    for X in PRICE_THRESHOLDS:
        req_ret = (X / current) - 1
        z = (req_ret - pred_ret) / sigma if sigma > 0 else np.inf
        prob = 1 - stats.norm.cdf(z)

        if prob > 0 and prob < 1:
            if prob >= 0.5:
                odds = f"{prob / (1 - prob):.2f}:1"
            else:
                odds = f"1:{(1 - prob) / prob:.1f}"
        else:
            odds = "—"

        print(f"  ${X:>10}  {req_ret * 100:>+11.2f}%  {z:>8.3f}  {prob * 100:>13.2f}%  {odds:>10}")

    # Multi-day probabilities (using random walk + drift)
    print_sub("Multi-Day Probabilities")
    print(f"  Assuming daily returns are i.i.d. N(μ, σ²), probability over h days:")
    print(f"  P(Brent > X at any point in h days) ≈ first-passage approximation\n")

    print(f"  {'Threshold':>12}  {'1 day':>10}  {'5 days':>10}  {'10 days':>10}  {'21 days':>10}")
    print(f"  {'─' * 12}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 10}")

    for X in [90, 95, 100, 105, 110, 115, 120]:
        req_ret = (X / current) - 1
        probs_h = []
        for h in [1, 5, 10, 21]:
            mu_h = pred_ret * h  # assumes daily drift compounds
            sigma_h = sigma * np.sqrt(h)
            z_h = (req_ret - mu_h) / sigma_h if sigma_h > 0 else np.inf
            p_h = 1 - stats.norm.cdf(z_h)
            probs_h.append(p_h)
        print(f"  ${X:>10}  {probs_h[0] * 100:>9.1f}%  {probs_h[1] * 100:>9.1f}%  {probs_h[2] * 100:>9.1f}%  {probs_h[3] * 100:>9.1f}%")


# ═════════════════════════════════════════════════════════════════════
# SECTION I: SAVE ROLLING DATA FOR CHART
# ═════════════════════════════════════════════════════════════════════

def save_rolling_data(rolling_data):
    """Save rolling predictions for plot_brent_vs_s3.py to use."""
    out_path = f"{DATA_DIR}/lagged_implied_series.csv"
    rolling_data[["implied_price"]].dropna().to_csv(out_path)
    print(f"\n  Saved rolling implied series → {out_path}")
    print(f"  ({rolling_data['implied_price'].dropna().shape[0]} data points)")


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    print("=" * 75)
    print("  LAGGED PREDICTIVE MODEL: S3 → BRENT")
    print("  Exploiting the 1-day lead of S3 over Brent crude")
    print("=" * 75)

    # Load data
    print("\n  Loading data...")
    data = load_data()
    print(f"  {len(data)} aligned trading days: {data.index[0].date()} → {data.index[-1].date()}")
    print(f"  Brent: ${data['brent'].iloc[0]:.2f} → ${data['brent'].iloc[-1]:.2f}")
    print(f"  S3:    {data['s3'].iloc[0]:.1f} → {data['s3'].iloc[-1]:.1f}")

    # A: Single-lag optimization
    single_results = run_single_lag_models(data)

    # B: Multi-lag models
    multi_results = run_multi_lag_models(data)

    # C: Level models
    level_results = run_level_models(data)

    # D: Best model selection
    best_name, best_res, oos_info = select_best_model(
        data, single_results, multi_results, level_results
    )

    if best_name is None:
        print("\n  ERROR: No valid model found. Exiting.")
        return

    # E: Today's implied price
    implied_info = compute_implied_price(data, best_name, best_res)

    # F: Rolling lagged model
    rolling_data = run_rolling_lagged(data, best_name)

    # G: Multi-horizon predictions
    run_multi_horizon(data)

    # H: Probability extraction
    extract_probabilities(implied_info)

    # I: Save for chart
    save_rolling_data(rolling_data)

    # Final summary
    print_header("SUMMARY")
    print(f"  Best model: {best_name}")
    print(f"  R² = {best_res['r_sq']:.6f}, AIC = {best_res['aic']:.1f}")
    if implied_info:
        print(f"  Current Brent: ${implied_info['current_brent']:.2f}")
        print(f"  Implied Brent (next day): ${implied_info['implied_next']:.2f}")
        print(f"  95% prediction interval: ${implied_info['lower']:.2f} — ${implied_info['upper']:.2f}")
        direction = "UP" if implied_info["pred_ret"] > 0 else "DOWN"
        print(f"  Direction signal: {direction} ({implied_info['pred_ret'] * 100:+.3f}%)")
    if oos_info:
        print(f"  Out-of-sample directional accuracy: {oos_info['dir_accuracy']:.1f}%")
    print(f"\n  Chart data saved to {DATA_DIR}/lagged_implied_series.csv")
    print(f"  Run plot_brent_vs_s3.py to see the lagged implied series on the chart.")


if __name__ == "__main__":
    main()
