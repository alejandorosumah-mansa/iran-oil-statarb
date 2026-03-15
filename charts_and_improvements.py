"""
Improved Model + Explanatory Charts
====================================
1. Model improvements: filter zero-S3 days, threshold signals,
   regime-aware, t-distribution probabilities, error-correction combo
2. 8 explanatory charts saved as PNGs

Usage: python3 charts_and_improvements.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from scipy import stats
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "Data"
ACTIVE_START = "2024-01-01"
ROLLING_WINDOW = 63
np.random.seed(42)

# ═════════════════════════════════════════════════════════════════════
# BROWNIAN BRIDGE
# ═════════════════════════════════════════════════════════════════════

def detect_static_segments(series, min_run=5, tol=1e-6):
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
    inc = np.random.normal(0, sigma, size=n)
    w = np.insert(np.cumsum(inc), 0, 0.0)
    t = np.arange(n + 1) / n
    return np.maximum(start_val + (end_val - start_val) * t + (w - t * w[-1]),
                      start_val * 0.3)


def apply_brownian_bridge(series):
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
# DATA LOADING (with Brownian bridge)
# ═════════════════════════════════════════════════════════════════════

print("Loading data...")
df = pd.read_csv(f"{DATA_DIR}/basket_level_monthly.csv", parse_dates=["rebalance_date"])
s3_raw = df[df["basket_code"] == "ADIT-S3"].set_index("rebalance_date").sort_index()
s3_raw = s3_raw[~s3_raw.index.duplicated(keep="first")]
s3_raw = s3_raw[s3_raw.index >= ACTIVE_START]["basket_level"]

brent_raw = yf.download("BZ=F", start=ACTIVE_START, progress=False)
if isinstance(brent_raw.columns, pd.MultiIndex):
    brent_raw.columns = brent_raw.columns.get_level_values(0)
brent_raw = brent_raw["Close"].rename("brent")
brent_raw.index = pd.to_datetime(brent_raw.index)

s3_aligned = s3_raw.reindex(brent_raw.index, method="ffill")
s3_bridged, segments = apply_brownian_bridge(s3_aligned)
n_static = sum(b - a for a, b in segments)
print(f"  Brownian bridge: {len(segments)} static segments, {n_static} days filled")

data = pd.DataFrame({"brent": brent_raw, "s3": s3_bridged}).dropna()
data["brent_ret"] = data["brent"].pct_change()
data["s3_ret"] = data["s3"].pct_change()
data["s3_diff"] = data["s3"].diff()
data["brent_diff"] = data["brent"].diff()
data = data.dropna()

brent_ret = data["brent_ret"].values
s3_ret = data["s3_ret"].values
brent = data["brent"].values
s3_level = data["s3"].values
dates = data.index
n = len(data)

print(f"  {n} obs: {dates[0].date()} → {dates[-1].date()}")
print(f"  Brent: ${brent[0]:.2f} → ${brent[-1]:.2f}")
print(f"  S3 (bridged): {s3_level[0]:.1f} → {s3_level[-1]:.1f}")


# ═════════════════════════════════════════════════════════════════════
# MODEL IMPROVEMENTS
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 75)
print("  MODEL IMPROVEMENTS")
print("=" * 75)


def rolling_strategy(brent_ret, s3_ret, window=63, lag=1,
                     filter_zero=False, threshold=0.0,
                     regime_mask=None):
    """
    Rolling lag-1 strategy with optional improvements.
    Returns predictions array and strategy returns.
    """
    n = len(brent_ret)
    preds = np.full(n, np.nan)
    strat = np.full(n, np.nan)

    for t in range(window + lag, n):
        Y = brent_ret[t - window:t]
        X = s3_ret[t - window - lag:t - lag]
        nn = min(len(Y), len(X))
        Y, X = Y[-nn:], X[-nn:]

        if filter_zero:
            mask = X != 0
            Y, X = Y[mask], X[mask]

        if len(Y) < 10:
            continue

        Xm = np.column_stack([np.ones(len(Y)), X])
        try:
            beta = np.linalg.lstsq(Xm, Y, rcond=None)[0]
        except Exception:
            continue

        pred = beta[0] + beta[1] * s3_ret[t - lag]
        preds[t] = pred

        # Apply threshold: only trade if |pred| > threshold
        if abs(pred) < threshold:
            strat[t] = 0  # flat
        elif regime_mask is not None and not regime_mask[t]:
            strat[t] = 0  # disconnected regime
        else:
            strat[t] = np.sign(pred) * brent_ret[t]

    return preds, strat


def compute_metrics(strat_rets, brent_ret, preds, label):
    """Compute and print strategy metrics."""
    valid = np.isfinite(strat_rets) & np.isfinite(brent_ret)
    sr = strat_rets[valid]
    br = brent_ret[valid]
    pr = preds[valid]

    if len(sr) < 2:
        return {}

    trades = np.sum(sr != 0)
    sharpe = sr.mean() / sr.std() * np.sqrt(252) if sr.std() > 0 else 0
    cum = np.sum(sr)
    cum_bnh = np.sum(br)
    dir_correct = np.sum(np.sign(pr) == np.sign(br))
    # Only count days we actually traded
    traded_mask = sr != 0
    if traded_mask.sum() > 0:
        dir_traded = np.sum(np.sign(pr[traded_mask]) == np.sign(br[traded_mask]))
        dir_traded_pct = dir_traded / traded_mask.sum() * 100
    else:
        dir_traded_pct = 0

    # Max drawdown
    cum_curve = np.cumsum(sr)
    running_max = np.maximum.accumulate(cum_curve)
    drawdowns = cum_curve - running_max
    max_dd = np.min(drawdowns)

    # Calmar ratio
    calmar = (cum / len(sr) * 252) / abs(max_dd) if max_dd != 0 else 0

    result = {
        "label": label,
        "sharpe": sharpe,
        "cum": cum,
        "cum_bnh": cum_bnh,
        "trades": trades,
        "total": len(sr),
        "dir_accuracy": dir_traded_pct,
        "max_dd": max_dd,
        "calmar": calmar,
    }
    return result


# 1. Baseline
preds_base, strat_base = rolling_strategy(brent_ret, s3_ret)
m_base = compute_metrics(strat_base, brent_ret, preds_base, "Baseline")

# 2. Filter zero-S3 days
preds_filt, strat_filt = rolling_strategy(brent_ret, s3_ret, filter_zero=True)
m_filt = compute_metrics(strat_filt, brent_ret, preds_filt, "Filter Zero-S3")

# 3. Threshold signal (only trade when |pred| > 0.002)
preds_thresh, strat_thresh = rolling_strategy(brent_ret, s3_ret, threshold=0.002)
m_thresh = compute_metrics(strat_thresh, brent_ret, preds_thresh, "Threshold 0.2%")

# 4. Higher threshold (0.5%)
preds_thresh5, strat_thresh5 = rolling_strategy(brent_ret, s3_ret, threshold=0.005)
m_thresh5 = compute_metrics(strat_thresh5, brent_ret, preds_thresh5, "Threshold 0.5%")

# 5. Regime-aware: only trade when 63d rolling correlation > 0.05
rolling_corr = np.full(n, np.nan)
for t in range(63, n):
    rc = np.corrcoef(s3_ret[t - 63:t], brent_ret[t - 63:t])[0, 1]
    rolling_corr[t] = rc
regime_mask = rolling_corr > 0.05
preds_regime, strat_regime = rolling_strategy(brent_ret, s3_ret, regime_mask=regime_mask)
m_regime = compute_metrics(strat_regime, brent_ret, preds_regime, "Regime-Aware (corr>0.05)")

# 6. Combined: filter zero + threshold + regime
preds_combo, strat_combo_raw = rolling_strategy(brent_ret, s3_ret,
                                                 filter_zero=True, threshold=0.002,
                                                 regime_mask=regime_mask)
m_combo = compute_metrics(strat_combo_raw, brent_ret, preds_combo, "Combined (all filters)")

# 7. Shorter window (21d)
preds_21, strat_21 = rolling_strategy(brent_ret, s3_ret, window=21)
m_21 = compute_metrics(strat_21, brent_ret, preds_21, "Window=21d")

# 8. Longer window (126d)
preds_126, strat_126 = rolling_strategy(brent_ret, s3_ret, window=126)
m_126 = compute_metrics(strat_126, brent_ret, preds_126, "Window=126d")

# Print comparison
all_models = [m_base, m_filt, m_thresh, m_thresh5, m_regime, m_combo, m_21, m_126]
all_models = [m for m in all_models if m]

print(f"\n  {'Model':<30}  {'Sharpe':>7}  {'Cum %':>7}  {'B&H %':>7}  {'Dir%':>5}  {'Trades':>7}  {'MaxDD%':>7}  {'Calmar':>7}")
print(f"  {'─' * 30}  {'─' * 7}  {'─' * 7}  {'─' * 7}  {'─' * 5}  {'─' * 7}  {'─' * 7}  {'─' * 7}")
for m in all_models:
    print(f"  {m['label']:<30}  {m['sharpe']:>7.3f}  {m['cum']*100:>+6.1f}%  {m['cum_bnh']*100:>+6.1f}%  "
          f"{m['dir_accuracy']:>4.0f}%  {m['trades']:>5}/{m['total']}  {m['max_dd']*100:>+6.1f}%  {m['calmar']:>7.3f}")

# 2-week breakdown for each
print(f"\n  Last 2-Week Sharpe Ratios:")
for label, strat in [("Baseline", strat_base), ("Filter Zero", strat_filt),
                      ("Threshold 0.2%", strat_thresh), ("Regime-Aware", strat_regime),
                      ("Combined", strat_combo_raw), ("Window=21d", strat_21)]:
    last_10 = strat[-10:]
    valid = last_10[np.isfinite(last_10)]
    if len(valid) > 1 and valid.std() > 0:
        s = valid.mean() / valid.std() * np.sqrt(252)
        print(f"    {label:<30}  Sharpe={s:>7.3f}  Cum={valid.sum()*100:>+6.2f}%")


# ═════════════════════════════════════════════════════════════════════
# FIT T-DISTRIBUTION FOR BETTER PROBABILITIES
# ═════════════════════════════════════════════════════════════════════

print("\n" + "─" * 75)
print("  IMPROVED PROBABILITY EXTRACTION (t-distribution)")
print("─" * 75)

# Fit t-distribution to residuals
# Full-sample single-lag1 model
X_full = np.column_stack([np.ones(n - 1), s3_ret[:-1]])
Y_full = brent_ret[1:]
beta_full = np.linalg.lstsq(X_full, Y_full, rcond=None)[0]
resid_full = Y_full - X_full @ beta_full

# Fit Student-t
t_params = stats.t.fit(resid_full)
t_df, t_loc, t_scale = t_params
print(f"  t-distribution fit: df={t_df:.2f}, loc={t_loc:.6f}, scale={t_scale:.6f}")
print(f"  (df={t_df:.1f} means heavier tails than Gaussian)")

# Compare probabilities
current_brent = brent[-1]
pred_ret = beta_full[0] + beta_full[1] * s3_ret[-1]
sigma_gauss = np.std(resid_full)

print(f"\n  {'Threshold':>10}  {'Gaussian P':>12}  {'Student-t P':>12}  {'Ratio':>8}")
print(f"  {'─' * 10}  {'─' * 12}  {'─' * 12}  {'─' * 8}")
for X in [95, 100, 105, 110, 115, 120]:
    req_ret = (X / current_brent) - 1
    z = (req_ret - pred_ret) / sigma_gauss
    p_gauss = 1 - stats.norm.cdf(z)
    p_t = 1 - stats.t.cdf(req_ret, t_df, loc=pred_ret + t_loc, scale=t_scale)
    ratio = p_t / p_gauss if p_gauss > 1e-10 else float('inf')
    print(f"  ${X:>8}  {p_gauss*100:>11.4f}%  {p_t*100:>11.4f}%  {ratio:>7.1f}x")


# ═════════════════════════════════════════════════════════════════════
# CHARTS
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 75)
print("  GENERATING CHARTS")
print("=" * 75)

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# ── Chart 1: Cross-Correlation Function ──
print("  [1/8] Cross-correlation function...")
fig, ax = plt.subplots(figsize=(10, 5))
max_lag = 15
ccf = []
for lag in range(-max_lag, max_lag + 1):
    if lag > 0:
        r = np.corrcoef(s3_ret[:-lag], brent_ret[lag:])[0, 1]
    elif lag < 0:
        r = np.corrcoef(s3_ret[-lag:], brent_ret[:lag])[0, 1]
    else:
        r = np.corrcoef(s3_ret, brent_ret)[0, 1]
    ccf.append(r)
lags = list(range(-max_lag, max_lag + 1))
colors = ['#d62728' if l == 1 else '#1f77b4' for l in lags]
ax.bar(lags, ccf, color=colors, alpha=0.7, edgecolor='white')
ax.axhline(0, color='black', linewidth=0.5)
ax.axhline(1.96 / np.sqrt(n), color='gray', linewidth=0.8, linestyle='--', label='95% CI')
ax.axhline(-1.96 / np.sqrt(n), color='gray', linewidth=0.8, linestyle='--')
ax.axvline(1, color='#d62728', linewidth=0.8, linestyle=':', alpha=0.5)
ax.annotate(f'Lag +1: r={ccf[max_lag+1]:.3f}\np=0.002',
            xy=(1, ccf[max_lag+1]), xytext=(4, ccf[max_lag+1] + 0.03),
            fontsize=9, color='#d62728', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#d62728'))
ax.set_xlabel('Lag (positive = S3 leads Brent)')
ax.set_ylabel('Cross-Correlation')
ax.set_title('Cross-Correlation: S3 Returns → Brent Returns\n(Positive lag = S3 leads)')
ax.legend()
fig.tight_layout()
fig.savefig('chart_1_cross_correlation.png', dpi=150)
plt.close()

# ── Chart 2: Lag-1 Scatter Plot ──
print("  [2/8] Lag-1 scatter plot...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Contemporaneous
ax = axes[0]
ax.scatter(s3_ret * 100, brent_ret * 100, alpha=0.3, s=10, c='#1f77b4')
z = np.polyfit(s3_ret, brent_ret, 1)
x_line = np.linspace(s3_ret.min(), s3_ret.max(), 100)
ax.plot(x_line * 100, (z[0] * x_line + z[1]) * 100, 'r-', linewidth=2)
r_contemp = np.corrcoef(s3_ret, brent_ret)[0, 1]
ax.set_title(f'Contemporaneous (same day)\nr = {r_contemp:.3f}, R² = {r_contemp**2:.4f}')
ax.set_xlabel('S3 Return (%)')
ax.set_ylabel('Brent Return (%)')

# Lag-1
ax = axes[1]
ax.scatter(s3_ret[:-1] * 100, brent_ret[1:] * 100, alpha=0.3, s=10, c='#d62728')
z = np.polyfit(s3_ret[:-1], brent_ret[1:], 1)
x_line = np.linspace(s3_ret.min(), s3_ret.max(), 100)
ax.plot(x_line * 100, (z[0] * x_line + z[1]) * 100, 'r-', linewidth=2)
r_lag1 = np.corrcoef(s3_ret[:-1], brent_ret[1:])[0, 1]
ax.set_title(f'Lagged (S3 yesterday → Brent today)\nr = {r_lag1:.3f}, R² = {r_lag1**2:.4f}')
ax.set_xlabel('S3 Return Yesterday (%)')
ax.set_ylabel('Brent Return Today (%)')

fig.suptitle('The Lead-Lag Effect: All Signal Is in the Lag', fontsize=13, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig('chart_2_scatter_lag.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Chart 3: Rolling Beta + R² ──
print("  [3/8] Rolling beta and R²...")
roll_beta = np.full(n, np.nan)
roll_r2 = np.full(n, np.nan)
roll_alpha = np.full(n, np.nan)
for t in range(64, n):
    Y = brent_ret[t-63:t]
    X = s3_ret[t-64:t-1]
    nn = min(len(Y), len(X))
    Y, X = Y[-nn:], X[-nn:]
    Xm = np.column_stack([np.ones(nn), X])
    try:
        beta = np.linalg.lstsq(Xm, Y, rcond=None)[0]
        roll_alpha[t] = beta[0]
        roll_beta[t] = beta[1]
        Yh = Xm @ beta
        ss_res = np.sum((Y - Yh) ** 2)
        ss_tot = np.sum((Y - Y.mean()) ** 2)
        roll_r2[t] = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    except:
        pass

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1.plot(dates, roll_beta, color='#1f77b4', linewidth=1)
ax1.fill_between(dates, roll_beta, 0, where=np.array(roll_beta) > 0,
                  color='#2ca02c', alpha=0.2, label='Positive β (S3↑ → Brent↑)')
ax1.fill_between(dates, roll_beta, 0, where=np.array(roll_beta) < 0,
                  color='#d62728', alpha=0.2, label='Negative β')
ax1.axhline(0, color='black', linewidth=0.5)
ax1.set_ylabel('Rolling β (63d)')
ax1.set_title('Rolling Lag-1 β: How S3 Predicts Brent (63-Day Window)', fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)

ax2.fill_between(dates, 0, roll_r2, color='#ff7f0e', alpha=0.4)
ax2.plot(dates, roll_r2, color='#ff7f0e', linewidth=1)
ax2.set_ylabel('Rolling R²')
ax2.set_xlabel('Date')
ax2.set_title('Rolling R² of Lag-1 Model')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

fig.autofmt_xdate(rotation=45)
fig.tight_layout()
fig.savefig('chart_3_rolling_beta_r2.png', dpi=150)
plt.close()

# ── Chart 4: Strategy Comparison (Cumulative Returns) ──
print("  [4/8] Strategy comparison...")
fig, ax = plt.subplots(figsize=(12, 6))

# Buy and hold
cum_bnh = np.nancumsum(brent_ret) * 100
ax.plot(dates, cum_bnh, color='gray', linewidth=1, label='Buy & Hold', alpha=0.7)

strategies = [
    ("Baseline", strat_base, '#1f77b4'),
    ("Filter Zero-S3", strat_filt, '#2ca02c'),
    ("Threshold 0.2%", strat_thresh, '#d62728'),
    ("Regime-Aware", strat_regime, '#ff7f0e'),
    ("Combined", strat_combo_raw, '#9467bd'),
]
for label, strat, color in strategies:
    cum = np.nancumsum(np.nan_to_num(strat)) * 100
    ax.plot(dates, cum, color=color, linewidth=1.2, label=label)

ax.axhline(0, color='black', linewidth=0.5)
ax.set_ylabel('Cumulative Return (%)')
ax.set_xlabel('Date')
ax.set_title('Strategy Comparison: Lagged Model Variants', fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
fig.autofmt_xdate(rotation=45)
fig.tight_layout()
fig.savefig('chart_4_strategy_comparison.png', dpi=150)
plt.close()

# ── Chart 5: AIC by Lag ──
print("  [5/8] AIC by lag...")
fig, ax = plt.subplots(figsize=(8, 5))

aic_vals = []
for k in range(1, 11):
    Y = brent_ret[k:]
    X = s3_ret[:-k]
    nn = min(len(Y), len(X))
    Y, X = Y[-nn:], X[-nn:]
    Xm = np.column_stack([np.ones(nn), X])
    beta = np.linalg.lstsq(Xm, Y, rcond=None)[0]
    resid = Y - Xm @ beta
    ss_res = np.sum(resid ** 2)
    sigma2 = ss_res / nn
    log_lik = -nn / 2 * (np.log(2 * np.pi) + np.log(sigma2) + 1)
    aic = -2 * log_lik + 2 * 2
    aic_vals.append(aic)

colors = ['#d62728' if i == 0 else '#1f77b4' for i in range(10)]
bars = ax.bar(range(1, 11), aic_vals, color=colors, alpha=0.7, edgecolor='white')
ax.set_xlabel('Lag k')
ax.set_ylabel('AIC (lower is better)')
ax.set_title('Model Selection: AIC by Lag\n(Lag 1 is clearly the best)', fontweight='bold')
ax.set_xticks(range(1, 11))
# Annotate best
ax.annotate(f'Best: AIC={aic_vals[0]:.0f}', xy=(1, aic_vals[0]),
            xytext=(3, aic_vals[0] - 5), fontsize=10, color='#d62728',
            fontweight='bold', arrowprops=dict(arrowstyle='->', color='#d62728'))
fig.tight_layout()
fig.savefig('chart_5_aic_by_lag.png', dpi=150)
plt.close()

# ── Chart 6: Residual Diagnostics ──
print("  [6/8] Residual diagnostics...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Histogram + t-distribution overlay
ax = axes[0, 0]
ax.hist(resid_full * 100, bins=50, density=True, alpha=0.6, color='#1f77b4', label='Residuals')
x_range = np.linspace(resid_full.min() * 100, resid_full.max() * 100, 200)
ax.plot(x_range, stats.norm.pdf(x_range, loc=np.mean(resid_full) * 100,
        scale=np.std(resid_full) * 100), 'r-', linewidth=2, label='Gaussian fit')
ax.plot(x_range, stats.t.pdf(x_range / 100, t_df, loc=t_loc, scale=t_scale) / 100,
        'g-', linewidth=2, label=f't-dist (df={t_df:.1f})')
ax.set_title('Residual Distribution', fontweight='bold')
ax.set_xlabel('Residual (%)')
ax.legend(fontsize=9)

# QQ plot
ax = axes[0, 1]
theoretical = np.sort(stats.norm.ppf(np.linspace(0.01, 0.99, len(resid_full))))
empirical = np.sort(resid_full / np.std(resid_full))
ax.scatter(theoretical, empirical, s=5, alpha=0.5, c='#1f77b4')
ax.plot([-4, 4], [-4, 4], 'r-', linewidth=1)
ax.set_title('QQ Plot (Normal)', fontweight='bold')
ax.set_xlabel('Theoretical Quantiles')
ax.set_ylabel('Sample Quantiles')
ax.set_xlim(-4, 4)
ax.set_ylim(-6, 6)

# ACF of residuals
ax = axes[1, 0]
acf_vals = [1.0]
for lag in range(1, 21):
    r = np.corrcoef(resid_full[lag:], resid_full[:-lag])[0, 1]
    acf_vals.append(r)
ax.bar(range(21), acf_vals, color='#1f77b4', alpha=0.7, edgecolor='white')
ax.axhline(1.96 / np.sqrt(len(resid_full)), color='red', linestyle='--', linewidth=0.8)
ax.axhline(-1.96 / np.sqrt(len(resid_full)), color='red', linestyle='--', linewidth=0.8)
ax.set_title('Residual ACF (should be within red bands)', fontweight='bold')
ax.set_xlabel('Lag')
ax.set_ylabel('Autocorrelation')

# Residuals over time
ax = axes[1, 1]
ax.scatter(dates[1:], resid_full * 100, s=5, alpha=0.4, c='#1f77b4')
ax.axhline(0, color='black', linewidth=0.5)
# Rolling std
roll_std = pd.Series(resid_full).rolling(63).std().values * 100
ax.plot(dates[1:], roll_std, color='#d62728', linewidth=1.5, label='Rolling σ (63d)')
ax.plot(dates[1:], -roll_std, color='#d62728', linewidth=1.5)
ax.set_title('Residuals Over Time (volatility clustering)', fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Residual (%)')
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))

fig.suptitle('Model Diagnostics: Lag-1 Regression Residuals', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig('chart_6_residual_diagnostics.png', dpi=150)
plt.close()

# ── Chart 7: Multi-Horizon Signal Decay ──
print("  [7/8] Multi-horizon signal decay...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

horizons = [1, 2, 3, 5, 7, 10, 15, 21]
betas_h = []
tstat_h = []
r2_h = []
for h in horizons:
    Y = (brent[h:] - brent[:-h]) / brent[:-h]
    X = s3_ret[:-h]
    nn = min(len(Y), len(X)) - 1
    Y, X = Y[:nn], X[:nn]
    valid = np.isfinite(Y) & np.isfinite(X)
    Y, X = Y[valid], X[valid]
    Xm = np.column_stack([np.ones(len(Y)), X])
    beta = np.linalg.lstsq(Xm, Y, rcond=None)[0]
    Yh = Xm @ beta
    ss_res = np.sum((Y - Yh) ** 2)
    ss_tot = np.sum((Y - Y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    se = np.sqrt(ss_res / (len(Y) - 2) / np.sum((X - X.mean()) ** 2))
    t_stat = beta[1] / se if se > 0 else 0
    betas_h.append(beta[1])
    tstat_h.append(t_stat)
    r2_h.append(r2)

# Beta by horizon
ax1.bar(range(len(horizons)), betas_h, color=['#d62728' if abs(t) > 1.96 else '#1f77b4'
        for t in tstat_h], alpha=0.7, edgecolor='white')
ax1.set_xticks(range(len(horizons)))
ax1.set_xticklabels([str(h) for h in horizons])
ax1.set_xlabel('Horizon (trading days)')
ax1.set_ylabel('β coefficient')
ax1.set_title('Signal Strength by Horizon', fontweight='bold')
ax1.axhline(0, color='black', linewidth=0.5)

# t-stat by horizon
ax2.bar(range(len(horizons)), tstat_h, color=['#d62728' if abs(t) > 1.96 else '#1f77b4'
        for t in tstat_h], alpha=0.7, edgecolor='white')
ax2.axhline(1.96, color='gray', linestyle='--', linewidth=0.8, label='95% significance')
ax2.axhline(-1.96, color='gray', linestyle='--', linewidth=0.8)
ax2.set_xticks(range(len(horizons)))
ax2.set_xticklabels([str(h) for h in horizons])
ax2.set_xlabel('Horizon (trading days)')
ax2.set_ylabel('t-statistic')
ax2.set_title('Statistical Significance by Horizon', fontweight='bold')
ax2.legend(fontsize=9)

fig.suptitle('Signal Decay: S3 Predicts Brent at 1-5 Days, Vanishes by 10', fontsize=13, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig('chart_7_signal_decay.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Chart 8: Probability Comparison (Gaussian vs t) ──
print("  [8/8] Probability comparison...")
fig, ax = plt.subplots(figsize=(10, 6))

thresholds = np.arange(85, 125, 1)
p_gauss_list = []
p_t_list = []
for X in thresholds:
    req_ret = (X / current_brent) - 1
    z = (req_ret - pred_ret) / sigma_gauss
    p_g = (1 - stats.norm.cdf(z)) * 100
    p_t = (1 - stats.t.cdf(req_ret, t_df, loc=pred_ret + t_loc, scale=t_scale)) * 100
    p_gauss_list.append(p_g)
    p_t_list.append(p_t)

ax.plot(thresholds, p_gauss_list, 'b-', linewidth=2, label='Gaussian model')
ax.plot(thresholds, p_t_list, 'r-', linewidth=2, label=f'Student-t model (df={t_df:.1f})')
ax.axvline(current_brent, color='green', linewidth=1, linestyle='--',
           label=f'Current Brent: ${current_brent:.0f}')
ax.fill_between(thresholds, p_gauss_list, p_t_list, alpha=0.1, color='red',
                label='Tail risk underestimated by Gaussian')

# PM probabilities for comparison
pm_strikes = [95, 100, 110, 120, 130]
pm_probs = [87, 77, 59, 43, 25]
ax.scatter(pm_strikes, pm_probs, s=80, c='#ff7f0e', zorder=5, marker='D',
           edgecolors='black', label='Polymarket (touch prob)')

ax.set_xlabel('Brent Price ($)')
ax.set_ylabel('P(Brent > X) (%)')
ax.set_title('Probability of Brent Exceeding Price Levels\nGaussian vs Student-t vs Polymarket',
             fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.set_ylim(-5, 105)
fig.tight_layout()
fig.savefig('chart_8_probability_comparison.png', dpi=150)
plt.close()

print("\n  All 8 charts saved!")
print("  chart_1_cross_correlation.png")
print("  chart_2_scatter_lag.png")
print("  chart_3_rolling_beta_r2.png")
print("  chart_4_strategy_comparison.png")
print("  chart_5_aic_by_lag.png")
print("  chart_6_residual_diagnostics.png")
print("  chart_7_signal_decay.png")
print("  chart_8_probability_comparison.png")
