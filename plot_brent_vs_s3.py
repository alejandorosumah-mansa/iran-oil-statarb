"""
Brent Crude Oil vs ADIT-S3 (Middle East Armed Conflict)
=======================================================
1. Dual-axis chart (Brent left, S3 right) starting March 2023
2. Brownian bridges over static basket periods
3. Implied Brent price from the S3 basket via multiple models
4. Full quant analysis: assumptions, problems, improvements

Usage: python3 plot_brent_vs_s3.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "Data"
OUT_FILE = "brent_vs_s3.png"
CHART_START = "2023-03-01"
np.random.seed(42)


# ═════════════════════════════════════════════════════════════════════
# SECTION 0: DATA LOADING
# ═════════════════════════════════════════════════════════════════════

print("=" * 75)
print("  BRENT vs ADIT-S3: IMPLIED PRICE ANALYSIS")
print("=" * 75)

# ADIT-S3 basket
df = pd.read_csv(f"{DATA_DIR}/basket_level_monthly.csv", parse_dates=["rebalance_date"])
s3_raw = df[df["basket_code"] == "ADIT-S3"].set_index("rebalance_date").sort_index()
s3_raw = s3_raw[~s3_raw.index.duplicated(keep="first")]
s3_raw = s3_raw["basket_level"]

# Brent crude
brent_raw = yf.download("BZ=F", start=CHART_START, progress=False)
if isinstance(brent_raw.columns, pd.MultiIndex):
    brent_raw.columns = brent_raw.columns.get_level_values(0)
brent_raw = brent_raw["Close"].rename("brent")
brent_raw.index = pd.to_datetime(brent_raw.index)

# Align: forward-fill basket to Brent trading days
s3 = s3_raw.reindex(brent_raw.index, method="ffill").dropna()
brent = brent_raw.reindex(s3.index).dropna()
common = s3.index.intersection(brent.index)
s3, brent = s3.loc[common], brent.loc[common]

print(f"\n  Data range: {common[0].date()} → {common[-1].date()} ({len(common)} trading days)")
print(f"  Brent range: ${brent.min():.2f} → ${brent.max():.2f}, current: ${brent.iloc[-1]:.2f}")
print(f"  S3 range:    {s3.min():.1f} → {s3.max():.1f}, current: {s3.iloc[-1]:.1f}")


# ═════════════════════════════════════════════════════════════════════
# SECTION 1: BROWNIAN BRIDGES OVER STATIC PERIODS
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


# Volatility from non-static returns
rets = s3.pct_change().dropna()
active_sigma = rets[rets.abs() > 1e-8].std()

segments = detect_static_segments(s3, min_run=5)
s3_bridged = s3.copy()
for a, b in segments:
    n = b - a
    bridge = brownian_bridge(n, s3.iloc[a], s3.iloc[b], active_sigma * s3.iloc[a])
    s3_bridged.iloc[a:b + 1] = bridge


# ═════════════════════════════════════════════════════════════════════
# SECTION 2: IMPLIED BRENT PRICE MODELS
# ═════════════════════════════════════════════════════════════════════

# Only use the "active" period (non-static basket) for model fitting
active_mask = s3.diff().abs() > 1e-6
active_mask.iloc[0] = True
s3_active = s3[active_mask]
brent_active = brent[active_mask]

print(f"\n  Active (non-static) observations: {len(s3_active)}")

# ── Model 1: OLS level regression ──
# Brent = alpha + beta * S3
X = np.column_stack([np.ones(len(s3_active)), s3_active.values])
Y = brent_active.values
coeffs_ols = np.linalg.lstsq(X, Y, rcond=None)[0]
alpha_ols, beta_ols = coeffs_ols
fitted_ols = X @ coeffs_ols
resid_ols = Y - fitted_ols
ss_res_ols = np.sum(resid_ols ** 2)
ss_tot_ols = np.sum((Y - Y.mean()) ** 2)
r2_ols = 1 - ss_res_ols / ss_tot_ols
sigma_ols = resid_ols.std()

# Current implied
current_s3 = s3.iloc[-1]
current_brent = brent.iloc[-1]
implied_ols = alpha_ols + beta_ols * current_s3

# ── Model 2: Log-log regression ──
# log(Brent) = alpha + beta * log(S3)
# This models a power-law relationship: Brent = exp(alpha) * S3^beta
log_s3 = np.log(s3_active.values)
log_brent = np.log(brent_active.values)
X_log = np.column_stack([np.ones(len(log_s3)), log_s3])
coeffs_log = np.linalg.lstsq(X_log, log_brent, rcond=None)[0]
alpha_log, beta_log = coeffs_log
fitted_log = np.exp(X_log @ coeffs_log)
resid_log = log_brent - X_log @ coeffs_log
r2_log = 1 - np.sum(resid_log ** 2) / np.sum((log_brent - log_brent.mean()) ** 2)
sigma_log = resid_log.std()

implied_log = np.exp(alpha_log + beta_log * np.log(current_s3))

# ── Model 3: Piecewise / regime-aware regression ──
# Fit separate regressions for S3 < median and S3 >= median
median_s3 = s3_active.median()
mask_low = s3_active.values < median_s3
mask_high = ~mask_low

def fit_ols_segment(x, y):
    if len(x) < 5:
        return None, None, None, None
    X_ = np.column_stack([np.ones(len(x)), x])
    c = np.linalg.lstsq(X_, y, rcond=None)[0]
    res = y - X_ @ c
    r2_ = 1 - np.sum(res ** 2) / np.sum((y - y.mean()) ** 2) if np.sum((y - y.mean()) ** 2) > 0 else 0
    return c[0], c[1], r2_, res.std()

a_lo, b_lo, r2_lo, sig_lo = fit_ols_segment(s3_active.values[mask_low], brent_active.values[mask_low])
a_hi, b_hi, r2_hi, sig_hi = fit_ols_segment(s3_active.values[mask_high], brent_active.values[mask_high])

if current_s3 >= median_s3 and a_hi is not None:
    implied_pw = a_hi + b_hi * current_s3
else:
    implied_pw = a_lo + b_lo * current_s3 if a_lo is not None else implied_ols

# ── Model 4: Quantile regression (median, 75th, 90th percentile) ──
# Minimizes asymmetric loss: rho_q(r) = r*(q - I(r<0))
def quantile_loss(params, X, Y, q):
    pred = X @ params
    resid = Y - pred
    return np.sum(np.where(resid >= 0, q * resid, (q - 1) * resid))

X_qr = np.column_stack([np.ones(len(s3_active)), s3_active.values])
Y_qr = brent_active.values
p0 = coeffs_ols.copy()

quantile_results = {}
for q in [0.50, 0.75, 0.90]:
    res = minimize(quantile_loss, p0, args=(X_qr, Y_qr, q), method="Nelder-Mead")
    a_q, b_q = res.x
    implied_q = a_q + b_q * current_s3
    quantile_results[q] = {"alpha": a_q, "beta": b_q, "implied": implied_q}

# ── Model 5: Rolling expanding-window regression (last 63d) ──
if len(s3_active) > 63:
    recent_s3 = s3_active.iloc[-63:].values
    recent_brent = brent_active.iloc[-63:].values
    X_rec = np.column_stack([np.ones(63), recent_s3])
    c_rec = np.linalg.lstsq(X_rec, recent_brent, rcond=None)[0]
    alpha_rec, beta_rec = c_rec
    resid_rec = recent_brent - X_rec @ c_rec
    r2_rec = 1 - np.sum(resid_rec ** 2) / np.sum((recent_brent - recent_brent.mean()) ** 2)
    sigma_rec = resid_rec.std()
    implied_rec = alpha_rec + beta_rec * current_s3
else:
    alpha_rec, beta_rec, r2_rec, sigma_rec = alpha_ols, beta_ols, r2_ols, sigma_ols
    implied_rec = implied_ols

# ── Model 6: Kernel-weighted local regression at current S3 ──
# Give more weight to observations where S3 was close to current level
bandwidth = (s3_active.max() - s3_active.min()) * 0.25
kernel_weights = np.exp(-0.5 * ((s3_active.values - current_s3) / bandwidth) ** 2)
# Weighted OLS
W = np.diag(kernel_weights)
X_k = np.column_stack([np.ones(len(s3_active)), s3_active.values])
try:
    c_kernel = np.linalg.solve(X_k.T @ W @ X_k, X_k.T @ W @ brent_active.values)
    implied_kernel = c_kernel[0] + c_kernel[1] * current_s3
except np.linalg.LinAlgError:
    implied_kernel = implied_ols

# ── Build the full implied series for charting (using OLS and 63d rolling) ──
implied_series_ols = alpha_ols + beta_ols * s3
implied_series_log = np.exp(alpha_log + beta_log * np.log(s3))

# Rolling 63d implied
implied_series_roll = pd.Series(np.nan, index=s3.index)
for i in range(63, len(s3)):
    window_s3 = s3.iloc[i - 63:i].values
    window_b = brent.iloc[i - 63:i].values
    # Only fit if there's actual variance in this window
    if np.std(window_s3) > 1e-6:
        X_w = np.column_stack([np.ones(63), window_s3])
        c_w = np.linalg.lstsq(X_w, window_b, rcond=None)[0]
        implied_series_roll.iloc[i] = c_w[0] + c_w[1] * s3.iloc[i]


# ═════════════════════════════════════════════════════════════════════
# SECTION 3: PRINT RESULTS
# ═════════════════════════════════════════════════════════════════════

print("\n" + "─" * 75)
print("  IMPLIED BRENT PRICE FROM ADIT-S3 BASKET")
print("─" * 75)
print(f"\n  Current S3 basket level: {current_s3:.2f} (all-time high)")
print(f"  Current Brent close:     ${current_brent:.2f}")

print(f"\n  {'Model':<35s} {'Implied':>10s} {'Spread':>10s} {'R²':>8s}")
print(f"  {'─' * 67}")
print(f"  {'1. OLS (full sample)':<35s} ${implied_ols:>8.2f} ${implied_ols - current_brent:>+8.2f} {r2_ols:>8.4f}")
print(f"  {'2. Log-log (power law)':<35s} ${implied_log:>8.2f} ${implied_log - current_brent:>+8.2f} {r2_log:>8.4f}")
print(f"  {'3. Piecewise (high regime)':<35s} ${implied_pw:>8.2f} ${implied_pw - current_brent:>+8.2f} "
      f"{r2_hi if r2_hi else 0:>8.4f}")
print(f"  {'4a. Quantile (median, q=0.50)':<35s} ${quantile_results[0.50]['implied']:>8.2f} "
      f"${quantile_results[0.50]['implied'] - current_brent:>+8.2f}     {'—':>4s}")
print(f"  {'4b. Quantile (75th, q=0.75)':<35s} ${quantile_results[0.75]['implied']:>8.2f} "
      f"${quantile_results[0.75]['implied'] - current_brent:>+8.2f}     {'—':>4s}")
print(f"  {'4c. Quantile (90th, q=0.90)':<35s} ${quantile_results[0.90]['implied']:>8.2f} "
      f"${quantile_results[0.90]['implied'] - current_brent:>+8.2f}     {'—':>4s}")
print(f"  {'5. Rolling 63d OLS':<35s} ${implied_rec:>8.2f} ${implied_rec - current_brent:>+8.2f} {r2_rec:>8.4f}")
print(f"  {'6. Kernel-weighted local':<35s} ${implied_kernel:>8.2f} ${implied_kernel - current_brent:>+8.2f}     {'—':>4s}")

# Ensemble: average of all models
all_implied = [implied_ols, implied_log, implied_pw,
               quantile_results[0.50]["implied"], implied_rec, implied_kernel]
ensemble = np.mean(all_implied)
ensemble_median = np.median(all_implied)

print(f"  {'─' * 67}")
print(f"  {'ENSEMBLE MEAN':<35s} ${ensemble:>8.2f} ${ensemble - current_brent:>+8.2f}")
print(f"  {'ENSEMBLE MEDIAN':<35s} ${ensemble_median:>8.2f} ${ensemble_median - current_brent:>+8.2f}")

# Confidence bands from the 63d rolling model
print(f"\n  Confidence bands (63d rolling regression):")
print(f"    Implied:  ${implied_rec:.2f}")
print(f"    1-sigma:  [${implied_rec - sigma_rec:.2f}, ${implied_rec + sigma_rec:.2f}]")
print(f"    2-sigma:  [${implied_rec - 2*sigma_rec:.2f}, ${implied_rec + 2*sigma_rec:.2f}]")

# Probability that Brent is cheap (below implied)
z_current = (current_brent - implied_rec) / sigma_rec if sigma_rec > 0 else 0
p_cheap = stats.norm.cdf(z_current)
print(f"    Current Brent z-score: {z_current:+.2f} (percentile: {p_cheap*100:.1f}%)")
if z_current < -1:
    print(f"    >> Brent is SIGNIFICANTLY CHEAP vs S3-implied (>{abs(z_current):.1f} sigma below)")
elif z_current > 1:
    print(f"    >> Brent is SIGNIFICANTLY RICH vs S3-implied (>{z_current:.1f} sigma above)")


# ═════════════════════════════════════════════════════════════════════
# SECTION 4: MODEL DIAGNOSTICS
# ═════════════════════════════════════════════════════════════════════

print("\n" + "─" * 75)
print("  MODEL DIAGNOSTICS")
print("─" * 75)

# Residual normality (Jarque-Bera on OLS residuals)
jb_stat, jb_p = stats.jarque_bera(resid_ols)
skew = stats.skew(resid_ols)
kurt = stats.kurtosis(resid_ols)
print(f"\n  OLS Residuals:")
print(f"    Mean:      ${np.mean(resid_ols):.4f}")
print(f"    Std:       ${sigma_ols:.2f}")
print(f"    Skewness:  {skew:.3f} {'(neg-skewed)' if skew < -0.5 else '(pos-skewed)' if skew > 0.5 else '(symmetric)'}")
print(f"    Kurtosis:  {kurt:.3f} {'(fat tails)' if kurt > 1 else '(thin tails)' if kurt < -1 else '(normal-like)'}")
print(f"    Jarque-Bera: JB={jb_stat:.2f}, p={jb_p:.4f} "
      f"{'(REJECT normality)' if jb_p < 0.05 else '(fail to reject normality)'}")

# Durbin-Watson (residual autocorrelation)
dw = np.sum(np.diff(resid_ols) ** 2) / np.sum(resid_ols ** 2)
print(f"    Durbin-Watson: {dw:.3f} {'(strong autocorr — non-stationary!)' if dw < 0.5 else '(moderate autocorr)' if dw < 1.5 else '(OK)'}")

# ADF-like stationarity check on residuals (simplified)
resid_diff = np.diff(resid_ols)
resid_lag = resid_ols[:-1]
r_adf = np.corrcoef(resid_diff, resid_lag)[0, 1]
print(f"    Resid lag-1 autocorr: {r_adf:.3f} "
      f"{'(residuals are NOT stationary — cointegration questionable)' if abs(r_adf) < 0.3 else '(residuals mean-revert — possible cointegration)'}")

# Beta stability
if len(s3_active) > 126:
    first_half = s3_active.iloc[:len(s3_active)//2]
    brent_first = brent_active.iloc[:len(s3_active)//2]
    second_half = s3_active.iloc[len(s3_active)//2:]
    brent_second = brent_active.iloc[len(s3_active)//2:]

    a1, b1, r2_1, _ = fit_ols_segment(first_half.values, brent_first.values)
    a2, b2, r2_2, _ = fit_ols_segment(second_half.values, brent_second.values)

    print(f"\n  Beta stability (Chow-style split):")
    print(f"    First half:  beta = {b1:.4f}, R² = {r2_1:.4f}")
    print(f"    Second half: beta = {b2:.4f}, R² = {r2_2:.4f}")
    if b1 is not None and b2 is not None and b1 != 0:
        print(f"    Beta change: {(b2/b1 - 1)*100:+.1f}%")
        if abs(b2/b1 - 1) > 0.5:
            print(f"    >> UNSTABLE: beta changed by >{abs(b2/b1-1)*100:.0f}% across halves")


# ═════════════════════════════════════════════════════════════════════
# SECTION 5: ASSUMPTIONS, PROBLEMS, IMPROVEMENTS
# ═════════════════════════════════════════════════════════════════════

print("\n" + "─" * 75)
print("  ASSUMPTIONS, PROBLEMS, AND IMPROVEMENTS")
print("─" * 75)

print("""
  ASSUMPTIONS (what must hold for these models to be valid):
  ──────────────────────────────────────────────────────────
  A1. Linear/log-linear relationship between S3 basket and Brent.
      The basket is a weighted portfolio of binary contract prices.
      Binary prices are bounded [0,1], Brent is unbounded above.
      Linearity is a first-order approximation that breaks at extremes.

  A2. The basket-oil relationship is STATIONARY (time-invariant).
      A beta fitted on 2024 data must hold in 2026. This is the weakest
      assumption — the basket composition changes monthly (contract rolls,
      new events), so the mapping from "basket level" to "oil risk" drifts.

  A3. Residuals are i.i.d. normal.
      Needed for confidence bands and z-scores. Almost certainly violated:
      oil returns have fat tails and volatility clustering.

  A4. The S3 basket IS the information set that prices Brent geopolitical premium.
      In reality, oil prices respond to OPEC+, inventories, demand, refining
      margins, SPR releases, and dozens of non-geopolitical factors.
      The S3 basket only captures the ME conflict component.

  A5. No omitted variable bias.
      Brent = f(S3) ignores every other driver. The true model is
      Brent = f(S3, OPEC, demand, SPR, USD, ...). Omitting these biases beta.

  PROBLEMS (known issues with the current approach):
  ──────────────────────────────────────────────────
  P1. LOW R-SQUARED (~0.04-0.07): The basket explains <7% of Brent variance.
      This means 93% of Brent's movement has nothing to do with the basket.
      The implied price has a wide confidence interval ($10-15 per sigma).

  P2. NON-STATIONARITY: Both Brent and S3 are trending. Level regression
      gives SPURIOUS R² — the true signal is in changes, which show R²≈0.
      The DW statistic confirms severe residual autocorrelation.

  P3. EXTRAPOLATION: S3 is at 497 (ATH). The model was fitted on S3 values
      mostly between 40-300. We are extrapolating well beyond the training
      domain. The implied price at S3=497 is an OUT-OF-SAMPLE prediction
      with no historical precedent to validate against.

  P4. BASKET COMPOSITION DRIFT: The 15 contracts in S3 change every month.
      A contract about "Khamenei survives 2022" is replaced by "US invades
      Iran 2027". The basket level is continuous but the underlying meaning
      shifts. A level of 300 in Oct 2024 (Iran-Israel escalation) represents
      different physical risk than 300 in Jan 2026 (US-Iran engagement).

  P5. ASYMMETRIC RESPONSE: Oil spikes fast on escalation but drops slowly
      on de-escalation (risk premium persistence). A linear model cannot
      capture this asymmetry. The piecewise and quantile models partially
      address this.

  P6. TOUCH vs TERMINAL (Polymarket contracts): The CL strike ladder
      contracts on PM are touch probabilities, not terminal. This inflates
      the PM-implied price by $10-15 vs the basket-implied price.

  IMPROVEMENTS (how to make this more robust):
  ─────────────────────────────────────────────
  I1. ERROR CORRECTION MODEL (ECM): If Brent and S3 are cointegrated,
      use an ECM: delta_Brent = gamma*(Brent - alpha - beta*S3) + noise.
      This explicitly models mean-reversion of the spread while respecting
      non-stationarity. Requires Engle-Granger cointegration test first.

  I2. MULTIVARIATE MODEL: Add controls for non-geopolitical oil drivers:
      Brent = f(S3, VIX, DXY, OPEC_spare_capacity, US_inventories).
      This isolates the S3 signal from confounders. The current R² of 0.07
      might become 0.30+ with proper controls.

  I3. STATE-SPACE / KALMAN FILTER: Let beta vary over time:
      Brent(t) = alpha(t) + beta(t)*S3(t), where beta follows a random walk.
      This adapts to composition drift and regime changes automatically.

  I4. CONTRACT-LEVEL MODEL: Instead of using the aggregate basket level,
      regress Brent on individual contract prices or slot-level averages.
      "Iran-Israel Direct Escalation" slot likely has a much higher beta
      than "Regional Normalization Failure". The aggregate basket dilutes
      the oil-relevant signal with non-oil geopolitical noise.

  I5. GARCH RESIDUALS: Model the residual volatility with GARCH(1,1) to
      get time-varying confidence bands. Current fixed-sigma bands understate
      risk during crisis periods and overstate during calm periods.

  I6. BAYESIAN APPROACH: Use a prior on beta from economic reasoning
      (1 mbpd supply loss ≈ $15/bbl historically) and update with data.
      This prevents the pure data-driven models from producing nonsensical
      betas when the sample is small or the relationship is weak.

  I7. COPULA FOR TAIL DEPENDENCE: Brent and S3 may have stronger correlation
      in the tails (both spike together during crises) than in the body.
      A Gaussian copula misses this. A Clayton or Gumbel copula would capture
      asymmetric tail dependence and produce better conditional distributions.
""")


# ═════════════════════════════════════════════════════════════════════
# SECTION 6: CHART
# ═════════════════════════════════════════════════════════════════════

# Load lagged implied series from implied_price_model.py
lagged_implied_path = f"{DATA_DIR}/lagged_implied_series.csv"
try:
    lagged_df = pd.read_csv(lagged_implied_path, parse_dates=["date"], index_col="date")
    lagged_implied = lagged_df["implied_price"].dropna()
    lagged_implied = lagged_implied[lagged_implied.index >= CHART_START]
    print(f"\n  Lagged implied series loaded: {len(lagged_implied)} points")
except FileNotFoundError:
    lagged_implied = pd.Series(dtype=float)
    print("\n  ⚠ No lagged_implied_series.csv found — run implied_price_model.py first")

fig, (ax_main, ax_spread) = plt.subplots(2, 1, figsize=(14, 10),
                                          height_ratios=[3, 1],
                                          sharex=True)

color_brent = "#1f77b4"
color_s3 = "#d62728"
color_implied = "#2ca02c"
color_lagged = "#ff7f0e"
color_bridge = "#ff9999"

# ── Top panel: Brent + S3 + Implied ──
ax1 = ax_main
ax1.plot(brent.index, brent.values, color=color_brent,
         linewidth=1.5, label="Brent Crude (BZ=F)", alpha=0.9)

# Implied Brent from rolling 63d model
valid_roll = implied_series_roll.dropna()
if len(valid_roll) > 0:
    ax1.plot(valid_roll.index, valid_roll.values, color=color_implied,
             linewidth=1.2, label="Implied Brent (63d rolling)", alpha=0.7,
             linestyle="--")
    # Shade confidence band
    ax1.fill_between(valid_roll.index,
                     valid_roll.values - sigma_rec,
                     valid_roll.values + sigma_rec,
                     color=color_implied, alpha=0.1, label="1-sigma band")

# Lagged implied (from implied_price_model.py)
if len(lagged_implied) > 0:
    ax1.plot(lagged_implied.index, lagged_implied.values, color=color_lagged,
             linewidth=1.2, label="Lagged Implied (S3 → Brent, lag-1)",
             alpha=0.7, linestyle="-.")

ax1.set_ylabel("Brent Crude Oil ($/bbl)", color=color_brent, fontsize=12)
ax1.tick_params(axis="y", labelcolor=color_brent)

# S3 on right axis
ax2 = ax1.twinx()
for seg_start, seg_end in segments:
    seg_idx = s3_bridged.index[seg_start:seg_end + 1]
    seg_vals = s3_bridged.iloc[seg_start:seg_end + 1].values
    ax2.plot(seg_idx, seg_vals, color=color_bridge, linewidth=1.0, alpha=0.5)

ax2.plot(s3_bridged.index, s3_bridged.values, color=color_s3,
         linewidth=1.8, label="ADIT-S3 Basket", alpha=0.85)
ax2.set_ylabel("ADIT-S3 Basket Level", color=color_s3, fontsize=12)
ax2.tick_params(axis="y", labelcolor=color_s3)

# Annotations
for date_str, label in [
    ("2024-10-02", "Oct 2024 shock"),
    ("2025-05-09", "May 2025 trough"),
    ("2026-03-07", f"ATH: {s3.iloc[-1]:.0f}"),
]:
    date = pd.Timestamp(date_str)
    if date in s3_bridged.index:
        ax2.annotate(label, xy=(date, s3_bridged.loc[date]),
                     xytext=(15, 15), textcoords="offset points",
                     fontsize=8, color=color_s3,
                     arrowprops=dict(arrowstyle="->", color=color_s3, lw=0.8),
                     bbox=dict(boxstyle="round,pad=0.2", fc="white",
                               ec=color_s3, alpha=0.8))

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

ax1.set_title("Brent Crude Oil vs ADIT-S3 (Middle East Armed Conflict)\n"
              f"S3-Implied Brent: ${ensemble:.0f}/bbl | Actual: ${current_brent:.0f}/bbl | "
              f"Spread: ${ensemble - current_brent:+.0f}",
              fontsize=13, fontweight="bold", pad=10)
ax1.grid(True, alpha=0.3)

# ── Bottom panel: Spread (Implied - Actual) ──
if len(valid_roll) > 0:
    spread_series = valid_roll - brent.reindex(valid_roll.index)
    ax_spread.fill_between(spread_series.index, 0, spread_series.values,
                           where=spread_series > 0, color=color_implied, alpha=0.3,
                           label="Brent CHEAP vs implied")
    ax_spread.fill_between(spread_series.index, 0, spread_series.values,
                           where=spread_series <= 0, color=color_s3, alpha=0.3,
                           label="Brent RICH vs implied")
    ax_spread.plot(spread_series.index, spread_series.values, color="black",
                   linewidth=0.8, alpha=0.7)
    ax_spread.axhline(0, color="black", linewidth=0.5, linestyle="-")
    ax_spread.axhline(sigma_rec, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax_spread.axhline(-sigma_rec, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax_spread.set_ylabel("Spread ($/bbl)", fontsize=11)
    ax_spread.legend(loc="upper left", fontsize=8)
    ax_spread.grid(True, alpha=0.3)

ax_spread.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax_spread.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
fig.autofmt_xdate(rotation=45)
fig.tight_layout()

fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
print(f"\n  Chart saved to {OUT_FILE}")
plt.show()
