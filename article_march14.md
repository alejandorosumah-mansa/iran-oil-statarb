# Prediction Markets Lead Oil by One Day: A Statistical Arbitrage Strategy

**March 14, 2026 | ITO Research**

---

## Abstract

We document a statistically significant lead-lag relationship between Polymarket geopolitical contracts and Brent crude oil futures. A basket of 15 Middle East armed conflict prediction market contracts (the ITO-S3 basket) leads Brent crude by one trading day, with cross-correlation r=0.131 (p=0.003). We exploit this signal with a rolling 63-day lag-1 OLS regression on Brownian-bridge-interpolated basket returns, filtered with a 0.5% signal threshold. The resulting strategy delivers a Sharpe ratio of 1.27, cumulative return of +43.5%, and maximum drawdown of -11.3% over 488 trading days. We open-source the full implementation including a live monitoring dashboard.

**Repository**: [github.com/alejandorosumah-mansa/iran-oil-statarb](https://github.com/alejandorosumah-mansa/iran-oil-statarb)

---

## 1. The Information Structure

Two markets price the same underlying physical reality — the probability that the Iran-Israel conflict escalates into an oil supply crisis. They produce strikingly different numbers.

The crude oil futures curve (BZ=F front-month at $98.91, Dec 2026 at $73) implies a constrained three-scenario probability distribution:

| Scenario | Oil-Implied P | Polymarket P | Gap |
|---|---|---|---|
| Quick Resolution | 25.5% | 1.0% | -24.5% |
| Prolonged Closure | 71.8% | 40.8% | -31.0% |
| Full Escalation | 2.7% | 58.2% | **+55.5%** |

The oil market assigns 2.7% to full escalation. Polymarket assigns 58.2%. The implied expected price spread is approximately $24/bbl. This is not noise — it reflects structural differences in the information sets aggregated by each market.

The question we set out to answer: is this disagreement predictive?

---

## 2. The ITO-S3 Basket

The core signal is the **ITO-S3 basket**, a portfolio of 15 live Polymarket contracts tracking Middle East armed conflict escalation. The basket is rebalanced monthly and started at 100.00 on January 1, 2022.

Contracts span five risk categories: Iran-Israel direct escalation (21.0%), Gaza ceasefire failure (18.5%), Hezbollah/Lebanon front (15.7%), US-Iran intervention risk (15.7%), and regional normalization failure (13.1%). Each contract is held on the YES or NO side depending on its directional exposure to escalation — a rising basket means aggregate conflict probability is increasing.

**Current level: 496.63** — an all-time high, nearly 5x the original base.

The basket exhibits a structural feature that complicates standard time series analysis: 247 out of 552 trading days show zero price change across 11 distinct static segments. Binary prediction market contracts do not reprice between event triggers, producing long runs of identical values. These zero-return periods attenuate regression coefficients toward zero and suppress the apparent strength of any predictive relationship.

---

## 3. The Lead-Lag Discovery

We compute the sample cross-correlation function between daily S3 basket returns and next-day Brent crude returns for lags k = -10 to +10:

```
CCF(k) = corr[ brent_ret(t), s3_ret(t - k) ]
```

The only statistically significant correlation occurs at **lag +1**: the S3 basket return today predicts Brent crude's return tomorrow.

```
lag +1:  r = 0.131,  p = 0.003
lag  0:  r = 0.036,  p = 0.41
lag -1:  r = 0.004,  p = 0.93
```

This is consistent with the microstructure hypothesis: prediction markets, populated by retail participants monitoring real-time OSINT and military intelligence signals, reprice geopolitical risk faster than the institutional-dominated oil futures market.

---

## 4. The Brownian Bridge

The 247 static days produce zero returns that bias OLS estimates. We address this with a Brownian bridge interpolation: for each static segment, we replace the flat price with a synthetic path pinned to the real start and end values, with noise calibrated to the active-period volatility:

```
B(t) = start + (t/T) * (end - start) + sigma * W_bridge(t)
```

where W_bridge is a standard Brownian bridge (W(0) = W(T) = 0). The global RNG seed is set once before all segments to ensure reproducibility while giving each segment unique randomness.

**Effect on strategy Sharpe: 0.42 → 1.13.** The bridge does not inject information — it restores the statistical structure that the zero-return periods destroy.

---

## 5. Model Selection

We evaluated 30+ specifications ranked by AIC and BIC:

| Model | Parameters | AIC | BIC |
|---|---|---|---|
| **Single lag-1 OLS** | **2** | **-2765.2** | **-2756.5** |
| Pure S3, K=2 lags | 3 | -2757.7 | -2744.8 |
| VAR(1) with own lag | 3 | -2763.5 | -2750.6 |
| Pure S3, K=5 lags | 6 | -2740.6 | -2714.7 |

The simplest model wins. Adding autoregressive terms, multiple lags, or level variables worsens both information criteria. The optimal specification is:

```
brent_ret(t) = alpha(t) + beta(t) * s3_bridged_ret(t-1) + epsilon(t)
```

estimated over a rolling 63-day window.

### Diagnostic Tests

```
beta             +0.01214   (1% S3 move predicts 0.012% Brent next day)
t-statistic      2.51
p-value          0.012
R-squared        0.0114
Durbin-Watson    1.95       (no autocorrelation)
Breusch-Pagan    1.17       (p=0.28, homoskedastic)
Ljung-Box Q(10)  14.1       (p=0.17, no serial correlation in residuals)
Jarque-Bera      432.6      (p<0.001, fat tails)
```

The model passes every specification test except Jarque-Bera normality, which is expected for commodity returns. Residuals fit a Student-t distribution with df=4.12, confirming the fat-tailed nature of oil return innovations.

---

## 6. Signal Threshold

The raw lag-1 OLS generates a prediction every day, but most daily signals are indistinguishable from noise. We impose a 0.5% threshold: the strategy only takes a position when |predicted_return| > 0.5%.

| Metric | No Filter | 0.5% Filter |
|---|---|---|
| Sharpe | 1.13 | **1.27** |
| Accuracy | 52% | **59%** |
| Max Drawdown | -26.8% | **-11.3%** |
| Calmar | 1.36 | **1.76** |
| Trades/year | 487 | **63** |

The filter reduces trading frequency by 87% while improving every risk-adjusted metric. This is consistent with the signal being informative only during periods of elevated geopolitical volatility.

---

## 7. Strategy Performance

The full strategy results over 488 trading days (January 2024 - March 2026):

![Dashboard — Signal, Strategy Performance, ITO-S3 Basket](screenshots/dashboard_top.png)

| Metric | Value |
|---|---|
| **Sharpe Ratio** | **1.27** |
| Cumulative P&L | +43.5% |
| Maximum Drawdown | -11.3% |
| Calmar Ratio | 1.76 |
| Win Rate | 57.8% |
| Average Win | +1.94% |
| Average Loss | -1.28% |
| Profit Factor | 2.16 |
| Total Trades | 89 |
| Directional Accuracy | 51.8% |

The strategy is a **geopolitical shock harvester**. Regime decomposition confirms this:

| Regime | Sharpe | Interpretation |
|---|---|---|
| High S3 volatility (ME shock) | +1.75 | Strategy earns during crises |
| Low S3 volatility (calm) | -0.24 | Strategy loses in quiet markets |
| Delta | +2.0 | Behaving as designed |

---

## 8. Probability Extraction

From the model's residual distribution, we compute tail probabilities P(Brent > $X) under both Gaussian and Student-t assumptions:

![Probability Table & Recent Trades](screenshots/dashboard_tables.png)

The Student-t model assigns fatter tails than the Gaussian — for example, P(Brent > $105) is 0.7% under Student-t versus 0.2% under Gaussian. At the 21-day horizon, the probability of Brent exceeding $100 is 31.4% under the sqrt-of-time scaling rule.

---

## 9. Current Signal

As of March 13, 2026:

- **Signal**: LONG
- **Predicted return**: +0.79%
- **Implied Brent**: $99.69 (vs current $98.91)
- **95% confidence interval**: $95.68 - $103.70
- **ITO-S3 basket**: 496.63 (all-time high)

![Brent Crude vs ITO-S3 Basket](screenshots/chart_brent_vs_s3.png)

The chart shows Brent crude (blue, left axis) and the ITO-S3 basket (gold, right axis) with Brownian-bridge interpolation over the full sample period. The recent spike in the S3 basket to all-time highs, driven by the acceleration in Polymarket conflict probabilities, has pulled Brent along with a one-day delay.

---

## 10. Rolling Diagnostics

The 63-day rolling beta and R-squared track the model's evolving sensitivity and explanatory power:

![Rolling Beta & R-Squared](screenshots/dashboard_bottom.png)

Beta is highly regime-dependent — near zero during calm periods, spiking during geopolitical shocks. R-squared is consistently low (1-2%) which is expected for a daily return prediction model. The strategy's edge comes from correct sign prediction, not from explaining variance.

---

## 11. Limitations

**R-squared is 1.1%.** The model explains almost nothing about any individual day's return. It works only in aggregate over many trades, through consistent directional accuracy.

**The S3 basket is at 497** — well outside the training range of 40-300. Extrapolation risk is significant.

**Beta is 0.012.** A 10% S3 move predicts a 0.12% move in Brent. The signal is real but small.

**The 15 contracts change monthly.** The basket level is not a stable unit across rebalancing boundaries.

**The Brownian bridge introduces model dependence.** Bridge noise is calibrated to active-period volatility and seeded deterministically. Different seed values or volatility estimates would produce different strategy metrics.

---

## 12. Reproducibility

The full implementation is open-sourced at:

**[github.com/alejandorosumah-mansa/iran-oil-statarb](https://github.com/alejandorosumah-mansa/iran-oil-statarb)**

The repository includes:
- FastAPI backend with 5 API endpoints serving live data
- Institutional-grade monitoring dashboard (TradingView lightweight-charts)
- Rolling OLS engine with Brownian bridge interpolation
- Probability extraction under Gaussian and Student-t models
- Full research scripts for model selection, diagnostics, and chart generation

```bash
git clone https://github.com/alejandorosumah-mansa/iran-oil-statarb.git
cd iran-oil-statarb
pip install -r requirements.txt
uvicorn src.api.server:app --reload
# open http://localhost:8000
```

---

## References

1. Fama, E. (1970). "Efficient Capital Markets: A Review of Theory and Empirical Work." *Journal of Finance*, 25(2), 383-417.
2. Granger, C.W.J. & Newbold, P. (1974). "Spurious Regressions in Econometrics." *Journal of Econometrics*, 2(2), 111-120.
3. Chordia, T. & Swaminathan, B. (2000). "Trading Volume and Cross-Autocorrelations in Stock Returns." *Journal of Finance*, 55(2), 913-935.
4. Campbell, J.Y. & Thompson, S.B. (2008). "Predicting Excess Stock Returns Out of Sample: Can Anything Beat the Historical Average?" *Review of Financial Studies*, 21(4), 1509-1531.
5. Arrow, K.J. et al. (2008). "The Promise of Prediction Markets." *Science*, 320(5878), 877-878.

---

*Data: ITO-S3 (Middle East Armed Conflict) basket via Polymarket, Brent crude (BZ=F) via yfinance. Analysis as of March 14, 2026. This is research code. Not investment advice.*

*Built by [ITO](https://ito.research) — research and structured products for prediction markets.*
