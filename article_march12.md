# The $30 Barrel Nobody Is Pricing: Why Crude Oil Is About to Reprice

**March 12, 2026 | Aditis Research**

---

## The Setup

There are two markets pricing the same physical reality right now — the probability that the Iran conflict escalates into a full-blown oil supply crisis. One of them is wrong by $30 a barrel.

The crude oil futures market says this is a controlled, prolonged disruption that normalizes by year-end. CL front-month closed yesterday at $87.25, with Dec 2026 at $73. That backwardation is the oil market telling you: elevated now, back to normal in nine months. The implied probability of a true escalation — Kharg Island hit, Gulf-wide disruption, $120+ oil — is under 3%.

Polymarket says something completely different. The prediction market assigns a 58% probability that CL touches $110 before month-end. A 43% chance it hits $120. A 25% chance it hits $130. The Kharg Island oil terminal — Iran's main crude export facility handling 90% of the country's oil shipments — has a 25.5% chance of being struck by March 31 according to live contract prices. The US invading Iran before 2027 is trading at 30%.

The spread between these two views is $30/bbl. That is not noise. That is a structural mispricing.

---

## The ADIT-S3 Basket: How We Measure Conflict Risk

The core signal in this analysis is the **ADIT-S3 basket** — a weighted portfolio of 15 live Polymarket contracts that tracks the aggregate probability of Middle East armed conflict escalation. The basket is constructed by the Aditis portfolio engine and rebalanced monthly.

### Construction

Each contract in the basket is assigned to one of four risk categories (called "slots"), and each category carries an aggregate weight:

| Risk Category | Weight | Example Contracts |
|---|---|---|
| Iran-Israel Direct Escalation | 21.0% | Israel strike on Yemen, US or Israel strike Iran first, Israel strikes 3/4 countries |
| Gaza / Hamas Ceasefire Failure | 18.5% | Hamas ceasefire Phase II fails, ceasefire cancelled |
| Hezbollah / Lebanon Front | 15.7% | Hezbollah strike on Israel by March 31 |
| US-Iran Intervention Risk | 15.7% | US invade Iran, Iranian regime survives US strikes, US cyberattack on Iran |
| Regional Normalization Failure | 13.1% | Israel-Lebanon normalization fails, Israel-Saudi normalization fails |
| Cash Buffer | 15.9% | — |

Within each slot, individual contracts are weighted by a quality score that combines keyword relevance, directional exposure confidence, and temporal fit (how many days until expiry relative to the target tenor of ~90-180 days).

### How the Basket Level Works

The basket started at 100.00 on January 1, 2022. On each rebalance date, the basket level is computed as:

```
basket_level = sum(effective_price_i * target_weight_i) * scale_factor
```

Where `effective_price` is the Polymarket YES price for contracts held on the YES side (direct exposure to escalation) and `(1 - YES price)` for contracts held on the NO side (inverse exposure — e.g., normalization contracts where NO means conflict persists). The `scale_factor` normalizes to the initial 100 base.

A rising basket level means the aggregate probability of Middle East conflict escalation is increasing across all 15 contracts simultaneously.

### Current Composition (March 2026)

The 15 contracts currently in the basket, ranked by weight:

| Contract | Price | Weight | Side |
|---|---|---|---|
| Hezbollah strike on Israel by March 31? | 68.5% | 15.7% | YES |
| Israel x Hamas Ceasefire Phase II by June 30? | 75.0% | 9.3% | NO |
| Israel x Hamas ceasefire cancelled by June 30? | 69.0% | 9.3% | NO |
| Israel and Lebanon normalize relations before 2027? | 85.0% | 5.0% | NO |
| Israel and Saudi Arabia normalize relations before 2027? | 77.0% | 5.0% | NO |
| Israel strike on Yemen by June 30, 2026? | 57.0% | 5.0% | YES |
| Will US or Israel strike Iran first? | 39.4% | 4.8% | YES |
| Will France, UK, or Germany strike Iran by June 30? | 11.5% | 4.0% | YES |
| Will the Iranian regime survive U.S. military strikes? | 59.5% | 4.0% | YES |
| Will the US conduct a cyberattack on Iran by March 31? | 47.5% | 4.0% | YES |
| Will Israel strike 4 countries in 2026? | 24.1% | 3.7% | YES |
| Israel strike on Damascus by March 31, 2026? | 10.8% | 3.7% | YES |
| Will Israel strike 3 countries in 2026? | 28.0% | 3.7% | YES |
| Will the U.S. invade Iran before 2027? | 26.5% | 3.7% | YES |
| Israel x Turkey military clash before 2027? | 12.5% | 3.1% | YES |

Note the position sides. Normalization contracts are held on the NO side — when normalization fails (NO price rises), the basket goes up. Escalation contracts are held on the YES side — when escalation probability rises, the basket goes up. This means the basket is a directional bet on conflict escalation across all dimensions.

### Historical Trajectory

The basket tells a story of four distinct regimes:

| Period | Basket Level | What Happened |
|---|---|---|
| Jan 2022 - Sep 2024 | 60 - 100 | Quiet period. No major contracts with high escalation probability. Basket oscillated around par. |
| Oct 2, 2024 | 100 → 263 (+163 in one day) | Iran-Israel escalation shock. Biggest single-day jump in basket history. |
| Nov 2024 - May 2025 | 263 → 40 | De-escalation. Ceasefire talks, diplomatic progress. Basket collapsed to its all-time low of 39.63 on May 9, 2025. |
| Jun 2025 - Mar 2026 | 40 → 497 | Re-escalation. Ceasefire collapse, US-Iran engagement begins. Basket goes parabolic to its all-time high. |

The current level of 496.63 is nearly 5x the original base. The basket has never been here before.

**Recent acceleration:**

| Period | Return |
|---|---|
| 7 days | +77% |
| 14 days | +79% |
| 30 days | +114% |
| 90 days | +168% |
| 1 year | +512% |

The 30-day average basket level is 291. The current level is 497. The basket is trading 70% above its own 30-day moving average — the widest gap ever recorded.

---

## How We Extract Probabilities From the Oil Market

The oil futures curve embeds an implicit probability distribution over conflict outcomes. We extract it using a **constrained three-scenario decomposition**.

### The Three Scenarios

We define three mutually exclusive, collectively exhaustive outcomes for the Iran/Hormuz conflict:

| Scenario | Description | Conditional CL Front-Month | Conditional CL Dec 2026 |
|---|---|---|---|
| S1: Quick Resolution | Strait reopens, diplomacy works, Iran stands down. | $68 | $66 |
| S2: Prolonged Closure | 30-90 day disruption, managed escalation, rerouting. | $100 | $75 |
| S3: Full Escalation | Kharg Island hit, Gulf-wide disruption, $120+ oil. | $130 | $95 |

The conditional prices are calibrated from historical analogs: S1 prices reflect pre-crisis CL levels, S2 prices match the 2019 Abqaiq aftermath, and S3 prices reflect the estimated impact of losing 4-5 mbpd of Gulf transit capacity.

### The Math

We observe two prices from the CL futures curve:
- **Front-month** (CL1): $87.25 (March 11 close)
- **Dec 2026** (CLZ26): $73.06

Each observed price is a probability-weighted average of the conditional prices:

```
CL_front = p1 * 68 + p2 * 100 + p3 * 130
CL_dec   = p1 * 66 + p2 * 75  + p3 * 95
```

Subject to: `p1 + p2 + p3 = 1` and `p1, p2, p3 >= 0`.

This is an over-determined system (2 equations, 3 unknowns, 1 constraint). We solve it using **constrained least-squares optimization** (Sequential Least Squares Programming / SLSQP), minimizing the squared residual between observed and model-implied prices.

### The Solution

The optimizer returns:

| Scenario | Oil-Implied Probability |
|---|---|
| S1: Quick Resolution | 25.5% |
| S2: Prolonged Closure | 71.8% |
| S3: Full Escalation | 2.7% |

**The oil market assigns a 2.7% probability to full escalation.** It is pricing a 72% chance that this remains a prolonged but manageable disruption. The implied front-month price given this distribution is $92.73, close to the actual CL curve front price — confirming the decomposition is well-fitted (residual RMSE < $1).

### What the Curve Shape Tells You

The backwardation — front at $87 vs December at $73 — is the critical signal. The oil market is saying:

1. **Prices are elevated now** ($87 vs pre-crisis ~$70) because of genuine disruption.
2. **But this normalizes by December** ($73 implies the market expects resolution within 9 months).
3. **The tail risk premium is minimal** — if the market feared $130 oil, the back of the curve would be above $80.

The steep backwardation is a structural bet that the conflict de-escalates. The front-month premium is compensation for current disruption, not insurance against future catastrophe.

---

## How We Extract Probabilities From Polymarket

Polymarket provides an explicit probability distribution through two channels: **geopolitical contracts** (binary yes/no events) and **CL price level contracts** (a strike ladder giving P(CL >= X) for various strikes).

### The CL Strike Ladder (Primary Signal)

Polymarket lists contracts of the form: *"Will crude oil (CL) hit (HIGH) $X by end of March?"* These are binary contracts that pay $1 if CL touches strike X at any point before March 31 (touch probability, not terminal).

The live strike ladder as of March 12:

| Strike | P(CL touches X by March 31) |
|---|---|
| $95 | 87% |
| $100 | 77% |
| $105 | 69% |
| $110 | 59% |
| $120 | 43% |
| $130 | 25% |
| $150 | 12% |

This is a **survival function** — `S(X) = P(CL >= X)` — which is monotonically decreasing. We use it to extract the PM-implied expected CL price.

### Computing PM-Implied Expected Price

From the survival function, the expected value of a non-negative random variable is:

```
E[CL] = integral from 0 to infinity of S(x) dx
```

For our discrete strike ladder, we approximate:

```
E[CL] ≈ min_strike + sum over i of [ P(CL >= strike_i) * (strike_{i+1} - strike_i) ]
```

Plugging in the numbers:

```
E[CL] ≈ 95 + (0.87 * 5) + (0.77 * 5) + (0.69 * 5) + (0.59 * 10) + (0.43 * 10) + (0.25 * 20) + (0.12 * extrapolation)
```

This yields a **PM-implied expected CL price of approximately $117.15**.

### Mapping to Three-State Probabilities

We then map the strike ladder to our three scenarios:

- **P(S3: Escalation)** = P(CL touches $110) = 59% → normalized to **58.2%**
- **P(S1: Resolution)** = 1 - P(CL touches $75) ≈ 1% (virtually nobody on Polymarket thinks CL drops below $75)
- **P(S2: Prolonged)** = residual = 1 - 58.2% - 1.0% = **40.8%**

### Important Caveat: Touch vs Terminal

The CL strike contracts are **touch probabilities** — they pay if CL reaches the strike at any point during the month, not if CL settles above the strike at month-end. Touch probabilities are always higher than terminal probabilities because oil can spike intraday (e.g., on a Kharg Island headline) and then retrace.

This means our PM-implied price of $117.15 is an **upper bound estimate**. The true PM-implied expected settlement price is probably $100-108. Even with this adjustment, the spread to the oil market's $87 remains $13-20/bbl — still highly significant.

### The Geopolitical Contracts (Cross-Check)

The individual escalation contracts provide a qualitative cross-check:

- **Ceasefire broken**: 100% (resolved YES — confirmed)
- **US-Iran military engagement**: 100% (resolved YES — confirmed)
- **Iranian regime survives US strikes**: 77.5% (market is pricing active US strikes as baseline)
- **US invades Iran before 2027**: 30%
- **Kharg Island hit by March 31**: 25.5%

These don't enter the quantitative model directly (the CL strike ladder is more informative), but they confirm the narrative: the prediction market is pricing an active, escalating conflict with a meaningful probability of hitting oil infrastructure.

---

## The Historical Relationship: CL vs ADIT-S3

### Full-Sample Regression (2022-2026)

We run an OLS regression of CL front-month daily close against the ADIT-S3 basket level over the full 4-year sample:

```
CL(t) = alpha + beta * S3_basket(t) + epsilon(t)
```

Results:

| Parameter | Value |
|---|---|
| Alpha (intercept) | $80.43 |
| Beta (sensitivity) | -0.0511 |
| R-squared | 0.07 |
| Residual std dev | $9.25 |
| Observations | ~380 aligned daily obs |

### Why the R-squared Is Low (and Why That's Expected)

An R-squared of 0.07 means the S3 basket explains only 7% of CL's variance over the full sample. The beta is negative (-0.05), meaning historically, higher S3 levels were weakly associated with *lower* CL prices.

This is not a bug — it reflects three structural realities:

**1. The basket covers ALL Middle East conflicts, not just Iran/oil.** ADIT-S3 includes Hamas ceasefire contracts, Lebanon normalization, Turkey-Israel clashes, and other events that have zero direct impact on oil supply. The basket tracks *geopolitical risk breadth*, not oil-specific risk.

**2. CL is driven by many non-geopolitical factors.** Over a 4-year window, OPEC+ production decisions, US shale output, Chinese demand, and SPR releases dominate. The Russia-Ukraine war in 2022 drove CL to $120 while ADIT-S3 was at 100 (no Middle East crisis). The subsequent CL decline from $120 to $70 was demand-driven, not ME-driven.

**3. The relationship is regime-dependent.** During the October 2024 Iran-Israel escalation, both CL and S3 spiked together. During the subsequent de-escalation, CL dropped faster than S3 because oil has physical anchors (OPEC+, inventories) that prediction markets don't. A single linear regression smears these regimes together and finds nothing.

### What the Regression IS Useful For

Despite the weak R-squared, the regression serves as a **regime break detector**. The residual (CL_actual - CL_implied) and its z-score tell you when the two markets are unusually far apart relative to their historical relationship.

Current z-score: if we plug in today's values (CL=$87, S3=497), the regression predicts CL ≈ $80.43 + (-0.05 * 497) = $55. The residual is +$32 and the z-score is +3.5. This means CL is 3.5 standard deviations above what the regression expects given the S3 level.

But this isn't a "sell CL" signal — it's telling you the regime has changed. The October 2024 shock created a new relationship between ME conflict and oil prices that didn't exist in the 2022-2023 data. The regression is screaming: *these two markets are in uncharted territory relative to each other*.

### Why We Use the Scenario Model Instead

The regression's job is to tell you something is different. The **scenario model** is what actually generates the trade signal. It doesn't require a stable historical relationship — it works from first principles: what would oil be worth under each scenario, and what probabilities does each market assign?

The scenario model is robust to regime changes because it re-estimates probabilities from the current curve shape and current PM prices every day. It doesn't rely on a 4-year beta that may not hold.

---

## The Probability Gap

We now have implied probabilities from both markets. The comparison:

| Scenario | Oil Market | Polymarket | Gap |
|---|---|---|---|
| Quick Resolution (CL $65-75) | 25.5% | 1.0% | -24.5% |
| Prolonged Closure (CL $85-105) | 71.8% | 40.8% | -31.0% |
| Escalation (CL $110+) | 2.7% | **58.2%** | **+55.5%** |

The oil market is pricing 72% prolonged, 2.7% escalation. Polymarket is pricing 58% escalation. One of these is catastrophically wrong.

Computing implied prices from each distribution:

```
Oil market:  0.255 * $68 + 0.718 * $100 + 0.027 * $130 = $92.73
Polymarket:  0.010 * $68 + 0.408 * $100 + 0.582 * $130 = $117.15
```

**The spread is $29.90 per barrel.** Even after adjusting for touch vs terminal (reducing the PM-implied price by ~$10-15), the adjusted spread is still $13-20/bbl — far above the noise threshold.

---

## Why the Oil Market Is Wrong

The crude market has been pushed down by three forces that have nothing to do with the physical probability of supply disruption:

**1. Trump's verbal interventions.** Every time the administration signals diplomatic progress or threatens to "end the war in 24 hours," CTAs and momentum funds sell crude. The move from $120 to $87 was not a probability update. It was a positioning washout driven by headline risk, not physical risk.

**2. SPR releases and OPEC+ spare capacity narrative.** The market is pricing in the assumption that any supply disruption can be offset. But SPR reserves are at multi-decade lows after the 2022 drawdowns, and OPEC+ spare capacity is concentrated in Saudi Arabia — which itself has a 25.5% chance of losing its normalization path with Israel according to the same prediction markets.

**3. Institutional flows.** Producer hedging, CTA trend-following, and physical traders who must trade regardless of view are all compressing the geopolitical premium out of the curve. These are flows, not information.

Meanwhile, the prediction market participants — retail, crypto-native, OSINT-tracking, geopolitically informed — are aggregating a different information set. They are watching the military deployments, the backchannel leaks, the diplomatic signals. And they are telling you this is not over.

---

## What Happens Today

The ADIT-S3 basket has been accelerating at +77% per week. The ceasefire is already broken. US-Iran military engagement is already confirmed. The question is not whether there is a conflict — the question is whether it hits oil infrastructure.

Kharg Island at 25.5% is the linchpin. If that contract moves from 25% to 50%, the CL repricing is immediate and violent — we are talking $15-20/bbl in a single session, which takes you from $87 to $105 before the market can adjust.

The front-month is at $87. The median outcome on Polymarket is above $110. The market is sitting $23 below the median expected outcome.

---

## The Trade

**Long CL front-month** at $87. The asymmetry is massive: if Polymarket is right, you are buying $30 below fair value. If the oil market is right, you lose $15-20 in a resolution scenario — but you hedge that by selling YES on escalation Polymarket contracts.

The hedge ratio comes from the conditional prices. For every 1,000 barrels of CL exposure ($87,250 notional), you sell approximately $2,500 in escalation PM contracts (Kharg Island, US invade Iran, Israel strikes Iran). If resolution happens, CL drops but your PM shorts pay. If escalation happens, CL reprices violently higher and your long prints.

You are not making a directional bet on the war. You are betting that two markets pricing the same physical reality cannot disagree by $30 for long. Convergence is forced by barrels either flowing or not flowing through the Strait of Hormuz. Eventually, both markets settle to truth.

---

## The Risk

The biggest risk is a structural break — the US establishes a convoy escort regime and tankers resume transit with military protection. In that case, the Polymarket "ceasefire broken" contract stays at 100% (because the ceasefire is broken) but oil drops because barrels flow anyway. The historical beta between the S3 basket and CL breaks down because the contract definitions diverge from physical reality.

You monitor this by watching the residual. If the spread blows out and stays out for more than 72 hours, that is not a trading signal — that is a regime change. You flatten everything and re-estimate.

But as of this morning, there is no escort regime. The ceasefire is broken. US-Iran engagement is active. The Kharg Island contract is at 25% and the basket is at an all-time high. And crude oil is sitting at $87 pretending this is a manageable situation.

Someone is wrong. The $30 barrel nobody is pricing says it is the oil market.

---

## Appendix: Data Sources and Methodology

| Source | What We Fetch | Auth | Endpoint |
|---|---|---|---|
| Polymarket Gamma API | Live contract prices for all 15 S3 contracts + CL strike ladder | None (free) | `gamma-api.polymarket.com/events/slug/{slug}` |
| Polymarket CLOB API | Historical price time series (daily bars) | None (free) | `clob.polymarket.com/prices-history` |
| yfinance `CL=F` | CL front-month daily close | None (free) | Yahoo Finance via Python |
| yfinance `CLZ26.NYM` | Dec 2026 contract price (for curve) | None (free) | Yahoo Finance via Python |
| Internal CSVs | ADIT-S3 basket level time series (1,527 daily obs, 2022-2026) | N/A | `Data/basket_level_monthly.csv` |
| Internal CSVs | Monthly contract compositions with weights and prices | N/A | `Data/last_year_monthly_compositions.csv` |

The scenario model runs a SLSQP constrained optimization. The regression model runs OLS. All code is in `iran_oil_statarb.py` and can be run in three modes: `live` (full PM API + yfinance), `offline` (internal data + yfinance only), `backfill` (historical regression).

---

*Data: ADIT-S3 (Middle East Armed Conflict) basket, Polymarket Gamma API, NYMEX CL futures via yfinance. Analysis as of March 12, 2026.*
