# DMPE preregistration: asymmetric high-reward/risk event model

**Registration date:** 2026-11-02 (UTC)  
**Status:** Frozen research specification; no final-holdout results have been inspected for this experiment.  
**Market:** Delta Exchange India BTCUSD perpetual.  
**Execution mode:** Research and paper observation only. No API credentials, order placement, or live execution.

## Hypothesis

Candle-only symmetric first-touch classification has not survived the first cost-aware replay. The next hypothesis is that separate long and short event models can identify fewer but higher-payoff events when the target is three times the stop distance, and that verified trade/order-book microstructure adds information beyond candles.

A 3:1 gross payoff does **not** guarantee profitability. The model must estimate directional target-first probability accurately enough to exceed the realized, side-specific cost-adjusted break-even probability, while accounting for unfilled passive entries. The Stage 1 asymmetric labels currently measure a signal-close reference path; they are diagnostic labels, not yet maker-fill-conditioned labels. They must not be used to claim maker-entry profitability until the event-level fill/queue model and fill-conditioned label semantics are implemented.

## Frozen primary variants

| Variant | Horizon | Long setup | Short setup | Gross RR |
|---|---:|---|---|---:|
| Primary | 15 minutes | +3x RV target / -1x RV stop | -3x RV target / +1x RV stop | 3:1 |
| Secondary | 1 hour | +3x RV target / -1x RV stop | -3x RV target / +1x RV stop | 3:1 |

- Volatility scale is the completed, trailing horizon-aligned realized volatility available at signal time (15m RV for 15m; 60m RV for 1h).
- Long and short outcomes are separate binary targets. For each direction: +1 = its target touched first, -1 = its stop touched first, 0 = neither touched by the horizon. Same-minute target/stop touches are ambiguous and excluded. Incomplete future windows are excluded.
- No additional target/stop grids, horizon variants, feature variants, or model families may be selected using the final holdout.
- 3h is excluded from this preregistered experiment.

## Prediction and selection

- Fit separate long-target and short-target probability models using only data available before each validation/test boundary.
- Select probability thresholds on validation only by maximizing estimated net expectancy under the frozen execution assumptions, subject to a minimum validation trade count and a positive lower confidence bound where sample size permits.
- At each timestamp, allow at most one direction. If both directions clear their thresholds, take neither unless a tie-break rule is frozen using validation only.
- Report calibration and Brier/log loss in addition to trade metrics. A probability is not a guarantee.

## Cost and fill policy

- Current public BTCUSD contract page lists maker 0.02%, taker 0.05%, and 8-hour funding interval. 18% GST applies to trading fees: 2.36 bps maker and 5.9 bps taker per side before any account-specific discounts. An idealized 3R trade with maker entry + maker target + taker stop has an approximate cost-adjusted break-even target probability around 27–34% when horizon RV is 1.0%–0.2%, respectively, assuming every maker order fills and the target maker exit executes. This is only a theoretical hurdle; queue priority, missed fills, spread, funding and adverse selection increase it.
- Model passive entry as a **limit order with an explicit no-fill outcome**, not as an automatic maker fill. A touch of the limit is insufficient evidence of queue priority. Require a conservative trade-through/queue buffer and measure fill rate, time-to-fill, adverse selection and missed moves using collected L2/trade data. The model labels must be conditioned on actual/credible passive fills, or the selection bias from fill-conditioned paths must be explicitly modelled.
- If passive entry fills, target exit may be maker only if the limit order is actually executable; stop exit is taker. Any unfilled or ambiguous order is not counted as a completed trade. Include cancellation/expiry and price movement while waiting.
- Until the fill model is calibrated from live event data, report a pessimistic scenario range and keep the strategy in research mode.
- Apply funding only at verified next-funding-realization timestamps using the contemporaneous published funding rate and signed position notional. Do not infer settlement from a generic hourly candle close.
- Include spread, measured slippage, fees, funding, INR conversion, and overlapping-position constraints in the final portfolio replay.

## Data splits and multiple-testing control

- Existing historical walk-forward test blocks have already been viewed and are **not** a final holdout for this new hypothesis.
- Do not tune this experiment on the previously reported Stage 3/Stage 4 test blocks.
- Collect a prospective, untouched period beginning after this registration. Earliest proposed final holdout: 2026-11-03 through 2027-01-02 UTC (60 days); if the continuous event feed is incomplete, extend the period rather than patching gaps.
- Train/validation only on data strictly before the final holdout, with horizon-length purging/embargo. Thresholds, features, model settings, fill assumptions and costs must be frozen before opening the final holdout.
- The only confirmatory comparisons are the two frozen horizons. Report the full pair; do not publish only the winner. Apply a family-wise multiple-testing correction (White's Reality Check or a documented Deflated Sharpe / equivalent) and block-bootstrap confidence intervals using day-sized blocks.
- Report trade-level net expectancy, win rate, profit factor, max drawdown, turnover, fill rate, no-fill rate, adverse selection, and 95% block-bootstrap intervals. Break down results by predeclared trend/range and low/medium/high trailing-volatility regimes; regime definitions must use past-only data.

## Deployment gates

1. Automated tests and timestamp/label alignment audit pass.
2. Microstructure feed runs continuously with gap/heartbeat monitoring for multiple weeks.
3. Cost model is calibrated from actual spread, fills, funding realization and fee records.
4. Prospective final holdout has positive net expectancy after all costs, with confidence interval and multiple-testing controls reported.
5. Only then may a 4–8 week paper-trading phase begin. Paper mode logs predictions, intended orders, hypothetical fills, actual observed market state, cancellations, costs and kill-switch events.
6. Fractional Kelly sizing is disabled until calibrated probabilities are stable; any later sizing must have a hard low cap, maximum daily loss, max drawdown stop and exchange-status kill switch.

A failed gate is a valid research result. No result in this document authorizes real-money trading.
