# DMPE prospective prediction-vs-structure comparison

**Status:** Frozen protocol for a new prospective comparison; no results have been inspected.
**Freeze date:** 2026-11-03 UTC.
**Planned prospective evaluation window:** 2026-11-04 00:00 UTC through 2027-01-03 00:00 UTC, conditional on the pre-period model/evaluator freeze checklist below. If the compatible model artifact and evaluator are not frozen before the start, move the start to the next 00:00 UTC boundary and amend this document before inspecting any outcomes.
**Market:** Delta Exchange India BTCUSD perpetual.
**Purpose:** Test whether the existing predictive engine and the separate frozen MTF structure engine provide complementary decision value.
**Execution:** Research / paper observation only. No exchange credentials or order placement.

## Why this is a new experiment

The initial MTF replay and its 36-case gate ablation are exploratory and already inspected. They are not eligible as confirmatory evidence. The gate ablation found no positive net expectancy or net profit factor above 1. This new prospective period is reserved for a single frozen comparison. No thresholds, entry rules, features, model family, target geometry or cost assumptions may be tuned after the period begins.

The MTF layer remains separate from the statistical predictive engine. The objective is not to rescue a negative structure backtest by repeatedly changing rules; it is to test whether independently generated evidence improves decisions when combined.

## Frozen signal arms

Evaluate three arms on the same completed 5-minute decision timestamps:

1. **Prediction-only:** the existing preregistered predictive model's directional call, only when it clears its validation-selected confidence / expected-value threshold. The model, features, weights, threshold and preprocessing must be frozen before the prospective period.
2. **Structure-only:** the original frozen MTF candidate from the 4h + 1h directional gate, 15m setup context and 5m entry module. No changes to pivots, EMA, lookback, entry definitions or module priority. If multiple modules conflict in direction on one timestamp, the structure decision is FLAT.
3. **Combined:** take a directional decision only when prediction-only and structure-only agree on the same timestamp. Disagreement, missing prediction, missing structure or a neutral output is FLAT. No weighted score or learned ensemble is permitted in this first comparison.

All arms are sampled on the identical 5-minute decision grid. Prediction-only must be sampled at the same timestamps as the structure decision; it must not receive extra opportunities from its native 1-minute cadence. **Important implementation prerequisite:** the current repeated walk-forward script evaluates symmetric barrier labels and does not yet emit a frozen model for the asymmetric 1x-stop / 3x-target labels. Before the prospective window opens, implement and test that exact target/model path, train only on pre-period data, select its threshold only on a pre-period validation block, then freeze the model artifact and hash. Do not substitute a symmetric-label model or a model trained on any prospective rows.

## Targets and execution comparability

- Primary horizon: 1 hour. Secondary horizon: 3 hours. The exact 1x horizon-aligned realized-volatility stop / 3x target event definitions in `PREREGISTERED_ASYMMETRIC_EXPERIMENT.md` remain unchanged.
- All three arms use identical target geometry, signal-close reference, next-1m-open entry convention, direction handling and position sizing. This experiment isolates decision selection, not differences between stop-placement styles.
- At most one active position per arm and horizon. Ignore new signals while that arm/horizon position is open. No pyramiding or overlapping exposure within an arm.
- If target and stop are touched in the same 1m candle, classify as ambiguous and exclude from directional outcome scoring; do not infer intraminute order.
- A missing minute inside the outcome window, missing entry open, incomplete future window or invalid feature join makes the sample invalid / censored; never forward-fill or impute an outcome.
- Apply the frozen taker fee and slippage scenarios from the predictive experiment. Funding may be included only after its settlement sign, timestamp and cashflow convention are verified against exchange documentation / records. If funding is not verified, label the comparison cost-incomplete and do not approve any signal.
- Report sensitivity to 0, 1, 2, 5 and 10 bps slippage per side, with the configured taker fee charged on both sides. Do not assume maker fills.

## Data and point-in-time safeguards

- Every decision uses only data available at the completed 5-minute close. HTF bars and pivots retain the original close-availability / right-confirmation rules.
- Predictive probabilities must come from a model trained only on data strictly before the prospective period. All preprocessing and thresholds are frozen from pre-period training / validation.
- Join prediction and structure decisions on exact completed 5-minute UTC close timestamps. No future as-of joins, backward filling, or carrying stale directional calls forward.
- Preserve missingness and data-gap flags. Report source coverage separately for Delta candles, Binance proxy flow and Delta live trade/L2. Binance proxy flow must never be labelled as Delta order flow.
- Record source artifact IDs, checksums, code commit SHA, model hash, feature schema version and fee assumptions before evaluation.

## Required reporting

For each horizon, arm and slippage scenario report:
- eligible decision timestamps, directional calls, coverage and abstention rate;
- executed trades, wins/losses/ambiguous/censored counts, trades/day and exposure;
- gross/net P&L, net expectancy in R, gross/net profit factor, win rate, turnover and max drawdown;
- calibration / Brier score for predictive probabilities where applicable;
- performance by predeclared trend/range and trailing-volatility regimes;
- 95% day-block-bootstrap intervals for net expectancy and profit factor;
- paired daily P&L differences: combined minus prediction-only, and combined minus structure-only.

The primary comparison is whether combined improves net expectancy and drawdown versus prediction-only without unacceptable loss of coverage. Win rate alone is not a selection criterion. Report all three arms, including negative or inactive results. Correct for the two predeclared horizons when interpreting significance.

## Decision and release gate

- No arm is promoted from a historical replay or from a single point estimate.
- A positive result must have adequate sample size, positive net expectancy after verified fees / spread / slippage / funding, robust block-bootstrap uncertainty, and no single regime or small cluster of days dominating returns.
- If funding or venue-specific execution assumptions remain unverified, results remain exploratory and the release gate is not passed.
- Even if the comparison passes, the next step is paper observation with explicit kill-switch rules; no real-money execution is authorized.
- FLAT is a valid and expected outcome. Do not force a minimum number of trades per day.

## Pre-period freeze checklist

Before 2026-11-04 00:00 UTC:
- [ ] Implement the predictive model path for the exact preregistered asymmetric 1x-stop / 3x-target labels; the current walk-forward script is not yet compatible.
- [ ] Verify the prediction model artifact, training cutoff, feature schema and validation threshold.
- [ ] Verify the frozen MTF engine commit and its point-in-time tests.
- [ ] Verify the common 5-minute timestamp join and no-lookahead tests.
- [ ] Verify funding cashflow convention or explicitly mark funding as unresolved.
- [ ] Freeze evaluator code, dependencies, source artifact manifest and code SHA.
- [ ] Run unit / synthetic tests only; do not inspect partial prospective outcomes.
