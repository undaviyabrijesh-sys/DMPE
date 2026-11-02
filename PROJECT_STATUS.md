# DMPE project status

## Milestone 0 — Data-pipeline bootstrap
- [x] Private GitHub repository and project layout
- [x] Historical REST client with 2,000-candle paging and rate-limit backoff
- [x] Historical bundle job for traded price, mark price, funding and open interest
- [x] Live public recorder for trades, L2, candles, ticker/OI, mark price and funding
- [x] Daily JSONL raw event storage
- [x] Candle data-quality audit
- [x] GitHub Actions workflows for tests, historical backfill and finite live capture
- [x] Historical backfill downloaded successfully; actual coverage profiled
- [x] Finite live capture completed and event types observed in a workflow artifact
- [ ] Validate event field semantics, reconnect gaps and timestamp behavior

## Milestone 1 — Streamlit research desk
- [x] Interactive candle/volume chart with timeframe aggregation and basic market descriptors
- [x] Local Parquet/CSV upload, dataset selection and CSV export
- [x] Auto-refreshing live JSONL tape with recent trade display and best-effort L2 size imbalance
- [x] Show exchange system/product trading-status events as future execution safety gates
- [x] Backfill/audit report panels and basic data-integrity checks
- [x] Explicit model/algorithm release gate; order execution remains disabled
- [ ] Smoke-test the app against extracted historical and live artifacts
- [ ] Deploy to a persistent Streamlit host after local verification

## Milestone 2 — Historical audit and first feature dataset
- [x] Profile all four historical series for cadence, missing intervals, duplicates and timestamp ranges
- [x] Report largest gaps and exact traded-price/mark-price timestamp overlap
- [x] Build a minute-aligned feature table without filling missing traded-price candles
- [x] Add lagged returns, realized volatility, ATR, trend distance, volume, candle-shape and time-of-week descriptors
- [x] Add mark-price basis/returns and delayed hourly funding/OI descriptors
- [x] Add forward-return and future-excursion targets for 15m, 1h and 3h
- [x] Add tests for timestamp gaps, future-data leakage and delayed hourly joins
- [x] Review feature null rates and preserve negative/zero funding-rate observations correctly
- [x] Store model inputs and forward targets as float32 to reduce artifact size
- [ ] Review final feature artifact's target distributions and barrier-label class balance
- [ ] Add target barrier labels and walk-forward model evaluation

## Stage 3 walk-forward findings (run 41492585989)

- Four expanding-window folds completed, each with a 60-day validation block and a following 60-day test block, horizon-sized embargoes, and validation-only confidence thresholds.
- The promising single-holdout 15m / 0.75x RV candidate did **not** replicate: across 22,989 combined untouched test rows it produced 1,898 calls (8.3% coverage), 48.2% exact label match, 45.0% wrong-direction outcomes and 6.7% no-touch outcomes. Approximate Wilson 95% interval: 46.0%–50.5%.
- 15m / 1.0x: 914 calls, 47.4% correct share (approximate 95% interval 44.2%–50.6%).
- 1h / 0.75x: 1,019 calls, 44.8% correct share (approximate 95% interval 41.8%–47.9%).
- 3h / 1.0x: 211 calls, 42.7% correct share (approximate 95% interval 36.2%–49.4%).
- All 9 variants had combined balanced accuracy below 44%; no variant established a repeatable directional edge. Larger 1.5x barriers mostly learned to predict no-touch/flat and generated few or no signals.
- Decision: the current candle-derived feature set is insufficient. Do not deploy these models or call their outputs trades. Next research priority is to improve the information set (verified live trade flow/L2), investigate event labels and regimes, then re-test with a fee-aware simulator and block-bootstrap uncertainty.

## Milestone 4 — Live microstructure foundation
- [x] Subscribe the public recorder to exchange-wide system status and product trading-status feeds
- [x] Verify Delta public trade field `r` means buyer_role (t=buyer taker, m=buyer maker)
- [x] Map documented buyer-role to aggressive buy/sell volume; unknown codes remain unclassified
- [x] Parse Delta ob_l2 top-15 snapshots into multi-level depth imbalance, spread and microprice pressure
- [x] Add minute-close event aggregation for trades and L2 with no forward filling
- [x] Add unit tests for aggressor inference, depth imbalance and missing-event minutes
- [ ] Collect materially longer continuous live trade/L2 data
- [ ] Reconstruct event-stream disconnect and stale-feed quality flags before using order-flow data in models
- [ ] Evaluate live microstructure features only on forward-collected data; historical candles cannot substitute for past tick/order-book events

## Historical coverage snapshot
- Requested: 1,825 days (five years).
- Traded-price candles: 1,461,177 observed 1-minute rows, from 2023-12-29 13:26 UTC to 2026-11-02 16:26 UTC.
- Price series: 35,164 missing minute intervals; gaps are not evenly distributed and include several multi-day outages.
- Mark-price candles: 1,512,171 rows; 26 missing minutes; exact timestamp match to available traded-price rows is 99.9982%.
- Funding: 25,170 hourly rows; 34 missing hourly intervals. Funding may be negative or zero and must not be treated as a positive price series.
- Open interest: 24,940 hourly rows; continuous hourly timestamps over its actual coverage window.
- Historical trades and historical order-book depth are not available from these candle files.

## First baseline findings (exploratory single holdout)

- The regularized histogram-gradient-boosting classifier improved balanced accuracy versus the prior-class benchmark in most variants, but overall balanced accuracy remained low (roughly 0.34–0.44 across the 9 variants).
- One candidate, 15-minute horizon with a 0.75x horizon-RV barrier, generated 87 confidence-gated test calls at a threshold of 0.60 selected on validation. Correct label share was 58.6%; wrong-direction share 35.6%; no-touch share 5.7%.
- Its approximate Wilson 95% binomial interval is about 48%–68%, which includes 50%. The interval also ignores serial dependence, and the 9 variants/models were compared on the same holdout. This is not evidence of a stable tradable edge.
- All other tested confidence-gated variants were weaker (approximately 39%–44% correct share) or inactive due to insufficient validation signals.
- Decision: no strategy is approved. The 15m/0.75x candidate is only a hypothesis to re-test in repeated walk-forward periods and a cost-aware simulator.

## Next milestones
1. Review the Stage 1 profile, especially large early traded-price outages and funding-rate interpretation.
2. Reconstruct trade aggressor flow and order-book changes only after validating the live exchange event schema.
3. Define event-based directional labels and barrier semantics.
4. Complete the four-fold expanding-window walk-forward experiment with validation-only threshold selection and purged/embargoed boundaries.
5. Compare simple baselines before ML; then evaluate candidate classifiers/regressors using walk-forward validation.
6. Include fees, spread, slippage, funding, latency, turnover, drawdown and calibration.
7. Register only validated models in the desk; add risk controls and paper trading before any live execution decision.

No order execution is enabled. Feature/label artifacts and experimental baseline models are research data, not trading signals.
