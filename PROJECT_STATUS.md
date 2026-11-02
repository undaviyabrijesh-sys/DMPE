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

## Historical coverage snapshot
- Requested: 1,825 days (five years).
- Traded-price candles: 1,461,177 observed 1-minute rows, from 2023-12-29 13:26 UTC to 2026-11-02 16:26 UTC.
- Price series: 35,164 missing minute intervals; gaps are not evenly distributed and include several multi-day outages.
- Mark-price candles: 1,512,171 rows; 26 missing minutes; exact timestamp match to available traded-price rows is 99.9982%.
- Funding: 25,170 hourly rows; 34 missing hourly intervals. Funding may be negative or zero and must not be treated as a positive price series.
- Open interest: 24,940 hourly rows; continuous hourly timestamps over its actual coverage window.
- Historical trades and historical order-book depth are not available from these candle files.

## Next milestones
1. Review the Stage 1 profile, especially large early traded-price outages and funding-rate interpretation.
2. Reconstruct trade aggressor flow and order-book changes only after validating the live exchange event schema.
3. Define event-based directional labels and barrier semantics.
4. Build chronological train/validation/test datasets with purge/embargo for overlapping forward labels.
5. Compare simple baselines before ML; then evaluate candidate classifiers/regressors using walk-forward validation.
6. Include fees, spread, slippage, funding, latency, turnover, drawdown and calibration.
7. Register only validated models in the desk; add risk controls and paper trading before any live execution decision.

No order execution is enabled. The current feature/label artifact is research data, not a trading signal.
