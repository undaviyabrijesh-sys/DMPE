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

## Live microstructure pipeline smoke test (run 41494766217)

- [x] Captured 3,941 raw WebSocket records including trade, L2 and operational status channels.
- [x] Automatically produced a separate derived order-flow artifact.
- Six minute-close rows were generated: trade events occurred in all 6 minutes and L2 snapshots in 5 minutes.
- Parsed 2,566 trade prints and 597 L2 snapshots; all observed trade rows had a recognized buyer-role code in this short capture.
- Verified the expected market fields: aggressive buy/sell volume, volume imbalance, spread, top-5 depth imbalance and microprice pressure.
- This is a schema/collector smoke test only. Six minutes is far too little history to train or validate an order-flow model. The next data task is a materially longer continuous capture and feed-gap analysis.

## First baseline findings (exploratory single holdout)

- The regularized histogram-gradient-boosting classifier improved balanced accuracy versus the prior-class benchmark in most variants, but overall balanced accuracy remained low (roughly 0.34–0.44 across the 9 variants).
- One candidate, 15-minute horizon with a 0.75x horizon-RV barrier, generated 87 confidence-gated test calls at a threshold of 0.60 selected on validation. Correct label share was 58.6%; wrong-direction share 35.6%; no-touch share 5.7%.
- Its approximate Wilson 95% binomial interval is about 48%–68%, which includes 50%. The interval also ignores serial dependence, and the 9 variants/models were compared on the same holdout. This is not evidence of a stable tradable edge.
- All other tested confidence-gated variants were weaker (approximately 39%–44% correct share) or inactive due to insufficient validation signals.
- Decision: no strategy is approved. The 15m/0.75x candidate is only a hypothesis to re-test in repeated walk-forward periods and a cost-aware simulator.

## Milestone 5 — Execution-cost simulation
- [x] Replay only out-of-sample walk-forward predictions against the original 1-minute candle history
- [x] Enter at the next 1-minute candle open after feature availability
- [x] Simulate horizon-aligned volatility barriers with conservative same-minute stop-first handling
- [x] Model stop-market gaps at the worse bar open
- [x] Include two-sided taker fees including published 18% GST and slippage sensitivity (0/1/2/5/10 bps per side)
- [x] Report trade-by-trade P&L, fees, win rate, profit factor, expectancy and drawdown
- [x] Validate simulator run 41497573112 and confirm all 25 tests pass (run 41497573126)
- [x] Correct barrier anchoring to the signal candle close and rerun execution simulation successfully (run 41508618660); tests passed (run 41508618604). This corrected run supersedes the initial Stage 4 metrics.
- [ ] Add funding settlement payments, verified spread/market-impact estimates and account-currency conversion
- [ ] Test stability on further unseen forward data and block-bootstrap confidence intervals

## Stage 4 initial execution-cost findings (run 41497573112)

- Simulation completed for 9 horizon/barrier variants and 5 slippage assumptions (0, 1, 2, 5 and 10 bps per side), using 5.9 bps taker fees per side including the configured GST assumption.
- Every variant was net negative even at the optimistic 0 bps slippage setting. Therefore the current candle-derived DMPE predictions do not pass the first cost-aware tradeability gate.
- Initial best of the tested variants at 0 bps assumed slippage was 3h / 1.0× volatility: 211 trades, 48.3% profitable trades, profit factor 0.65, net P&L −$19.41, fees $22.22, expectancy −$0.092 per configured 0.001 BTC contract.
- 1h / 1.0× at 0 bps: 183 trades, 42.6% profitable, profit factor 0.50, net P&L −$12.98.
- 15m / 0.75× at 0 bps: 1,898 trades, 6.6% profitable, profit factor 0.023, net P&L −$175.58; the very tight barriers are overwhelmed by round-trip fees.
- Increasing assumed slippage made all results worse. Funding, market impact, queue position, INR conversion, leverage/liquidation and portfolio-level overlapping-position constraints are not yet modelled.
- Interpretation: no current model/threshold is suitable for paper trading, let alone live orders. These figures are an initial diagnostic, not a deployable backtest. The next work is to audit the execution/label alignment and build a portfolio-level simulator with verified funding and spread/impact before attempting new features or model tuning.

## Execution/label alignment audit and next experiment (2026-11-02)

- [x] Audited the time convention: feature rows are timestamped at completed candle close; the same timestamp maps to the next 1m candle open for simulated entry. Label windows and simulator windows both begin with that next minute.
- [x] Found and corrected a barrier-reference mismatch: symmetric labels are anchored to the completed signal candle close, but the original execution simulator had re-anchored barriers to the next candle's open/fill. The simulator now retains the signal-close reference for barrier geometry while entry gaps/slippage affect realized P&L. Added a regression test.
- [x] Added separate asymmetric long and short first-touch reference labels for 15m and 1h: 1x horizon-RV stop / 3x target. Stage 1 run 41508187531 built the labels successfully (artifact 13172261495). Target-first=+1, stop-first=-1, no-touch=0; same-minute ambiguity=state 2 and excluded; incomplete windows=state -2. These are signal-close labels, not yet maker-fill-conditioned targets.
- [x] Added a frozen experiment specification in `research/PREREGISTERED_ASYMMETRIC_EXPERIMENT.md`. Only 15m and 1h are confirmatory variants; 3h and extra RR grids are not to be searched on the final holdout.
- [x] Added extraction of live funding snapshots with Delta's exact `next_funding_realization` timestamp and rate. Live order-flow feature build run 41508345064 succeeded. This is observation capture only; funding P&L is not yet charged until the cashflow convention is validated.
- [x] Added exact-time order-flow feature join with no as-of carry-forward, plus 5m/15m trailing descriptors and feed coverage fields.
- [x] Added regression tests for the signal-close barrier anchor, asymmetric target/stop labels, funding snapshot timestamps and point-in-time order-flow join; latest test run 41508618604 passed.
- [x] Added a hardened systemd service and setup guide for an always-on public-data recorder in `deploy/vps/`. The VPS itself has not been provisioned.
- [ ] Complete binary long/short model training, validation-only expected-value thresholding, and asymmetric maker-entry / taker-stop replay. Labels must be fill-conditioned before this is treated as a final trading model.
- [ ] Calibrate maker fill/no-fill and adverse selection from multiple weeks of actual L2/trade captures; no candle-only touch may be treated as guaranteed maker fill.
- [ ] Reconcile funding snapshots against actual funding transaction/realization records before charging funding in replay.
- [ ] Add block-bootstrap confidence intervals, family-wise multiple-testing correction, and predeclared trend/range/volatility regime reports.
- [ ] Reserve prospective final holdout data from 2026-11-03 through 2027-01-02 UTC; previously inspected Stage 3/4 test windows are not eligible as the final holdout.
- [ ] After all research gates pass, run 4–8 weeks of paper observation with a kill switch and no real orders.

## Corrected Stage 4 replay (run 41508618660)

- Replayed after anchoring stop/target barriers to the completed signal candle close, matching the label builder rather than shifting them to the next candle's entry open.
- All 9 symmetric variants remained net negative at every tested slippage assumption.
- Best zero-slippage result remained 3h / 1.0x: 211 trades, 48.3% win rate, profit factor 0.647, net P&L −$19.37 per configured 0.001 BTC contract, fees $22.22, expectancy −$0.092 per trade.
- 1h / 1.0x: 183 trades, 41.5% win rate, profit factor 0.469, net P&L −$14.11.
- 15m / 1.0x: 914 trades, 12.3% win rate, profit factor 0.058, net P&L −$79.77.
- Conclusion unchanged: candle-only symmetric predictions do not pass the cost-aware gate. This replay still uses taker entry/exit and does not test the new asymmetric labels or maker fills.

## Next milestones
1. Provision the VPS using `deploy/vps/README.md`; verify recorder uptime, heartbeats, event coverage and off-host backups.
2. Collect multiple weeks of raw trades, L2 and funding-rate snapshots; validate queue-ahead and funding cashflow semantics.
3. Build fill-conditioned long/short 1R-stop / 3R-target labels and a maker-entry / maker-target / taker-stop execution replay with explicit no-fill outcomes.
4. Train separate binary long/short models; choose thresholds only from validation expected value after costs.
5. Add block-bootstrap confidence intervals, family-wise multiple-testing control and predeclared trend/range/volatility regimes.
6. Use only the prospective final holdout specified in the preregistration after all code, thresholds and assumptions are frozen.
7. Register only validated models in the desk; add fractional-Kelly sizing with a hard cap, kill switch and 4–8 week paper observation before any live execution decision.

No order execution is enabled. Feature/label artifacts and experimental baseline models are research data, not trading signals.
