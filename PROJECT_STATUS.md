# DMPE project status

## Milestone 7 — Event-label balance diagnostic
- [x] Added a separate exploratory first-touch label-distribution experiment for 1h and 3h horizons across a predeclared 0.50x/0.75x/1.00x/1.50x/2.00x horizon-RV grid.
- [x] Preserved ambiguous same-minute touches and incomplete/gapped windows as distinct diagnostic states.
- [x] Sampled one UTC-aligned observation per horizon and added chronological 70/15/15 summaries with boundary purges.
- [x] Added focused unit tests, a reproducible GitHub Actions workflow and protocol documentation.
- [ ] Review the completed diagnostic report before deciding whether to preregister a new event definition.
- Decision guardrail: this is label-distribution research only. It does not fit models, select a profitable barrier, change confirmatory asymmetric 1R-stop/3R-target labels, inspect the prospective holdout, or approve trading.
- Protocol: [research/EVENT_LABEL_BALANCE_DIAGNOSTIC.md](research/EVENT_LABEL_BALANCE_DIAGNOSTIC.md).



## Milestone 6 — MTF structure & entry research
- [x] Added completed 5m/15m/1h/4h resampling, confirmed swing structure, 4h/1h context, 15m setup context and 5m pullback/sweep/breakout-retest candidates.
- [x] Added gap-aware taker-fee/slippage replay; stale entries and trades crossing missing data are excluded/censored.
- [x] Completed first replay (run 41567256039), 36-case post-hoc gate ablation (run 41567250195), and tests (run 41567285342).
- [x] Froze a new prospective protocol for prediction-only vs structure-only vs combined comparison: [research/PREREGISTERED_MTF_PREDICTION_COMPARISON.md](research/PREREGISTERED_MTF_PREDICTION_COMPARISON.md).
- [ ] Implement the predictive model path for the exact preregistered asymmetric 1x-stop / 3x-target labels; the current walk-forward script is not yet compatible.
- [ ] Before the prospective window opens, freeze the model artifact/training cutoff, common 5m timestamp alignment, point-in-time tests, funding convention and evaluator hash.
- [ ] Planned comparison window: 2026-11-04 00:00 UTC through 2027-01-03 00:00 UTC, only if the pre-period freeze checklist is complete. Otherwise shift the start to the next 00:00 UTC boundary and amend the preregistration before inspecting outcomes.
- Decision: MTF remains a separate research layer, not an approved signal. All 36 post-hoc gate-ablation cases had negative net expectancy and net PF below 1. The MTF engine must not be promoted based on these already-inspected results. Keep FLAT valid; no live orders.
- Full exploratory results and assumptions: [research/MTF_STRUCTURE_RESULTS.md](research/MTF_STRUCTURE_RESULTS.md).



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


## Revised two-horizon Binance flow experiment (2026-11-02)

- [x] Checked Binance public data archive root; it is reachable. Official Binance public-data documentation confirms USD-M futures aggTrades include timestamp and buyer-maker flag, with monthly/daily archives and SHA-256 checksum files.
- [x] Added monthly BTCUSDT USD-M aggTrades downloader with checksum verification and a raw-archive manifest via verified ZIP + checksum artifact.
- [x] Added 1-minute proxy flow features: signed base-volume imbalance, trade count, fixed-$100k large-print notional share, signed-notional imbalance and trailing 5m flow acceleration versus the prior 5m window.
- [x] Added exact UTC close-time merge with no as-of join and no forward-fill. Binance proxy fields are explicitly separate from Delta live order-flow fields.
- [x] Added a month-at-a-time GitHub Actions workflow that uploads source archive, checksum and derived feature artifacts for 90 days.
- [x] Changed the current Stage 1 asymmetric label build to exactly two confirmatory horizons: 1h and 3h; 1R stop / 3R target. The 15m horizon is removed from this experiment.
- [x] Replaced the prior preregistration with a frozen 1h/3h-only specification. Previously viewed historical test blocks are disqualified; prospective holdout remains 2026-11-03 to 2027-01-02 UTC.
- [ ] Confirm the first monthly archive workflow completes, verify its checksum, feature artifact and row counts, then process the remaining training months incrementally.
- [ ] Run the two-horizon model with Binance proxy features on chronological training/validation only; do not inspect prospective holdout.
- [ ] Keep cross-venue transfer as a hypothesis; validate against multiple weeks of Delta WS capture and calibrate maker fill/no-fill from Delta L2 before any execution claims.
- [ ] VPS is not provisioned. Account owner must provision and authorize the host; no credentials are requested or stored here.

The Binance archive can accelerate historical trade-flow research, but it does not replace Delta event capture or historical queue data. The 1m output is an explanatory feature set, not a strategy or trading signal.


## Binance historical flow backfill execution

- [x] Confirmed the archive is genuinely downloadable from a GitHub Actions runner, not merely visible in the browser: BTCUSDT USD-M aggTrades for 2024-01 downloaded successfully (522,743,749 bytes) and SHA-256 matched the published checksum.
- [x] The 2024-01 monthly feature job generated 44,640 minute-close rows, covering 2024-01-01 00:01 UTC through 2024-02-01 00:00 UTC. All 44,640 minutes had at least one aggregate trade. This is a single-month smoke test, not the full 2-year history.
- [x] Added provenance manifests containing archive URL, checksum URL, SHA-256, byte size and verification result.
- [x] Added a 24-month matrix backfill for 2024-11 through 2026-10 with six monthly jobs in parallel and a final strict continuity check. Monthly ZIPs are removed after processing in this bulk workflow; checksums and source URLs are retained. The one-month workflow retains a raw archive artifact when needed.
- [ ] Wait for the 24-month matrix and combine job to complete; verify exactly 24 monthly manifests and 1,051,200 minute-close rows before merging to candle features.
- [ ] The final 24-month flow artifact is a Binance proxy dataset, not Delta history. Keep it distinct from the Delta live feed in all model and execution reports.


## Quantitative regime research — initial implementation (2026-11-03)

- [x] Added a constrained Gaussian-QMLE GARCH(1,1) estimator with stationarity guard (alpha + beta < 0.999) and causal next-period variance recursion.
- [x] Added a diagonal-Gaussian HMM with training-only feature scaling, train-only parameter estimation, low-to-high volatility state ordering and forward-filtered probabilities. No backward smoothing is used for decision-time state estimates.
- [x] Added past-only trailing realized-volatility observations for regime estimation.
- [x] Added synthetic tests for positive/stationary GARCH forecasts, causal forecasts, normalized HMM probabilities, future-perturbation invariance and unfitted-model protection.
- [ ] Verify the new test workflow passes on GitHub Actions before accepting the implementation.
- [ ] Run GARCH and HMM as independent research features on chronological train/validation/test partitions; compare against existing DMPE baselines and simple rolling-volatility/regime baselines.
- [ ] Test robustness to gaps, heavy tails, regime-count choices and parameter instability; fit all scalers and model parameters on training data only.
- [ ] Do not promote these estimates to trade signals until they improve net-of-cost out-of-sample expectancy and uncertainty versus the existing baseline. No execution enabled.

These modules are research components, not yet integrated into the production feature pipeline or the decision engine. GARCH forecasts volatility, not direction; HMM states are latent statistical clusters, not guaranteed economic regimes. The next models (Kalman, Bayesian updating and Monte Carlo) remain planned, not implemented.


## Additional quantitative components (2026-11-03)

- [x] Added a causal local-linear-trend Kalman filter; noise scales are estimated from the training slice and test observations are filtered sequentially without smoothing.
- [x] Added a sequential Bayesian odds updater that accepts externally calibrated likelihood ratios. It deliberately does not estimate likelihood ratios from the evaluation sample.
- [x] Added empirical block-bootstrap Monte Carlo barrier simulation with explicit target-first, stop-first and no-touch probabilities; block size is configurable and seeded runs are reproducible.
- [x] Added synthetic tests for Kalman future-perturbation invariance, Bayesian odds arithmetic, and Monte Carlo probability normalization/reproducibility.
- [ ] Verify the latest GitHub Actions test run passes for all five quantitative components.
- [ ] Calibrate Bayesian likelihood ratios and Kalman noise parameters strictly on chronological training/validation data; compare each module with simple baselines.
- [ ] Evaluate the five components individually and incrementally, then only retain those that improve net-of-cost out-of-sample expectancy, drawdown or calibrated risk estimates.

The five methods now exist as independent research utilities, not as an integrated trading model. In particular, Monte Carlo probabilities inherit the empirical bootstrap assumptions, Bayesian updates are only meaningful with calibrated likelihood ratios, and none of these components has demonstrated profitability yet.


## Quantitative model implementation and ablation (2026-11-03)

- [x] Added constrained GARCH(1,1), three-state Gaussian HMM, causal Kalman local-trend filter, sequential Bayesian odds updater and empirical block-bootstrap Monte Carlo barrier simulator as separate research utilities.
- [x] Added synthetic unit tests for stationarity, causal filtering, probability normalization, Bayesian odds arithmetic and reproducible Monte Carlo outputs.
- [x] Added a reproducible chronological GARCH/HMM feature-ablation workflow using the existing Stage 1 feature artifact.
- [x] Derived a three-class directional event target from the already-frozen separate long/short 1R-stop / 3R-target labels; conflicting long-and-short target wins, same-minute ambiguity and incomplete outcomes are excluded.
- [x] Added complete-5m-bucket construction and gap-reset tests; quant features use completed 5m observations and are joined backward with a five-minute staleness limit.
- [ ] Complete the current GARCH/HMM ablation run and review its report. A previous attempt exposed that the Stage 1 artifact contains asymmetric event labels rather than the assumed symmetric label names; the evaluator was corrected to use the actual preregistered labels.
- [ ] If either model improves validation/test classification and calibration, run the exact same out-of-sample predictions through the fee/slippage/funding-aware execution simulator. Classification improvement alone is not a profitability result.
- [ ] Cross-fit training-period regime features before any confirmatory or prospective use; current ablation is exploratory and model parameters are fitted on a contiguous training segment only.
- [ ] Extend the ablation to Kalman and calibrated Bayesian state-conditional probabilities; evaluate Monte Carlo as a separate risk/target probability estimator, not as an assumed alpha source.
- [ ] Add block-bootstrap uncertainty and multiple-comparison controls before selecting any model variant.

**Important:** the five quantitative utilities are implemented, but only GARCH and HMM are currently wired into the new ablation. None has demonstrated incremental net trading profit. No model is approved for paper/live execution.


## GARCH/HMM directional feature ablation — completed 2026-11-03

- [x] Completed the chronological GARCH/HMM ablation using the actual preregistered asymmetric event labels (workflow run 41577802028).
- [x] Baseline, +GARCH, and +GARCH+HMM variants were evaluated for both 1h and 3h outcomes using Logistic Regression and Histogram Gradient Boosting.
- [x] Recorded full findings in [research/QUANT_REGIME_ABLATION_RESULTS.md](research/QUANT_REGIME_ABLATION_RESULTS.md).
- **Decision:** no consistent improvement in balanced accuracy or calibration. Do not add GARCH/HMM as directional alpha features at this stage. No net-PnL simulation was performed in this screening experiment.
- [ ] Evaluate GARCH in its proper volatility-forecasting role against trailing-RV baselines using QLIKE and out-of-sample loss comparisons.
- [ ] Evaluate HMM state persistence / volatility separation and then regime-conditioned target geometry using validation-only selection.
- [ ] Cross-fit training-period regime features before confirmatory use; then run the exact fee/slippage/funding-aware trade simulator with block-bootstrap uncertainty.
- [ ] Only after the above, test whether Kalman, Bayesian updating or Monte Carlo adds incremental value. These three have not yet been evaluated in this ablation.

Workflow: https://github.com/undaviyabrijesh-sys/DMPE/actions/runs/41577802028



## GARCH volatility forecast evaluation — implementation in progress (2026-11-03)

- [x] Added a dedicated QLIKE experiment for 1h and 3h variance forecasts from completed 5-minute BTCUSD bars.
- [x] Benchmarks are horizon-matched trailing realized variance and fixed EWMA(0.94), alongside Gaussian-QMLE GARCH(1,1).
- [x] GARCH parameters are fitted only on the first chronological 70% minus a 3-hour purge; the 15% validation interval is not used for tuning; evaluation starts after a 3-hour test-boundary purge.
- [x] Forecast at close t targets only the next 12 or 36 completed 5-minute returns. Recursive states update only with observations available at each forecast close.
- [x] Longest contiguous complete 5-minute segment is selected; partial buckets and gaps are not filled or bridged.
- [x] Added paired day-block bootstrap intervals for GARCH-minus-baseline QLIKE differences and saved per-timestamp test forecasts.
- [x] Added unit tests for QLIKE behavior, causal GARCH recursion, incomplete 5-minute buckets and bootstrap output.
- [ ] Confirm GitHub Actions tests pass and run the QLIKE workflow against Stage 1 artifact run 41509621537.
- [ ] Review 1h and 3h QLIKE rankings and paired intervals. Retain GARCH for risk forecasting only if it beats both simple baselines robustly; otherwise keep the simpler forecast.
- [ ] Only after this benchmark, consider Student-t innovations or asymmetric EGARCH/GJR-GARCH to address Bitcoin heavy tails and volatility asymmetry; compare each as a separately declared follow-up, not by tuning against this test set.
- **Scope:** volatility forecasting only. This does not change the frozen 1h/3h directional 1R-stop / 3R-target experiment, does not use its prospective holdout for tuning, and does not authorize signals, paper trading or live execution.


### GARCH QMLE implementation audit — correction required before accepting results (2026-11-03)

- [x] Audited the first QLIKE run (41587956808) after implausible forecast errors and found the shared GARCH likelihood scored each return using a variance updated with that same return. This is a timing/lookahead defect in parameter estimation.
- [x] Corrected the Gaussian-QMLE recursion: score return t with variance known through t-1, then update variance using return t for the forecast of t+1.
- [x] Added a synthetic persistent-GARCH recovery regression test (known alpha=0.08, beta=0.90).
- [ ] Re-run all CI and the QLIKE workflow using the corrected likelihood. The first QLIKE artifact and its metrics are invalid and must not be interpreted.
- [ ] Re-run the GARCH/HMM feature ablation after the correction before relying on its GARCH-related comparison; the HMM implementation/results are not changed by this likelihood correction.


### Corrected GARCH QLIKE results — completed 2026-11-03

- [x] Corrected likelihood run completed successfully: workflow run [41588407996](https://github.com/undaviyabrijesh-sys/DMPE/actions/runs/41588407996); report artifact dmpe-garch-qlike-41588407996.
- [x] CI passed: 60 tests, including the synthetic known-parameter GARCH recovery test.
- Evaluation used the longest contiguous complete 5m segment (249,334 returns), with 37,353 1h and 37,329 3h out-of-sample forecasts from 2026-06-25 through 2026-11-02. Training ended 2026-02-15; a 3h purge was applied.
- Corrected Gaussian-QMLE fit: omega=1.79899e-8, alpha=0.11443, beta=0.88456, alpha+beta=0.99899 (near the stationarity cap).
- **1h QLIKE:** GARCH -10.4313; trailing RV - (score 97.8708, affected by one near-zero trailing forecast); EWMA(0.94) -10.3769. GARCH-minus-EWMA paired day-block 95% interval: [-0.0804, -0.0262].
- **3h QLIKE:** GARCH -9.2096; trailing RV -9.0518; EWMA(0.94) -9.1039. GARCH-minus-trailing-RV interval: [-0.2181, -0.0971]; GARCH-minus-EWMA interval: [-0.1530, -0.0527].
- Calibration diagnostic: aggregate forecast/realized variance ratio was 1.20 (1h) and 1.37 (3h) for GARCH, versus approximately 1.00 for both simple baselines. Mean absolute log-variance error was also higher for GARCH (0.85/0.87) than EWMA (0.72/0.69).
- **Interpretation:** GARCH wins on mean QLIKE versus EWMA in both horizons, but over-forecasts aggregate variance and has worse absolute log error. Treat it as a promising conservative risk-forecast candidate, not a calibrated forecast or directional alpha. The 1h trailing-RV comparison is especially sensitive to a near-zero forecast; rely more on the paired GARCH-vs-EWMA comparison.
- [ ] Test Student-t innovations and asymmetric EGARCH/GJR-GARCH as separately declared follow-ups for Bitcoin heavy tails and leverage effects. Do not tune them on the current test segment; use fresh validation/forward data.
- [ ] Review the corrected GARCH/HMM directional ablation rerun (workflow run 41588160449). The earlier run 41577802028 used the defective likelihood and is superseded for any GARCH-related conclusion.
- **Release gate:** no direction prediction, trading signal, paper trading or live execution is approved by this volatility-only result.
