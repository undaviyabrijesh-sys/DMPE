# DMPE — Dynamic Market Prediction Engine

Research-first BTC perpetual-futures market-data and prediction project. The priority is a reliable, auditable data pipeline. No order execution is included.

## Initial scope
- Venue: Delta Exchange India; initial product: BTCUSD perpetual
- Historical: BTCUSD 1m traded-price candles; MARK:BTCUSD 1m mark-price candles; FUNDING:BTCUSD and OI:BTCUSD at 1h where history is available
- Live: trades, top-15 L2 order book, 1m candles, ticker (includes OI fields), mark price and funding rate
- Storage: Parquet historical files and raw JSONL live events
- Prediction horizons: 15m, 1h and 3h
- Signal convention: +1 long, 0 flat/exit, -1 short

## Historical backfill
The initial automated GitHub Actions job requests **up to five years** of history and uploads the data as a workflow artifact. Actual coverage depends on what Delta returns for each series; the earliest available timestamps are recorded in the backfill report. The historical candles endpoint has a documented 2,000-candle response cap; the collector pages across time windows and uses a pause/retry policy.

Run locally:
```bash
pip install -r requirements.txt
python scripts/backfill_bundle.py --days 1825
python scripts/audit_history.py --directory data/raw/history
```

## Live capture
Run locally until interrupted:
```bash
python scripts/capture_live.py --symbol BTCUSD
```
Or start a finite capture in GitHub Actions from **Actions → DMPE live market-data capture → Run workflow** (1–120 minutes). For meaningful microstructure research, use a longer session rather than the 5-minute push-triggered smoke test. The workflow uploads both the raw JSONL capture and a derived order-flow feature artifact automatically. GitHub Actions is not a permanent WebSocket host; multi-day continuous capture needs a VPS or another always-on service.

Raw data are intentionally excluded from Git commits. Download the workflow artifacts from the repository's Actions page.

## Streamlit Quant Desk
DMPE includes a local Streamlit research desk with:
- Interactive BTCUSD candlesticks, EMA overlays, volume, multi-horizon returns, ATR and realized-volatility descriptors.
- Historical Parquet/CSV upload and timeframe aggregation (only to coarser timeframes than the source).
- Auto-refreshing live JSONL tape, recent trades and L2 displayed-size imbalance when those events are present.
- Backfill report, candle audit and basic data-integrity panels.
- An explicit model gate showing whether a validated model is registered.

Extract the historical backfill artifact so its `history/` and `metadata/` folders sit under this repository's `data/raw/` directory. To enable the Feature Lab, extract the Stage 1 artifact's `processed/` folder to `stage1_output/processed/` (or point `DMPE_FEATURE_DIR` at the extracted folder). Then launch:
```bash
pip install -r requirements.txt
streamlit run app.py
```
For live events, run the recorder separately in another terminal:
```bash
python scripts/capture_live.py --symbol BTCUSD
```
The live pane reads the JSONL files produced by that recorder. The Streamlit app does not start a recorder, request exchange credentials, or place orders. The app can also be deployed as a private Streamlit service by pointing it at a persistent data directory using the `DMPE_DATA_DIR` environment variable.

**Important:** the current desk is a data and research interface, not a proven trading system. It intentionally displays FLAT (0) until a model passes chronological walk-forward validation and net-of-cost evaluation. EMA, return, volatility and order-book metrics are descriptive candidate features, not trade recommendations.

## Stage 1 — Historical audit and feature engineering

The first reproducible feature workflow reads the historical backfill artifact, profiles each dataset's cadence and largest gaps, then creates a 1-minute research table. It builds 40 candidate point-in-time descriptors, 9 continuous forward-return/excursion targets, 9 symmetric first-touch labels across 15m, 1h and 3h, and separate asymmetric long/short event labels for the preregistered 15m and 1h experiments (1x RV stop / 3x RV target). Each horizon uses a matching trailing realized-volatility scale: 15m, 60m and 180m respectively. The barrier grid tests symmetric distances of 0.75, 1.0 and 1.5 times that horizon-aligned scale; these are research variants, not selected trading parameters.

Candidate descriptors include lagged returns, realized volatility, ATR, normalized moving-average distance, volume surprise, candle shape, UTC time seasonality, mark-price basis/returns, hourly funding-rate changes and hourly open-interest changes. Exchange candle timestamps denote minute OPEN, but each processed feature row is timestamped at the completed minute CLOSE (source timestamp + 1 minute). The model may only act after that close; target windows begin with the following minute. This avoids using a completed candle's high/low/close at the candle open. Funding/OI observations are delayed by one full hour plus one minute before joining. Negative funding is preserved as valid data. The minute grid makes missing candles explicit; price is not forward-filled, rolling features require complete lookback windows, and forward labels are null when the future window crosses missing data.

Run **Actions → DMPE feature engineering**. The default source run ID points to the completed historical backfill (41481361783); choose another backfill run ID when using a newer dataset. The workflow uploads:
- `metadata/history_profile.json` and `history_profile.csv`: cadence, missing intervals, largest gaps and exact mark/price timestamp alignment.
- `processed/dmpe_BTCUSD_1m_features.parquet`: features plus separate `target_*` columns.
- `processed/feature_schema.json` and `feature_build_report.json`: feature/label contract, timestamp semantics, null rates and barrier class balance.

Barrier labels are +1 when the upper horizon-aligned realized-volatility barrier is touched first, -1 when the lower barrier is touched first, and 0 when neither is touched or both are touched in the same minute. The corresponding `barrier_state_*` diagnostics distinguish no-touch (0), ambiguous same-minute touches (2), and invalid/incomplete windows (-2). Exclude state=2 and -2 rows from directional model training; never infer intraminute order from 1-minute OHLC data. The same-minute ambiguity is deliberately not guessed.

The target columns are for research evaluation only. Never include `target_*` columns as model inputs. No predictive model or trading signal is trained in this stage.

## Stage 2 — Chronological baseline evaluation

Run **Actions → DMPE baseline model evaluation** after a successful Stage 1 feature-engineering workflow. The default source feature run ID is `41490148033`; enter a newer feature-workflow run ID when using a newer artifact. The job compares a prior-class baseline, an experimental logistic-regression classifier and a regularized histogram-gradient-boosting classifier for each 15m/1h/3h horizon and each 0.75x/1.0x/1.5x horizon-aligned volatility barrier.

Evaluation safeguards:
- One UTC-aligned sample per target horizon, so adjacent labels in the sample do not overlap.
- Chronological 70/15/15 train/validation/test periods, with one full target-horizon embargo on each side of the split boundaries.
- Imputation, scaling and model fitting use the training period only.
- Confidence threshold is selected on validation only; final metrics are reported on the untouched test period.
- A long/short call must be the highest-probability class (it must beat FLAT as well as the opposite direction) and clear the validation-selected threshold.
- Ambiguous same-minute barriers and incomplete windows are excluded from directional fitting.
- Future target columns are never model inputs.

The output artifact `dmpe-baseline-evaluation` contains the JSON/CSV scorecard and experimental model bundles. Metrics include accuracy, balanced accuracy, macro-F1, log loss, multiclass Brier score, confusion matrix, and confidence-gated directional precision/coverage.

This is only the first classification baseline. It is **not** a backtest of realized trading P&L and does not include fills, fees, spread, slippage, funding or execution latency. A model is not eligible for live signals merely because its classification metrics look promising. The next gate is a trade simulator with realistic execution assumptions and additional walk-forward periods.

## Live microstructure feature capture

The live recorder also subscribes to Delta's `system_status` and `product_updates` public channels. These are operational safety feeds, not alpha features.

After collecting raw JSONL data, build minute-level trade and L2 descriptors locally:
```bash
python scripts/build_live_orderflow_dataset.py --input-dir data/raw/live --output-dir data/processed
```
This produces `dmpe_BTCUSD_live_orderflow_1m.parquet`, CSV and a report. It also extracts `funding_rate` snapshots into separate `dmpe_BTCUSD_funding_observations.parquet/csv` files, retaining the observed rate and Delta's explicit `next_funding_realization` timestamp. Those observations are not charged in the simulator until the settlement cashflow convention is reconciled. It summarizes aggressive buy/sell contract volume, signed volume imbalance, trade VWAP/range/arrival pace, top-1/5/10/15 order-book depth imbalance, spread and microprice pressure. Unknown trade-role codes remain unclassified and missing event minutes are not filled.

Delta's public trade event documents `r` as `buyer_role`: `t` means the buyer was taker and `m` means the buyer was maker. DMPE therefore maps `t` to buyer-taker (aggressive buy) and `m` to seller-taker (aggressive sell); other values are not guessed. The event data are microsecond-timestamped. The L2 public channel provides top-15 bid/ask price-size levels. See the [official Delta WebSocket public-channel documentation](https://docs.delta.exchange/#trades).

Alternatively, run **Actions → DMPE live order-flow feature build** and provide the run ID of a completed live-capture workflow. This creates the same Parquet/CSV/report bundle as a 30-day artifact. Extract the resulting Parquet into `orderflow_output/processed/`; the Streamlit Live Tape panel will show its latest aggregated pressure, depth and spread readings.

To join live features to the candle feature dataset without lookahead or forward-fill:
```bash
python scripts/merge_live_orderflow_features.py \
  --feature-file stage1_output/processed/dmpe_BTCUSD_1m_features.parquet \
  --orderflow-file orderflow_output/processed/dmpe_BTCUSD_live_orderflow_1m.parquet \
  --schema-file stage1_output/processed/feature_schema.json \
  --output-dir orderflow_merged
```
The merge is exact on UTC minute-close timestamps. Missing capture rows remain missing; rolling 5m/15m descriptors require complete windows. Training must use a chronological period after the actual recorder start and should include feed-coverage checks.

These microstructure features start only when the WebSocket recorder is running; the historical candle dataset cannot reconstruct past trades or order-book states. They must be collected over time and independently evaluated before being added to a trading model.

## Stage 3 — Repeated walk-forward confirmation

Run **Actions → DMPE repeated walk-forward evaluation** after Stage 1 is complete. The default feature artifact source is run `41490148033`; choose the appropriate Stage 1 run ID when the feature dataset is rebuilt.

This experiment evaluates the same regularized histogram-gradient-boosting baseline over four expanding-window folds. Each fold has a 60-day validation block and a following 60-day untouched test block. One full prediction horizon is purged/embargoed at each boundary. The model is fitted only on data before validation; its confidence threshold is selected on validation only; then the following test block is evaluated without tuning.

Outputs:
- `walk_forward_report.json`: fold-level and combined test metrics, class counts, threshold selections, confidence-gated signal rates and uncertainty intervals.
- `walk_forward_summary.csv`: flattened variant/fold summary.
- `walk_forward_test_signals.csv`: timestamps, predicted directions, true outcomes and class probabilities for test-period directional calls.

This is a stronger confirmation than a single 70/15/15 holdout, but still not an execution backtest. Test windows are sequential, and later training folds may include earlier folds' already-observed test history. The report explicitly documents this. We must still test fee-aware fills, slippage, funding, market impact and block-bootstrap uncertainty before deciding whether any pattern is tradeable.

## Stage 4 — Execution-cost simulation

Run **Actions → DMPE execution-cost simulation** after the repeated walk-forward evaluation completes. The workflow downloads three immutable inputs: the walk-forward out-of-sample signal artifact, the Stage 1 feature dataset, and the original historical BTCUSD 1-minute candles. You can change all three run IDs from the workflow form.

The simulator trades each horizon/barrier variant separately and tests 0, 1, 2, 5 and 10 bps of assumed slippage per side. It enters at the next 1-minute candle open after a signal, anchors barrier geometry to the completed signal candle close (matching the label reference), applies adverse entry/exit slippage, and charges taker commission on both entry and exit. It uses one BTCUSD contract by default (lot size 0.001 BTC); change the script arguments to scale contract count.

The default fee assumption is standard BTCUSD perpetual taker commission of 0.05% plus 18% GST on the fee (= 5.9 bps per side). Current published contract specifications show BTCUSD lot size 0.001 BTC, 0.05% taker commission, and 8-hour funding interval. The public fee page also confirms GST on trading fees. Check the fee page before any future live deployment because account promotions or fee schedules may differ:
- [BTCUSD contract specifications](https://www.delta.exchange/contracts)
- [Delta trading fee schedule](https://www.delta.exchange/fees?tab=trading-fees)

Outputs in `dmpe-execution-cost-simulation`:
- `execution_sensitivity.csv`: P&L and risk metrics for each horizon, barrier multiple and slippage scenario.
- `execution_trades.csv`: individual simulated fills, exits, fees and returns.
- `execution_simulation.json`: assumptions, exclusions and full scorecard.

Important exclusions in version 1: funding payments, account INR conversion, leverage, liquidation, queue position, market impact and measured historical spread. Funding is deliberately excluded until the meaning/timing of historical funding candles is verified against actual settlement realization times. The configured slippage values are scenarios, not measured historical estimates. P&L is USD-equivalent for the configured contract size, fixed-size/non-compounded, and is not a live fill replay. No orders are submitted. Do not approve any strategy from this report until funding and realistic execution assumptions are included and results remain robust on additional unseen data.

## Install locally
Python 3.11+ recommended:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src pytest -q
```

## Research safeguards
1. Use chronological splits and walk-forward validation; never randomly shuffle time-series rows.
2. Preserve raw events before feature engineering.
3. Include fees, spread, slippage, funding and latency in evaluations.
4. Probabilities are not guarantees; risk controls remain separate from the model.
5. Never commit API keys, secrets, private fills or personal data. Current collectors use public endpoints only.

## Layout
- `.github/workflows/tests.yml`: automated tests
- `.github/workflows/historical-backfill.yml`: 5-year historical data job and artifact
- `.github/workflows/live-capture.yml`: manually started finite live recorder
- `app.py`: Streamlit Quant Desk
- `scripts/`: backfill, live capture and quality audit
- `src/dmpe/`: REST client and WebSocket recorder
- `data/raw/`: local market data, intentionally excluded from Git

## Limitations
Historical trades and historical order-book depth are not returned by the candle endpoint. The WebSocket recorder starts collecting those events only when it is running. Historical funding and OI symbols may have shorter coverage than price candles. The first backfill report will show which series were actually available. No model training or live trading is enabled.


## Always-on live recorder VPS

GitHub Actions is intended for finite captures, not a persistent WebSocket process. A hardened systemd unit and installation guide are included in `deploy/vps/`:
- `deploy/vps/dmpe-live-recorder.service`
- `deploy/vps/README.md`

The service runs the public BTCUSD recorder under a non-root account, restarts after disconnects, and writes daily JSONL files to persistent storage. It requires no API keys and cannot place orders. Provisioning the VPS and configuring durable off-host backups still require the account owner.


## Binance historical aggregate-trade proxy

Binance's public data archive is reachable at the archive root and its official repository documents USD-M futures aggTrades with timestamp and buyer-maker flag. DMPE now includes a checksum-verified monthly downloader and a minute-feature builder:

- `scripts/download_binance_aggtrades.py` downloads one BTCUSDT USD-M futures monthly archive and verifies its SHA-256 checksum.
- `scripts/build_binance_flow_features.py` creates UTC minute-close features: signed base-volume imbalance, trade count, large-print notional share (fixed $100,000 notional threshold), signed-notional imbalance and 5m flow acceleration.
- `scripts/merge_binance_flow_features.py` performs a one-to-one exact UTC close-timestamp join. Unmatched minutes remain missing; no as-of join or forward-fill is permitted.
- `.github/workflows/binance-aggtrades-monthly.yml` runs a month at a time and uploads both the verified source ZIP/checksum and derived Parquet/report artifacts for 90 days. Dispatch it with a year and month.

Example local workflow:
```bash
python scripts/download_binance_aggtrades.py --symbol BTCUSDT --year 2024 --month 1 --output-dir binance_input
python scripts/build_binance_flow_features.py --archive binance_input/BTCUSDT-aggTrades-2024-01.zip --year 2024 --month 1 --output-dir binance_output
python scripts/merge_binance_flow_features.py --feature-file stage1_output/processed/dmpe_BTCUSD_1m_features.parquet --flow-file binance_output/binance_BTCUSDT_aggtrade_flow_1m_2024-01.parquet --output-dir binance_merged
```

**Important:** this is Binance aggressor-flow proxy data, not historical Delta order flow. BTC cross-venue price correlation does not guarantee that signed-flow features transfer to Delta. Binance aggTrades contain no historical L2 queue position and cannot calibrate Delta maker fills. Keep the Binance proxy and Delta live-flow columns distinct; only forward Delta capture can validate venue transfer and fill assumptions.

The revised preregistration in `research/PREREGISTERED_ASYMMETRIC_EXPERIMENT.md` freezes exactly two confirmatory horizons: 1h and 3h, each with a 1R stop and 3R target. The 15m horizon is excluded. Existing inspected Stage 3/4 test blocks are disqualified; the prospective final holdout remains 2026-11-03 through 2027-01-02 UTC.


## 24-month Binance proxy backfill

The repository now includes `.github/workflows/binance-flow-backfill-24m.yml`, which downloads BTCUSDT USD-M aggTrades for **2024-11 through 2026-10** in monthly jobs (six at a time), verifies each monthly checksum, creates the predeclared 1m flow features, and combines the results only if the final minute grid is complete and gap-free. Monthly source ZIPs are discarded after processing in this bulk workflow to control artifact storage; the exact public archive URL, SHA-256, byte count and checksum provenance are retained per month. The single-month workflow remains available when an actual raw ZIP artifact is required.

After the combined artifact is produced, join it to candle features using:
```bash
python scripts/merge_binance_flow_features.py \
  --feature-file stage1_output/processed/dmpe_BTCUSD_1m_features.parquet \
  --flow-file binance_BTCUSDT_aggtrade_flow_1m_2024-11_to_2026-10.parquet \
  --output-dir binance_merged
```


## Stage 5 — Multi-timeframe structure and entry research

DMPE now has a separate experimental Structure & Entry Engine alongside (not instead of) the predictive engine. It builds completed 5m, 15m, 1h and 4h bars without filling missing candles, exposes swing pivots only after right-side confirmation, derives 4h/1h directional context and 15m setup context, then tests 5m pullback, liquidity-sweep and breakout/retest candidates.

Run **Actions → DMPE multi-timeframe structure evaluation** for the original gated modules, or **Actions → DMPE MTF gate ablation** to compare the original 4h/1h/15m context, 15m-only context and no directional context gate. The scripts are:
- `scripts/evaluate_structure_engine.py`
- `scripts/evaluate_structure_gate_ablation.py`

The replay requires an exact next-1m open at the signal availability timestamp. It censors positions crossing missing 1m data and positions still open at end-of-file; censored trades are excluded from performance metrics. It assumes 5.9 bps taker fee and 1 bp slippage per side. Funding, measured spread, market impact and latency are not yet included.

**Current status:** The first historical replay and post-hoc gate ablation are negative after costs. No tested variant passed the net expectancy / profit-factor gate. The structure engine is a research layer only; it is not integrated into production signal logic and does not place orders. The current evidence does not justify loosening HTF filters merely to increase trade frequency. See [the full MTF results and limitations](research/MTF_STRUCTURE_RESULTS.md) and [the ablation protocol](research/MTF_GATE_ABLATION_PROTOCOL.md).

The intended DMPE architecture remains four distinct layers:
1. Market intelligence.
2. Predictive engine.
3. Structure and entry engine.
4. Decision engine, which may combine only independently validated evidence and can choose FLAT.

