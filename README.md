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
Or start a finite capture in GitHub Actions from **Actions → DMPE live market-data capture → Run workflow** (1–120 minutes). The output is uploaded as an artifact. GitHub Actions is not a permanent WebSocket host; uninterrupted capture needs a VPS or another always-on service.

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

The first reproducible feature workflow reads the historical backfill artifact, profiles each dataset's cadence and largest gaps, then creates a 1-minute research table. It builds 39 candidate point-in-time descriptors, 9 continuous forward-return/excursion targets, and 9 first-touch realized-volatility barrier labels across 15m, 1h and 3h horizons. Each horizon uses a matching trailing realized-volatility scale: 15m, 60m and 180m respectively. The barrier grid tests symmetric distances of 0.75, 1.0 and 1.5 times that horizon-aligned scale; these are research variants, not selected trading parameters.

Candidate descriptors include lagged returns, realized volatility, ATR, normalized moving-average distance, volume surprise, candle shape, UTC time seasonality, mark-price basis/returns, hourly funding-rate changes and hourly open-interest changes. Funding/OI observations are delayed by one full hour plus one minute before joining. Negative funding is preserved as valid data. The minute grid makes missing candles explicit; price is not forward-filled, rolling features require complete lookback windows, and forward labels are null when the future window crosses missing data.

Run **Actions → DMPE feature engineering**. The default source run ID points to the completed historical backfill (41481361783); choose another backfill run ID when using a newer dataset. The workflow uploads:
- `metadata/history_profile.json` and `history_profile.csv`: cadence, missing intervals, largest gaps and exact mark/price timestamp alignment.
- `processed/dmpe_BTCUSD_1m_features.parquet`: features plus separate `target_*` columns.
- `processed/feature_schema.json` and `feature_build_report.json`: feature/label contract, null rates and barrier class balance.

Barrier labels are +1 when the upper horizon-aligned realized-volatility barrier is touched first, -1 when the lower barrier is touched first, and 0 when neither is touched or both are touched in the same minute. The corresponding `barrier_state_*` diagnostics distinguish no-touch (0), ambiguous same-minute touches (2), and invalid/incomplete windows (-2). Exclude state=2 and -2 rows from directional model training; never infer intraminute order from 1-minute OHLC data. The same-minute ambiguity is deliberately not guessed.

The target columns are for research evaluation only. Never include `target_*` columns as model inputs. No predictive model or trading signal is trained in this stage.

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
