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
- `scripts/`: backfill, live capture and quality audit
- `src/dmpe/`: REST client and WebSocket recorder
- `data/raw/`: local market data, intentionally excluded from Git

## Limitations
Historical trades and historical order-book depth are not returned by the candle endpoint. The WebSocket recorder starts collecting those events only when it is running. Historical funding and OI symbols may have shorter coverage than price candles. The first backfill report will show which series were actually available. No model training or live trading is enabled.
