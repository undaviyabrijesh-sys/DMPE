# DMPE — Dynamic Market Prediction Engine

Research-first BTC perpetual-futures market-data and prediction project. The first milestone is a reliable, auditable data pipeline. No order execution is included.

## Initial scope
- Venue: Delta Exchange India
- Initial contract: BTCUSD (verify product metadata before changing symbols)
- Historical data: 1-minute OHLCV; optional mark, funding and OI series only where confirmed available
- Live data: public trades, L2 order-book updates and 1-minute candles
- Storage: Parquet (historical) and JSONL (raw live events)
- Planned horizons: 15m, 1h and 3h
- Planned signals: +1 long, 0 flat/exit, -1 short

## Research safeguards
1. Use chronological splits and walk-forward validation; never randomly shuffle time-series rows.
2. Preserve raw events before feature engineering.
3. Include fees, spread, slippage, funding and latency in evaluations.
4. Probabilities are not guarantees; risk controls remain separate from the model.
5. Never commit API keys, secrets, private fills or personal data. This bootstrap uses public endpoints only.

## Install
Python 3.11+ recommended.
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Historical candle backfill
```bash
python scripts/backfill_candles.py --symbol BTCUSD --resolution 1m --days 30
```
Optional API series may include MARK/FUNDING/OI-prefixed symbols, but availability must be checked for the product and date range.

## Live public feed recorder
```bash
python scripts/capture_live.py --symbol BTCUSD
```
Records public trades, L2 and 1-minute candles to daily JSONL files. This is a foreground process; use a continuously running VPS/server for uninterrupted capture. GitHub Actions is not a persistent WebSocket host.

## Validate a candle file
```bash
python scripts/validate_candles.py data/raw/candles/BTCUSD_1m.parquet
```

## Layout
- `.github/workflows/tests.yml`: automated tests
- `scripts/`: backfill, live capture and validation commands
- `src/dmpe/`: Delta REST client and live recorder
- `data/raw/`: local market data, intentionally excluded from Git

## Current limitations
This is a data-pipeline bootstrap, not a profitable trading system. REST candles are fetched in bounded windows. Live trade/order-book history starts only when recording begins. Historical tick/order-book depth and liquidation/positioning feeds require separate source research. No order execution is implemented.
