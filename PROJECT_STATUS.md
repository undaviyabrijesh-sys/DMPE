# DMPE project status

## Milestone 0 — Data-pipeline bootstrap
- [x] Private GitHub repository and project layout
- [x] Historical REST client with 2,000-candle paging and rate-limit backoff
- [x] Historical bundle job for traded price, mark price, funding and open interest
- [x] Live public recorder for trades, L2, candles, ticker/OI, mark price and funding
- [x] Daily JSONL raw event storage
- [x] Candle data-quality audit
- [x] GitHub Actions workflows for tests, historical backfill and finite live capture
- [ ] Verify historical API availability and row coverage from the first workflow artifact
- [x] Finite live capture completed and event types observed in a workflow artifact
- [ ] Continue validating event field semantics, reconnect gaps and timestamp behavior

## Milestone 1 — Streamlit research desk
- [x] Interactive candle/volume chart with timeframe aggregation and basic market descriptors
- [x] Local Parquet/CSV upload, dataset selection and CSV export
- [x] Auto-refreshing live JSONL tape with recent trade display and best-effort L2 size imbalance
- [x] Backfill/audit report panels and basic data-integrity checks
- [x] Explicit model/algorithm release gate; order execution remains disabled
- [ ] Smoke-test the app against extracted historical and live artifacts
- [ ] Deploy to a persistent Streamlit host after local verification

## Initial historical request
- Five years requested by default (1825 days).
- BTCUSD and MARK:BTCUSD: 1-minute bars.
- FUNDING:BTCUSD and OI:BTCUSD: 1-hour bars.
- Actual coverage must be read from the report; no minimum history is assumed.

## Next milestones
1. Verify downloaded history and run the data audit.
2. Validate raw live event schema and reconstruct trade aggressor flow/order-book changes only after checking exchange field semantics.
3. Determine whether additional third-party historical tick/order-book sources are required.
4. Define event-based labels, then build leak-safe chronological feature sets.
5. Compare baseline models using walk-forward validation and net-of-cost metrics.
6. Register only validated models in the desk; add risk controls and paper-trading before considering execution.

No order execution is enabled. Research signals are not trading instructions.
