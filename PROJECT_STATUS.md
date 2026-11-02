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
- [ ] Start first finite live capture and inspect event schema/gaps

## Initial historical request
- Five years requested by default (1825 days).
- BTCUSD and MARK:BTCUSD: 1-minute bars.
- FUNDING:BTCUSD and OI:BTCUSD: 1-hour bars.
- Actual coverage must be read from the report; no minimum history is assumed.

## Next milestones
1. Audit gaps, duplicates, timestamps and data cadence.
2. Reconstruct trade aggressor flow and order-book imbalance from raw live events.
3. Determine whether additional third-party historical tick/order-book sources are required.
4. Define event-based labels, then build leak-safe chronological feature sets.
5. Compare baseline models using walk-forward validation and net-of-cost metrics.

No order execution is enabled. Research signals are not trading instructions.
