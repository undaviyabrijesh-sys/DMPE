# DMPE project status

## Milestone 0 — Bootstrap
- [x] Project layout and research guardrails
- [x] Delta Exchange India historical candle client
- [x] Live public feed recorder (trades, L2 top-15, 1m candles)
- [x] Basic candle validation script
- [x] GitHub Actions test workflow
- [x] Publish into dedicated private GitHub repository
- [ ] Run first live capture and inspect schema/gaps
- [ ] Build ingestion monitoring and raw-feed reconciliation

## Milestone 1 — Historical data audit
- [ ] Retrieve 1m BTCUSD candles across available history
- [ ] Retrieve mark-price, funding and open-interest series where supported
- [ ] Audit missing intervals, duplicate bars and contract metadata
- [ ] Determine historical availability of trade-by-trade and order-book data

## Milestone 2 — Features and labels
- [ ] Define event-based labels before model selection
- [ ] Build flow, volatility, trend, liquidity and positioning features
- [ ] Use chronological and walk-forward validation only

## Milestone 3 — Models and evaluation
- [ ] Establish simple baselines
- [ ] Compare calibrated logistic regression and gradient boosting
- [ ] Evaluate net expectancy, profit factor, drawdown, calibration and turnover
- [ ] Paper-trade only after robust out-of-sample validation

No live order execution is enabled.
