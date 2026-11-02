# DMPE preregistration: Binance trade-flow + asymmetric event model

**Registration timestamp:** 2026-11-02 19:22:48 UTC  
**Status:** Frozen before evaluation of any test block for this revised experiment.  
**Market to trade:** Delta Exchange India BTCUSD perpetual.  
**Research proxy:** Binance USD-M BTCUSDT perpetual aggregate trades.  
**Execution:** Research and paper observation only; no API credentials, order placement, or live execution.

## Hypothesis and venue-transfer limitation

Historical Binance USD-M BTCUSDT aggregate trades provide trade-level aggressor-side information and can extend signed-flow history beyond the short Delta WebSocket capture. Binance isBuyerMaker=true means the buyer was maker, hence the seller was the aggressor; false means the buyer was aggressor. These prints can support minute-level signed volume, count, large-trade share and flow-acceleration features.

This is a cross-venue proxy, not Delta historical order flow. BTC prices co-move across venues, but aggressor flow, liquidity, basis, participant mix, contract specifications and latency can differ. Binance-derived flow must be labelled as proxy data, never as observed Delta flow. Final trading conclusions require forward Delta event capture and fill calibration. Historical Binance aggTrades do not contain historical L2 queue position.

## Exactly two confirmatory variants

| Variant | Horizon | Long barrier | Short barrier | Gross RR |
|---|---:|---|---|---:|
| A (primary) | 1 hour | +3x horizon RV target / -1x stop | -3x target / +1x stop | 3:1 |
| B (secondary) | 3 hours | +3x horizon RV target / -1x stop | -3x target / +1x stop | 3:1 |

- 15-minute horizon is excluded because fee drag is excessive in prior replay.
- No other horizon, stop/target ratio, feature set, model family, threshold grid, or variant is confirmatory.
- Long and short are separate probability tasks within each horizon. At most one direction may be selected per timestamp.
- Volatility scale is trailing completed horizon-aligned realized volatility available at signal time: 60m RV for 1h; 180m RV for 3h.
- For each direction: +1 = target touched first, -1 = stop touched first, 0 = neither touched by horizon. Same-minute target/stop touches are ambiguous and excluded; incomplete future windows are invalid.
- Barriers are referenced to completed signal-candle close. Entry can occur no earlier than the next minute open.

## Data and point-in-time rules

- Binance USD-M BTCUSDT monthly/daily aggTrades are the historical trade-flow source. Preserve source archives, checksums, download manifest, source timestamps, and parser version.
- Aggregate trade timestamp is assigned to its UTC minute; output feature timestamp is minute-open + 1 minute, matching DMPE completed-candle-close timestamps.
- Join candle features and Binance flow only on exact equal UTC close timestamps. No backward/forward/as-of join and no forward-fill.
- isBuyerMaker=true maps to negative signed quantity/notional (aggressive sell); false maps to positive (aggressive buy).
- Predeclared features: signed base-volume imbalance, trade count, large-trade notional share using fixed $100,000 aggregate-print notional threshold, and short-window flow acceleration (current trailing 5m signed-notional imbalance minus the immediately preceding 5m imbalance). Additional feature discovery is exploratory only.
- Monthly files must pass checksum verification and expected timestamp coverage checks. Incomplete months are not silently treated as zero-flow months.
- Binance proxy coverage and Delta live coverage are separate feature families with separate observed/missing flags.

## Model selection, costs and reporting

- Train chronologically before the final holdout, with 3-hour purge/embargo at boundaries.
- Thresholds are selected on validation only using estimated net expected value under frozen cost/fill assumptions. No test-set threshold selection.
- Passive entries must have explicit no-fill outcomes. A candle touch is not a maker fill; queue and adverse-selection calibration requires Delta L2/trade capture. Until calibrated, report pessimistic scenarios and do not call the proxy replay a realistic maker backtest.
- Include maker/taker fees plus GST, spread, slippage, funding at verified settlement timestamps, INR conversion and overlapping-position constraints. Funding cashflow sign/timing must first be reconciled to exchange records.
- Report trade-level net expectancy, win rate, PF, max drawdown, turnover, fill/no-fill rate, adverse selection, calibration, Brier/log loss and 95% day-block-bootstrap intervals.
- Predeclare trend/range and low/medium/high trailing-volatility regimes using past-only data.
- Apply family-wise correction across the two horizons (White's Reality Check or documented Deflated Sharpe/equivalent). Report both variants, not only the winner.

## Holdout and multiple-testing protection

- Previously inspected Stage 3/Stage 4 historical test blocks are disqualified for this revised hypothesis and may be used only for debugging or exploratory diagnostics.
- Prospective untouched final holdout: **2026-11-03 00:00 UTC through 2027-01-02 00:00 UTC**. If Binance archive or Delta event capture has gaps, extend the period; do not patch missing event data or inspect partial results to tune.
- Before opening the holdout, freeze source-data manifest, feature definitions, model code, hyperparameters, validation threshold, execution assumptions and analysis code.
- No test block is to be opened until those items are frozen.

## Deployment gates

1. Archive download/checksum, completeness, timestamp and exact-join tests pass.
2. Continuous Delta WS capture has heartbeat/gap monitoring and durable backups for multiple weeks.
3. Funding and fee conventions are reconciled against actual account/exchange records.
4. The prospective holdout shows positive net expectancy after all costs, with uncertainty and family-wise correction reported.
5. Only then consider 4–8 weeks of paper observation. No real-money execution is authorized.
6. Fractional Kelly remains disabled until calibrated probabilities are stable. Any later sizing requires a hard cap, daily-loss/drawdown kill switch and exchange-status kill switch.

A failed gate is a valid research result. A Binance proxy signal is not proof of Delta execution edge.