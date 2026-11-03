# DMPE multi-timeframe market-structure experiment

**Status:** Initial rule definitions frozen for engineering and exploratory evaluation; not yet a validated strategy.  
**Market:** Delta Exchange India BTCUSD perpetual.  
**Research-only:** no exchange credentials, order placement or live execution.

## Objective

Test whether a discretionary-style hierarchy improves DMPE beyond the existing prediction engine:

- **4H + 1H:** directional market structure and regime gate.
- **15m:** setup context.
- **5m:** candidate entry trigger.
- **1m:** source data only; not used to pretend intrabar execution is known.

This engine remains separate from the statistical prediction engine. Results must be compared independently before considering any ensemble.

## Point-in-time and data integrity rules

- Source 1m timestamps are candle OPEN times. Resampled bars are timestamped at their CLOSE / availability time.
- Only complete resampled buckets containing every expected 1m source candle are retained. Missing minutes invalidate the affected bucket; no synthetic OHLC or forward-fill.
- Swing pivots use 3 bars to the left and 2 bars to the right. A pivot is visible only after the two right-side confirmation bars have completed.
- HTF context is joined backward by availability timestamp. A 5m decision cannot see an unfinished 15m, 1h or 4h candle.
- No future high/low/close, centered rolling value or unconfirmed swing may be used at decision time.
- Signals are generated at the completed 5m close and any hypothetical execution must occur no earlier than the next available 1m open.

## Frozen first-pass rules

### Directional gate
A directional HTF bias requires both 1h and 4h structure to agree. Structure is bullish when the latest confirmed swing high and low are both higher than their previous same-type swings, with close and EMA20 above EMA50. Bearish is the symmetric condition. Otherwise bias is neutral and directional entries are blocked.

### Setup context
15m structure bias must agree with the 1h/4h gate for pullback and breakout/retest modules. The liquidity-sweep module requires 1h/4h agreement and a close back inside the prior 20-bar 5m range.

### Entry modules (independent experiments)
1. **Pullback continuation:** 5m EMA20 retest/reclaim and close beyond prior 5m high (long); inverse for short.
2. **Liquidity sweep/reversal:** sweep the prior 20 completed 5m-bar low/high and close back inside that level, with HTF bias agreement.
3. **Breakout/retest:** recent 5m close beyond the prior 20-bar range, retest of the broken boundary, then close back in the breakout direction; 1h/4h and 15m must agree.

These are deliberately simple, deterministic first-pass definitions. They are not discretionary SMC labels and must not be silently modified after inspecting test results.

## Evaluation plan

1. Unit-test resampling completeness, pivot confirmation timing, and HTF as-of alignment.
2. Run each module independently, then the combined gate, on chronological data.
3. Use next-1m-open entries; model stop beyond the signal/setup swing with a predeclared ATR buffer and compare fixed 1R/2R/3R exits as separately registered experiments.
4. Include taker fees, GST, spread, slippage, funding at verified settlement times and conservative same-minute stop-first handling. Do not count candle touches as maker fills.
5. Report trade count, trades/day, win rate, expectancy in R, profit factor, max drawdown, exposure, net P&L and performance by trend/range and volatility regime.
6. Compare against (a) no-trade baseline, (b) existing DMPE prediction-only outputs and (c) each entry module without HTF gate.
7. Preserve a chronological untouched holdout. Do not tune thresholds, pivots, lookbacks, stop rules or module definitions on it.

## Release gate

No module becomes a live signal unless it demonstrates positive net expectancy after realistic costs, stable out-of-sample performance and adequate trade count with uncertainty intervals. A high win rate alone is insufficient. The existing DMPE desk remains FLAT until its independent release criteria are met.
