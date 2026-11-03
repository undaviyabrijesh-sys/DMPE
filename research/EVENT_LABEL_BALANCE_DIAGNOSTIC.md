# DMPE exploratory first-touch event-label balance diagnostic

**Status:** Exploratory diagnostic; not a confirmatory strategy experiment.  
**Market:** BTCUSD perpetual research series.  
**Source:** Frozen Stage 1 feature artifact from run 41509621537.  
**Execution:** No orders, no paper-trading promotion, no prospective holdout access.

## Question

Does the current 1h/3h directional class imbalance arise partly from the 1R-stop / 3R-target payoff geometry, and how do first-touch event frequencies vary across a small, predeclared symmetric barrier grid?

## Frozen diagnostic grid

| Horizon | Barrier multiples of completed horizon-aligned RV |
|---|---|
| 1 hour | 0.50x, 0.75x, 1.00x, 1.50x, 2.00x |
| 3 hours | 0.50x, 0.75x, 1.00x, 1.50x, 2.00x |

This is a label-distribution sensitivity diagnostic only. The grid is not a search for a profitable stop/target setting. The existing confirmatory asymmetric 1R-stop / 3R-target labels remain unchanged.

## Label and sampling rules

- Reuse the existing symmetric first-touch label implementation.
- +1 means the upper barrier was touched first; -1 means the lower barrier was touched first; 0 means neither barrier was touched by the horizon.
- If both barriers are first touched in the same 1-minute candle, retain an ambiguous diagnostic state; do not infer intraminute order from OHLC.
- Incomplete future windows and windows containing missing OHLC candles remain invalid.
- Use only complete 1-minute timestamp grids with gaps represented as NaN; never compress time or fill candles.
- Sample one timestamp per horizon, aligned to UTC horizon boundaries, so forward outcome windows do not overlap.
- Report chronological 70/15/15 partitions with one horizon-spaced sample purged at each boundary.
- Report up/down counts, no-touch share, ambiguity, invalid rows, directional-event rate and median/mean bars to first touch.

## Interpretation guardrails

- No predictive models, feature selection, threshold selection, P&L calculation or execution simulation are performed.
- Do not choose a trading barrier solely because it creates more directional labels.
- Any candidate label definition must subsequently be preregistered and evaluated with a new chronological protocol, calibration checks and realistic fee/slippage/funding assumptions.
- The prospective holdout beginning 2026-11-03 00:00 UTC is not accessed or tuned against.
- A successful workflow indicates reproducibility, not trading profitability.

## Decision rule

The output is used to decide whether the severe neutral-class concentration is primarily a consequence of the current 1R/3R asymmetric geometry or persists under symmetric first-touch event definitions. It cannot approve a model or strategy.
