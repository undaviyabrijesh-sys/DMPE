# DMPE MTF gate ablation — exploratory protocol

## Status and interpretation

This is a **post-hoc diagnostic**, written after the first gated MTF replay was observed. It is not a preregistered confirmatory experiment and cannot be used to select a strategy for deployment. Its only question is whether the existing 5-minute entry patterns behave differently when the higher-timeframe directional gates are removed.

Any variant that looks promising must be frozen and evaluated on a new, untouched chronological period with the corrected gap-aware replay, realistic funding and execution assumptions, and the original predictive-engine comparison.

## Fixed setup definitions

The 5-minute entry setup definitions remain unchanged from the initial MTF experiment:

- Pullback continuation: EMA20 retest/reclaim and prior-bar high/low break.
- Liquidity sweep: 20-bar prior high/low sweep and close back inside the prior range.
- Breakout/retest: recent range break followed by the existing retest/reclaim condition.

The stop, target and replay assumptions also remain unchanged for this diagnostic:
- Stop: completed signal 5-minute candle extreme plus/minus 0.1 × rolling 20-bar 5-minute ATR.
- Entry: first 1-minute open at or after the completed 5-minute signal close.
- Same-minute stop/target ambiguity: stop first.
- One active position per module / gate mode / R:R replay.
- Maximum holding period: 60 observed 1-minute rows.
- Taker fee: 5.9 bps per side; slippage scenario: 1 bp per side.
- Funding is not included.

## Context modes

1. **4h1h15m** — original directional gate: 4-hour and 1-hour bias must agree; the 15-minute bias must also agree for modules that originally used it. The sweep module retains its original 4h/1h gate, matching the frozen engine definition.
2. **15m_only** — remove the 4-hour and 1-hour directional gate; retain the 15-minute bias for all three 5-minute entry patterns.
3. **ungated** — retain the same 5-minute setup triggers but remove all directional structure-bias gates.

Three independent modules are scored separately. The combined-consensus diagnostic is long only when at least one module is long and none is short, short only when at least one module is short and none is long; opposing same-bar module signals are discarded as flat. This avoids a long-first tie-break.

## Required metrics

Report candidate count, executed trade count, trades/day, win rate after costs, gross and net P&L points, gross and net expectancy in R, gross and net profit factor, average and median cost per trade, and net-R maximum drawdown.

Do not rank variants by win rate alone. A variant is not useful unless net expectancy, profit factor, drawdown, stability and sample size support it after realistic costs.

## Known limitations

The current first-pass replay advances across observed 1-minute rows and does not yet censor a position if a missing minute occurs during its holding window. The source price history contains missing intervals. Therefore this diagnostic is explicitly non-final.

Funding, latency, market impact, measured spread, liquidation and account-specific contract sizing are also not represented. The 5.9 bps fee and 1 bp slippage are assumptions, not measured fills.

## Decision rule

- Do not promote any gate mode based on this post-hoc comparison.
- Do not tune thresholds or entry definitions on these results.
- If any mode appears materially better, freeze it and test it on a new untouched period using a gap-aware simulator.
- Compare prediction-only, structure-only and combined predictive-plus-structure evidence before deciding whether the structure layer belongs in DMPE's decision engine.
- Keep FLAT as a valid outcome. No minimum daily trade requirement and no live orders.
