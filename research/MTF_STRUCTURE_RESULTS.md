# DMPE multi-timeframe structure research — first evaluation

**Status: exploratory and not approved for trading.** This report records the completed first replay and post-hoc gate ablation. It does not replace the original prediction engine.

## Dataset and replay

- Delta BTCUSD 1-minute candles: 1,461,177 rows.
- Coverage: 2023-12-29 13:26 UTC to 2026-11-02 16:26 UTC.
- Missing data: 35,164 missing minute intervals across 16 gap events; longest missing run 15,311 minutes.
- Entry: exact 1-minute open matching the completed 5-minute signal availability time. Stale signals are skipped.
- A position crossing a missing minute or running into end-of-file is censored and excluded from performance metrics.
- Stop: completed 5-minute signal candle extreme plus/minus 0.1 × rolling 20-bar 5-minute ATR.
- Same-minute stop/target ambiguity: stop first.
- One active trade per independent module / R:R replay; maximum holding period 60 minutes.
- Costs: 5.9 bps taker fee and 1 bp slippage per side. Funding and market impact are not included.
- P&L is raw BTCUSD price points, not INR account returns or portfolio-level compounding.

## Original 4h/1h/15m gated results

| Module | R:R | Trades | Trades/day | Win rate | Gross PF | Net PF | Net expectancy | Net P&L points |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pullback | 3R | 514 | 0.52 | 25.7% | 0.91 | 0.38 | -0.923R | -63,533.6 |
| Sweep | 3R | 1,453 | 1.47 | 24.0% | 1.08 | 0.35 | -1.944R | -153,373.2 |
| Breakout/retest | 1R | 238 | 0.24 | 39.1% | 1.20 | 0.54 | -0.617R | -20,184.5 |
| Breakout/retest | 3R | 171 | 0.17 | 29.8% | 1.04 | 0.56 | -0.647R | -17,816.6 |
| Combined | 3R | 2,015 | 2.03 | 24.8% | 1.03 | 0.38 | -1.626R | -220,025.9 |

The best net profit factor was only 0.56. Average estimated round-trip cost was approximately 110–113 BTCUSD price points per trade. Some modules were marginally positive before costs, but none survived the configured taker fees and slippage.

## Post-hoc gate ablation

The diagnostic compared 4h/1h/15m context, 15m-only context and no directional context gate across pullback, sweep, breakout/retest and a collision-safe combined consensus module at 1R, 2R and 3R: 36 cases total.

- Positive net expectancy: 0 of 36.
- Net profit factor above 1: 0 of 36.
- Removing the 4h/1h gates increased trade frequency substantially but did not create an edge. Ungated variants often generated roughly 10–36 trades/day and remained deeply negative after costs.
- The original combined module did not become better by loosening the context filter.

This is a post-hoc diagnostic because the first gated results had already been inspected. It must not be used to select thresholds or approve a strategy.

## Decision

Keep the MTF structure and entry engine as a separate research layer. Do not merge it into production signal logic or promote it to actionable trade recommendations. Keep the original DMPE architecture:

1. Market intelligence.
2. Predictive engine.
3. Structure and entry engine.
4. Decision engine combining validated evidence, expected value and risk.

The next valid comparison is prediction-only versus structure-only versus combined predictive-plus-structure on the same untouched prospective period. Before that, freeze the design and add verified funding, measured spread/market-impact assumptions, and block-bootstrap uncertainty. FLAT remains a valid decision; there is no minimum daily trade requirement and no live order execution.

## Reproducibility

- MTF replay run: https://github.com/undaviyabrijesh-sys/DMPE/actions/runs/41567256039
- Gate-ablation run: https://github.com/undaviyabrijesh-sys/DMPE/actions/runs/41567250195
- Latest tests: https://github.com/undaviyabrijesh-sys/DMPE/actions/runs/41567285342

Artifacts are retained by GitHub Actions for 30 days.
