# DMPE: frozen symmetric-label historical holdout results

**Workflow:** [Run 41590623745](https://github.com/undaviyabrijesh-sys/DMPE/actions/runs/41590623745)  
**Protocol:** [research/FROZEN_SYMMETRIC_LABEL_HOLDOUT.md](FROZEN_SYMMETRIC_LABEL_HOLDOUT.md)  
**Status:** Historical reserved block passed; no prospective or trading approval.

## Frozen candidate

- BTCUSD, 3-hour horizon.
- Symmetric barriers at 1.00x trailing 180-minute realized volatility.
- Classes: lower barrier first (-1), no touch (0), upper barrier first (+1).
- HistGradientBoostingClassifier with balanced sample weights.
- Hyperparameters fixed in the protocol and code; median imputation fitted on the fit partition.
- 39 point-in-time Stage 1 feature columns; no future target columns.
- Fit partition: 5,761 non-overlapping observations, 2024-01-17 12:00 UTC through 2026-01-23 06:00 UTC.
- Reserved holdout: 1,017 observations, 2026-01-23 15:00 UTC through 2026-05-30 15:00 UTC.
- One 3-hour horizon embargo at the final boundary.

## Holdout metrics

| Metric | Prior-only baseline | Frozen candidate |
|---|---:|---:|
| Accuracy | 46.41% | 46.80% |
| Balanced accuracy | 33.33% | **41.90%** |
| Macro-F1 | 21.13% | **41.36%** |
| Multiclass log loss | 1.0623 | **1.0241** |
| Multiclass Brier | 0.6412 | **0.6172** |

The baseline predicts the majority no-touch class for every observation. The candidate predicted 342 down-first, 506 no-touch and 169 up-first observations; actual counts were 293, 472 and 252.

## Paired day-block bootstrap

1,000 paired resamples of 128 UTC-day blocks:
- Balanced-accuracy difference (candidate minus baseline), 95% CI: **+5.85 to +11.55 percentage points**.
- Macro-F1 difference, 95% CI: **+17.08 to +23.44 percentage points**.

Both intervals exclude zero. This is a useful historical predictive result and supports continuing research on this frozen candidate.

## What this does not establish

- It is a reserved historical classification test, not a prospective out-of-sample period.
- It does not estimate trade expectancy, win rate at an executable entry, profit factor, drawdown, turnover or net P&L.
- It does not include Delta Exchange taker/maker fees, GST, spread, slippage, funding, fill/no-fill, latency or venue-transfer effects.
- It does not select a probability threshold or authorize paper/live trading.
- It does not replace or modify the separate frozen asymmetric 1R-stop / 3R-target confirmatory experiment.

## Next gate

Freeze a separate forward protocol before 2026-11-04 00:00 UTC. The prospective test must lock the model artifact, training cutoff, probability calibration and threshold selection using validation only, execution cost/funding conventions, entry timing, one-position/overlap rules, risk limits, and evaluator hash. Only a positive net result with uncertainty and robust execution assumptions can justify paper observation.
