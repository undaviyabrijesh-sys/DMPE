# Quantitative regime feature ablation — exploratory results

**Run:** [DMPE quantitative regime ablation #4](https://github.com/undaviyabrijesh-sys/DMPE/actions/runs/41577802028)  
**Code commit:** `1ba5bd7ae3516800e3d405d4a1f157d24ded428b`  
**Stage 1 source:** feature artifact from run `41509621537`  
**Status:** Completed successfully; exploratory screening only.

## Question

Do GARCH volatility features and HMM volatility-state probabilities add directional classification value to the existing DMPE feature set?

## Method

- Reused the existing chronological train/validation/test evaluation, purge/embargo and validation-only confidence-threshold selection.
- Used the existing separate 1h and 3h long/short 1R-stop / 3R-target first-touch labels.
- Derived a direction class: +1 when the long target wins and the short target does not also win; -1 for the reverse; 0 when neither target wins. Conflicting target wins, same-minute ambiguities and incomplete outcomes were excluded.
- Built complete 5-minute bars only; incomplete buckets and gaps were not filled.
- Fitted GARCH(1,1) and a three-state Gaussian HMM on a contiguous training segment only. HMM state probabilities were forward-filtered, never backward-smoothed, and filters reset after data gaps.
- Compared baseline features, baseline + GARCH, and baseline + GARCH + HMM for Logistic Regression and Histogram Gradient Boosting.
- Fixed reference geometry was 1R stop / 3R target. This was not a search over stop/target geometry.

## Test results

Balanced accuracy and multiclass Brier score (lower Brier is better):

| Horizon | Feature set | Logistic BA | Logistic Brier | HistGB BA | HistGB Brier |
|---|---|---:|---:|---:|---:|
| 1h | Baseline | 0.3333 | 0.088060 | 0.3333 | 0.087901 |
| 1h | + GARCH | 0.3333 | 0.088013 | 0.3332 | 0.088264 |
| 1h | + GARCH + HMM | 0.3333 | 0.087936 | 0.3333 | 0.088449 |
| 3h | Baseline | 0.3333 | 0.117117 | 0.3511 | 0.114997 |
| 3h | + GARCH | 0.3333 | 0.117260 | 0.3426 | 0.114727 |
| 3h | + GARCH + HMM | 0.3333 | 0.117143 | 0.3426 | 0.116072 |

## Interpretation

1. **No consistent directional improvement.** Balanced accuracy is unchanged or lower when GARCH/HMM features are added.
2. Small Brier-score movements are mixed and not enough to establish a meaningful calibration improvement.
3. The 1h directional classifiers remain at approximately chance balanced accuracy. The 3h HistGB baseline is only modestly above chance and degrades when GARCH/HMM features are added.
4. This experiment did **not** simulate net P&L. It is a feature-screening test, not evidence of profitability.
5. GARCH's primary role may be volatility forecasting and risk geometry rather than direction. This ablation does not test that role. HMM may be more useful for regime-conditioned risk/target policies than as a generic directional feature.

## Decision

- Do not add GARCH/HMM as directional alpha features to the current prediction model.
- Do not promote any of these variants to paper or live trading.
- Keep both modules as research components pending separate tests:
  - GARCH: compare 1h/3h realized-volatility forecasts against trailing-RV baselines using QLIKE and robust loss comparisons.
  - HMM: test out-of-sample regime persistence and volatility separation, then test a small preregistered regime-to-geometry mapping on validation only.
- Before any confirmatory use, cross-fit training-period regime features and run a fee, slippage and verified-funding-aware trade replay with block-bootstrap uncertainty.
- Kalman, Bayesian updating and Monte Carlo have not yet been included in this ablation.

A successful workflow run means the code and experiment completed; it does not mean the models are profitable.
