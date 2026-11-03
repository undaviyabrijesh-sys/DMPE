# DMPE: event-label and regime-ablation findings

**As of:** 2026-11-03 UTC  
**Status:** Exploratory findings; no trading approval.

## 1. Symmetric first-touch label distribution

The reproducible diagnostic completed successfully on the frozen Stage 1 feature artifact (1,496,341 one-minute rows; 2023-12-29 13:27 UTC through 2026-11-02 16:27 UTC). It compared symmetric barriers at 0.50x, 0.75x, 1.00x, 1.50x and 2.00x trailing horizon-aligned realized volatility.

The following table reports the non-overlapping UTC-aligned samples from the diagnostic. The final historical test partition is shown only as a descriptive label-stability check; it is not used for model fitting, barrier selection or performance evaluation in the follow-on predictability screen.

| Horizon | Barrier | Train directional / no-touch | Validation directional / no-touch | Historical test directional / no-touch |
|---|---:|---:|---:|---:|
| 1h | 0.75x RV | 75.0% / 25.0% | 74.6% / 25.4% | 73.8% / 26.2% |
| 1h | 1.00x RV | 54.6% / 45.4% | 54.5% / 45.5% | 54.0% / 46.0% |
| 3h | 0.75x RV | 71.6% / 28.4% | 71.6% / 28.4% | 71.0% / 29.0% |
| 3h | 1.00x RV | 53.4% / 46.6% | 53.3% / 46.7% | 51.5% / 48.5% |

Up-first and down-first counts are close in each partition. Same-minute dual touches are rare and remain ambiguous. Incomplete/gapped windows remain invalid.

Median time to first touch in the 0.75x variant was approximately 21 minutes for 1h and 64–68 minutes for 3h in train/validation. At 1.00x it was approximately 27 minutes for 1h and 80–87 minutes for 3h.

**Interpretation:** 0.75x creates a higher directional-event frequency and near-balanced three-class proportions. 1.00x creates a more conservative event frequency with a larger no-touch class. Neither has predictive edge established. The barrier is not selected automatically from this diagnostic.

## 2. Corrected GARCH/HMM directional ablation

The corrected GARCH QMLE timing implementation was rerun with the existing asymmetric 1R-stop / 3R-target directional labels.

- 1h baseline, baseline+GARCH and baseline+GARCH+HMM all had 0.333 balanced accuracy; models defaulted to the neutral class.
- 3h baseline HistGradientBoosting historical test balanced accuracy was 0.3511; baseline+GARCH was 0.3426 and baseline+GARCH+HMM was 0.3426.
- Validation confidence gates did not produce the minimum 30 directional signals, so no strategy was activated.
- This is a negative result for the current directional target/model setup. It does not invalidate GARCH for volatility forecasting or risk sizing.

## 3. Follow-on predictability screen

PR #3 adds a train/validation-only screen of symmetric labels. It compares prior-only, logistic regression and HistGradientBoosting, each unweighted and class-balanced, across the predeclared grid.

- 1h signal timestamps stop at 2026-05-30 18:00 UTC.
- 3h signal timestamps stop at 2026-05-30 15:00 UTC.
- One full horizon is purged around the train/validation boundary.
- The final 15% is reserved and unscored.
- The previously inspected historical test period is not used for fitting, barrier selection or model evaluation.
- No confidence threshold, P&L, execution-cost or trading decision is produced.

The experiment is only a screen for whether existing features contain directional information. Any positive validation result requires a separate frozen forward protocol before a prospective test.
