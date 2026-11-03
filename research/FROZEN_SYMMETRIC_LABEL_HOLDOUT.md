# DMPE frozen candidate: one-time reserved historical holdout

**Status:** Frozen exploratory candidate evaluation; not a prospective test.  
**Candidate selection source:** Train/validation-only symmetric-label screen, workflow run 41590214109.  
**No live/paper trading approval.**

## Frozen candidate

- Market: BTCUSD perpetual research series.
- Horizon: 3 hours.
- Barrier: symmetric 1.00x trailing 180-minute realized volatility.
- Classes: -1 lower barrier first, 0 neither barrier touched, +1 upper barrier first.
- Same-minute dual touches and incomplete/gapped windows are excluded.
- Model: HistGradientBoostingClassifier with balanced sample weights.
- Hyperparameters: learning_rate=0.05, max_iter=160, max_leaf_nodes=15, min_samples_leaf=60, l2_regularization=5.0, early_stopping=False, random_state=42.
- Preprocessing: median imputation fitted on the fit partition.
- Features: the frozen Stage 1 schema feature columns only; no future target/excursion fields.
- No confidence threshold, barrier, model family, feature or hyperparameter may be changed after opening this reserved block.

## Why this candidate was frozen

On the preceding validation partition (1,015 samples), the candidate recorded:
- Balanced accuracy: 0.4586 versus 0.3333 for the prior-only baseline.
- Macro-F1: 0.4572 versus 0.2076.
- Multiclass log loss: 1.019 versus 1.069.
- Multiclass Brier score: 0.611 versus 0.646.

The validation screen compared multiple horizons, barrier widths and model variants. These results are therefore subject to selection bias and are not proof of edge. The candidate is being frozen only to make the reserved-block evaluation a genuine one-time check.

## Reserved historical block

- Reuse the same pre-test signal-time cutoff: 2026-05-30 15:00 UTC.
- Reconstruct 3h samples aligned to UTC boundaries.
- Refit on all eligible samples before the final reserved block, excluding one full horizon around the train/validation and final holdout boundaries.
- Evaluate once on the previously reserved final 15%.
- Compare against a prior-only dummy baseline.
- Report balanced accuracy, macro-F1, accuracy, log loss, Brier score, confusion matrix and predicted/actual class counts.
- Compute paired UTC-day block-bootstrap 95% intervals for model-minus-baseline balanced-accuracy and macro-F1 differences (1,000 resamples).

This is a historical reserved block from data ending 2026-05-30. It is not the prospective Delta Exchange holdout. Do not tune or replace the candidate based on this result. A failed result means reject this candidate; do not retry on the same block.

## Next gate

If the candidate beats the prior baseline and the paired block-bootstrap intervals support a positive improvement, freeze a separate prospective protocol before the new window begins. The earliest clean start for that new protocol is 2026-11-04 00:00 UTC. The forward protocol must include fees, GST, spread, slippage, funding, entry timing, fill/no-fill assumptions and Delta Exchange venue-transfer validation.

**Trading approval:** No.
