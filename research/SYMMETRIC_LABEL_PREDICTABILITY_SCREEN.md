# DMPE exploratory protocol: symmetric first-touch predictability screen

**Status:** Exploratory screening only; no confirmatory strategy or deployment status.  
**Market:** BTCUSD perpetual research series.  
**Feature artifact:** Stage 1 run 41509621537.  
**Signal-time cutoffs:** 1h through 2026-05-30 18:00 UTC; 3h through 2026-05-30 15:00 UTC.  
**Execution:** No orders, no P&L replay, no paper/live promotion.

## Motivation

The existing frozen asymmetric 1R-stop / 3R-target directional labels are extremely neutral-heavy. A corrected GARCH/HMM ablation using those labels showed all models predicting the neutral class; balanced accuracy remained at chance and validation produced no qualifying directional signals. That ablation is a negative directional result for the current label/model combination.

The separate symmetric first-touch diagnostic found that symmetric volatility-scaled barriers create more balanced three-class distributions. This protocol asks whether the existing point-in-time features contain any out-of-sample information about those symmetric first-touch outcomes.

## Frozen exploratory grid

- Horizons: 1h and 3h only.
- Barrier multiples: 0.50x, 0.75x, 1.00x, 1.50x and 2.00x of completed horizon-aligned trailing realized volatility.
- Classes: -1 lower barrier first, 0 neither barrier touched within horizon, +1 upper barrier first.
- Same-minute dual touches and incomplete/gapped future windows are excluded, not assigned a direction.
- One sample per horizon, aligned to UTC horizon boundaries; this prevents overlapping forward outcome windows.

The grid is descriptive and exploratory. It is not a search for a profitable stop/target setting. No multiplier is automatically selected.

## Data and split policy

- Reuse the frozen Stage 1 BTCUSD 1-minute feature artifact and its declared feature columns.
- Features must be available at the completed signal-candle close; future target/excursion columns are never inputs.
- Labels use only future OHLC bars after the signal close.
- Use horizon-specific signal cutoffs: 1h no later than 2026-05-30 18:00 UTC; 3h no later than 2026-05-30 15:00 UTC. Retain only the additional future bars required to resolve labels at or before those cutoffs. The last 15% of each pre-cutoff sample is reserved and unscored.
- Chronological train/validation partition: first 70% train, next 15% validation, final 15% reserved. Purge one full horizon on both sides of the train/validation boundary.
- No historical test metrics are generated. In particular, data after the cutoff through 2026-11-02 are not used for model fitting, barrier selection or performance evaluation in this screen.
- The previously declared prospective window is not accessed. Because this is a new exploratory hypothesis, any later forward confirmation must be frozen separately before its start.

## Models and metrics

For every horizon/multiplier combination, compare:
1. Prior-only dummy classifier.
2. Logistic regression, unweighted and balanced class weights.
3. Histogram gradient boosting, unweighted and balanced sample weights.

Report validation class counts, predicted class counts, accuracy, balanced accuracy, macro-F1, multiclass log loss, Brier score and confusion matrix. Do not select a directional confidence threshold in this screen. Balanced class weighting changes effective class priors; its probabilities are not presumed calibrated.

No p-values, P&L, net expectancy, fees, slippage, funding, fills, Sharpe or drawdown claims are produced. The grid contains multiple exploratory comparisons; a promising result is a hypothesis for a separate preregistered test, not evidence of a robust edge.

## Decision gates

- If models do not beat the prior-only benchmark on validation in balanced accuracy and macro-F1, do not add more model complexity. Revisit feature information and label economics.
- If validation results are consistently better than prior-only across a predeclared subset, freeze one hypothesis, features, model, parameters and cost assumptions in a new protocol.
- A new prospective forward window must begin only after that freeze. Historical periods already inspected in prior DMPE experiments remain exploratory and cannot be repurposed as untouched confirmation.
- Even a successful directional screen must pass net execution-cost and prospective Delta Exchange validation gates before paper trading is considered.

**Trading approval:** No.
