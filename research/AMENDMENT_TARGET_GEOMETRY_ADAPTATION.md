# DMPE amendment: target geometry and regime-adaptive model research

**Amendment date:** 2026-11-03 UTC  
**Applies to:** Future predictive-model development and the prospective prediction-vs-structure comparison.  
**Status:** This amendment supersedes the fixed 1R-stop / 3R-target requirement for the *new* experiment only. The original preregistration remains preserved as the historical record of its original hypothesis.  
**Execution:** Research and paper observation only. No live orders.

## Decision

A 1R stop / 3R target is a reference configuration, not a compulsory universal setting. DMPE may investigate alternative barrier geometries and regime-conditioned choices. The purpose is to discover whether the market's volatility, trend and liquidity state supports different risk/reward geometry, while preventing hindsight selection.

## Research tracks

1. **Fixed-geometry benchmark:** retain 1R stop / 3R target as a reference so results remain comparable with the original hypothesis.
2. **Bounded geometry exploration:** evaluate only a small, predeclared grid:
   - stop: 0.75R, 1.0R, 1.25R, 1.5R;
   - target: 1.5R, 2.0R, 2.5R, 3.0R, 4.0R.
   - Include only combinations with positive target and stop distances. Report every tested combination, not only the winner.
3. **Regime-adaptive geometry:** test a small set of past-only regimes defined before evaluation (trend/range and low/medium/high trailing volatility). A regime may select a geometry only when that mapping is chosen on the pre-period training/validation data and then frozen. The regime classifier must use information available at the decision timestamp; no future labels, centered windows or retrospective regime assignment.
4. **Continuous adaptation:** do not begin with an unconstrained optimizer that changes stops/targets every bar. It may be explored later only as a separate experiment with hard bounds, turnover/complexity penalties, and a strictly walk-forward policy.

## Selection and leakage controls

- Separate model probability estimation from barrier geometry. Do not change labels, train separate classifiers for every geometry, and then choose the best result on the final holdout.
- Candidate geometries are evaluated using chronological train/validation partitions with a purge/embargo at least as long as the maximum outcome horizon (3 hours).
- Choose one geometry policy using pre-period validation only. Freeze its mapping, thresholds, code, feature schema, training cutoff, dependencies and artifact hashes before the prospective evaluation begins.
- The prospective holdout is used once for evaluation, never for selecting geometry, regimes, thresholds or model family.
- Report the complete candidate grid and multiple-testing correction. Any post-hoc geometry or regime discovery is exploratory and requires a new untouched prospective period.
- Compare against a simple fixed benchmark. A more complex adaptive policy must demonstrate incremental net benefit after fees, spread, slippage and verified funding, with uncertainty intervals and stability across regimes.
- Incomplete windows, missing candles and same-minute barrier ambiguity remain censored/excluded under the existing label policy. Never invent intraminute order.
- If no geometry clears validation robustness checks, the model must abstain / output FLAT rather than force a trade.

## Live-market adaptation boundary

"Adapt to live market" means using a frozen, point-in-time policy whose inputs update as new completed market data arrive. It does **not** mean manually changing parameters after seeing live outcomes or silently retuning on the evaluation period.

The future live/paper decision engine may update its regime state every completed decision bar, but:
- regime features must be past-only;
- the regime-to-geometry mapping must be frozen before the evaluation window;
- any scheduled retraining or recalibration must have a declared cadence, data cutoff, validation procedure, versioning and rollback rule;
- drift may trigger a FLAT / model-suspended state, not automatic unvalidated parameter changes.

## Required reporting

For each fixed and adaptive policy report coverage, trades/day, win rate, gross/net expectancy, profit factor, drawdown, turnover, regime-level results, sensitivity to costs, and day-block-bootstrap uncertainty. Report the benchmark and all tested variants. A higher win rate or more trades is not sufficient evidence of improvement.

## Prospective comparison impact

The prediction-only arm may use the single geometry policy selected and frozen on pre-period validation. Structure-only and combined arms must use the same frozen geometry policy for fair comparison. The original fixed 1R/3R result may also be reported as a benchmark, but it is not the only eligible geometry. Do not inspect prospective outcomes until model, geometry policy, evaluator and data manifest are frozen.
