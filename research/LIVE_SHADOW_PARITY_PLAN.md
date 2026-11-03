# Live shadow prediction: feature-parity gate

**Status:** design and data-contract audit; no predictions or orders are enabled.

## Frozen model contract

The shadow process must reproduce the frozen 3-hour BTCUSD candidate exactly:
- 3-hour horizon; symmetric barriers at 1.00x trailing 180-minute realized volatility.
- HistGradientBoostingClassifier with balanced sample weights and the frozen preprocessing/configuration.
- The 39 Stage 1 point-in-time features, in the exact saved schema order.
- No new order-flow, GARCH, HMM, Kalman, Bayesian or Monte Carlo columns may be appended to this model.
- No threshold, calibration mapping or model change may be selected using the historical reserved block.

The fitted estimator and imputer are not yet available as a verified deployable artifact in this branch. The shadow service must refuse to start prediction mode until an immutable model bundle, feature schema, training cutoff and SHA-256 manifest are present and verified.

## Data-source audit

The recorder writes raw Delta WebSocket messages as daily JSONL records with a local UTC `received_at` timestamp. Its current subscriptions include trades, L2, 1-minute traded-price candles, ticker, mark price, funding rate and exchange status.

The existing `build_feature_frame` is a batch feature builder over normalized OHLCV frames:
- traded-price 1-minute candles;
- mark-price candles;
- hourly funding-rate observations;
- hourly open-interest observations.

It timestamps completed candle features at candle close and delays hourly features by one hour plus one minute. This causal availability convention must remain unchanged.

The existing order-flow processor handles trades, L2 and explicit funding snapshots, but it is not a complete live adapter for the frozen Stage 1 feature schema. Its order-flow columns are not part of the frozen candidate.

## Current Delta compatibility issue

Delta's current India API documentation states that the legacy `funding_rate` WebSocket channel was fully deprecated on 31 October 2026. The recorder must not rely on that channel for a new prospective run.

The current public ticker sample exposes open interest in the `oi` field, and the REST historical-candle API documents `FUNDING:BTCUSD`, `OI:BTCUSD` and `MARK:...` symbol conventions. Before using these for live feature construction, validate:
1. response units and candle timestamp semantics;
2. whether the current hourly bar is complete;
3. publication delay and revision behaviour;
4. missing/stale data behaviour;
5. exact agreement with the historical Stage 1 input convention.

Do not substitute a ticker snapshot for an hourly close without an explicit, tested aggregation rule. Do not forward-fill missing minute prices or stale funding/OI observations.

## Implementation sequence

1. **Raw-event contract tests:** parse actual Delta event envelopes, event timestamps, symbol identity and candle OHLCV fields. Keep unknown event shapes visible in diagnostics rather than silently coercing them.
2. **Causal live bar store:** assemble one completed minute at a time; reject duplicates, late revisions and incomplete bars from prediction; preserve gap markers.
3. **Funding/OI adapter:** use only verified current sources and completed hourly observations; apply the same one-hour-plus-one-minute availability lag as Stage 1.
4. **Feature parity:** call the shared Stage 1 feature builder and compare the emitted columns, order, dtypes, null masks and numeric values against a deterministic replay fixture built from the same raw source data. Require exact schema equality and documented numerical tolerances.
5. **Model bundle:** export the frozen fitted imputer and classifier, ordered feature names, class order, training cutoff, package versions, source commit and artifact hashes. Refuse mismatched bundles.
6. **Shadow logger:** write timestamped class probabilities and data-quality metadata only after all gates pass. Never submit, stage or simulate an order.
7. **Operational checks:** record feed heartbeats, event counts, reconnects, minute coverage, stale-feed duration, prediction count and exceptions. Fail closed on missing data or schema drift.

## Shadow validation boundaries

- No exchange credentials are required.
- No private endpoints or order permissions are permitted.
- No orders, paper orders or execution simulation are started by this service.
- Predictions are research observations, not trade instructions.
- Keep a separate shadow calibration/development window and a later untouched prospective evaluation window. Select any calibration or confidence threshold only in the development window, then freeze it before the prospective test.
- Before any later P&L analysis, preregister fees, GST, spread, slippage, funding cashflows, fill/no-fill, entry timing, overlap rules and risk limits.

## Current blocker

The VPS has not been provisioned according to the existing deployment notes, and no representative raw capture sample has been audited in this task. Synthetic tests can validate parser logic, but live schema parity cannot be claimed until an actual recorder sample is checked against the documented Delta payloads.
