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

## Current Delta compatibility and capture findings

The recorder uses the new public India WebSocket endpoint, `wss://public-socket.india.delta.exchange`. Delta's documentation describes `funding_rate`, `mark_price` and `candlesticks` as migrated public channels on this endpoint. The July 2026 retirement notice applied to the old channels on the legacy private WebSocket endpoint; it does not mean the migrated public `funding_rate` channel should be omitted.

The REST historical-candle documentation explicitly supports:
- traded-price candles: `BTCUSD`;
- mark-price candles: `MARK:BTCUSD`;
- funding history: `FUNDING:BTCUSD`;
- open-interest history: `OI:BTCUSD`.

Official reference: https://docs.delta.exchange/ (India API, Historical OHLC Candles and WebSocket candlesticks/channel migration sections).

Two public five-minute captures have been audited:
- first capture: trades, L2, traded candles, mark point updates and ticker/OI snapshots;
- second capture: additionally verified `funding_rate` snapshots (`fr,fi,nfr,sy,ts`) and mark-price candle messages (`sy=MARK:BTCUSD`, `cst`, `ts`, OHLC).

The ticker `oi` field is a snapshot, not an hourly historical candle. The short capture's five funding snapshots are also not enough to reproduce the historical hourly funding series. Therefore the live feature adapter must use verified completed 1h REST observations for funding and OI, with the exact Stage 1 one-hour-plus-one-minute availability lag. Ticker OI and funding WebSocket snapshots can be stored for monitoring and later diagnostics, but must not silently replace the model inputs.

The second capture exposed a timing defect in the first live order-flow output: partial startup and in-progress final minutes were stamped as if complete. A five-second receive-time watermark is now implemented in the order-flow aggregator. A separate candle finalizer has also been added to:
- retain the latest update per 1m candle;
- exclude a partially captured startup minute;
- withhold any candle until its close is behind the receive-time watermark;
- preserve Delta candle-open timestamps, which the shared feature builder converts to feature-close timestamps.

The live candle dataset build is now part of the capture workflow. Its next successful run must be inspected before declaring candle finalization validated on live data.

## Implementation sequence

1. **Raw-event contract:** actual Delta payload shapes and timestamp units are now observed for trades, L2, traded candles, mark candles, mark updates, ticker, funding and system status. Keep diagnostics for unknown shapes.
2. **Causal live bars:** implemented receive-time watermarking for event minutes and a separate finalizer for traded/mark candles. Synthetic tests cover startup/current-minute exclusion, duplicate update selection and mark candles without volume. Confirm this behavior against the next live capture artifact.
3. **Funding/OI adapter:** add a public REST adapter for completed 1h `FUNDING:BTCUSD` and `OI:BTCUSD` candles. Filter out any currently forming hourly candle and preserve original candle-open timestamps.
4. **Historical/live comparison:** over a shared closed timestamp window, compare WebSocket traded and mark candles to REST history. Report missing bars, duplicates, OHLC differences, candle revisions and end-to-end receive latency.
5. **Feature parity:** assemble sufficient warm-up history (at least 5h of 1m traded/mark candles and 48h of 1h funding/OI history), call the shared Stage 1 feature builder, and compare the exact saved 39-feature schema, order, dtypes, null masks and numeric values. Require exact schema equality and documented numerical tolerances.
6. **Model bundle:** export the frozen fitted imputer and classifier, ordered feature names, class order, training cutoff, package versions, source commit and artifact hashes. Refuse mismatched bundles.
7. **Shadow logger:** write timestamped class probabilities and data-quality metadata only after all gates pass. Never submit, stage or simulate an order.
8. **Operational checks:** record feed heartbeats, event counts, reconnects, minute coverage, stale-feed duration, prediction count and exceptions. Fail closed on missing data or schema drift.

## Shadow validation boundaries

- No exchange credentials are required.
- No private endpoints or order permissions are permitted.
- No orders, paper orders or execution simulation are started by this service.
- Predictions are research observations, not trade instructions.
- Keep a separate shadow calibration/development window and a later untouched prospective evaluation window. Select any calibration or confidence threshold only in the development window, then freeze it before the prospective test.
- Before any later P&L analysis, preregister fees, GST, spread, slippage, funding cashflows, fill/no-fill, entry timing, overlap rules and risk limits.

## Current status and blockers

- Public recorder capture works through GitHub Actions without exchange credentials.
- The latest repository test suite passed at commit `e2e042ebb623dc66d7f2c50b1f2cdb3ccd647b2c` before the final test-fixture correction; re-run status for the newest commit must be green before merging.
- The capture workflow now builds order-flow features and causally finalized traded/mark candles; its latest full run is still being validated.
- Feature parity is not established. The 39-feature training schema and the fitted model bundle still need to be loaded from the historical workflow artifacts and pinned by checksum.
- The VPS is not provisioned. This is not a blocker for REST-vs-WebSocket parity checks in GitHub Actions, but it is required for durable continuous shadow collection.
- No shadow predictions, paper orders or live orders are enabled.
