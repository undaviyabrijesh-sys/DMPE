# Live Capture Audit — 2026-11-03

## Run

- Repository: `undaviyabrijesh-sys/DMPE`
- Branch: `research/live-shadow-parity`
- Workflow run: [41591739802](https://github.com/undaviyabrijesh-sys/DMPE/actions/runs/41591739802)
- Raw artifact: `dmpe-live-capture-41591739802`
- Derived artifact: `dmpe-live-orderflow-41591739802`
- Capture window: 2026-11-03 06:44:19.683 UTC to 06:49:18.580 UTC
- Raw records: 2,823 across approximately five minutes
- Workflow and repository tests: passed

## Observed public event contract

| Event | Count | Observed payload fields / notes |
|---|---:|---|
| `trades` | 1,527 | `p,r,s,sy,t,ts,type`; trade time and publish time are microsecond epoch values; buyer role `m` / `t` observed |
| `ob_l2` | 597 | `a,b,lts,sy,ts,type`; top 15 ask/bid levels; event timestamps in microseconds |
| `candlestick_1m` | 489 | `c,cst,h,l,o,res,sy,ts,type,v`; `cst` identifies candle start; OHLCV values present for traded-price candles |
| `mark_price` | 148 | `p,sy,ts,type`; mark price point updates |
| `ticker` | 60 | `d,sp,sy,ts,type`; nested `d[0].oi` array observed |
| `system_status` | 1 | Initial snapshot reported `status=live` |
| `subscriptions` | 1 | Subscription acknowledgement |

## Important ticker OI interpretation

The current Delta public ticker documentation describes the nested `oi` array as:
- element 0: open interest in contracts;
- element 1: six-hour change in open-interest value in the settling currency.

The capture sample contained `oi=["1147484","5158925.2800"]`. This is a ticker snapshot, not an hourly OI candle. It must not be substituted into the historical hourly OI input or used to synthesize an hourly close.

## Derived order-flow output

- Six minute-close rows were emitted (06:45 through 06:50 UTC).
- Trade and L2 feed-observed flags were true for all six rows.
- Trade counts ranged from 52 to 406 per minute.
- L2 snapshots ranged from 37 to 120 per minute.
- No missing-minute forward fill was applied.
- Funding observations: zero, because this capture was run while the recorder did not subscribe to funding.

The six-row output validates that the current trade/L2 parser can produce a small live feature file. It is not a statistical sample and is not evidence of predictive value.

## Corrections and next capture

The official Delta documentation currently describes `funding_rate` as a channel on the new public WebSocket endpoint. The 31 July 2026 retirement applies to the old channels on the legacy private endpoint, not the migrated public channel. The previous recorder change that removed funding entirely was therefore too broad.

The recorder has now been corrected to:
1. Subscribe to `funding_rate` on `wss://public-socket.india.delta.exchange`.
2. Subscribe to `candlestick_1m` for both `BTCUSD` and `MARK:BTCUSD`, so traded and mark OHLC candles can be compared.
3. Continue recording ticker/OI snapshots separately from historical OI candles.

Official reference: [Delta Exchange WebSocket public channels and migration notes](https://docs.delta.exchange/).

## Parity status

Not yet established. The next capture must confirm:
- funding event fields `fr,fi,nfr,sy,ts` and causal availability;
- mark-candle message shape and timestamp semantics;
- ticker OI values and update cadence, kept separate from hourly OI candles;
- candle start/close semantics and duplicate/revision handling;
- exact 39-feature schema, null masks, and numeric parity with the frozen historical feature builder.

No live prediction or order routing is enabled by this audit.


## Second capture: migrated feeds verified

- Workflow run: [41601974295](https://github.com/undaviyabrijesh-sys/DMPE/actions/runs/41601974295)
- Raw artifact: `dmpe-live-capture-41601974295`
- Derived artifact: `dmpe-live-orderflow-41601974295`
- Capture window: 2026-11-03 08:16:15.780 UTC to 08:21:14.627 UTC
- Raw records: 1,938
- Workflow and repository tests: passed

| Event | Count | Notes |
|---|---:|---|
| `trades` | 618 | Same trade payload; microsecond event time |
| `ob_l2` | 598 | Top-15 bid/ask snapshots |
| `candlestick_1m` | 508 | Six traded-price minute buckets and six mark-price minute buckets; repeated in-progress updates |
| `mark_price` | 148 | Mark point updates |
| `ticker` | 59 | Nested ticker OI snapshots |
| `funding_rate` | 5 | `fr,fi,nfr,sy,ts` verified |
| `system_status` | 1 | `status=live` |
| `subscriptions` | 1 | Server acknowledged all requested subscriptions |

The observed funding sample had `fi=28800`, an explicit `nfr` timestamp, and a numeric `fr`. The observed mark-candle payload used `sy=MARK:BTCUSD`, `cst` (microsecond candle-start epoch), `ts` (microsecond update timestamp), and OHLC fields. Traded-price candles used the same shape with `sy=BTCUSD`.

### Critical timing issue found in the derived output

The pre-fix aggregation reported six minute rows from 08:17 to 08:22 even though the capture began at 08:16:15 and ended at 08:21:14. That means it included:
- the partial startup minute (08:16), before a full minute of capture coverage existed;
- the still-open final minute (08:21), incorrectly timestamped as available at 08:22.

Those rows must not be passed into a live model. They create incomplete inputs and can make a feature timestamp appear later than the latest observation actually received.

A receive-time completeness watermark has now been added to `aggregate_event_records`:
- exclude the minute in progress when capture starts;
- withhold a minute until its close is at least five seconds behind the latest recorder receive time;
- keep internal missing minutes as explicit gaps / NaNs rather than forward-filling them.

Tests were added for partial startup/current-minute exclusion and invalid watermark settings. The next workflow run must verify that the corrected output contains only complete minutes.

### Current interpretation

The live capture confirms that the migrated funding and mark-candle channels are available and parsable. It does not establish hourly OI parity: ticker OI remains a point-in-time snapshot and must be stored separately from hourly OI candle observations. Feature parity with the frozen 39-column classifier remains unproven.

No shadow predictions or order routing are enabled.
