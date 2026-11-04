import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dmpe.orderflow import (aggregate_event_records, extract_funding_observations, parse_book_event, parse_funding_event, parse_trade_event)


def test_delta_buyer_role_infers_aggressor_side_without_guessing_unknowns():
    buy = parse_trade_event({"type": "trades", "p": "100", "s": "2", "r": "t"})
    sell = parse_trade_event({"type": "trades", "p": "101", "s": "3", "r": "m"})
    unknown = parse_trade_event({"type": "trades", "p": "100", "s": "1", "r": "x"})
    assert buy["aggressor_side"] == "buy"
    assert buy["signed_size"] == 2
    assert sell["aggressor_side"] == "sell"
    assert sell["signed_size"] == -3
    assert unknown["aggressor_side"] == "unknown"
    assert np.isnan(unknown["signed_size"])


def test_l2_snapshot_calculates_spread_depth_and_microprice_pressure():
    event = {
        "type": "ob_l2",
        "b": [["99.0", "10"], ["98.0", "20"], ["97.0", "30"]],
        "a": [["101.0", "5"], ["102.0", "10"], ["103.0", "15"]],
    }
    parsed = parse_book_event(event)
    assert parsed is not None
    assert parsed["mid_price"] == 100.0
    assert parsed["spread_bps"] == 200.0
    assert parsed["depth_imbalance_1"] == 1 / 3
    assert parsed["depth_imbalance_5"] == 1 / 3
    assert parsed["microprice_mid_bps"] > 0


def test_aggregation_timestamps_completed_minute_and_preserves_missingness():
    records = [
        {"received_at": "2024-01-01T00:00:00.000Z", "event": {
            "type": "trades", "p": "100", "s": 2, "r": "t", "t": 1704067200010000
        }},
        {"received_at": "2024-01-01T00:00:00.020Z", "event": {
            "type": "trades", "p": "101", "s": 3, "r": "m", "t": 1704067200020000
        }},
        {"received_at": "2024-01-01T00:00:00.030Z", "event": {
            "type": "ob_l2",
            "b": [["99", "10"]], "a": [["101", "5"]],
            "ts": 1704067200030000,
        }},
        {"received_at": "2024-01-01T00:03:00.010Z", "event": {
            "type": "trades", "p": "102", "s": 4, "r": "t", "t": 1704067380010000
        }},
    ]
    frame = aggregate_event_records(records)
    assert len(frame) == 3
    assert frame.loc[0, "time"] == pd.Timestamp("2024-01-01T00:01:00Z")
    assert frame.loc[0, "trade_buy_taker_volume"] == 2
    assert frame.loc[0, "trade_sell_taker_volume"] == 3
    assert frame.loc[0, "trade_volume_imbalance"] == -0.2
    assert frame.loc[0, "trade_feed_observed"] == 1
    assert frame.loc[1, "trade_feed_observed"] == 0
    assert pd.isna(frame.loc[1, "trade_total_volume"])
    assert frame.loc[2, "trade_feed_observed"] == 1
    assert frame.loc[2, "l2_feed_observed"] == 0


def test_aggregation_excludes_partial_startup_and_current_minutes():
    start = pd.Timestamp("2024-01-01T00:00:15Z")
    middle = pd.Timestamp("2024-01-01T00:01:10Z")
    current = pd.Timestamp("2024-01-01T00:02:15Z")

    def trade_record(received_at, event_time, price):
        return {
            "received_at": received_at.isoformat(),
            "event": {
                "type": "trades",
                "p": str(price),
                "s": 1,
                "r": "t",
                "t": event_time.value // 1000,
            },
        }

    records = [
        trade_record(start, start, 99),
        trade_record(middle, middle, 100),
        trade_record(current, current, 101),
    ]
    frame = aggregate_event_records(records)
    # The first minute began before capture; the last minute is still open.
    assert len(frame) == 1
    assert frame.loc[0, "time"] == pd.Timestamp("2024-01-01T00:02:00Z")
    assert frame.loc[0, "trade_count"] == 1
    assert frame.loc[0, "trade_last_price"] == 100


def test_aggregation_rejects_invalid_close_grace():
    try:
        aggregate_event_records([], close_grace_seconds=60)
    except ValueError as exc:
        assert "between 0 and 59" in str(exc)
    else:
        raise AssertionError("Expected invalid close grace to be rejected")


def test_funding_snapshot_preserves_rate_and_exact_next_realization_time():
    record = {
        "received_at": "2024-01-01T00:00:00Z",
        "event": {
            "type": "funding_rate",
            "sy": "BTCUSD",
            "fr": "0.0001",
            "fi": 28800,
            "ts": 1704067200000000,
            "nfr": 1704096000000000,
        },
    }
    parsed = parse_funding_event(record, record["event"])
    assert parsed is not None
    assert parsed["funding_rate"] == 0.0001
    assert parsed["funding_interval_seconds"] == 28800
    assert parsed["next_funding_realization"] == pd.Timestamp("2024-01-01T08:00:00Z")
    frame = extract_funding_observations([record])
    assert len(frame) == 1
    assert frame.loc[0, "symbol"] == "BTCUSD"


def test_funding_snapshot_rejects_missing_or_past_realization_timestamp():
    record = {"event": {
        "type": "funding_rate", "fr": "0.0001", "fi": 28800,
        "ts": 1704067200000000, "nfr": 1704067100000000,
    }}
    assert parse_funding_event(record, record["event"]) is None


def test_orderflow_features_join_exact_minute_and_keep_missingness():
    from dmpe.orderflow import attach_orderflow_features

    times = pd.date_range("2024-01-01T00:00:00Z", periods=8, freq="1min")
    base = pd.DataFrame({"time": times, "close": np.arange(100, 108)})
    flow = pd.DataFrame({
        "time": times[2:],
        "trade_feed_observed": 1,
        "l2_feed_observed": 1,
        "trade_signed_volume": np.arange(6, dtype=float),
        "trade_volume_imbalance": np.full(6, 0.25),
        "l2_depth_imbalance_5_mean": np.full(6, 0.10),
        "l2_microprice_mid_bps_mean": np.full(6, 0.5),
        "l2_spread_bps_mean": np.full(6, 1.0),
    })
    merged, added = attach_orderflow_features(base, flow)
    assert "of_trade_signed_volume" in added
    assert pd.isna(merged.loc[0, "of_trade_signed_volume"])
    assert merged.loc[2, "of_trade_signed_volume"] == 0.0
    assert pd.isna(merged.loc[5, "of_trade_signed_volume_mean_5m"])
    assert merged.loc[6, "of_trade_signed_volume_mean_5m"] == 2.0
    assert merged.loc[6, "of_l2_feed_coverage_5m"] == 1.0


def test_orderflow_join_rejects_duplicate_timestamps():
    from dmpe.orderflow import attach_orderflow_features

    times = pd.date_range("2024-01-01T00:00:00Z", periods=2, freq="1min")
    base = pd.DataFrame({"time": times})
    flow = pd.DataFrame({"time": [times[0], times[0]], "trade_signed_volume": [1, 2]})
    try:
        attach_orderflow_features(base, flow)
    except ValueError as exc:
        assert "unique minute-close" in str(exc)
    else:
        raise AssertionError("Expected duplicate minute timestamps to be rejected")
