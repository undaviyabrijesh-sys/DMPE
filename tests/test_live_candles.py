import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dmpe.live_candles import build_closed_candle_frame


def _record(received_at, candle_open, close, *, symbol="BTCUSD", volume=10, high=None, low=None):
    received_at = pd.Timestamp(received_at)
    candle_open = pd.Timestamp(candle_open)
    return {
        "received_at": received_at.isoformat(),
        "event": {
            "type": "candlestick_1m",
            "res": "1m",
            "sy": symbol,
            "cst": candle_open.value // 1000,
            "ts": received_at.value // 1000,
            "o": close - 1,
            "h": close if high is None else high,
            "l": close - 2 if low is None else low,
            "c": close,
            **({"v": volume} if volume is not None else {}),
        },
    }


def test_closed_candle_builder_uses_latest_update_and_watermark():
    records = [
        _record("2024-01-01T00:00:15Z", "2024-01-01T00:00:00Z", 100),
        _record("2024-01-01T00:01:30Z", "2024-01-01T00:01:00Z", 101),
        _record("2024-01-01T00:01:59Z", "2024-01-01T00:01:00Z", 102),
        _record("2024-01-01T00:02:59Z", "2024-01-01T00:02:00Z", 103),
        _record("2024-01-01T00:03:59Z", "2024-01-01T00:03:00Z", 104),
        _record("2024-01-01T00:04:10Z", "2024-01-01T00:04:00Z", 105),
    ]
    frame, report = build_closed_candle_frame(records, symbol="BTCUSD")

    assert frame["time"].tolist() == [
        pd.Timestamp("2024-01-01T00:01:00Z"),
        pd.Timestamp("2024-01-01T00:02:00Z"),
        pd.Timestamp("2024-01-01T00:03:00Z"),
    ]
    assert frame.loc[0, "close"] == 102
    assert report["updates_seen"] == 6
    assert report["closed_candles"] == 3
    assert report["duplicate_updates_collapsed"] == 1


def test_mark_candle_without_volume_is_accepted_as_zero_volume():
    records = [
        _record(
            "2024-01-01T00:00:15Z",
            "2024-01-01T00:00:00Z",
            100,
            symbol="MARK:BTCUSD",
            volume=None,
        ),
        _record(
            "2024-01-01T00:01:10Z",
            "2024-01-01T00:01:00Z",
            101,
            symbol="MARK:BTCUSD",
            volume=None,
        ),
        {"received_at": "2024-01-01T00:02:10Z", "event": {"type": "ticker"}},
    ]
    frame, report = build_closed_candle_frame(records, symbol="MARK:BTCUSD")
    assert len(frame) == 1
    assert frame.loc[0, "volume"] == 0.0
    assert report["closed_candles"] == 1


def test_candle_builder_rejects_invalid_ohlc_and_missing_receive_times():
    bad = _record(
        "2024-01-01T00:00:00Z",
        "2024-01-01T00:00:00Z",
        100,
        high=99,
    )
    frame, report = build_closed_candle_frame([bad], symbol="BTCUSD")
    assert frame.empty
    assert report["invalid_updates"] == 1

    frame, report = build_closed_candle_frame([
        {"event": {"type": "candlestick_1m", "sy": "BTCUSD"}}
    ])
    assert frame.empty
    assert report["reason"] == "missing_recorder_receive_timestamps"
