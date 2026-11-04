import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dmpe.live_parity import compare_candle_frames, trailing_contiguous_rows


def test_live_rest_candle_parity_passes_within_tolerance():
    times = pd.date_range("2024-01-01T00:00:00Z", periods=3, freq="1min")
    live = pd.DataFrame({
        "time": times,
        "open": [100.0, 101.0, 102.0],
        "high": [101.0, 102.0, 103.0],
        "low": [99.0, 100.0, 101.0],
        "close": [100.5, 101.5, 102.5],
    })
    rest = live.copy()
    rest.loc[1, "close"] += 1e-10
    report = compare_candle_frames(live, rest)
    assert report["matched_rows"] == 3
    assert report["all_live_bars_match_rest"] is True


def test_live_rest_parity_reports_missing_and_materially_different_candles():
    times = pd.date_range("2024-01-01T00:00:00Z", periods=2, freq="1min")
    live = pd.DataFrame({
        "time": times,
        "open": [100.0, 101.0],
        "high": [101.0, 102.0],
        "low": [99.0, 100.0],
        "close": [100.5, 101.5],
    })
    rest = live.iloc[:1].copy()
    rest.loc[rest.index[0], "close"] = 100.0
    report = compare_candle_frames(live, rest)
    assert report["missing_in_rest"] == 1
    assert report["mismatched_values"]["close"]["mismatches"] == 1
    assert report["all_live_bars_match_rest"] is False


def test_trailing_contiguous_rows_stops_at_gap():
    times = pd.to_datetime([
        "2024-01-01T00:00:00Z",
        "2024-01-01T00:01:00Z",
        "2024-01-01T00:02:00Z",
        "2024-01-01T00:04:00Z",
        "2024-01-01T00:05:00Z",
    ])
    frame = pd.DataFrame({"time": times})
    assert trailing_contiguous_rows(frame, "1m") == 2
    assert trailing_contiguous_rows(frame, "1h") == 1
