import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dmpe.live_parity import (
    assess_feature_warmup,
    compare_candle_frames,
    trailing_contiguous_rows,
)


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



def test_feature_warmup_requires_contiguous_and_fresh_sources():
    profiles = {
        "traded_price_1m": {"trailing_contiguous_rows": 241, "last_complete_bar_age_seconds": 90},
        "mark_price_1m": {"trailing_contiguous_rows": 250, "last_complete_bar_age_seconds": 100},
        "funding_1h": {"trailing_contiguous_rows": 25, "last_complete_bar_age_seconds": 3600},
        "open_interest_1h": {"trailing_contiguous_rows": 30, "last_complete_bar_age_seconds": 3700},
    }
    result = assess_feature_warmup(profiles)
    assert result["ready"] is True


def test_feature_warmup_fails_closed_on_stale_or_short_hourly_history():
    profiles = {
        "traded_price_1m": {"trailing_contiguous_rows": 300, "last_complete_bar_age_seconds": 90},
        "mark_price_1m": {"trailing_contiguous_rows": 300, "last_complete_bar_age_seconds": 90},
        "funding_1h": {"trailing_contiguous_rows": 25, "last_complete_bar_age_seconds": 4000},
        "open_interest_1h": {"trailing_contiguous_rows": 24, "last_complete_bar_age_seconds": 3600},
    }
    result = assess_feature_warmup(profiles)
    assert result["ready"] is False
    assert result["checks"]["funding_1h"]["freshness_ok"] is False
    assert result["checks"]["open_interest_1h"]["rows_ok"] is False
