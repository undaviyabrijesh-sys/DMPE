import numpy as np
import pandas as pd

from dmpe.structure_engine import (
    build_mtf_signal_frame, build_structure_frame, resample_completed_ohlcv,
)


def make_minutes(periods=6000):
    times = pd.date_range("2024-01-01T00:00:00Z", periods=periods, freq="1min")
    x = np.arange(periods, dtype=float)
    close = 30000 + x * 0.5 + np.sin(x / 37) * 12
    return pd.DataFrame({
        "time": times, "open": close - 0.2, "high": close + 2,
        "low": close - 2, "close": close, "volume": 1 + (x % 20),
    })


def test_resample_uses_only_complete_bars_and_close_availability_time():
    source = make_minutes(12)
    result = resample_completed_ohlcv(source, 5)
    assert len(result) == 2
    assert result["available_at"].iloc[0] == pd.Timestamp("2024-01-01T00:05:00Z")
    assert result["available_at"].iloc[-1] == pd.Timestamp("2024-01-01T00:10:00Z")


def test_missing_minute_invalidates_affected_higher_timeframe_bucket():
    source = make_minutes(20).drop(index=7)
    result = resample_completed_ohlcv(source, 5)
    assert len(result) == 3
    assert pd.Timestamp("2024-01-01T00:10:00Z") not in set(result["available_at"])


def test_pivot_is_not_available_until_right_confirmation_bars_close():
    frame = pd.DataFrame({
        "available_at": pd.date_range("2024-01-01", periods=80, freq="1h", tz="UTC"),
        "open": np.arange(80) + 100.0,
        "high": np.array([100 + i for i in range(40)] + [140 - i for i in range(40)], dtype=float),
        "low": np.arange(80) + 90.0,
        "close": np.arange(80) + 99.0,
        "volume": 1.0,
    })
    result = build_structure_frame(frame, left=2, right=2)
    # The high at row 40 is confirmed only on row 42.
    assert np.isnan(result.loc[40, "confirmed_swing_high"])
    assert result.loc[42, "confirmed_swing_high"] == 140.0


def test_mtf_builder_exposes_all_three_modules_and_never_future_context():
    source = make_minutes(24 * 60 * 8)
    result = build_mtf_signal_frame(source)
    required = {
        "4h_structure_bias", "1h_structure_bias", "15m_structure_bias",
        "entry_pullback", "entry_sweep", "entry_breakout_retest", "entry_candidate",
    }
    assert required.issubset(result.columns)
    # The latest minute cannot see a context bar whose close is in the future.
    assert (result["4h_available_at"] <= result["available_at"]).all()
    assert (result["1h_available_at"] <= result["available_at"]).all()
