import numpy as np
import pandas as pd

from dmpe.features import build_feature_frame, normalize_ohlcv, profile_series


def make_price_frame(periods=420):
    times = pd.date_range("2024-01-01T00:00:00Z", periods=periods, freq="1min")
    close = 100.0 * np.power(1.0001, np.arange(periods))
    return pd.DataFrame({
        "time": times,
        "open": close * 0.9999,
        "high": close * 1.0005,
        "low": close * 0.9995,
        "close": close,
        "volume": np.linspace(10.0, 30.0, periods),
    })


def test_normalize_ohlcv_sorts_deduplicates_and_does_not_fill_gaps():
    frame = make_price_frame(10).iloc[[0, 1, 3, 4, 5, 5]].copy()
    normalized = normalize_ohlcv(frame)
    assert normalized["time"].is_monotonic_increasing
    assert not normalized["time"].duplicated().any()
    assert len(normalized) == 5
    assert normalized["time"].diff().dt.total_seconds().max() == 120


def test_profile_counts_missing_intervals_and_gap_events():
    frame = make_price_frame(10).drop(index=[4, 5]).reset_index(drop=True)
    profile = profile_series("sample", frame, 60)
    assert profile["rows"] == 8
    assert profile["missing_intervals"] == 2
    assert profile["gap_count"] == 1
    assert profile["longest_gap_minutes"] == 3.0


def test_feature_builder_creates_targets_and_respects_gaps():
    frame = make_price_frame(420).drop(index=[100, 101]).reset_index(drop=True)
    output, feature_cols, label_cols = build_feature_frame(frame)
    assert "log_return_15m_pct" in feature_cols
    assert "realized_vol_60m_pct" in feature_cols
    assert "target_forward_return_15m_pct" in label_cols
    assert "target_future_max_up_3h_pct" in label_cols
    assert len(output) == 420
    # The missing minute rows remain explicit NaN OHLC rows.
    assert output["close"].isna().sum() == 2
    # 90th minute's 15-minute forward window crosses the missing candles.
    row_90 = output.loc[output["time"] == frame.loc[90, "time"] + pd.Timedelta(minutes=1)].iloc[0]
    assert pd.isna(row_90["target_forward_return_15m_pct"])
    # Rolling returns also reset around a market-data gap.
    time_after_gap = make_price_frame(420).loc[102, "time"] + pd.Timedelta(minutes=1)
    row_102 = output.loc[output["time"] == time_after_gap].iloc[0]
    assert pd.isna(row_102["log_return_1m_pct"])


def test_feature_columns_do_not_depend_on_future_prices():
    original = make_price_frame(420)
    altered = original.copy()
    altered.loc[250:, ["open", "high", "low", "close", "volume"]] *= 1.8
    first, feature_cols, _ = build_feature_frame(original)
    second, _, _ = build_feature_frame(altered)
    cutoff = original.loc[200, "time"]
    left = first.loc[first["time"] <= cutoff, feature_cols]
    right = second.loc[second["time"] <= cutoff, feature_cols]
    pd.testing.assert_frame_equal(left, right)


def test_hourly_funding_is_delayed_before_join():
    price = make_price_frame(240)
    hour_times = pd.date_range("2024-01-01T00:00:00Z", periods=4, freq="1h")
    rates = [-0.0001, 0.0, 0.0003, 0.0004]
    funding = pd.DataFrame({
        "time": hour_times, "open": rates, "high": rates,
        "low": rates, "close": rates, "volume": 0.0,
    })
    output, feature_cols, _ = build_feature_frame(price, funding_frame=funding)
    assert "funding_funding_rate" in feature_cols
    before_available = output.loc[output["time"] == pd.Timestamp("2024-01-01T01:00:00Z"),
                                  "funding_funding_rate"].iloc[0]
    at_available = output.loc[output["time"] == pd.Timestamp("2024-01-01T01:01:00Z"),
                              "funding_funding_rate"].iloc[0]
    assert pd.isna(before_available)
    assert at_available == rates[0]
    quality = profile_series("funding_rate_1h", funding, 3600)
    assert quality["invalid_ohlc_rows"] == 0


def test_feature_timestamps_represent_completed_bar_close():
    frame = make_price_frame(30)
    output, _, _ = build_feature_frame(frame)
    assert output["time"].iloc[0] == frame["time"].iloc[0] + pd.Timedelta(minutes=1)
    assert output["time"].iloc[-1] == frame["time"].iloc[-1] + pd.Timedelta(minutes=1)
