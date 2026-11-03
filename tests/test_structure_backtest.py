import numpy as np
import pandas as pd

from dmpe.structure_backtest import replay_structure_candidates


def test_replay_enters_after_signal_and_stop_first_on_ambiguous_bar():
    signal_time = pd.Timestamp("2024-01-01T00:05:00Z")
    signals = pd.DataFrame([{
        "available_at": signal_time, "open": 100.0, "high": 102.0,
        "low": 98.0, "close": 101.0, "atr20": 2.0, "signal": 1,
    }])
    prices = pd.DataFrame([
        {"time": signal_time, "open": 100.0, "high": 105.0, "low": 97.0, "close": 101.0},
        {"time": signal_time + pd.Timedelta(minutes=1), "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
    ])
    trades = replay_structure_candidates(
        signals, prices, reward_risk=2.0, fee_bps_per_side=0,
        slippage_bps_per_side=0, max_holding_minutes=2,
    )
    assert len(trades) == 1
    assert trades.loc[0, "entry_time"] == signal_time
    assert trades.loc[0, "exit_reason"] == "stop"
    assert trades.loc[0, "net_pnl_points"] < 0


def test_replay_skips_overlapping_signals_until_previous_trade_exits():
    base = pd.Timestamp("2024-01-01T00:05:00Z")
    signals = pd.DataFrame([
        {"available_at": base, "open": 100, "high": 101, "low": 99, "close": 100, "atr20": 2, "signal": 1},
        {"available_at": base + pd.Timedelta(minutes=1), "open": 100, "high": 101, "low": 99, "close": 100, "atr20": 2, "signal": 1},
    ])
    prices = pd.DataFrame([
        {"time": base + pd.Timedelta(minutes=i), "open": 100, "high": 101, "low": 99.5, "close": 100}
        for i in range(5)
    ])
    trades = replay_structure_candidates(
        signals, prices, fee_bps_per_side=0, slippage_bps_per_side=0,
        max_holding_minutes=2,
    )
    assert len(trades) == 1


def test_replay_censors_position_when_a_missing_minute_occurs():
    base = pd.Timestamp("2024-01-01T00:05:00Z")
    signals = pd.DataFrame([{
        "available_at": base, "open": 100.0, "high": 101.0,
        "low": 99.0, "close": 100.0, "atr20": 2.0, "signal": 1,
    }])
    prices = pd.DataFrame([
        {"time": base, "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.0},
        {"time": base + pd.Timedelta(minutes=1), "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.0},
        # 00:07 is missing; the bar at 00:08 cannot safely be used to resolve the trade.
        {"time": base + pd.Timedelta(minutes=3), "open": 100.0, "high": 110.0, "low": 99.5, "close": 109.0},
    ])
    trades = replay_structure_candidates(
        signals, prices, reward_risk=2.0, fee_bps_per_side=0,
        slippage_bps_per_side=0, max_holding_minutes=5,
    )
    assert len(trades) == 1
    assert trades.loc[0, "exit_reason"] == "data_gap"
    assert bool(trades.loc[0, "censored"])
    assert pd.isna(trades.loc[0, "net_pnl_points"])
    assert trades.loc[0, "gap_minutes"] == 1


def test_replay_does_not_enter_a_stale_signal_after_a_missing_entry_minute():
    signal_time = pd.Timestamp("2024-01-01T00:05:00Z")
    signals = pd.DataFrame([{
        "available_at": signal_time, "open": 100.0, "high": 101.0,
        "low": 99.0, "close": 100.0, "atr20": 2.0, "signal": 1,
    }])
    prices = pd.DataFrame([
        {"time": signal_time + pd.Timedelta(minutes=1), "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.0},
        {"time": signal_time + pd.Timedelta(minutes=2), "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.0},
    ])
    trades = replay_structure_candidates(
        signals, prices, fee_bps_per_side=0, slippage_bps_per_side=0,
        max_holding_minutes=2,
    )
    assert trades.empty


def test_replay_censors_open_position_at_end_of_data():
    signal_time = pd.Timestamp("2024-01-01T00:05:00Z")
    signals = pd.DataFrame([{
        "available_at": signal_time, "open": 100.0, "high": 101.0,
        "low": 99.0, "close": 100.0, "atr20": 2.0, "signal": 1,
    }])
    prices = pd.DataFrame([
        {"time": signal_time, "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.0},
        {"time": signal_time + pd.Timedelta(minutes=1), "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.0},
    ])
    trades = replay_structure_candidates(
        signals, prices, fee_bps_per_side=0, slippage_bps_per_side=0,
        max_holding_minutes=3,
    )
    assert len(trades) == 1
    assert trades.loc[0, "exit_reason"] == "end_of_data"
    assert bool(trades.loc[0, "censored"])
    assert pd.isna(trades.loc[0, "net_pnl_points"])
