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
