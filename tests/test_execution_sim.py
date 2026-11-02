import numpy as np
import pandas as pd

from dmpe.execution_sim import (
    ExecutionAssumptions,
    simulate_signal_set,
)


def make_inputs():
    times = pd.date_range("2024-01-01T00:00:00Z", periods=20, freq="1min")
    opens = np.full(20, 100.0)
    highs = np.full(20, 100.05)
    lows = np.full(20, 99.95)
    closes = np.full(20, 100.0)
    # First future bar reaches the long target, while not hitting the stop.
    highs[0], lows[0] = 100.30, 100.10
    prices = pd.DataFrame({
        "time": times, "open": opens, "high": highs,
        "low": lows, "close": closes,
    })
    features = pd.DataFrame({
        "time": times,
        "realized_vol_15m_pct": 0.20,
        "realized_vol_60m_pct": 0.20,
        "realized_vol_180m_pct": 0.20,
    })
    signals = pd.DataFrame([{
        "time": times[0], "fold": 1, "horizon": "15m",
        "barrier_multiple": 1.0, "signal": 1,
        "p_down": 0.10, "p_flat": 0.20, "p_up": 0.70,
        "validation_threshold": 0.60,
    }])
    return signals, prices, features


def test_target_exit_includes_taker_fees_and_two_sided_slippage():
    signals, prices, features = make_inputs()
    assumptions = ExecutionAssumptions(
        taker_fee_bps_per_side_including_gst=5.9,
        slippage_bps_per_side=2.0,
        lot_size_btc=0.001,
        contracts=1.0,
    )
    trades, report = simulate_signal_set(signals, prices, features, assumptions)
    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["exit_reason"] == "target"
    assert trade["entry_fill_price"] > trade["raw_entry_price"]
    assert trade["exit_fill_price"] < trade["raw_exit_price"]
    assert trade["fees_usd"] > 0
    assert trade["net_pnl_usd"] < trade["gross_pnl_usd"]
    assert report["executed_trades"] == 1
    assert report["fees_usd"] == trade["fees_usd"]


def test_same_minute_stop_and_target_uses_conservative_stop_first():
    signals, prices, features = make_inputs()
    prices.loc[0, ["high", "low"]] = [100.50, 99.50]
    trades, _ = simulate_signal_set(signals, prices, features, ExecutionAssumptions())
    assert trades.iloc[0]["exit_reason"] == "stop_and_target_same_bar_stop_first"
    assert trades.iloc[0]["net_pnl_usd"] < 0


def test_stop_gap_fills_at_worse_open_not_better_stop_level():
    signals, prices, features = make_inputs()
    # Entry is at row 0 open; the following candle gaps below the long stop.
    prices.loc[1, ["open", "high", "low", "close"]] = [99.50, 99.60, 99.30, 99.40]
    trades, _ = simulate_signal_set(signals, prices, features, ExecutionAssumptions(
        slippage_bps_per_side=0.0
    ))
    trade = trades.iloc[0]
    assert trade["exit_reason"] == "stop"
    assert trade["holding_minutes"] == 2
    assert trade["raw_exit_price"] == 99.50


def test_trade_is_skipped_if_any_minute_in_holding_window_is_missing():
    signals, prices, features = make_inputs()
    prices = prices.drop(index=5).reset_index(drop=True)
    trades, report = simulate_signal_set(signals, prices, features, ExecutionAssumptions())
    assert trades.empty
    assert report["skipped"]["incomplete_future_price_window"] == 1
