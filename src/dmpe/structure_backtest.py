"""Conservative next-minute-open replay for DMPE structure candidates."""
from __future__ import annotations

import numpy as np
import pandas as pd


def replay_structure_candidates(
    signals: pd.DataFrame,
    one_minute: pd.DataFrame,
    *,
    reward_risk: float = 2.0,
    fee_bps_per_side: float = 5.9,
    slippage_bps_per_side: float = 1.0,
    max_holding_minutes: int = 60,
    atr_buffer_fraction: float = 0.1,
) -> pd.DataFrame:
    """Replay one independent module's candidates; one position at a time.

    Stops use the completed 5m signal candle extreme plus an ATR buffer.
    Entries occur at the first 1m open at/after signal availability. If stop and
    target are both touched in one 1m bar, stop is assumed first. No maker fills.
    """
    if reward_risk <= 0 or max_holding_minutes < 1:
        raise ValueError("reward_risk and max_holding_minutes must be positive")
    if fee_bps_per_side < 0 or slippage_bps_per_side < 0:
        raise ValueError("cost assumptions cannot be negative")
    required_signal = {"available_at", "open", "high", "low", "close", "atr20", "signal"}
    required_price = {"time", "open", "high", "low", "close"}
    if not required_signal.issubset(signals.columns):
        raise ValueError(f"signals missing columns: {sorted(required_signal - set(signals.columns))}")
    if not required_price.issubset(one_minute.columns):
        raise ValueError(f"one_minute missing columns: {sorted(required_price - set(one_minute.columns))}")

    sig = signals.copy().sort_values("available_at").reset_index(drop=True)
    px = one_minute.copy().sort_values("time").reset_index(drop=True)
    sig["available_at"] = pd.to_datetime(sig["available_at"], utc=True)
    px["time"] = pd.to_datetime(px["time"], utc=True)
    px = px.dropna(subset=["time", "open", "high", "low", "close"]).reset_index(drop=True)
    times = px["time"].astype("int64").to_numpy()
    cost_rate = (fee_bps_per_side + slippage_bps_per_side) / 10000.0
    trades: list[dict] = []
    next_allowed_index = 0

    for row in sig.itertuples(index=False):
        direction = int(row.signal)
        if direction not in (-1, 1):
            continue
        signal_time = row.available_at
        entry_idx = int(np.searchsorted(times, signal_time.value, side="left"))
        if entry_idx >= len(px) or entry_idx < next_allowed_index:
            continue
        atr = float(row.atr20) if pd.notna(row.atr20) else np.nan
        if not np.isfinite(atr) or atr <= 0:
            continue
        raw_entry = float(px.iloc[entry_idx]["open"])
        buffer = atr * atr_buffer_fraction
        stop = float(row.low) - buffer if direction == 1 else float(row.high) + buffer
        risk = raw_entry - stop if direction == 1 else stop - raw_entry
        if not np.isfinite(risk) or risk <= 0:
            continue
        target = raw_entry + direction * risk * reward_risk
        exit_idx = min(entry_idx + max_holding_minutes - 1, len(px) - 1)
        exit_price = float(px.iloc[exit_idx]["close"])
        exit_reason = "time"
        for idx in range(entry_idx, exit_idx + 1):
            candle = px.iloc[idx]
            o, h, l = float(candle.open), float(candle.high), float(candle.low)
            if direction == 1:
                stop_hit, target_hit = l <= stop, h >= target
                if stop_hit:
                    exit_price = min(o, stop)
                    exit_idx, exit_reason = idx, "stop"
                    break
                if target_hit:
                    exit_price = target
                    exit_idx, exit_reason = idx, "target"
                    break
            else:
                stop_hit, target_hit = h >= stop, l <= target
                if stop_hit:
                    exit_price = max(o, stop)
                    exit_idx, exit_reason = idx, "stop"
                    break
                if target_hit:
                    exit_price = target
                    exit_idx, exit_reason = idx, "target"
                    break

        gross_pnl = direction * (exit_price - raw_entry)
        costs = (raw_entry + exit_price) * cost_rate
        net_pnl = gross_pnl - costs
        trades.append({
            "signal_time": signal_time,
            "entry_time": px.iloc[entry_idx]["time"],
            "exit_time": px.iloc[exit_idx]["time"],
            "direction": direction,
            "entry": raw_entry,
            "stop": stop,
            "target": target,
            "exit": exit_price,
            "exit_reason": exit_reason,
            "risk_points": risk,
            "gross_pnl_points": gross_pnl,
            "cost_points": costs,
            "net_pnl_points": net_pnl,
            "net_r": net_pnl / risk,
        })
        next_allowed_index = exit_idx + 1

    return pd.DataFrame(trades)
