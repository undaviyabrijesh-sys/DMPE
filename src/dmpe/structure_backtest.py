"""Gap-aware conservative next-minute-open replay for DMPE structure candidates."""
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

    The signal must be executable at the exact 1m open equal to its completed
    5m availability timestamp. Stale signals are skipped rather than entered
    after a data outage. If a 1m gap appears while a position is open, or the
    dataset ends before the time exit, the trade is marked censored and carries
    no P&L. Such rows must be excluded from performance metrics.

    Stops use the completed 5m signal candle extreme plus an ATR buffer. If
    stop and target are both touched in one 1m candle, stop is assumed first.
    No maker fills are assumed.
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
    px = (
        px.dropna(subset=["time", "open", "high", "low", "close"])
        .drop_duplicates("time", keep="last")
        .reset_index(drop=True)
    )
    if px.empty:
        return pd.DataFrame()

    times = px["time"].astype("int64").to_numpy()
    opens = px["open"].to_numpy(dtype=float)
    highs = px["high"].to_numpy(dtype=float)
    lows = px["low"].to_numpy(dtype=float)
    closes = px["close"].to_numpy(dtype=float)
    minute_ns = pd.Timedelta(minutes=1).value
    max_hold_ns = max_holding_minutes * minute_ns
    cost_rate = (fee_bps_per_side + slippage_bps_per_side) / 10000.0

    trades: list[dict] = []
    next_allowed_index = 0

    for row in sig.itertuples(index=False):
        direction = int(row.signal)
        if direction not in (-1, 1):
            continue
        signal_time = row.available_at
        entry_idx = int(np.searchsorted(times, signal_time.value, side="left"))
        # An entry is valid only at the very next 1m open. Never chase a stale
        # signal after one or more missing minutes.
        if entry_idx >= len(px) or times[entry_idx] != signal_time.value:
            continue
        if entry_idx < next_allowed_index:
            continue

        atr = float(row.atr20) if pd.notna(row.atr20) else np.nan
        if not np.isfinite(atr) or atr <= 0:
            continue

        raw_entry = opens[entry_idx]
        buffer = atr * atr_buffer_fraction
        stop = float(row.low) - buffer if direction == 1 else float(row.high) + buffer
        risk = raw_entry - stop if direction == 1 else stop - raw_entry
        if not np.isfinite(risk) or risk <= 0:
            continue
        target = raw_entry + direction * risk * reward_risk

        max_exit_ns = times[entry_idx] + max_hold_ns - minute_ns
        exit_idx = int(np.searchsorted(times, max_exit_ns, side="right") - 1)
        exit_idx = min(exit_idx, len(px) - 1)
        final_reason = "time"
        exit_price = np.nan
        censored = False
        gap_minutes = 0
        last_observed_time = pd.NaT
        last_observed_close = np.nan
        gap_resume_time = pd.NaT

        for idx in range(entry_idx, exit_idx + 1):
            if idx > entry_idx and times[idx] - times[idx - 1] != minute_ns:
                # Intratrade path is unknowable across a missing interval.
                censored = True
                final_reason = "data_gap"
                gap_minutes = max(int((times[idx] - times[idx - 1]) // minute_ns) - 1, 1)
                last_observed_time = px.iloc[idx - 1]["time"]
                last_observed_close = closes[idx - 1]
                gap_resume_time = px.iloc[idx]["time"]
                break

            o, h, l = opens[idx], highs[idx], lows[idx]
            if direction == 1:
                if l <= stop:
                    exit_price = min(o, stop)
                    exit_idx, final_reason = idx, "stop"
                    break
                if h >= target:
                    exit_price = target
                    exit_idx, final_reason = idx, "target"
                    break
            else:
                if h >= stop:
                    exit_price = max(o, stop)
                    exit_idx, final_reason = idx, "stop"
                    break
                if l <= target:
                    exit_price = target
                    exit_idx, final_reason = idx, "target"
                    break

        # If the file ends before the requested time-exit horizon, no reliable
        # realized outcome exists; keep the audit row but censor its P&L.
        expected_exit_ns = times[entry_idx] + max_hold_ns - minute_ns
        if (
            not censored
            and final_reason == "time"
            and exit_idx == len(px) - 1
            and times[exit_idx] < expected_exit_ns
        ):
            censored = True
            final_reason = "end_of_data"
            last_observed_time = px.iloc[exit_idx]["time"]
            last_observed_close = closes[exit_idx]

        if censored:
            trades.append({
                "signal_time": signal_time,
                "entry_time": px.iloc[entry_idx]["time"],
                "exit_time": pd.NaT,
                "direction": direction,
                "entry": raw_entry,
                "stop": stop,
                "target": target,
                "exit": np.nan,
                "exit_reason": final_reason,
                "risk_points": risk,
                "gross_pnl_points": np.nan,
                "cost_points": np.nan,
                "net_pnl_points": np.nan,
                "net_r": np.nan,
                "censored": True,
                "gap_minutes": gap_minutes,
                "last_observed_time": last_observed_time,
                "last_observed_close": last_observed_close,
                "gap_resume_time": gap_resume_time,
            })
            # Treat the censored position as closed at the data discontinuity;
            # no subsequent price path is inferred across the outage.
            next_allowed_index = entry_idx + 1 if final_reason == "end_of_data" else int(
                np.searchsorted(times, gap_resume_time.value, side="left")
            )
            continue

        exit_price = float(exit_price) if np.isfinite(exit_price) else closes[exit_idx]
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
            "exit_reason": final_reason,
            "risk_points": risk,
            "gross_pnl_points": gross_pnl,
            "cost_points": costs,
            "net_pnl_points": net_pnl,
            "net_r": net_pnl / risk,
            "censored": False,
            "gap_minutes": 0,
            "last_observed_time": pd.NaT,
            "last_observed_close": np.nan,
            "gap_resume_time": pd.NaT,
        })
        next_allowed_index = exit_idx + 1

    return pd.DataFrame(trades)
