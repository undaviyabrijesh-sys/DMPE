"""Causally finalize Delta 1-minute traded-price and mark-price candle updates."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from dmpe.orderflow import to_utc_timestamp

CANDLE_COLUMNS = ["time", "open", "high", "low", "close", "volume"]


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def build_closed_candle_frame(
    records: Iterable[dict[str, Any]],
    *,
    symbol: str = "BTCUSD",
    close_grace_seconds: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return one final row per fully closed candle, timestamped at candle OPEN.

    Delta's candlestick WebSocket sends repeated snapshots while a candle is
    forming. We retain the latest update by recorder receive time, but only
    after the candle close is behind the receive-time watermark. The first
    partially captured minute and the current/incomplete minute are excluded.
    """
    if not 0 <= close_grace_seconds <= 59:
        raise ValueError("close_grace_seconds must be between 0 and 59")

    record_list = [row for row in records if isinstance(row, dict)]
    received = [
        to_utc_timestamp(row.get("received_at"))
        for row in record_list
        if row.get("received_at") is not None
    ]
    received = [ts for ts in received if ts is not None and not pd.isna(ts)]
    if not received:
        return pd.DataFrame(columns=CANDLE_COLUMNS), {
            "symbol": symbol,
            "updates_seen": 0,
            "closed_candles": 0,
            "reason": "missing_recorder_receive_timestamps",
        }

    capture_start = min(received)
    capture_end = max(received)
    first_eligible_open = capture_start.ceil("min")
    watermark = capture_end - pd.Timedelta(seconds=close_grace_seconds)

    updates: list[dict[str, Any]] = []
    symbol_updates = 0
    invalid_updates = 0
    for record in record_list:
        event = record.get("event", record)
        if not isinstance(event, dict):
            continue
        if str(event.get("type", "")).lower() != "candlestick_1m":
            continue
        if str(event.get("res", "")) != "1m" or event.get("sy") != symbol:
            continue
        symbol_updates += 1

        candle_open = to_utc_timestamp(event.get("cst"))
        received_at = to_utc_timestamp(record.get("received_at"))
        if candle_open is None or received_at is None or pd.isna(candle_open) or pd.isna(received_at):
            invalid_updates += 1
            continue
        candle_close = candle_open + pd.Timedelta(minutes=1)
        if candle_open < first_eligible_open or candle_close > watermark:
            continue

        values = {name: _finite_float(event.get(key)) for name, key in (
            ("open", "o"), ("high", "h"), ("low", "l"), ("close", "c")
        )}
        if any(value is None for value in values.values()):
            invalid_updates += 1
            continue
        open_, high, low, close = (
            values["open"], values["high"], values["low"], values["close"]
        )
        if (
            min(open_, high, low, close) <= 0
            or low > high
            or not low <= open_ <= high
            or not low <= close <= high
        ):
            invalid_updates += 1
            continue

        # Mark-price candles omit volume. Use zero only for that absent field;
        # this value is ignored by the historical feature builder for mark data.
        volume_value = _finite_float(event.get("v"))
        volume = 0.0 if volume_value is None and symbol.startswith("MARK:") else volume_value
        if volume is None or volume < 0:
            invalid_updates += 1
            continue

        updates.append({
            "time": candle_open,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "_received_at": received_at,
            "_event_ts": to_utc_timestamp(event.get("ts")),
        })

    if not updates:
        return pd.DataFrame(columns=CANDLE_COLUMNS), {
            "symbol": symbol,
            "capture_start_utc": capture_start.isoformat(),
            "capture_end_utc": capture_end.isoformat(),
            "receive_watermark_utc": watermark.isoformat(),
            "first_eligible_open_utc": first_eligible_open.isoformat(),
            "updates_seen": symbol_updates,
            "invalid_updates": invalid_updates,
            "closed_candles": 0,
        }

    frame = pd.DataFrame(updates).sort_values(["time", "_received_at", "_event_ts"])
    frame = frame.drop_duplicates("time", keep="last")
    frame = frame[CANDLE_COLUMNS].sort_values("time").reset_index(drop=True)
    report = {
        "symbol": symbol,
        "capture_start_utc": capture_start.isoformat(),
        "capture_end_utc": capture_end.isoformat(),
        "receive_watermark_utc": watermark.isoformat(),
        "first_eligible_open_utc": first_eligible_open.isoformat(),
        "updates_seen": symbol_updates,
        "invalid_updates": invalid_updates,
        "closed_candles": int(len(frame)),
        "first_candle_open_utc": frame["time"].min().isoformat(),
        "last_candle_open_utc": frame["time"].max().isoformat(),
        "last_candle_close_utc": (frame["time"].max() + pd.Timedelta(minutes=1)).isoformat(),
        "duplicate_updates_collapsed": int(len(updates) - len(frame)),
        "volume_policy": "Mark-price candle volume absent -> 0; traded-price volume must be present.",
    }
    return frame, report
