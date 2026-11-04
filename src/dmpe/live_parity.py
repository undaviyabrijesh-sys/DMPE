"""Compare finalized Delta WebSocket candles with REST history for the same bars."""
from __future__ import annotations

import pandas as pd


def compare_candle_frames(
    live: pd.DataFrame,
    historical: pd.DataFrame,
    *,
    columns: tuple[str, ...] = ("open", "high", "low", "close"),
    absolute_tolerance: float = 1e-8,
    relative_tolerance: float = 1e-10,
) -> dict:
    """Compare every live candle to a same-timestamp historical REST candle."""
    for name, frame in (("live", live), ("historical", historical)):
        if "time" not in frame:
            raise ValueError(f"{name} candle frame is missing time.")
        if frame["time"].duplicated().any():
            raise ValueError(f"{name} candle frame contains duplicate timestamps.")
        missing = set(columns).difference(frame.columns)
        if missing:
            raise ValueError(f"{name} candle frame is missing columns: {sorted(missing)}")

    if live.empty:
        return {
            "live_rows": 0,
            "matched_rows": 0,
            "missing_in_rest": 0,
            "mismatched_values": {},
            "all_live_bars_match_rest": False,
            "reason": "no_finalized_live_candles",
        }

    left = live[["time", *columns]].copy()
    right = historical[["time", *columns]].copy()
    left["time"] = pd.to_datetime(left["time"], utc=True, errors="coerce")
    right["time"] = pd.to_datetime(right["time"], utc=True, errors="coerce")
    if left["time"].isna().any() or right["time"].isna().any():
        raise ValueError("Candle frames contain invalid timestamps.")

    merged = left.merge(right, on="time", how="left", suffixes=("_live", "_rest"), indicator=True)
    matched = merged["_merge"].eq("both")
    missing_in_rest = int((~matched).sum())
    mismatch_summary = {}
    for column in columns:
        live_values = pd.to_numeric(merged[f"{column}_live"], errors="coerce")
        rest_values = pd.to_numeric(merged[f"{column}_rest"], errors="coerce")
        missing_values = matched & (live_values.isna() | rest_values.isna())
        comparable = matched & live_values.notna() & rest_values.notna()
        allowed = absolute_tolerance + relative_tolerance * rest_values.abs()
        mismatch = comparable & ((live_values - rest_values).abs() > allowed)
        mismatch_summary[column] = {
            "compared": int(comparable.sum()),
            "missing_values": int(missing_values.sum()),
            "mismatches": int(mismatch.sum()),
            "max_absolute_difference": (
                float((live_values[comparable] - rest_values[comparable]).abs().max())
                if comparable.any() else None
            ),
        }

    total_mismatches = sum(
        item["mismatches"] + item["missing_values"]
        for item in mismatch_summary.values()
    )
    return {
        "live_rows": int(len(live)),
        "historical_rows": int(len(historical)),
        "matched_rows": int(matched.sum()),
        "missing_in_rest": missing_in_rest,
        "mismatched_values": mismatch_summary,
        "all_live_bars_match_rest": missing_in_rest == 0 and total_mismatches == 0,
        "tolerance": {
            "absolute": absolute_tolerance,
            "relative": relative_tolerance,
        },
    }


def trailing_contiguous_rows(
    frame: pd.DataFrame,
    resolution: str,
) -> int:
    """Count the consecutive cadence-aligned rows at the newest end of a frame."""
    seconds_by_resolution = {"1m": 60, "1h": 3600}
    if resolution not in seconds_by_resolution:
        raise ValueError(f"Unsupported resolution: {resolution}")
    if frame.empty or "time" not in frame:
        return 0
    times = pd.to_datetime(frame["time"], utc=True, errors="coerce").dropna().drop_duplicates().sort_values()
    if times.empty:
        return 0
    differences = times.diff().dt.total_seconds().dropna().to_numpy()
    expected = seconds_by_resolution[resolution]
    count = 1
    for difference in differences[::-1]:
        if difference != expected:
            break
        count += 1
    return count
