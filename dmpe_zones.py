"""Structure-based demand/supply and role-reversal zones for DMPE.

The detector uses completed OHLC bars only. It identifies compact bases followed
by displacement, tracks retests/invalidation, and promotes broken zones to RBS
or SBR only after a close-through followed by a successful retest.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def _atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    prev = frame["close"].shift(1)
    tr = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev).abs(),
            (frame["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=3).mean()


def detect_structure_zones(
    candles: pd.DataFrame,
    *,
    impulse_atr: float = 1.6,
    base_atr_max: float = 1.15,
    max_zones_per_side: int = 6,
) -> list[dict[str, Any]]:
    """Return ranked, non-overlapping zones with real price boundaries.

    A demand base is the last bearish/small-bodied candle before a bullish
    displacement; supply is the inverse. Zones are not stretched to the chart
    floor and are not merged merely because they are close in price.
    """
    required = {"time", "open", "high", "low", "close"}
    if candles is None or candles.empty or not required.issubset(candles.columns):
        return []

    frame = candles.sort_values("time").drop_duplicates("time").reset_index(drop=True).copy()
    for col in ("open", "high", "low", "close"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    if len(frame) < 12:
        return []

    frame["atr"] = _atr(frame)
    raw: list[dict[str, Any]] = []
    # A displacement is measured over the next three completed bars. The
    # base is selected from the preceding three bars, avoiding one zone per
    # candle during a single impulsive leg.
    for end in range(6, len(frame)):
        atr_value = frame.at[end - 3, "atr"]
        if pd.isna(atr_value) or atr_value <= 0:
            continue
        start = end - 3
        leg = frame.iloc[start + 1 : end + 1]
        displacement_up = float(leg["close"].max() - frame.at[start, "close"])
        displacement_down = float(frame.at[start, "close"] - leg["close"].min())
        direction = None
        strength = 0.0
        if displacement_up >= impulse_atr * atr_value:
            direction, strength = "DEMAND", displacement_up / atr_value
        if displacement_down >= impulse_atr * atr_value and displacement_down > displacement_up:
            direction, strength = "SUPPLY", displacement_down / atr_value
        if direction is None:
            continue

        base_start = max(0, start - 2)
        base_candidates = frame.iloc[base_start : start + 1]
        if direction == "DEMAND":
            eligible = base_candidates[
                (base_candidates["close"] <= base_candidates["open"])
                | ((base_candidates["close"] - base_candidates["open"]).abs()
                   <= (base_candidates["high"] - base_candidates["low"]) * 0.35)
            ]
        else:
            eligible = base_candidates[
                (base_candidates["close"] >= base_candidates["open"])
                | ((base_candidates["close"] - base_candidates["open"]).abs()
                   <= (base_candidates["high"] - base_candidates["low"]) * 0.35)
            ]
        if eligible.empty:
            continue
        base_idx = int(eligible.index[-1])
        base = frame.loc[base_idx]
        base_range = float(base["high"] - base["low"])
        if base_range <= 0 or base_range > base_atr_max * atr_value:
            continue

        raw.append({
            "kind": direction,
            "low": float(base["low"]),
            "high": float(base["high"]),
            "created_at": base["time"],
            "origin_index": base_idx,
            "strength": round(float(strength), 2),
            "origin_type": "BASE_LEG",
            "origin_time": base["time"],
        })

    # Add confirmed swing-origin zones. The complete high-low range of the
    # pivot candle is the zone. A pivot is actionable only after three right-
    # hand candles close and price has displaced away from the origin.
    left_right = 3
    for pivot in range(left_right, len(frame) - left_right):
        pivot_row = frame.iloc[pivot]
        local = frame.iloc[pivot - left_right:pivot + left_right + 1]
        pivot_atr = frame.at[pivot, "atr"]
        if pd.isna(pivot_atr) or float(pivot_atr) <= 0:
            continue
        pivot_high, pivot_low = float(pivot_row["high"]), float(pivot_row["low"])
        is_swing_high = pivot_high >= float(local["high"].max())
        is_swing_low = pivot_low <= float(local["low"].min())
        confirm_index = pivot + left_right
        after = frame.iloc[pivot + 1:confirm_index + 1]
        if after.empty:
            continue
        confirm_time = frame.at[confirm_index, "time"]
        origin_time = pivot_row["time"]

        if is_swing_high:
            departure = pivot_low - float(after["close"].min())
            if departure >= 1.5 * float(pivot_atr):
                raw.append({
                    "kind": "SUPPLY", "low": pivot_low, "high": pivot_high,
                    "created_at": confirm_time, "origin_time": origin_time,
                    "origin_index": confirm_index,
                    "strength": round(min(4.0, departure / float(pivot_atr)), 2),
                    "origin_type": "SWING_HIGH",
                })
        if is_swing_low:
            departure = float(after["close"].max()) - pivot_high
            if departure >= 1.5 * float(pivot_atr):
                raw.append({
                    "kind": "DEMAND", "low": pivot_low, "high": pivot_high,
                    "created_at": confirm_time, "origin_time": origin_time,
                    "origin_index": confirm_index,
                    "strength": round(min(4.0, departure / float(pivot_atr)), 2),
                    "origin_type": "SWING_LOW",
                })

    # Keep one representative zone per structure cluster. Major swing origins
    # outrank nearby minor bases; preserve the selected candle's exact bounds.
    typical_range = float((frame["high"] - frame["low"]).tail(20).median())
    merged: list[dict[str, Any]] = []
    ranked = sorted(
        raw,
        key=lambda z: (1 if str(z.get("origin_type", "")).startswith("SWING_") else 0,
                       float(z["strength"])),
        reverse=True,
    )
    for zone in ranked:
        z_mid = (float(zone["low"]) + float(zone["high"])) / 2.0
        is_swing = str(zone.get("origin_type", "")).startswith("SWING_")
        duplicate = False
        for existing in merged:
            if existing["kind"] != zone["kind"]:
                continue
            overlap = max(0.0, min(existing["high"], zone["high"]) - max(existing["low"], zone["low"]))
            smaller = max(1e-12, min(existing["high"] - existing["low"], zone["high"] - zone["low"]))
            overlap_ratio = overlap / smaller
            e_mid = (float(existing["low"]) + float(existing["high"])) / 2.0
            kept_swing = str(existing.get("origin_type", "")).startswith("SWING_")
            near_major = (
                (is_swing != kept_swing and abs(z_mid - e_mid) <= typical_range * 0.90)
                or (is_swing and kept_swing and abs(z_mid - e_mid) <= typical_range * 0.45)
            )
            if overlap_ratio >= 0.60 or near_major:
                duplicate = True
                break
        if not duplicate:
            merged.append(zone.copy())

    last_close = float(frame["close"].iloc[-1])
    last_time = frame["time"].iloc[-1]
    output: list[dict[str, Any]] = []
    for zone in merged:
        low, high = zone["low"], zone["high"]
        origin = zone["origin_index"]
        later = frame.iloc[origin + 1 :]
        touches = int(((later["low"] <= high) & (later["high"] >= low)).sum())
        # Ignore the first bar that initiates the displacement when counting
        # meaningful returns to the origin.
        if zone["kind"] == "DEMAND":
            broken = bool((later["close"] < low).any())
            if broken and last_close > high:
                # Broken demand reclaimed: it is only a candidate RBS after
                # price closes back above and revisits the band.
                post_break = later.loc[later["close"] < low]
                after = later.loc[post_break.index[-1] + 1 :] if not post_break.empty else later.iloc[0:0]
                if not after.empty and ((after["low"] <= high) & (after["close"] >= low)).any():
                    zone["kind"] = "RBS"
                else:
                    continue
            elif broken:
                continue
        else:
            broken = bool((later["close"] > high).any())
            if broken and last_close < low:
                post_break = later.loc[later["close"] > high]
                after = later.loc[post_break.index[-1] + 1 :] if not post_break.empty else later.iloc[0:0]
                if not after.empty and ((after["high"] >= low) & (after["close"] <= high)).any():
                    zone["kind"] = "SBR"
                else:
                    continue
            elif broken:
                continue

        distance = abs((low + high) / 2 - last_close)
        atr_now = float(frame["atr"].iloc[-1]) if pd.notna(frame["atr"].iloc[-1]) else 0.0
        distance_atr = distance / atr_now if atr_now > 0 else 999.0
        output.append({
            **zone,
            "touches": touches,
            "distance_atr": round(distance_atr, 2),
            "active": True,
            "last_price": last_close,
            "last_time": last_time,
        })

    # Keep the nearest, freshest, strongest zones, while preserving multiple
    # distinct price levels. A zone far from price can still survive if strong.
    demand = [z for z in output if z["kind"] in ("DEMAND", "RBS")]
    supply = [z for z in output if z["kind"] in ("SUPPLY", "SBR")]
    key = lambda z: (z["distance_atr"] - min(z["strength"], 4) * 0.20 + z["touches"] * 0.10)
    selected = sorted(demand, key=key)[:max_zones_per_side]
    selected += sorted(supply, key=key)[:max_zones_per_side]
    return sorted(selected, key=lambda z: z["low"])


def add_zones_to_chart(fig: Any, zones: list[dict[str, Any]], view: pd.DataFrame) -> Any:
    """Draw price-aligned horizontal bands from origin to the visible right edge."""
    if not zones or view.empty:
        return fig
    x0 = view["time"].iloc[0]
    x1 = view["time"].iloc[-1]
    if len(view) > 1:
        extension = view["time"].iloc[-1] - view["time"].iloc[-2]
        x1 = x1 + extension * max(8, int(len(view) * 0.12))
    palette = {
        "DEMAND": ("rgba(34,197,94,0.16)", "#4ade80"),
        "RBS": ("rgba(16,185,129,0.16)", "#34d399"),
        "SUPPLY": ("rgba(249,115,22,0.16)", "#fb923c"),
        "SBR": ("rgba(239,68,68,0.15)", "#f87171"),
    }
    view_start = view["time"].iloc[0]
    for index, zone in enumerate(zones):
        created = zone["created_at"]
        if created < view_start:
            created = view_start
        fill, line = palette[zone["kind"]]
        fig.add_shape(
            type="rect", xref="x", yref="y",
            x0=created, x1=x1, y0=zone["low"], y1=zone["high"],
            fillcolor=fill, line={"color": line, "width": 1.2},
            layer="below", row=1, col=1,
        )
        fig.add_annotation(
            x=x1, y=(zone["low"] + zone["high"]) / 2,
            xref="x", yref="y",
            text=f'{zone["kind"]} · {zone.get("origin_type", "BASE_LEG")} · {zone["strength"]:.1f}x',
            showarrow=False, xanchor="left", font={"size": 9, "color": line},
            bgcolor="rgba(15,23,42,0.65)", row=1, col=1,
        )
    return fig
