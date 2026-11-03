"""Multi-timeframe market-structure and entry research engine.

Input candle timestamps are candle OPEN times (Delta convention). Resampled bars
are timestamped at their CLOSE/availability time. Only fully formed bars and
confirmed pivots are exposed to lower-timeframe decisions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dmpe.features import normalize_ohlcv

TIMEFRAME_MINUTES = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}


def resample_completed_ohlcv(frame: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Aggregate 1m candles; reject incomplete buckets and buckets with gaps."""
    if minutes < 1:
        raise ValueError("minutes must be positive")
    one = normalize_ohlcv(frame).set_index("time").sort_index()
    rule = f"{minutes}min"
    grouped = one.resample(rule, origin="epoch", label="right", closed="left")
    out = grouped.agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"), source_rows=("close", "count"),
    )
    # Do not construct a synthetic HTF candle across missing source minutes.
    out = out.loc[out["source_rows"] == minutes].dropna(subset=["open", "high", "low", "close"])
    out = out.drop(columns="source_rows").reset_index()
    return out.rename(columns={"time": "available_at"})


def _confirmed_pivots(frame: pd.DataFrame, left: int, right: int) -> pd.DataFrame:
    """Return pivot values only on the bar where right-side confirmation completes."""
    out = frame.copy()
    width = left + right + 1
    highs = out["high"].to_numpy(dtype=float)
    lows = out["low"].to_numpy(dtype=float)
    pivot_high = np.full(len(out), np.nan)
    pivot_low = np.full(len(out), np.nan)
    for confirm_idx in range(width - 1, len(out)):
        pivot_idx = confirm_idx - right
        start = pivot_idx - left
        end = pivot_idx + right + 1
        high_window = highs[start:end]
        low_window = lows[start:end]
        if np.isfinite(high_window).all() and highs[pivot_idx] == np.max(high_window):
            # Require a unique extreme to avoid ambiguous flat-top/bottom pivots.
            if np.count_nonzero(high_window == highs[pivot_idx]) == 1:
                pivot_high[confirm_idx] = highs[pivot_idx]
        if np.isfinite(low_window).all() and lows[pivot_idx] == np.min(low_window):
            if np.count_nonzero(low_window == lows[pivot_idx]) == 1:
                pivot_low[confirm_idx] = lows[pivot_idx]
    out["confirmed_swing_high"] = pivot_high
    out["confirmed_swing_low"] = pivot_low
    return out


def build_structure_frame(
    candles: pd.DataFrame, *, left: int = 3, right: int = 2
) -> pd.DataFrame:
    """Build trend/structure descriptors from a completed timeframe candle frame."""
    if left < 1 or right < 1:
        raise ValueError("Pivot left/right confirmation bars must be at least one.")
    bars = candles.copy().sort_values("available_at").reset_index(drop=True)
    if bars.empty:
        return bars
    bars = _confirmed_pivots(bars, left, right)
    bars["ema20"] = bars["close"].ewm(span=20, adjust=False, min_periods=20).mean()
    bars["ema50"] = bars["close"].ewm(span=50, adjust=False, min_periods=50).mean()

    high_events = bars["confirmed_swing_high"].dropna()
    low_events = bars["confirmed_swing_low"].dropna()
    bars["last_swing_high"] = bars["confirmed_swing_high"].ffill()
    bars["last_swing_low"] = bars["confirmed_swing_low"].ffill()
    bars["previous_swing_high"] = np.nan
    bars["previous_swing_low"] = np.nan
    if not high_events.empty:
        previous = high_events.shift(1)
        bars.loc[high_events.index, "previous_swing_high"] = previous
        bars["previous_swing_high"] = bars["previous_swing_high"].ffill()
    if not low_events.empty:
        previous = low_events.shift(1)
        bars.loc[low_events.index, "previous_swing_low"] = previous
        bars["previous_swing_low"] = bars["previous_swing_low"].ffill()

    higher_high = bars["last_swing_high"] > bars["previous_swing_high"]
    lower_high = bars["last_swing_high"] < bars["previous_swing_high"]
    higher_low = bars["last_swing_low"] > bars["previous_swing_low"]
    lower_low = bars["last_swing_low"] < bars["previous_swing_low"]
    bullish_structure = higher_high & higher_low
    bearish_structure = lower_high & lower_low
    ema_bull = (bars["close"] > bars["ema50"]) & (bars["ema20"] > bars["ema50"])
    ema_bear = (bars["close"] < bars["ema50"]) & (bars["ema20"] < bars["ema50"])
    bars["structure_bias"] = np.select(
        [bullish_structure & ema_bull, bearish_structure & ema_bear],
        [1, -1], default=0,
    ).astype("int8")
    bars["structure_break_up"] = (
        bars["close"] > bars["last_swing_high"].shift(1)
    ) & bars["last_swing_high"].shift(1).notna()
    bars["structure_break_down"] = (
        bars["close"] < bars["last_swing_low"].shift(1)
    ) & bars["last_swing_low"].shift(1).notna()
    return bars


def _merge_context(base: pd.DataFrame, context: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cols = [
        "available_at", "structure_bias", "last_swing_high", "last_swing_low",
        "ema20", "ema50", "structure_break_up", "structure_break_down",
    ]
    right = context[cols].copy().rename(
        columns={col: f"{prefix}_{col}" for col in cols}
    )
    return pd.merge_asof(
        base.sort_values("available_at"),
        right.sort_values(f"{prefix}_available_at"),
        left_on="available_at", right_on=f"{prefix}_available_at",
        direction="backward",
    )


def build_mtf_signal_frame(
    price_1m: pd.DataFrame,
    *,
    pivot_left: int = 3,
    pivot_right: int = 2,
    lookback: int = 20,
) -> pd.DataFrame:
    """Create experimental 5m entries gated by completed 4h/1h/15m context.

    Entry modules:
      1. pullback: trend-aligned EMA20 retest followed by prior-bar break;
      2. sweep: 20-bar liquidity sweep and close back inside the prior range;
      3. breakout_retest: range break followed by a retest within five 5m bars.

    The function creates research candidates only; it does not place orders.
    """
    if lookback < 5:
        raise ValueError("lookback must be at least 5")
    bars = resample_completed_ohlcv(price_1m, 5)
    if bars.empty:
        return bars
    bars["ema20"] = bars["close"].ewm(span=20, adjust=False, min_periods=20).mean()
    bars["prior_high"] = bars["high"].shift(1).rolling(lookback, min_periods=lookback).max()
    bars["prior_low"] = bars["low"].shift(1).rolling(lookback, min_periods=lookback).min()
    bars["atr20"] = (bars["high"] - bars["low"]).rolling(20, min_periods=20).mean()

    for timeframe in ("15m", "1h", "4h"):
        context_bars = resample_completed_ohlcv(price_1m, TIMEFRAME_MINUTES[timeframe])
        context = build_structure_frame(context_bars, left=pivot_left, right=pivot_right)
        bars = _merge_context(bars, context, timeframe)

    htf_long = (bars["1h_structure_bias"] == 1) & (bars["4h_structure_bias"] == 1)
    htf_short = (bars["1h_structure_bias"] == -1) & (bars["4h_structure_bias"] == -1)
    setup_long = bars["15m_structure_bias"] == 1
    setup_short = bars["15m_structure_bias"] == -1
    bars["htf_bias"] = np.select([htf_long, htf_short], [1, -1], default=0).astype("int8")

    touched_ema = (bars["low"] <= bars["ema20"]) & (bars["close"] >= bars["ema20"])
    touched_ema_short = (bars["high"] >= bars["ema20"]) & (bars["close"] <= bars["ema20"])
    bars["entry_pullback_long"] = (
        htf_long & setup_long & touched_ema & (bars["close"] > bars["high"].shift(1))
    )
    bars["entry_pullback_short"] = (
        htf_short & setup_short & touched_ema_short & (bars["close"] < bars["low"].shift(1))
    )

    bars["entry_sweep_long"] = (
        htf_long & (bars["low"] < bars["prior_low"]) & (bars["close"] > bars["prior_low"])
    )
    bars["entry_sweep_short"] = (
        htf_short & (bars["high"] > bars["prior_high"]) & (bars["close"] < bars["prior_high"])
    )

    broke_up = bars["close"].shift(1) > bars["prior_high"].shift(1)
    broke_down = bars["close"].shift(1) < bars["prior_low"].shift(1)
    recent_up = broke_up.rolling(5, min_periods=1).max().astype(bool)
    recent_down = broke_down.rolling(5, min_periods=1).max().astype(bool)
    bars["entry_breakout_retest_long"] = (
        htf_long & setup_long & recent_up & (bars["low"] <= bars["prior_high"])
        & (bars["close"] > bars["prior_high"])
    )
    bars["entry_breakout_retest_short"] = (
        htf_short & setup_short & recent_down & (bars["high"] >= bars["prior_low"])
        & (bars["close"] < bars["prior_low"])
    )

    for module in ("pullback", "sweep", "breakout_retest"):
        bars[f"entry_{module}"] = (
            bars[f"entry_{module}_long"].fillna(False).astype(bool).astype("int8")
            - bars[f"entry_{module}_short"].fillna(False).astype(bool).astype("int8")
        )
    bars["entry_candidate"] = np.select(
        [bars[[f"entry_{m}_long" for m in ("pullback", "sweep", "breakout_retest")]].any(axis=1),
         bars[[f"entry_{m}_short" for m in ("pullback", "sweep", "breakout_retest")]].any(axis=1)],
        [1, -1], default=0,
    ).astype("int8")
    return bars
