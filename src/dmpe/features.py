"""Leak-conscious first-pass feature and target builder for DMPE.

All features are point-in-time descriptors. Hourly data are delayed by one full
hour plus one minute before being joined to minute bars. Forward-return and
excursion columns are labels only and must never be passed to a model as inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

PRICE_COLUMNS = ["time", "open", "high", "low", "close", "volume"]
HORIZONS_MINUTES = {"15m": 15, "1h": 60, "3h": 180}
LAG_MINUTES = (1, 3, 5, 15, 30, 60, 180)
VOL_WINDOWS = (15, 60, 180, 240)
TREND_WINDOWS = (15, 60, 240)


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    timeframe_seconds: int
    rows: int
    first_utc: str | None
    last_utc: str | None
    expected_rows_between_endpoints: int
    missing_intervals: int
    gap_count: int
    longest_gap_minutes: float
    median_step_seconds: float | None
    duplicate_timestamps: int
    invalid_ohlc_rows: int


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize exchange candles while preserving missing time intervals."""
    if frame.empty:
        raise ValueError("Candle dataset is empty.")
    aliases = {str(col).strip().lower(): col for col in frame.columns}
    if "time" not in aliases:
        raise ValueError("Candle dataset must contain a 'time' column.")
    missing = set(("open", "high", "low", "close")).difference(aliases)
    if missing:
        raise ValueError(f"Missing required OHLC columns: {sorted(missing)}")

    out = frame.rename(
        columns={aliases[k]: k for k in ("time", "open", "high", "low", "close", "volume") if k in aliases}
    ).copy()
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        if col not in out:
            out[col] = np.nan if col != "volume" else 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["time", "open", "high", "low", "close"])
    out = out.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    invalid = (
        (out["low"] > out["high"])
        | (out["open"] < out["low"]) | (out["open"] > out["high"])
        | (out["close"] < out["low"]) | (out["close"] > out["high"])
        | (out[["open", "high", "low", "close"]] <= 0).any(axis=1)
    )
    out = out.loc[~invalid, PRICE_COLUMNS].reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid OHLC rows remain after validation.")
    return out


def profile_series(name: str, frame: pd.DataFrame, timeframe_seconds: int) -> dict[str, Any]:
    """Return cadence and integrity statistics, including actual gap lengths."""
    if frame.empty:
        return DatasetProfile(
            name, timeframe_seconds, 0, None, None, 0, 0, 0, 0.0, None, 0, 0
        ).__dict__

    work = frame.copy()
    work["time"] = pd.to_datetime(work["time"], utc=True, errors="coerce")
    invalid_times = int(work["time"].isna().sum())
    work = work.dropna(subset=["time"]).sort_values("time")
    duplicate_count = int(work["time"].duplicated().sum())
    work = work.drop_duplicates("time")
    deltas = work["time"].diff().dt.total_seconds().dropna()
    expected = int((work["time"].iloc[-1] - work["time"].iloc[0]).total_seconds() // timeframe_seconds) + 1
    missing = max(expected - len(work), 0)
    gap_deltas = deltas[deltas > timeframe_seconds]
    gap_count = int(len(gap_deltas))
    longest_gap = float(gap_deltas.max() / 60) if gap_count else 0.0
    invalid_ohlc = 0
    if {"open", "high", "low", "close"}.issubset(work.columns):
        bad = (
            (work["low"] > work["high"])
            | (work["open"] < work["low"]) | (work["open"] > work["high"])
            | (work["close"] < work["low"]) | (work["close"] > work["high"])
        )
        # Market prices and open interest must be positive; funding rates may
        # legitimately be negative or zero.
        if name in {"traded_price_1m", "mark_price_1m", "open_interest_1h"}:
            bad = bad | (work[["open", "high", "low", "close"]] <= 0).any(axis=1)
        invalid_ohlc = int(bad.sum())

    return DatasetProfile(
        name=name,
        timeframe_seconds=timeframe_seconds,
        rows=int(len(work)),
        first_utc=work["time"].iloc[0].isoformat(),
        last_utc=work["time"].iloc[-1].isoformat(),
        expected_rows_between_endpoints=expected,
        missing_intervals=missing,
        gap_count=gap_count,
        longest_gap_minutes=longest_gap,
        median_step_seconds=float(deltas.median()) if len(deltas) else None,
        duplicate_timestamps=duplicate_count,
        invalid_ohlc_rows=invalid_ohlc + invalid_times,
    ).__dict__


def _rolling_forward(values: pd.Series, window: int, operation: str) -> pd.Series:
    future = values.shift(-1).iloc[::-1]
    rolled = getattr(future.rolling(window=window, min_periods=window), operation)()
    return rolled.iloc[::-1]


def _regular_hourly(frame: pd.DataFrame | None, value_name: str) -> pd.DataFrame | None:
    if frame is None or frame.empty:
        return None
    # Funding can be negative or exactly zero; do not run it through the
    # positive-price OHLC validator or logarithms. For both hourly series only
    # the timestamped close observation is used as a point-in-time scalar.
    hourly = frame[["time", "close"]].copy()
    hourly["time"] = pd.to_datetime(hourly["time"], utc=True, errors="coerce")
    hourly["close"] = pd.to_numeric(hourly["close"], errors="coerce")
    hourly = (
        hourly.dropna(subset=["time", "close"])
        .sort_values("time")
        .drop_duplicates("time")
        .set_index("time")
    )
    start, end = hourly.index.min().floor("h"), hourly.index.max().floor("h")
    full_index = pd.date_range(start, end, freq="1h", tz="UTC")
    hourly = hourly.reindex(full_index)
    out = pd.DataFrame(index=full_index)
    out[value_name] = hourly["close"].astype(float)
    if value_name == "open_interest":
        out[value_name] = out[value_name].where(out[value_name] > 0)
    safe_lag = pd.Timedelta(hours=1, minutes=1)
    out["available_at"] = out.index + safe_lag

    if value_name == "funding_rate":
        # Differences remain meaningful across both positive and negative rates.
        for hours in (1, 4, 24):
            col = f"{value_name}_change_{hours}h"
            out[col] = out[value_name].diff(hours)
            valid_count = out[value_name].notna().astype("int8").rolling(
                hours + 1, min_periods=hours + 1
            ).sum()
            out.loc[valid_count < hours + 1, col] = np.nan
    else:
        positive = out[value_name].where(out[value_name] > 0)
        log_value = np.log(positive)
        for hours in (1, 4, 24):
            col = f"{value_name}_log_change_{hours}h"
            out[col] = log_value.diff(hours)
            valid_count = log_value.notna().astype("int8").rolling(
                hours + 1, min_periods=hours + 1
            ).sum()
            out.loc[valid_count < hours + 1, col] = np.nan

    return out.reset_index(names="source_time")


def _asof_hourly(
    base: pd.DataFrame,
    hourly: pd.DataFrame | None,
    prefix: str,
    tolerance_hours: int,
) -> pd.DataFrame:
    if hourly is None:
        return base
    right = hourly.rename(columns={
        "available_at": f"{prefix}_available_at",
        "source_time": f"{prefix}_source_time",
    }).sort_values(f"{prefix}_available_at")
    right = right.rename(columns={
        col: f"{prefix}_{col}"
        for col in right.columns
        if col not in (f"{prefix}_available_at", f"{prefix}_source_time")
    })
    merged = pd.merge_asof(
        base.sort_values("time"),
        right.sort_values(f"{prefix}_available_at"),
        left_on="time",
        right_on=f"{prefix}_available_at",
        direction="backward",
        tolerance=pd.Timedelta(hours=tolerance_hours),
    )
    age_col = f"{prefix}_age_minutes"
    if f"{prefix}_source_time" in merged:
        merged[age_col] = (
            merged["time"] - merged[f"{prefix}_source_time"]
        ).dt.total_seconds() / 60
    return merged


def build_feature_frame(
    price_frame: pd.DataFrame,
    mark_frame: pd.DataFrame | None = None,
    funding_frame: pd.DataFrame | None = None,
    oi_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build 1-minute descriptors and forward labels without filling market prices.

    Features use only current/previous completed bars. Labels use strictly future
    bars and are returned separately in the label-column list.
    """
    price = normalize_ohlcv(price_frame)
    minute_index = pd.date_range(
        price["time"].iloc[0], price["time"].iloc[-1], freq="1min", tz="UTC"
    )
    base = price.set_index("time").reindex(minute_index)
    base.index.name = "time"
    base = base.reset_index()

    close = base["close"].astype(float)
    open_ = base["open"].astype(float)
    high = base["high"].astype(float)
    low = base["low"].astype(float)
    volume = base["volume"].astype(float)
    log_close = np.log(close.where(close > 0))
    one_min_log_return = log_close.diff()
    features: list[str] = []

    def add(name: str, values: pd.Series | np.ndarray) -> None:
        base[name] = values
        features.append(name)

    add("log_return_1m_pct", one_min_log_return * 100.0)
    for minutes in LAG_MINUTES:
        summed = one_min_log_return.rolling(minutes, min_periods=minutes).sum()
        add(f"log_return_{minutes}m_pct", summed * 100.0)
    for window in VOL_WINDOWS:
        rv = one_min_log_return.rolling(window, min_periods=window).std(ddof=0) * np.sqrt(window) * 100
        add(f"realized_vol_{window}m_pct", rv)

    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    true_range = true_range.where(one_min_log_return.notna())
    for window in (14, 60):
        atr = true_range.rolling(window, min_periods=window).mean()
        add(f"atr_{window}m_pct", atr / close * 100.0)
    for window in TREND_WINDOWS:
        average = close.rolling(window, min_periods=window).mean()
        add(f"sma_distance_{window}m_pct", (close / average - 1.0) * 100.0)

    volume_mean = volume.rolling(60, min_periods=60).mean()
    volume_std = volume.rolling(60, min_periods=60).std(ddof=0)
    add("volume_zscore_60m", (volume - volume_mean) / volume_std.replace(0, np.nan))
    add("volume_ratio_60m", volume / volume_mean.replace(0, np.nan))

    range_abs = high - low
    add("candle_body_pct", (close - open_) / open_ * 100.0)
    add("candle_range_pct", range_abs / open_ * 100.0)
    add("close_location", ((close - low) - (high - close)) / range_abs.replace(0, np.nan))
    add("upper_wick_pct", (high - pd.concat([open_, close], axis=1).max(axis=1)) / open_ * 100.0)
    add("lower_wick_pct", (pd.concat([open_, close], axis=1).min(axis=1) - low) / open_ * 100.0)

    hour = base["time"].dt.hour.astype(float)
    weekday = base["time"].dt.dayofweek.astype(float)
    add("hour_sin", np.sin(2 * np.pi * hour / 24.0))
    add("hour_cos", np.cos(2 * np.pi * hour / 24.0))
    add("weekday_sin", np.sin(2 * np.pi * weekday / 7.0))
    add("weekday_cos", np.cos(2 * np.pi * weekday / 7.0))

    if mark_frame is not None and not mark_frame.empty:
        mark = normalize_ohlcv(mark_frame)[["time", "close"]].rename(columns={"close": "mark_close"})
        base = base.merge(mark, on="time", how="left", validate="one_to_one")
        mark_log = np.log(base["mark_close"].where(base["mark_close"] > 0))
        base["mark_basis_pct"] = (base["mark_close"] / base["close"] - 1.0) * 100.0
        base["mark_log_return_1m_pct"] = mark_log.diff() * 100.0
        features.extend(["mark_basis_pct", "mark_log_return_1m_pct"])
    else:
        base["mark_close"] = np.nan
        base["mark_basis_pct"] = np.nan
        base["mark_log_return_1m_pct"] = np.nan
        features.extend(["mark_basis_pct", "mark_log_return_1m_pct"])

    funding_hourly = _regular_hourly(funding_frame, "funding_rate")
    oi_hourly = _regular_hourly(oi_frame, "open_interest")
    base = _asof_hourly(base, funding_hourly, "funding", tolerance_hours=3)
    base = _asof_hourly(base, oi_hourly, "oi", tolerance_hours=2)
    for col in ("funding_funding_rate", "funding_funding_rate_change_1h",
                "funding_funding_rate_change_4h", "funding_funding_rate_change_24h",
                "funding_age_minutes", "oi_open_interest", "oi_open_interest_log_change_1h",
                "oi_open_interest_log_change_4h", "oi_open_interest_log_change_24h",
                "oi_age_minutes"):
        if col in base:
            features.append(col)

    labels: list[str] = []
    future_valid = one_min_log_return.notna().astype("int8")
    for label, horizon in HORIZONS_MINUTES.items():
        future_log_return = _rolling_forward(one_min_log_return, horizon, "sum")
        valid_count = _rolling_forward(future_valid, horizon, "sum")
        fwd_return = (np.exp(future_log_return) - 1.0) * 100.0
        fwd_return = fwd_return.where(valid_count == horizon)
        future_high = _rolling_forward(high, horizon, "max")
        future_low = _rolling_forward(low, horizon, "min")
        valid_price_count = _rolling_forward(base["close"].notna().astype("int8"), horizon, "sum")
        fwd_up = (future_high / close - 1.0) * 100.0
        fwd_down = (future_low / close - 1.0) * 100.0
        fwd_up = fwd_up.where((valid_count == horizon) & (valid_price_count == horizon))
        fwd_down = fwd_down.where((valid_count == horizon) & (valid_price_count == horizon))
        for name, values in (
            (f"target_forward_return_{label}_pct", fwd_return),
            (f"target_future_max_up_{label}_pct", fwd_up),
            (f"target_future_max_down_{label}_pct", fwd_down),
        ):
            base[name] = values
            labels.append(name)

    base = base.replace([np.inf, -np.inf], np.nan)
    # Raw OHLCV columns are retained for inspection; do not use them blindly as
    # features if model architecture expects the normalized descriptor set.
    return base, features, labels
