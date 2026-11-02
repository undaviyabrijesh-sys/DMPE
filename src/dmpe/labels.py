"""First-touch, horizon-aligned realized-volatility barrier labels for DMPE.

Each horizon uses its own trailing realized-volatility estimate, available at
signal time: 15m uses realized_vol_15m_pct, 1h uses realized_vol_60m_pct,
and 3h uses realized_vol_180m_pct. Barriers are symmetric around the signal
close. If both levels are first touched in the same 1-minute candle, direction
is ambiguous (diagnostic state=2); do not infer intraminute order from OHLC.
Incomplete future windows are null and must be excluded from training.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

DEFAULT_HORIZONS = {"15m": 15, "1h": 60, "3h": 180}
DEFAULT_MULTIPLIERS = (0.75, 1.0, 1.5)
CHUNK_ROWS = 20_000
DEFAULT_VOLATILITY_COLUMNS = {
    "15m": "realized_vol_15m_pct",
    "1h": "realized_vol_60m_pct",
    "3h": "realized_vol_180m_pct",
}


def _multiplier_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def add_barrier_labels(
    frame: pd.DataFrame,
    horizons: dict[str, int] | None = None,
    multipliers: tuple[float, ...] = DEFAULT_MULTIPLIERS,
    volatility_columns: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str], dict[str, Any]]:
    """Append first-touch labels and diagnostics to a minute-grid feature frame."""
    horizons = horizons or DEFAULT_HORIZONS
    if not multipliers or any(not np.isfinite(x) or x <= 0 for x in multipliers):
        raise ValueError("Barrier multipliers must be finite positive values.")
    volatility_columns = volatility_columns or DEFAULT_VOLATILITY_COLUMNS
    missing_horizons = set(horizons).difference(volatility_columns)
    if missing_horizons:
        raise ValueError(
            "No volatility-column mapping supplied for horizons: "
            f"{sorted(missing_horizons)}"
        )
    required = {"time", "high", "low", "close"}
    required.update(volatility_columns[name] for name in horizons)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Barrier labelling requires columns: {sorted(missing)}")
    out = frame.copy()
    times = pd.to_datetime(out["time"], utc=True, errors="coerce")
    if times.isna().any() or not times.is_monotonic_increasing or times.duplicated().any():
        raise ValueError("Input timestamps must be valid, unique and chronological.")
    if len(times) > 1 and not times.diff().dropna().eq(pd.Timedelta(minutes=1)).all():
        raise ValueError("Input must be a complete 1-minute grid with missing candles left as NaN.")

    high = pd.to_numeric(out["high"], errors="coerce").to_numpy(dtype="float64")
    low = pd.to_numeric(out["low"], errors="coerce").to_numpy(dtype="float64")
    close = pd.to_numeric(out["close"], errors="coerce").to_numpy(dtype="float64")
    volatility_scales = {
        horizon_name: pd.to_numeric(out[column], errors="coerce").to_numpy(dtype="float64")
        for horizon_name, column in volatility_columns.items()
        if horizon_name in horizons
    }
    n_rows = len(out)

    label_columns: list[str] = []
    diagnostic_columns: list[str] = []
    report: dict[str, Any] = {}

    for horizon_name, horizon in horizons.items():
        if horizon <= 0:
            raise ValueError(f"Horizon {horizon_name} must be positive.")
        if n_rows <= horizon:
            windows_high = windows_low = windows_close = None
        else:
            windows_high = sliding_window_view(high, horizon + 1)[:, 1:]
            windows_low = sliding_window_view(low, horizon + 1)[:, 1:]
            windows_close = sliding_window_view(close, horizon + 1)[:, 1:]

        max_starts = max(n_rows - horizon, 0)
        for multiplier in multipliers:
            tag = _multiplier_tag(float(multiplier))
            label_name = f"barrier_label_{horizon_name}_{tag}rv"
            state_name = f"barrier_state_{horizon_name}_{tag}rv"
            bars_name = f"barrier_bars_{horizon_name}_{tag}rv"

            labels = np.full(n_rows, np.nan, dtype="float32")
            # -2 = invalid/incomplete; -1 = down first; 0 = no touch;
            # +1 = up first; +2 = both touched in the same 1m candle.
            states = np.full(n_rows, -2, dtype="int8")
            bars_to_event = np.full(n_rows, np.nan, dtype="float32")

            if max_starts:
                for start in range(0, max_starts, CHUNK_ROWS):
                    end = min(start + CHUNK_ROWS, max_starts)
                    h_win = windows_high[start:end]
                    l_win = windows_low[start:end]
                    c_win = windows_close[start:end]
                    entry = close[start:end]
                    volatility_pct = volatility_scales[horizon_name][start:end]

                    valid = (
                        np.isfinite(entry)
                        & np.isfinite(volatility_pct)
                        & (entry > 0)
                        & (volatility_pct > 0)
                        & np.isfinite(h_win).all(axis=1)
                        & np.isfinite(l_win).all(axis=1)
                        & np.isfinite(c_win).all(axis=1)
                    )
                    if not valid.any():
                        continue

                    upper = entry * (1.0 + multiplier * volatility_pct / 100.0)
                    lower = entry * (1.0 - multiplier * volatility_pct / 100.0)
                    hit_up = h_win >= upper[:, None]
                    hit_down = l_win <= lower[:, None]
                    has_up = hit_up.any(axis=1)
                    has_down = hit_down.any(axis=1)
                    first_up = hit_up.argmax(axis=1)
                    first_down = hit_down.argmax(axis=1)

                    local_labels = np.full(end - start, np.nan, dtype="float32")
                    local_states = np.full(end - start, -2, dtype="int8")
                    local_bars = np.full(end - start, np.nan, dtype="float32")
                    local_labels[valid] = 0.0
                    local_states[valid] = 0

                    up_first = valid & has_up & (~has_down | (first_up < first_down))
                    down_first = valid & has_down & (~has_up | (first_down < first_up))
                    ambiguous = valid & has_up & has_down & (first_up == first_down)

                    local_labels[up_first] = 1.0
                    local_states[up_first] = 1
                    local_bars[up_first] = first_up[up_first] + 1

                    local_labels[down_first] = -1.0
                    local_states[down_first] = -1
                    local_bars[down_first] = first_down[down_first] + 1

                    local_states[ambiguous] = 2
                    local_bars[ambiguous] = first_up[ambiguous] + 1

                    labels[start:end] = local_labels
                    states[start:end] = local_states
                    bars_to_event[start:end] = local_bars

            out[label_name] = labels
            out[state_name] = states
            out[bars_name] = bars_to_event
            label_columns.append(label_name)
            diagnostic_columns.extend([state_name, bars_name])

            state_counts = {
                "up_first": int((states == 1).sum()),
                "down_first": int((states == -1).sum()),
                "no_touch": int((states == 0).sum()),
                "ambiguous_same_minute": int((states == 2).sum()),
                "invalid_or_incomplete": int((states == -2).sum()),
            }
            valid_count = sum(state_counts[key] for key in (
                "up_first", "down_first", "no_touch", "ambiguous_same_minute"
            ))
            report[label_name] = {
                "horizon_minutes": horizon,
                "symmetric_barrier_multiple_of_horizon_aligned_realized_volatility": float(multiplier),
                "volatility_column": volatility_columns[horizon_name],
                "label_mapping": {"-1": "down barrier first", "0": "no touch or ambiguous",
                                  "+1": "up barrier first", "null": "incomplete/invalid"},
                "state_counts": state_counts,
                "valid_rows": valid_count,
                "up_first_share_of_valid": (
                    state_counts["up_first"] / valid_count if valid_count else None
                ),
                "down_first_share_of_valid": (
                    state_counts["down_first"] / valid_count if valid_count else None
                ),
                "ambiguous_share_of_valid": (
                    state_counts["ambiguous_same_minute"] / valid_count if valid_count else None
                ),
            }

    return out, label_columns, diagnostic_columns, report



def add_asymmetric_directional_labels(
    frame: pd.DataFrame,
    horizons: dict[str, int] | None = None,
    stop_multipliers: tuple[float, ...] = (1.0,),
    target_multipliers: tuple[float, ...] = (3.0,),
    volatility_columns: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str], dict[str, Any]]:
    """Add separate long/short first-touch labels with asymmetric 1R stop / >=3R target.

    For a long label, the upper target is +target_mult * RV and the stop is
    -stop_mult * RV. For a short label, the lower target is -target_mult * RV
    and the stop is +stop_mult * RV. Each direction is evaluated independently.
    Labels: +1 target first, -1 stop first, 0 neither touched, NaN invalid.
    State 2 means target and stop were both touched in the same 1m bar; -2 is
    invalid/incomplete. Ambiguous bars are never assigned an invented order.
    """
    horizons = horizons or DEFAULT_HORIZONS
    volatility_columns = volatility_columns or DEFAULT_VOLATILITY_COLUMNS
    if not stop_multipliers or not target_multipliers:
        raise ValueError("Stop and target multipliers must be non-empty.")
    if any(not np.isfinite(x) or x <= 0 for x in (*stop_multipliers, *target_multipliers)):
        raise ValueError("Stop and target multipliers must be finite and positive.")
    missing_horizons = set(horizons).difference(volatility_columns)
    if missing_horizons:
        raise ValueError(f"No volatility mapping for horizons: {sorted(missing_horizons)}")
    required = {"time", "high", "low", "close"}
    required.update(volatility_columns[name] for name in horizons)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Asymmetric labelling requires columns: {sorted(missing)}")

    out = frame.copy()
    times = pd.to_datetime(out["time"], utc=True, errors="coerce")
    if times.isna().any() or not times.is_monotonic_increasing or times.duplicated().any():
        raise ValueError("Input timestamps must be valid, unique and chronological.")
    if len(times) > 1 and not times.diff().dropna().eq(pd.Timedelta(minutes=1)).all():
        raise ValueError("Input must be a complete 1-minute grid with missing candles left as NaN.")

    high = pd.to_numeric(out["high"], errors="coerce").to_numpy(dtype="float64")
    low = pd.to_numeric(out["low"], errors="coerce").to_numpy(dtype="float64")
    close = pd.to_numeric(out["close"], errors="coerce").to_numpy(dtype="float64")
    vol = {
        name: pd.to_numeric(out[column], errors="coerce").to_numpy(dtype="float64")
        for name, column in volatility_columns.items() if name in horizons
    }
    n_rows = len(out)
    label_columns: list[str] = []
    diagnostic_columns: list[str] = []
    report: dict[str, Any] = {}

    for horizon_name, horizon in horizons.items():
        if horizon <= 0:
            raise ValueError(f"Horizon {horizon_name} must be positive.")
        if n_rows <= horizon:
            windows_high = windows_low = None
        else:
            windows_high = sliding_window_view(high, horizon + 1)[:, 1:]
            windows_low = sliding_window_view(low, horizon + 1)[:, 1:]
        max_starts = max(n_rows - horizon, 0)

        for stop_mult in stop_multipliers:
            for target_mult in target_multipliers:
                tag = f"sl{stop_mult:g}_tp{target_mult:g}".replace(".", "p")
                for direction in ("long", "short"):
                    label_name = f"asym_{direction}_label_{horizon_name}_{tag}rv"
                    state_name = f"asym_{direction}_state_{horizon_name}_{tag}rv"
                    bars_name = f"asym_{direction}_bars_{horizon_name}_{tag}rv"
                    labels = np.full(n_rows, np.nan, dtype="float32")
                    states = np.full(n_rows, -2, dtype="int8")
                    bars_to_event = np.full(n_rows, np.nan, dtype="float32")

                    if max_starts:
                        for start in range(0, max_starts, CHUNK_ROWS):
                            end = min(start + CHUNK_ROWS, max_starts)
                            h_win = windows_high[start:end]
                            l_win = windows_low[start:end]
                            entry = close[start:end]
                            scale = vol[horizon_name][start:end]
                            valid = (
                                np.isfinite(entry) & (entry > 0)
                                & np.isfinite(scale) & (scale > 0)
                                & np.isfinite(h_win).all(axis=1)
                                & np.isfinite(l_win).all(axis=1)
                            )
                            local_labels = np.full(end - start, np.nan, dtype="float32")
                            local_states = np.full(end - start, -2, dtype="int8")
                            local_bars = np.full(end - start, np.nan, dtype="float32")
                            local_labels[valid] = 0.0
                            local_states[valid] = 0

                            if direction == "long":
                                target_price = entry * (1.0 + target_mult * scale / 100.0)
                                stop_price = entry * (1.0 - stop_mult * scale / 100.0)
                                target_hits = h_win >= target_price[:, None]
                                stop_hits = l_win <= stop_price[:, None]
                            else:
                                target_price = entry * (1.0 - target_mult * scale / 100.0)
                                stop_price = entry * (1.0 + stop_mult * scale / 100.0)
                                target_hits = l_win <= target_price[:, None]
                                stop_hits = h_win >= stop_price[:, None]

                            has_target = target_hits.any(axis=1)
                            has_stop = stop_hits.any(axis=1)
                            first_target = target_hits.argmax(axis=1)
                            first_stop = stop_hits.argmax(axis=1)
                            target_first = valid & has_target & (~has_stop | (first_target < first_stop))
                            stop_first = valid & has_stop & (~has_target | (first_stop < first_target))
                            ambiguous = valid & has_target & has_stop & (first_target == first_stop)

                            local_labels[target_first] = 1.0
                            local_states[target_first] = 1
                            local_bars[target_first] = first_target[target_first] + 1
                            local_labels[stop_first] = -1.0
                            local_states[stop_first] = -1
                            local_bars[stop_first] = first_stop[stop_first] + 1
                            local_states[ambiguous] = 2
                            local_bars[ambiguous] = first_target[ambiguous] + 1

                            labels[start:end] = local_labels
                            states[start:end] = local_states
                            bars_to_event[start:end] = local_bars

                    out[label_name] = labels
                    out[state_name] = states
                    out[bars_name] = bars_to_event
                    label_columns.append(label_name)
                    diagnostic_columns.extend([state_name, bars_name])
                    counts = {
                        "target_first": int((states == 1).sum()),
                        "stop_first": int((states == -1).sum()),
                        "no_touch": int((states == 0).sum()),
                        "ambiguous_same_minute": int((states == 2).sum()),
                        "invalid_or_incomplete": int((states == -2).sum()),
                    }
                    valid_count = sum(counts[key] for key in (
                        "target_first", "stop_first", "no_touch", "ambiguous_same_minute"
                    ))
                    report[label_name] = {
                        "direction": direction,
                        "horizon_minutes": horizon,
                        "stop_multiple": float(stop_mult),
                        "target_multiple": float(target_mult),
                        "reward_risk_ratio_before_costs": float(target_mult / stop_mult),
                        "volatility_column": volatility_columns[horizon_name],
                        "label_mapping": {"+1": "target first", "-1": "stop first",
                                          "0": "neither touched", "null": "invalid/incomplete"},
                        "state_counts": counts,
                        "valid_rows": valid_count,
                    }

    return out, label_columns, diagnostic_columns, report
