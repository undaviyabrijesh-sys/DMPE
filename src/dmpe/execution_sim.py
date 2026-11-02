"""Execution-aware simulator for DMPE walk-forward test signals.

Version 1 models market entry/exit, BTCUSD contract lot size, taker fees incl.
GST, and configurable per-side slippage. Funding is intentionally excluded
until settlement timestamps/rates can be verified against the venue's actual
funding realization schedule. No orders are submitted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

HORIZONS = {"15m": 15, "1h": 60, "3h": 180}
VOL_COLUMNS = {
    "15m": "realized_vol_15m_pct",
    "1h": "realized_vol_60m_pct",
    "3h": "realized_vol_180m_pct",
}


@dataclass(frozen=True)
class ExecutionAssumptions:
    taker_fee_bps_per_side_including_gst: float = 5.9
    slippage_bps_per_side: float = 2.0
    lot_size_btc: float = 0.001
    contracts: float = 1.0


def _normalize_prices(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"time", "open", "high", "low", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing candle columns: {sorted(missing)}")
    out = frame[["time", "open", "high", "low", "close"]].copy()
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    for col in ("open", "high", "low", "close"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["time"]).sort_values("time").drop_duplicates("time")
    return out.set_index("time").sort_index()




@dataclass(frozen=True)
class PreparedExecutionData:
    """Large static market arrays built once and reused across variants/cost scenarios."""
    price_positions: dict[pd.Timestamp, int]
    times_ns: np.ndarray
    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    feature_positions: dict[pd.Timestamp, int]
    feature_volatility: dict[str, np.ndarray]
    feature_close: np.ndarray


def prepare_execution_data(
    price_frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
) -> PreparedExecutionData:
    """Normalize and index million-row source data once per simulation run."""
    prices = _normalize_prices(price_frame)
    features = feature_frame.copy()
    features["time"] = pd.to_datetime(features["time"], utc=True, errors="coerce")
    features = features.dropna(subset=["time"]).drop_duplicates("time").set_index("time").sort_index()
    price_index = prices.index
    volatility_arrays = {
        column: features[column].to_numpy(dtype="float64")
        for column in VOL_COLUMNS.values()
        if column in features.columns
    }
    return PreparedExecutionData(
        price_positions={timestamp: idx for idx, timestamp in enumerate(price_index)},
        feature_positions={timestamp: idx for idx, timestamp in enumerate(features.index)},
        times_ns=price_index.asi8,
        opens=prices["open"].to_numpy(dtype="float64"),
        highs=prices["high"].to_numpy(dtype="float64"),
        lows=prices["low"].to_numpy(dtype="float64"),
        closes=prices["close"].to_numpy(dtype="float64"),
        feature_volatility=volatility_arrays,
        feature_close=pd.to_numeric(features["close"], errors="coerce").to_numpy(dtype="float64"),
    )


def max_drawdown(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    equity = values.cumsum()
    drawdown = equity - equity.cummax().clip(lower=0.0)
    return float(drawdown.min())


def summarize_trades(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {
            "trades": 0,
            "net_pnl_usd_per_configured_size": 0.0,
            "gross_pnl_usd": 0.0,
            "fees_usd": 0.0,
            "win_rate": None,
            "profit_factor": None,
            "expectancy_usd_per_trade": None,
            "mean_net_return_pct_on_entry_notional": None,
            "median_net_return_pct_on_entry_notional": None,
            "max_drawdown_usd": 0.0,
        }
    pnl = trades["net_pnl_usd"].astype(float)
    winners = pnl[pnl > 0]
    losers = pnl[pnl < 0]
    profit_factor = (
        float(winners.sum() / abs(losers.sum()))
        if len(losers) and losers.sum() != 0 else None
    )
    return {
        "trades": int(len(trades)),
        "long_trades": int((trades["signal"] == 1).sum()),
        "short_trades": int((trades["signal"] == -1).sum()),
        "net_pnl_usd_per_configured_size": float(pnl.sum()),
        "gross_pnl_usd": float(trades["gross_pnl_usd"].sum()),
        "fees_usd": float(trades["fees_usd"].sum()),
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": profit_factor,
        "expectancy_usd_per_trade": float(pnl.mean()),
        "mean_net_return_pct_on_entry_notional": float(trades["net_return_pct"].mean()),
        "median_net_return_pct_on_entry_notional": float(trades["net_return_pct"].median()),
        "max_drawdown_usd": max_drawdown(pnl.reset_index(drop=True)),
        "average_holding_minutes": float(trades["holding_minutes"].mean()),
        "exit_reasons": {
            str(key): int(value) for key, value in trades["exit_reason"].value_counts().items()
        },
    }


def simulate_signal_set(
    signals: pd.DataFrame,
    price_frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
    assumptions: ExecutionAssumptions,
    _prepared: PreparedExecutionData | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Simulate one horizon/barrier signal set with fixed size; use NumPy OHLC arrays."""
    if assumptions.taker_fee_bps_per_side_including_gst < 0:
        raise ValueError("Taker fee bps must be non-negative.")
    if assumptions.slippage_bps_per_side < 0:
        raise ValueError("Slippage bps must be non-negative.")
    if assumptions.lot_size_btc <= 0 or assumptions.contracts <= 0:
        raise ValueError("Contract lot size and contract count must be positive.")
    if signals.empty:
        empty = pd.DataFrame()
        return empty, {"input_signals": 0, "skipped": {}, **summarize_trades(empty)}

    prepared = _prepared or prepare_execution_data(price_frame, feature_frame)

    horizon_name = str(signals["horizon"].iloc[0])
    if horizon_name not in HORIZONS or not signals["horizon"].eq(horizon_name).all():
        raise ValueError("Pass exactly one supported horizon per simulation call.")
    horizon_minutes = HORIZONS[horizon_name]
    if signals["barrier_multiple"].nunique() != 1:
        raise ValueError("Pass exactly one barrier multiplier per simulation call.")
    multiplier = float(signals["barrier_multiple"].iloc[0])
    volatility_column = VOL_COLUMNS[horizon_name]
    # Feature rows are timestamped at the previous candle close; this same time
    # is the next 1-minute candle's OPEN timestamp in Delta's raw candle history.
    fee_rate = assumptions.taker_fee_bps_per_side_including_gst / 10_000.0
    slippage = assumptions.slippage_bps_per_side / 10_000.0
    btc_quantity = assumptions.lot_size_btc * assumptions.contracts

    try:
        feature_volatility = prepared.feature_volatility[volatility_column]
    except KeyError as exc:
        raise ValueError(f"Feature dataset is missing {volatility_column}.") from exc
    price_positions = prepared.price_positions
    feature_positions = prepared.feature_positions
    times_ns = prepared.times_ns
    feature_close = prepared.feature_close
    opens, highs = prepared.opens, prepared.highs
    lows, closes = prepared.lows, prepared.closes

    rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    expected_step_ns = 60_000_000_000

    for signal_row in signals.sort_values("time").to_dict(orient="records"):
        signal_time = pd.to_datetime(signal_row["time"], utc=True)
        direction = int(signal_row["signal"])
        if direction not in (-1, 1):
            skipped["non_directional_signal"] = skipped.get("non_directional_signal", 0) + 1
            continue
        entry_position = price_positions.get(signal_time)
        if entry_position is None:
            skipped["missing_entry_candle"] = skipped.get("missing_entry_candle", 0) + 1
            continue
        feature_position = feature_positions.get(signal_time)
        if feature_position is None:
            skipped["missing_point_in_time_features"] = skipped.get("missing_point_in_time_features", 0) + 1
            continue
        vol_pct = feature_volatility[feature_position]
        if pd.isna(vol_pct) or not np.isfinite(float(vol_pct)) or float(vol_pct) <= 0:
            skipped["invalid_volatility_scale"] = skipped.get("invalid_volatility_scale", 0) + 1
            continue

        end_position = entry_position + horizon_minutes
        if end_position > len(times_ns):
            skipped["incomplete_future_price_window"] = skipped.get("incomplete_future_price_window", 0) + 1
            continue
        window_times_ns = times_ns[entry_position:end_position]
        if (
            len(window_times_ns) != horizon_minutes
            or (horizon_minutes > 1 and not np.all(np.diff(window_times_ns) == expected_step_ns))
        ):
            skipped["incomplete_future_price_window"] = skipped.get("incomplete_future_price_window", 0) + 1
            continue

        open_window = opens[entry_position:end_position]
        high_window = highs[entry_position:end_position]
        low_window = lows[entry_position:end_position]
        close_window = closes[entry_position:end_position]
        if not (
            np.isfinite(open_window).all()
            and np.isfinite(high_window).all()
            and np.isfinite(low_window).all()
            and np.isfinite(close_window).all()
        ):
            skipped["invalid_future_ohlc"] = skipped.get("invalid_future_ohlc", 0) + 1
            continue

        raw_entry = float(open_window[0])
        entry_fill = raw_entry * (1.0 + direction * slippage)
        reference_price = float(feature_close[feature_position])
        if not np.isfinite(reference_price) or reference_price <= 0:
            skipped["invalid_signal_close_reference"] = skipped.get("invalid_signal_close_reference", 0) + 1
            continue
        # Labels are anchored to the completed signal candle close. Keep the
        # execution barriers anchored to that same point-in-time reference;
        # entry gaps/slippage affect realized P&L, not the label geometry.
        distance_pct = multiplier * float(vol_pct)
        if distance_pct <= 0 or distance_pct >= 100:
            skipped["invalid_barrier_distance"] = skipped.get("invalid_barrier_distance", 0) + 1
            continue

        target = reference_price * (1.0 + direction * distance_pct / 100.0)
        stop = reference_price * (1.0 - direction * distance_pct / 100.0)
        if direction == 1:
            stop_hits = np.flatnonzero(low_window <= stop)
            target_hits = np.flatnonzero(high_window >= target)
        else:
            stop_hits = np.flatnonzero(high_window >= stop)
            target_hits = np.flatnonzero(low_window <= target)

        first_stop = int(stop_hits[0]) if len(stop_hits) else None
        first_target = int(target_hits[0]) if len(target_hits) else None
        raw_exit: float
        holding_minutes: int
        if first_stop is not None and (first_target is None or first_stop <= first_target):
            stop_bar_open = float(open_window[first_stop])
            raw_exit = min(stop_bar_open, stop) if direction == 1 else max(stop_bar_open, stop)
            exit_reason = (
                "stop_and_target_same_bar_stop_first"
                if first_target == first_stop else "stop"
            )
            holding_minutes = first_stop + 1
        elif first_target is not None:
            raw_exit = target
            exit_reason = "target"
            holding_minutes = first_target + 1
        else:
            raw_exit = float(close_window[-1])
            exit_reason = "time_exit"
            holding_minutes = horizon_minutes

        exit_fill = raw_exit * (1.0 - direction * slippage)
        entry_notional = btc_quantity * entry_fill
        exit_notional = btc_quantity * exit_fill
        gross_pnl = btc_quantity * direction * (exit_fill - entry_fill)
        entry_fee = entry_notional * fee_rate
        exit_fee = exit_notional * fee_rate
        fees = entry_fee + exit_fee
        net_pnl = gross_pnl - fees
        net_return_pct = net_pnl / entry_notional * 100.0 if entry_notional else np.nan

        rows.append({
            "time": signal_time,
            "fold": signal_row.get("fold"),
            "horizon": horizon_name,
            "barrier_multiple": multiplier,
            "signal": direction,
            "p_down": signal_row.get("p_down"),
            "p_flat": signal_row.get("p_flat"),
            "p_up": signal_row.get("p_up"),
            "validation_threshold": signal_row.get("validation_threshold"),
            "volatility_scale_pct": float(vol_pct),
            "barrier_distance_pct": distance_pct,
            "signal_close_reference_price": reference_price,
            "raw_entry_price": raw_entry,
            "entry_fill_price": entry_fill,
            "target_price": target,
            "stop_price": stop,
            "raw_exit_price": raw_exit,
            "exit_fill_price": exit_fill,
            "exit_reason": exit_reason,
            "holding_minutes": holding_minutes,
            "entry_notional_usd": entry_notional,
            "gross_pnl_usd": gross_pnl,
            "entry_fee_usd": entry_fee,
            "exit_fee_usd": exit_fee,
            "fees_usd": fees,
            "net_pnl_usd": net_pnl,
            "gross_return_pct": gross_pnl / entry_notional * 100.0 if entry_notional else np.nan,
            "net_return_pct": net_return_pct,
        })

    trades = pd.DataFrame(rows)
    summary = {
        "input_signals": int(len(signals)),
        "executed_trades": int(len(trades)),
        "skipped": skipped,
        **summarize_trades(trades),
    }
    return trades, summary

def run_slippage_sensitivity(
    signals_frame: pd.DataFrame,
    price_frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
    fee_bps_per_side_including_gst: float = 5.9,
    slippage_scenarios_bps: tuple[float, ...] = (0.0, 1.0, 2.0, 5.0, 10.0),
    lot_size_btc: float = 0.001,
    contracts: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Evaluate every model/horizon/barrier separately across slippage scenarios."""
    summaries = []
    all_trades = []
    # Reuse the large 1.5M-row timestamp maps and OHLC arrays across all scenarios.
    prepared = prepare_execution_data(price_frame, feature_frame)
    variants = signals_frame.groupby(["horizon", "barrier_multiple"], sort=True)
    for (horizon, multiplier), variant_signals in variants:
        for slip in slippage_scenarios_bps:
            assumptions = ExecutionAssumptions(
                taker_fee_bps_per_side_including_gst=fee_bps_per_side_including_gst,
                slippage_bps_per_side=float(slip),
                lot_size_btc=lot_size_btc,
                contracts=contracts,
            )
            trades, summary = simulate_signal_set(
                variant_signals, price_frame, feature_frame, assumptions, _prepared=prepared
            )
            summary.update({
                "horizon": horizon,
                "barrier_multiple": float(multiplier),
                "taker_fee_bps_per_side_including_gst": fee_bps_per_side_including_gst,
                "slippage_bps_per_side": float(slip),
                "funding_included": False,
            })
            summaries.append(summary)
            if not trades.empty:
                trades["slippage_bps_per_side"] = float(slip)
                all_trades.append(trades)

    return (
        pd.DataFrame(summaries),
        pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame(),
        {
            "fee_basis": "Taker 0.05% plus 18% GST on trading fees = 5.9 bps per side",
            "slippage_basis": "User-configurable scenario assumption per side; not historically measured.",
            "funding_basis": (
                "Excluded in this first simulator version until actual historical funding "
                "realization timestamps and the funding-candle meaning are independently verified."
            ),
            "pnl_basis": (
                "USD-equivalent, fixed configured BTC lot size, no compounding, one position "
                "per signal, no leverage or liquidation simulation."
            ),
            "entry_policy": "Next 1-minute candle open after signal feature timestamp, with adverse slippage.",
            "same_bar_policy": "If stop and target both touched in one 1-minute candle, assume stop first.",
            "gap_policy": "Skip trades with missing 1-minute candles in the holding window; stop-market gaps fill at the worse bar open.",
        },
    )
