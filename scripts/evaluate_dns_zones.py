"""Cost-aware causal backtest for the DMPE base/leg + swing DNS detector.

Uses only zones confirmed by the current completed 15m candle. Entries occur
at the next candle open after a later retest candle is confirmed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dmpe_zones import detect_structure_zones


def load_15m(path: Path, days: int) -> pd.DataFrame:
    one = pd.read_parquet(path, columns=["time", "open", "high", "low", "close", "volume"])
    one["time"] = pd.to_datetime(one["time"], utc=True)
    one = one.sort_values("time").drop_duplicates("time").set_index("time")
    bars = one.resample("15min", label="right", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"), count=("close", "count"),
    )
    bars = bars[bars["count"] == 15].drop(columns="count").dropna().reset_index()
    if not bars.empty:
        cutoff = bars["time"].max() - pd.Timedelta(days=days)
        bars = bars[bars["time"] >= cutoff].reset_index(drop=True)
    if len(bars) < 500:
        raise ValueError(f"Only {len(bars)} complete 15m bars; need at least 500.")
    prev = bars["close"].shift(1)
    tr = pd.concat([
        bars["high"] - bars["low"],
        (bars["high"] - prev).abs(),
        (bars["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    bars["atr"] = tr.rolling(14, min_periods=14).mean()
    return bars


def run_one(bars: pd.DataFrame, rr: float, fee_bps: float, slippage_bps: float,
            window: int, max_hold: int) -> pd.DataFrame:
    active: list[dict] = []
    seen: set[tuple] = set()
    trades: list[dict] = []
    position = None
    pending = None

    for i in range(max(60, window), len(bars) - 1):
        row = bars.iloc[i]
        stamp = row["time"]
        local = bars.iloc[max(0, i - window + 1):i + 1].copy()
        zones = detect_structure_zones(local)
        for z in zones:
            if z["created_at"] != stamp:
                continue
            key = (z["kind"], z.get("origin_type"), z.get("origin_time"),
                   round(float(z["low"]), 8), round(float(z["high"]), 8))
            if key in seen:
                continue
            seen.add(key)
            active.append({
                "zone": z, "touched": False, "touch_idx": None,
                "touch_high": None, "touch_low": None, "invalid": False,
            })

        # A confirmed setup enters at the next completed-bar open.
        if pending is not None and position is None:
            entry = float(row["open"])
            stop = float(pending["stop"])
            risk = entry - stop if pending["side"] == "LONG" else stop - entry
            if risk > 0:
                target = entry + rr * risk if pending["side"] == "LONG" else entry - rr * risk
                position = {**pending, "entry": entry, "risk": risk, "target": target,
                            "entry_idx": i, "entry_time": stamp}
            pending = None

        if position is not None:
            side = position["side"]
            hit_sl = float(row["low"]) <= position["stop"] if side == "LONG" else float(row["high"]) >= position["stop"]
            hit_tp = float(row["high"]) >= position["target"] if side == "LONG" else float(row["low"]) <= position["target"]
            timed_out = i - position["entry_idx"] >= max_hold
            if hit_sl or hit_tp or timed_out:
                if hit_sl:  # conservative: stop first if both barriers hit
                    exit_price = min(float(row["open"]), position["stop"]) if side == "LONG" else max(float(row["open"]), position["stop"])
                    reason = "stop"
                elif hit_tp:
                    exit_price, reason = position["target"], "target"
                else:
                    exit_price, reason = float(row["close"]), "time"
                direction = 1 if side == "LONG" else -1
                gross_r = direction * (exit_price - position["entry"]) / position["risk"]
                cost_r = ((position["entry"] + exit_price) * (fee_bps + slippage_bps) / 10000.0) / position["risk"]
                trades.append({
                    "side": side, "origin_type": position["origin_type"],
                    "zone_low": position["zone_low"], "zone_high": position["zone_high"],
                    "zone_strength_atr": position["strength"],
                    "origin_time": position["origin_time"],
                    "confirmed_at": position["confirmed_at"],
                    "touch_time": position["touch_time"],
                    "entry_time": position["entry_time"], "exit_time": stamp,
                    "entry": position["entry"], "stop": position["stop"],
                    "target": position["target"], "exit": exit_price,
                    "exit_reason": reason, "gross_r": gross_r,
                    "cost_r": cost_r, "net_r": gross_r - cost_r,
                    "bars_held": i - position["entry_idx"],
                })
                position = None

        if position is not None or pending is not None:
            continue

        for state in active:
            if state["invalid"] or state["touched"]:
                continue
            z = state["zone"]
            low, high = float(z["low"]), float(z["high"])
            side = "LONG" if z["kind"] in ("DEMAND", "RBS") else "SHORT"
            close = float(row["close"])
            if (side == "LONG" and close < low) or (side == "SHORT" and close > high):
                state["invalid"] = True
                continue

            intersects = float(row["low"]) <= high and float(row["high"]) >= low
            if state["touch_idx"] is None:
                if intersects and z["created_at"] != stamp:
                    state["touch_idx"] = i
                    state["touch_high"] = float(row["high"])
                    state["touch_low"] = float(row["low"])
                    state["touch_time"] = stamp
                continue

            if i <= state["touch_idx"]:
                continue
            confirmed = close > state["touch_high"] if side == "LONG" else close < state["touch_low"]
            if confirmed:
                atr = float(row["atr"]) if pd.notna(row["atr"]) else float(row["high"] - row["low"])
                stop = state["touch_low"] - 0.1 * atr if side == "LONG" else state["touch_high"] + 0.1 * atr
                pending = {
                    "side": side, "stop": stop, "origin_type": z.get("origin_type", "BASE_LEG"),
                    "zone_low": low, "zone_high": high, "strength": float(z["strength"]),
                    "origin_time": z.get("origin_time", z["created_at"]),
                    "confirmed_at": stamp, "touch_time": state["touch_time"],
                }
                state["touched"] = True
                break
            # A close through the far boundary invalidates before confirmation.
            if (side == "LONG" and close < low) or (side == "SHORT" and close > high):
                state["invalid"] = True

    return pd.DataFrame(trades)


def summarize(trades: pd.DataFrame, rr: float, fee_bps: float, slippage_bps: float) -> dict:
    if trades.empty:
        return {"rr": rr, "trades": 0, "win_rate_pct": None, "expectancy_r": None,
                "profit_factor": None, "net_total_r": 0.0, "max_drawdown_r": 0.0}
    net = trades["net_r"].astype(float)
    wins, losses = net[net > 0], net[net <= 0]
    equity = net.cumsum()
    dd = equity - equity.cummax()
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    return {
        "rr": rr, "trades": int(len(trades)),
        "win_rate_pct": round(float((net > 0).mean() * 100), 2),
        "expectancy_r": round(float(net.mean()), 4),
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss else None,
        "net_total_r": round(float(net.sum()), 2),
        "max_drawdown_r": round(float(dd.min()), 2),
        "avg_win_r": round(float(wins.mean()), 3) if len(wins) else 0.0,
        "avg_loss_r": round(float(losses.mean()), 3) if len(losses) else 0.0,
        "stop_rate_pct": round(float((trades["exit_reason"] == "stop").mean() * 100), 2),
        "target_rate_pct": round(float((trades["exit_reason"] == "target").mean() * 100), 2),
        "fee_bps_per_side": fee_bps, "slippage_bps_per_side": slippage_bps,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--price-file", required=True)
    p.add_argument("--output-dir", default="dmpe_dns_backtest")
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--rr", default="1,2,3")
    p.add_argument("--fee-bps-per-side", type=float, default=5.9)
    p.add_argument("--slippage-bps-per-side", type=float, default=1.0)
    p.add_argument("--window", type=int, default=240)
    p.add_argument("--max-hold-bars", type=int, default=96)
    args = p.parse_args()

    bars = load_15m(Path(args.price_file), args.days)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_trades, summaries = [], []
    for rr in [float(x) for x in args.rr.split(",") if x.strip()]:
        trades = run_one(bars, rr, args.fee_bps_per_side, args.slippage_bps_per_side,
                         args.window, args.max_hold_bars)
        trades.to_csv(out / f"trades_{rr:g}R.csv", index=False)
        if not trades.empty:
            trades["rr"] = rr
            all_trades.append(trades)
        summaries.append(summarize(trades, rr, args.fee_bps_per_side, args.slippage_bps_per_side))

    pd.concat(all_trades, ignore_index=True).to_csv(out / "all_trades.csv", index=False) if all_trades else pd.DataFrame().to_csv(out / "all_trades.csv", index=False)
    report = {
        "purpose": "Exploratory backtest of DMPE structure DNS detector; not approved for trading.",
        "data_file": str(args.price_file), "timeframe": "15m",
        "bars": int(len(bars)), "start_utc": bars["time"].min().isoformat(),
        "end_utc": bars["time"].max().isoformat(),
        "fees_and_slippage": "5.9 bps fee + 1.0 bps slippage per side",
        "funding": "not included",
        "execution": "next 15m open after completed retest confirmation; one position at a time; stop first if SL and TP hit in same candle; stop gap filled at worse open",
        "zone_availability": "base/leg zones only after displacement confirmation; swing zones only after three right-side candles close",
        "max_holding_bars": args.max_hold_bars,
        "results": summaries,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
