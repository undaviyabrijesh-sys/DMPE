"""Post-hoc diagnostic ablation of DMPE MTF direction gates.

This does not tune entries. It replays the same frozen 5m entry patterns under
three context modes: original 4h+1h+15m gate, 15m-only gate, and no directional
context gate. This is exploratory because the first gated results were already
observed; any promising result must be retested on a new untouched period.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dmpe.structure_engine import build_mtf_signal_frame  # noqa: E402
from dmpe.structure_backtest import replay_structure_candidates  # noqa: E402


def _raw_module_signals(c: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series]]:
    """Return (long, short) raw setup triggers before 4h/1h/15m directional gates."""
    touched_long = (c["low"] <= c["ema20"]) & (c["close"] >= c["ema20"])
    touched_short = (c["high"] >= c["ema20"]) & (c["close"] <= c["ema20"])

    pullback_long = touched_long & (c["close"] > c["high"].shift(1))
    pullback_short = touched_short & (c["close"] < c["low"].shift(1))

    sweep_long = (c["low"] < c["prior_low"]) & (c["close"] > c["prior_low"])
    sweep_short = (c["high"] > c["prior_high"]) & (c["close"] < c["prior_high"])

    broke_up = c["close"].shift(1) > c["prior_high"].shift(1)
    broke_down = c["close"].shift(1) < c["prior_low"].shift(1)
    recent_up = broke_up.rolling(5, min_periods=1).max().fillna(False).astype(bool)
    recent_down = broke_down.rolling(5, min_periods=1).max().fillna(False).astype(bool)
    breakout_long = recent_up & (c["low"] <= c["prior_high"]) & (c["close"] > c["prior_high"])
    breakout_short = recent_down & (c["high"] >= c["prior_low"]) & (c["close"] < c["prior_low"])

    return {
        "pullback": (pullback_long.fillna(False), pullback_short.fillna(False)),
        "sweep": (sweep_long.fillna(False), sweep_short.fillna(False)),
        "breakout_retest": (breakout_long.fillna(False), breakout_short.fillna(False)),
    }


def _direction(long_flag: pd.Series, short_flag: pd.Series) -> pd.Series:
    """Ambiguous same-bar long+short triggers are dropped, never tie-broken."""
    long_flag = long_flag.fillna(False).astype(bool)
    short_flag = short_flag.fillna(False).astype(bool)
    values = (long_flag & ~short_flag).astype("int8") - (short_flag & ~long_flag).astype("int8")
    return values.astype("int8")


def build_gate_variants(c: pd.DataFrame) -> tuple[dict[str, pd.Series], dict[str, int]]:
    """Build frozen setup signals under gated, 15m-only and ungated contexts."""
    raw = _raw_module_signals(c)
    variants: dict[str, pd.Series] = {}
    counts: dict[str, int] = {}

    for module, (raw_long, raw_short) in raw.items():
        gated = c[f"entry_{module}"].fillna(0).astype("int8")
        fifteen_long = raw_long & (c["15m_structure_bias"] == 1)
        fifteen_short = raw_short & (c["15m_structure_bias"] == -1)
        variants[f"{module}__4h1h15m"] = gated
        variants[f"{module}__15m_only"] = _direction(fifteen_long, fifteen_short)
        variants[f"{module}__ungated"] = _direction(raw_long, raw_short)

    # Collision-safe union: if different modules point both ways on one bar,
    # that timestamp is FLAT rather than silently giving either side priority.
    for mode in ("4h1h15m", "15m_only", "ungated"):
        cols = [variants[f"{m}__{mode}"] for m in ("pullback", "sweep", "breakout_retest")]
        long_any = pd.concat([(s == 1) for s in cols], axis=1).any(axis=1)
        short_any = pd.concat([(s == -1) for s in cols], axis=1).any(axis=1)
        variants[f"combined_consensus__{mode}"] = _direction(long_any, short_any)

    for name, signal in variants.items():
        counts[name] = int((signal != 0).sum())
    return variants, counts


def _profit_factor(pnl: pd.Series) -> float | None:
    gains = float(pnl[pnl > 0].sum())
    losses = float(-pnl[pnl < 0].sum())
    if losses == 0:
        return None if gains == 0 else float("inf")
    return gains / losses


def _score(trades: pd.DataFrame, candidate_count: int, fee: float, slip: float) -> dict:
    if trades.empty:
        return {
            "candidate_signals": candidate_count, "trades": 0, "trades_per_day": 0.0,
            "win_rate_pct": None, "gross_pnl_points": 0.0, "net_pnl_points": 0.0,
            "gross_expectancy_R": None, "net_expectancy_R": None,
            "gross_profit_factor": None, "net_profit_factor": None,
            "avg_cost_points": None, "median_cost_points": None,
            "max_drawdown_R": None, "fee_bps_per_side": fee,
            "slippage_bps_per_side": slip,
        }

    gross = trades["gross_pnl_points"].astype(float)
    net = trades["net_pnl_points"].astype(float)
    equity = trades["net_r"].astype(float).cumsum()
    drawdown = equity - equity.cummax()
    start = pd.to_datetime(trades["entry_time"], utc=True).min()
    end = pd.to_datetime(trades["exit_time"], utc=True).max()
    days = max((end - start).total_seconds() / 86400.0, 1 / 1440)
    return {
        "candidate_signals": candidate_count,
        "trades": int(len(trades)),
        "trades_per_day": float(len(trades) / days),
        "win_rate_pct": float((net > 0).mean() * 100),
        "gross_pnl_points": float(gross.sum()),
        "net_pnl_points": float(net.sum()),
        "gross_expectancy_R": float((gross / trades["risk_points"]).mean()),
        "net_expectancy_R": float(trades["net_r"].mean()),
        "gross_profit_factor": _profit_factor(gross),
        "net_profit_factor": _profit_factor(net),
        "avg_cost_points": float(trades["cost_points"].mean()),
        "median_cost_points": float(trades["cost_points"].median()),
        "max_drawdown_R": float(drawdown.min()),
        "fee_bps_per_side": fee,
        "slippage_bps_per_side": slip,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-file", required=True, help="Delta BTCUSD 1m candle Parquet")
    parser.add_argument("--output-dir", default="mtf_gate_ablation_output")
    parser.add_argument("--rr", default="1,2,3")
    parser.add_argument("--fee-bps-per-side", type=float, default=5.9)
    parser.add_argument("--slippage-bps-per-side", type=float, default=1.0)
    parser.add_argument("--max-holding-minutes", type=int, default=60)
    args = parser.parse_args()

    if args.fee_bps_per_side < 0 or args.slippage_bps_per_side < 0:
        raise SystemExit("fee and slippage assumptions cannot be negative")
    rr_values = [float(x.strip()) for x in args.rr.split(",") if x.strip()]
    if not rr_values or any(x <= 0 for x in rr_values):
        raise SystemExit("--rr must contain positive comma-separated values")

    price_path = Path(args.price_file)
    if not price_path.exists():
        raise SystemExit(f"Price file not found: {price_path}")
    prices = pd.read_parquet(price_path, columns=["time", "open", "high", "low", "close", "volume"])
    prices["time"] = pd.to_datetime(prices["time"], utc=True)
    prices = prices.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    candidates = build_mtf_signal_frame(prices)
    variants, candidate_counts = build_gate_variants(candidates)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_rows, trade_frames = [], []
    base_cols = ["available_at", "open", "high", "low", "close", "atr20"]

    for variant_name, signal in variants.items():
        selected = candidates[base_cols].copy()
        selected["signal"] = signal.to_numpy()
        candidate_count = candidate_counts[variant_name]
        module, gate_mode = variant_name.rsplit("__", 1)
        for rr in rr_values:
            trades = replay_structure_candidates(
                selected, prices, reward_risk=rr,
                fee_bps_per_side=args.fee_bps_per_side,
                slippage_bps_per_side=args.slippage_bps_per_side,
                max_holding_minutes=args.max_holding_minutes,
            )
            if not trades.empty:
                trades["module"] = module
                trades["gate_mode"] = gate_mode
                trades["reward_risk"] = rr
                trade_frames.append(trades)
            score = _score(trades, candidate_count, args.fee_bps_per_side, args.slippage_bps_per_side)
            summary_rows.append({
                "module": module, "gate_mode": gate_mode, "reward_risk": rr, **score,
            })

    summary = pd.DataFrame(summary_rows)
    trades_all = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    summary.to_csv(out / "mtf_gate_ablation_summary.csv", index=False)
    trades_all.to_csv(out / "mtf_gate_ablation_trades.csv", index=False)

    time_diff = prices["time"].diff().dropna()
    report = {
        "purpose": (
            "Post-hoc diagnostic ablation of the already-evaluated MTF gate. "
            "Exploratory only; not a confirmatory test or strategy approval."
        ),
        "source_file": str(price_path),
        "rows_1m": int(len(prices)),
        "price_first_utc": prices["time"].min().isoformat(),
        "price_last_utc": prices["time"].max().isoformat(),
        "missing_minute_intervals": int((time_diff > pd.Timedelta(minutes=1)).sum()),
        "candidate_rows_5m": int(len(candidates)),
        "gate_modes": {
            "4h1h15m": "Original frozen directional gate: 4h and 1h must agree; 15m bias must agree where specified by each module.",
            "15m_only": "Same raw 5m entry setups; remove 4h/1h gate and retain 15m directional bias.",
            "ungated": "Same raw 5m entry setups with no directional structure-bias gate.",
        },
        "assumptions": {
            "entry": "first 1m open at or after completed 5m signal close",
            "stop": "signal 5m candle extreme plus 0.1 x rolling 20-bar 5m ATR",
            "same_minute_stop_target": "stop first",
            "overlapping_trades": "one active position per module/gate-mode/RR replay",
            "max_holding_minutes": args.max_holding_minutes,
            "fee_bps_per_side": args.fee_bps_per_side,
            "slippage_bps_per_side": args.slippage_bps_per_side,
            "funding": "not included",
            "maker_fills": "not assumed; entries and exits modelled as taker",
            "data_gap_caveat": (
                "The current replay advances across observed 1m rows; it does not yet censor a position "
                "when a missing minute occurs during its holding window. Results remain diagnostic."
            ),
            "combined_rule": (
                "Combined consensus is long only when one or more modules signal long and none signal short; "
                "short only when one or more modules signal short and none signal long; opposing same-bar "
                "module signals are dropped as FLAT."
            ),
        },
        "results": summary_rows,
    }
    (out / "mtf_gate_ablation_report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
