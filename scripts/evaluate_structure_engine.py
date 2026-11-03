"""Evaluate the frozen MTF entry modules with gap-aware execution costs."""
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


def _profit_factor(pnl: pd.Series) -> float | None:
    gains = float(pnl[pnl > 0].sum())
    losses = float(-pnl[pnl < 0].sum())
    return gains / losses if losses > 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-file", required=True, help="Delta BTCUSD 1m candle Parquet")
    parser.add_argument("--output-dir", default="mtf_structure_output")
    parser.add_argument("--rr", default="1,2,3")
    parser.add_argument("--fee-bps-per-side", type=float, default=5.9)
    parser.add_argument("--slippage-bps-per-side", type=float, default=1.0)
    parser.add_argument("--max-holding-minutes", type=int, default=60)
    args = parser.parse_args()

    price_path = Path(args.price_file)
    if not price_path.exists():
        raise SystemExit(f"Price file not found: {price_path}")
    prices = pd.read_parquet(price_path, columns=["time", "open", "high", "low", "close", "volume"])
    prices["time"] = pd.to_datetime(prices["time"], utc=True)
    prices = prices.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    candidates = build_mtf_signal_frame(prices)
    modules = ("pullback", "sweep", "breakout_retest", "combined")
    rr_values = [float(x.strip()) for x in args.rr.split(",") if x.strip()]
    if not rr_values or any(x <= 0 for x in rr_values):
        raise SystemExit("--rr must contain positive comma-separated values")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_rows, trade_frames = [], []
    for module in modules:
        signal_col = "entry_candidate" if module == "combined" else f"entry_{module}"
        selected = candidates[["available_at", "open", "high", "low", "close", "atr20", signal_col]].rename(
            columns={signal_col: "signal"}
        )
        candidate_count = int((selected["signal"] != 0).sum())
        for rr in rr_values:
            trades = replay_structure_candidates(
                selected, prices, reward_risk=rr,
                fee_bps_per_side=args.fee_bps_per_side,
                slippage_bps_per_side=args.slippage_bps_per_side,
                max_holding_minutes=args.max_holding_minutes,
            )
            censored_count = int(trades["censored"].fillna(False).sum()) if not trades.empty else 0
            scored = trades.loc[~trades["censored"].fillna(False)].copy() if not trades.empty else trades
            if not trades.empty:
                trades["module"] = module
                trades["reward_risk"] = rr
                trade_frames.append(trades)

            if scored.empty:
                summary_rows.append({
                    "module": module, "reward_risk": rr, "candidate_signals": candidate_count,
                    "trades": 0, "censored_trades": censored_count, "trades_per_day": 0.0,
                    "win_rate_pct": None, "gross_pnl_points": 0.0, "net_pnl_points": 0.0,
                    "gross_expectancy_R": None, "net_expectancy_R": None,
                    "gross_profit_factor": None, "net_profit_factor": None,
                    "avg_cost_points": None, "median_cost_points": None,
                    "max_drawdown_R": None, "fee_bps_per_side": args.fee_bps_per_side,
                    "slippage_bps_per_side": args.slippage_bps_per_side,
                })
                continue

            gross = scored["gross_pnl_points"].astype(float)
            net = scored["net_pnl_points"].astype(float)
            equity = scored["net_r"].astype(float).cumsum()
            drawdown = equity - equity.cummax()
            start = pd.to_datetime(scored["entry_time"], utc=True).min()
            end = pd.to_datetime(scored["exit_time"], utc=True).max()
            days = max((end - start).total_seconds() / 86400.0, 1 / 1440)
            summary_rows.append({
                "module": module,
                "reward_risk": rr,
                "candidate_signals": candidate_count,
                "trades": int(len(scored)),
                "censored_trades": censored_count,
                "trades_per_day": float(len(scored) / days),
                "win_rate_pct": float((net > 0).mean() * 100),
                "gross_pnl_points": float(gross.sum()),
                "net_pnl_points": float(net.sum()),
                "gross_expectancy_R": float((gross / scored["risk_points"]).mean()),
                "net_expectancy_R": float(scored["net_r"].mean()),
                "gross_profit_factor": _profit_factor(gross),
                "net_profit_factor": _profit_factor(net),
                "avg_cost_points": float(scored["cost_points"].mean()),
                "median_cost_points": float(scored["cost_points"].median()),
                "max_drawdown_R": float(drawdown.min()),
                "fee_bps_per_side": args.fee_bps_per_side,
                "slippage_bps_per_side": args.slippage_bps_per_side,
            })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "mtf_structure_summary.csv", index=False)
    trades_all = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    trades_all.to_csv(out / "mtf_structure_trades.csv", index=False)
    candidates.to_parquet(out / "mtf_structure_candidates.parquet", index=False)

    time_diff = prices["time"].diff().dropna()
    gap_durations = time_diff[time_diff > pd.Timedelta(minutes=1)]
    missing_minutes = int(((gap_durations / pd.Timedelta(minutes=1)).round().astype("int64") - 1).sum())
    max_missing_run = int((gap_durations.max() / pd.Timedelta(minutes=1)) - 1) if not gap_durations.empty else 0
    report = {
        "purpose": "Exploratory structure-module replay; not an approved strategy.",
        "source_file": str(price_path),
        "rows_1m": int(len(prices)),
        "price_first_utc": prices["time"].min().isoformat(),
        "price_last_utc": prices["time"].max().isoformat(),
        "missing_minute_intervals": missing_minutes,
        "data_gap_events": int(len(gap_durations)),
        "max_missing_run_minutes": max_missing_run,
        "candidate_rows_5m": int(len(candidates)),
        "assumptions": {
            "entry": "exact 1m open at the completed 5m signal availability timestamp; stale signals skipped",
            "stop": "signal 5m candle extreme plus 0.1 x rolling 20-bar 5m ATR",
            "same_minute_stop_target": "stop first",
            "overlapping_trades": "one active position per module/RR replay",
            "max_holding_minutes": args.max_holding_minutes,
            "fee_bps_per_side": args.fee_bps_per_side,
            "slippage_bps_per_side": args.slippage_bps_per_side,
            "funding": "not included; must be added after settlement convention validation",
            "maker_fills": "not assumed; all entries and exits modelled as taker",
            "data_gaps": "positions crossing a missing 1m interval are censored and excluded from performance metrics; missing outcomes are not imputed",
            "end_of_data": "positions that cannot reach their defined time exit before the file ends are censored",
        },
        "results": summary_rows,
    }
    (out / "mtf_structure_report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
