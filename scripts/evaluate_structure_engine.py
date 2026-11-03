"""Evaluate MTF entry modules independently across cost and R:R assumptions."""
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
        for rr in rr_values:
            trades = replay_structure_candidates(
                selected, prices, reward_risk=rr,
                fee_bps_per_side=args.fee_bps_per_side,
                slippage_bps_per_side=args.slippage_bps_per_side,
                max_holding_minutes=args.max_holding_minutes,
            )
            if not trades.empty:
                trades["module"] = module
                trades["reward_risk"] = rr
                trade_frames.append(trades)
                wins = int((trades["net_pnl_points"] > 0).sum())
                gross_profit = float(trades.loc[trades["net_pnl_points"] > 0, "net_pnl_points"].sum())
                gross_loss = float(-trades.loc[trades["net_pnl_points"] < 0, "net_pnl_points"].sum())
                equity = trades["net_r"].cumsum()
                drawdown = equity - equity.cummax()
                net_sum = float(trades["net_pnl_points"].sum())
                expectancy = float(trades["net_r"].mean())
                pf = gross_profit / gross_loss if gross_loss > 0 else None
                start = pd.to_datetime(trades["entry_time"], utc=True).min()
                end = pd.to_datetime(trades["exit_time"], utc=True).max()
                days = max((end - start).total_seconds() / 86400, 1 / 1440)
                summary_rows.append({
                    "module": module, "reward_risk": rr, "trades": len(trades),
                    "trades_per_day": len(trades) / days, "win_rate_pct": wins / len(trades) * 100,
                    "net_pnl_points": net_sum, "expectancy_R": expectancy,
                    "profit_factor": pf, "max_drawdown_R": float(drawdown.min()),
                    "fee_bps_per_side": args.fee_bps_per_side,
                    "slippage_bps_per_side": args.slippage_bps_per_side,
                })
            else:
                summary_rows.append({
                    "module": module, "reward_risk": rr, "trades": 0,
                    "trades_per_day": 0, "win_rate_pct": None, "net_pnl_points": 0,
                    "expectancy_R": None, "profit_factor": None, "max_drawdown_R": None,
                    "fee_bps_per_side": args.fee_bps_per_side,
                    "slippage_bps_per_side": args.slippage_bps_per_side,
                })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "mtf_structure_summary.csv", index=False)
    trades_all = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    trades_all.to_csv(out / "mtf_structure_trades.csv", index=False)
    candidates.to_parquet(out / "mtf_structure_candidates.parquet", index=False)
    report = {
        "purpose": "Exploratory structure-module replay; not an approved strategy.",
        "source_file": str(price_path),
        "rows_1m": int(len(prices)),
        "candidate_rows_5m": int(len(candidates)),
        "assumptions": {
            "entry": "first 1m open at or after completed 5m signal close",
            "stop": "signal 5m candle extreme plus 0.1 x rolling 20-bar 5m ATR",
            "same_minute_stop_target": "stop first",
            "overlapping_trades": "one active position per module/RR replay",
            "max_holding_minutes": args.max_holding_minutes,
            "fee_bps_per_side": args.fee_bps_per_side,
            "slippage_bps_per_side": args.slippage_bps_per_side,
            "funding": "not included; must be added after settlement convention validation",
            "maker_fills": "not assumed; all entries and exits modelled as taker",
        },
        "results": summary_rows,
    }
    (out / "mtf_structure_report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
