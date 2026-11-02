"""Run fee/slippage execution simulation on repeated walk-forward predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dmpe.execution_sim import run_slippage_sensitivity  # noqa: E402

DEFAULT_SLIPPAGE_SCENARIOS = (0.0, 1.0, 2.0, 5.0, 10.0)


def parse_float_list(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Provide comma-separated numeric values.") from exc
    if not values or any(number < 0 for number in values):
        raise argparse.ArgumentTypeError("Values must be non-negative and non-empty.")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate DMPE walk-forward signals with execution costs")
    parser.add_argument("--signals-file", default="wf_input/walk_forward_test_signals.csv")
    parser.add_argument("--price-file", default="stage1_input/history/BTCUSD_1m.parquet")
    parser.add_argument("--feature-file", default="stage1_input/processed/dmpe_BTCUSD_1m_features.parquet")
    parser.add_argument("--output-dir", default="stage4_output")
    parser.add_argument("--fee-bps-per-side", type=float, default=5.9,
                        help="Taker commission including GST, in bps per side.")
    parser.add_argument("--slippage-scenarios-bps", type=parse_float_list,
                        default=DEFAULT_SLIPPAGE_SCENARIOS)
    parser.add_argument("--lot-size-btc", type=float, default=0.001)
    parser.add_argument("--contracts", type=float, default=1.0)
    args = parser.parse_args()

    signals_path, price_path, feature_path = (
        Path(args.signals_file), Path(args.price_file), Path(args.feature_file)
    )
    for path in (signals_path, price_path, feature_path):
        if not path.exists():
            raise SystemExit(f"Required input file not found: {path}")

    signals = pd.read_csv(signals_path)
    required_signal_columns = {
        "time", "fold", "horizon", "barrier_multiple", "signal",
        "p_down", "p_flat", "p_up", "validation_threshold",
    }
    missing = required_signal_columns.difference(signals.columns)
    if missing:
        raise SystemExit(f"Walk-forward signal file missing columns: {sorted(missing)}")
    if signals.empty:
        raise SystemExit("Walk-forward signal file is empty; there are no directional calls to simulate.")
    signals["time"] = pd.to_datetime(signals["time"], utc=True, errors="coerce")
    signals = signals.dropna(subset=["time"])
    signals["signal"] = pd.to_numeric(signals["signal"], errors="coerce")
    signals["barrier_multiple"] = pd.to_numeric(signals["barrier_multiple"], errors="coerce")
    signals = signals.dropna(subset=["signal", "barrier_multiple"])
    signals["signal"] = signals["signal"].astype("int8")

    prices = pd.read_parquet(price_path, columns=["time", "open", "high", "low", "close"])
    prices["time"] = pd.to_datetime(prices["time"], utc=True, errors="coerce")
    prices = prices.dropna(subset=["time"]).sort_values("time")

    needed_volatility = [
        "realized_vol_15m_pct",
        "realized_vol_60m_pct",
        "realized_vol_180m_pct",
    ]
    features = pd.read_parquet(feature_path, columns=["time", *needed_volatility])
    features["time"] = pd.to_datetime(features["time"], utc=True, errors="coerce")
    features = features.dropna(subset=["time"]).sort_values("time")

    summary, trades, assumptions = run_slippage_sensitivity(
        signals_frame=signals,
        price_frame=prices,
        feature_frame=features,
        fee_bps_per_side_including_gst=args.fee_bps_per_side,
        slippage_scenarios_bps=args.slippage_scenarios_bps,
        lot_size_btc=args.lot_size_btc,
        contracts=args.contracts,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "execution_sensitivity.csv"
    trades_path = output_dir / "execution_trades.csv"
    report_path = output_dir / "execution_simulation.json"
    summary.to_csv(summary_path, index=False)
    trades.to_csv(trades_path, index=False)

    report = {
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "purpose": "Execution-cost sensitivity analysis of out-of-sample DMPE predictions; not live trading.",
        "inputs": {
            "signals_file": str(signals_path),
            "price_file": str(price_path),
            "feature_file": str(feature_path),
            "input_signal_rows": int(len(signals)),
            "slippage_scenarios_bps_per_side": list(args.slippage_scenarios_bps),
            "fee_bps_per_side_including_gst": args.fee_bps_per_side,
            "lot_size_btc": args.lot_size_btc,
            "contracts": args.contracts,
        },
        "execution_assumptions": assumptions,
        "results": summary.to_dict(orient="records"),
        "output_files": {
            "summary": str(summary_path),
            "trades": str(trades_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "input_signal_rows": len(signals),
        "summary": summary.to_dict(orient="records"),
        "output_dir": str(output_dir),
        "funding_included": False,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
