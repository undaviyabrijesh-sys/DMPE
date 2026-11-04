"""Build causally finalized live 1-minute traded-price and mark-price candles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dmpe.live_candles import build_closed_candle_frame  # noqa: E402
from dmpe.orderflow import read_jsonl_files  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/raw/live")
    parser.add_argument("--output-dir", default="data/processed/live_candles")
    parser.add_argument("--symbol", default="BTCUSD")
    parser.add_argument("--close-grace-seconds", type=int, default=5)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    paths = sorted(input_dir.glob("delta_*.jsonl")) if input_dir.exists() else []
    if not paths:
        raise SystemExit(f"No Delta raw JSONL captures found in {input_dir}")
    records, ingest = read_jsonl_files(paths)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {}
    reports = {}
    for label, symbol in (
        ("traded_price", args.symbol),
        ("mark_price", f"MARK:{args.symbol}"),
    ):
        frame, report = build_closed_candle_frame(
            records, symbol=symbol, close_grace_seconds=args.close_grace_seconds
        )
        frames[label] = frame
        reports[label] = report
        if not frame.empty:
            filename = f"{symbol.replace(':', '_')}_1m_live.parquet"
            frame.to_parquet(output_dir / filename, index=False, compression="zstd")
            frame.to_csv(output_dir / filename.replace(".parquet", ".csv"), index=False)

    report = {
        "input": ingest,
        "traded_price": reports["traded_price"],
        "mark_price": reports["mark_price"],
        "timestamp_semantics": (
            "Output time is candle OPEN, matching Delta REST history and the feature "
            "builder. The candle is eligible only after close plus configured grace."
        ),
        "feature_parity_status": "not_yet_tested",
        "output_dir": str(output_dir),
    }
    (output_dir / "live_candle_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
