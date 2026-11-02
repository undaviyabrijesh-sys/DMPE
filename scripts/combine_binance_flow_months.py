"""Combine monthly Binance proxy flow Parquets into a chronological point-in-time dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="binance_combined")
    args = parser.parse_args()

    paths = sorted(Path(args.input_dir).rglob("binance_BTCUSDT_aggtrade_flow_1m_*.parquet"))
    if not paths:
        raise SystemExit("No monthly Binance flow Parquet files found.")
    frames = [pd.read_parquet(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    combined["time"] = pd.to_datetime(combined["time"], utc=True, errors="coerce")
    if combined["time"].isna().any() or combined["time"].duplicated().any():
        raise ValueError("Combined monthly flow timestamps are invalid or duplicated.")
    combined = combined.sort_values("time").reset_index(drop=True)
    if len(combined) > 1:
        gaps = combined["time"].diff().dropna()
        if not gaps.eq(pd.Timedelta(minutes=1)).all():
            raise ValueError("Combined monthly archive coverage has a gap or overlap; do not silently fill it.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "binance_BTCUSDT_aggtrade_flow_1m_2024-11_to_2026-10.parquet"
    combined.to_parquet(output, index=False, compression="zstd")
    report = {
        "monthly_files": [path.name for path in paths],
        "month_count": len(paths),
        "rows": len(combined),
        "first_close_utc": combined["time"].min().isoformat(),
        "last_close_utc": combined["time"].max().isoformat(),
        "minute_gaps": 0,
        "venue_semantics": "Binance USD-M BTCUSDT proxy, not Delta observed flow",
        "output_file": str(output),
    }
    (out_dir / "combined_flow_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
