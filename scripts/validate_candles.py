from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main() -> None:
    parser = argparse.ArgumentParser(description="Check basic candle data integrity")
    parser.add_argument("file", type=Path)
    args = parser.parse_args()
    frame = pd.read_parquet(args.file)
    required = {"time", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    invalid_ohlc = ((frame["low"] > frame["high"]) | (frame["open"] < frame["low"]) |
        (frame["open"] > frame["high"]) | (frame["close"] < frame["low"]) |
        (frame["close"] > frame["high"]))
    duplicates = int(frame["time"].duplicated().sum())
    null_times = int(frame["time"].isna().sum())
    print(f"Rows: {len(frame):,}")
    print(f"Duplicate timestamps: {duplicates}")
    print(f"Invalid OHLC rows: {int(invalid_ohlc.sum())}")
    print(f"Invalid timestamps: {null_times}")
    if len(frame):
        print(f"UTC range: {frame['time'].min()} -> {frame['time'].max()}")
    if duplicates or invalid_ohlc.any() or null_times:
        raise SystemExit("Data validation failed.")
    print("Basic candle validation passed.")

if __name__ == "__main__":
    main()
