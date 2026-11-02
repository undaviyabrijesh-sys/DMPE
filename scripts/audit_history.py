"""Audit candle Parquet files for duplicates, malformed OHLC and time gaps."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

RESOLUTION_SECONDS = {"1m": 60, "1h": 3600}


def audit_file(path: Path) -> dict:
    frame = pd.read_parquet(path)
    required = {"time", "open", "high", "low", "close", "volume"}
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        return {"file": str(path), "status": "invalid", "error": f"Missing columns: {missing_columns}"}

    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    invalid_times = int(frame["time"].isna().sum())
    frame = frame.dropna(subset=["time"])
    duplicate_timestamps = int(frame["time"].duplicated().sum())
    invalid_ohlc_mask = (
        (frame["low"] > frame["high"])
        | (frame["open"] < frame["low"]) | (frame["open"] > frame["high"])
        | (frame["close"] < frame["low"]) | (frame["close"] > frame["high"])
    )
    invalid_ohlc = int(invalid_ohlc_mask.sum())
    step = next(
        (seconds for suffix, seconds in RESOLUTION_SECONDS.items()
         if path.stem.endswith(f"_{suffix}")),
        None,
    )
    missing_intervals = None
    if step and not frame.empty:
        unique_times = frame["time"].drop_duplicates().sort_values()
        expected = int((unique_times.iloc[-1] - unique_times.iloc[0]).total_seconds() // step) + 1
        missing_intervals = max(expected - len(unique_times), 0)

    status = "ok" if not (invalid_times or duplicate_timestamps or invalid_ohlc) else "issues"
    return {
        "file": str(path), "status": status, "rows": int(len(frame)),
        "first_utc": frame["time"].min().isoformat() if not frame.empty else None,
        "last_utc": frame["time"].max().isoformat() if not frame.empty else None,
        "invalid_timestamps": invalid_times, "duplicate_timestamps": duplicate_timestamps,
        "invalid_ohlc_rows": invalid_ohlc,
        "missing_intervals_between_first_last": missing_intervals, "error": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit DMPE market-history files")
    parser.add_argument("--directory", default="data/raw/history")
    parser.add_argument("--output", default="data/raw/metadata/data_quality_report.csv")
    args = parser.parse_args()
    directory = Path(args.directory)
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No Parquet files found in {directory}")

    results = [audit_file(path) for path in files]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output, index=False)
    print(pd.DataFrame(results).to_string(index=False))
    if any(row["status"] == "invalid" for row in results):
        raise SystemExit("At least one dataset has an invalid schema.")


if __name__ == "__main__":
    main()
