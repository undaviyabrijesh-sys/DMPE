"""Profile DMPE historical Parquet datasets and report exact time gaps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dmpe.features import profile_series  # noqa: E402

SPECS = {
    "BTCUSD_1m.parquet": ("traded_price_1m", 60),
    "MARK_BTCUSD_1m.parquet": ("mark_price_1m", 60),
    "FUNDING_BTCUSD_1h.parquet": ("funding_rate_1h", 3600),
    "OI_BTCUSD_1h.parquet": ("open_interest_1h", 3600),
}


def largest_gaps(frame: pd.DataFrame, expected_seconds: int, limit: int = 25) -> list[dict]:
    times = pd.to_datetime(frame["time"], utc=True, errors="coerce").dropna().drop_duplicates().sort_values()
    delta = times.diff().dt.total_seconds()
    indexes = delta[delta > expected_seconds].nlargest(limit).index
    rows = []
    for idx in indexes:
        pos = times.index.get_loc(idx)
        previous = times.iloc[pos - 1]
        current = times.iloc[pos]
        seconds = float((current - previous).total_seconds())
        rows.append({
            "previous_timestamp_utc": previous.isoformat(),
            "next_timestamp_utc": current.isoformat(),
            "gap_minutes": seconds / 60,
            "missing_intervals": max(int(round(seconds / expected_seconds)) - 1, 0),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile DMPE market history and timestamp gaps")
    parser.add_argument("--directory", default="data/raw/history")
    parser.add_argument("--output-dir", default="data/raw/metadata")
    args = parser.parse_args()
    directory = Path(args.directory)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries, gaps = [], {}
    loaded: dict[str, pd.DataFrame] = {}
    for filename, (name, seconds) in SPECS.items():
        path = directory / filename
        if not path.exists():
            summaries.append({"series": name, "status": "missing_file", "file": str(path)})
            continue
        frame = pd.read_parquet(path)
        loaded[name] = frame
        row = profile_series(name, frame, seconds)
        row["file"] = str(path)
        row["status"] = "ok" if not (
            row["duplicate_timestamps"] or row["invalid_ohlc_rows"]
        ) else "issues"
        summaries.append(row)
        gaps[name] = largest_gaps(frame, seconds)

    # Coverage intersections matter before any multi-source feature join.
    if "traded_price_1m" in loaded and "mark_price_1m" in loaded:
        px = pd.to_datetime(loaded["traded_price_1m"]["time"], utc=True)
        mk = pd.to_datetime(loaded["mark_price_1m"]["time"], utc=True)
        overlap = px.isin(set(mk))
        cross = {
            "price_rows": int(len(px)),
            "price_rows_with_exact_mark_timestamp": int(overlap.sum()),
            "price_rows_without_exact_mark_timestamp": int((~overlap).sum()),
            "exact_mark_match_rate_pct": float(overlap.mean() * 100),
        }
    else:
        cross = {}

    report = {
        "profiled_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "datasets": summaries,
        "largest_gaps": gaps,
        "cross_series_alignment": cross,
        "policy": [
            "Do not forward-fill missing traded-price OHLC candles.",
            "Do not calculate lagged features across missing one-minute intervals.",
            "Hourly funding and open-interest bars are delayed by one full hour plus one minute before feature joins.",
            "Missing target horizons remain null; they are not imputed.",
        ],
    }
    (output_dir / "history_profile.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(summaries).to_csv(output_dir / "history_profile.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
