"""Download a research bundle of BTC traded price, mark, funding and OI history."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dmpe.delta_rest import DeltaPublicClient, utc_epoch_days_ago  # noqa: E402

SERIES = [
    {"name": "traded_price_1m", "symbol": "BTCUSD", "resolution": "1m", "required": True},
    {"name": "mark_price_1m", "symbol": "MARK:BTCUSD", "resolution": "1m", "required": False},
    {"name": "funding_rate_1h", "symbol": "FUNDING:BTCUSD", "resolution": "1h", "required": False},
    {"name": "open_interest_1h", "symbol": "OI:BTCUSD", "resolution": "1h", "required": False},
]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


def build_report_row(item: dict, frame: pd.DataFrame, output: Path) -> dict:
    if frame.empty:
        return {
            "series": item["name"], "symbol": item["symbol"], "resolution": item["resolution"],
            "status": "empty", "rows": 0, "first_utc": None, "last_utc": None,
            "missing_intervals_between_first_last": None, "file": None, "error": None,
        }

    first = frame["time"].min()
    last = frame["time"].max()
    expected = None
    # Only compare expected cadence for price/OI bar series. Funding events may be less frequent.
    if item["name"] in {"traded_price_1m", "mark_price_1m", "open_interest_1h"}:
        step = {"1m": 60, "1h": 3600}[item["resolution"]]
        expected = int((last - first).total_seconds() // step) + 1
    missing = max(expected - len(frame), 0) if expected is not None else None

    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    return {
        "series": item["name"], "symbol": item["symbol"], "resolution": item["resolution"],
        "status": "ok", "rows": int(len(frame)),
        "first_utc": first.isoformat(), "last_utc": last.isoformat(),
        "missing_intervals_between_first_last": missing,
        "file": str(output.relative_to(ROOT)), "error": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill a Delta Exchange BTC research bundle")
    parser.add_argument("--days", type=int, default=1825, help="History window (default: five years)")
    parser.add_argument("--output-dir", default="data/raw/history")
    args = parser.parse_args()
    if args.days <= 0:
        parser.error("--days must be a positive integer")

    start_epoch, end_epoch = utc_epoch_days_ago(args.days)
    start_utc = datetime.fromtimestamp(start_epoch, timezone.utc).isoformat()
    end_utc = datetime.fromtimestamp(end_epoch, timezone.utc).isoformat()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    client = DeltaPublicClient()
    rows: list[dict] = []
    errors: list[dict] = []
    required_ok = False

    for item in SERIES:
        output = output_dir / f"{safe_name(item['symbol'])}_{item['resolution']}.parquet"
        print(
            f"Fetching {item['name']} ({item['symbol']} / {item['resolution']}) "
            f"from {start_utc} to {end_utc}..."
        )
        try:
            frame = client.backfill_candles(
                item["symbol"], item["resolution"], start_epoch, end_epoch
            )
            row = build_report_row(item, frame, output)
            rows.append(row)
            print(
                f"  {row['status']}: {row['rows']:,} rows; "
                f"{row['first_utc']} -> {row['last_utc']}"
            )
            if item["required"]:
                required_ok = not frame.empty
                if not required_ok:
                    errors.append({"series": item["name"], "error": "No traded-price candles returned"})
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            errors.append({"series": item["name"], "error": error})
            rows.append({
                "series": item["name"], "symbol": item["symbol"],
                "resolution": item["resolution"], "status": "error",
                "rows": 0, "first_utc": None, "last_utc": None,
                "missing_intervals_between_first_last": None, "file": None, "error": error,
            })
            print(f"  ERROR: {error}")
        time.sleep(1)

    report = {
        "project": "DMPE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "venue": "Delta Exchange India",
        "requested_days": args.days,
        "requested_start_utc": start_utc,
        "requested_end_utc": end_utc,
        "api_max_candles_per_response": 2000,
        "series": rows,
        "errors": errors,
        "notes": [
            "The historical candles API documents a maximum of 2,000 candles per response; data are paged in time windows.",
            "Actual earliest/latest available timestamps are determined from returned data, not assumed.",
            "Live trade and order-book history starts only when the WebSocket recorder is running.",
            "Funding and OI are requested at 1-hour resolution; inspect actual coverage and cadence.",
            "No API keys, account data, or order execution are used.",
        ],
    }
    metadata_dir = ROOT / "data/raw/metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "backfill_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(metadata_dir / "backfill_report.csv", index=False)
    print(f"Backfill report written to {metadata_dir / 'backfill_report.json'}")

    if not required_ok:
        raise SystemExit("Required BTCUSD 1m history did not download; see report for details.")


if __name__ == "__main__":
    main()
