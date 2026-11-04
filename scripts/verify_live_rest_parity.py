"""Verify live Delta candles against REST and measure Stage 1 warm-up availability."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dmpe.delta_rest import DeltaPublicClient, filter_completed_candles  # noqa: E402
from dmpe.live_parity import compare_candle_frames, trailing_contiguous_rows  # noqa: E402


def _profile(frame: pd.DataFrame, resolution: str, as_of: pd.Timestamp) -> dict:
    seconds = 60 if resolution == "1m" else 3600
    contiguous = trailing_contiguous_rows(frame, resolution)
    latest_close = (
        pd.to_datetime(frame["time"], utc=True).max() + pd.Timedelta(seconds=seconds)
        if not frame.empty else None
    )
    age_seconds = (
        float((as_of - latest_close).total_seconds()) if latest_close is not None else None
    )
    return {
        "rows": int(len(frame)),
        "trailing_contiguous_rows": int(contiguous),
        "first_open_utc": pd.to_datetime(frame["time"], utc=True).min().isoformat() if not frame.empty else None,
        "last_open_utc": pd.to_datetime(frame["time"], utc=True).max().isoformat() if not frame.empty else None,
        "last_complete_close_utc": latest_close.isoformat() if latest_close is not None else None,
        "last_complete_bar_age_seconds": age_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candle-dir", default="candle_output/processed")
    parser.add_argument("--output-dir", default="parity_output/processed")
    parser.add_argument("--symbol", default="BTCUSD")
    parser.add_argument("--price-lookback-hours", type=int, default=6)
    parser.add_argument("--hourly-lookback-hours", type=int, default=48)
    parser.add_argument("--finality-delay-seconds", type=int, default=5)
    parser.add_argument("--strict", action="store_true", help="Fail if any live candle is missing or differs from REST.")
    args = parser.parse_args()

    if args.price_lookback_hours < 5 or args.hourly_lookback_hours < 25:
        parser.error("Use at least 5 hours of 1m data and 25 hours of 1h funding/OI history.")

    candle_dir = Path(args.candle_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    as_of = pd.Timestamp.now(tz="UTC")

    live_frames = {}
    for label, symbol in (
        ("traded_price", args.symbol),
        ("mark_price", f"MARK:{args.symbol}"),
    ):
        path = candle_dir / f"{symbol.replace(':', '_')}_1m_live.parquet"
        if not path.exists():
            raise SystemExit(f"Finalized live candle file not found: {path}")
        live_frames[label] = pd.read_parquet(path)

    client = DeltaPublicClient()
    one_min_start = int((as_of - pd.Timedelta(hours=args.price_lookback_hours)).timestamp())
    hourly_start = int((as_of - pd.Timedelta(hours=args.hourly_lookback_hours)).timestamp())
    end_epoch = int(as_of.timestamp())

    rest_frames = {}
    for label, symbol in (
        ("traded_price", args.symbol),
        ("mark_price", f"MARK:{args.symbol}"),
    ):
        raw = client.backfill_candles(symbol, "1m", one_min_start, end_epoch)
        rest_frames[label] = filter_completed_candles(
            raw, "1m", as_of, finality_delay_seconds=args.finality_delay_seconds
        )
    hourly_frames = {}
    for label, symbol in (
        ("funding", f"FUNDING:{args.symbol}"),
        ("open_interest", f"OI:{args.symbol}"),
    ):
        raw = client.backfill_candles(symbol, "1h", hourly_start, end_epoch)
        hourly_frames[label] = filter_completed_candles(
            raw, "1h", as_of, finality_delay_seconds=args.finality_delay_seconds
        )

    comparisons = {}
    for label in ("traded_price", "mark_price"):
        comparison_columns = ("open", "high", "low", "close")
        if label == "traded_price":
            comparison_columns = (*comparison_columns, "volume")
        comparisons[label] = compare_candle_frames(
            live_frames[label],
            rest_frames[label],
            columns=comparison_columns,
        )

    for label, frame in rest_frames.items():
        frame.to_parquet(output_dir / f"{label}_1m_rest.parquet", index=False)
    for label, frame in hourly_frames.items():
        frame.to_parquet(output_dir / f"{label}_1h_rest.parquet", index=False)

    profiles = {
        "traded_price_1m": _profile(rest_frames["traded_price"], "1m", as_of),
        "mark_price_1m": _profile(rest_frames["mark_price"], "1m", as_of),
        "funding_1h": _profile(hourly_frames["funding"], "1h", as_of),
        "open_interest_1h": _profile(hourly_frames["open_interest"], "1h", as_of),
    }
    warmup_requirements = {
        "traded_price_1m_contiguous_rows": 241,
        "mark_price_1m_contiguous_rows": 241,
        "funding_1h_contiguous_rows": 25,
        "open_interest_1h_contiguous_rows": 25,
    }
    warmup_ready = (
        profiles["traded_price_1m"]["trailing_contiguous_rows"] >= 241
        and profiles["mark_price_1m"]["trailing_contiguous_rows"] >= 241
        and profiles["funding_1h"]["trailing_contiguous_rows"] >= 25
        and profiles["open_interest_1h"]["trailing_contiguous_rows"] >= 25
    )
    parity_pass = all(
        result["all_live_bars_match_rest"] for result in comparisons.values()
    )

    report = {
        "built_at_utc": as_of.isoformat(),
        "symbol": args.symbol,
        "data_source": "Delta Exchange India public REST historical candles",
        "resolution_symbols": {
            "traded_price_1m": args.symbol,
            "mark_price_1m": f"MARK:{args.symbol}",
            "funding_1h": f"FUNDING:{args.symbol}",
            "open_interest_1h": f"OI:{args.symbol}",
        },
        "live_vs_rest": comparisons,
        "rest_profiles": profiles,
        "warmup_requirements": warmup_requirements,
        "feature_warmup_ready": bool(warmup_ready),
        "candle_parity_pass": bool(parity_pass),
        "notes": [
            "All REST candles are filtered to source-open plus resolution plus finality delay <= as_of.",
            "A ticker OI snapshot or funding WebSocket snapshot is not substituted for the hourly REST inputs.",
            "The 39-feature schema and model bundle are not loaded by this diagnostic.",
        ],
    }
    (output_dir / "live_rest_parity_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if args.strict and not parity_pass:
        raise SystemExit("Live WebSocket candle data did not pass REST parity.")


if __name__ == "__main__":
    main()
