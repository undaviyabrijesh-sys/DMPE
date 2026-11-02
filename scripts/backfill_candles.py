from __future__ import annotations
import argparse
from pathlib import Path
import re
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dmpe.delta_rest import DeltaPublicClient, utc_epoch_days_ago  # noqa: E402

def safe_symbol(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", symbol)

def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Delta Exchange India candle history")
    parser.add_argument("--symbol", default="BTCUSD")
    parser.add_argument("--resolution", default="1m")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--output-dir", default="data/raw/candles")
    args = parser.parse_args()
    start, end = utc_epoch_days_ago(args.days)
    frame = DeltaPublicClient().backfill_candles(args.symbol, args.resolution, start, end)
    if frame.empty:
        raise SystemExit("No candles returned. Check symbol, resolution and date range.")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{safe_symbol(args.symbol)}_{args.resolution}.parquet"
    frame.to_parquet(output_path, index=False)
    print(f"Saved {len(frame):,} rows to {output_path}")
    print(f"UTC range: {frame['time'].min()} -> {frame['time'].max()}")

if __name__ == "__main__":
    main()
