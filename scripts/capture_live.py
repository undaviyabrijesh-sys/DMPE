from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dmpe.live_recorder import LiveRecorder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture Delta Exchange public live market data")
    parser.add_argument("--symbol", default="BTCUSD")
    parser.add_argument("--output-dir", default="data/raw/live")
    parser.add_argument(
        "--duration-minutes", type=int, default=None,
        help="Optional finite capture duration; omit to run until interrupted",
    )
    args = parser.parse_args()
    if args.duration_minutes is not None and not 1 <= args.duration_minutes <= 120:
        parser.error("--duration-minutes must be between 1 and 120")
    recorder = LiveRecorder(symbol=args.symbol, output_dir=ROOT / args.output_dir)
    seconds = args.duration_minutes * 60 if args.duration_minutes is not None else None
    recorder.run_forever(duration_seconds=seconds)


if __name__ == "__main__":
    main()
