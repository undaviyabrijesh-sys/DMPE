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
    args = parser.parse_args()
    LiveRecorder(symbol=args.symbol, output_dir=ROOT / args.output_dir).run_forever()

if __name__ == "__main__":
    main()
