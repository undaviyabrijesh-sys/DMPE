"""Build live minute-level trade and order-book features from raw JSONL."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dmpe.orderflow import build_live_orderflow_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DMPE live order-flow feature dataset")
    parser.add_argument("--input-dir", default="data/raw/live")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()
    report = build_live_orderflow_dataset(args.input_dir, args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
