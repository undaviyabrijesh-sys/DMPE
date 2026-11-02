"""Download one verified Binance USD-M BTCUSDT monthly aggTrades archive."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import requests


BASE = "https://data.binance.vision/data/futures/um/monthly/aggTrades"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True, choices=range(1, 13))
    parser.add_argument("--output-dir", default="binance_raw")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    symbol = args.symbol.upper()
    if not re.fullmatch(r"[A-Z0-9]+", symbol):
        raise SystemExit("Invalid symbol.")
    filename = f"{symbol}-aggTrades-{args.year}-{args.month:02d}.zip"
    url = f"{BASE}/{symbol}/{filename}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / filename
    checksum_file = out_dir / f"{filename}.CHECKSUM"

    with requests.get(url, stream=True, timeout=args.timeout) as response:
        response.raise_for_status()
        with archive.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    checksum_url = url + ".CHECKSUM"
    checksum_response = requests.get(checksum_url, timeout=args.timeout)
    checksum_response.raise_for_status()
    checksum_file.write_text(checksum_response.text, encoding="utf-8")
    expected = checksum_response.text.strip().split()[0].lower()
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != expected:
        archive.unlink(missing_ok=True)
        raise SystemExit(f"SHA256 mismatch for {filename}: expected {expected}, got {digest}")

    manifest = {
        "symbol": symbol,
        "market": "USD-M perpetual futures",
        "archive": filename,
        "url": url,
        "sha256": digest,
        "bytes": archive.stat().st_size,
        "checksum_url": checksum_url,
        "checksum_verified": True,
    }
    (out_dir / f"{filename}.manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(manifest)


if __name__ == "__main__":
    main()
