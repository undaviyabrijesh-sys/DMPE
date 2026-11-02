"""Build a DMPE first-pass 1-minute feature and research-target dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dmpe.features import build_feature_frame  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DMPE 1m feature and label dataset")
    parser.add_argument("--input-dir", default="data/raw/history")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    price_path = input_dir / "BTCUSD_1m.parquet"
    if not price_path.exists():
        raise SystemExit(f"Required traded-price dataset not found: {price_path}")
    price = pd.read_parquet(price_path)
    mark_path = input_dir / "MARK_BTCUSD_1m.parquet"
    funding_path = input_dir / "FUNDING_BTCUSD_1h.parquet"
    oi_path = input_dir / "OI_BTCUSD_1h.parquet"

    mark = pd.read_parquet(mark_path) if mark_path.exists() else None
    funding = pd.read_parquet(funding_path) if funding_path.exists() else None
    oi = pd.read_parquet(oi_path) if oi_path.exists() else None

    features, feature_columns, label_columns = build_feature_frame(
        price, mark_frame=mark, funding_frame=funding, oi_frame=oi
    )
    output_path = output_dir / "dmpe_BTCUSD_1m_features.parquet"
    features.to_parquet(
        output_path, index=False, engine="pyarrow", compression="zstd", row_group_size=100_000
    )

    schema = {
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "rows": int(len(features)),
        "first_time_utc": pd.to_datetime(features["time"], utc=True).min().isoformat(),
        "last_time_utc": pd.to_datetime(features["time"], utc=True).max().isoformat(),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "label_count": len(label_columns),
        "label_columns": label_columns,
        "label_policy": (
            "Forward return and future max excursion columns are evaluation targets only. "
            "Never include target_* columns in model inputs."
        ),
        "hourly_data_availability": (
            "Funding and OI values become eligible only after source timestamp + 1h 1m."
        ),
        "gap_policy": (
            "Minute grid is explicit; OHLCV is not forward-filled. Rolling descriptors require "
            "complete windows. Forward labels are null if any future one-minute return is missing."
        ),
        "data_sources": {
            "traded_price": str(price_path),
            "mark_price": str(mark_path) if mark is not None else None,
            "funding": str(funding_path) if funding is not None else None,
            "open_interest": str(oi_path) if oi is not None else None,
        },
        "output_file": str(output_path),
    }
    (output_dir / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    summary = {
        "rows": int(len(features)),
        "features": len(feature_columns),
        "labels": len(label_columns),
        "output_file": str(output_path),
        "null_rates": {
            col: float(features[col].isna().mean())
            for col in feature_columns + label_columns
        },
    }
    (output_dir / "feature_build_report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
