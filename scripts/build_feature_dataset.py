"""Build point-in-time DMPE features and the two preregistered asymmetric targets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dmpe.features import build_feature_frame  # noqa: E402
from dmpe.labels import add_asymmetric_directional_labels  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DMPE 1m features and frozen 1h/3h labels")
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
    # Exactly two confirmatory horizons: 1h and 3h, each 1R stop / 3R target.
    features, asymmetric_labels, asymmetric_diagnostics, asymmetric_summary = (
        add_asymmetric_directional_labels(
            features,
            horizons={"1h": 60, "3h": 180},
            stop_multipliers=(1.0,),
            target_multipliers=(3.0,),
        )
    )
    label_columns.extend(asymmetric_labels)

    for column in feature_columns + label_columns:
        features[column] = pd.to_numeric(features[column], errors="coerce").astype("float32")
    for column in asymmetric_diagnostics:
        if "_state_" in column:
            features[column] = features[column].astype("int8")
        else:
            features[column] = pd.to_numeric(features[column], errors="coerce").astype("float32")

    output_path = output_dir / "dmpe_BTCUSD_1m_features.parquet"
    features.to_parquet(
        output_path, index=False, engine="pyarrow", compression="zstd", row_group_size=100_000
    )

    schema = {
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "rows": int(len(features)),
        "time_semantics": (
            "Source exchange candle timestamp denotes minute OPEN. Feature rows are timestamped "
            "at completed candle CLOSE (source timestamp + 1 minute); all features are available "
            "no earlier than this close timestamp."
        ),
        "first_time_utc": pd.to_datetime(features["time"], utc=True).min().isoformat(),
        "last_time_utc": pd.to_datetime(features["time"], utc=True).max().isoformat(),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "label_count": len(label_columns),
        "label_columns": label_columns,
        "diagnostic_count": len(asymmetric_diagnostics),
        "diagnostic_columns": asymmetric_diagnostics,
        "continuous_label_policy": (
            "Forward return and future max/min excursion columns are evaluation targets only. "
            "Never include target_* columns in model inputs."
        ),
        "asymmetric_label_policy": (
            "Only two confirmatory horizons: 1h and 3h. Separate long and short first-touch "
            "targets, 1x horizon-aligned trailing RV stop / 3x target. +1=target first, "
            "-1=stop first, 0=neither, state=2=same-minute ambiguous (exclude), "
            "state=-2=invalid/incomplete. These are signal-close reference labels, not maker-fill labels."
        ),
        "asymmetric_barrier_configuration": {
            "stop_multipliers": [1.0],
            "target_multipliers": [3.0],
            "reward_risk_ratio_before_costs": 3.0,
            "horizons_minutes": {"1h": 60, "3h": 180},
            "volatility_columns": {
                "1h": "realized_vol_60m_pct",
                "3h": "realized_vol_180m_pct",
            },
            "policy": "Frozen research-only two-horizon experiment; 15m excluded."
        },
        "hourly_data_availability": (
            "Funding and OI values become eligible only after source timestamp + 1h."
        ),
        "gap_policy": (
            "Minute grid is explicit; OHLCV is not forward-filled. Rolling descriptors require "
            "complete lookback windows. Labels are null when their future window crosses a missing candle."
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
        "diagnostics": len(asymmetric_diagnostics),
        "output_file": str(output_path),
        "asymmetric_class_counts": asymmetric_summary,
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
