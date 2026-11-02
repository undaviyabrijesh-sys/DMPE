"""Build a DMPE 1-minute feature, continuous-target and barrier-label dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dmpe.features import build_feature_frame  # noqa: E402
from dmpe.labels import add_asymmetric_directional_labels, add_barrier_labels  # noqa: E402


def parse_multipliers(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(token.strip()) for token in value.split(",") if token.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated positive numbers.") from exc
    if not values or any(not (number > 0) for number in values):
        raise argparse.ArgumentTypeError("Barrier multipliers must be positive.")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DMPE 1m feature and label dataset")
    parser.add_argument("--input-dir", default="data/raw/history")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument(
        "--barrier-multipliers", type=parse_multipliers, default=(0.75, 1.0, 1.5),
        help="Symmetric upper/lower horizon-aligned realized-volatility multiples to evaluate.",
    )
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
    features, barrier_labels, diagnostic_columns, barrier_summary = add_barrier_labels(
        features, multipliers=args.barrier_multipliers
    )
    label_columns.extend(barrier_labels)
    features, asymmetric_labels, asymmetric_diagnostics, asymmetric_summary = (
        add_asymmetric_directional_labels(
            features,
            horizons={"15m": 15, "1h": 60},
            stop_multipliers=(1.0,),
            target_multipliers=(3.0,),
        )
    )
    label_columns.extend(asymmetric_labels)
    diagnostic_columns.extend(asymmetric_diagnostics)

    # Compact model inputs and targets without downcasting retained raw OHLCV.
    for column in feature_columns + label_columns:
        features[column] = pd.to_numeric(features[column], errors="coerce").astype("float32")
    for column in diagnostic_columns:
        if column.startswith("barrier_state_"):
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
            "The source exchange candle timestamp denotes minute OPEN. Feature rows are "
            "timestamped at the completed candle CLOSE (source timestamp + 1 minute); "
            "all features are available no earlier than this close timestamp."
        ),
        "first_time_utc": pd.to_datetime(features["time"], utc=True).min().isoformat(),
        "last_time_utc": pd.to_datetime(features["time"], utc=True).max().isoformat(),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "label_count": len(label_columns),
        "label_columns": label_columns,
        "diagnostic_count": len(diagnostic_columns),
        "diagnostic_columns": diagnostic_columns,
        "continuous_label_policy": (
            "Forward return and future max/min excursion columns are evaluation targets only. "
            "Never include target_* columns in model inputs."
        ),
        "asymmetric_label_policy": (
            "Separate long and short first-touch outcomes. For long: target +3x horizon RV, "
            "stop -1x; for short: target -3x, stop +1x. +1=target first, -1=stop first, "
            "0=neither touched, state=2=same-minute ambiguous (exclude), state=-2=invalid/incomplete. "
            "The long and short targets are independent binary prediction tasks; do not collapse "
            "them into a single three-class label. The preregistered experiment covers 15m and 1h only."
        ),
        "asymmetric_barrier_configuration": {
            "stop_multipliers": [1.0],
            "target_multipliers": [3.0],
            "reward_risk_ratio_before_costs": 3.0,
            "horizons_minutes": {"15m": 15, "1h": 60, "3h": 180},
            "policy": "Research-only preregistered 1R stop / 3R target; not selected from test results."
        },
        "barrier_label_policy": (
            "barrier_label_* is +1 when the symmetric upper horizon-aligned realized-volatility "
            "barrier is touched first, -1 when the lower barrier is touched first, and 0 for no touch or same-minute "
            "ambiguous touch. The barrier_state_* diagnostic distinguishes these cases; "
            "state=2 is ambiguous and should be excluded from directional training."
        ),
        "barrier_configuration": {
            "volatility_scale": {
                "15m": "realized_vol_15m_pct",
                "1h": "realized_vol_60m_pct",
                "3h": "realized_vol_180m_pct",
            },
            "volatility_scale_definition": (
                "Trailing standard deviation of completed 1-minute log returns multiplied "
                "by sqrt(window), expressed as percent; horizon aligned and known at signal time."
            ),
            "horizons_minutes": {"15m": 15, "1h": 60, "3h": 180},
            "symmetric_multipliers": list(args.barrier_multipliers),
            "ambiguous_intraminute_order": "not guessed; state=2 and directional label=0",
        },
        "hourly_data_availability": (
            "Funding and OI values become eligible only after source timestamp + 1h 1m."
        ),
        "gap_policy": (
            "Minute grid is explicit; OHLCV is not forward-filled. Rolling descriptors require "
            "complete lookback windows. Continuous labels and barrier labels are null when "
            "their required future window crosses a missing candle."
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
        "diagnostics": len(diagnostic_columns),
        "output_file": str(output_path),
        "barrier_class_counts": barrier_summary,
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
