"""Exact-close join Binance proxy flow features to the candle feature dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-file", required=True)
    parser.add_argument("--flow-file", required=True)
    parser.add_argument("--output-dir", default="binance_merged")
    args = parser.parse_args()

    features = pd.read_parquet(args.feature_file)
    flow = pd.read_parquet(args.flow_file)
    for name, frame in (("features", features), ("flow", flow)):
        if "time" not in frame:
            raise ValueError(f"{name} file is missing the UTC close timestamp column 'time'")
        frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
        if frame["time"].isna().any() or frame["time"].duplicated().any():
            raise ValueError(f"{name} timestamps must be valid and unique")

    flow_columns = [column for column in flow.columns if column != "time"]
    collisions = set(flow_columns).intersection(features.columns)
    if collisions:
        raise ValueError(f"Refusing to overwrite existing feature columns: {sorted(collisions)}")

    merged = features.merge(flow, on="time", how="left", validate="one_to_one", sort=False)
    # Missing archive coverage stays missing. No as-of join or forward-fill is allowed.
    observed = merged["binance_aggtrade_feed_observed"].fillna(0).astype("int8")
    merged["binance_aggtrade_feed_observed"] = observed
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "dmpe_BTCUSD_1m_features_with_binance_flow.parquet"
    merged.to_parquet(output_path, index=False, compression="zstd")

    matched = int(observed.sum())
    report = {
        "rows": int(len(merged)),
        "feature_rows_with_binance_archive_coverage": matched,
        "coverage_share": float(matched / len(merged)) if len(merged) else None,
        "flow_columns": flow_columns,
        "join": "left exact UTC close timestamp; one-to-one; no as-of join; no forward-fill",
        "missingness_policy": "Unmatched archive minutes remain null; observed flag is 0.",
        "output_file": str(output_path),
    }
    (out_dir / "binance_flow_join_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
