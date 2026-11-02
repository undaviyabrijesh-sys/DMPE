"""Join collected live order-flow features to the candle feature table without lookahead."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dmpe.orderflow import attach_orderflow_features  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-file", required=True)
    parser.add_argument("--orderflow-file", required=True)
    parser.add_argument("--schema-file", required=True)
    parser.add_argument("--output-dir", default="orderflow_merged")
    args = parser.parse_args()

    feature_path = Path(args.feature_file)
    flow_path = Path(args.orderflow_file)
    schema_path = Path(args.schema_file)
    for path in (feature_path, flow_path, schema_path):
        if not path.exists():
            raise SystemExit(f"Required input not found: {path}")

    features = pd.read_parquet(feature_path)
    flow = pd.read_parquet(flow_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    merged, added_columns = attach_orderflow_features(features, flow)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "dmpe_BTCUSD_1m_candle_orderflow_features.parquet"
    merged.to_parquet(output_file, index=False, compression="zstd", row_group_size=100_000)

    schema["feature_columns"] = list(dict.fromkeys(
        [*schema.get("feature_columns", []), *added_columns]
    ))
    schema["feature_count"] = len(schema["feature_columns"])
    schema["live_orderflow_policy"] = {
        "join": "Exact UTC minute-close timestamp; no as-of join or forward-fill.",
        "availability": "Only rows from actual live capture have event-derived measurements.",
        "missingness": "Missing feed/event minutes remain NaN; rolling descriptors require complete windows.",
        "added_columns": added_columns,
        "source_file": str(flow_path),
        "warning": "Use a chronological split after the capture start. Never assume pre-capture rows contain historical order-flow data."
    }
    schema_out = output_dir / "feature_schema_with_orderflow.json"
    schema_out.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    report = {
        "rows": int(len(merged)),
        "live_orderflow_rows": int(len(flow)),
        "first_orderflow_time": pd.to_datetime(flow["time"], utc=True).min().isoformat() if len(flow) else None,
        "last_orderflow_time": pd.to_datetime(flow["time"], utc=True).max().isoformat() if len(flow) else None,
        "added_feature_count": len(added_columns),
        "rows_with_trade_feed": int(merged.get("of_trade_feed_observed", pd.Series(dtype=float)).fillna(0).sum()),
        "rows_with_l2_feed": int(merged.get("of_l2_feed_observed", pd.Series(dtype=float)).fillna(0).sum()),
        "output_file": str(output_file),
        "schema_file": str(schema_out),
        "no_future_fill": True,
    }
    (output_dir / "orderflow_merge_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
