"""Exploratory BTC first-touch event-label balance diagnostics.

This script tests label distributions only. It does not fit a predictive model,
select a trading threshold, alter confirmatory asymmetric labels, or approve
trading. Samples are UTC-aligned and one per horizon to avoid overlapping
forward outcome windows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dmpe.labels import add_barrier_labels  # noqa: E402

HORIZONS = {"1h": 60, "3h": 180}
MULTIPLIERS = (0.5, 0.75, 1.0, 1.5, 2.0)
VOL_COLUMNS = {"1h": "realized_vol_60m_pct", "3h": "realized_vol_180m_pct"}
STATES = {
    -1: "down_first",
    0: "no_touch",
    1: "up_first",
    2: "ambiguous_same_minute",
    -2: "invalid_or_incomplete",
}


def summarize_state_counts(states: pd.Series | np.ndarray) -> dict:
    values = pd.to_numeric(pd.Series(states), errors="coerce").to_numpy()
    counts = {name: int(np.sum(values == code)) for code, name in STATES.items()}
    total = int(len(values))
    valid = counts["down_first"] + counts["no_touch"] + counts["up_first"] + counts["ambiguous_same_minute"]
    directional = counts["down_first"] + counts["up_first"]
    directional_balance = (
        min(counts["down_first"], counts["up_first"])
        / max(counts["down_first"], counts["up_first"])
        if directional and max(counts["down_first"], counts["up_first"]) else None
    )
    return {
        "samples": total,
        **counts,
        "valid_outcomes": valid,
        "directional_events": directional,
        "directional_event_rate_of_valid": directional / valid if valid else None,
        "no_touch_rate_of_valid": counts["no_touch"] / valid if valid else None,
        "ambiguous_rate_of_valid": counts["ambiguous_same_minute"] / valid if valid else None,
        "up_down_count_balance_min_over_max": directional_balance,
    }


def chronological_partition_masks(length: int) -> dict[str, np.ndarray]:
    """70/15/15 chronological partitions with one-sample boundary purges."""
    if length < 10:
        raise ValueError("At least 10 horizon-spaced samples are required.")
    train_cut = int(length * 0.70)
    validation_cut = int(length * 0.85)
    masks = {
        "train": np.arange(length) < max(train_cut - 1, 0),
        "validation": (np.arange(length) > train_cut) & (np.arange(length) < max(validation_cut - 1, 0)),
        "test": np.arange(length) > validation_cut,
    }
    return masks


def evaluate_event_grid(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {"time", "high", "low", "close", *VOL_COLUMNS.values()}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = frame[list(required)].copy()
    data["time"] = pd.to_datetime(data["time"], utc=True, errors="coerce")
    if data["time"].isna().any():
        raise ValueError("Feature dataset contains invalid timestamps.")
    data = data.sort_values("time").reset_index(drop=True)
    if data["time"].duplicated().any():
        raise ValueError("Feature dataset contains duplicate timestamps.")
    if len(data) > 1 and not data["time"].diff().dropna().eq(pd.Timedelta(minutes=1)).all():
        raise ValueError("Expected explicit 1-minute timestamp grid; gaps must remain as NaN rows.")

    labelled, label_cols, diagnostic_cols, label_report = add_barrier_labels(
        data,
        horizons=HORIZONS,
        multipliers=MULTIPLIERS,
        volatility_columns=VOL_COLUMNS,
    )
    rows: list[dict] = []
    for horizon, minutes in HORIZONS.items():
        epoch_minutes = labelled["time"].astype("int64") // 60_000_000_000
        cadence = (epoch_minutes % minutes).eq(0)
        sampled = labelled.loc[cadence].reset_index(drop=True)
        masks = chronological_partition_masks(len(sampled))
        for multiple in MULTIPLIERS:
            tag = f"{multiple:g}".replace(".", "p")
            state_col = f"barrier_state_{horizon}_{tag}rv"
            label_col = f"barrier_label_{horizon}_{tag}rv"
            bars_col = f"barrier_bars_{horizon}_{tag}rv"
            for partition, mask in masks.items():
                subset = sampled.loc[mask]
                summary = summarize_state_counts(subset[state_col])
                event_bars = pd.to_numeric(subset[bars_col], errors="coerce").dropna()
                rows.append({
                    "horizon": horizon,
                    "barrier_multiple": multiple,
                    "partition": partition,
                    "start_utc": subset["time"].min().isoformat() if len(subset) else None,
                    "end_utc": subset["time"].max().isoformat() if len(subset) else None,
                    **summary,
                    "median_bars_to_first_touch": float(event_bars.median()) if len(event_bars) else None,
                    "mean_bars_to_first_touch": float(event_bars.mean()) if len(event_bars) else None,
                    "label_column": label_col,
                })
    report = {
        "project": "DMPE",
        "purpose": "Exploratory event-label distribution diagnostics only; not a model or trading test.",
        "source_rows": int(len(data)),
        "source_start_utc": data["time"].min().isoformat(),
        "source_end_utc": data["time"].max().isoformat(),
        "horizons_minutes": HORIZONS,
        "barrier_multipliers": list(MULTIPLIERS),
        "volatility_columns": VOL_COLUMNS,
        "sampling": "One sample per horizon, UTC aligned; forward outcome windows do not overlap.",
        "partition_policy": "Chronological 70/15/15; one horizon-spaced sample purged at each partition boundary.",
        "state_semantics": STATES,
        "labeling_implementation": "Existing symmetric first-touch labeler; same-minute dual touches remain ambiguous; incomplete/gapped future windows remain invalid.",
        "confirmatory_labels_modified": False,
        "prospective_holdout_accessed": False,
        "trading_approval": False,
        "per_label_diagnostics": label_report,
    }
    return pd.DataFrame(rows), report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-file", required=True)
    parser.add_argument("--output-dir", default="event_label_output")
    args = parser.parse_args()
    source = Path(args.feature_file)
    frame = pd.read_parquet(
        source,
        columns=["time", "high", "low", "close", *VOL_COLUMNS.values()],
    )
    results, report = evaluate_event_grid(frame)
    report["feature_file"] = str(source)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "event_label_diagnostics.csv", index=False)
    (output / "event_label_diagnostics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(results.to_string(index=False))
    print(json.dumps({
        "output_dir": str(output),
        "rows": int(len(results)),
        "purpose": report["purpose"],
        "confirmatory_labels_modified": False,
        "trading_approval": False,
    }, indent=2))


if __name__ == "__main__":
    main()
