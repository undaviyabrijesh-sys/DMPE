"""Repeated expanding-window walk-forward confirmation for DMPE.

Each fold uses an earlier training period, a separate validation period for
confidence-threshold selection, and a later untouched test period. Test
windows are not reused for threshold or model selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_baselines import (  # noqa: E402
    HORIZONS,
    MULTIPLIERS,
    _multiclass_metrics,
    _predict_direction,
    _signal_metrics,
    choose_confidence_threshold,
)

VALIDATION_STARTS = (
    "2025-07-01T00:00:00Z",
    "2025-11-01T00:00:00Z",
    "2026-03-01T00:00:00Z",
    "2026-07-01T00:00:00Z",
)
VALIDATION_DAYS = 60
TEST_DAYS = 60
MIN_TRAIN_ROWS = 500
MIN_VALIDATION_ROWS = 100
MIN_TEST_ROWS = 100


def make_fold_masks(times: pd.Series, horizon_minutes: int) -> list[dict[str, Any]]:
    """Make expanding train / 60d validation / 60d test folds with purging."""
    times = pd.to_datetime(times, utc=True)
    embargo = pd.Timedelta(minutes=horizon_minutes)
    folds = []
    for fold_number, raw_val_start in enumerate(VALIDATION_STARTS, start=1):
        val_start = pd.Timestamp(raw_val_start)
        test_start = val_start + pd.Timedelta(days=VALIDATION_DAYS)
        test_end = test_start + pd.Timedelta(days=TEST_DAYS)
        train_mask = times < val_start - embargo
        validation_mask = (
            (times >= val_start + embargo)
            & (times < test_start - embargo)
        )
        test_mask = (
            (times >= test_start + embargo)
            & (times < test_end)
        )
        folds.append({
            "fold": fold_number,
            "train_mask": train_mask.to_numpy(),
            "validation_mask": validation_mask.to_numpy(),
            "test_mask": test_mask.to_numpy(),
            "nominal_validation_start": val_start.isoformat(),
            "nominal_test_start": test_start.isoformat(),
            "nominal_test_end": test_end.isoformat(),
            "embargo_minutes": horizon_minutes,
        })
    return folds


def class_counts(y: np.ndarray) -> dict[str, int]:
    return {str(int(c)): int((y == c).sum()) for c in (-1, 0, 1)}


def score_variant(
    sample: pd.DataFrame,
    y: np.ndarray,
    feature_columns: list[str],
    horizon_name: str,
    horizon_minutes: int,
    multiplier: float,
) -> tuple[dict[str, Any], list[pd.DataFrame]]:
    times = pd.to_datetime(sample["time"], utc=True)
    x = sample[feature_columns].apply(pd.to_numeric, errors="coerce").astype("float32")
    fold_results = []
    signal_frames = []
    combined_y: list[np.ndarray] = []
    combined_predictions: list[np.ndarray] = []
    combined_probabilities: list[np.ndarray] = []
    combined_classes: np.ndarray | None = None
    combined_signals: list[np.ndarray] = []
    combined_signal_y: list[np.ndarray] = []

    for fold in make_fold_masks(times, horizon_minutes):
        train_idx = np.flatnonzero(fold["train_mask"])
        val_idx = np.flatnonzero(fold["validation_mask"])
        test_idx = np.flatnonzero(fold["test_mask"])
        fold_row: dict[str, Any] = {
            "fold": fold["fold"],
            "nominal_validation_start": fold["nominal_validation_start"],
            "nominal_test_start": fold["nominal_test_start"],
            "nominal_test_end": fold["nominal_test_end"],
            "embargo_minutes": horizon_minutes,
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(val_idx)),
            "test_rows": int(len(test_idx)),
        }
        if (
            len(train_idx) < MIN_TRAIN_ROWS
            or len(val_idx) < MIN_VALIDATION_ROWS
            or len(test_idx) < MIN_TEST_ROWS
        ):
            fold_row["status"] = "skipped_insufficient_partition"
            fold_results.append(fold_row)
            continue

        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
        if set(np.unique(y_train)) != {-1, 0, 1}:
            fold_row["status"] = "skipped_training_partition_missing_class"
            fold_row["train_class_counts"] = class_counts(y_train)
            fold_results.append(fold_row)
            continue

        train_non_null = x.iloc[train_idx].notna().any(axis=0).to_numpy()
        usable_features = [
            col for col, has_values in zip(x.columns, train_non_null) if bool(has_values)
        ]
        if not usable_features:
            fold_row["status"] = "skipped_no_training_features"
            fold_results.append(fold_row)
            continue

        model = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=160,
                max_leaf_nodes=15,
                min_samples_leaf=60,
                l2_regularization=5.0,
                early_stopping=False,
                random_state=42,
            )),
        ])
        model.fit(x.iloc[train_idx][usable_features], y_train)
        val_prob = model.predict_proba(x.iloc[val_idx][usable_features])
        val_pred = model.predict(x.iloc[val_idx][usable_features])
        threshold_selection = choose_confidence_threshold(y_val, val_prob, model.classes_)
        test_prob = model.predict_proba(x.iloc[test_idx][usable_features])
        test_pred = model.predict(x.iloc[test_idx][usable_features])

        fold_row.update({
            "status": "completed",
            "train_class_counts": class_counts(y_train),
            "validation_class_counts": class_counts(y_val),
            "test_class_counts": class_counts(y_test),
            "feature_count_used": len(usable_features),
            "validation_metrics": _multiclass_metrics(y_val, val_pred, val_prob, model.classes_),
            "test_metrics": _multiclass_metrics(y_test, test_pred, test_prob, model.classes_),
            "threshold_selection_on_validation": threshold_selection,
        })
        chosen = threshold_selection["selected_threshold"]
        if chosen is not None:
            test_signals = _predict_direction(test_prob, model.classes_, chosen)
            signal_metrics = _signal_metrics(y_test, test_signals)
            fold_row["test_confidence_gated_signals"] = {
                "threshold_selected_on_validation": chosen,
                **signal_metrics,
            }
            signal_rows = np.flatnonzero(test_signals != 0)
            if len(signal_rows):
                aligned = np.zeros((len(test_prob), 3), dtype="float64")
                map_col = {-1: 0, 0: 1, 1: 2}
                for idx, cls in enumerate(model.classes_):
                    aligned[:, map_col[int(cls)]] = test_prob[:, idx]
                signal_frames.append(pd.DataFrame({
                    "time": times.iloc[test_idx[signal_rows]].to_numpy(),
                    "fold": fold["fold"],
                    "horizon": horizon_name,
                    "barrier_multiple": multiplier,
                    "signal": test_signals[signal_rows],
                    "true_label": y_test[signal_rows],
                    "p_down": aligned[signal_rows, 0],
                    "p_flat": aligned[signal_rows, 1],
                    "p_up": aligned[signal_rows, 2],
                    "validation_threshold": chosen,
                }))
            combined_signals.append(test_signals)
            combined_signal_y.append(y_test)
        else:
            fold_row["test_confidence_gated_signals"] = {
                "status": "inactive_no_validation_threshold_met_minimum_signal_count",
                "signals": 0,
                "coverage": 0.0,
            }
            # Keep inactive folds in the denominator of combined coverage.
            combined_signals.append(np.zeros(len(y_test), dtype="int8"))
            combined_signal_y.append(y_test)

        combined_y.append(y_test)
        combined_predictions.append(test_pred)
        combined_probabilities.append(test_prob)
        combined_classes = model.classes_
        fold_results.append(fold_row)

    completed = [row for row in fold_results if row["status"] == "completed"]
    variant: dict[str, Any] = {
        "horizon": horizon_name,
        "horizon_minutes": horizon_minutes,
        "barrier_multiple": multiplier,
        "sample_rows": int(len(sample)),
        "folds": fold_results,
        "completed_folds": len(completed),
    }
    if not completed or combined_classes is None:
        variant["status"] = "no_completed_folds"
        return variant, signal_frames

    overall_y = np.concatenate(combined_y)
    overall_pred = np.concatenate(combined_predictions)
    overall_prob = np.vstack(combined_probabilities)
    variant["status"] = "completed"
    variant["combined_untouched_test_metrics"] = _multiclass_metrics(
        overall_y, overall_pred, overall_prob, combined_classes
    )
    if combined_signals:
        variant["combined_confidence_gated_test_signals"] = {
            **_signal_metrics(
                np.concatenate(combined_signal_y),
                np.concatenate(combined_signals),
            ),
            "folds_with_selected_threshold": sum(
                row.get("threshold_selection_on_validation", {}).get("selected_threshold") is not None
                for row in completed
            ),
        }
    else:
        variant["combined_confidence_gated_test_signals"] = {
            "signals": 0,
            "coverage": 0.0,
            "status": "no_directional_signals",
        }
    return variant, signal_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated DMPE expanding-window walk-forward tests")
    parser.add_argument("--feature-file", default="stage1_input/processed/dmpe_BTCUSD_1m_features.parquet")
    parser.add_argument("--schema-file", default="stage1_input/processed/feature_schema.json")
    parser.add_argument("--output-dir", default="stage3_output")
    parser.add_argument("--horizons", default="15m,1h,3h")
    parser.add_argument("--multipliers", default="0.75,1.0,1.5")
    args = parser.parse_args()

    feature_path = Path(args.feature_file)
    schema_path = Path(args.schema_file)
    if not feature_path.exists() or not schema_path.exists():
        raise SystemExit("Feature Parquet and feature_schema.json must both exist.")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    feature_columns = list(dict.fromkeys(schema.get("feature_columns", [])))
    if not feature_columns:
        raise SystemExit("Feature schema has no point-in-time feature columns.")

    horizons = [part.strip() for part in args.horizons.split(",") if part.strip()]
    multipliers = [float(part.strip()) for part in args.multipliers.split(",") if part.strip()]
    if any(value not in HORIZONS for value in horizons):
        raise SystemExit(f"Unsupported horizons: {horizons}")
    if any(value not in MULTIPLIERS for value in multipliers):
        raise SystemExit(f"Unsupported barrier multipliers: {multipliers}")

    required_columns = ["time", *feature_columns]
    for horizon in horizons:
        for multiplier in multipliers:
            tag = f"{multiplier:g}".replace(".", "p")
            required_columns.extend([
                f"barrier_label_{horizon}_{tag}rv",
                f"barrier_state_{horizon}_{tag}rv",
            ])
    required_columns = list(dict.fromkeys(required_columns))
    frame = pd.read_parquet(feature_path, columns=required_columns)
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    if frame["time"].isna().any():
        raise SystemExit("Invalid feature timestamps detected.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_variants = []
    all_signal_frames = []

    for horizon in horizons:
        horizon_minutes = HORIZONS[horizon]
        minute_epoch = frame["time"].astype("int64") // 60_000_000_000
        cadence = (minute_epoch % horizon_minutes) == 0
        for multiplier in multipliers:
            tag = f"{multiplier:g}".replace(".", "p")
            label_col = f"barrier_label_{horizon}_{tag}rv"
            state_col = f"barrier_state_{horizon}_{tag}rv"
            y_series = pd.to_numeric(frame[label_col], errors="coerce")
            state = pd.to_numeric(frame[state_col], errors="coerce")
            valid = cadence & y_series.notna() & state.isin([-1, 0, 1])
            sample = frame.loc[valid, ["time", *feature_columns]].copy().reset_index(drop=True)
            y = y_series.loc[valid].astype("int8").to_numpy()
            print(
                f"Walk-forward: {horizon}, {multiplier:g}x RV, {len(sample):,} samples",
                flush=True,
            )
            result, signal_frames = score_variant(
                sample, y, feature_columns, horizon, horizon_minutes, multiplier
            )
            all_variants.append(result)
            all_signal_frames.extend(signal_frames)

    report = {
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "project": "DMPE",
        "purpose": "Repeated expanding-window out-of-sample confirmation; no live trading approval.",
        "feature_dataset": str(feature_path),
        "feature_count": len(feature_columns),
        "fold_design": {
            "type": "expanding training window with validation and test blocks",
            "validation_starts_utc": list(VALIDATION_STARTS),
            "validation_days": VALIDATION_DAYS,
            "test_days": TEST_DAYS,
            "embargo_each_boundary_minutes": "one full prediction horizon",
            "threshold_policy": "validation-only confidence threshold; untouched test metrics per fold",
            "sample_policy": "one UTC-aligned sample per prediction horizon to avoid overlapping labels",
        },
        "warnings": [
            "The folds are sequential and expanding; later training folds may include earlier folds' already-observed test history.",
            "The same predefined model and barrier variants are reported across folds; do not choose a model using combined test metrics and call that same score unbiased.",
            "Wilson intervals do not account for regime clustering; use block bootstrap and more forward data for inference.",
            "These are classification/event labels, not a fee-aware execution backtest.",
        ],
        "results": all_variants,
    }
    (output_dir / "walk_forward_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    pd.json_normalize(all_variants).to_csv(output_dir / "walk_forward_summary.csv", index=False)
    if all_signal_frames:
        pd.concat(all_signal_frames, ignore_index=True).to_csv(
            output_dir / "walk_forward_test_signals.csv", index=False
        )
    print(json.dumps({
        "output_dir": str(output_dir),
        "variants": len(all_variants),
        "completed_variants": sum(row.get("status") == "completed" for row in all_variants),
        "results": [{
            "horizon": row["horizon"],
            "barrier_multiple": row["barrier_multiple"],
            "completed_folds": row.get("completed_folds"),
            "combined_test": row.get("combined_untouched_test_metrics"),
            "combined_signals": row.get("combined_confidence_gated_test_signals"),
            "folds": [{
                "fold": fold["fold"],
                "status": fold["status"],
                "test_metrics": fold.get("test_metrics"),
                "test_signals": fold.get("test_confidence_gated_signals"),
            } for fold in row.get("folds", [])],
        } for row in all_variants],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
