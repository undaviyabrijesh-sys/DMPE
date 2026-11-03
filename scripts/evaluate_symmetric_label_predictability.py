"""Exploratory train/validation-only predictability screen for symmetric BTC event labels.

The historical period after 2026-05-30 18:00 UTC is deliberately excluded
from this experiment. No historical test scores, P&L, threshold selection or
trading decisions are produced. The prospective holdout remains untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dmpe.labels import add_barrier_labels  # noqa: E402

HORIZONS = {"1h": 60, "3h": 180}
MULTIPLIERS = (0.5, 0.75, 1.0, 1.5, 2.0)
VOL_COLUMNS = {"1h": "realized_vol_60m_pct", "3h": "realized_vol_180m_pct"}
CLASSES = np.array([-1, 0, 1], dtype="int8")
SIGNAL_CUTOFFS = {
    "1h": pd.Timestamp("2026-05-30T18:00:00Z"),
    "3h": pd.Timestamp("2026-05-30T15:00:00Z"),
}


def chronological_train_validation(
    sample: pd.DataFrame, horizon_minutes: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    if len(sample) < 300:
        raise ValueError(f"Insufficient samples: {len(sample)}")
    times = pd.to_datetime(sample["time"], utc=True)
    cut_1 = times.iloc[int(len(sample) * 0.70)]
    cut_2 = times.iloc[int(len(sample) * 0.85)]
    embargo = pd.Timedelta(minutes=horizon_minutes)
    train_mask = times < cut_1 - embargo
    val_mask = (times >= cut_1 + embargo) & (times < cut_2 - embargo)
    # The final 15% and all data after SIGNAL_CUTOFF are not scored or fitted.
    train = np.flatnonzero(train_mask.to_numpy())
    validation = np.flatnonzero(val_mask.to_numpy())
    if min(len(train), len(validation)) == 0:
        raise ValueError("Empty chronological train or validation partition.")
    return train, validation, {
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "reserved_unscored_rows": int(len(sample) - len(train) - len(validation)),
        "train_start_utc": times.iloc[train[0]].isoformat(),
        "train_end_utc": times.iloc[train[-1]].isoformat(),
        "validation_start_utc": times.iloc[validation[0]].isoformat(),
        "validation_end_utc": times.iloc[validation[-1]].isoformat(),
        "nominal_cut_1_utc": cut_1.isoformat(),
        "nominal_cut_2_utc": cut_2.isoformat(),
        "embargo_minutes_each_side": horizon_minutes,
    }


def metrics(y: np.ndarray, prediction: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> dict:
    aligned = np.zeros((len(y), 3), dtype="float64")
    positions = {-1: 0, 0: 1, 1: 2}
    for col, value in enumerate(classes):
        aligned[:, positions[int(value)]] = probabilities[:, col]
    one_hot = np.column_stack([(y == value).astype(float) for value in CLASSES])
    return {
        "rows": int(len(y)),
        "class_counts": {str(int(c)): int((y == c).sum()) for c in CLASSES},
        "predicted_class_counts": {str(int(c)): int((prediction == c).sum()) for c in CLASSES},
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, labels=CLASSES, average="macro", zero_division=0)),
        "multiclass_log_loss": float(log_loss(y, aligned, labels=CLASSES)),
        "multiclass_brier_score": float(np.mean(np.sum((aligned - one_hot) ** 2, axis=1))),
        "confusion_matrix_minus1_0_plus1": confusion_matrix(y, prediction, labels=CLASSES).tolist(),
    }


def fit_and_score(x_train: pd.DataFrame, y_train: np.ndarray, x_val: pd.DataFrame, y_val: np.ndarray) -> dict:
    if set(np.unique(y_train)) != set(CLASSES.tolist()) or set(np.unique(y_val)) != set(CLASSES.tolist()):
        return {"status": "skipped_missing_class_in_train_or_validation"}

    # Prior-only benchmark.
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    dummy_prediction = dummy.predict(np.zeros((len(y_val), 1)))
    dummy_prob = dummy.predict_proba(np.zeros((len(y_val), 1)))
    result = {
        "status": "completed",
        "prior_baseline": metrics(y_val, dummy_prediction, dummy_prob, dummy.classes_),
    }

    for weighting in ("none", "balanced"):
        logistic = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(
                C=0.1, max_iter=500, solver="lbfgs",
                class_weight=None if weighting == "none" else "balanced",
                random_state=42,
            )),
        ])
        logistic.fit(x_train, y_train)
        prediction = logistic.predict(x_val)
        probabilities = logistic.predict_proba(x_val)
        result[f"logistic_{weighting}"] = metrics(y_val, prediction, probabilities, logistic.classes_)

        hist = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=160, max_leaf_nodes=15,
                min_samples_leaf=60, l2_regularization=5.0,
                early_stopping=False, random_state=42,
            )),
        ])
        weights = None if weighting == "none" else compute_sample_weight("balanced", y_train)
        hist.fit(x_train, y_train, classifier__sample_weight=weights)
        hist_prediction = hist.predict(x_val)
        hist_probabilities = hist.predict_proba(x_val)
        result[f"histgb_{weighting}"] = metrics(
            y_val, hist_prediction, hist_probabilities, hist.classes_
        )
    return result


def run(frame: pd.DataFrame, feature_columns: list[str]) -> dict:
    required = {"time", "high", "low", "close", *VOL_COLUMNS.values(), *feature_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = frame.copy()
    data["time"] = pd.to_datetime(data["time"], utc=True, errors="coerce")
    if data["time"].isna().any():
        raise ValueError("Invalid timestamps in feature data.")
    data = data.sort_values("time").reset_index(drop=True)

    # The last needed future bar is the latest signal cutoff plus its horizon.
    max_data_time = max(
        SIGNAL_CUTOFFS[h] + pd.Timedelta(minutes=minutes)
        for h, minutes in HORIZONS.items()
    )
    data = data.loc[data["time"] <= max_data_time].copy()
    labelled, _, _, _ = add_barrier_labels(
        data,
        horizons=HORIZONS,
        multipliers=MULTIPLIERS,
        volatility_columns=VOL_COLUMNS,
    )
    results = []
    cutoff_diagnostics = {}
    for horizon, minutes in HORIZONS.items():
        horizon_data = labelled.loc[labelled["time"] <= SIGNAL_CUTOFFS[horizon]].copy()
        epoch_minute = horizon_data["time"].astype("int64") // 60_000_000_000
        cadence = epoch_minute.mod(minutes).eq(0)
        for multiple in MULTIPLIERS:
            tag = f"{multiple:g}".replace(".", "p")
            label_col = f"barrier_label_{horizon}_{tag}rv"
            state_col = f"barrier_state_{horizon}_{tag}rv"
            y_series = pd.to_numeric(horizon_data[label_col], errors="coerce")
            state = pd.to_numeric(horizon_data[state_col], errors="coerce")
            valid = cadence & y_series.notna() & state.isin([-1, 0, 1])
            sample = horizon_data.loc[valid, ["time", *feature_columns]].copy().reset_index(drop=True)
            y = y_series.loc[valid].astype("int8").to_numpy()
            train_idx, val_idx, split = chronological_train_validation(sample, minutes)
            x = sample[feature_columns].apply(pd.to_numeric, errors="coerce")
            train_non_null = x.iloc[train_idx].notna().any(axis=0)
            usable = train_non_null[train_non_null].index.tolist()
            if not usable:
                raise ValueError(f"No usable training features for {horizon}/{multiple:g}x.")
            x_train = x.iloc[train_idx][usable].astype("float32")
            x_val = x.iloc[val_idx][usable].astype("float32")
            y_train, y_val = y[train_idx], y[val_idx]
            score = fit_and_score(x_train, y_train, x_val, y_val)
            results.append({
                "horizon": horizon,
                "barrier_multiple": multiple,
                "label_column": label_col,
                "samples": int(len(sample)),
                "class_counts_all_pre_cutoff": {str(int(c)): int((y == c).sum()) for c in CLASSES},
                "train_class_counts": {str(int(c)): int((y_train == c).sum()) for c in CLASSES},
                "validation_class_counts": {str(int(c)): int((y_val == c).sum()) for c in CLASSES},
                "feature_count": len(usable),
                "split": split,
                **score,
            })
            state_values = pd.to_numeric(
                horizon_data[state_col], errors="coerce"
            )
            cutoff_diagnostics[state_col] = {
                "signal_rows_through_cutoff": int(len(state_values)),
                "up_first": int((state_values == 1).sum()),
                "down_first": int((state_values == -1).sum()),
                "no_touch": int((state_values == 0).sum()),
                "ambiguous_same_minute": int((state_values == 2).sum()),
                "invalid_or_incomplete": int((state_values == -2).sum()),
            }
    return {
        "project": "DMPE",
        "purpose": "Exploratory symmetric event-label predictability screen; not a trading test.",
        "signal_time_cutoffs_utc": {h: value.isoformat() for h, value in SIGNAL_CUTOFFS.items()},
        "data_policy": "Signal timestamps after each horizon cutoff are excluded. Only future bars needed to resolve labels up to the cutoff are retained. Final 15% of each pre-cutoff sample is reserved and unscored.",
        "holdout_policy": "No historical test metrics are produced. The prospective holdout is not accessed.",
        "horizons_minutes": HORIZONS,
        "barrier_multipliers": list(MULTIPLIERS),
        "volatility_columns": VOL_COLUMNS,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "model_families": ["prior baseline", "logistic regression", "histogram gradient boosting"],
        "weighting_variants": ["unweighted", "balanced class weights"],
        "metric_policy": "Report validation balanced accuracy, macro-F1, log loss and multiclass Brier; no threshold or barrier is selected automatically.",
        "label_diagnostics_through_cutoff_only": cutoff_diagnostics,
        "results": results,
        "trading_approval": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-file", required=True)
    parser.add_argument("--schema-file", required=True)
    parser.add_argument("--output-dir", default="symmetric_label_screen")
    args = parser.parse_args()
    schema = json.loads(Path(args.schema_file).read_text(encoding="utf-8"))
    feature_columns = list(dict.fromkeys(schema.get("feature_columns", [])))
    if not feature_columns:
        raise SystemExit("Feature schema has no feature_columns.")
    required = ["time", "high", "low", "close", *VOL_COLUMNS.values(), *feature_columns]
    frame = pd.read_parquet(args.feature_file, columns=list(dict.fromkeys(required)))
    report = run(frame, feature_columns)
    report["feature_file"] = str(args.feature_file)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "symmetric_label_predictability.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    pd.json_normalize(report["results"]).to_csv(
        output / "symmetric_label_predictability.csv", index=False
    )
    print(json.dumps({
        "output_dir": str(output),
        "variants": len(report["results"]),
        "signal_time_cutoffs": report["signal_time_cutoffs_utc"],
        "trading_approval": False,
    }, indent=2))


if __name__ == "__main__":
    main()
