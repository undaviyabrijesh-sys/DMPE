"""Chronological, non-overlapping first-pass DMPE classifier evaluation.

This is a research baseline only. It never trains on forward targets as inputs
and never uses the final test period for model fitting or threshold selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
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

CLASSES = np.array([-1, 0, 1], dtype="int8")
HORIZONS = {"15m": 15, "1h": 60, "3h": 180}
MULTIPLIERS = (0.75, 1.0, 1.5)
CONFIDENCE_THRESHOLDS = tuple(float(x) for x in np.arange(0.45, 0.801, 0.05))
MIN_VALIDATION_SIGNALS = 30


def _predict_direction(probabilities: np.ndarray, classes: np.ndarray, threshold: float) -> np.ndarray:
    """Emit +/-1 only when directional class probability clears threshold."""
    aligned = np.zeros((probabilities.shape[0], 3), dtype="float64")
    class_to_col = {-1: 0, 0: 1, 1: 2}
    for idx, class_value in enumerate(classes):
        aligned[:, class_to_col[int(class_value)]] = probabilities[:, idx]
    p_down, p_flat, p_up = aligned.T
    prediction = np.zeros(len(aligned), dtype="int8")
    # A directional call must be the model's most likely class, not merely
    # higher than the opposite direction while FLAT remains more probable.
    up = (p_up >= threshold) & (p_up > p_down) & (p_up > p_flat)
    down = (p_down >= threshold) & (p_down > p_up) & (p_down > p_flat)
    prediction[up] = 1
    prediction[down] = -1
    return prediction




def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float] | None:
    """Approximate two-sided Wilson interval for a binomial proportion."""
    if trials <= 0:
        return None
    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denominator
    radius = (
        z / denominator
        * np.sqrt((p * (1.0 - p) / trials) + z2 / (4.0 * trials * trials))
    )
    return [max(0.0, float(center - radius)), min(1.0, float(center + radius))]


def _signal_metrics(y_true: np.ndarray, signals: np.ndarray) -> dict[str, Any]:
    active = signals != 0
    count = int(active.sum())
    if count == 0:
        precision = None
        wrong_direction = None
        neutral_outcome = None
        precision_ci = None
    else:
        actual = y_true[active]
        called = signals[active]
        successes = int((actual == called).sum())
        precision = successes / count
        precision_ci = wilson_interval(successes, count)
        wrong_direction = float(((actual != 0) & (actual == -called)).mean())
        neutral_outcome = float((actual == 0).mean())
    return {
        "signals": count,
        "coverage": float(active.mean()) if len(active) else 0.0,
        "correct_directional_share_of_signals": precision,
        "correct_share_wilson_95_ci": precision_ci,
        "wrong_direction_share_of_signals": wrong_direction,
        "no_barrier_touch_share_of_signals": neutral_outcome,
    }


def _multiclass_metrics(y_true: np.ndarray, pred: np.ndarray, probabilities: np.ndarray,
                        classes: np.ndarray) -> dict[str, Any]:
    aligned = np.zeros((len(y_true), 3), dtype="float64")
    class_to_col = {-1: 0, 0: 1, 1: 2}
    for idx, class_value in enumerate(classes):
        aligned[:, class_to_col[int(class_value)]] = probabilities[:, idx]
    one_hot = np.column_stack([(y_true == c).astype("float64") for c in CLASSES])
    brier = float(np.mean(np.sum((aligned - one_hot) ** 2, axis=1)))
    return {
        "rows": int(len(y_true)),
        "class_counts": {str(int(c)): int((y_true == c).sum()) for c in CLASSES},
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, labels=CLASSES, average="macro", zero_division=0)),
        "multiclass_log_loss": float(log_loss(y_true, aligned, labels=CLASSES)),
        "multiclass_brier_score": brier,
        "confusion_matrix_labels_minus1_0_plus1": confusion_matrix(
            y_true, pred, labels=CLASSES
        ).tolist(),
    }


def choose_confidence_threshold(y_validation: np.ndarray, probabilities: np.ndarray,
                                classes: np.ndarray) -> dict[str, Any]:
    """Select threshold on validation only; prefer precision with minimum signal count."""
    candidates = []
    for threshold in CONFIDENCE_THRESHOLDS:
        signals = _predict_direction(probabilities, classes, threshold)
        metrics = _signal_metrics(y_validation, signals)
        candidates.append({"threshold": threshold, **metrics})
    eligible = [row for row in candidates if row["signals"] >= MIN_VALIDATION_SIGNALS]
    if not eligible:
        return {
            "selected_threshold": None,
            "selection_reason": (
                f"No validation threshold produced {MIN_VALIDATION_SIGNALS} directional signals; "
                "the test-period strategy is marked inactive."
            ),
            "candidates": candidates,
        }
    selected = max(
        eligible,
        key=lambda row: (
            row["correct_directional_share_of_signals"],
            row["coverage"],
            -row["threshold"],
        ),
    )
    return {
        "selected_threshold": float(selected["threshold"]),
        "selection_reason": (
            "Highest validation directional precision among thresholds meeting the minimum "
            "signal-count requirement; test data were not used for selection."
        ),
        "selected_validation_metrics": selected,
        "candidates": candidates,
    }


def chronological_splits(
    sample: pd.DataFrame, horizon_minutes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    """70/15/15 chronological periods with horizon-sized embargoes around cutoffs."""
    if len(sample) < 300:
        raise ValueError(f"Only {len(sample)} valid non-overlapping samples; at least 300 are required.")
    times = pd.to_datetime(sample["time"], utc=True)
    cut_1 = times.iloc[int(len(sample) * 0.70)]
    cut_2 = times.iloc[int(len(sample) * 0.85)]
    embargo = pd.Timedelta(minutes=horizon_minutes)
    train_mask = times < (cut_1 - embargo)
    validation_mask = (times >= (cut_1 + embargo)) & (times < (cut_2 - embargo))
    test_mask = times >= (cut_2 + embargo)

    train = np.flatnonzero(train_mask.to_numpy())
    validation = np.flatnonzero(validation_mask.to_numpy())
    test = np.flatnonzero(test_mask.to_numpy())
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Chronological split produced an empty partition.")
    return train, validation, test, {
        "train_end": times.iloc[train[-1]].isoformat(),
        "validation_start": times.iloc[validation[0]].isoformat(),
        "validation_end": times.iloc[validation[-1]].isoformat(),
        "test_start": times.iloc[test[0]].isoformat(),
        "embargo_minutes_each_side_of_cut": horizon_minutes,
        "nominal_cut_1": cut_1.isoformat(),
        "nominal_cut_2": cut_2.isoformat(),
    }


def evaluate_one(
    frame: pd.DataFrame,
    feature_columns: list[str],
    horizon_name: str,
    horizon_minutes: int,
    multiplier: float,
    model_dir: Path,
) -> dict[str, Any]:
    tag = f"{multiplier:g}".replace(".", "p")
    label_column = f"barrier_label_{horizon_name}_{tag}rv"
    state_column = f"barrier_state_{horizon_name}_{tag}rv"
    if label_column not in frame or state_column not in frame:
        raise ValueError(f"Missing label/diagnostic columns: {label_column}, {state_column}")

    # Sample at exactly one observation per target horizon. This prevents
    # successive target windows from overlapping in the reported baseline.
    minute_epoch = pd.to_datetime(frame["time"], utc=True).astype("int64") // 60_000_000_000
    cadence_mask = (minute_epoch % horizon_minutes) == 0
    y_series = pd.to_numeric(frame[label_column], errors="coerce")
    state = pd.to_numeric(frame[state_column], errors="coerce")
    valid = cadence_mask & y_series.notna() & state.isin([-1, 0, 1])
    sample = frame.loc[valid, ["time", *feature_columns]].copy().reset_index(drop=True)
    y = y_series.loc[valid].astype("int8").to_numpy()

    result: dict[str, Any] = {
        "horizon": horizon_name,
        "horizon_minutes": horizon_minutes,
        "barrier_multiple": multiplier,
        "label_column": label_column,
        "sampling_policy": f"one sample every {horizon_minutes} minutes, UTC aligned",
        "samples_before_split": int(len(sample)),
        "sample_class_counts": {str(int(c)): int((y == c).sum()) for c in CLASSES},
    }
    if len(sample) < 300 or set(np.unique(y)) != set(CLASSES.tolist()):
        result["status"] = "skipped_insufficient_samples_or_classes"
        return result

    train_idx, val_idx, test_idx, split_info = chronological_splits(sample, horizon_minutes)
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    if set(np.unique(y_train)) != set(CLASSES.tolist()):
        result["status"] = "skipped_training_partition_missing_class"
        result["split"] = {
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(val_idx)),
            "test_rows": int(len(test_idx)),
            **split_info,
        }
        return result

    x = sample[feature_columns].apply(pd.to_numeric, errors="coerce")
    # Remove columns that were wholly missing in the training period only.
    train_non_null = x.iloc[train_idx].notna().any(axis=0).to_numpy()
    usable_features = [
        col for col, has_training_values in zip(x.columns, train_non_null)
        if bool(has_training_values)
    ]
    if not usable_features:
        result["status"] = "skipped_no_nonnull_training_features"
        return result
    x_train = x.iloc[train_idx][usable_features].astype("float32")
    x_val = x.iloc[val_idx][usable_features].astype("float32")
    x_test = x.iloc[test_idx][usable_features].astype("float32")

    dummy = DummyClassifier(strategy="prior")
    dummy_x_train = np.zeros((len(y_train), 1), dtype="float32")
    dummy_x_test = np.zeros((len(y_test), 1), dtype="float32")
    dummy.fit(dummy_x_train, y_train)
    dummy_pred = dummy.predict(dummy_x_test)
    dummy_prob = dummy.predict_proba(dummy_x_test)
    result["majority_prior_baseline"] = _multiclass_metrics(
        y_test, dummy_pred, dummy_prob, dummy.classes_
    )

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(
            C=0.1, max_iter=300, solver="lbfgs", class_weight=None,
            random_state=42,
        )),
    ])
    model.fit(x_train, y_train)
    val_prob = model.predict_proba(x_val)
    val_pred = model.predict(x_val)
    threshold_selection = choose_confidence_threshold(y_val, val_prob, model.classes_)

    test_prob = model.predict_proba(x_test)
    test_pred = model.predict(x_test)
    result["status"] = "completed"
    result["split"] = {
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "feature_count_used": len(usable_features),
        "features_used": usable_features,
        **split_info,
    }
    result["logistic_regression_validation"] = _multiclass_metrics(
        y_val, val_pred, val_prob, model.classes_
    )
    result["logistic_regression_test"] = _multiclass_metrics(
        y_test, test_pred, test_prob, model.classes_
    )
    result["logistic_validation_threshold_selection"] = threshold_selection

    logistic_threshold = threshold_selection["selected_threshold"]
    if logistic_threshold is not None:
        logistic_signals = _predict_direction(test_prob, model.classes_, logistic_threshold)
        result["logistic_test_confidence_gated_signals"] = {
            "threshold_selected_on_validation": logistic_threshold,
            **_signal_metrics(y_test, logistic_signals),
        }
    else:
        result["logistic_test_confidence_gated_signals"] = {
            "threshold_selected_on_validation": None,
            "status": "inactive_no_validation_threshold_met_minimum_signal_count",
            "signals": 0,
            "coverage": 0.0,
            "correct_directional_share_of_signals": None,
        }

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"dmpe_logistic_{horizon_name}_{tag}rv.joblib"
    joblib.dump({
        "pipeline": model,
        "feature_columns": usable_features,
        "classes": model.classes_.tolist(),
        "horizon": horizon_name,
        "barrier_multiple": multiplier,
        "label_column": label_column,
        "train_end_utc": split_info["train_end"],
        "validation_start_utc": split_info["validation_start"],
        "test_start_utc": split_info["test_start"],
        "confidence_threshold_selected_on_validation": logistic_threshold,
        "signal_policy": "direction must be argmax class and exceed validation-selected threshold",
        "validated_for_live_trading": False,
    }, model_path, compress=3)
    result["experimental_logistic_model_file"] = str(model_path)

    # Nonlinear benchmark with conservative tree depth/leaf size. This is also
    # fitted on train only; validation selects only the confidence gate.
    tree_model = Pipeline(steps=[
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
    tree_model.fit(x_train, y_train)
    tree_val_prob = tree_model.predict_proba(x_val)
    tree_val_pred = tree_model.predict(x_val)
    tree_threshold_selection = choose_confidence_threshold(
        y_val, tree_val_prob, tree_model.classes_
    )
    tree_test_prob = tree_model.predict_proba(x_test)
    tree_test_pred = tree_model.predict(x_test)
    result["hist_gradient_boosting_validation"] = _multiclass_metrics(
        y_val, tree_val_pred, tree_val_prob, tree_model.classes_
    )
    result["hist_gradient_boosting_test"] = _multiclass_metrics(
        y_test, tree_test_pred, tree_test_prob, tree_model.classes_
    )
    result["hist_gradient_boosting_validation_threshold_selection"] = tree_threshold_selection
    tree_threshold = tree_threshold_selection["selected_threshold"]
    if tree_threshold is not None:
        tree_signals = _predict_direction(tree_test_prob, tree_model.classes_, tree_threshold)
        result["hist_gradient_boosting_test_confidence_gated_signals"] = {
            "threshold_selected_on_validation": tree_threshold,
            **_signal_metrics(y_test, tree_signals),
        }
    else:
        result["hist_gradient_boosting_test_confidence_gated_signals"] = {
            "threshold_selected_on_validation": None,
            "status": "inactive_no_validation_threshold_met_minimum_signal_count",
            "signals": 0,
            "coverage": 0.0,
            "correct_directional_share_of_signals": None,
        }

    tree_path = model_dir / f"dmpe_histgb_{horizon_name}_{tag}rv.joblib"
    joblib.dump({
        "pipeline": tree_model,
        "feature_columns": usable_features,
        "classes": tree_model.classes_.tolist(),
        "horizon": horizon_name,
        "barrier_multiple": multiplier,
        "label_column": label_column,
        "train_end_utc": split_info["train_end"],
        "validation_start_utc": split_info["validation_start"],
        "test_start_utc": split_info["test_start"],
        "confidence_threshold_selected_on_validation": tree_threshold,
        "signal_policy": "direction must be argmax class and exceed validation-selected threshold",
        "validated_for_live_trading": False,
    }, tree_path, compress=3)
    result["experimental_hist_gradient_boosting_model_file"] = str(tree_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DMPE walk-forward classification baselines")
    parser.add_argument("--feature-file", default="stage1_input/processed/dmpe_BTCUSD_1m_features.parquet")
    parser.add_argument("--schema-file", default="stage1_input/processed/feature_schema.json")
    parser.add_argument("--output-dir", default="stage2_output")
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
        raise SystemExit("Feature schema does not contain feature_columns.")
    horizons = [value.strip() for value in args.horizons.split(",") if value.strip()]
    multipliers = [float(value.strip()) for value in args.multipliers.split(",") if value.strip()]
    for h in horizons:
        if h not in HORIZONS:
            raise SystemExit(f"Unsupported horizon {h}; choose from {sorted(HORIZONS)}.")
    for value in multipliers:
        if value not in MULTIPLIERS:
            raise SystemExit(f"Unsupported barrier multiplier {value}; choose from {MULTIPLIERS}.")

    all_columns = ["time", *feature_columns]
    for horizon in horizons:
        for multiplier in multipliers:
            tag = f"{multiplier:g}".replace(".", "p")
            all_columns.extend([
                f"barrier_label_{horizon}_{tag}rv",
                f"barrier_state_{horizon}_{tag}rv",
            ])
    all_columns = list(dict.fromkeys(all_columns))
    frame = pd.read_parquet(feature_path, columns=all_columns)
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    if frame["time"].isna().any():
        raise SystemExit("Feature dataset contains invalid timestamps.")

    output_dir = Path(args.output_dir)
    model_dir = output_dir / "models"
    evaluations = []
    for horizon in horizons:
        for multiplier in multipliers:
            print(f"Evaluating {horizon}, barrier {multiplier:g}x horizon RV ...", flush=True)
            evaluations.append(evaluate_one(
                frame, feature_columns, horizon, HORIZONS[horizon], multiplier, model_dir
            ))

    report = {
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "project": "DMPE",
        "purpose": "Initial exploratory holdout comparison only; no live-trading approval.",
        "selection_bias_warning": (
            "The 9 horizon/barrier combinations and two model families are compared on one "
            "holdout. Do not select a winner from these test results and treat it as unbiased; "
            "use the next rolling walk-forward experiment as a fresh confirmation."
        ),
        "confidence_interval_warning": (
            "Wilson intervals are binomial approximations and do not account for market-regime "
            "clustering or serial dependence. Use block-bootstrap intervals in subsequent tests."
        ),
        "feature_dataset": str(feature_path),
        "feature_count_in_schema": len(feature_columns),
        "training_policy": [
            "No target_* columns are model inputs.",
            "Sample one observation per target horizon to avoid overlapping label windows.",
            "Use chronological 70/15/15 train/validation/test periods.",
            "Purge and embargo one full target horizon at both split boundaries.",
            "Fit imputation, scaling and model only on the train partition.",
            "Select signal confidence thresholds on validation only.",
            "A directional output must be the highest-probability class and clear the threshold.",
            "Evaluate the final frozen models on the untouched test partition.",
            "Ambiguous same-minute touch and invalid/incomplete states are excluded."
        ],
        "results": evaluations,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "baseline_evaluation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    pd.json_normalize(evaluations).to_csv(output_dir / "baseline_evaluation.csv", index=False)
    print(json.dumps({
        "output_dir": str(output_dir),
        "models_completed": sum(row.get("status") == "completed" for row in evaluations),
        "models_skipped": sum(row.get("status", "").startswith("skipped") for row in evaluations),
        "results": [{
            "horizon": row["horizon"],
            "multiplier": row["barrier_multiple"],
            "status": row["status"],
            "samples": row["samples_before_split"],
            "prior_baseline_test": row.get("majority_prior_baseline"),
            "logistic_test": row.get("logistic_regression_test"),
            "logistic_signals": row.get("logistic_test_confidence_gated_signals"),
            "hist_gradient_boosting_test": row.get("hist_gradient_boosting_test"),
            "hist_gradient_boosting_signals": row.get("hist_gradient_boosting_test_confidence_gated_signals"),
        } for row in evaluations],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
