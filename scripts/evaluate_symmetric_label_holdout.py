"""One-time reserved historical holdout for the frozen 3h symmetric-label candidate.

Candidate fixed from the preceding train/validation screen:
3h, 1.0x trailing horizon RV, HistGradientBoosting with balanced sample weights.
No candidate/hyperparameter/threshold changes are allowed after this holdout is opened.
This is still historical research, not prospective confirmation or a trading test.
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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dmpe.labels import add_barrier_labels  # noqa: E402

HORIZON = "3h"
HORIZON_MINUTES = 180
MULTIPLIER = 1.0
SIGNAL_CUTOFF = pd.Timestamp("2026-05-30T15:00:00Z")
CLASSES = np.array([-1, 0, 1], dtype="int8")
BOOTSTRAP_REPS = 1_000


def chronological_holdout_indices(sample: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    if len(sample) < 300:
        raise ValueError(f"Insufficient samples: {len(sample)}")
    times = pd.to_datetime(sample["time"], utc=True)
    cut_1 = times.iloc[int(len(sample) * 0.70)]
    cut_2 = times.iloc[int(len(sample) * 0.85)]
    embargo = pd.Timedelta(minutes=HORIZON_MINUTES)
    # Refit on train + validation after the candidate is frozen, while preserving
    # the purge around the train/validation boundary and the final holdout boundary.
    fit_mask = (times < cut_2 - embargo) & (
        (times < cut_1 - embargo) | (times >= cut_1 + embargo)
    )
    holdout_mask = times >= cut_2 + embargo
    fit_idx = np.flatnonzero(fit_mask.to_numpy())
    holdout_idx = np.flatnonzero(holdout_mask.to_numpy())
    if min(len(fit_idx), len(holdout_idx)) == 0:
        raise ValueError("Empty fit or reserved holdout partition.")
    return fit_idx, holdout_idx, {
        "fit_rows": int(len(fit_idx)),
        "holdout_rows": int(len(holdout_idx)),
        "fit_start_utc": times.iloc[fit_idx[0]].isoformat(),
        "fit_end_utc": times.iloc[fit_idx[-1]].isoformat(),
        "holdout_start_utc": times.iloc[holdout_idx[0]].isoformat(),
        "holdout_end_utc": times.iloc[holdout_idx[-1]].isoformat(),
        "nominal_train_validation_cut_utc": cut_1.isoformat(),
        "nominal_holdout_cut_utc": cut_2.isoformat(),
        "embargo_minutes": HORIZON_MINUTES,
    }


def metric_pack(y: np.ndarray, pred: np.ndarray, prob: np.ndarray, classes: np.ndarray) -> dict:
    aligned = np.zeros((len(y), 3), dtype="float64")
    class_pos = {-1: 0, 0: 1, 1: 2}
    for col, value in enumerate(classes):
        aligned[:, class_pos[int(value)]] = prob[:, col]
    one_hot = np.column_stack([(y == value).astype(float) for value in CLASSES])
    return {
        "rows": int(len(y)),
        "class_counts": {str(int(c)): int((y == c).sum()) for c in CLASSES},
        "predicted_class_counts": {str(int(c)): int((pred == c).sum()) for c in CLASSES},
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)),
        "multiclass_log_loss": float(log_loss(y, aligned, labels=CLASSES)),
        "multiclass_brier_score": float(np.mean(np.sum((aligned - one_hot) ** 2, axis=1))),
        "confusion_matrix_minus1_0_plus1": confusion_matrix(y, pred, labels=CLASSES).tolist(),
    }


def day_block_bootstrap_differences(
    times: pd.Series,
    y: np.ndarray,
    model_pred: np.ndarray,
    baseline_pred: np.ndarray,
    reps: int = BOOTSTRAP_REPS,
    seed: int = 42,
) -> dict:
    day_keys = pd.to_datetime(times, utc=True).dt.strftime("%Y-%m-%d").to_numpy()
    unique_days = np.unique(day_keys)
    blocks = [np.flatnonzero(day_keys == day) for day in unique_days]
    rng = np.random.default_rng(seed)
    ba_diffs = []
    f1_diffs = []
    for _ in range(reps):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        idx = np.concatenate([blocks[j] for j in chosen])
        ba_diffs.append(
            balanced_accuracy_score(y[idx], model_pred[idx])
            - balanced_accuracy_score(y[idx], baseline_pred[idx])
        )
        f1_diffs.append(
            f1_score(y[idx], model_pred[idx], labels=CLASSES, average="macro", zero_division=0)
            - f1_score(y[idx], baseline_pred[idx], labels=CLASSES, average="macro", zero_division=0)
        )
    return {
        "method": "UTC-day block bootstrap; paired model-minus-prior-baseline differences",
        "replicates": reps,
        "day_blocks": int(len(unique_days)),
        "balanced_accuracy_difference_95pct_ci": [
            float(np.quantile(ba_diffs, 0.025)), float(np.quantile(ba_diffs, 0.975))
        ],
        "macro_f1_difference_95pct_ci": [
            float(np.quantile(f1_diffs, 0.025)), float(np.quantile(f1_diffs, 0.975))
        ],
    }


def evaluate(frame: pd.DataFrame, feature_columns: list[str]) -> dict:
    required = {"time", "high", "low", "close", "realized_vol_180m_pct", *feature_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = frame.copy()
    data["time"] = pd.to_datetime(data["time"], utc=True, errors="coerce")
    if data["time"].isna().any():
        raise ValueError("Invalid timestamps.")
    data = data.sort_values("time").reset_index(drop=True)
    data = data.loc[data["time"] <= SIGNAL_CUTOFF + pd.Timedelta(minutes=HORIZON_MINUTES)].copy()
    labelled, _, _, _ = add_barrier_labels(
        data,
        horizons={HORIZON: HORIZON_MINUTES},
        multipliers=(MULTIPLIER,),
        volatility_columns={HORIZON: "realized_vol_180m_pct"},
    )
    labelled = labelled.loc[labelled["time"] <= SIGNAL_CUTOFF].copy()
    epoch_minutes = labelled["time"].astype("int64") // 60_000_000_000
    label_col = "barrier_label_3h_1rv"
    state_col = "barrier_state_3h_1rv"
    labels = pd.to_numeric(labelled[label_col], errors="coerce")
    states = pd.to_numeric(labelled[state_col], errors="coerce")
    valid = epoch_minutes.mod(HORIZON_MINUTES).eq(0) & labels.notna() & states.isin([-1, 0, 1])
    sample = labelled.loc[valid, ["time", *feature_columns]].copy().reset_index(drop=True)
    y = labels.loc[valid].astype("int8").to_numpy()
    fit_idx, holdout_idx, split = chronological_holdout_indices(sample)
    x = sample[feature_columns].apply(pd.to_numeric, errors="coerce")
    train_non_null = x.iloc[fit_idx].notna().any(axis=0)
    usable = train_non_null[train_non_null].index.tolist()
    if not usable:
        raise ValueError("No usable training features.")
    x_fit = x.iloc[fit_idx][usable].astype("float32")
    x_holdout = x.iloc[holdout_idx][usable].astype("float32")
    y_fit, y_holdout = y[fit_idx], y[holdout_idx]
    if set(np.unique(y_fit)) != set(CLASSES.tolist()) or set(np.unique(y_holdout)) != set(CLASSES.tolist()):
        raise ValueError("All three classes must be present in fit and reserved holdout.")

    dummy = DummyClassifier(strategy="prior")
    dummy.fit(np.zeros((len(y_fit), 1)), y_fit)
    dummy_pred = dummy.predict(np.zeros((len(y_holdout), 1)))
    dummy_prob = dummy.predict_proba(np.zeros((len(y_holdout), 1)))

    model = Pipeline([
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
    weights = compute_sample_weight("balanced", y_fit)
    model.fit(x_fit, y_fit, classifier__sample_weight=weights)
    model_pred = model.predict(x_holdout)
    model_prob = model.predict_proba(x_holdout)
    holdout_times = sample.loc[holdout_idx, "time"].reset_index(drop=True)
    return {
        "project": "DMPE",
        "purpose": "One-time reserved historical holdout evaluation of a frozen exploratory candidate; not prospective confirmation.",
        "candidate": {
            "horizon": HORIZON,
            "barrier_multiple_of_trailing_horizon_rv": MULTIPLIER,
            "model": "HistGradientBoostingClassifier",
            "class_weighting": "balanced sample weights computed on fit partition",
            "hyperparameters": {
                "learning_rate": 0.05,
                "max_iter": 160,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 60,
                "l2_regularization": 5.0,
                "early_stopping": False,
                "random_state": 42,
            },
            "feature_columns": usable,
        },
        "signal_time_cutoff_utc": SIGNAL_CUTOFF.isoformat(),
        "split": split,
        "fit_class_counts": {str(int(c)): int((y_fit == c).sum()) for c in CLASSES},
        "prior_baseline_holdout": metric_pack(y_holdout, dummy_pred, dummy_prob, dummy.classes_),
        "candidate_holdout": metric_pack(y_holdout, model_pred, model_prob, model.classes_),
        "paired_day_block_bootstrap": day_block_bootstrap_differences(
            holdout_times, y_holdout, model_pred, dummy_pred
        ),
        "selection_policy": "Candidate frozen from the preceding train/validation screen; no tuning, threshold selection or candidate substitution on this reserved block.",
        "holdout_policy": "This reserved historical 15% is opened once. It is not the prospective Delta Exchange holdout.",
        "trading_approval": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-file", required=True)
    parser.add_argument("--schema-file", required=True)
    parser.add_argument("--output-dir", default="symmetric_label_holdout")
    args = parser.parse_args()
    schema = json.loads(Path(args.schema_file).read_text(encoding="utf-8"))
    feature_columns = list(dict.fromkeys(schema.get("feature_columns", [])))
    required = ["time", "high", "low", "close", "realized_vol_180m_pct", *feature_columns]
    frame = pd.read_parquet(args.feature_file, columns=list(dict.fromkeys(required)))
    report = evaluate(frame, feature_columns)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "symmetric_label_holdout.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output / "symmetric_label_holdout.csv")
    pd.json_normalize([report]).to_csv(output / "symmetric_label_holdout.csv", index=False)
    print(json.dumps({
        "candidate_balanced_accuracy": report["candidate_holdout"]["balanced_accuracy"],
        "prior_balanced_accuracy": report["prior_baseline_holdout"]["balanced_accuracy"],
        "bootstrap_ci": report["paired_day_block_bootstrap"]["balanced_accuracy_difference_95pct_ci"],
        "trading_approval": False,
    }, indent=2))


if __name__ == "__main__":
    main()
