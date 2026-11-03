import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_symmetric_label_predictability import (  # noqa: E402
    CLASSES,
    SIGNAL_CUTOFFS,
    chronological_train_validation,
    metrics,
)


def test_train_validation_split_is_chronological_and_purged():
    sample = pd.DataFrame({
        "time": pd.date_range("2024-01-01T00:00:00Z", periods=1_000, freq="1h")
    })
    train, validation, info = chronological_train_validation(sample, 60)
    assert train.max() < validation.min()
    assert info["train_rows"] > 0
    assert info["validation_rows"] > 0
    assert info["reserved_unscored_rows"] > 0
    times = sample["time"]
    assert times.iloc[train[-1]] + pd.Timedelta(minutes=60) < times.iloc[validation[0]]


def test_validation_metrics_align_all_three_classes():
    y = np.array([-1, 0, 1, -1, 0, 1], dtype="int8")
    prediction = np.array([-1, 0, 1, 0, 0, 1], dtype="int8")
    probabilities = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.2, 0.7, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.2, 0.7],
    ])
    result = metrics(y, prediction, probabilities, CLASSES)
    assert result["rows"] == 6
    assert result["balanced_accuracy"] > 0.8
    assert result["multiclass_log_loss"] > 0
    assert result["multiclass_brier_score"] >= 0


def test_signal_cutoff_is_before_original_historical_test_window():
    assert SIGNAL_CUTOFFS["1h"] == pd.Timestamp("2026-05-30T18:00:00Z")
    assert SIGNAL_CUTOFFS["3h"] == pd.Timestamp("2026-05-30T15:00:00Z")
