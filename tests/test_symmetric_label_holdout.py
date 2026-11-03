import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_symmetric_label_holdout import (  # noqa: E402
    CLASSES,
    chronological_holdout_indices,
    day_block_bootstrap_differences,
    metric_pack,
)


def test_holdout_split_is_chronological_with_embargo():
    sample = pd.DataFrame({
        "time": pd.date_range("2024-01-01T00:00:00Z", periods=1_000, freq="3h")
    })
    fit, holdout, info = chronological_holdout_indices(sample)
    assert fit.max() < holdout.min()
    assert info["fit_rows"] > 0
    assert info["holdout_rows"] > 0
    times = sample["time"]
    assert times.iloc[fit[-1]] + pd.Timedelta(minutes=180) < times.iloc[holdout[0]]


def test_day_block_bootstrap_returns_paired_intervals():
    times = pd.Series(pd.date_range("2026-01-01T00:00:00Z", periods=240, freq="3h"))
    y = np.tile(np.array([-1, 0, 1], dtype="int8"), 80)
    model_pred = y.copy()
    baseline_pred = np.zeros_like(y)
    result = day_block_bootstrap_differences(
        times, y, model_pred, baseline_pred, reps=100, seed=7
    )
    assert result["day_blocks"] == 30
    assert result["replicates"] == 100
    assert result["balanced_accuracy_difference_95pct_ci"][0] > 0


def test_metric_pack_reports_all_classes():
    y = np.array([-1, 0, 1], dtype="int8")
    probabilities = np.eye(3)
    result = metric_pack(y, y, probabilities, CLASSES)
    assert result["balanced_accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
