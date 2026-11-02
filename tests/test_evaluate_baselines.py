import numpy as np
import pandas as pd

from scripts.evaluate_baselines import (
    _predict_direction,
    _signal_metrics,
    chronological_splits,
)


def test_probability_gate_requires_confident_direction():
    classes = np.array([-1, 0, 1], dtype="int8")
    probabilities = np.array([
        [0.70, 0.10, 0.20],
        [0.10, 0.20, 0.70],
        [0.25, 0.50, 0.25],
        [0.45, 0.10, 0.45],
    ])
    signals = _predict_direction(probabilities, classes, 0.60)
    assert signals.tolist() == [-1, 1, 0, 0]


def test_signal_metrics_separate_directional_hits_and_neutral_outcomes():
    y = np.array([-1, 0, 1, 1], dtype="int8")
    signals = np.array([-1, 1, -1, 0], dtype="int8")
    metrics = _signal_metrics(y, signals)
    assert metrics["signals"] == 3
    assert metrics["correct_directional_share_of_signals"] == 1 / 3
    assert metrics["wrong_direction_share_of_signals"] == 1 / 3
    assert metrics["no_barrier_touch_share_of_signals"] == 1 / 3


def test_chronological_splits_purge_and_embargo_one_full_horizon():
    times = pd.date_range("2024-01-01T00:00:00Z", periods=2000, freq="15min")
    frame = pd.DataFrame({"time": times})
    train, validation, test, info = chronological_splits(frame, horizon_minutes=15)
    assert len(train) > len(validation) > 0
    assert len(test) > 0
    train_end = frame.loc[train[-1], "time"]
    val_start = frame.loc[validation[0], "time"]
    val_end = frame.loc[validation[-1], "time"]
    test_start = frame.loc[test[0], "time"]
    assert val_start - train_end >= pd.Timedelta(minutes=30)
    assert test_start - val_end >= pd.Timedelta(minutes=30)
    assert info["embargo_minutes_each_side_of_cut"] == 15
