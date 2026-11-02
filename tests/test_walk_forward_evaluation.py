import pandas as pd

from walk_forward_evaluation import make_fold_masks


def test_walk_forward_folds_are_purged_and_test_blocks_are_separate():
    times = pd.Series(pd.date_range(
        "2024-01-01T00:00:00Z",
        "2026-11-01T00:00:00Z",
        freq="15min",
    ))
    folds = make_fold_masks(times, horizon_minutes=15)
    assert len(folds) == 4
    for fold in folds:
        train_times = times[fold["train_mask"]]
        validation_times = times[fold["validation_mask"]]
        test_times = times[fold["test_mask"]]
        assert len(train_times) > 500
        assert len(validation_times) > 100
        assert len(test_times) > 100
        assert validation_times.min() - train_times.max() >= pd.Timedelta(minutes=30)
        assert test_times.min() - validation_times.max() >= pd.Timedelta(minutes=30)
        assert validation_times.min() >= pd.Timestamp(fold["nominal_validation_start"]) + pd.Timedelta(minutes=15)
        assert test_times.min() >= pd.Timestamp(fold["nominal_test_start"]) + pd.Timedelta(minutes=15)


def test_expanding_training_windows_grow_over_time():
    times = pd.Series(pd.date_range(
        "2024-01-01T00:00:00Z",
        "2026-11-01T00:00:00Z",
        freq="15min",
    ))
    folds = make_fold_masks(times, horizon_minutes=15)
    train_sizes = [int(fold["train_mask"].sum()) for fold in folds]
    assert train_sizes == sorted(train_sizes)
    assert len(set(train_sizes)) == len(train_sizes)
