import numpy as np
import pandas as pd
import pytest

from diagnose_event_label_balance import (
    chronological_partition_masks,
    evaluate_event_grid,
    summarize_state_counts,
)


def test_state_summary_keeps_ambiguous_and_invalid_separate():
    summary = summarize_state_counts(np.array([-1, -1, 0, 1, 2, -2]))
    assert summary["samples"] == 6
    assert summary["valid_outcomes"] == 5
    assert summary["directional_events"] == 3
    assert summary["ambiguous_same_minute"] == 1
    assert summary["invalid_or_incomplete"] == 1
    assert summary["directional_event_rate_of_valid"] == pytest.approx(0.6)
    assert summary["up_down_count_balance_min_over_max"] == pytest.approx(0.5)


def test_partition_masks_are_chronological_and_purge_boundaries():
    masks = chronological_partition_masks(100)
    train = np.flatnonzero(masks["train"])
    validation = np.flatnonzero(masks["validation"])
    test = np.flatnonzero(masks["test"])
    assert train.max() < validation.min()
    assert validation.max() < test.min()
    assert train.max() + 1 < validation.min()
    assert validation.max() + 1 < test.min()


def test_event_grid_reports_all_horizons_and_multipliers():
    n = 1_200
    time = pd.date_range("2025-01-01T00:00:00Z", periods=n, freq="1min")
    close = 100 + np.cumsum(np.sin(np.arange(n) / 8) * 0.03)
    frame = pd.DataFrame({
        "time": time,
        "high": close * 1.001,
        "low": close * 0.999,
        "close": close,
        "realized_vol_60m_pct": 0.15,
        "realized_vol_180m_pct": 0.25,
    })
    results, report = evaluate_event_grid(frame)
    assert len(results) == 2 * 5 * 3
    assert set(results["horizon"]) == {"1h", "3h"}
    assert set(results["partition"]) == {"train", "validation", "test"}
    assert report["confirmatory_labels_modified"] is False
    assert report["trading_approval"] is False


def test_event_grid_rejects_missing_timestamp_rows():
    n = 300
    time = pd.date_range("2025-01-01T00:00:00Z", periods=n, freq="1min").delete(100)
    close = np.full(len(time), 100.0)
    frame = pd.DataFrame({
        "time": time, "high": close * 1.01, "low": close * 0.99,
        "close": close, "realized_vol_60m_pct": 0.2,
        "realized_vol_180m_pct": 0.3,
    })
    with pytest.raises(ValueError, match="explicit 1-minute timestamp grid"):
        evaluate_event_grid(frame)
