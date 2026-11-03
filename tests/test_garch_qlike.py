import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_garch_qlike import (
    _day_block_ci, _garch_path, build_complete_5m_bars, qlike,
)
from dmpe.quant_regimes import fit_garch11


def test_qlike_prefers_correct_positive_variance_forecast():
    realized = np.array([0.01, 0.02, 0.03])
    exact = qlike(realized, realized)
    poor = qlike(realized * 4.0, realized)
    assert exact.mean() < poor.mean()


def test_qlike_rejects_nonpositive_forecasts():
    with pytest.raises(ValueError):
        qlike(np.array([0.0]), np.array([0.1]))


def test_garch_path_is_causal_to_each_observed_return():
    rng = np.random.default_rng(7)
    train = rng.normal(0, 0.002, 800)
    fit = fit_garch11(train)
    future = rng.normal(0, 0.002, 100)
    first = _garch_path(future, fit)
    changed = future.copy()
    changed[40:] *= 10
    second = _garch_path(changed, fit)
    np.testing.assert_allclose(first[:40], second[:40])


def test_complete_5m_bars_reject_partial_buckets_and_mark_gaps():
    times = pd.date_range("2026-01-01T00:01:00Z", periods=20, freq="min")
    frame = pd.DataFrame({"time": times, "close": np.arange(100, 120, dtype=float)})
    frame = frame.drop(index=[7])
    bars = build_complete_5m_bars(frame)
    assert len(bars) == 3
    assert bars["time"].iloc[0] == pd.Timestamp("2026-01-01T00:05:00Z")
    assert bars["time"].iloc[1] == pd.Timestamp("2026-01-01T00:15:00Z")
    assert pd.isna(bars["log_return"].iloc[0])
    assert pd.isna(bars["log_return"].iloc[1])


def test_day_block_bootstrap_returns_interval_for_enough_days():
    times = pd.Series(pd.date_range("2026-01-01", periods=20 * 24, freq="h", tz="UTC"))
    diff = np.linspace(-0.1, 0.1, len(times))
    result = _day_block_ci(diff, times, reps=100)
    assert result["blocks"] == 20
    assert result["lower_95"] <= result["upper_95"]
