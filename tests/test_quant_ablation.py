import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_quant_ablation import build_complete_5m_bars


def test_complete_5m_bars_reject_partial_buckets_and_gaps():
    times = pd.date_range("2026-01-01T00:01:00Z", periods=15, freq="min")
    frame = pd.DataFrame({"time": times, "close": range(100, 115)})
    result = build_complete_5m_bars(frame)
    assert result["time"].tolist() == list(pd.to_datetime([
        "2026-01-01T00:05:00Z", "2026-01-01T00:10:00Z", "2026-01-01T00:15:00Z"
    ]))

    gapped = frame.loc[frame["time"] != pd.Timestamp("2026-01-01T00:08:00Z")]
    result_gap = build_complete_5m_bars(gapped)
    assert result_gap["time"].tolist() == list(pd.to_datetime([
        "2026-01-01T00:05:00Z", "2026-01-01T00:15:00Z"
    ]))


def test_5m_return_resets_after_missing_bucket():
    times = pd.date_range("2026-01-01T00:01:00Z", periods=20, freq="min")
    frame = pd.DataFrame({"time": times, "close": [100 + i for i in range(20)]})
    frame = frame.loc[~frame["time"].isin(pd.to_datetime([
        "2026-01-01T00:08:00Z", "2026-01-01T00:09:00Z", "2026-01-01T00:10:00Z"
    ]))]
    result = build_complete_5m_bars(frame)
    assert result["log_return"].isna().any()
    assert result["segment"].nunique() >= 2
