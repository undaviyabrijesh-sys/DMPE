import numpy as np
import pandas as pd

from dmpe.labels import add_barrier_labels


def make_barrier_frame(periods=24):
    times = pd.date_range("2024-01-01T00:00:00Z", periods=periods, freq="1min")
    close = np.full(periods, 100.0)
    high = np.full(periods, 100.2)
    low = np.full(periods, 99.8)
    # From row 0: upper is touched on bar 2, lower on bar 3 => +1.
    high[2], low[2] = 101.2, 99.8
    high[3], low[3] = 100.2, 98.8
    # From row 5: lower is touched on bar 1, upper on bar 2 => -1.
    high[6], low[6] = 100.2, 98.8
    high[7], low[7] = 101.2, 99.8
    # From row 9: both barriers are touched within the same minute => ambiguous.
    high[10], low[10] = 101.2, 98.8
    return pd.DataFrame({
        "time": times, "open": close, "high": high, "low": low,
        "close": close, "volume": 10.0,
        "realized_vol_15m_pct": 1.0,
    })


def test_first_touch_barrier_labeling_and_ambiguous_minute():
    output, labels, diagnostics, report = add_barrier_labels(
        make_barrier_frame(), horizons={"15m": 4}, multipliers=(1.0,)
    )
    label = "barrier_label_15m_1rv"
    state = "barrier_state_15m_1rv"
    bars = "barrier_bars_15m_1rv"
    assert label in labels
    assert state in diagnostics and bars in diagnostics
    assert output.loc[0, label] == 1
    assert output.loc[0, state] == 1
    assert output.loc[0, bars] == 2
    assert output.loc[5, label] == -1
    assert output.loc[5, state] == -1
    assert output.loc[5, bars] == 1
    assert output.loc[9, label] == 0
    assert output.loc[9, state] == 2
    assert output.loc[9, bars] == 1
    assert output.loc[13, label] == 0
    assert output.loc[13, state] == 0
    assert report[label]["state_counts"]["ambiguous_same_minute"] >= 1


def test_barrier_labels_are_null_when_future_window_crosses_data_gap():
    frame = make_barrier_frame()
    frame.loc[2, ["high", "low", "close"]] = np.nan
    output, labels, diagnostics, _ = add_barrier_labels(
        frame, horizons={"15m": 4}, multipliers=(1.0,)
    )
    assert pd.isna(output.loc[0, labels[0]])
    assert output.loc[0, diagnostics[0]] == -2


def test_barrier_labels_require_complete_minute_grid():
    frame = make_barrier_frame().drop(index=4).reset_index(drop=True)
    try:
        add_barrier_labels(frame, horizons={"15m": 4}, multipliers=(1.0,))
    except ValueError as exc:
        assert "complete 1-minute grid" in str(exc)
    else:
        raise AssertionError("Expected a ValueError for irregular minute timestamps")
