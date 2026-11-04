from dmpe.delta_rest import (
    MAX_CANDLES_PER_RESPONSE,
    RESOLUTION_SECONDS,
    DeltaPublicClient,
    filter_completed_candles,
)


def test_resolution_table_contains_model_timeframes():
    assert RESOLUTION_SECONDS["1m"] == 60
    assert RESOLUTION_SECONDS["15m"] == 900
    assert RESOLUTION_SECONDS["1h"] == 3600
    assert RESOLUTION_SECONDS["12h"] == 43200


def test_candle_api_limit_is_explicit():
    assert MAX_CANDLES_PER_RESPONSE == 2000


def test_backfill_splits_time_windows_without_overlapping_seconds(monkeypatch):
    client = DeltaPublicClient(request_pause_seconds=0)
    calls = []

    def fake_fetch(symbol, resolution, start_epoch, end_epoch):
        calls.append((start_epoch, end_epoch))
        import pandas as pd
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    monkeypatch.setattr(client, "fetch_candle_window", fake_fetch)
    client.backfill_candles("BTCUSD", "1m", 0, 2000 * 60 * 2)
    assert len(calls) > 1
    assert calls[1][0] == calls[0][1] + 1



def test_completed_candle_filter_removes_in_progress_and_grace_window():
    import pandas as pd

    frame = pd.DataFrame({
        "time": pd.to_datetime([
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:01:00Z",
            "2024-01-01T00:02:00Z",
        ]),
        "open": [100, 101, 102],
        "high": [101, 102, 103],
        "low": [99, 100, 101],
        "close": [101, 102, 103],
        "volume": [10, 10, 10],
    })
    result = filter_completed_candles(
        frame, "1m", pd.Timestamp("2024-01-01T00:02:05Z"),
        finality_delay_seconds=5,
    )
    assert result["time"].tolist() == [
        pd.Timestamp("2024-01-01T00:00:00Z"),
        pd.Timestamp("2024-01-01T00:01:00Z"),
    ]


def test_completed_hourly_candle_filter_keeps_only_fully_closed_hours():
    import pandas as pd

    frame = pd.DataFrame({
        "time": pd.to_datetime([
            "2024-01-01T00:00:00Z",
            "2024-01-01T01:00:00Z",
            "2024-01-01T02:00:00Z",
        ]),
        "close": [0.0001, 0.0002, 0.0003],
    })
    result = filter_completed_candles(
        frame, "1h", pd.Timestamp("2024-01-01T02:00:04Z"),
        finality_delay_seconds=5,
    )
    assert result["time"].tolist() == [
        pd.Timestamp("2024-01-01T00:00:00Z"),
    ]
