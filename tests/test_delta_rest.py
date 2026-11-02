from dmpe.delta_rest import (
    MAX_CANDLES_PER_RESPONSE,
    RESOLUTION_SECONDS,
    DeltaPublicClient,
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
