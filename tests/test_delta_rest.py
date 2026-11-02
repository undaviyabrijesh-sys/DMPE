from dmpe.delta_rest import RESOLUTION_SECONDS, MAX_CANDLES_PER_RESPONSE

def test_resolution_table_contains_initial_timeframes():
    assert RESOLUTION_SECONDS["1m"] == 60
    assert RESOLUTION_SECONDS["15m"] == 900
    assert RESOLUTION_SECONDS["1h"] == 3600

def test_candle_api_limit_is_explicit():
    assert MAX_CANDLES_PER_RESPONSE == 2000
