import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dmpe.orderflow import aggregate_event_records, parse_book_event, parse_trade_event


def test_delta_buyer_role_infers_aggressor_side_without_guessing_unknowns():
    buy = parse_trade_event({"type": "trades", "p": "100", "s": "2", "r": "t"})
    sell = parse_trade_event({"type": "trades", "p": "101", "s": "3", "r": "m"})
    unknown = parse_trade_event({"type": "trades", "p": "100", "s": "1", "r": "x"})
    assert buy["aggressor_side"] == "buy"
    assert buy["signed_size"] == 2
    assert sell["aggressor_side"] == "sell"
    assert sell["signed_size"] == -3
    assert unknown["aggressor_side"] == "unknown"
    assert np.isnan(unknown["signed_size"])


def test_l2_snapshot_calculates_spread_depth_and_microprice_pressure():
    event = {
        "type": "ob_l2",
        "b": [["99.0", "10"], ["98.0", "20"], ["97.0", "30"]],
        "a": [["101.0", "5"], ["102.0", "10"], ["103.0", "15"]],
    }
    parsed = parse_book_event(event)
    assert parsed is not None
    assert parsed["mid_price"] == 100.0
    assert parsed["spread_bps"] == 200.0
    assert parsed["depth_imbalance_1"] == 1 / 3
    assert parsed["depth_imbalance_3"] == 0.2
    assert parsed["microprice_mid_bps"] > 0


def test_aggregation_timestamps_completed_minute_and_preserves_missingness():
    records = [
        {"received_at": "2024-01-01T00:00:00.010Z", "event": {
            "type": "trades", "p": "100", "s": 2, "r": "t", "t": 1704067200010000
        }},
        {"received_at": "2024-01-01T00:00:00.020Z", "event": {
            "type": "trades", "p": "101", "s": 3, "r": "m", "t": 1704067200020000
        }},
        {"received_at": "2024-01-01T00:00:00.030Z", "event": {
            "type": "ob_l2",
            "b": [["99", "10"]], "a": [["101", "5"]],
            "ts": 1704067200030000,
        }},
        {"received_at": "2024-01-01T00:02:00.010Z", "event": {
            "type": "trades", "p": "102", "s": 4, "r": "t", "t": 1704067320010000
        }},
    ]
    frame = aggregate_event_records(records)
    assert len(frame) == 3
    assert frame.loc[0, "time"] == pd.Timestamp("2024-01-01T00:01:00Z")
    assert frame.loc[0, "trade_buy_taker_volume"] == 2
    assert frame.loc[0, "trade_sell_taker_volume"] == 3
    assert frame.loc[0, "trade_volume_imbalance"] == -0.2
    assert frame.loc[0, "trade_feed_observed"] == 1
    assert frame.loc[1, "trade_feed_observed"] == 0
    assert pd.isna(frame.loc[1, "trade_total_volume"])
    assert frame.loc[2, "trade_feed_observed"] == 1
    assert frame.loc[2, "l2_feed_observed"] == 0
