import pandas as pd

from dmpe_zones import detect_structure_zones


def test_empty_or_incomplete_candles_return_no_zones():
    assert detect_structure_zones(pd.DataFrame()) == []
    assert detect_structure_zones(pd.DataFrame({"close": [1, 2]})) == []


def test_zone_boundaries_are_ordered_and_never_zero_width():
    # Deterministic alternating market with two sharp displacement legs.
    close = []
    price = 100.0
    for i in range(90):
        if 20 <= i < 24:
            price += 2.2
        elif 55 <= i < 59:
            price -= 2.4
        else:
            price += 0.15 if i % 2 else -0.12
        close.append(price)

    rows = []
    for i, value in enumerate(close):
        rows.append({
            "time": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(minutes=15 * i),
            "open": value - (0.2 if i % 2 else -0.2),
            "high": value + 0.35,
            "low": value - 0.35,
            "close": value,
        })
    zones = detect_structure_zones(pd.DataFrame(rows), impulse_atr=1.2)
    for zone in zones:
        assert zone["low"] < zone["high"]
        assert zone["kind"] in {"DEMAND", "SUPPLY", "RBS", "SBR"}
        assert zone["strength"] > 0


def test_swing_high_uses_full_candle_range_and_waits_for_confirmation():
    rows = []
    for i in range(24):
        close = 100.0 + (0.12 if i % 2 else -0.12)
        rows.append({
            "time": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(minutes=15 * i),
            "open": close - 0.05,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
        })
    rows[10] = {
        "time": rows[10]["time"], "open": 106.0,
        "high": 108.0, "low": 105.0, "close": 107.0,
    }
    for i, close in zip((11, 12, 13), (103.0, 100.0, 97.0)):
        rows[i] = {
            "time": rows[i]["time"], "open": close + 0.5,
            "high": close + 1.0, "low": close - 1.0, "close": close,
        }

    zones = detect_structure_zones(pd.DataFrame(rows))
    swing = [z for z in zones if z.get("origin_type") == "SWING_HIGH"]
    assert swing
    assert swing[0]["low"] == 105.0
    assert swing[0]["high"] == 108.0
    assert swing[0]["origin_time"] == rows[10]["time"]
    assert swing[0]["created_at"] == rows[13]["time"]
    assert swing[0]["origin_index"] == 10
