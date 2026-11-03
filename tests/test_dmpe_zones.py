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
