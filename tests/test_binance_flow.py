import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_binance_flow_features import build_features, load_aggregate_minutes


def test_binance_buyer_maker_sign_and_minute_close(tmp_path):
    archive = tmp_path / "BTCUSDT-aggTrades-2024-01.zip"
    rows = [
        "1,50000,0.01,1,1,1704067200123,false",
        "2,50010,0.02,2,2,1704067210123,true",
    ]
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("BTCUSDT-aggTrades-2024-01.csv", "\n".join(rows) + "\n")

    aggregates = load_aggregate_minutes(archive)
    assert aggregates.loc[pd.Timestamp("2024-01-01T00:00:00Z"), "signed_base_volume"] == 0.01
    assert aggregates.loc[pd.Timestamp("2024-01-01T00:01:00Z"), "signed_base_volume"] == -0.02

    features = build_features(aggregates, 2024, 1)
    first = features.loc[features["time"] == pd.Timestamp("2024-01-01T00:01:00Z")].iloc[0]
    second = features.loc[features["time"] == pd.Timestamp("2024-01-01T00:02:00Z")].iloc[0]
    assert first["signed_volume_imbalance"] == 1.0
    assert second["signed_volume_imbalance"] == -1.0
    assert first["binance_aggtrade_feed_observed"] == 1
    assert features["time"].iloc[0] == pd.Timestamp("2024-01-01T00:01:00Z")
    assert len(features) == 31 * 24 * 60
