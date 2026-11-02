"""Build causal 1m signed-flow features from Binance USD-M aggTrades ZIP data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

COLUMNS = [
    "aggregate_trade_id", "price", "quantity", "first_trade_id",
    "last_trade_id", "timestamp_ms", "is_buyer_maker",
]
LARGE_PRINT_USD = 100_000.0


def load_aggregate_minutes(archive: Path, chunksize: int = 1_000_000) -> pd.DataFrame:
    partials = []
    with zipfile.ZipFile(archive) as zipped:
        csv_names = [name for name in zipped.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Expected exactly one CSV in {archive.name}; found {csv_names}")
        with zipped.open(csv_names[0]) as stream:
            for chunk in pd.read_csv(stream, header=None, names=COLUMNS, dtype=str, chunksize=chunksize):
                chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce")
                chunk["quantity"] = pd.to_numeric(chunk["quantity"], errors="coerce")
                chunk["timestamp_ms"] = pd.to_numeric(chunk["timestamp_ms"], errors="coerce")
                chunk = chunk.dropna(subset=["price", "quantity", "timestamp_ms"])
                chunk = chunk[(chunk["price"] > 0) & (chunk["quantity"] > 0)]
                if chunk.empty:
                    continue
                buyer_maker_text = chunk["is_buyer_maker"].astype(str).str.strip().str.lower()
                valid_flag = buyer_maker_text.isin({"true", "false", "1", "0"})
                chunk = chunk.loc[valid_flag].copy()
                buyer_maker_text = buyer_maker_text.loc[chunk.index]
                if chunk.empty:
                    continue
                # Binance isBuyerMaker=true => aggressive seller; false => aggressive buyer.
                is_buyer_maker = buyer_maker_text.isin({"true", "1"}).to_numpy()
                signed_sign = np.where(is_buyer_maker, -1.0, 1.0)
                chunk["signed_qty"] = chunk["quantity"].to_numpy() * signed_sign
                chunk["notional"] = chunk["price"] * chunk["quantity"]
                chunk["signed_notional"] = chunk["notional"] * signed_sign
                chunk["large_notional"] = np.where(
                    chunk["notional"] >= LARGE_PRINT_USD, chunk["notional"], 0.0
                )
                chunk["minute_open"] = pd.to_datetime(
                    chunk["timestamp_ms"].astype("int64"), unit="ms", utc=True
                ).dt.floor("min")
                grouped = chunk.groupby("minute_open", sort=True).agg(
                    trade_count=("quantity", "size"),
                    total_base_volume=("quantity", "sum"),
                    signed_base_volume=("signed_qty", "sum"),
                    total_notional=("notional", "sum"),
                    signed_notional=("signed_notional", "sum"),
                    large_print_notional=("large_notional", "sum"),
                )
                partials.append(grouped)
    if not partials:
        raise ValueError(f"No valid aggTrades rows found in {archive}")
    return pd.concat(partials).groupby(level=0).sum().sort_index()


def build_features(aggregates: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    grid = pd.date_range(start, end, freq="1min", inclusive="left")
    frame = aggregates.reindex(grid)
    count_cols = ["trade_count"]
    volume_cols = [
        "total_base_volume", "signed_base_volume", "total_notional",
        "signed_notional", "large_print_notional",
    ]
    frame[count_cols] = frame[count_cols].fillna(0)
    frame[volume_cols] = frame[volume_cols].fillna(0)
    frame["signed_volume_imbalance"] = np.divide(
        frame["signed_base_volume"], frame["total_base_volume"],
        out=np.full(len(frame), np.nan), where=frame["total_base_volume"].to_numpy() > 0,
    )
    frame["signed_notional_imbalance"] = np.divide(
        frame["signed_notional"], frame["total_notional"],
        out=np.full(len(frame), np.nan), where=frame["total_notional"].to_numpy() > 0,
    )
    frame["large_trade_share"] = np.divide(
        frame["large_print_notional"], frame["total_notional"],
        out=np.full(len(frame), np.nan), where=frame["total_notional"].to_numpy() > 0,
    )
    current_5m_signed = frame["signed_notional"].rolling(5, min_periods=5).sum()
    current_5m_total = frame["total_notional"].rolling(5, min_periods=5).sum()
    previous_5m_signed = current_5m_signed.shift(5)
    previous_5m_total = current_5m_total.shift(5)
    current_imbalance = current_5m_signed / current_5m_total.replace(0, np.nan)
    previous_imbalance = previous_5m_signed / previous_5m_total.replace(0, np.nan)
    frame["flow_acceleration_5m"] = current_imbalance - previous_imbalance
    frame["binance_aggtrade_feed_observed"] = 1
    frame["time"] = frame.index + pd.Timedelta(minutes=1)
    frame["flow_source"] = "binance_usdm_btcusdt_proxy"
    return frame.reset_index(drop=True)[[
        "time", "trade_count", "total_base_volume", "signed_base_volume",
        "signed_volume_imbalance", "total_notional", "signed_notional",
        "signed_notional_imbalance", "large_print_notional", "large_trade_share",
        "flow_acceleration_5m", "binance_aggtrade_feed_observed", "flow_source",
    ]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--output-dir", default="binance_flow_output")
    args = parser.parse_args()
    archive = Path(args.archive)
    aggregates = load_aggregate_minutes(archive)
    features = build_features(aggregates, args.year, args.month)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"binance_BTCUSDT_aggtrade_flow_1m_{args.year}-{args.month:02d}"
    parquet = out_dir / f"{stem}.parquet"
    features.to_parquet(parquet, index=False, compression="zstd")
    report = {
        "source_archive": archive.name,
        "rows": len(features),
        "first_close_utc": features["time"].min().isoformat(),
        "last_close_utc": features["time"].max().isoformat(),
        "observed_minutes_with_trades": int((features["trade_count"] > 0).sum()),
        "zero_trade_minutes": int((features["trade_count"] == 0).sum()),
        "large_print_threshold_usd": LARGE_PRINT_USD,
        "timestamp_rule": "UTC minute open + 1 minute (completed minute close)",
        "buyer_maker_sign": "true=aggressive sell negative; false=aggressive buy positive",
        "venue_semantics": "Binance USD-M proxy, not Delta observed flow",
    }
    (out_dir / f"{stem}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
