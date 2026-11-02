"""Reconstruct auditable minute-level features from Delta public WS events.

Trade aggressor direction is inferred from the documented buyer_role field:
r='t' means buyer was taker (aggressive buy); r='m' means buyer was maker,
therefore the seller was taker (aggressive sell). Unknown values remain
unclassified rather than being guessed.

The input is a locally captured raw JSONL event stream. These features are
only available from the start of WS capture; historical candles do not contain
the underlying trade or order-book events.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

LEVELS = (1, 5, 10, 15)


def to_utc_timestamp(value: Any, *, numeric_unit: str = "us") -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
            return pd.to_datetime(int(value), unit=numeric_unit, utc=True, errors="coerce")
        return pd.to_datetime(value, utc=True, errors="coerce")
    except (TypeError, ValueError, OverflowError):
        return None


def unwrap_record(record: dict[str, Any]) -> dict[str, Any]:
    event = record.get("event", record)
    return event if isinstance(event, dict) else {}


def event_timestamp(record: dict[str, Any], event: dict[str, Any]) -> pd.Timestamp | None:
    kind = str(event.get("type", "")).lower()
    if kind == "trades":
        ts = to_utc_timestamp(event.get("t"))
        if ts is not None and not pd.isna(ts):
            return ts
    ts = to_utc_timestamp(event.get("ts"))
    if ts is not None and not pd.isna(ts):
        return ts
    received_at = to_utc_timestamp(record.get("received_at"))
    if received_at is not None and not pd.isna(received_at):
        return received_at
    return None


def parse_trade_event(event: dict[str, Any]) -> dict[str, Any] | None:
    if str(event.get("type", "")).lower() != "trades":
        return None
    try:
        price = float(event["p"])
        size = float(event["s"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(price) or not np.isfinite(size) or price <= 0 or size <= 0:
        return None

    buyer_role = str(event.get("r", "")).lower()
    if buyer_role == "t":
        aggressor_side = "buy"
        signed_size = size
    elif buyer_role == "m":
        aggressor_side = "sell"
        signed_size = -size
    else:
        aggressor_side = "unknown"
        signed_size = np.nan

    return {
        "price": price,
        "size": size,
        "buyer_role": buyer_role or None,
        "aggressor_side": aggressor_side,
        "signed_size": signed_size,
    }


def _parse_levels(levels: Any, side: str) -> list[tuple[float, float]]:
    if not isinstance(levels, list):
        return []
    parsed = []
    for level in levels:
        if not isinstance(level, (list, tuple)) or len(level) < 2:
            continue
        try:
            price, size = float(level[0]), float(level[1])
        except (TypeError, ValueError):
            continue
        if np.isfinite(price) and np.isfinite(size) and price > 0 and size >= 0:
            parsed.append((price, size))
    return sorted(parsed, key=lambda x: x[0], reverse=(side == "bid"))


def parse_book_event(event: dict[str, Any]) -> dict[str, Any] | None:
    if str(event.get("type", "")).lower() != "ob_l2":
        return None
    bids = _parse_levels(event.get("b"), "bid")
    asks = _parse_levels(event.get("a"), "ask")
    if not bids or not asks:
        return None
    best_bid, bid_qty = bids[0]
    best_ask, ask_qty = asks[0]
    if best_bid <= 0 or best_ask <= best_bid:
        return None
    mid = (best_bid + best_ask) / 2.0
    total_top = bid_qty + ask_qty
    result: dict[str, Any] = {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid_price": mid,
        "spread_bps": (best_ask - best_bid) / mid * 10_000.0,
        "microprice_mid_bps": (
            (((best_ask * bid_qty + best_bid * ask_qty) / total_top) / mid - 1.0) * 10_000.0
            if total_top > 0 else np.nan
        ),
        "bid_levels": len(bids),
        "ask_levels": len(asks),
    }
    for n_levels in LEVELS:
        bid_depth = sum(size for _, size in bids[:n_levels])
        ask_depth = sum(size for _, size in asks[:n_levels])
        total = bid_depth + ask_depth
        result[f"bid_depth_{n_levels}"] = bid_depth
        result[f"ask_depth_{n_levels}"] = ask_depth
        result[f"depth_imbalance_{n_levels}"] = (
            (bid_depth - ask_depth) / total if total > 0 else np.nan
        )
        weights = np.array([1.0 / (i + 1) for i in range(n_levels)], dtype="float64")
        weighted_bid = sum(size * weights[i] for i, (_, size) in enumerate(bids[:n_levels]))
        weighted_ask = sum(size * weights[i] for i, (_, size) in enumerate(asks[:n_levels]))
        weighted_total = weighted_bid + weighted_ask
        result[f"weighted_depth_imbalance_{n_levels}"] = (
            (weighted_bid - weighted_ask) / weighted_total if weighted_total > 0 else np.nan
        )
    return result



def parse_funding_event(record: dict[str, Any], event: dict[str, Any]) -> dict[str, Any] | None:
    """Parse Delta's funding_rate snapshot with its explicit next realization timestamp."""
    if str(event.get("type", "")).lower() != "funding_rate":
        return None
    observed_at = event_timestamp(record, event)
    next_realization = to_utc_timestamp(event.get("nfr"))
    try:
        rate = float(event["fr"])
        interval_seconds = int(event["fi"])
    except (KeyError, TypeError, ValueError):
        return None
    if (
        observed_at is None or pd.isna(observed_at)
        or next_realization is None or pd.isna(next_realization)
        or not np.isfinite(rate) or interval_seconds <= 0
    ):
        return None
    if next_realization <= observed_at:
        return None
    return {
        "observed_at": observed_at,
        "next_funding_realization": next_realization,
        "funding_rate": rate,
        "funding_interval_seconds": interval_seconds,
        "symbol": event.get("sy") or record.get("symbol"),
    }


def extract_funding_observations(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Return timestamped funding snapshots; never infer settlement from hourly OHLC."""
    rows = []
    for record in records:
        if not isinstance(record, dict):
            continue
        event = unwrap_record(record)
        parsed = parse_funding_event(record, event)
        if parsed is not None:
            rows.append(parsed)
    if not rows:
        return pd.DataFrame(columns=[
            "observed_at", "next_funding_realization", "funding_rate",
            "funding_interval_seconds", "symbol",
        ])
    return (
        pd.DataFrame(rows)
        .drop_duplicates(["observed_at", "next_funding_realization", "symbol"])
        .sort_values(["next_funding_realization", "observed_at"])
        .reset_index(drop=True)
    )

def _aggregate_trade_minute(events: list[dict[str, Any]]) -> dict[str, Any]:
    events = sorted(events, key=lambda item: item["_time"])
    sizes = np.array([item["size"] for item in events], dtype="float64")
    prices = np.array([item["price"] for item in events], dtype="float64")
    sides = [item["aggressor_side"] for item in events]
    buy_volume = float(sum(item["size"] for item in events if item["aggressor_side"] == "buy"))
    sell_volume = float(sum(item["size"] for item in events if item["aggressor_side"] == "sell"))
    unknown_volume = float(sum(item["size"] for item in events if item["aggressor_side"] == "unknown"))
    classified_volume = buy_volume + sell_volume
    total_volume = float(sizes.sum())
    signed = buy_volume - sell_volume
    vwap = float(np.sum(prices * sizes) / total_volume) if total_volume > 0 else np.nan

    time_deltas_ms = []
    event_times = [item.get("_time") for item in events]
    valid_times = [value for value in event_times if value is not None]
    if len(valid_times) > 1:
        sorted_times = sorted(valid_times)
        time_deltas_ms = np.diff(np.array([value.value for value in sorted_times])) / 1e6

    log_tick_returns = np.diff(np.log(prices)) if len(prices) > 1 else np.array([], dtype="float64")
    return {
        "trade_count": int(len(events)),
        "trade_buy_taker_volume": buy_volume,
        "trade_sell_taker_volume": sell_volume,
        "trade_unknown_role_volume": unknown_volume,
        "trade_total_volume": total_volume,
        "trade_classified_volume": classified_volume,
        "trade_signed_volume": signed,
        "trade_volume_imbalance": signed / classified_volume if classified_volume > 0 else np.nan,
        "trade_buy_taker_share": buy_volume / classified_volume if classified_volume > 0 else np.nan,
        "trade_vwap": vwap,
        "trade_last_price": float(prices[-1]),
        "trade_range_bps": (float(prices.max()) / float(prices.min()) - 1.0) * 10_000.0,
        "trade_tick_return_std": float(np.std(log_tick_returns)) if len(log_tick_returns) else np.nan,
        "trade_mean_interarrival_ms": float(np.mean(time_deltas_ms)) if len(time_deltas_ms) else np.nan,
        "trade_median_size": float(np.median(sizes)),
        "trade_large_print_count": int((sizes >= np.quantile(sizes, 0.95)).sum()),
        "trade_unknown_role_count": int(sum(side == "unknown" for side in sides)),
    }


def _aggregate_book_minute(events: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(events)
    result: dict[str, Any] = {"l2_snapshot_count": int(len(frame))}
    columns = [
        "spread_bps", "microprice_mid_bps",
        *[f"depth_imbalance_{n}" for n in LEVELS],
        *[f"weighted_depth_imbalance_{n}" for n in LEVELS],
        *[f"bid_depth_{n}" for n in LEVELS],
        *[f"ask_depth_{n}" for n in LEVELS],
    ]
    for col in columns:
        values = pd.to_numeric(frame[col], errors="coerce").dropna()
        result[f"l2_{col}_last"] = float(values.iloc[-1]) if len(values) else np.nan
        result[f"l2_{col}_mean"] = float(values.mean()) if len(values) else np.nan
    result["l2_spread_bps_max"] = float(frame["spread_bps"].max())
    return result


def aggregate_event_records(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Convert raw records to a close-timestamped 1-minute event feature grid."""
    trades_by_minute: dict[pd.Timestamp, list[dict[str, Any]]] = defaultdict(list)
    books_by_minute: dict[pd.Timestamp, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        if not isinstance(record, dict):
            continue
        event = unwrap_record(record)
        kind = str(event.get("type", "")).lower()
        if kind not in {"trades", "ob_l2"}:
            continue
        timestamp = event_timestamp(record, event)
        if timestamp is None or pd.isna(timestamp):
            continue
        minute_open = timestamp.floor("min")
        if kind == "trades":
            parsed = parse_trade_event(event)
            if parsed is not None:
                parsed["_time"] = timestamp
                trades_by_minute[minute_open].append(parsed)
        else:
            parsed = parse_book_event(event)
            if parsed is not None:
                books_by_minute[minute_open].append(parsed)

    if not trades_by_minute and not books_by_minute:
        return pd.DataFrame(columns=["time"])

    first = min([*trades_by_minute.keys(), *books_by_minute.keys()])
    last = max([*trades_by_minute.keys(), *books_by_minute.keys()])
    minute_grid = pd.date_range(first, last, freq="1min", tz="UTC")
    rows: list[dict[str, Any]] = []
    for minute_open in minute_grid:
        row: dict[str, Any] = {"time": minute_open + pd.Timedelta(minutes=1)}
        trades = trades_by_minute.get(minute_open, [])
        books = books_by_minute.get(minute_open, [])
        row["trade_feed_observed"] = int(bool(trades))
        row["l2_feed_observed"] = int(bool(books))
        if trades:
            row.update(_aggregate_trade_minute(trades))
        if books:
            row.update(_aggregate_book_minute(books))
        rows.append(row)
    out = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    return out


def read_jsonl_files(paths: Iterable[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    file_stats = []
    for path in sorted(paths):
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
                    count += 1
        file_stats.append({"file": str(path), "parsed_records": count})
    return records, {"files": file_stats, "parsed_records": len(records)}


def build_live_orderflow_dataset(
    input_dir: str | Path = "data/raw/live",
    output_dir: str | Path = "data/processed",
) -> dict[str, Any]:
    input_path = Path(input_dir)
    files = sorted(input_path.glob("delta_*.jsonl")) if input_path.exists() else []
    if not files:
        raise FileNotFoundError(f"No Delta JSONL recorder files found in {input_path}")
    records, ingest = read_jsonl_files(files)
    features = aggregate_event_records(records)
    funding_observations = extract_funding_observations(records)
    if features.empty and funding_observations.empty:
        raise ValueError("No usable trades, L2 snapshots or funding observations were found in the JSONL files.")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    parquet_path = output_path / "dmpe_BTCUSD_live_orderflow_1m.parquet"
    csv_path = output_path / "dmpe_BTCUSD_live_orderflow_1m.csv"
    if not features.empty:
        features.to_parquet(parquet_path, index=False, compression="zstd")
        features.to_csv(csv_path, index=False)
    funding_parquet = output_path / "dmpe_BTCUSD_funding_observations.parquet"
    funding_csv = output_path / "dmpe_BTCUSD_funding_observations.csv"
    if not funding_observations.empty:
        funding_observations.to_parquet(funding_parquet, index=False, compression="zstd")
        funding_observations.to_csv(funding_csv, index=False)
    report = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": ingest,
        "first_close_time_utc": features["time"].min().isoformat() if not features.empty else None,
        "last_close_time_utc": features["time"].max().isoformat() if not features.empty else None,
        "minutes": int(len(features)),
        "minutes_with_trade_events": int(features["trade_feed_observed"].sum()) if "trade_feed_observed" in features else 0,
        "minutes_with_l2_events": int(features["l2_feed_observed"].sum()) if "l2_feed_observed" in features else 0,
        "funding_observations": int(len(funding_observations)),
        "funding_observations_parquet": str(funding_parquet) if not funding_observations.empty else None,
        "funding_observations_csv": str(funding_csv) if not funding_observations.empty else None,
        "feature_columns": [col for col in features.columns if col not in {"time"}],
        "semantics": {
            "trade_side": (
                "Signed size uses Delta trades.r buyer_role: t=buyer taker (buy aggressor); "
                "m=buyer maker (sell aggressor). Other codes are unclassified."
            ),
            "orderbook": "ob_l2 snapshots aggregate top-1/5/10/15 depth imbalances and spread/microprice.",
            "timestamps": "Event features are timestamped at the end of the minute they summarize.",
            "missingness": (
                "No forward fill. Minutes without a trade or L2 event retain NaN measurements "
                "and explicit feed_observed flags."
            ),
            "funding": (
                "Funding snapshots preserve observed_at, published funding_rate and explicit "
                "next_funding_realization. They are not yet charged by the execution simulator; "
                "the signed cashflow convention and realized payment must be reconciled first."
            ),
            "scope": "Live capture only; does not reconstruct historical tick or order-book history.",
        },
        "output_parquet": str(parquet_path) if not features.empty else None,
        "output_csv": str(csv_path) if not features.empty else None,
    }
    (output_path / "live_orderflow_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
