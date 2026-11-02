"""Public REST client for Delta Exchange India historical candles."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import time
from typing import Any

import pandas as pd
import requests

BASE_URL = "https://api.india.delta.exchange"
MAX_CANDLES_PER_RESPONSE = 2000
RESOLUTION_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600,
    "12h": 43200, "1d": 86400, "1w": 604800,
}


@dataclass
class DeltaPublicClient:
    base_url: str = BASE_URL
    timeout: int = 30
    max_retries: int = 6
    request_pause_seconds: float = 0.10

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "DMPE-research/0.2 (public market-data client)",
        })

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}{path}"
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                if response.status_code == 429:
                    reset_ms = response.headers.get("X-RATE-LIMIT-RESET")
                    retry_after = response.headers.get("Retry-After")
                    wait_seconds = (
                        max(float(reset_ms) / 1000.0, 1.0) if reset_ms
                        else max(float(retry_after), 1.0) if retry_after
                        else min(2 ** attempt, 30)
                    )
                    last_error = RuntimeError("HTTP 429 rate limited")
                    if attempt == self.max_retries - 1:
                        break
                    time.sleep(wait_seconds + 0.25)
                    continue

                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict) or payload.get("success") is False:
                    raise RuntimeError(f"Unexpected Delta API response: {payload}")
                return payload
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                last_error = exc
                if attempt == self.max_retries - 1:
                    break
                time.sleep(min(2 ** attempt, 20))

        raise RuntimeError(f"Delta API request failed after retries: {last_error}") from last_error

    def fetch_candle_window(
        self, symbol: str, resolution: str, start_epoch: int, end_epoch: int
    ) -> pd.DataFrame:
        if resolution not in RESOLUTION_SECONDS:
            raise ValueError(f"Unsupported resolution {resolution!r}")
        if start_epoch >= end_epoch:
            raise ValueError("start_epoch must be earlier than end_epoch")

        payload = self._get("/v2/history/candles", {
            "resolution": resolution,
            "symbol": symbol,
            "start": int(start_epoch),
            "end": int(end_epoch),
        })
        rows = payload.get("result", [])
        if not isinstance(rows, list):
            raise RuntimeError(f"Unexpected candle result format: {type(rows).__name__}")
        if len(rows) > MAX_CANDLES_PER_RESPONSE:
            raise RuntimeError(
                f"API returned {len(rows)} rows; documented maximum is {MAX_CANDLES_PER_RESPONSE}"
            )
        columns = ["time", "open", "high", "low", "close", "volume"]
        if not rows:
            return pd.DataFrame(columns=columns)

        frame = pd.DataFrame(rows)
        required = {"time", "open", "high", "low", "close"}
        missing = required.difference(frame.columns)
        if missing:
            raise RuntimeError(f"Candle response is missing columns: {sorted(missing)}")
        if "volume" not in frame.columns:
            frame["volume"] = pd.NA

        frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True, errors="coerce")
        for column in ("open", "high", "low", "close", "volume"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["time", "open", "high", "low", "close"])
        return (
            frame[columns]
            .sort_values("time")
            .drop_duplicates("time")
            .reset_index(drop=True)
        )

    def backfill_candles(
        self, symbol: str, resolution: str, start_epoch: int, end_epoch: int
    ) -> pd.DataFrame:
        if resolution not in RESOLUTION_SECONDS:
            raise ValueError(f"Unsupported resolution {resolution!r}")
        if start_epoch >= end_epoch:
            raise ValueError("start_epoch must be earlier than end_epoch")

        step_seconds = RESOLUTION_SECONDS[resolution]
        cursor = int(start_epoch)
        end_epoch = int(end_epoch)
        # A window spanning 1,999 intervals can contain at most 2,000 aligned candles.
        window_seconds = step_seconds * (MAX_CANDLES_PER_RESPONSE - 1)
        frames: list[pd.DataFrame] = []

        while cursor < end_epoch:
            window_end = min(end_epoch, cursor + window_seconds)
            chunk = self.fetch_candle_window(symbol, resolution, cursor, window_end)
            if not chunk.empty:
                frames.append(chunk)
            cursor = window_end + 1
            if self.request_pause_seconds > 0:
                time.sleep(self.request_pause_seconds)

        if not frames:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
        return (
            pd.concat(frames, ignore_index=True)
            .sort_values("time")
            .drop_duplicates("time")
            .reset_index(drop=True)
        )


def utc_epoch_days_ago(days: int) -> tuple[int, int]:
    if days <= 0:
        raise ValueError("days must be a positive integer")
    end = int(datetime.now(timezone.utc).timestamp())
    return end - days * 86400, end
