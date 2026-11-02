"""Record public Delta Exchange WebSocket events as daily JSONL files."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
from typing import Any
import websocket

WS_URL = "wss://public-socket.india.delta.exchange"

class LiveRecorder:
    def __init__(self, symbol: str = "BTCUSD", output_dir: str | Path = "data/raw/live") -> None:
        self.symbol = symbol
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._file_date: str | None = None
        self._lock = threading.Lock()

    def _write(self, event: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc)
        date_key = now.strftime("%Y%m%d")
        if self._file is None or self._file_date != date_key:
            if self._file is not None:
                self._file.close()
            self._file = (self.output_dir / f"delta_{self.symbol}_{date_key}.jsonl").open("a", encoding="utf-8")
            self._file_date = date_key
        record = {"received_at": now.isoformat(), "symbol": self.symbol, "event": event}
        self._file.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\\n")
        self._file.flush()

    def on_open(self, ws: websocket.WebSocketApp) -> None:
        subscription = {"type": "subscribe", "payload": {"channels": [
            {"name": "trades", "symbols": [self.symbol]},
            {"name": "ob_l2", "symbols": [self.symbol]},
            {"name": "candlestick_1m", "symbols": [self.symbol]},
        ]}}
        ws.send(json.dumps(subscription))
        print(f"Connected; subscribed to trades, ob_l2 and 1m candles for {self.symbol}.")

    def on_message(self, ws: websocket.WebSocketApp, message: str) -> None:
        try:
            event = json.loads(message)
        except json.JSONDecodeError:
            event = {"type": "unparsed", "raw": message}
        with self._lock:
            self._write(event if isinstance(event, dict) else {"payload": event})

    def on_error(self, ws: websocket.WebSocketApp, error: Any) -> None:
        print(f"WebSocket error: {error}")

    def on_close(self, ws: websocket.WebSocketApp, close_status_code: Any, close_msg: Any) -> None:
        print(f"WebSocket closed: code={close_status_code}, message={close_msg}")

    def run_forever(self) -> None:
        print(f"Starting public market-data recorder for {self.symbol}; Ctrl+C to stop.")
        try:
            while True:
                app = websocket.WebSocketApp(
                    WS_URL, on_open=self.on_open, on_message=self.on_message,
                    on_error=self.on_error, on_close=self.on_close,
                )
                app.run_forever(ping_interval=20, ping_timeout=10)
                print("Disconnected; reconnecting in 5 seconds...")
                time.sleep(5)
        except KeyboardInterrupt:
            print("Stopping live recorder.")
        finally:
            if self._file is not None:
                self._file.close()
                self._file = None
