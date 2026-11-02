"""DMPE Quant Desk — research and paper-trading interface.
It reads local DMPE Parquet/JSONL data. No execution or unvalidated signals.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.getenv("DMPE_DATA_DIR", ROOT / "data" / "raw"))
HISTORY_DIR = DATA_ROOT / "history"
LIVE_DIR = DATA_ROOT / "live"
METADATA_DIR = DATA_ROOT / "metadata"
FEATURE_DIRS = [
    Path(os.getenv("DMPE_FEATURE_DIR", ROOT / "stage1_output" / "processed")),
    ROOT / "data" / "processed",
    ROOT / "processed",
]
BASELINE_REPORT_PATHS = [
    Path(os.getenv("DMPE_STAGE2_DIR", ROOT / "stage2_output")) / "baseline_evaluation.json",
    ROOT / "artifacts" / "stage2_output" / "baseline_evaluation.json",
]
WALK_FORWARD_REPORT_PATHS = [
    Path(os.getenv("DMPE_STAGE3_DIR", ROOT / "stage3_output")) / "walk_forward_report.json",
    ROOT / "artifacts" / "stage3_output" / "walk_forward_report.json",
]
ORDERFLOW_FEATURE_PATH = Path(os.getenv(
    "DMPE_ORDERFLOW_FILE",
    ROOT / "orderflow_output" / "processed" / "dmpe_BTCUSD_live_orderflow_1m.parquet",
))

TF_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "3h": 10800, "4h": 14400,
    "6h": 21600, "12h": 43200, "1d": 86400,
}
TF_RULES = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min",
    "30m": "30min", "1h": "1h", "2h": "2h", "3h": "3h",
    "4h": "4h", "6h": "6h", "12h": "12h", "1d": "1D",
}
OHLCV = ["open", "high", "low", "close", "volume"]

st.set_page_config(page_title="DMPE | Quant Desk", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.25rem; padding-bottom: 2rem;}
      .desk-banner {padding: 0.8rem 1rem; border: 1px solid #334155;
        border-radius: 10px; background: #111827; margin-bottom: 1rem;}
      .muted {color: #94a3b8; font-size: 0.86rem;}
      div[data-testid="stMetric"] {background: #111827; padding: 0.7rem;
        border: 1px solid #263244; border-radius: 9px;}
    </style>
    """, unsafe_allow_html=True,
)


def discover_price_files() -> list[Path]:
    if not HISTORY_DIR.exists():
        return []
    return [
        p for p in sorted(HISTORY_DIR.glob("*.parquet"))
        if not any(tag in p.stem.upper() for tag in ("MARK", "FUNDING"))
        and not p.stem.upper().startswith("OI")
    ]


def normalize_candles(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("The selected file contains no rows.")
    aliases = {str(c).strip().lower(): c for c in frame.columns}
    time_col = next((aliases[k] for k in ("time", "timestamp", "datetime", "date") if k in aliases), None)
    if time_col is None:
        raise ValueError("No time/timestamp column was found.")
    out = frame.rename(
        columns={aliases[k]: k for k in ("open", "high", "low", "close", "volume") if k in aliases}
    ).copy()
    missing = {"open", "high", "low", "close"}.difference(out.columns)
    if missing:
        raise ValueError(f"Missing candle columns: {sorted(missing)}")
    out["time"] = pd.to_datetime(frame[time_col], utc=True, errors="coerce")
    for col in OHLCV:
        if col not in out.columns:
            out[col] = 0.0 if col == "volume" else pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["time", "open", "high", "low", "close"])
    out = out.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    bad = (
        (out["low"] > out["high"])
        | (out["open"] < out["low"]) | (out["open"] > out["high"])
        | (out["close"] < out["low"]) | (out["close"] > out["high"])
    )
    out = out.loc[~bad].copy()
    if out.empty:
        raise ValueError("No valid OHLC rows remain after parsing.")
    return out[["time", *OHLCV]].reset_index(drop=True)


@st.cache_data(ttl=20, show_spinner=False)
def read_candle_path(path_str: str, modified_ns: int) -> pd.DataFrame:
    del modified_ns
    return normalize_candles(pd.read_parquet(path_str))


def read_uploaded_candles(upload) -> pd.DataFrame:
    raw = upload.getvalue()
    frame = pd.read_csv(io.BytesIO(raw)) if upload.name.lower().endswith(".csv") else pd.read_parquet(io.BytesIO(raw))
    return normalize_candles(frame)


def resample_candles(frame: pd.DataFrame, target_tf: str, source_tf: str) -> pd.DataFrame:
    if target_tf not in TF_RULES or source_tf not in TF_SECONDS:
        raise ValueError("Unsupported timeframe.")
    if TF_SECONDS[target_tf] < TF_SECONDS[source_tf]:
        raise ValueError(
            f"{target_tf} is finer than the source data ({source_tf}); "
            "missing intrabar detail cannot be reconstructed."
        )
    if target_tf == source_tf:
        return frame.copy()
    aggregated = frame.set_index("time").resample(
        TF_RULES[target_tf], label="left", closed="left"
    ).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    })
    return aggregated.dropna(subset=["open", "high", "low", "close"]).reset_index()


def timeframe_from_name(filename: str) -> str:
    stem = Path(filename).stem.lower()
    for tf in sorted(TF_SECONDS, key=len, reverse=True):
        if stem.endswith("_" + tf):
            return tf
    return "1m"


def pct_change_at(frame: pd.DataFrame, minutes: int, tf: str) -> float | None:
    if minutes * 60 < TF_SECONDS[tf]:
        return None
    bars = max(int(round(minutes * 60 / TF_SECONDS[tf])), 1)
    if len(frame) <= bars:
        return None
    old, last = float(frame["close"].iloc[-bars - 1]), float(frame["close"].iloc[-1])
    return (last / old - 1.0) * 100.0 if old else None


def calculate_metrics(frame: pd.DataFrame, tf: str) -> dict[str, Any]:
    close, high, low = frame["close"].astype(float), frame["high"].astype(float), frame["low"].astype(float)
    previous = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous).abs(), (low - previous).abs()], axis=1
    ).max(axis=1)
    atr = true_range.rolling(14, min_periods=14).mean()
    returns = close.pct_change().dropna()
    bars_per_day = max(1, int(86400 / TF_SECONDS[tf]))
    rv = float(returns.tail(bars_per_day).std() * bars_per_day ** 0.5 * 100) if len(returns) > 2 else None
    atr_now = float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else None
    last_price = float(close.iloc[-1])
    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
    return {
        "price": last_price,
        "change_24h": pct_change_at(frame, 1440, tf),
        "return_15m": pct_change_at(frame, 15, tf),
        "return_1h": pct_change_at(frame, 60, tf),
        "return_3h": pct_change_at(frame, 180, tf),
        "atr": atr_now,
        "atr_pct": atr_now / last_price * 100 if atr_now is not None and last_price else None,
        "rv_24h": rv, "ema20": ema20, "ema50": ema50,
    }


def fmt_pct(value: float | None) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:+.2f}%"


def build_price_chart(frame: pd.DataFrame, max_bars: int) -> go.Figure:
    view = frame.tail(max_bars).copy()
    close = view["close"].astype(float)
    view["ema20"] = close.ewm(span=20, adjust=False).mean()
    view["ema50"] = close.ewm(span=50, adjust=False).mean()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.035,
                        row_heights=[0.76, 0.24])
    fig.add_trace(go.Candlestick(
        x=view["time"], open=view["open"], high=view["high"], low=view["low"],
        close=view["close"], name="BTCUSD",
        increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=view["time"], y=view["ema20"], name="EMA 20",
                             line={"color": "#38bdf8", "width": 1.2}), row=1, col=1)
    fig.add_trace(go.Scatter(x=view["time"], y=view["ema50"], name="EMA 50",
                             line={"color": "#f59e0b", "width": 1.2}), row=1, col=1)
    fig.add_trace(go.Bar(x=view["time"], y=view["volume"], name="Volume",
                         marker_color="#64748b", opacity=0.75), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=640,
                      margin={"l": 10, "r": 10, "t": 30, "b": 10},
                      xaxis_rangeslider_visible=False, hovermode="x unified",
                      legend={"orientation": "h", "y": 1.02, "x": 0})
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def read_jsonl_tail(path: Path, limit: int = 1500) -> list[dict[str, Any]]:
    rows: deque[dict[str, Any]] = deque(maxlen=limit)
    if not path.exists():
        return []
    try:
        # Read backwards in blocks: daily live files can become large, and
        # the UI refreshes this tail repeatedly.
        with path.open("rb") as handle:
            end = handle.seek(0, os.SEEK_END)
            chunks = bytearray()
            newline_count = 0
            while end > 0 and newline_count <= limit:
                size = min(65536, end)
                end -= size
                handle.seek(end)
                block = handle.read(size)
                chunks[:0] = block
                newline_count += block.count(b"\n")
        lines = bytes(chunks).decode("utf-8", errors="replace").splitlines()[-limit:]
        for line in lines:
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
            except json.JSONDecodeError:
                continue
    except OSError:
        return []
    return list(rows)


def unwrap_event(record: dict[str, Any]) -> dict[str, Any]:
    event = record.get("event", record)
    return event if isinstance(event, dict) else {"payload": event}


def event_kind(event: dict[str, Any]) -> str:
    return str(event.get("type") or event.get("channel") or event.get("event_type") or "unknown").lower()


def is_kind(event: dict[str, Any], fragments: tuple[str, ...]) -> bool:
    return any(fragment in event_kind(event) for fragment in fragments)


def book_depth_imbalance(event: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    def total(levels: Any) -> float | None:
        if not isinstance(levels, list):
            return None
        sizes = []
        for level in levels:
            try:
                if isinstance(level, dict):
                    value = level.get("size", level.get("quantity", level.get("q")))
                elif isinstance(level, (list, tuple)) and len(level) >= 2:
                    value = level[1]
                else:
                    continue
                sizes.append(float(value))
            except (TypeError, ValueError):
                continue
        return sum(sizes) if sizes else None

    bids, asks = total(event.get("b")), total(event.get("a"))
    if bids is None or asks is None or bids + asks == 0:
        return bids, asks, None
    return bids, asks, (bids - asks) / (bids + asks)




def infer_trade_aggressor(event: dict[str, Any]) -> str:
    """Infer taker side from Delta's documented buyer_role code only."""
    buyer_role = str(event.get("r", "")).lower()
    if buyer_role == "t":
        return "BUY taker"
    if buyer_role == "m":
        return "SELL taker"
    return "Unknown"


def latest_events() -> tuple[list[dict[str, Any]], Path | None]:
    files = sorted(LIVE_DIR.glob("delta_*.jsonl")) if LIVE_DIR.exists() else []
    if not files:
        return [], None
    records: list[dict[str, Any]] = []
    for path in files[-2:]:
        records.extend(read_jsonl_tail(path, 1200))
    records.sort(key=lambda row: str(row.get("received_at", "")))
    return records[-1500:], files[-1]


def load_report(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None




def discover_feature_dataset() -> tuple[Path | None, Path | None]:
    for directory in FEATURE_DIRS:
        data_path = directory / "dmpe_BTCUSD_1m_features.parquet"
        schema_path = directory / "feature_schema.json"
        if data_path.exists() and schema_path.exists():
            return data_path, schema_path
    return None, None


@st.cache_data(ttl=60, show_spinner=False)
def read_feature_columns(path_str: str, modified_ns: int, columns: tuple[str, ...]) -> pd.DataFrame:
    del modified_ns
    return pd.read_parquet(path_str, columns=["time", *columns])




def discover_baseline_report() -> Path | None:
    return next((path for path in BASELINE_REPORT_PATHS if path.exists()), None)


def baseline_scorecard(report: dict[str, Any]) -> pd.DataFrame:
    """Flatten model test scores without ranking variants by a reused test set."""
    rows = []
    for item in report.get("results", []):
        if item.get("status") != "completed":
            continue
        for model_name, metric_key, signal_key in (
            ("Logistic regression", "logistic_regression_test", "logistic_test_confidence_gated_signals"),
            ("Histogram gradient boosting", "hist_gradient_boosting_test", "hist_gradient_boosting_test_confidence_gated_signals"),
        ):
            metrics = item.get(metric_key) or {}
            signals = item.get(signal_key) or {}
            if not metrics:
                continue
            interval = signals.get("correct_share_wilson_95_ci")
            rows.append({
                "Model": model_name,
                "Horizon": item.get("horizon"),
                "Barrier × RV": item.get("barrier_multiple"),
                "Test rows": metrics.get("rows"),
                "Balanced accuracy": metrics.get("balanced_accuracy"),
                "Macro F1": metrics.get("macro_f1"),
                "Log loss": metrics.get("multiclass_log_loss"),
                "Signals": signals.get("signals", 0),
                "Signal coverage": signals.get("coverage"),
                "Correct signal share": signals.get("correct_directional_share_of_signals"),
                "Approx. 95% CI low": interval[0] if interval else None,
                "Approx. 95% CI high": interval[1] if interval else None,
                "Wrong direction share": signals.get("wrong_direction_share_of_signals"),
                "No-touch share": signals.get("no_barrier_touch_share_of_signals"),
                "Threshold": signals.get("threshold_selected_on_validation"),
            })
    return pd.DataFrame(rows)




def discover_walk_forward_report() -> Path | None:
    return next((path for path in WALK_FORWARD_REPORT_PATHS if path.exists()), None)


def walk_forward_scorecard(report: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in report.get("results", []):
        if item.get("status") != "completed":
            continue
        metrics = item.get("combined_untouched_test_metrics") or {}
        signals = item.get("combined_confidence_gated_test_signals") or {}
        interval = signals.get("correct_share_wilson_95_ci")
        rows.append({
            "Horizon": item.get("horizon"),
            "Barrier × RV": item.get("barrier_multiple"),
            "Folds": item.get("completed_folds"),
            "Test rows": metrics.get("rows"),
            "Balanced accuracy": metrics.get("balanced_accuracy"),
            "Macro F1": metrics.get("macro_f1"),
            "Directional calls": signals.get("signals", 0),
            "Signal coverage": signals.get("coverage"),
            "Correct signal share": signals.get("correct_directional_share_of_signals"),
            "Approx. 95% CI low": interval[0] if interval else None,
            "Approx. 95% CI high": interval[1] if interval else None,
            "Wrong direction share": signals.get("wrong_direction_share_of_signals"),
            "No-touch share": signals.get("no_barrier_touch_share_of_signals"),
            "Folds with validation threshold": signals.get("folds_with_selected_threshold"),
        })
    return pd.DataFrame(rows)




@st.cache_data(ttl=30, show_spinner=False)
def read_orderflow_tail(path_str: str, modified_ns: int) -> pd.DataFrame:
    del modified_ns
    columns = [
        "time", "trade_count", "trade_buy_taker_volume", "trade_sell_taker_volume",
        "trade_volume_imbalance", "trade_total_volume", "l2_snapshot_count",
        "l2_spread_bps_last", "l2_depth_imbalance_5_last",
        "l2_microprice_mid_bps_last",
    ]
    available = pd.read_parquet(path_str, columns=columns)
    return available.tail(60).reset_index(drop=True)


def main() -> None:
    st.title("DMPE Quant Desk")
    st.caption("Dynamic Market Prediction Engine · BTCUSD perpetual · research workspace")
    st.markdown(
        '<div class="desk-banner"><b>RESEARCH MODE · EXECUTION DISABLED</b>'
        '<div class="muted">The desk visualizes and audits data. No model has passed '
        'walk-forward validation, so the official DMPE decision remains FLAT (0). '
        'No orders can be placed from this app.</div></div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Desk controls")
        st.caption(f"Data directory: {DATA_ROOT}")
        st.caption("Venue: Delta Exchange India · BTCUSD")
        source_files = discover_price_files()
        uploaded = st.file_uploader("Load candle file (CSV or Parquet)",
                                    type=["csv", "parquet"], accept_multiple_files=False)
        selected_path = None
        source_tf = "1m"
        candle_data = None
        if uploaded is not None:
            try:
                candle_data = read_uploaded_candles(uploaded)
                source_tf = timeframe_from_name(uploaded.name)
                st.success(f"Loaded {len(candle_data):,} rows from upload")
            except Exception as exc:
                st.error(f"Upload could not be read: {exc}")
        elif source_files:
            names = [p.name for p in source_files]
            selected_name = st.selectbox("Historical dataset", names, index=len(names) - 1)
            selected_path = next(p for p in source_files if p.name == selected_name)
            source_tf = timeframe_from_name(selected_path.name)
            try:
                candle_data = read_candle_path(str(selected_path), selected_path.stat().st_mtime_ns)
            except Exception as exc:
                st.error(f"Could not read {selected_path.name}: {exc}")
        else:
            st.info("No local traded-price Parquet file found. Run the backfill workflow or upload candles.")
        target_options = [tf for tf, seconds in TF_SECONDS.items() if seconds >= TF_SECONDS[source_tf]]
        target_options = target_options or [source_tf]
        default_tf = "15m" if "15m" in target_options else target_options[0]
        timeframe = st.selectbox("Desk timeframe", target_options, index=target_options.index(default_tf))
        chart_bars = st.select_slider("Chart bars", options=[300, 600, 1000, 2000, 4000], value=1000)
        auto_refresh = st.toggle("Auto-refresh live tape", value=True)
        st.caption("The live pane reads the local JSONL recorder; it does not start it.")

    if candle_data is None:
        st.subheader("Welcome to the DMPE desk")
        st.write(
            "The interface is ready. Run the historical backfill, download its GitHub Actions "
            "artifact, and extract data/raw/history into this repository. You can also upload "
            "a candle CSV or Parquet file from the sidebar."
        )
        st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")
        st.stop()

    try:
        candles = resample_candles(candle_data, timeframe, source_tf)
    except Exception as exc:
        st.error(str(exc))
        st.stop()
    if candles.empty:
        st.warning("No candles are available for the selected timeframe.")
        st.stop()

    metrics = calculate_metrics(candles, timeframe)
    latest_time = candles["time"].iloc[-1]
    age_mins = max((datetime.now(timezone.utc) - latest_time.to_pydatetime()).total_seconds() / 60, 0)
    tabs = st.tabs(["Market Desk", "Live Tape", "Data Quality", "Feature Lab", "Model & Algo"])

    with tabs[0]:
        st.subheader(f"BTCUSD · {timeframe} · Market overview")
        top = st.columns(5)
        top[0].metric("Last price", f"USD {metrics['price']:,.2f}")
        top[1].metric("24h change", fmt_pct(metrics["change_24h"]))
        top[2].metric("15m return", fmt_pct(metrics["return_15m"]))
        top[3].metric("1h return", fmt_pct(metrics["return_1h"]))
        top[4].metric("3h return", fmt_pct(metrics["return_3h"]))
        st.plotly_chart(build_price_chart(candles, chart_bars), use_container_width=True,
                        config={"displaylogo": False, "scrollZoom": True})
        secondary = st.columns(4)
        secondary[0].metric("ATR(14)", "—" if metrics["atr"] is None else f"USD {metrics['atr']:,.2f}")
        secondary[1].metric("ATR / price", "—" if metrics["atr_pct"] is None else f"{metrics['atr_pct']:.3f}%")
        secondary[2].metric("24h realized vol*", "—" if metrics["rv_24h"] is None else f"{metrics['rv_24h']:.2f}%")
        trend = "EMA20 > EMA50" if metrics["ema20"] > metrics["ema50"] else "EMA20 ≤ EMA50"
        secondary[3].metric("Trend descriptor", trend)
        dataset_name = selected_path.name if selected_path else uploaded.name
        st.caption(
            f"Dataset: {dataset_name} · {len(candles):,} bars · "
            f"last candle {latest_time.isoformat()} · data age {age_mins:.1f} min. "
            "*Realized volatility is a rolling descriptive estimate, not a forecast."
        )
        with st.expander("Recent candle rows"):
            st.dataframe(candles.tail(100).sort_values("time", ascending=False),
                         use_container_width=True, hide_index=True)
        st.download_button("Download selected candles as CSV",
                           data=candles.to_csv(index=False).encode("utf-8"),
                           file_name=f"dmpe_BTCUSD_{timeframe}.csv", mime="text/csv")

    with tabs[1]:
        st.subheader("Live market-data tape")
        if not LIVE_DIR.exists():
            st.info("No data/raw/live folder found. Start the recorder separately:")
            st.code("python scripts/capture_live.py --symbol BTCUSD", language="bash")
        else:
            @st.fragment(run_every=5 if auto_refresh else None)
            def render_live_tape() -> None:
                records, newest_file = latest_events()
                if not records:
                    st.info("No JSONL events found yet. Start the public WebSocket recorder.")
                    return
                events = [(record, unwrap_event(record)) for record in records]
                c1, c2, c3, c4 = st.columns(4)
                trade_events = [(r, e) for r, e in events if is_kind(e, ("trade",))]
                book_events = [(r, e) for r, e in events if is_kind(e, ("ob_l2", "orderbook", "order_book"))]
                mark_events = [(r, e) for r, e in events if is_kind(e, ("mark_price", "markprice"))]
                system_events = [(r, e) for r, e in events if is_kind(e, ("system_status",))]
                product_events = [(r, e) for r, e in events if is_kind(e, ("product_updates",))]
                c1.metric("Trades captured", f"{len(trade_events):,}")
                c2.metric("Order-book updates", f"{len(book_events):,}")
                c3.metric("Mark-price updates", f"{len(mark_events):,}")
                mark_value = mark_events[-1][1].get("p") if mark_events else None
                try:
                    mark_value = float(mark_value) if mark_value is not None else None
                except (TypeError, ValueError):
                    mark_value = None
                c4.metric("Latest mark price", "—" if mark_value is None else f"USD {mark_value:,.2f}")
                status_cols = st.columns(2)
                system_status = system_events[-1][1].get("status") if system_events else None
                product_event = product_events[-1][1] if product_events else {}
                product_info = product_event.get("product", {})
                product_status = product_info.get("trading_status") if isinstance(product_info, dict) else None
                status_cols[0].metric("Exchange system status", str(system_status or "Not reported"))
                status_cols[1].metric("BTCUSD trading status", str(product_status or "Not reported"))
                if system_status and str(system_status).lower() not in {"live", "operational"}:
                    st.error(f"Exchange system status is {system_status}. Future execution must be locked.")
                if product_status and str(product_status).lower() != "operational":
                    st.error(f"BTCUSD product status is {product_status}. Future execution must be locked.")
                st.caption(
                    "Status feeds are advisory safety gates. Missing or stale status data must not be treated as operational. "
                    "This dashboard does not submit orders."
                )
                st.caption(
                    f"Source: {newest_file.name if newest_file else 'unknown'} · "
                    f"latest received {records[-1].get('received_at', 'timestamp unavailable')} · "
                    f"{len(records):,} recent records loaded"
                )
                if book_events:
                    _, book = book_events[-1]
                    bid_size, ask_size, imbalance = book_depth_imbalance(book)
                    b1, b2, b3 = st.columns(3)
                    b1.metric("Displayed bid size", "—" if bid_size is None else f"{bid_size:,.4f}")
                    b2.metric("Displayed ask size", "—" if ask_size is None else f"{ask_size:,.4f}")
                    b3.metric("Top-book size imbalance", "—" if imbalance is None else f"{imbalance:+.1%}")
                    st.caption(
                        "Imbalance = (sum bid sizes − sum ask sizes) / total displayed size. "
                        "Snapshot descriptor only, not a standalone signal."
                    )
                else:
                    st.info("No L2 order-book event was recognized in the current JSONL tail.")
                if trade_events:
                    rows = [{
                        "received_at": r.get("received_at"),
                        "price": e.get("p", e.get("price")),
                        "size": e.get("s", e.get("size")),
                        "buyer_role_code": e.get("r"),
                        "inferred_aggressor": infer_trade_aggressor(e),
                        "event_time": e.get("ts", e.get("t")),
                    } for r, e in trade_events[-30:][::-1]]
                    st.markdown("**Recent trades**")
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    st.caption(
                        "Delta documents r as buyer_role: t = buyer taker (buy aggressor); "
                        "m = buyer maker (sell aggressor). Unknown codes remain unclassified."
                    )

                st.markdown("**Aggregated live order-flow features**")
                if ORDERFLOW_FEATURE_PATH.exists():
                    try:
                        flow = read_orderflow_tail(
                            str(ORDERFLOW_FEATURE_PATH), ORDERFLOW_FEATURE_PATH.stat().st_mtime_ns
                        )
                        if not flow.empty:
                            latest = flow.iloc[-1]
                            fcols = st.columns(5)
                            fcols[0].metric("Aggressor volume imbalance",
                                "—" if pd.isna(latest["trade_volume_imbalance"]) else f"{latest['trade_volume_imbalance']:+.1%}")
                            fcols[1].metric("Buy-taker volume",
                                "—" if pd.isna(latest["trade_buy_taker_volume"]) else f"{latest['trade_buy_taker_volume']:,.2f}")
                            fcols[2].metric("Sell-taker volume",
                                "—" if pd.isna(latest["trade_sell_taker_volume"]) else f"{latest['trade_sell_taker_volume']:,.2f}")
                            fcols[3].metric("Top-5 depth imbalance",
                                "—" if pd.isna(latest["l2_depth_imbalance_5_last"]) else f"{latest['l2_depth_imbalance_5_last']:+.1%}")
                            fcols[4].metric("Spread (bps)",
                                "—" if pd.isna(latest["l2_spread_bps_last"]) else f"{latest['l2_spread_bps_last']:.2f}")
                            st.caption(
                                f"Aggregated minute close: {latest['time']} · "
                                f"trade prints: {int(latest['trade_count']) if pd.notna(latest['trade_count']) else 0} · "
                                f"L2 snapshots: {int(latest['l2_snapshot_count']) if pd.notna(latest['l2_snapshot_count']) else 0}. "
                                "These are observed microstructure descriptors, not validated trade signals."
                            )
                            st.dataframe(flow.tail(10).sort_values("time", ascending=False),
                                         use_container_width=True, hide_index=True)
                    except Exception as exc:
                        st.error(f"Could not read aggregated order-flow file: {exc}")
                else:
                    st.info(
                        "No aggregated order-flow feature file found. Run "
                        "scripts/build_live_orderflow_dataset.py or the DMPE live order-flow "
                        "feature workflow after downloading a capture artifact."
                    )

                raw_rows = [{
                    "received_at": r.get("received_at"),
                    "kind": event_kind(e),
                    "symbol": e.get("sy", e.get("symbol")),
                    "payload_preview": json.dumps(e, ensure_ascii=False)[:350],
                } for r, e in events[-25:][::-1]]
                with st.expander("Recent raw events"):
                    st.dataframe(pd.DataFrame(raw_rows), use_container_width=True, hide_index=True)
            render_live_tape()

    with tabs[2]:
        st.subheader("Data quality & coverage")
        backfill = load_report(METADATA_DIR / "backfill_report.json")
        audit_path = METADATA_DIR / "data_quality_report.csv"
        a1, a2, a3 = st.columns(3)
        a1.metric("Rows in selected chart", f"{len(candles):,}")
        a2.metric("First candle", candles["time"].iloc[0].strftime("%Y-%m-%d %H:%M UTC"))
        a3.metric("Last candle age", f"{age_mins:.1f} min")
        if backfill:
            st.markdown("**Historical backfill report**")
            st.caption(
                f"Requested {backfill.get('requested_days', 'n/a')} days · "
                f"created {backfill.get('created_at_utc', 'timestamp unavailable')}"
            )
            series = pd.DataFrame(backfill.get("series", []))
            if not series.empty:
                st.dataframe(series, use_container_width=True, hide_index=True)
            if backfill.get("errors"):
                st.warning(f"{len(backfill['errors'])} series reported download errors.")
        else:
            st.info("No backfill report found at data/raw/metadata/backfill_report.json.")
        if audit_path.exists():
            st.markdown("**Candle audit report**")
            try:
                st.dataframe(pd.read_csv(audit_path), use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(f"Could not read audit report: {exc}")
        else:
            st.info("No audit report found. Run python scripts/audit_history.py --directory data/raw/history.")
        st.markdown("**Current file checks**")
        st.write({
            "Rows loaded": int(len(candle_data)),
            "Duplicate timestamps after normalization": int(candle_data["time"].duplicated().sum()),
            "Missing values in OHLC": int(candle_data[["open", "high", "low", "close"]].isna().sum().sum()),
            "Chronological order": bool(candle_data["time"].is_monotonic_increasing),
        })

    with tabs[3]:
        st.subheader("Feature Lab · point-in-time data vs. future targets")
        feature_path, schema_path = discover_feature_dataset()
        if feature_path is None or schema_path is None:
            st.info(
                "No processed feature dataset found. Download the "
                "dmpe-stage1-feature-engineering artifact and extract its processed/ "
                "folder into stage1_output/processed (or set DMPE_FEATURE_DIR)."
            )
        else:
            schema = load_report(schema_path)
            if not schema:
                st.error("Feature schema could not be read.")
            else:
                feature_names = schema.get("feature_columns", [])
                label_names = schema.get("label_columns", [])
                if not feature_names or not label_names:
                    st.warning("The feature schema does not contain the expected feature/label contract.")
                else:
                    selectors = st.columns(2)
                    selected_feature = selectors[0].selectbox(
                        "Model input candidate (past/current data only)",
                        feature_names, key="feature_lab_input",
                    )
                    selected_target = selectors[1].selectbox(
                        "Research target (future data; never a model input)",
                        label_names, key="feature_lab_target",
                    )
                    try:
                        feature_view = read_feature_columns(
                            str(feature_path), feature_path.stat().st_mtime_ns,
                            (selected_feature, selected_target),
                        )
                        feature_values = pd.to_numeric(feature_view[selected_feature], errors="coerce")
                        target_values = pd.to_numeric(feature_view[selected_target], errors="coerce")
                        nonnull = feature_values.dropna()
                        target_nonnull = target_values.dropna()
                        f1, f2, f3, f4 = st.columns(4)
                        f1.metric("Feature rows", f"{len(feature_view):,}")
                        f2.metric("Feature null rate", f"{feature_values.isna().mean():.2%}")
                        f3.metric("Target null rate", f"{target_values.isna().mean():.2%}")
                        f4.metric("Feature observations", f"{len(nonnull):,}")

                        st.markdown(f"**Feature: {selected_feature}**")
                        if not nonnull.empty:
                            stats = nonnull.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
                            st.dataframe(stats.rename("value").to_frame(), use_container_width=True)
                            plot_data = feature_view.loc[feature_values.notna(), ["time"]].copy()
                            plot_data[selected_feature] = feature_values.loc[feature_values.notna()].to_numpy()
                            if len(plot_data) > 6000:
                                stride = max(len(plot_data) // 6000, 1)
                                plot_data = plot_data.iloc[::stride]
                            line = go.Figure(go.Scattergl(
                                x=plot_data["time"], y=plot_data[selected_feature],
                                mode="lines", name=selected_feature,
                            ))
                            line.update_layout(
                                template="plotly_dark", height=360,
                                margin={"l": 10, "r": 10, "t": 25, "b": 10},
                                title=f"{selected_feature} · sampled time series",
                            )
                            st.plotly_chart(line, use_container_width=True)
                            hist = go.Figure(go.Histogram(x=nonnull, nbinsx=70, name=selected_feature))
                            hist.update_layout(
                                template="plotly_dark", height=300,
                                margin={"l": 10, "r": 10, "t": 25, "b": 10},
                                title=f"{selected_feature} · distribution",
                            )
                            st.plotly_chart(hist, use_container_width=True)
                        else:
                            st.warning("The selected feature is entirely null in this dataset.")

                        st.markdown(f"**Future target: {selected_target}**")
                        if not target_nonnull.empty:
                            target_stats = target_nonnull.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
                            st.dataframe(target_stats.rename("value").to_frame(), use_container_width=True)
                            st.caption(
                                f"Positive: {(target_nonnull > 0).mean():.1%} · "
                                f"Negative: {(target_nonnull < 0).mean():.1%} · "
                                f"Exactly zero: {(target_nonnull == 0).mean():.1%}"
                            )
                        st.caption(
                            "Future targets are outcomes for research evaluation only. They must be "
                            "excluded from training features to avoid look-ahead leakage."
                        )
                    except Exception as exc:
                        st.error(f"Could not read the selected feature columns: {exc}")

    with tabs[4]:
        st.subheader("Model gate & algo desk readiness")
        gate = st.columns(3)
        gate[0].metric("Active model", "NOT REGISTERED")
        gate[1].metric("Validated signal", "FLAT (0)")
        gate[2].metric("Order execution", "DISABLED")
        st.warning(
            "The app intentionally does not convert EMA, returns, volatility or book imbalance "
            "into buy/sell calls. Those are candidate features, not a tested trading edge."
        )
        st.markdown("**DMPE signal contract**")
        st.code("+1  = LONG / hold long\n 0  = FLAT / exit\n-1  = SHORT / hold short", language="text")
        st.markdown("**Release gates before any live algo connection**")
        gates = [
            ("Data", "History coverage, timestamps, gaps, duplicates and feed disconnects audited"),
            ("Features", "Point-in-time features with documented definitions and no look-ahead"),
            ("Labels", "Event-based labels with explicit horizon, barriers and overlap handling"),
            ("Model", "Chronological train/validation/test and walk-forward evaluation"),
            ("Economics", "Fees, spread, slippage, funding, latency and turnover included"),
            ("Risk", "Position sizing, exposure caps, daily loss limit, kill switch and stale-data lockout"),
            ("Paper", "Shadow/paper trading with live forward metrics before execution approval"),
        ]
        for title, description in gates:
            st.markdown(f"- **{title}:** {description}")
        report_path = discover_baseline_report()
        if report_path is None:
            st.info(
                "No baseline evaluation report found. Download the "
                "dmpe-baseline-evaluation artifact and extract baseline_evaluation.json "
                "into stage2_output/ (or set DMPE_STAGE2_DIR)."
            )
        else:
            baseline_report = load_report(report_path)
            if baseline_report:
                st.markdown("**Exploratory out-of-sample baseline scorecard**")
                st.caption(
                    f"Report created {baseline_report.get('created_at_utc', 'unknown')} · "
                    f"{baseline_report.get('feature_count_in_schema', 'n/a')} candidate inputs"
                )
                if baseline_report.get("selection_bias_warning"):
                    st.warning(baseline_report["selection_bias_warning"])
                scorecard = baseline_scorecard(baseline_report)
                if not scorecard.empty:
                    st.dataframe(
                        scorecard.style.format({
                            "Balanced accuracy": "{:.1%}",
                            "Macro F1": "{:.3f}",
                            "Log loss": "{:.3f}",
                            "Signal coverage": "{:.2%}",
                            "Correct signal share": "{:.1%}",
                            "Approx. 95% CI low": "{:.1%}",
                            "Approx. 95% CI high": "{:.1%}",
                            "Wrong direction share": "{:.1%}",
                            "No-touch share": "{:.1%}",
                            "Threshold": "{:.2f}",
                        }, na_rep="—"),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption(
                        "The confidence interval is an approximate binomial Wilson interval. "
                        "It does not correct for regime clustering or serial dependence."
                    )
                else:
                    st.warning("The baseline report contains no completed model evaluations.")
            else:
                st.error("The baseline evaluation JSON could not be read.")

        walk_forward_path = discover_walk_forward_report()
        if walk_forward_path is None:
            st.info(
                "No repeated walk-forward report found. Download the "
                "dmpe-walk-forward-evaluation artifact and extract walk_forward_report.json "
                "into stage3_output/ (or set DMPE_STAGE3_DIR)."
            )
        else:
            wf_report = load_report(walk_forward_path)
            if wf_report:
                st.markdown("**Repeated expanding-window walk-forward results**")
                st.caption(
                    f"Four 60-day test folds · report created "
                    f"{wf_report.get('created_at_utc', 'unknown')}"
                )
                wf_table = walk_forward_scorecard(wf_report)
                if not wf_table.empty:
                    st.dataframe(
                        wf_table.style.format({
                            "Balanced accuracy": "{:.1%}",
                            "Macro F1": "{:.3f}",
                            "Signal coverage": "{:.2%}",
                            "Correct signal share": "{:.1%}",
                            "Approx. 95% CI low": "{:.1%}",
                            "Approx. 95% CI high": "{:.1%}",
                            "Wrong direction share": "{:.1%}",
                            "No-touch share": "{:.1%}",
                        }, na_rep="—"),
                        use_container_width=True,
                        hide_index=True,
                    )
                    ci_low = pd.to_numeric(wf_table["Approx. 95% CI low"], errors="coerce")
                    if not (ci_low > 0.50).any():
                        st.error(
                            "Research gate: NOT PASSED. No variant has a lower approximate "
                            "95% Wilson confidence bound above 50% directional correctness. "
                            "The current feature-only models do not establish a stable edge."
                        )
                    st.caption(
                        "Wilson intervals are approximate and ignore dependence/clustering. "
                        "A positive interval would still require block-bootstrap confirmation "
                        "and a net-of-cost execution backtest."
                    )
                else:
                    st.warning("No completed variants were recorded in the walk-forward report.")
            else:
                st.error("The walk-forward report JSON could not be read.")

        st.info(
            "Planned progression: enrich inputs with verified live trade/order-book flow, "
            "run repeated walk-forward tests, then simulate fills/costs and paper trade. "
            "The current scorecard is experimental research, not a live signal or approval."
        )

    st.divider()
    st.caption(
        "DMPE is a research tool, not investment advice. Model probabilities are uncertain; "
        "never deploy capital based only on a dashboard display."
    )


if __name__ == "__main__":
    main()
