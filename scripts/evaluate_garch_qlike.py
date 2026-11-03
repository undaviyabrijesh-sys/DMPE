"""Out-of-sample GARCH volatility forecast evaluation using QLIKE.

Compares Gaussian-QMLE GARCH(1,1) with horizon-matched trailing realized
variance and EWMA(0.94). Parameters are fitted on the chronological training
slice only. Forecasts are made after a completed 5-minute close and target the
next 12 or 36 completed 5-minute returns (1h and 3h). No labels or test tuning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dmpe.quant_regimes import fit_garch11  # noqa: E402

HORIZONS = {"1h": 12, "3h": 36}
EWMA_LAMBDA = 0.94
BOOTSTRAP_REPS = 1000
SEED = 20261103


def qlike(forecast: np.ndarray, realized: np.ndarray) -> np.ndarray:
    """Patton-style QLIKE up to the realized-only additive term."""
    f = np.asarray(forecast, dtype=float)
    y = np.asarray(realized, dtype=float)
    if f.shape != y.shape or np.any(~np.isfinite(f)) or np.any(~np.isfinite(y)):
        raise ValueError("QLIKE inputs must be aligned and finite.")
    if np.any(f <= 0) or np.any(y < 0):
        raise ValueError("Forecast variance must be positive and realized variance nonnegative.")
    return np.log(f) + y / f


def build_complete_5m_bars(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame[["time", "close"]].copy()
    work["time"] = pd.to_datetime(work["time"], utc=True)
    work["close"] = pd.to_numeric(work["close"], errors="coerce")
    work = work.dropna().sort_values("time").drop_duplicates("time")
    indexed = work.set_index("time")
    grouped = indexed["close"].resample("5min", closed="right", label="right")
    bars = grouped.agg(close="last", count="count")
    first_time = indexed["close"].resample("5min", closed="right", label="right").apply(
        lambda s: s.index.min() if len(s) else pd.NaT
    )
    last_time = indexed["close"].resample("5min", closed="right", label="right").apply(
        lambda s: s.index.max() if len(s) else pd.NaT
    )
    complete = (
        bars["count"].eq(5)
        & first_time.eq(bars.index - pd.Timedelta(minutes=4))
        & last_time.eq(bars.index)
    )
    bars = bars.loc[complete, ["close"]].copy()
    bars["log_return"] = np.log(bars["close"]).diff()
    delta = bars.index.to_series().diff()
    bars.loc[delta != pd.Timedelta(minutes=5), "log_return"] = np.nan
    bars["segment"] = (delta != pd.Timedelta(minutes=5)).cumsum().astype("int32")
    return bars.reset_index(names="time")


def _longest_contiguous_segment(bars: pd.DataFrame) -> pd.DataFrame:
    valid = bars.dropna(subset=["log_return"]).copy()
    sizes = valid.groupby("segment").size()
    if sizes.empty:
        raise ValueError("No complete contiguous 5-minute return segment found.")
    selected = int(sizes.idxmax())
    out = valid.loc[valid["segment"] == selected].reset_index(drop=True)
    if len(out) < 5000:
        raise ValueError(f"Longest contiguous segment has only {len(out)} returns; need 5000.")
    return out


def _garch_path(returns: np.ndarray, fit) -> np.ndarray:
    """Forecast total variance over H future bars from each completed close."""
    out = np.empty(len(returns), dtype=float)
    h = fit.train_variance
    for i, value in enumerate(returns):
        eps = float(value) - fit.train_mean
        h = max(fit.omega + fit.alpha * eps * eps + fit.beta * h, 1e-12)
        out[i] = h
    return out


def _ewma_path(returns: np.ndarray, train_start: int, train_end: int) -> np.ndarray:
    train_var = max(float(np.var(returns[train_start:train_end + 1])), 1e-12)
    out = np.empty(len(returns), dtype=float)
    variance = train_var
    for i, value in enumerate(returns):
        variance = EWMA_LAMBDA * variance + (1.0 - EWMA_LAMBDA) * float(value) ** 2
        out[i] = max(variance, 1e-12)
    return out


def _day_block_ci(loss_diff: np.ndarray, times: pd.Series, reps: int = BOOTSTRAP_REPS):
    """Paired day-block bootstrap CI for mean loss difference (model - baseline)."""
    day = pd.to_datetime(times, utc=True).dt.floor("D").astype(str).to_numpy()
    unique = np.unique(day)
    if len(unique) < 10:
        return {"lower_95": None, "upper_95": None, "blocks": int(len(unique))}
    groups = [loss_diff[day == d] for d in unique]
    rng = np.random.default_rng(SEED)
    means = np.empty(reps, dtype=float)
    for i in range(reps):
        chosen = rng.integers(0, len(groups), size=len(groups))
        sample = np.concatenate([groups[j] for j in chosen])
        means[i] = float(sample.mean())
    return {
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
        "blocks": int(len(unique)),
        "replicates": reps,
    }


def evaluate_horizon(bars: pd.DataFrame, name: str, horizon_bars: int):
    r = bars["log_return"].to_numpy(dtype=float)
    times = pd.to_datetime(bars["time"], utc=True)
    n = len(r)
    cut_train = int(n * 0.70)
    cut_validation = int(n * 0.85)
    # Reserve a full 3h purge before training cutoff and after test boundary.
    train_end = cut_train - HORIZONS["3h"] - 1
    test_start = cut_validation + HORIZONS["3h"]
    if train_end < 100 or test_start + horizon_bars >= n:
        raise ValueError(f"Insufficient observations after chronological purge for {name}.")
    fit = fit_garch11(r[:train_end + 1])
    if not fit.success:
        raise RuntimeError(f"GARCH optimizer did not converge for {name}; no results reported.")

    # Forecast states are updated sequentially using only returns observed by each close.
    garch_one_step = _garch_path(r, fit)
    ewma_one_step = _ewma_path(r, 0, train_end)
    records = []
    for t in range(test_start, n - horizon_bars):
        future = r[t + 1:t + horizon_bars + 1]
        realized = float(np.sum(future * future))
        # GARCH multi-step conditional variance: E[h(t+k)|F_t] follows
        # omega + (alpha+beta)*E[h(t+k-1)|F_t] after the first step.
        h_next = float(garch_one_step[t])
        garch_total = 0.0
        h = h_next
        for k in range(horizon_bars):
            if k:
                h = fit.omega + (fit.alpha + fit.beta) * h
            garch_total += h
        trailing = r[max(0, t - horizon_bars + 1):t + 1]
        if len(trailing) != horizon_bars:
            continue
        rv_total = max(float(np.mean(trailing * trailing)) * horizon_bars, 1e-12)
        ewma_total = max(float(ewma_one_step[t]) * horizon_bars, 1e-12)
        records.append({
            "time": times.iloc[t],
            "horizon": name,
            "realized_variance": realized,
            "garch_forecast_variance": max(garch_total, 1e-12),
            "trailing_rv_forecast_variance": rv_total,
            "ewma_forecast_variance": ewma_total,
        })
    result = pd.DataFrame(records)
    if len(result) < 100:
        raise ValueError(f"Only {len(result)} test forecasts for {name}; need at least 100.")
    y = result["realized_variance"].to_numpy()
    summaries = {}
    losses = {}
    for model, col in [
        ("garch11", "garch_forecast_variance"),
        ("trailing_rv", "trailing_rv_forecast_variance"),
        ("ewma_094", "ewma_forecast_variance"),
    ]:
        forecast = result[col].to_numpy()
        loss = qlike(forecast, y)
        losses[model] = loss
        summaries[model] = {
            "mean_qlike": float(loss.mean()),
            "median_qlike": float(np.median(loss)),
            "mean_forecast_variance": float(np.mean(forecast)),
            "mean_realized_variance": float(np.mean(y)),
            # Aggregate ratio is stable when an individual realized window is zero.
            "aggregate_forecast_to_realized_variance_ratio": float(
                np.mean(forecast) / max(float(np.mean(y)), 1e-18)
            ),
            "mean_absolute_log_variance_error": float(np.mean(np.abs(
                np.log(forecast) - np.log(np.maximum(y, 1e-18))
            ))),
            "zero_realized_variance_windows": int(np.sum(y == 0.0)),
            "forecast_at_numerical_floor": int(np.sum(forecast <= 1.0001e-12)),
        }
    comparisons = {}
    for baseline in ("trailing_rv", "ewma_094"):
        diff = losses["garch11"] - losses[baseline]
        comparisons[f"garch_minus_{baseline}"] = {
            "mean_qlike_difference": float(diff.mean()),
            "day_block_bootstrap_95_ci": _day_block_ci(diff, result["time"]),
            "interpretation": "negative favors GARCH; positive favors baseline",
        }
    report = {
        "horizon": name,
        "horizon_bars_5m": horizon_bars,
        "train_start_utc": times.iloc[0].isoformat(),
        "train_end_utc": times.iloc[train_end].isoformat(),
        "test_start_utc": result["time"].iloc[0].isoformat(),
        "test_end_utc": result["time"].iloc[-1].isoformat(),
        "training_returns": int(train_end + 1),
        "test_forecasts": int(len(result)),
        "garch_parameters": {
            "omega": fit.omega, "alpha": fit.alpha, "beta": fit.beta,
            "alpha_plus_beta": fit.alpha + fit.beta,
            "gaussian_qmle_optimizer_success": fit.success,
        },
        "models": summaries,
        "paired_comparisons": comparisons,
        "notes": [
            "5-minute completed-bar log returns; forecasts made after close t for returns t+1 through t+H.",
            "GARCH parameters fitted on the first 70% minus a 3-hour purge; validation period is not used for tuning.",
            "Recursive GARCH and EWMA states update only with observations available by each forecast timestamp.",
            "Trailing RV uses the immediately preceding H completed 5-minute returns.",
            "QLIKE is evaluated only on the untouched chronological test segment; no parameter or model selection uses test outcomes.",
            "Overlapping 1h/3h realized-variance targets are dependent; paired day-block bootstrap is descriptive uncertainty, not an iid confidence interval.",
            "This is a volatility forecast evaluation, not a directional prediction or net-PnL backtest.",
        ],
    }
    return result, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-file", required=True)
    parser.add_argument("--output-dir", default="garch_qlike_output")
    args = parser.parse_args()
    frame = pd.read_parquet(args.feature_file, columns=["time", "close"])
    bars = _longest_contiguous_segment(build_complete_5m_bars(frame))
    all_rows, reports = [], {}
    for horizon, count in HORIZONS.items():
        rows, report = evaluate_horizon(bars, horizon, count)
        all_rows.append(rows)
        reports[horizon] = report
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    forecasts = pd.concat(all_rows, ignore_index=True)
    forecasts.to_csv(output / "garch_qlike_test_forecasts.csv", index=False)
    payload = {
        "experiment": "GARCH(1,1) volatility forecasting versus horizon-matched trailing RV and EWMA(0.94)",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "source_feature_file": str(args.feature_file),
        "source_rows": int(len(frame)),
        "complete_5m_bars_in_longest_segment": int(len(bars)),
        "method": "Gaussian-QMLE GARCH(1,1), fixed EWMA lambda 0.94, trailing RV window equals forecast horizon",
        "results": reports,
        "decision_rule": (
            "GARCH is considered useful only if its test QLIKE is lower than both simple baselines "
            "and paired day-block intervals support the improvement. Otherwise retain the simpler "
            "forecast. This experiment does not authorize directional features or trading."
        ),
    }
    (output / "garch_qlike_report.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({
        horizon: {
            "test_forecasts": item["test_forecasts"],
            "qlike": {model: round(stats["mean_qlike"], 8)
                      for model, stats in item["models"].items()},
            "comparisons": {
                key: value["mean_qlike_difference"]
                for key, value in item["paired_comparisons"].items()
            },
        } for horizon, item in reports.items()
    }, indent=2))


if __name__ == "__main__":
    main()
