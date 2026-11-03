"""Chronological GARCH/HMM feature ablation against existing DMPE baselines.

This is a predictive-screening experiment, not a profitability claim. It reuses
the existing train/validation/test split and validation-only signal threshold.
Only the fixed 1.0x symmetric 1h/3h barrier labels are used as a common benchmark.
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
sys.path.insert(0, str(ROOT / "scripts"))

from dmpe.quant_regimes import GaussianHMM, fit_garch11  # noqa: E402
from evaluate_baselines import chronological_splits, evaluate_one  # noqa: E402

HORIZONS = {"1h": 60, "3h": 180}


def build_complete_5m_bars(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame[["time", "close"]].copy()
    work["time"] = pd.to_datetime(work["time"], utc=True)
    work["close"] = pd.to_numeric(work["close"], errors="coerce")
    work = work.dropna().sort_values("time").drop_duplicates("time")
    indexed = work.set_index("time")
    bars = indexed["close"].resample("5min", closed="right", label="right").agg(
        close="last", count="count"
    )
    # Explicitly verify all five minute closes are present and contiguous.
    complete = pd.Series(False, index=bars.index)
    grouped = indexed["close"].resample("5min", closed="right", label="right")
    for stamp, values in grouped:
        if len(values) == 5:
            idx = values.index
            if idx[0] == stamp - pd.Timedelta(minutes=4) and idx[-1] == stamp:
                if (idx.to_series().diff().dropna() == pd.Timedelta(minutes=1)).all():
                    complete.loc[stamp] = True
    bars = bars.loc[complete, ["close"]].copy()
    bars["log_return"] = np.log(bars["close"]).diff()
    time_step = bars.index.to_series().diff()
    bars.loc[time_step != pd.Timedelta(minutes=5), "log_return"] = np.nan
    bars["segment"] = (time_step != pd.Timedelta(minutes=5)).cumsum().astype("int32")
    bars["trailing_vol"] = bars.groupby("segment")["log_return"].transform(
        lambda s: s.rolling(12, min_periods=12).std(ddof=0)
    )
    bars["log_trailing_vol"] = np.log(bars["trailing_vol"].clip(lower=1e-12))
    return bars.reset_index(names="time")


def add_directional_event_labels(frame: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """Combine independent long/short 1R-stop/3R-target events into a direction class.

    +1 = long target first and short target did not also win.
    -1 = short target first and long target did not also win.
     0 = neither side's target won.
    Conflicting target wins, same-minute ambiguities and incomplete outcomes are excluded.
    """
    tag = "sl1_tp3rv"
    long_label = pd.to_numeric(frame[f"asym_long_label_{horizon}_{tag}"], errors="coerce")
    short_label = pd.to_numeric(frame[f"asym_short_label_{horizon}_{tag}"], errors="coerce")
    long_state = pd.to_numeric(frame[f"asym_long_state_{horizon}_{tag}"], errors="coerce")
    short_state = pd.to_numeric(frame[f"asym_short_state_{horizon}_{tag}"], errors="coerce")
    valid = long_state.isin([-1, 0, 1]) & short_state.isin([-1, 0, 1])
    long_win = long_label.eq(1)
    short_win = short_label.eq(1)
    conflict = long_win & short_win
    direction = np.where(long_win & ~short_win, 1,
                np.where(short_win & ~long_win, -1, 0)).astype("int8")
    label_col = f"barrier_label_{horizon}_1rv"
    state_col = f"barrier_state_{horizon}_1rv"
    out = frame.copy()
    out[label_col] = pd.Series(direction, index=out.index).where(valid & ~conflict, np.nan)
    out[state_col] = np.where(valid & ~conflict, direction, -2).astype("int8")
    return out


def make_sample(frame: pd.DataFrame, horizon: str) -> pd.DataFrame:
    minutes = HORIZONS[horizon]
    label = f"barrier_label_{horizon}_1rv"
    state = f"barrier_state_{horizon}_1rv"
    minute_epoch = pd.to_datetime(frame["time"], utc=True).astype("int64") // 60_000_000_000
    y = pd.to_numeric(frame[label], errors="coerce")
    st = pd.to_numeric(frame[state], errors="coerce")
    mask = (minute_epoch % minutes == 0) & y.notna() & st.isin([-1, 0, 1])
    return frame.loc[mask, ["time"]].reset_index(drop=True)


def add_causal_regime_features(
    frame: pd.DataFrame, bars: pd.DataFrame, train_end: pd.Timestamp,
    max_hmm_iter: int = 30,
) -> tuple[pd.DataFrame, dict]:
    valid_obs = np.isfinite(bars[["log_return", "log_trailing_vol"]].to_numpy(dtype=float)).all(axis=1)
    train_mask = (bars["time"] <= train_end).to_numpy()
    train_candidates = bars.loc[train_mask & bars["log_return"].notna()]
    if train_candidates.empty:
        raise ValueError("No 5-minute training returns before the chronological cutoff.")
    segment_sizes = train_candidates.groupby("segment").size().sort_values(ascending=False)
    training_segment = int(segment_sizes.index[0])
    selected_train = train_mask & (bars["segment"].to_numpy() == training_segment)
    returns_train = bars.loc[selected_train & bars["log_return"].notna(), "log_return"].to_numpy(dtype=float)
    if len(returns_train) < 100:
        raise ValueError("Insufficient contiguous 5-minute returns in the selected training segment.")
    garch = fit_garch11(returns_train)
    if not garch.success:
        raise RuntimeError("GARCH optimizer did not converge; refusing to report an ablation.")

    obs = bars[["log_return", "log_trailing_vol"]].to_numpy(dtype=float)
    train_obs = obs[selected_train & valid_obs]
    hmm = GaussianHMM(n_states=3, max_iter=max_hmm_iter).fit(train_obs)

    enriched = bars.copy()
    enriched["garch_log_variance"] = np.nan
    enriched["hmm_p_low_vol"] = np.nan
    enriched["hmm_p_mid_vol"] = np.nan
    enriched["hmm_p_high_vol"] = np.nan

    # Reset recursive state after data gaps; never carry a stale variance across a gap.
    for _, idx in enriched.groupby("segment", sort=False).groups.items():
        positions = np.asarray(list(idx), dtype=int)
        h = garch.train_variance
        for pos in positions:
            ret = enriched.at[pos, "log_return"]
            if not np.isfinite(ret):
                h = garch.train_variance
                continue
            eps = float(ret) - garch.train_mean
            h = max(garch.omega + garch.alpha * eps * eps + garch.beta * h, 1e-12)
            enriched.at[pos, "garch_log_variance"] = np.log(h)

        segment_valid = positions[valid_obs[positions]]
        if len(segment_valid):
            probs = hmm.filtered_probabilities(obs[segment_valid])
            enriched.loc[segment_valid, [
                "hmm_p_low_vol", "hmm_p_mid_vol", "hmm_p_high_vol"
            ]] = probs

    # Attach only the latest completed 5m observation; reject stale values after gaps.
    quant_cols = [
        "time", "garch_log_variance", "hmm_p_low_vol", "hmm_p_mid_vol",
        "hmm_p_high_vol", "trailing_vol",
    ]
    merged = pd.merge_asof(
        frame.sort_values("time"),
        enriched[quant_cols].sort_values("time"),
        on="time", direction="backward", tolerance=pd.Timedelta(minutes=5),
    )
    merged["garch_vs_realized_vol"] = (
        np.exp(0.5 * merged["garch_log_variance"])
        / merged["trailing_vol"].replace(0, np.nan)
    )
    report = {
        "train_end_utc": train_end.isoformat(),
        "garch": {
            "omega": garch.omega, "alpha": garch.alpha, "beta": garch.beta,
            "alpha_plus_beta": garch.alpha + garch.beta,
            "optimizer_success": garch.success,
            "training_returns": int(len(returns_train)),
            "training_segment_id": training_segment,
            "training_segment_start_utc": bars.loc[selected_train, "time"].min().isoformat(),
            "training_segment_end_utc": bars.loc[selected_train, "time"].max().isoformat(),
        },
        "hmm": {
            "states": 3, "max_iterations": max_hmm_iter,
            "training_observations": int(len(train_obs)),
            "state_order": "low-to-high volatility by fitted mean log trailing volatility",
            "probability_method": "forward filtered only; reset after data gaps",
        },
        "feature_missing_rates": {
            col: float(merged[col].isna().mean())
            for col in quant_cols[1:] + ["garch_vs_realized_vol"]
        },
    }
    return merged, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-file", required=True)
    parser.add_argument("--schema-file", required=True)
    parser.add_argument("--output-dir", default="quant_ablation_output")
    parser.add_argument("--hmm-iterations", type=int, default=30)
    args = parser.parse_args()

    feature_path, schema_path = Path(args.feature_file), Path(args.schema_file)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    base_features = list(dict.fromkeys(schema.get("feature_columns", [])))
    required = ["time", "close", *base_features]
    for horizon in HORIZONS:
        for direction in ("long", "short"):
            required += [
                f"asym_{direction}_label_{horizon}_sl1_tp3rv",
                f"asym_{direction}_state_{horizon}_sl1_tp3rv",
            ]
    frame = pd.read_parquet(feature_path, columns=list(dict.fromkeys(required)))
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame = frame.sort_values("time").reset_index(drop=True)
    for horizon in HORIZONS:
        frame = add_directional_event_labels(frame, horizon)

    bars = build_complete_5m_bars(frame)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    feature_reports = {}

    for horizon, horizon_minutes in HORIZONS.items():
        sample = make_sample(frame, horizon)
        if len(sample) < 300:
            raise RuntimeError(f"Insufficient {horizon} samples: {len(sample)}")
        train_idx, _, _, split = chronological_splits(sample, horizon_minutes)
        train_end = pd.Timestamp(split["train_end"])
        enriched, model_report = add_causal_regime_features(
            frame, bars, train_end, max_hmm_iter=args.hmm_iterations
        )
        feature_reports[horizon] = model_report
        quant_features = [
            "garch_log_variance", "garch_vs_realized_vol",
            "hmm_p_low_vol", "hmm_p_mid_vol", "hmm_p_high_vol",
        ]
        variants = [
            ("baseline", base_features),
            ("baseline_plus_garch", base_features + quant_features[:2]),
            ("baseline_plus_garch_hmm", base_features + quant_features),
        ]
        for variant, columns in variants:
            print(f"Evaluating {horizon}: {variant}", flush=True)
            result = evaluate_one(
                enriched, columns, horizon, horizon_minutes, 1.0,
                output / "models" / horizon / variant,
            )
            result["variant"] = variant
            result["quant_feature_columns"] = [c for c in columns if c in quant_features]
            reports.append(result)

    payload = {
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "label_construction": (
            "For each horizon, +1 means the long 3R target was first and the short target was not also first; "
            "-1 means the short 3R target was first and the long target was not also first; "
            "0 means neither directional target was first. Conflicts, same-minute ambiguity and incomplete rows are excluded."
        ),
        "purpose": (
            "Exploratory chronological feature ablation. This is not a net-PnL test "
            "and does not approve trading. A promising variant must pass the separate "
            "execution-cost and prospective validation gates."
        ),
        "data_file": str(feature_path),
        "decision_timeframe": "5m completed-bar regime features merged backward with 5m tolerance",
        "training_policy": [
            "GARCH and HMM parameters are fitted only before each horizon's training cutoff.",
            "GARCH recursion and HMM probabilities are causal and reset after data gaps.",
            "HMM uses forward filtering, never backward smoothing.",
            "Existing baseline train/validation/test splits, purging and validation-only thresholds are reused.",
            "Directional labels are derived from the existing separate 1h/3h asymmetric 1R-stop/3R-target event labels: long target-first=+1, short target-first=-1, neither target-first=0.",
            "Conflicting long-and-short target wins, same-minute ambiguity and incomplete outcomes are excluded.",
            "No prospective comparison-period outcomes are used.",
        ],
        "model_fits": feature_reports,
        "results": reports,
    }
    (output / "quant_ablation_report.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    pd.json_normalize(reports).to_csv(output / "quant_ablation_summary.csv", index=False)
    print(json.dumps({
        "output_dir": str(output),
        "completed": sum(x.get("status") == "completed" for x in reports),
        "results": [{
            "horizon": x.get("horizon"), "variant": x.get("variant"),
            "status": x.get("status"),
            "logistic_balanced_accuracy": x.get("logistic_regression_test", {}).get("balanced_accuracy"),
            "logistic_brier": x.get("logistic_regression_test", {}).get("multiclass_brier_score"),
            "histgb_balanced_accuracy": x.get("hist_gradient_boosting_test", {}).get("balanced_accuracy"),
            "histgb_brier": x.get("hist_gradient_boosting_test", {}).get("multiclass_brier_score"),
        } for x in reports],
    }, indent=2))


if __name__ == "__main__":
    main()
