"""Leakage-aware SKAB classroom experiment: classical detectors and score fusion.

Only approved normal prefixes of fit runs train detectors. Normal calibration,
labelled validation, and held-out runs have disjoint IDs. All timestamps shown
in results are actually observed; short causal fills supply history only.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import time
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

SKAB_COMMIT = "b2c0d46c2971dcbfe71e26087b6d231998bb91c2"
CHANNELS = ["Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure",
            "Temperature", "Thermocouple", "Voltage", "Volume Flow RateRMS"]
FEATURE_NAMES = [f"{kind}__{channel}" for kind in ["current", "difference", "mean30", "std30", "mean120", "std120"] for channel in CHANNELS]
SPLITS = {
    "fit": [*(f"valve1/{i}" for i in range(4)), "valve2/0"],
    "calibration": [*(f"valve1/{i}" for i in range(4, 8)), "valve2/1"],
    "validation": [*(f"valve1/{i}" for i in range(8, 12)), "valve2/2"],
    "test": [*(f"valve1/{i}" for i in range(12, 16)), "valve2/3"],
}
CONFIG = {
    "skab_commit": SKAB_COMMIT, "splits": SPLITS, "seed": 2026,
    "frequency": "1s", "max_fill_seconds": 2, "warmup_seconds": 128,
    "windows_seconds": [30, 120], "pca_components": 3,
    "if_trees": 200, "if_max_samples": 256, "normal_alarm_fraction_budget": .01,
    "score_direction": "larger means more unusual",
    "normalization": "(1 + normal-calibration count <= score) / (n + 2)",
    "threshold_rule": "validation macro unadjusted state F1 maximum under pooled normal alarm fraction <= 1%; ties prefer fewer false positives then higher threshold",
    "decision_time": "current observation is available; all window/forecast context is causal",
}


def sha256_bytes(payload: bytes) -> str:
    """Hash literal bytes.

    Parameters
    ----------
    payload : bytes
        Bytes to identify.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(payload).hexdigest()


def json_safe(value: object) -> object:
    """Convert missing numerical report values into strict JSON nulls.

    Parameters
    ----------
    value : object
        Nested ordinary dictionaries/lists/scalars from a report.

    Returns
    -------
    object
        Same report with nonfinite floats represented by None.
    """
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def download_runs(root: Path) -> dict[str, dict[str, object]]:
    """Download the fixed public run set without credentials.

    Parameters
    ----------
    root : Path
        Cache directory; existing raw files are retained and hashed.

    Returns
    -------
    dict
        Per-run URL, SHA-256 and byte count provenance.
    """
    provenance = {}
    for run_id in [r for group in SPLITS.values() for r in group]:
        path = root / (run_id + ".csv")
        url = f"https://raw.githubusercontent.com/waico/SKAB/{SKAB_COMMIT}/data/{run_id}.csv"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(url, headers={"User-Agent": "IntroML-SKAB-classroom"})
            with urllib.request.urlopen(req, timeout=45) as response:
                payload = response.read()
            if not payload.startswith(b"datetime;"):
                raise ValueError(f"Unexpected CSV header for {run_id}")
            path.write_bytes(payload)
        payload = path.read_bytes()
        provenance[run_id] = {"url": url, "sha256": sha256_bytes(payload), "bytes": len(payload)}
    return provenance


def preprocess_run(path: Path, run_id: str) -> tuple[pd.DataFrame, dict[str, object]]:
    """Regularize one run causally, retaining labels only at observed rows.

    Parameters
    ----------
    path : Path
        Pinned semicolon-separated SKAB source.
    run_id : str
        Stable family/file ID.

    Returns
    -------
    frame : DataFrame
        One-second grid with channels, observed, segment_id, age_seconds,
        eligible, normal_prefix, anomaly and changepoint. Gaps longer than
        two missing seconds split segments; windows may not cross them.
    audit : dict
        Excluded and inserted sample counts, cadence and prefix lengths.

    Raises
    ------
    ValueError
        Duplicate/reversed timestamps, missing channels or no anomaly.
    """
    raw = pd.read_csv(path, sep=";", parse_dates=["datetime"])
    if not raw.datetime.is_monotonic_increasing or raw.datetime.duplicated().any():
        raise ValueError(f"Non-unique or unordered timestamps: {run_id}")
    if not set(CHANNELS + ["anomaly", "changepoint"]).issubset(raw.columns):
        raise ValueError(f"Unexpected columns in {run_id}")
    raw = raw.set_index("datetime")
    if raw[CHANNELS].isna().any().any():
        raise ValueError(f"Observed sensor NaNs require explicit policy: {run_id}")
    if not (raw.anomaly == 1).any():
        raise ValueError(f"Expected an annotated anomaly in {run_id}")
    onset = raw.index[raw.anomaly == 1][0]
    starts = (raw.anomaly == 1) & (raw.anomaly.shift(fill_value=0) != 1)
    raw["event_id"] = starts.cumsum().where(raw.anomaly == 1, 0)
    event_onsets = {int(raw.loc[timestamp, "event_id"]): timestamp.isoformat() for timestamp in raw.index[starts]}
    raw["event_onset"] = raw.event_id.map(event_onsets)
    cadence = raw.index.to_series().diff().dt.total_seconds()
    # Three seconds between observed rows means exactly two missing seconds.
    raw["source_segment"] = (cadence.fillna(1) > 3).cumsum().to_numpy()
    pieces = []
    for segment_id, segment in raw.groupby("source_segment", sort=False):
        grid = pd.date_range(segment.index[0], segment.index[-1], freq="1s")
        block = segment.reindex(grid)
        block.index.name = "datetime"
        block["observed"] = block.index.isin(segment.index)
        block[CHANNELS] = block[CHANNELS].ffill(limit=2)
        if block[CHANNELS].isna().any().any():
            raise ValueError(f"Unfilled sensor values after segmentation: {run_id}")
        block["segment_id"] = int(segment_id)
        block["age_seconds"] = (block.index - block.index[0]).total_seconds()
        block["eligible"] = block.observed & (block.age_seconds >= CONFIG["warmup_seconds"])
        block["normal_prefix"] = block.index < onset
        block["run_id"] = run_id
        # Inserted rows intentionally retain NaN labels and are never scored.
        pieces.append(block.drop(columns="source_segment"))
    frame = pd.concat(pieces)
    obs = frame.observed
    elig = frame.eligible
    audit = {
        "run_id": run_id, "observed_rows": int(obs.sum()), "grid_rows": len(frame),
        "inserted_history_rows": int((~obs).sum()), "segments": int(frame.segment_id.nunique()),
        "raw_cadence_counts_seconds": {str(k): int(v) for k, v in cadence.value_counts().sort_index().items()},
        "eligible_rows": int(elig.sum()), "excluded_observed_warmup": int((obs & ~elig).sum()),
        "eligible_observed_fraction": float(elig.sum() / obs.sum()),
        "anomalous_observed_rows": int((obs & (frame.anomaly == 1)).sum()),
        "eligible_anomalous_rows": int((elig & (frame.anomaly == 1)).sum()),
        "normal_prefix_eligible_rows": int((elig & frame.normal_prefix).sum()),
        "onset": onset.isoformat(), "first_timestamp": raw.index[0].isoformat(),
        "last_timestamp": raw.index[-1].isoformat(),
    }
    return frame, audit


# window_features is supplied by the student's notebook after its visible checks.


# pca_q is supplied by the student's notebook after its visible checks.


# ecdf_transform is supplied by the student's notebook after its visible checks.


# fuse_scores is supplied by the student's notebook after its visible checks.


def threshold_search(rows: pd.DataFrame, score: str, budget: float = .01) -> dict[str, float]:
    """Select a validation threshold with an explicit pooled false-alarm budget.

    Parameters
    ----------
    rows : DataFrame
        Observed eligible validation rows; columns run_id, anomaly and score.
    score : str
        Detector score column name.
    budget : float
        Maximum pooled fraction of normal observations in alarm.

    Returns
    -------
    dict
        Threshold, validation macro state F1 and pooled normal alarm fraction.
        Alarm is defined as score >= threshold; no point adjustment is used.
    """
    unique = np.unique(rows[score].to_numpy())
    candidates = np.r_[unique, np.nextafter(unique[-1], np.inf)]
    f1s, pooled_fp, normal_count = [], np.zeros(len(candidates), dtype=int), 0
    for _, block in rows.groupby("run_id", sort=False):
        positives = np.sort(block.loc[block.anomaly == 1, score].to_numpy())
        negatives = np.sort(block.loc[block.anomaly == 0, score].to_numpy())
        tp = len(positives) - np.searchsorted(positives, candidates, side="left")
        fp = len(negatives) - np.searchsorted(negatives, candidates, side="left")
        fn = len(positives) - tp
        denominator = 2 * tp + fp + fn
        f1s.append(np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0))
        pooled_fp += fp
        normal_count += len(negatives)
    macro = np.mean(f1s, axis=0)
    fpr = pooled_fp / normal_count
    feasible = np.flatnonzero(fpr <= budget + 1e-12)
    best = max(feasible, key=lambda j: (macro[j], -pooled_fp[j], candidates[j]))
    return {"threshold": float(candidates[best]), "validation_macro_f1": float(macro[best]),
            "validation_normal_alarm_fraction": float(fpr[best]), "budget": budget}


def evaluate_runs(rows: pd.DataFrame, score: str, threshold: float) -> pd.DataFrame:
    """Evaluate unadjusted state predictions, coverage and secondary onset delay.

    Parameters
    ----------
    rows : DataFrame
        Observed eligible rows with run_id, datetime, anomaly and score.
    score : str
        Larger-is-more-unusual score column.
    threshold : float
        Frozen validation-selected threshold; score >= threshold is alarm.

    Returns
    -------
    DataFrame
        Per-run AP, precision/recall/F1, state coverage, normal alarm time,
        alarm episode rate and first-onset delay within a 60-second window.
        Missing onset delay denotes no timely detection, never zero delay.
    """
    from sklearn.metrics import average_precision_score
    records = []
    for run_id, block in rows.groupby("run_id", sort=False):
        y = block.anomaly.to_numpy(dtype=int)
        alarm = block[score].to_numpy() >= threshold
        tp, fp, fn = int((alarm & (y == 1)).sum()), int((alarm & (y == 0)).sum()), int((~alarm & (y == 1)).sum())
        normal_n, anomaly_n = int((y == 0).sum()), int((y == 1).sum())
        # Episodes restart across gaps; only episodes beginning at a normal row
        # contribute to the normal false-alarm episode count.
        times = pd.to_datetime(block.datetime)
        segment_break = block.segment_id.to_numpy() != np.r_[-1, block.segment_id.to_numpy()[:-1]]
        episodes = alarm & (~np.r_[False, alarm[:-1]] | segment_break)
        false_episodes = int((episodes & (y == 0)).sum())
        changes = np.r_[True, (y[1:] != y[:-1]) | segment_break[1:]]
        event_ids = block.event_id.to_numpy(dtype=int) if "event_id" in block else changes.cumsum()
        coverages, delays = [], []
        for event in np.unique(event_ids[y == 1]):
            ids = np.flatnonzero((event_ids == event) & (y == 1))
            coverages.append(float(alarm[ids].mean()))
            onset_time = pd.Timestamp(block.event_onset.iloc[ids[0]]) if "event_onset" in block else times.iloc[ids[0]]
            hits = ids[alarm[ids] & ((times.iloc[ids] - onset_time).dt.total_seconds().to_numpy() <= 60)]
            delays.append(float((times.iloc[hits[0]] - onset_time).total_seconds()) if len(hits) else np.nan)
        precision = tp / (tp + fp) if tp + fp else 0.
        recall = tp / anomaly_n if anomaly_n else 0.
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.
        records.append({"run_id": run_id, "detector": score, "n": len(block), "normal_n": normal_n,
                        "anomaly_n": anomaly_n, "ap": float(average_precision_score(y, block[score])),
                        "precision": precision, "recall": recall, "f1": f1,
                        "state_coverage_mean": float(np.mean(coverages)) if coverages else np.nan,
                        "normal_alarm_fraction": fp / normal_n if normal_n else np.nan,
                        "false_alarm_minutes_per_normal_hour": 60 * fp / normal_n if normal_n else np.nan,
                        "false_alarm_episodes_per_normal_hour": 3600 * false_episodes / normal_n if normal_n else np.nan,
                        "events": len(coverages), "onsets_detected_60s": int(np.isfinite(delays).sum()),
                        "onset_delay_seconds_mean_detected": float(np.nanmean(delays)) if np.isfinite(delays).any() else np.nan,
                        "threshold": threshold})
    return pd.DataFrame(records)


def write_figures(rows: pd.DataFrame, thresholds: dict[str, dict[str, float]], summary: pd.DataFrame, root: Path) -> None:
    """Save shared-axis diagnostics and primary-metric comparison figures.

    Parameters
    ----------
    rows : DataFrame
        Scored eligible observed rows.
    thresholds : dict
        Frozen validation threshold records.
    summary : DataFrame
        Per-detector/per-split macro metrics.
    root : Path
        Figure output directory.

    Returns
    -------
    None
        Writes PNG and SVG figures.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    root.mkdir(exist_ok=True, parents=True)
    colors = {"robust": "#e9a12b", "pca_q": "#3eb9b5", "if_raw": "#9d85e8",
              "if_window": "#3f81d0", "ensemble_mean": "#e86571", "ensemble_max": "#9aa6bc"}
    for split in ["validation", "test"]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
        block = rows[rows.split == split]
        for detector, color in colors.items():
            precision, recall, _ = precision_recall_curve(block.anomaly, block[detector])
            axes[0].plot(recall, precision, color=color, label=detector)
        axes[0].axhline(block.anomaly.mean(), color="#999999", ls=":", label="prevalence")
        axes[0].set(xlabel="State recall", ylabel="State precision", title=f"{split.title()}: pooled PR curves", xlim=(0,1), ylim=(0,1.02))
        table = summary[summary.split == split]
        axes[1].barh(table.detector, table.f1, color=[colors[x] for x in table.detector])
        axes[1].set(xlabel="Macro run state F1 (unadjusted)", xlim=(0,1), title="Frozen validation thresholds")
        axes[0].legend(fontsize=7, loc="lower left")
        fig.tight_layout()
        for suffix in ["png", "svg"]:
            fig.savefig(root / f"{split}_comparison.{suffix}", dpi=160, bbox_inches="tight")
        plt.close(fig)
    # Every run is exported: reviewers need not trust a cherry-picked example.
    for run_id, block in rows[rows.split.isin(["validation", "test"])].groupby("run_id", sort=False):
        t = (pd.to_datetime(block.datetime) - pd.to_datetime(block.datetime).iloc[0]).dt.total_seconds()
        fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)
        for channel in ["Accelerometer1RMS", "Current", "Volume Flow RateRMS"]:
            axes[0].plot(t, block["z__" + channel], label=channel)
        for name in ["robust", "pca_q", "if_window", "ensemble_mean"]:
            axes[1].plot(t, block[name], label=name, color=colors[name], lw=1)
            axes[2].step(t, (block[name] >= thresholds[name]["threshold"]).astype(int) + .18 * list(colors).index(name), label=name, color=colors[name], lw=.9, where="post")
        for axis in axes:
            axis.fill_between(t, 0, 1, where=block.anomaly.to_numpy() == 1, transform=axis.get_xaxis_transform(), color="#e86571", alpha=.12, label="anomalous state" if axis is axes[0] else None)
            axis.grid(alpha=.15)
        axes[0].set(ylabel="Normal-fit robust z-score", title=f"{run_id}: observed eligible rows; full state interval shaded")
        axes[1].set(ylabel="Normal-calibration ECDF")
        axes[2].set(xlabel="Seconds after first eligible observation", ylabel="Alarms (offset for visibility)")
        for axis in axes[:2]:axis.legend(ncol=4, fontsize=7, loc="upper right")
        axes[2].legend(ncol=4, fontsize=7, loc="upper right")
        fig.tight_layout()
        for suffix in ["png", "svg"]:
            fig.savefig(root / (run_id.replace("/", "_") + f"_timeline.{suffix}"), dpi=140, bbox_inches="tight")
        plt.close(fig)


def run_experiment(output: Path) -> dict[str, object]:
    """Execute the fixed classical experiment and preserve genuine results.

    Parameters
    ----------
    output : Path
        Owned experiment directory. Raw sources and results remain here.

    Returns
    -------
    dict
        Provenance, timing, selected pipeline and saved artifact paths.
    """
    import joblib
    import matplotlib
    import scipy
    import sklearn
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    output.mkdir(exist_ok=True, parents=True)
    started = time.perf_counter()
    provenance = download_runs(output / "raw")
    frames, audits = {}, []
    for split, run_ids in SPLITS.items():
        for run_id in run_ids:
            frame, audit = preprocess_run(output / "raw" / (run_id + ".csv"), run_id)
            frame["split"] = split
            frames[run_id] = frame
            audits.append(dict(audit, split=split))
    fit = pd.concat([frames[r].loc[frames[r].eligible & frames[r].normal_prefix] for r in SPLITS["fit"]])
    x_fit = fit[CHANNELS].to_numpy()
    center = np.median(x_fit, axis=0)
    mad_scale = 1.4826 * np.median(np.abs(x_fit - center), axis=0)
    fallback = np.std(x_fit, axis=0)
    scale = np.where(mad_scale > 1e-10, mad_scale, np.where(fallback > 1e-10, fallback, 1.))
    z_fit = (x_fit - center) / scale
    features = {r: window_features(f, center, scale) for r, f in frames.items()}
    window_fit = np.concatenate([features[r][(frames[r].eligible & frames[r].normal_prefix).to_numpy()] for r in SPLITS["fit"]])
    if not np.isfinite(window_fit).all():
        raise ValueError("Training window features contain nonfinite values")
    training_times = {}
    model_objects = {}
    for name, model, x in [
        ("pca_q", PCA(n_components=CONFIG["pca_components"], svd_solver="full"), z_fit),
        ("if_raw", IsolationForest(n_estimators=200, max_samples=256, random_state=2026, n_jobs=1), z_fit),
        ("if_window", IsolationForest(n_estimators=200, max_samples=256, random_state=2026, n_jobs=1), window_fit),
    ]:
        before = time.perf_counter();model.fit(x)
        training_times[name] = time.perf_counter() - before
        model_objects[name] = model
    raw_scores = {}
    eligible_rows = []
    inference_times = {name: 0. for name in ["robust", "pca_q", "if_raw", "if_window"]}
    for run_id, frame in frames.items():
        eligible = frame.eligible.to_numpy()
        observed = frame.loc[eligible].copy()
        z = (observed[CHANNELS].to_numpy() - center) / scale
        for index, channel in enumerate(CHANNELS):
            observed["z__" + channel] = z[:, index]
        scores = {}
        for name, operation in [
            ("robust", lambda: np.abs(z).max(axis=1)),
            ("pca_q", lambda: pca_q(z, model_objects["pca_q"].inverse_transform(model_objects["pca_q"].transform(z)))),
            ("if_raw", lambda: -model_objects["if_raw"].score_samples(z)),
            ("if_window", lambda: -model_objects["if_window"].score_samples(features[run_id][eligible])),
        ]:
            before = time.perf_counter();scores[name] = operation()
            inference_times[name] += time.perf_counter() - before
            observed[name + "_raw_score"] = scores[name]
        raw_scores[run_id] = scores
        eligible_rows.append(observed.reset_index())
    rows = pd.concat(eligible_rows, ignore_index=True)
    calibration = rows.loc[(rows.split == "calibration") & rows.normal_prefix]
    references = {name: calibration[name + "_raw_score"].to_numpy() for name in inference_times}
    for name in references:
        rows[name] = ecdf_transform(rows[name + "_raw_score"].to_numpy(), references[name])
    three = rows[["robust", "pca_q", "if_window"]].to_numpy()
    rows["ensemble_mean"] = fuse_scores(three, np.ones(3))
    rows["ensemble_max"] = three.max(axis=1)
    detectors = [*references, "ensemble_mean", "ensemble_max"]
    thresholds = {name: threshold_search(rows[rows.split == "validation"], name) for name in detectors}
    metrics = pd.concat([evaluate_runs(rows[rows.split == split], name, thresholds[name]["threshold"]).assign(split=split)
                         for split in ["validation", "test"] for name in detectors], ignore_index=True)
    metric_names = ["ap", "precision", "recall", "f1", "state_coverage_mean", "normal_alarm_fraction",
                    "false_alarm_minutes_per_normal_hour", "false_alarm_episodes_per_normal_hour", "onset_delay_seconds_mean_detected"]
    summary = metrics.groupby(["split", "detector"], sort=False)[metric_names].mean().reset_index()
    # Pooled budget is deliberately distinct from macro normal alarm fraction.
    summary["pooled_normal_alarm_fraction"] = [float(((rows[(rows.split == split) & (rows.anomaly == 0)][name] >= thresholds[name]["threshold"]).mean()))
                                                  for split, name in zip(summary.split, summary.detector)]
    selected = max(detectors, key=lambda name: (thresholds[name]["validation_macro_f1"], -detectors.index(name)))
    for run_id, frame in frames.items():
        destination = output / "regularized" / (run_id + ".csv.gz")
        destination.parent.mkdir(exist_ok=True, parents=True)
        frame.to_csv(destination, index=True)
        arrays = output / "arrays" / (run_id + ".npz")
        arrays.parent.mkdir(exist_ok=True, parents=True)
        np.savez_compressed(arrays, datetime_ns=frame.index.asi8, observed=frame.observed.to_numpy(),
                            segment_id=frame.segment_id.to_numpy(), eligible=frame.eligible.to_numpy(),
                            normal_prefix=frame.normal_prefix.to_numpy(), anomaly=frame.anomaly.to_numpy(),
                            changepoint=frame.changepoint.to_numpy(), z=(frame[CHANNELS].to_numpy() - center) / scale,
                            features=features[run_id], feature_names=np.array(FEATURE_NAMES), channels=np.array(CHANNELS))
    rows.to_csv(output / "scores.csv.gz", index=False)
    pd.DataFrame(audits).to_json(output / "preprocessing_audit.json", orient="records", indent=2)
    metrics.to_csv(output / "metrics_per_run.csv", index=False)
    summary.to_csv(output / "metrics_macro.csv", index=False)
    (output / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    (output / "config.json").write_text(json.dumps(CONFIG, indent=2))
    (output / "raw_provenance.json").write_text(json.dumps(provenance, indent=2))
    np.savez_compressed(output / "normal_fit_and_calibration.npz", center=center, scale=scale, normal_fit=x_fit,
                        normal_fit_z=z_fit, normal_fit_window=window_fit, **{name + "_calibration": ref for name, ref in references.items()})
    joblib.dump({"center": center, "scale": scale, "models": model_objects, "references": references, "thresholds": thresholds}, output / "classical_models.joblib")
    write_figures(rows, thresholds, summary, output / "figures")
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "scipy": scipy.__version__, "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__, "joblib": joblib.__version__}
    receipt = {"status": "actual CPU execution completed", "selected_on_validation": selected,
               "selected_test_macro": summary[(summary.split == "test") & (summary.detector == selected)].to_dict(orient="records")[0],
               "versions": versions, "platform": platform.platform(), "cpu": platform.processor(),
               "normal_fit_n": len(fit), "normal_calibration_n": len(calibration), "training_seconds": training_times,
               "all_run_score_seconds": inference_times, "total_wall_seconds": time.perf_counter() - started,
               "source_sha256": sha256_bytes(Path(__file__).read_bytes()), "point_adjustment": False,
               "row_identity": ["run_id", "datetime"], "labelled_selection": "validation only; test never selects thresholds or weights",
               "teacher_test_visibility": "Results below are instructor-only preflight; students reveal holdout after freezing decisions."}
    receipt = json_safe(receipt)
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False))
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(receipt, indent=2), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "actual_run")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        provenance = download_runs(args.output / "raw")
        audits = []
        for split, run_ids in SPLITS.items():
            for run_id in run_ids:
                frame, audit = preprocess_run(args.output / "raw" / (run_id + ".csv"), run_id)
                frame["split"] = split
                destination = args.output / "regularized" / (run_id + ".csv.gz")
                destination.parent.mkdir(exist_ok=True, parents=True)
                frame.to_csv(destination, index=True)
                audits.append(dict(audit, split=split))
        (args.output / "preprocessing_audit.json").write_text(json.dumps(audits, indent=2))
        (args.output / "config.json").write_text(json.dumps(CONFIG, indent=2))
        (args.output / "raw_provenance.json").write_text(json.dumps(provenance, indent=2))
        print("Prepared", len(audits), "regularized runs", flush=True)
    else:
        run_experiment(args.output)
