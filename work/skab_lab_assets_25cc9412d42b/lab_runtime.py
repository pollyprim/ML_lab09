"""Provided infrastructure for the SKAB lab; the four student algorithms are absent.

Data loading, controlled model fitting, metric evaluation and plotting are supplied.
The notebook installs its four student functions into skab_lab before using them.
"""
from __future__ import annotations

import copy
import hashlib
import json
import marshal
import time
from pathlib import Path
from typing import Callable

import joblib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve

import skab_lab as core

FORECAST_COLUMNS = {
    "chronos_residual": "chronos_residual_raw_score",
    "chronos_interval": "chronos_interval_raw_score",
    "naive_residual": "naive_residual_raw_score",
}
CLASSIC = ["robust", "pca_q", "if_raw", "if_window"]
DETECTORS = [*CLASSIC, "ensemble_mean", "ensemble_max", *FORECAST_COLUMNS]
COLORS = dict(zip(DETECTORS, ["#d99016", "#18a59f", "#aa86d8", "#367acf", "#e35d77", "#7e8798", "#182e66", "#c76fc0", "#62974b"]))


class LabRuntime:
    """Keep provided model/data state separate from the visible student algorithms."""

    def __init__(self, assets: Path, attempt: Path) -> None:
        """Load exact public inputs and create a fresh owned results directory.

        Parameters
        ----------
        assets : Path
            Verified extracted package directory.
        attempt : Path
            New directory for this notebook attempt; must not already exist.

        Returns
        -------
        None
            Stores regularized frames, fixed normal fit data and recorded forecasts.
        """
        self.assets, self.attempt = Path(assets), Path(attempt)
        self.attempt.mkdir(parents=True, exist_ok=False)
        self.frames, audit = {}, []
        for split, run_ids in core.SPLITS.items():
            for run_id in run_ids:
                frame, record = core.preprocess_run(self.assets / "raw" / (run_id + ".csv"), run_id)
                frame["split"] = split
                self.frames[run_id] = frame
                audit.append(dict(record, split=split))
        self.audit = pd.DataFrame(audit)
        fit = pd.concat([self.frames[r].loc[self.frames[r].eligible & self.frames[r].normal_prefix] for r in core.SPLITS["fit"]])
        x = fit[core.CHANNELS].to_numpy()
        self.center = np.median(x, axis=0)
        mad = 1.4826 * np.median(np.abs(x - self.center), axis=0)
        sd = x.std(axis=0)
        self.scale = np.where(mad > 1e-10, mad, np.where(sd > 1e-10, sd, 1.))
        self.fit_z = (x - self.center) / self.scale
        self.models, self.training_seconds, self.references = {}, {}, {}
        self.rows, self.thresholds, self.decision = None, {}, None
        self.revealed = False
        self._stages = set()
        self._frozen_integrity = None
        self._holdout_score_hash = None
        self.forecasts = pd.read_csv(self.assets / "forecast_scores.csv.gz", parse_dates=["datetime"], float_precision="round_trip")
        self.forecasts = self.forecasts.set_index(["run_id", "datetime"])
        if self.forecasts.index.has_duplicates:
            raise ValueError("Recorded forecast cache has duplicate row IDs")
        self.forecast_receipt = json.loads((self.assets / "forecast_receipt.json").read_text())

    def _require_stage(self, *stages: str) -> None:
        """Fail close to an out-of-order cell rather than report stale metrics."""
        missing = set(stages).difference(self._stages)
        if missing:
            raise RuntimeError("Run the earlier notebook stages first: " + ", ".join(sorted(missing)))

    def _require_unfrozen(self) -> None:
        """An already reviewed holdout must not change a development decision."""
        if self.decision is not None:
            raise RuntimeError("This attempt is frozen. Start a new top-to-bottom attempt to change development steps.")

    @staticmethod
    def _strict_json(value: object) -> str:
        """Use portable JSON and reject NaN/Infinity in decision evidence."""
        return json.dumps(value, sort_keys=True, allow_nan=False)

    @staticmethod
    def _score_rows_hash(frame: pd.DataFrame) -> str:
        """Hash displayed score values, independently of pandas block layout."""
        digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
        digest.update(json.dumps([(str(name), str(dtype)) for name, dtype in frame.dtypes.items()]).encode())
        return digest.hexdigest()

    def _integrity_snapshot(self) -> dict[str, str]:
        """Identify the actual fitted pipeline used by the holdout adapter.

        This is an accidental-state-change guard, not a security sandbox. The
        hashes contain no fitted weights or sensor values in the decision file.
        """
        functions = {}
        for name in ("pca_q", "window_features", "ecdf_transform", "fuse_scores"):
            function = getattr(core, name)
            code = getattr(function, "__code__", None)
            if code is None:
                raise RuntimeError("Expected the checked notebook function: " + name)
            functions[name] = hashlib.sha256(marshal.dumps(code)).hexdigest()
        return {
            "config": hashlib.sha256(self._strict_json(core.CONFIG).encode()).hexdigest(),
            "thresholds": hashlib.sha256(self._strict_json(self.thresholds).encode()).hexdigest(),
            "fit_state": joblib.hash((self.models, self.center, self.scale)),
            "normal_references": joblib.hash(self.references),
            "features_and_frames": joblib.hash((self.features, self.frames)),
            "recorded_forecasts": joblib.hash(self.forecasts),
            "task_functions": hashlib.sha256(self._strict_json(functions).encode()).hexdigest(),
            "development_scores": self._score_rows_hash(self.rows[self.rows.split != "test"]),
        }

    def _verify_frozen(self) -> None:
        """Check both the stored decision and the live pipeline before every view."""
        if self.decision is None or self._frozen_integrity is None:
            raise RuntimeError("Freeze your validation choice and explanation first")
        snapshot = self._integrity_snapshot()
        if snapshot != self._frozen_integrity:
            changed = sorted(key for key in snapshot if snapshot[key] != self._frozen_integrity[key])
            raise RuntimeError("Frozen pipeline changed (" + ", ".join(changed) + "). Restore the original attempt; do not reinterpret altered holdout results.")
        if self.revealed:
            actual = self._score_rows_hash(self.rows[self.rows.split == "test"])
            if actual != self._holdout_score_hash:
                raise RuntimeError("Cached heldout scores changed after their frozen reveal")
        payload = copy.deepcopy(self.decision)
        decision_id = payload.pop("decision_id")
        if hashlib.sha256(self._strict_json(payload).encode()).hexdigest()[:16] != decision_id:
            raise RuntimeError("The frozen decision record changed after it was saved")
        disk = json.loads((self.attempt / "frozen_decision.json").read_text())
        if disk != self.decision:
            raise RuntimeError("The saved frozen decision differs from this attempt")

    def split_table(self) -> pd.DataFrame:
        """Return a compact split, exclusion and prefix audit.

        Returns
        -------
        DataFrame
            Per-split counts; no held-out signal values, labels or model metrics.
        """
        return self.audit.groupby("split", sort=False).agg(runs=("run_id", "size"), observed=("observed_rows", "sum"), eligible=("eligible_rows", "sum"), inserted_history=("inserted_history_rows", "sum"), reset_segments=("segments", "sum"), healthy_prefix_eligible=("normal_prefix_eligible_rows", "sum"))

    def fit_pca(self) -> pd.DataFrame:
        """Fit the fixed PCA on normal fit observations only.

        Returns
        -------
        DataFrame
            Fitted axes and normal-training explained variance fractions.
        """
        self._require_unfrozen()
        started = time.perf_counter()
        self.models["pca_q"] = PCA(n_components=3, svd_solver="full").fit(self.fit_z)
        self.training_seconds["pca_q"] = time.perf_counter() - started
        self._stages.add("pca")
        self._stages.difference_update({"scores", "normalized", "fused", "thresholds"})
        return pd.DataFrame({"component": [1,2,3], "normal_fit_explained_variance": self.models["pca_q"].explained_variance_ratio_})

    def plot_observations(self, run_id: str="valve1/8") -> None:
        """Show real validation sensors and distinct state/changepoint annotations.

        Parameters
        ----------
        run_id : str
            A validation run ID; held-out observations are not shown here.

        Returns
        -------
        None
            One three-panel figure with raw physical units and annotation overlays.
        """
        if run_id not in core.SPLITS["validation"]:raise ValueError("Choose a validation run")
        b=self.frames[run_id];b=b[b.observed]
        t=(b.index-b.index[0]).total_seconds()
        fig,axes=plt.subplots(3,1,figsize=(11,6),sharex=True)
        for ax,channel,unit in zip(axes,["Accelerometer1RMS","Current","Volume Flow RateRMS"],["g RMS","A","L/min"]):
            ax.plot(t,b[channel],lw=1,color="#367acf")
            ax.fill_between(t,0,1,where=b.anomaly.to_numpy()==1,transform=ax.get_xaxis_transform(),color="#f9a5b5",alpha=.3,label="anomalous state")
            for cp in t[b.changepoint.to_numpy()==1]:ax.axvline(cp,color="#5f566b",ls=":",lw=1)
            ax.set(ylabel=f"{channel}\n({unit})");ax.grid(alpha=.15)
        axes[0].set_title(f"{run_id}: real pump measurements; dotted lines are annotated changepoints")
        axes[-1].set_xlabel("Elapsed seconds in this run");fig.tight_layout();plt.show()

    def fit_forests(self) -> pd.DataFrame:
        """Fit identical forests on raw and student-defined causal features.

        Returns
        -------
        DataFrame
            Exact fixed tree count, samples, feature dimensions and measured CPU time.
        """
        self._require_unfrozen()
        self._require_stage("pca")
        self.features = {r: core.window_features(f, self.center, self.scale) for r, f in self.frames.items()}
        xw = np.concatenate([self.features[r][(self.frames[r].eligible & self.frames[r].normal_prefix).to_numpy()] for r in core.SPLITS["fit"]])
        if xw.shape[1] != 48 or not np.isfinite(xw).all():
            raise ValueError("Expected finite 48-column training features")
        records = []
        for name, x in [("if_raw", self.fit_z), ("if_window", xw)]:
            started = time.perf_counter()
            self.models[name] = IsolationForest(n_estimators=200, max_samples=256, random_state=2026, n_jobs=1).fit(x)
            self.training_seconds[name] = time.perf_counter() - started
            records.append({"detector": name, "normal_fit_rows": len(x), "features": x.shape[1], "trees": 200, "max_samples": 256, "fit_CPU_seconds": self.training_seconds[name]})
        self._stages.add("forests")
        self._stages.difference_update({"scores", "normalized", "fused", "thresholds"})
        return pd.DataFrame(records)

    def _raw_scores(self, split: str) -> pd.DataFrame:
        """Compute supplied raw scores using notebook-injected student algorithms.

        Parameters
        ----------
        split : str
            calibration, validation or test.

        Returns
        -------
        DataFrame
            Observed eligible rows, classical raw scores and recorded forecast scores.
        """
        blocks = []
        for run_id in core.SPLITS[split]:
            f = self.frames[run_id]
            mask = f.eligible.to_numpy(copy=True)
            if split == "calibration":
                mask &= f.normal_prefix.to_numpy()
            block = f.loc[mask].copy().reset_index()
            z = (block[core.CHANNELS].to_numpy() - self.center) / self.scale
            for j, channel in enumerate(core.CHANNELS):block["z__" + channel] = z[:,j]
            block["robust_raw_score"] = np.abs(z).max(axis=1)
            reconstruction = self.models["pca_q"].inverse_transform(self.models["pca_q"].transform(z))
            block["pca_q_raw_score"] = core.pca_q(z, reconstruction)
            block["if_raw_raw_score"] = -self.models["if_raw"].score_samples(z)
            block["if_window_raw_score"] = -self.models["if_window"].score_samples(self.features[run_id][mask])
            key = pd.MultiIndex.from_frame(block[["run_id", "datetime"]])
            missing = key.difference(self.forecasts.index)
            if len(missing):raise ValueError(f"Forecast cache missing {len(missing)} exact eligible row keys for {run_id}")
            cached = self.forecasts.loc[key]
            for _, column in FORECAST_COLUMNS.items():block[column] = cached[column].to_numpy()
            blocks.append(block)
        return pd.concat(blocks, ignore_index=True)

    def score_development(self) -> pd.DataFrame:
        """Score calibration and validation only; held-out scoring waits for freeze.

        Returns
        -------
        DataFrame
            Raw score distribution summaries on normal calibration observations.
        """
        self._require_unfrozen()
        self._require_stage("pca", "forests")
        self.rows = pd.concat([self._raw_scores("calibration"), self._raw_scores("validation")], ignore_index=True)
        self._stages.add("scores")
        self._stages.difference_update({"normalized", "fused", "thresholds"})
        self.references, self.thresholds = {}, {}
        calibration = self.rows[self.rows.split == "calibration"]
        validation_normal = self.rows[(self.rows.split == "validation") & (self.rows.anomaly == 0)]
        validation_fault = self.rows[(self.rows.split == "validation") & (self.rows.anomaly == 1)]
        return pd.DataFrame([{"detector": name,
                              "cal_normal_median": float(calibration[name+"_raw_score"].median()),
                              "cal_normal_max": float(calibration[name+"_raw_score"].max()),
                              "val_normal_above_cal_max_fraction": float((validation_normal[name+"_raw_score"] > calibration[name+"_raw_score"].max()).mean()),
                              "val_fault_above_cal_max_fraction": float((validation_fault[name+"_raw_score"] > calibration[name+"_raw_score"].max()).mean())}
                             for name in CLASSIC])

    def normalize_scores(self) -> pd.DataFrame:
        """Apply the student's ECDF against independent normal calibration only.

        Returns
        -------
        DataFrame
            Calibration sizes and validation saturation fractions by detector.
        """
        self._require_unfrozen()
        self._require_stage("scores")
        self._stages.difference_update({"fused", "thresholds"})
        self.thresholds = {}
        records=[]
        for name in [*CLASSIC, *FORECAST_COLUMNS]:
            column = name + "_raw_score" if name in CLASSIC else FORECAST_COLUMNS[name]
            self.references[name] = self.rows.loc[self.rows.split == "calibration", column].to_numpy()
            self.rows[name] = core.ecdf_transform(self.rows[column].to_numpy(), self.references[name])
            v = self.rows[self.rows.split == "validation"]
            records.append({"detector": name, "normal_calibration_n":len(self.references[name]),
                            "validation_normal_fraction_above_calibration_max":float((v.loc[v.anomaly==0,column] > self.references[name].max()).mean()),
                            "validation_fault_fraction_above_calibration_max":float((v.loc[v.anomaly==1,column] > self.references[name].max()).mean())})
        self._stages.add("normalized")
        return pd.DataFrame(records)

    def fuse(self) -> None:
        """Compute classical fusion using the student-defined weighted mean.

        Returns
        -------
        None
            Adds mean and max of robust/PCA/window-IF normalized scores.
        """
        self._require_unfrozen()
        self._require_stage("normalized")
        self._fuse_frame(self.rows)
        self._stages.add("fused")
        self._stages.discard("thresholds")
        self.thresholds = {}

    def _fuse_frame(self, frame: pd.DataFrame) -> None:
        """Add aligned ensemble columns to a provided score frame.

        Parameters
        ----------
        frame : DataFrame
            Frame with robust, pca_q and if_window ECDF scores.

        Returns
        -------
        None
            Updates ensemble columns in place.
        """
        values = frame[["robust", "pca_q", "if_window"]].to_numpy()
        frame["ensemble_mean"] = core.fuse_scores(values, np.ones(3))
        frame["ensemble_max"] = values.max(axis=1)

    def select_thresholds(self) -> pd.DataFrame:
        """Select validation thresholds at the same 1% normal alarm-time budget.

        Returns
        -------
        DataFrame
            Validation-only macro state metrics, pooled false-alarm fractions and thresholds.
        """
        self._require_unfrozen()
        self._require_stage("fused")
        validation=self.rows[self.rows.split == "validation"]
        self.thresholds = {name: core.threshold_search(validation, name, budget=.01) for name in DETECTORS}
        self._stages.add("thresholds")
        return self.metric_table("validation")

    def metric_table(self, split: str) -> pd.DataFrame:
        """Compute the shared metric table without any point adjustment.

        Parameters
        ----------
        split : str
            validation, or test after final reveal.

        Returns
        -------
        DataFrame
            Macro AP/F1/coverage, pooled normal alarm fraction and onset detection counts.
        """
        self._require_stage("thresholds")
        if self.decision is not None:self._verify_frozen()
        if split not in ("validation", "test"):raise ValueError("Choose validation or test")
        if split == "test" and not self.revealed:raise RuntimeError("Freeze a decision before revealing held-out results")
        rows=self.rows[self.rows.split == split]
        reports=[]
        for name in DETECTORS:
            per_run=core.evaluate_runs(rows,name,self.thresholds[name]["threshold"])
            reports.append({"detector":name,"macro_AP":per_run.ap.mean(),"macro_state_F1":per_run.f1.mean(),
                            "macro_state_coverage":per_run.state_coverage_mean.mean(),
                            "pooled_normal_alarm_fraction":float((rows.loc[rows.anomaly==0,name]>=self.thresholds[name]["threshold"]).mean()),
                            "macro_false_alarm_episodes_per_normal_hour":per_run.false_alarm_episodes_per_normal_hour.mean(),
                            "onsets_detected_within_60s":int(per_run.onsets_detected_60s.sum()),
                            "events":int(per_run.events.sum()),
                            "mean_delay_seconds_detected_only":per_run.onset_delay_seconds_mean_detected.mean(),
                            "threshold":self.thresholds[name]["threshold"],
                            "policy_on_this_split":"quiet: cutoff above max" if self.thresholds[name]["threshold"]>rows[name].max() else "active"})
        return pd.DataFrame(reports)

    def freeze(self, selected: str, reason: str) -> dict[str, object]:
        """Persist one validation-based decision before opening held-out labels.

        Parameters
        ----------
        selected : str
            One detector from the fixed candidate list.
        reason : str
            Student's concise evidence-based justification.

        Returns
        -------
        dict
            Decision ID, selected detector and immutable protocol/thresholds.

        Raises
        ------
        ValueError
            Missing justification, unknown detector or an already-frozen attempt.
        """
        if not isinstance(selected, str) or selected not in DETECTORS or not isinstance(reason, str) or len(reason.strip()) < 20:
            raise ValueError("Choose an available detector and explain your validation evidence")
        reason = reason.strip()
        if self.decision is not None:
            self._verify_frozen()
            if selected != self.decision["selected"] or reason != self.decision["reason"]:
                raise ValueError("Already frozen; preserve this attempt instead of revising it after test")
            return {"decision_id": self.decision["decision_id"], "selected": selected, "test_revealed": self.revealed, "reused": True}
        self._require_stage("thresholds")
        integrity = self._integrity_snapshot()
        payload = {
            "decision_version": 2,
            "selected": selected,
            "reason": reason,
            "thresholds": copy.deepcopy(self.thresholds),
            "config": copy.deepcopy(core.CONFIG),
            "pipeline_integrity": copy.deepcopy(integrity),
            "provenance": {
                "attempt": str(self.attempt.resolve()),
                "assets": str(self.assets.resolve()),
                "forecast_receipt_sha256": hashlib.sha256((self.assets / "forecast_receipt.json").read_bytes()).hexdigest(),
                "skab_revision": core.CONFIG["skab_commit"],
                "forecast_model": self.forecast_receipt["model"],
                "forecast_revision": self.forecast_receipt["revision"],
                "test_revealed_at_freeze": False,
            },
        }
        payload["decision_id"] = hashlib.sha256(self._strict_json(payload).encode()).hexdigest()[:16]
        (self.attempt / "frozen_decision.json").write_text(json.dumps(payload, indent=2, allow_nan=False))
        self.decision, self._frozen_integrity = payload, integrity
        return {"decision_id": payload["decision_id"], "selected": selected, "test_revealed": False, "reused": False}

    def reveal_holdout(self) -> pd.DataFrame:
        """Evaluate held-out runs with all existing fits, references and thresholds frozen.

        Returns
        -------
        DataFrame
            One final held-out comparison. Repeated views reuse saved results.
        """
        self._verify_frozen()
        if not self.revealed:
            test=self._raw_scores("test")
            for name in [*CLASSIC,*FORECAST_COLUMNS]:
                column=name+"_raw_score" if name in CLASSIC else FORECAST_COLUMNS[name]
                test[name]=core.ecdf_transform(test[column].to_numpy(),self.references[name])
            self._fuse_frame(test)
            self.rows=pd.concat([self.rows,test],ignore_index=True)
            self._holdout_score_hash = self._score_rows_hash(self.rows[self.rows.split == "test"])
            self.revealed=True
            self.rows.to_csv(self.attempt/"scores.csv.gz",index=False)
        report=self.metric_table("test")
        report.to_csv(self.attempt/"holdout_metrics.csv",index=False)
        return report

    def plot_timeline(self, run_id: str, names: list[str], raw: bool=False) -> None:
        """Show one run's observations, scores and alarms as a single aligned artifact.

        Parameters
        ----------
        run_id : str
            Validation run, or test run after reveal.
        names : list[str]
            Up to four detector column names.
        raw : bool
            Show raw rather than ECDF scores; alarm panel omitted in this mode.

        Returns
        -------
        None
            Displays a three-panel figure, with original annotated state shading.
        """
        if self.decision is not None:self._verify_frozen()
        if run_id in core.SPLITS["test"] and not self.revealed:raise RuntimeError("Held-out viewer is locked until freeze")
        b=self.rows[self.rows.run_id==run_id]
        if b.empty:raise ValueError("Run is not present in the current scored stage")
        t=(pd.to_datetime(b.datetime)-pd.to_datetime(b.datetime).iloc[0]).dt.total_seconds().to_numpy()
        fig,axes=plt.subplots(2 if raw else 3,1,figsize=(11,5 if raw else 7),sharex=True)
        for channel in ["Accelerometer1RMS","Current","Volume Flow RateRMS"]:
            axes[0].plot(t,b["z__"+channel],label=channel,lw=1)
        for index,name in enumerate(names):
            column=(name+"_raw_score" if name in CLASSIC else FORECAST_COLUMNS.get(name,name)) if raw else name
            axes[1].plot(t,b[column],label=name,color=COLORS.get(name),lw=1)
            if not raw and name in self.thresholds:
                alarm=(b[name]>=self.thresholds[name]["threshold"]).astype(int)
                axes[2].bar(t[alarm.to_numpy()==1],np.full(int(alarm.sum()),.7),bottom=index-.35,width=1.,align="edge",color=COLORS.get(name),edgecolor="none")
        for axis in axes:
            axis.fill_between(t,0,1,where=b.anomaly.to_numpy()==1,transform=axis.get_xaxis_transform(),color="#f9a5b5",alpha=.24)
            axis.grid(alpha=.15)
        axes[0].set(ylabel="Normal-fit robust z",title=f"{run_id} — shaded interval: anomalous state")
        axes[1].set(ylabel="Raw score" if raw else "Normal-reference rank")
        axes[-1].set_xlabel("Seconds after first scored observation")
        if not raw:
            axes[2].set_yticks(range(len(names)),names)
            axes[2].set_ylim(-.6,len(names)-.4)
            axes[2].set_ylabel("Alarm active")
        for axis in axes[:2]:axis.legend(loc="upper right",fontsize=8,ncol=3)
        fig.tight_layout();plt.show()

    def validation_fusion_case(self) -> str:
        """Choose one explanatory run using validation results only.

        Returns
        -------
        str
            Run with the largest mean-fusion F1 gain over window IF at existing
            thresholds. This chooses a diagnostic view, never a new model or setting.
        """
        rows=self.rows[self.rows.split=="validation"]
        a=core.evaluate_runs(rows,"ensemble_mean",self.thresholds["ensemble_mean"]["threshold"]).set_index("run_id")
        b=core.evaluate_runs(rows,"if_window",self.thresholds["if_window"]["threshold"]).set_index("run_id")
        return str((a.f1-b.f1).sort_values(ascending=False,kind="stable").index[0])

    def plot_pr(self, split: str, names: list[str]) -> None:
        """Display threshold-free pooled state precision–recall curves.

        Parameters
        ----------
        split : str
            validation or revealed test.
        names : list[str]
            Detector columns to compare.

        Returns
        -------
        None
            One labelled PR figure. Main tables additionally use macro per-run AP.
        """
        if self.decision is not None:self._verify_frozen()
        if split=="test" and not self.revealed:raise RuntimeError("Holdout not revealed")
        b=self.rows[self.rows.split==split]
        fig,ax=plt.subplots(figsize=(7,4.5))
        for name in names:
            precision,recall,_=precision_recall_curve(b.anomaly,b[name]);ax.plot(recall,precision,label=name,color=COLORS.get(name))
        ax.axhline(b.anomaly.mean(),ls=":",color="gray",label="anomaly prevalence")
        ax.set(xlabel="State recall",ylabel="State precision",title=f"{split.title()}: pooled unadjusted PR",xlim=(0,1),ylim=(0,1.02));ax.legend(fontsize=8);fig.tight_layout();plt.show()

    def plot_complementarity(self) -> None:
        """Display score correlation and validation alarm overlap at fixed thresholds.

        Returns
        -------
        None
            Two panels answering whether the three classical sources differ.
        """
        if self.decision is not None:self._verify_frozen()
        b=self.rows[self.rows.split=="validation"]
        names=["robust","pca_q","if_window"]
        corr=b[names].corr(method="spearman").to_numpy()
        alarms=np.column_stack([b[n]>=self.thresholds[n]["threshold"] for n in names])
        jaccard=np.full((3,3),np.nan)
        for i in range(3):
            for j in range(3):
                union=(alarms[:,i]|alarms[:,j]).sum();jaccard[i,j]=(alarms[:,i]&alarms[:,j]).sum()/union if union else np.nan
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
        cmap=plt.get_cmap("Blues").copy();cmap.set_bad("#eeeeee")
        for ax,matrix,title in [(axes[0],corr,"Score rank correlation"),(axes[1],jaccard,"Alarm-set overlap (Jaccard)")]:
            image=ax.imshow(matrix,vmin=0,vmax=1,cmap=cmap);ax.set_xticks(range(3),names,rotation=20);ax.set_yticks(range(3),names);ax.set_title(title)
            for i in range(3):
                for j in range(3):ax.text(j,i,f"{matrix[i,j]:.2f}" if np.isfinite(matrix[i,j]) else "no\nalarms",ha="center",va="center",color="white" if matrix[i,j]>.65 else "black")
        fig.colorbar(image,ax=axes.ravel().tolist(),shrink=.7);plt.show()

    def plot_calibration_tails(self) -> None:
        """Compare raw healthy calibration/validation tails without retuning any method.

        Returns
        -------
        None
            Four panels with separate raw-score axes and a visible calibration maximum.
        """
        fig,axes=plt.subplots(2,2,figsize=(10,6))
        for ax,name in zip(axes.ravel(),CLASSIC):
            column=name+"_raw_score"
            for split,color in [("calibration","#18a59f"),("validation","#e35d77")]:
                mask=(self.rows.split==split)&(self.rows.anomaly==0)
                if split=="calibration":mask &= self.rows.normal_prefix
                values=np.sort(self.rows.loc[mask,column].to_numpy())
                survival=(len(values)-np.arange(len(values)))/(len(values)+1)
                ax.step(values,survival,label=f"{split} normal",color=color,where="post",lw=1)
            maximum=self.rows.loc[(self.rows.split=="calibration")&self.rows.normal_prefix,column].max()
            ax.axvline(maximum,color="#777777",ls=":",lw=1,label="calibration maximum")
            ax.axhline(.01,color="#999999",ls="--",lw=.8,label="1% reference line")
            ax.set(yscale="log",xlabel=f"Raw {name} score",ylabel="Normal fraction at/above score",title=name,ylim=(.0002,1.1))
            ax.grid(alpha=.15)
            ax.legend(fontsize=7)
        fig.suptitle("Does the independent normal tail transfer? Diagnostic only; protocol stays frozen")
        fig.tight_layout();plt.show()

    def plot_forecast(self, run_id: str="valve1/8", channel: str="Volume Flow RateRMS") -> None:
        """Inspect recorded median/interval forecasts with exact block origins.

        Parameters
        ----------
        run_id : str
            Validation run ID; held-out runs require prior reveal.
        channel : str
            One original sensor channel.

        Returns
        -------
        None
            One forecast-versus-observation figure; forecasts are explicitly recorded.
        """
        if run_id in core.SPLITS["test"] and not self.revealed:raise RuntimeError("Held-out trace is locked")
        trace=pd.read_csv(self.assets/"forecast_trace.csv.gz",parse_dates=["datetime","origin_datetime"],float_precision="round_trip")
        b=trace[(trace.run_id==run_id)&(trace.channel==channel)].sort_values("datetime")
        if b.empty:raise ValueError("No exact forecast trace for this run/channel")
        t=(b.datetime-b.datetime.iloc[0]).dt.total_seconds()
        fig,ax=plt.subplots(figsize=(11,3.8));ax.fill_between(t,b.q10,b.q90,alpha=.22,color=COLORS["chronos_residual"],label="recorded q10–q90")
        state=self.frames[run_id].reindex(pd.DatetimeIndex(b.datetime)).anomaly.to_numpy()
        ax.fill_between(t,0,1,where=state==1,transform=ax.get_xaxis_transform(),color="#f9a5b5",alpha=.18,label="anomalous state")
        ax.plot(t,b.actual,color="black",lw=1,label="observed");ax.plot(t,b.q50,color=COLORS["chronos_residual"],lw=1,label="recorded median");ax.plot(t,b.naive,color=COLORS["naive_residual"],alpha=.8,lw=.8,label="matched naive")
        ax.set(xlabel="Seconds after first forecast target",ylabel=channel,title=f"{run_id}: recorded Chronos-2-small forecast; fixed context=128, horizon=16")
        ax.legend(fontsize=8,ncol=4);ax.grid(alpha=.15);fig.tight_layout();plt.show()

    def save_submission(self, answers: dict[str,str]) -> dict[str,str]:
        """Save the decision, results and written interpretations without model caches.

        Parameters
        ----------
        answers : dict[str,str]
            Student responses to final interpretation questions.

        Returns
        -------
        dict[str,str]
            Paths to the saved decision, holdout metrics and written answers.
        """
        if not self.revealed:raise RuntimeError("Finish the frozen holdout evaluation first")
        self._verify_frozen()
        required = {"representation", "fusion", "forecasting", "transfer", "scope"}
        if not isinstance(answers, dict) or set(answers) != required:
            raise ValueError("Provide exactly the five required interpretation answers: " + ", ".join(sorted(required)))
        placeholders = ("write your", "write the", "compare chronos, matched naive", "report the frozen choice", "state the limits and your next")
        for name, answer in answers.items():
            if not isinstance(answer, str) or len(answer.strip()) < 20 or any(phrase in answer.strip().lower() for phrase in placeholders):
                raise ValueError("Replace the placeholder with a substantive evidence-based answer: " + name)
        destination=self.attempt/"interpretations.json";destination.write_text(json.dumps(answers,indent=2,allow_nan=False))
        return {"decision":str(self.attempt/"frozen_decision.json"),"metrics":str(self.attempt/"holdout_metrics.csv"),"interpretations":str(destination)}


def compact_report(report: pd.DataFrame) -> pd.DataFrame:
    """Make the primary comparison readable without changing stored metric values.

    Parameters
    ----------
    report : DataFrame
        Full metric table returned by LabRuntime.metric_table.

    Returns
    -------
    DataFrame
        Compact scalar columns; full precision and secondary fields remain in report.
    """
    columns={"detector":"detector","macro_AP":"AP","macro_state_F1":"state F1",
             "macro_state_coverage":"coverage","pooled_normal_alarm_fraction":"normal alarm fraction",
             "onsets_detected_within_60s":"onsets ≤60s","policy_on_this_split":"policy"}
    return report[list(columns)].rename(columns=columns).round(4)
