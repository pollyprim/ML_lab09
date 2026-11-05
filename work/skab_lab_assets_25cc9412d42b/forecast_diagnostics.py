"""Provided validation-only diagnostics for the genuine recorded forecasts.

Forecast accuracy and anomaly-state detection answer different questions. This
module computes forecast accuracy without fitting, threshold selection, repair
of quantiles, or access to held-out labels in its returned diagnostics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from skab_lab import CHANNELS, SPLITS


def forecast_quality_validation(session) -> dict[str, pd.DataFrame]:
    """Compare recorded Chronos medians with matched block persistence.

    Parameters
    ----------
    session : LabRuntime
        A session after development scoring. Its validation observations and
        healthy-fit scales define the exact comparison population.

    Returns
    -------
    dict[str, DataFrame]
        ``summary`` has all/healthy/abnormal rows. Each metric is first averaged
        over timestamps in a run x channel cell, then equally across those
        cells, matching the lecture's descriptive validation analysis.
        ``by_channel`` averages run-cell means within each sensor;
        ``per_run_channel`` exposes the 40 paired cell comparisons; ``audit``
        reports exact-key, causality and raw-quantile checks. No test diagnostics
        or forecast arrays are returned.

    Notes
    -----
    Normalized MAE divides each absolute error by that sensor's fixed healthy-
    fit robust scale. Persistence holds the last preprocessed context value for
    the same 16-step forecast block. Intervals belong only to Chronos. Quantiles
    are retained literally, including any q50 outside [q10, q90].
    """
    if session.rows is None:
        raise RuntimeError("Score development runs before inspecting forecast quality")
    expected = session.rows.loc[
        session.rows.split.eq("validation"),
        ["run_id", "datetime", "anomaly", *CHANNELS],
    ].copy()
    if set(expected.run_id) != set(SPLITS["validation"]):
        raise ValueError("Expected every fixed validation run, and no other split")
    if expected.duplicated(["run_id", "datetime"]).any():
        raise ValueError("Validation observations have duplicate run/time identities")
    expected["datetime"] = pd.to_datetime(expected.datetime)
    expected = expected.melt(
        id_vars=["run_id", "datetime", "anomaly"],
        value_vars=CHANNELS,
        var_name="channel",
        value_name="expected_actual",
    )
    keys = ["run_id", "datetime", "channel"]
    expected = expected.sort_values(keys).reset_index(drop=True)
    trace = pd.read_csv(
        session.assets / "forecast_trace.csv.gz",
        parse_dates=["datetime", "origin_datetime"],
        float_precision="round_trip",
    )
    # Filter first: no held-out labels or results enter any aggregation below.
    trace = trace.loc[trace.run_id.isin(SPLITS["validation"])].copy()
    if trace.duplicated(keys).any():
        raise ValueError("Validation forecast cache has duplicate run/time/channel keys")
    trace = trace.sort_values(keys).reset_index(drop=True)
    expected_keys = pd.MultiIndex.from_frame(expected[keys])
    trace_keys = pd.MultiIndex.from_frame(trace[keys])
    if not trace_keys.equals(expected_keys):
        raise ValueError("Forecast and validation populations differ; no rows may silently disappear")
    required = ["actual", "q10", "q50", "q90", "naive"]
    if not np.isfinite(trace[required].to_numpy()).all():
        raise ValueError("Recorded validation forecast values must be finite")
    actual_error = float(np.max(abs(trace.actual.to_numpy() - expected.expected_actual.to_numpy())))
    if actual_error > 1e-10:
        raise ValueError("Recorded target values differ from the matched validation observations")
    elapsed = (trace.datetime - trace.origin_datetime).dt.total_seconds().to_numpy()
    if not ((elapsed > 0).all() and np.array_equal(elapsed, trace.horizon_step.to_numpy())):
        raise ValueError("Forecast origins must precede their targets by the declared horizon")
    if not trace.horizon_step.between(1, 16).all():
        raise ValueError("Expected the fixed 16-step issuance protocol")
    persistence_error = 0.0
    imputed_origins, origin_count = 0, 0
    for run_id in SPLITS["validation"]:
        frame = session.frames[run_id]
        run = trace.loc[trace.run_id.eq(run_id)]
        origins = pd.DatetimeIndex(run.origin_datetime.unique())
        if not origins.isin(frame.index).all():
            raise ValueError("Forecast origin is absent from its causal preprocessed run")
        origin_count += len(origins)
        imputed_origins += int((~frame.loc[origins, "observed"].astype(bool)).sum())
        for channel in CHANNELS:
            block = run.loc[run.channel.eq(channel)]
            matched = frame.loc[pd.DatetimeIndex(block.origin_datetime), channel].to_numpy(dtype=np.float32).astype(float)
            persistence_error = max(persistence_error, float(np.max(abs(matched - block.naive.to_numpy()))))
    if persistence_error > 1e-10:
        raise ValueError("Persistence must repeat the same last context value at each forecast origin")
    scale = np.asarray(session.scale, dtype=float)
    if scale.shape != (len(CHANNELS),) or not (np.isfinite(scale).all() and (scale > 0).all()):
        raise ValueError("Expected one positive fixed healthy-fit scale per sensor")
    trace["anomaly"] = expected.anomaly.to_numpy()
    trace["scale"] = trace.channel.map(dict(zip(CHANNELS, scale)))
    trace["chronos_nmae"] = abs(trace.actual - trace.q50) / trace.scale
    trace["persistence_nmae"] = abs(trace.actual - trace.naive) / trace.scale
    trace["chronos_mae_physical"] = abs(trace.actual - trace.q50)
    trace["persistence_mae_physical"] = abs(trace.actual - trace.naive)
    trace["coverage80"] = ((trace.q10 <= trace.actual) & (trace.actual <= trace.q90)).astype(float)
    trace["width80_normalized"] = (trace.q90 - trace.q10) / trace.scale
    numeric = ["chronos_nmae", "persistence_nmae", "coverage80", "width80_normalized", "chronos_mae_physical", "persistence_mae_physical"]

    def paired_cells(values):
        groups = values.groupby(["run_id", "channel"], sort=True)
        result = groups[numeric].mean().reset_index()
        result["forecast_values"] = groups.size().to_numpy()
        result["relative_nmae_reduction"] = 1 - result.chronos_nmae / result.persistence_nmae
        return result

    per_run_channel = paired_cells(trace)
    records = []
    for regime, values in [
        ("all", trace),
        ("healthy", trace.loc[trace.anomaly.eq(0)]),
        ("abnormal", trace.loc[trace.anomaly.eq(1)]),
    ]:
        cells = paired_cells(values)
        means = cells[numeric].mean()
        records.append({
            "regime": regime,
            "timestamps": len(values[["run_id", "datetime"]].drop_duplicates()),
            "forecast_values": len(values),
            "paired_run_channel_cells": len(cells),
            "chronos_nmae": means.chronos_nmae,
            "persistence_nmae": means.persistence_nmae,
            "relative_nmae_reduction": 1 - means.chronos_nmae / means.persistence_nmae,
            "chronos_coverage80": means.coverage80,
            "chronos_width80_normalized": means.width80_normalized,
            "chronos_lower_mae_cells": int((cells.chronos_nmae < cells.persistence_nmae).sum()),
        })
    by_channel = per_run_channel.groupby("channel", sort=True)[numeric].mean().reset_index()
    by_channel["relative_nmae_reduction"] = 1 - by_channel.chronos_nmae / by_channel.persistence_nmae
    audit = pd.DataFrame([{
        "split": "validation only",
        "exact_run_time_channel_match": True,
        "actual_join_max_error": actual_error,
        "persistence_origin_max_error": persistence_error,
        "causal_origins": True,
        "unique_forecast_origins": origin_count,
        "causally_filled_origin_endpoints": imputed_origins,
        "q10_above_q90": int((trace.q10 > trace.q90).sum()),
        "q50_outside_q10_q90": int(((trace.q50 < trace.q10) | (trace.q50 > trace.q90)).sum()),
        "quantiles_repaired_or_sorted": False,
    }])
    return {"summary": pd.DataFrame(records), "by_channel": by_channel,
            "per_run_channel": per_run_channel, "audit": audit}
