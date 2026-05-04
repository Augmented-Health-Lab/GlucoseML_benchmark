"""
clarke/ceg_seg.py

Clarke Error Grid (CEG) and Surveillance Error Grid (SEG) utilities for
glucose-prediction evaluation.

Contents
--------
1. Core scoring + plots (extracted from results/plot_CEG_SEG.ipynb)
   - classify_ceg_zone, plot_ceg
   - _build_seg_table, plot_seg
   - ceg_zone_pcts, seg_scores  (lightweight, no-plot variants)

2. Per-method summary (the pattern repeated across Chronos, LSTM,
   TimeLLM, TimesFM, CALF, persistent baseline)
   - summarize_method

3. Per-test-dataset Zone A summary (NEW)
   - load_dataset_participants
   - patient_to_dataset
   - summarize_zone_a_per_dataset
   - zone_a_pivot

Usage from results/plot_CEG_SEG.ipynb
-------------------------------------
    import sys, os
    sys.path.insert(0, os.path.abspath('..'))
    from clarke.ceg_seg import (
        plot_ceg, plot_seg,
        summarize_method,
        summarize_zone_a_per_dataset, zone_a_pivot,
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ─── Clarke Error Grid ────────────────────────────────────────────────────────
def classify_ceg_zone(ref, meas):
    """Classify a single (reference, measurement) pair into CEG zones A-E."""
    if (ref <= 70 and meas >= 180) or (ref >= 180 and meas <= 70):
        return 'E'
    if ref < 70 and meas < 70:
        return 'A'
    if abs(meas - ref) / ref <= 0.20:
        return 'A'
    if ref <= 70 and 70 <= meas <= 180:
        return 'D'
    if ref >= 240 and 70 <= meas <= 180:
        return 'D'
    if ref >= 70 and meas >= ref + 110:
        return 'C'
    if ref >= 130 and meas <= 70 and meas <= 1.4 * (ref - 130):
        return 'C'
    return 'B'


def plot_ceg(ref_values, pred_values, title='Clarke Error Grid',
             ax=None, s=2, alpha=0.3):
    """Plot Clarke Error Grid; return {'A':pct, ..., 'E':pct}."""
    ref_values  = np.asarray(ref_values,  dtype=float)
    pred_values = np.asarray(pred_values, dtype=float)

    zones = np.array([classify_ceg_zone(r, p)
                      for r, p in zip(ref_values, pred_values)])
    total = len(zones)
    pct = {z: np.sum(zones == z) / total * 100 for z in 'ABCDE'}

    zone_colors = {'A': '#2ecc71', 'B': '#3498db', 'C': '#f39c12',
                   'D': '#e67e22', 'E': '#e74c3c'}

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    for zone, color in zone_colors.items():
        mask = zones == zone
        if mask.any():
            ax.scatter(ref_values[mask], pred_values[mask],
                       c=color, s=s, alpha=alpha, rasterized=True,
                       label=f'Zone {zone} ({pct[zone]:.1f}%)')

    lw, lc = 1.2, 'black'
    ax.plot([0, 400], [0, 400], color=lc, lw=lw, ls='--', zorder=3)
    ax.plot([0,    58.3 ], [70,  70 ], color=lc, lw=lw)
    ax.plot([58.3, 333.3], [70,  400], color=lc, lw=lw)
    ax.plot([0, 400], [0, 320], color=lc, lw=lw)
    ax.plot([70, 70], [0,   180], color=lc, lw=lw)
    ax.plot([0,  70], [180, 180], color=lc, lw=lw)
    ax.plot([70, 400], [70, 70], color=lc, lw=lw)
    ax.plot([130, 180], [0,  70], color=lc, lw=lw)
    ax.plot([180, 180], [0,  70], color=lc, lw=lw)
    ax.plot([240, 240], [70,  180], color=lc, lw=lw)
    ax.plot([240, 400], [180, 180], color=lc, lw=lw)
    ax.plot([70, 290], [180, 400], color=lc, lw=lw)

    fs = 13
    for x, y, lbl, c in [
        (30,  20,  'A', '#27ae60'), (355, 330, 'A', '#27ae60'),
        (105, 165, 'B', '#2980b9'), (185, 105, 'B', '#2980b9'),
        (110, 330, 'C', '#d35400'), (162,  20, 'C', '#d35400'),
        (25,  125, 'D', '#e67e22'), (305, 125, 'D', '#e67e22'),
        (20,  360, 'E', '#c0392b'), (290,  25, 'E', '#c0392b'),
    ]:
        ax.text(x, y, lbl, fontsize=fs, fontweight='bold', color=c)

    ax.set_xlim(0, 400); ax.set_ylim(0, 400)
    ax.set_xlabel('Reference Glucose (mg/dL)', fontsize=12)
    ax.set_ylabel('Predicted Glucose (mg/dL)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc='upper left', fontsize=9, markerscale=4)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    return pct


# ─── Surveillance Error Grid ──────────────────────────────────────────────────
def _build_seg_table(max_glucose: int = 600) -> np.ndarray:
    """Pre-compute SEG risk-score lookup table (Klonoff et al. 2014)."""
    ref_ax  = np.arange(max_glucose, dtype=float)
    meas_ax = np.arange(max_glucose, dtype=float)
    REF, MEAS = np.meshgrid(ref_ax, meas_ax, indexing='ij')

    fda_lo = np.where(REF < 100, REF - 15, REF * 0.85)
    fda_hi = np.where(REF < 100, REF + 15, REF * 1.15)
    in_fda = (MEAS >= fda_lo) & (MEAS <= fda_hi)

    error = MEAS - REF
    abs_e = np.abs(error)
    ref_safe = np.maximum(REF, 1.0)
    pct_e = error / ref_safe * 100.0

    table = np.zeros((max_glucose, max_glucose), dtype=np.float32)

    def _set(mask, value):
        table[mask] = np.maximum(table[mask], value)

    lower = error < 0
    _set(lower & ~in_fda & (REF <  70) & (abs_e <= 30), 0.5)
    _set(lower & ~in_fda & (REF >= 70) & (REF <= 180) & (abs_e <= 40), 0.5)
    _set(lower & ~in_fda & (REF >  180) & (pct_e >= -20), 0.5)
    _set(lower & ~in_fda & (REF <  70) & (abs_e > 30) & (abs_e <= 60), 1.0)
    _set(lower & ~in_fda & (REF >= 70) & (REF <= 180) & (abs_e > 40) & (abs_e <= 80), 1.0)
    _set(lower & ~in_fda & (REF >  180) & (pct_e < -20) & (pct_e >= -40), 1.0)
    _set(lower & ~in_fda & (REF <  70) & (abs_e > 60) & (abs_e <= 90), 2.0)
    _set(lower & ~in_fda & (REF >= 70) & (REF <= 180) & (abs_e > 80) & (abs_e <= 120), 2.0)
    _set(lower & ~in_fda & (REF >  180) & (pct_e < -40) & (pct_e >= -60), 2.0)
    _set(lower & ~in_fda & (REF <  70) & (abs_e > 90), 3.0)
    _set(lower & ~in_fda & (REF >= 70) & (REF <= 180) & (abs_e > 120), 3.0)
    _set(lower & ~in_fda & (REF >  180) & (pct_e < -60), 3.0)

    upper = error > 0
    _set(upper & ~in_fda & (REF <  70) & (abs_e <= 30), 0.5)
    _set(upper & ~in_fda & (REF >= 70) & (REF <= 180) & (abs_e <= 40), 0.5)
    _set(upper & ~in_fda & (REF >  180) & (pct_e <= 20), 0.5)
    _set(upper & ~in_fda & (REF <  70) & (abs_e > 30) & (abs_e <= 50), 1.0)
    _set(upper & ~in_fda & (REF >= 70) & (REF <= 180) & (abs_e > 40) & (abs_e <= 80), 1.0)
    _set(upper & ~in_fda & (REF >  180) & (pct_e > 20) & (pct_e <= 40), 1.0)
    _set(upper & ~in_fda & (REF <  70) & (abs_e > 50) & (abs_e <= 80), 2.0)
    _set(upper & ~in_fda & (REF >= 70) & (REF <= 180) & (abs_e > 80) & (abs_e <= 120), 2.0)
    _set(upper & ~in_fda & (REF >  180) & (pct_e > 40) & (pct_e <= 70), 2.0)
    _set(upper & ~in_fda & (REF <  70) & (abs_e > 80), 3.0)
    _set(upper & ~in_fda & (REF >= 70) & (REF <= 180) & (abs_e > 120), 3.0)
    _set(upper & ~in_fda & (REF >  180) & (pct_e > 70), 3.0)

    return table


SEG_TABLE = _build_seg_table(600)


def plot_seg(ref_values, pred_values, title='Surveillance Error Grid',
             ax=None, s=2, alpha=0.3):
    """Plot SEG; return (mean_risk, zone_a_pct)."""
    ref_values  = np.asarray(ref_values,  dtype=float)
    pred_values = np.asarray(pred_values, dtype=float)

    ref_idx  = np.clip(ref_values.astype(int),  0, SEG_TABLE.shape[0] - 1)
    pred_idx = np.clip(pred_values.astype(int), 0, SEG_TABLE.shape[1] - 1)
    risk_scores = SEG_TABLE[ref_idx, pred_idx].astype(float)
    mean_risk  = float(np.mean(risk_scores))
    zone_a_pct = float(np.mean(risk_scores == 0.0) * 100)

    seg_cmap = ListedColormap(['#00b050', '#92d050', '#ffff00', '#ff9900', '#ff0000'])
    risk_bins   = [0.0, 0.4, 0.9, 1.5, 2.5, 3.5]
    zone_labels = ['0 – None (A)', '0.5 – Slight', '1.0 – Moderate',
                   '2.0 – Great',  '3.0 – Extreme']
    zone_colors_list = ['#00b050', '#92d050', '#ffff00', '#ff9900', '#ff0000']
    color_idx = np.digitize(risk_scores, risk_bins[1:])

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    bg_ref  = np.arange(0, 401)
    bg_meas = np.arange(0, 401)
    BG_R, BG_M = np.meshgrid(bg_ref, bg_meas)
    bg_risk = SEG_TABLE[np.clip(BG_R, 0, SEG_TABLE.shape[0] - 1),
                        np.clip(BG_M, 0, SEG_TABLE.shape[1] - 1)]
    bg_zone = np.digitize(bg_risk, risk_bins[1:])
    ax.pcolormesh(bg_ref, bg_meas, bg_zone,
                  cmap=seg_cmap, vmin=0, vmax=4, alpha=0.25, zorder=0)

    for i, (label, color) in enumerate(zip(zone_labels, zone_colors_list)):
        mask = color_idx == i
        if mask.any():
            ax.scatter(ref_values[mask], pred_values[mask],
                       c=color, s=s, alpha=alpha, rasterized=True,
                       label=f'Risk {label} ({np.sum(mask)/len(mask)*100:.1f}%)')

    ax.plot([0, 400], [0, 400], 'k--', lw=1.0, zorder=3)
    ax.set_xlim(0, 400); ax.set_ylim(0, 400)
    ax.set_xlabel('Reference Glucose (mg/dL)', fontsize=12)
    ax.set_ylabel('Predicted Glucose (mg/dL)', fontsize=12)
    ax.set_title(f'{title}\nMean Risk = {mean_risk:.4f}  |  '
                 f'Zone A (no risk) = {zone_a_pct:.1f}%', fontsize=13)
    ax.legend(loc='upper left', fontsize=8, markerscale=4)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.2)

    return mean_risk, zone_a_pct


# ─── Lightweight scoring (no plot) ────────────────────────────────────────────
def ceg_zone_pcts(ref, pred) -> dict:
    """Return {'A':pct, ..., 'E':pct} without plotting."""
    ref  = np.asarray(ref,  dtype=float)
    pred = np.asarray(pred, dtype=float)
    zones = np.array([classify_ceg_zone(r, p) for r, p in zip(ref, pred)])
    return {z: float(np.mean(zones == z) * 100) for z in 'ABCDE'}


def seg_scores(ref, pred) -> tuple[float, float]:
    """Return (mean_risk, zone_a_pct) without plotting."""
    ref  = np.asarray(ref,  dtype=float)
    pred = np.asarray(pred, dtype=float)
    ref_idx  = np.clip(ref.astype(int),  0, SEG_TABLE.shape[0] - 1)
    pred_idx = np.clip(pred.astype(int), 0, SEG_TABLE.shape[1] - 1)
    risk = SEG_TABLE[ref_idx, pred_idx].astype(float)
    return float(risk.mean()), float((risk == 0).mean() * 100)


# ─── Method-level summary (the cells 13/20/26/33/44/50 pattern) ───────────────
def summarize_method(results: Mapping, default_dataset: str | None = None) -> pd.DataFrame:
    """
    Convert a per-method results dict into a tidy summary DataFrame.

    Accepts the three key formats used in the notebook:
      - (ds_name, horizon)  e.g. results_summary  (Chronos), calf_results
      - ds_name (str)       e.g. lstm_results, timellm_results, persistent_results
      - horizon (int/str)   e.g. timesfm_results — uses ``default_dataset``
                            (or 'Method') as the dataset label

    Each value must be ``{'ceg': {'A':..,'B':..}, 'seg_mean_risk': float,
    'seg_zone_a_pct': float}``.

    Returns
    -------
    DataFrame indexed by (Dataset, Horizon) with CEG zone %, SEG mean risk,
    and SEG Zone A %.
    """
    rows = []
    for key, r in results.items():
        if isinstance(key, tuple):
            ds_name, horizon = key
        elif isinstance(key, (int, np.integer)):
            ds_name, horizon = (default_dataset or 'Method'), f'{int(key)}min'
        else:
            ds_name, horizon = key, '30min'
        row = {'Dataset': ds_name, 'Horizon': str(horizon)}
        for z in 'ABCDE':
            row[f'CEG Zone {z} (%)'] = round(r['ceg'][z], 2)
        row['SEG Mean Risk']  = round(r['seg_mean_risk'], 4)
        row['SEG Zone A (%)'] = round(r['seg_zone_a_pct'], 2)
        rows.append(row)
    return pd.DataFrame(rows).set_index(['Dataset', 'Horizon'])


# ─── Per-test-dataset Zone A summary (NEW) ────────────────────────────────────
def load_dataset_participants(path: str | Path) -> dict:
    """Load test_dataset/dataset_participants.json -> {dataset: [patient_id]}."""
    with open(path) as fh:
        return json.load(fh)


def patient_to_dataset(mapping: Mapping[str, Sequence[str]]) -> dict:
    """Reverse {dataset: [patients]} into {patient: dataset}; error on collisions."""
    rev: dict = {}
    dupes: dict = {}
    for ds, patients in mapping.items():
        for p in patients:
            if p in rev and rev[p] != ds:
                dupes.setdefault(p, [rev[p]]).append(ds)
            rev[p] = ds
    if dupes:
        raise ValueError(
            "Patient id collision across datasets (first 5): "
            f"{dict(list(dupes.items())[:5])}"
        )
    return rev


def _default_participants_path() -> Path:
    return (Path(__file__).resolve().parent.parent
            / 'test_dataset' / 'dataset_participants.json')


def summarize_zone_a_per_dataset(
    df: pd.DataFrame,
    *,
    patient_col: str | None = None,
    dataset_col: str | None = None,
    pred_col: str = 'Prediction',
    truth_col: str = 'GroundTruth',
    horizon_col: str | None = 'Horizon',
    participants_path: str | Path | None = None,
    method_name: str = 'Method',
    warn_unmapped: bool = True,
) -> pd.DataFrame:
    """
    Summarize CEG Zone A %, SEG Zone A %, and SEG mean risk per
    (test_dataset, horizon) for the predictions in ``df``.

    Parameters
    ----------
    df : DataFrame containing prediction / ground-truth columns.
    patient_col : column with patient ids (matched against the keys of
        dataset_participants.json). Required if ``dataset_col`` is None.
    dataset_col : column already containing dataset names (TimesFM and the
        persistent-baseline output store this directly). When set,
        ``patient_col`` / ``participants_path`` are ignored.
    pred_col, truth_col, horizon_col : DataFrame column names. Pass
        ``horizon_col=None`` if the data covers only a single horizon and
        you want one row per dataset.
    participants_path : override the default path
        (`<repo>/test_dataset/dataset_participants.json`).
    method_name : label stored in the output's ``Method`` column.

    Returns
    -------
    DataFrame with columns:
        Method, Dataset, Horizon, N, CEG Zone A (%), SEG Zone A (%), SEG Mean Risk
    """
    if dataset_col is None and patient_col is None:
        raise ValueError("Provide either dataset_col or patient_col.")

    needed = [c for c in (patient_col, dataset_col, pred_col,
                          truth_col, horizon_col) if c is not None]
    work = df.loc[:, needed].copy()

    if dataset_col is None:
        path = participants_path or _default_participants_path()
        rev  = patient_to_dataset(load_dataset_participants(path))
        work['Dataset'] = work[patient_col].map(rev)
        unmapped = work.loc[work['Dataset'].isna(), patient_col].unique()
        if len(unmapped):
            if warn_unmapped:
                print(f"[summarize_zone_a_per_dataset] {len(unmapped)} "
                      f"patient id(s) not in {Path(path).name}; dropping. "
                      f"Examples: {list(unmapped)[:5]}")
            work = work.dropna(subset=['Dataset'])
        ds_col_used = 'Dataset'
    else:
        ds_col_used = dataset_col

    group_keys = [ds_col_used] + ([horizon_col] if horizon_col is not None else [])

    rows = []
    for keys, sub in work.groupby(group_keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        ds_name = keys[0]
        horizon = keys[1] if horizon_col is not None else '-'
        ref  = sub[truth_col].to_numpy(dtype=float)
        pred = sub[pred_col].to_numpy(dtype=float)
        ceg = ceg_zone_pcts(ref, pred)
        seg_mean, seg_a = seg_scores(ref, pred)
        rows.append({
            'Method': method_name,
            'Dataset': ds_name,
            'Horizon': str(horizon),
            'N': int(len(sub)),
            'CEG Zone A (%)': round(ceg['A'], 2),
            'SEG Zone A (%)': round(seg_a, 2),
            'SEG Mean Risk':  round(seg_mean, 4),
        })
    return (pd.DataFrame(rows)
              .sort_values(['Method', 'Dataset', 'Horizon'])
              .reset_index(drop=True))


def zone_a_pivot(per_method_dfs: Iterable[pd.DataFrame], *,
                 value_col: str = 'CEG Zone A (%)',
                 horizon: str | None = None) -> pd.DataFrame:
    """
    Combine per-method per-dataset summaries into a Dataset × Method pivot of
    Zone A %, optionally filtered to a single horizon.
    """
    combined = pd.concat(list(per_method_dfs), ignore_index=True)
    if horizon is not None:
        combined = combined[combined['Horizon'].astype(str) == str(horizon)]
    return combined.pivot_table(index='Dataset',
                                columns='Method',
                                values=value_col)


# ─── Per-participant Zone A + cross-participant aggregation ───────────────────
def summarize_zone_a_per_participant(
    df: pd.DataFrame,
    *,
    patient_col: str = 'participant_id',
    pred_col: str = 'prediction',
    truth_col: str = 'ground_truth',
    dataset_col: str = 'dataset',
    horizon_col: str | None = 'horizon_minutes',
    method_name: str = 'Method',
) -> pd.DataFrame:
    """
    For each (dataset, [horizon,] participant), compute CEG Zone A %, SEG Zone A %,
    and SEG mean risk.

    Default column names match the *_with_raw_quantile.py output CSVs (timer,
    uni2ts, timesfm, ...). Pass ``horizon_col=None`` if your data covers a
    single horizon.
    """
    group_cols = [dataset_col]
    if horizon_col is not None:
        group_cols.append(horizon_col)
    group_cols.append(patient_col)

    rows = []
    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        ds_name = keys[0]
        if horizon_col is not None:
            horizon = keys[1]
            participant = keys[2]
        else:
            horizon = None
            participant = keys[1]
        ref  = sub[truth_col].to_numpy(dtype=float)
        pred = sub[pred_col].to_numpy(dtype=float)
        ceg = ceg_zone_pcts(ref, pred)
        seg_mean, seg_a = seg_scores(ref, pred)
        row = {
            'Method': method_name,
            'Dataset': ds_name,
            'Participant': participant,
            'N': int(len(sub)),
            'CEG Zone A (%)': float(ceg['A']),
            'SEG Zone A (%)': float(seg_a),
            'SEG Mean Risk':  float(seg_mean),
        }
        if horizon_col is not None:
            row['Horizon'] = horizon
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_zone_a_across_participants(
    per_participant_df: pd.DataFrame,
    *,
    open_excluded: Iterable[str] = ('5_T1DEXI', 'OhioT1DM', '8_DiaTrend'),
    horizon_col: str | None = 'Horizon',
    digits: int = 2,
) -> pd.DataFrame:
    """
    Given the per-(dataset, [horizon,] participant) Zone A table from
    ``summarize_zone_a_per_participant``, aggregate across participants to
    mean ± std per (Method, Dataset, [Horizon]) and append two overall rows:

      - ``Overall (Open)``  : pool all participants from datasets NOT in
                              ``open_excluded`` (default excludes 5_T1DEXI,
                              OhioT1DM, 8_DiaTrend — the closed/restricted
                              datasets).
      - ``Overall (Total)`` : pool all participants from all datasets.

    Overall rows are unweighted across participants (each participant counts
    once regardless of which dataset they came from).
    """
    open_set = set(open_excluded)
    has_horizon = horizon_col is not None and horizon_col in per_participant_df.columns

    def _row(sub: pd.DataFrame, method, dataset, horizon=None) -> dict:
        row = {'Method': method, 'Dataset': dataset}
        if has_horizon:
            row['Horizon'] = horizon
        row['N_Participants']     = int(len(sub))
        row['CEG Zone A mean (%)'] = round(float(sub['CEG Zone A (%)'].mean()), digits)
        row['CEG Zone A std (%)']  = round(float(sub['CEG Zone A (%)'].std()),  digits)
        row['SEG Zone A mean (%)'] = round(float(sub['SEG Zone A (%)'].mean()), digits)
        row['SEG Zone A std (%)']  = round(float(sub['SEG Zone A (%)'].std()),  digits)
        row['SEG Mean Risk mean']  = round(float(sub['SEG Mean Risk'].mean()),  4)
        row['SEG Mean Risk std']   = round(float(sub['SEG Mean Risk'].std()),   4)
        return row

    rows = []
    per_ds_keys = ['Method', 'Dataset'] + (['Horizon'] if has_horizon else [])
    for keys, sub in per_participant_df.groupby(per_ds_keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        method, dataset = keys[0], keys[1]
        horizon = keys[2] if has_horizon else None
        rows.append(_row(sub, method, dataset, horizon))

    overall_keys = ['Method'] + (['Horizon'] if has_horizon else [])
    open_df = per_participant_df[~per_participant_df['Dataset'].isin(open_set)]
    for keys, sub in open_df.groupby(overall_keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        method = keys[0]
        horizon = keys[1] if has_horizon else None
        rows.append(_row(sub, method, 'Overall (Open)', horizon))
    for keys, sub in per_participant_df.groupby(overall_keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        method = keys[0]
        horizon = keys[1] if has_horizon else None
        rows.append(_row(sub, method, 'Overall (Total)', horizon))

    out = pd.DataFrame(rows)
    sort_keys = ['Method', 'Dataset'] + (['Horizon'] if has_horizon else [])
    return out.sort_values(sort_keys).reset_index(drop=True)
