"""
clarke/plot_ceg_seg_horizon30_table_aligned.py

Re-render the horizon=30min CEG/SEG figures in results/ceg_seg_figures/ so
both the title AND the per-zone legend percentages are computed with the
SAME per-participant aggregation rule used by the *_zone_a_summary.csv
tables (mean across participants of per-participant zone %s).

Title numbers are read directly from each method's *_zone_a_summary.csv
(Dataset = "Overall (Total)", Horizon = 30min). Legend %s are computed
on-the-fly from the per-method raw folders.

Scatter density (point cloud + colour) still reflects the pooled point
distribution — only the printed numbers are per-participant means.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from clarke.ceg_seg import classify_ceg_zone, plot_ceg                  # noqa: E402
from clarke.ceg_seg_upgrade import (                                    # noqa: E402
    plot_seg, seg_risk_scores, _ZONE_BINS, _ZONE_LABELS,
)

# Per-method assemble_* helpers used by the table-building scripts.
chronos_ft = importlib.import_module('clarke.run_zone_a_tables_chronos_finetuned')
lstm_mod   = importlib.import_module('clarke.run_zone_a_tables_lstm')
timellm    = importlib.import_module('clarke.run_zone_a_tables_timellm')
calf_mod   = importlib.import_module('clarke.run_zone_a_tables_calf')

FIG_DIR        = REPO / 'results' / 'ceg_seg_figures'
RESULTS_DIR    = REPO / 'results'
H_MIN          = 30
TABLE_DATASET  = 'Overall (Total)'

FIG_DIR.mkdir(parents=True, exist_ok=True)


# ─── Per-method DataFrame loaders ─────────────────────────────────────────────
# All return columns: ['patient', 'ref', 'pred'] at horizon = H_MIN.
def _glob_concat(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob('*.csv'))
    if not files:
        raise FileNotFoundError(f'No CSVs in {folder}')
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True)


def _std_canonical(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        'patient': df['participant_id'].astype(str).values,
        'ref':     df['ground_truth'].astype(float).values,
        'pred':    df['prediction'].astype(float).values,
    })


def _std_assembled(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        'patient': df['Patient'].astype(str).values,
        'ref':     df['GroundTruth'].astype(float).values,
        'pred':    df['Prediction'].astype(float).values,
    })


def load_canonical_df(folder: Path) -> pd.DataFrame:
    df = _glob_concat(folder)
    df = df[df['horizon_minutes'] == H_MIN]
    return _std_canonical(df)


def load_gpformer_df(folder: Path) -> pd.DataFrame:
    df = _glob_concat(folder)
    df = df[(df['horizon_minutes'] == H_MIN) & (df['horizon_offset_min'] == H_MIN)]
    return _std_canonical(df)


def load_chronos_df(folder: Path) -> pd.DataFrame:
    df = chronos_ft.assemble_chronos(folder)
    df = df[df['Horizon'] == f'{H_MIN}min']
    return _std_assembled(df)


def load_lstm_df(folder: Path, infix: str) -> pd.DataFrame:
    df = lstm_mod.assemble_lstm(folder, infix)
    df = df[df['Horizon'] == f'{H_MIN}min']
    return _std_assembled(df)


def load_timellm_df(folder: Path) -> pd.DataFrame:
    df = timellm.assemble_timellm(folder)
    df = df[df['Horizon'] == f'{H_MIN}min']
    return _std_assembled(df)


def load_calf_df(folder: Path) -> pd.DataFrame:
    df = calf_mod.assemble_calf(folder)
    df = df[df['Horizon'] == f'{H_MIN}min']
    return _std_assembled(df)


# ─── Per-participant zone-% aggregation ───────────────────────────────────────
def per_patient_zone_means(zone_array: np.ndarray,
                           patient_array: np.ndarray,
                           zone_keys) -> dict:
    """For each zone in `zone_keys`, return mean across participants of the
    per-participant percentage of points landing in that zone.
    """
    df = pd.DataFrame({'zone': zone_array, 'patient': patient_array})
    counts = (df.groupby('patient')['zone']
                .value_counts()
                .unstack(fill_value=0))
    totals = counts.sum(axis=1).replace(0, np.nan)
    pcts = counts.div(totals, axis=0) * 100
    out = {}
    for z in zone_keys:
        out[z] = float(pcts[z].mean()) if z in pcts.columns else 0.0
    return out


# ─── Title sourcing ───────────────────────────────────────────────────────────
def _horizon_to_int(value) -> int | None:
    s = str(value).strip().lower().replace('min', '')
    try:
        return int(s)
    except ValueError:
        return None


def get_overall_total_row(csv_path: Path) -> pd.Series:
    t = pd.read_csv(csv_path)
    h_int = t['Horizon'].apply(_horizon_to_int)
    mask = (t['Dataset'] == TABLE_DATASET) & (h_int == H_MIN)
    sub = t[mask]
    if sub.empty:
        raise ValueError(
            f'No row with Dataset="{TABLE_DATASET}" & Horizon={H_MIN}min in {csv_path}'
        )
    return sub.iloc[0]


# ─── Legend overrides ─────────────────────────────────────────────────────────
def _replace_legend(ax, new_labels: list[str], **legend_kw) -> None:
    handles, _ = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(handles, new_labels, **legend_kw)


def update_ceg_legend(ax, ceg_pcts: dict) -> None:
    handles, labels = ax.get_legend_handles_labels()
    new = []
    for lbl in labels:
        # 'Zone A (75.3%)' -> 'A'
        zone = lbl.split()[1]
        new.append(f'Zone {zone} ({ceg_pcts.get(zone, 0):.1f}%)')
    if handles:
        ax.legend(handles, new, loc='upper left', fontsize=9, markerscale=4)


def update_seg_legend(ax, seg_pcts: dict) -> None:
    name_to_idx = {n: i for i, n in enumerate(_ZONE_LABELS)}
    handles, labels = ax.get_legend_handles_labels()
    new = []
    for lbl in labels:
        # 'None (75.3%)' -> 'None'
        name = lbl.split('(')[0].strip()
        idx = name_to_idx[name]
        new.append(f'{name} ({seg_pcts.get(idx, 0):.1f}%)')
    if handles:
        ax.legend(handles, new, loc='upper left', fontsize=8, markerscale=4)


# ─── Job table ────────────────────────────────────────────────────────────────
# (label, pretty, raw_subfolder, summary_csv_basename, df_loader)
JOBS = [
    ('chronos_fewshot',   'Chronos few-shot',   'chronos_fewshot_raw',
     'chronos_fewshot_zone_a_summary.csv',           load_chronos_df),
    ('chronos_fullshot',  'Chronos full-shot',  'chronos_fullshot_raw',
     'chronos_fullshot_zone_a_summary.csv',          load_chronos_df),
    ('timer_fewshot',     'Timer few-shot',     'timer_fewshot_raw',
     'timer_fewshot_raw_zone_a_summary.csv',         load_canonical_df),
    ('timer_fullshot',    'Timer full-shot',    'timer_fullshot_raw',
     'timer_fullshot_raw_zone_a_summary.csv',        load_canonical_df),
    ('timesfm_fewshot',   'TimesFM few-shot',   'timesfm_fewshot_raw_quantile',
     'timesfm_fewshot_raw_quantile_zone_a_summary.csv',  load_canonical_df),
    ('timesfm_fullshot',  'TimesFM full-shot',  'timesfm_fullshot_raw_for_plot',
     'timesfm_fullshot_raw_quantile_zone_a_summary.csv', load_canonical_df),
    ('uni2ts_fewshot',    'Uni2TS few-shot',    'uni2ts_fewshot_raw_quantile',
     'uni2ts_fewshot_raw_quantile_zone_a_summary.csv',   load_canonical_df),
    ('uni2ts_fullshot',   'Uni2TS full-shot',   'uni2ts_fullshot_raw_quantile',
     'uni2ts_fullshot_raw_quantile_zone_a_summary.csv',  load_canonical_df),
    ('lstm_fewshot',      'LSTM few-shot',      'lstm_fewshot_raw',
     'lstm_fewshot_zone_a_summary.csv',  lambda f: load_lstm_df(f, 'few')),
    ('lstm_fullshot',     'LSTM full-shot',     'lstm_fullshot_raw',
     'lstm_fullshot_zone_a_summary.csv', lambda f: load_lstm_df(f, 'full')),
    ('timellm_fewshot',   'Time-LLM few-shot',  'timellm_fewshot_raw',
     'timellm_fewshot_zone_a_summary.csv',  load_timellm_df),
    ('timellm_fullshot',  'Time-LLM full-shot', 'timellm_fullshot_raw',
     'timellm_fullshot_zone_a_summary.csv', load_timellm_df),
    ('calf_fewshot',      'CALF few-shot',      'calf_fewshot_raw',
     'calf_fewshot_zone_a_summary.csv',     load_calf_df),
    ('calf_fullshot',     'CALF full-shot',     'calf_fullshot_raw',
     'calf_fullshot_zone_a_summary.csv',    load_calf_df),
    ('gpformer_fewshot',  'GPFormer few-shot',  'gpformer_fewshot_raw',
     'gpformer_fewshot_zone_a_summary.csv',  load_gpformer_df),
    ('gpformer_fullshot', 'GPFormer full-shot', 'gpformer_fullshot_raw',
     'gpformer_fullshot_zone_a_summary.csv', load_gpformer_df),
]


# ─── Main plot routine ────────────────────────────────────────────────────────
def plot_one(label, pretty, raw_folder, csv_path, df_loader):
    if not raw_folder.is_dir():
        print(f'[skip] {pretty}: {raw_folder} not found', flush=True)
        return
    if not csv_path.exists():
        print(f'[skip] {pretty}: {csv_path} not found', flush=True)
        return

    df = df_loader(raw_folder)
    if df.empty:
        print(f'[skip] {pretty}: no rows at horizon={H_MIN}min', flush=True)
        return

    ref      = df['ref'].to_numpy(float)
    pred     = df['pred'].to_numpy(float)
    patient  = df['patient'].to_numpy()

    # Per-point zone classifications, used both for the per-patient legend
    # means and to verify nothing is unexpected.
    ceg_zones = np.array([classify_ceg_zone(r, p) for r, p in zip(ref, pred)])
    seg_risk  = seg_risk_scores(ref, pred)
    seg_zones = np.clip(np.digitize(seg_risk, _ZONE_BINS[1:]), 0, 4)

    ceg_pcts = per_patient_zone_means(ceg_zones, patient, list('ABCDE'))
    seg_pcts = per_patient_zone_means(seg_zones, patient, list(range(5)))

    n_unique_patients = pd.Series(patient).nunique()

    # Title numbers from the table.
    row         = get_overall_total_row(csv_path)
    n_table     = int(row['N_Participants'])
    ceg_a_mean  = float(row['CEG Zone A mean (%)'])
    ceg_a_std   = float(row['CEG Zone A std (%)'])
    seg_a_mean  = float(row['SEG Zone A mean (%)'])
    seg_a_std   = float(row['SEG Zone A std (%)'])
    seg_r_mean  = float(row['SEG |Risk| mean'])
    seg_r_std   = float(row['SEG |Risk| std'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    base = (f'{pretty} — Overall (Total) '
            f'(horizon={H_MIN}min, N={n_table} participants)')

    plot_ceg(ref, pred, title=base, ax=axes[0])
    plot_seg(ref, pred, title=base, ax=axes[1])

    update_ceg_legend(axes[0], ceg_pcts)
    update_seg_legend(axes[1], seg_pcts)

    axes[0].set_title(
        f'{base}\nCEG Zone A = {ceg_a_mean:.2f} ± {ceg_a_std:.2f}%',
        fontsize=13,
    )
    axes[1].set_title(
        f'{base}\nMean |Risk| = {seg_r_mean:.4f} ± {seg_r_std:.4f}  |  '
        f'No-risk zone = {seg_a_mean:.2f} ± {seg_a_std:.2f}%',
        fontsize=13,
    )
    fig.tight_layout()

    out_path = FIG_DIR / f'CEG_SEG_{label}_{H_MIN}min.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(
        f'Saved {out_path}  '
        f'(scatter pts={ref.size:,}, raw participants={n_unique_patients}, '
        f'table N={n_table}, CEG-A={ceg_pcts["A"]:.2f}%, '
        f'SEG-A={seg_pcts[0]:.2f}%)',
        flush=True,
    )


def main():
    for label, pretty, sub, csv_name, loader in JOBS:
        raw_folder = RESULTS_DIR / sub
        csv_path   = RESULTS_DIR / csv_name
        try:
            plot_one(label, pretty, raw_folder, csv_path, loader)
        except Exception as exc:
            print(f'[error] {pretty}: {exc}', flush=True)


if __name__ == '__main__':
    main()
