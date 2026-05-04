"""
clarke/plot_ceg_seg_horizon30_table_aligned.py

Re-render the horizon=30min CEG/SEG figures in results/ceg_seg_figures/ so the
title numbers come directly from each method's *_zone_a_summary.csv table
(Dataset = "Overall (Total)", Horizon = "30min") instead of being recomputed
from the pooled scatter.

Scatter density still comes from the same per-method raw folders used by
clarke/plot_ceg_seg_horizon30_overall.py — only the printed Mean |Risk|, SEG
no-risk zone %, and CEG Zone A % are read from the summary tables.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from clarke.ceg_seg import plot_ceg                          # noqa: E402
from clarke.ceg_seg_upgrade import plot_seg                  # noqa: E402

# Reuse the existing per-method scatter loaders.
plot_mod = importlib.import_module('clarke.plot_ceg_seg_horizon30_overall')
load_canonical = plot_mod.load_canonical
load_gpformer  = plot_mod.load_gpformer
load_chronos   = plot_mod.load_chronos
load_lstm      = plot_mod.load_lstm
load_timellm   = plot_mod.load_timellm
load_calf      = plot_mod.load_calf

FIG_DIR        = REPO / 'results' / 'ceg_seg_figures'
RESULTS_DIR    = REPO / 'results'
H_MIN          = 30
TABLE_DATASET  = 'Overall (Total)'
TABLE_HORIZON  = f'{H_MIN}min'

FIG_DIR.mkdir(parents=True, exist_ok=True)

# (label, pretty, raw_subfolder, summary_csv_basename, scatter_loader)
JOBS = [
    ('chronos_fewshot',   'Chronos few-shot',   'chronos_fewshot_raw',
     'chronos_fewshot_zone_a_summary.csv',           load_chronos),
    ('chronos_fullshot',  'Chronos full-shot',  'chronos_fullshot_raw',
     'chronos_fullshot_zone_a_summary.csv',          load_chronos),
    ('timer_fewshot',     'Timer few-shot',     'timer_fewshot_raw',
     'timer_fewshot_raw_zone_a_summary.csv',         load_canonical),
    ('timer_fullshot',    'Timer full-shot',    'timer_fullshot_raw',
     'timer_fullshot_raw_zone_a_summary.csv',        load_canonical),
    ('timesfm_fewshot',   'TimesFM few-shot',   'timesfm_fewshot_raw_quantile',
     'timesfm_fewshot_raw_quantile_zone_a_summary.csv',  load_canonical),
    ('timesfm_fullshot',  'TimesFM full-shot',  'timesfm_fullshot_raw_for_plot',
     'timesfm_fullshot_raw_quantile_zone_a_summary.csv', load_canonical),
    ('uni2ts_fewshot',    'Uni2TS few-shot',    'uni2ts_fewshot_raw_quantile',
     'uni2ts_fewshot_raw_quantile_zone_a_summary.csv',   load_canonical),
    ('uni2ts_fullshot',   'Uni2TS full-shot',   'uni2ts_fullshot_raw_quantile',
     'uni2ts_fullshot_raw_quantile_zone_a_summary.csv',  load_canonical),
    ('lstm_fewshot',      'LSTM few-shot',      'lstm_fewshot_raw',
     'lstm_fewshot_zone_a_summary.csv',  lambda f: load_lstm(f, 'few')),
    ('lstm_fullshot',     'LSTM full-shot',     'lstm_fullshot_raw',
     'lstm_fullshot_zone_a_summary.csv', lambda f: load_lstm(f, 'full')),
    ('timellm_fewshot',   'Time-LLM few-shot',  'timellm_fewshot_raw',
     'timellm_fewshot_zone_a_summary.csv',  load_timellm),
    ('timellm_fullshot',  'Time-LLM full-shot', 'timellm_fullshot_raw',
     'timellm_fullshot_zone_a_summary.csv', load_timellm),
    ('calf_fewshot',      'CALF few-shot',      'calf_fewshot_raw',
     'calf_fewshot_zone_a_summary.csv',     load_calf),
    ('calf_fullshot',     'CALF full-shot',     'calf_fullshot_raw',
     'calf_fullshot_zone_a_summary.csv',    load_calf),
    ('gpformer_fewshot',  'GPFormer few-shot',  'gpformer_fewshot_raw',
     'gpformer_fewshot_zone_a_summary.csv',  load_gpformer),
    ('gpformer_fullshot', 'GPFormer full-shot', 'gpformer_fullshot_raw',
     'gpformer_fullshot_zone_a_summary.csv', load_gpformer),
]


def _horizon_to_int(value) -> int | None:
    """Coerce a Horizon cell to an integer minutes value (handles '30', '30min', 30)."""
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


def plot_one(label, pretty, raw_folder, csv_path, scatter_loader):
    if not raw_folder.is_dir():
        print(f'[skip] {pretty}: {raw_folder} not found')
        return
    if not csv_path.exists():
        print(f'[skip] {pretty}: {csv_path} not found')
        return

    ref, pred = scatter_loader(raw_folder)
    if ref.size == 0:
        print(f'[skip] {pretty}: no scatter rows at horizon={H_MIN}min')
        return

    row         = get_overall_total_row(csv_path)
    n_part      = int(row['N_Participants'])
    ceg_a_mean  = float(row['CEG Zone A mean (%)'])
    ceg_a_std   = float(row['CEG Zone A std (%)'])
    seg_a_mean  = float(row['SEG Zone A mean (%)'])
    seg_a_std   = float(row['SEG Zone A std (%)'])
    seg_r_mean  = float(row['SEG |Risk| mean'])
    seg_r_std   = float(row['SEG |Risk| std'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    base = f'{pretty} — Overall (Total) (horizon={H_MIN}min, N={n_part} participants)'

    plot_ceg(ref, pred, title=base, ax=axes[0])
    plot_seg(ref, pred, title=base, ax=axes[1])

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
    print(f'Saved {out_path}  (scatter pts={ref.size:,}, table N={n_part}, '
          f'CEG-A={ceg_a_mean:.2f}%, SEG-A={seg_a_mean:.2f}%)')


def main():
    for label, pretty, sub, csv_name, loader in JOBS:
        raw_folder = RESULTS_DIR / sub
        csv_path   = RESULTS_DIR / csv_name
        try:
            plot_one(label, pretty, raw_folder, csv_path, loader)
        except Exception as exc:
            print(f'[error] {pretty}: {exc}')


if __name__ == '__main__':
    main()
