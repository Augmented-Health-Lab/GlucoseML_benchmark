"""
clarke/plot_ceg_seg_horizon30_overall.py

For every {method}_{few/full}shot_raw* folder under results/, pool ALL
(reference, prediction) pairs at horizon=30min across every dataset and
emit a 1x2 (CEG | SEG) figure to results/ceg_seg_figures/.

Reuses the per-method assemble_* helpers we already wrote for the Zone A
summaries so the data normalization stays identical across methods.
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

from clarke.ceg_seg import plot_ceg                          # noqa: E402
from clarke.ceg_seg_upgrade import plot_seg, seg_scores      # noqa: E402

# Existing per-method assemblers
chronos_ft = importlib.import_module('clarke.run_zone_a_tables_chronos_finetuned')
lstm_mod   = importlib.import_module('clarke.run_zone_a_tables_lstm')
timellm    = importlib.import_module('clarke.run_zone_a_tables_timellm')
calf_mod   = importlib.import_module('clarke.run_zone_a_tables_calf')

FIG_DIR = REPO / 'results' / 'ceg_seg_figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
H_MIN = 30


def _glob_concat(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob('*.csv'))
    if not files:
        raise FileNotFoundError(f'No CSVs in {folder}')
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True)


# --- per-method (ref, pred) loaders at horizon=30 ----------------------------
def load_canonical(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    """For folders that already use {dataset, participant_id, horizon_minutes,
    prediction, ground_truth} (timer / timesfm / uni2ts)."""
    df = _glob_concat(folder)
    df = df[df['horizon_minutes'] == H_MIN]
    return df['ground_truth'].to_numpy(float), df['prediction'].to_numpy(float)


def load_gpformer(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    """GPFormer stores all sub-steps; mirror metric_mode=final by keeping
    rows where horizon_offset_min == horizon_minutes."""
    df = _glob_concat(folder)
    df = df[(df['horizon_minutes'] == H_MIN) &
            (df['horizon_offset_min'] == H_MIN)]
    return df['ground_truth'].to_numpy(float), df['prediction'].to_numpy(float)


def load_chronos(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    df = chronos_ft.assemble_chronos(folder)
    df = df[df['Horizon'] == f'{H_MIN}min']
    return df['GroundTruth'].to_numpy(float), df['Prediction'].to_numpy(float)


def load_lstm(folder: Path, infix: str) -> tuple[np.ndarray, np.ndarray]:
    df = lstm_mod.assemble_lstm(folder, infix)
    df = df[df['Horizon'] == f'{H_MIN}min']
    return df['GroundTruth'].to_numpy(float), df['Prediction'].to_numpy(float)


def load_timellm(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    df = timellm.assemble_timellm(folder)
    df = df[df['Horizon'] == f'{H_MIN}min']
    return df['GroundTruth'].to_numpy(float), df['Prediction'].to_numpy(float)


def load_calf(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    df = calf_mod.assemble_calf(folder)
    df = df[df['Horizon'] == f'{H_MIN}min']
    return df['GroundTruth'].to_numpy(float), df['Prediction'].to_numpy(float)


# (label_for_filename, pretty_method, results_subfolder, loader)
JOBS = [
    ('chronos_fewshot',   'Chronos few-shot',   'chronos_fewshot_raw',   load_chronos),
    ('chronos_fullshot',  'Chronos full-shot',  'chronos_fullshot_raw',  load_chronos),
    ('timer_fewshot',     'Timer few-shot',     'timer_fewshot_raw',     load_canonical),
    ('timer_fullshot',    'Timer full-shot',    'timer_fullshot_raw',    load_canonical),
    ('timesfm_fewshot',   'TimesFM few-shot',   'timesfm_fewshot_raw_quantile',  load_canonical),
    ('timesfm_fullshot',  'TimesFM full-shot',  'timesfm_fullshot_raw_quantile', load_canonical),
    ('uni2ts_fewshot',    'Uni2TS few-shot',    'uni2ts_fewshot_raw_quantile',   load_canonical),
    ('uni2ts_fullshot',   'Uni2TS full-shot',   'uni2ts_fullshot_raw_quantile',  load_canonical),
    ('lstm_fewshot',      'LSTM few-shot',      'lstm_fewshot_raw',      lambda f: load_lstm(f, 'few')),
    ('lstm_fullshot',     'LSTM full-shot',     'lstm_fullshot_raw',     lambda f: load_lstm(f, 'full')),
    ('timellm_fewshot',   'Time-LLM few-shot',  'timellm_fewshot_raw',   load_timellm),
    ('timellm_fullshot',  'Time-LLM full-shot', 'timellm_fullshot_raw',  load_timellm),
    ('calf_fewshot',      'CALF few-shot',      'calf_fewshot_raw',      load_calf),
    ('calf_fullshot',     'CALF full-shot',     'calf_fullshot_raw',     load_calf),
    ('gpformer_fewshot',  'GPFormer few-shot',  'gpformer_fewshot_raw',  load_gpformer),
    ('gpformer_fullshot', 'GPFormer full-shot', 'gpformer_fullshot_raw', load_gpformer),
]


def plot_one(label: str, pretty: str, folder: Path, loader) -> Path | None:
    if not folder.is_dir():
        print(f'[skip] {pretty}: {folder} not found')
        return None
    ref, pred = loader(folder)
    if ref.size == 0:
        print(f'[skip] {pretty}: no rows at horizon={H_MIN}min')
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    title = f'{pretty} — Overall (horizon={H_MIN}min)'
    plot_ceg(ref, pred, title=title, ax=axes[0])
    plot_seg(ref, pred, title=title, ax=axes[1])

    # Override SEG subplot title: drop "Zone A", phrase as "No-risk zone".
    mean_risk, zone_a = seg_scores(ref, pred)
    axes[1].set_title(
        f'{title}\nMean |Risk| = {mean_risk:.4f}  |  '
        f'No-risk zone = {zone_a:.1f}%',
        fontsize=13,
    )
    fig.tight_layout()

    out_path = FIG_DIR / f'CEG_SEG_{label}_{H_MIN}min.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path} ({ref.size:,} points)')
    return out_path


def main() -> None:
    for label, pretty, sub, loader in JOBS:
        folder = REPO / 'results' / sub
        try:
            plot_one(label, pretty, folder, loader)
        except Exception as exc:
            print(f'[error] {pretty}: {exc}')


if __name__ == '__main__':
    main()
