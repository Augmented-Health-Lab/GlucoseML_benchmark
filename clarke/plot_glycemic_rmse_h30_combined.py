"""
clarke/plot_glycemic_rmse_h30_combined.py

Render the zero-shot / few-shot / full-shot RMSE figures as a single 1x3
figure that fits on one paper page. Panels share a y-axis (only the
leftmost shows the y-label/ticks) and a single category-grouped legend
sits above all three. The zero-shot panel uses tighter zone spacing
(only 4 methods to display) and gets a narrower width-ratio so the
overall figure stays slim.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO    = Path(__file__).resolve().parent.parent
SRC_DIR = REPO / 'results'
OUT     = REPO / 'results' / 'figures' / 'glycemic_rmse_h30_combined.png'

# Strata, left -> right. (CSV row key, x-axis tick label).
STRATA: list[tuple[str, str]] = [
    ('Hypo (<70)',        'Hypo'),
    ('In-range [70,180]', 'In-range'),
    ('Hyper (>180)',      'Hyper'),
]
ZONE_BG = ['#F8D7DA', '#D7EFD2', '#FFF4B0']

METHOD_COLORS = {
    'LSTM':      '#2196F3',
    'GPFormer':  '#4CAF50',
    'Chronos':   '#FF9800',
    'Moirai':    '#9C27B0',  # CSV column "Uni2TS"
    'Timer':     '#F44336',
    'TimesFM':   '#E91E63',
    'Time-LLM':  '#00BCD4',
    'CALF':      '#795548',
}
METHOD_MARKERS = {
    'LSTM':      'o',
    'GPFormer':  's',
    'Chronos':   '^',
    'Moirai':    'D',
    'Timer':     'v',
    'TimesFM':   'X',
    'Time-LLM':  'P',
    'CALF':      '*',
}
DISPLAY_TO_CSV = {'Moirai': 'Uni2TS'}

# Per-method marker-size overrides. Square ('s') and diamond ('D') glyphs
# read visually larger than circles/triangles at the same point size, so
# shrink GPFormer / Moirai by 1pt to match the others.
_MS_OVERRIDES = {'GPFormer': 9, 'Moirai': 9}


def _ms(method: str) -> int:
    if METHOD_MARKERS[method] == '*':
        return 13
    return _MS_OVERRIDES.get(method, 10)

CATEGORIES_FULL = [
    ('Supervised DL',     ['LSTM', 'GPFormer']),
    ('Pre-trained TSFMs', ['Chronos', 'Moirai', 'Timer', 'TimesFM']),
    ('TS LLM-based',      ['Time-LLM', 'CALF']),
]
CATEGORIES_ZERO = [
    ('Pre-trained TSFMs', ['Chronos', 'Moirai', 'Timer', 'TimesFM']),
]

# Spacing within / between method groups, in zone-x units.
INTRA_STEP = 0.085
INTER_GAP  = 0.18

# (panel title, csv path, categories drawn in panel, zone width).
# Zone width also drives the panel's gridspec width-ratio so each strata
# zone spans roughly the same physical width across the three panels.
PANELS = [
    ('Zero-shot',  SRC_DIR / 'glycemic_rmse_h30_zeroshot.csv', CATEGORIES_ZERO, 0.7),
    ('Few-shot',   SRC_DIR / 'glycemic_rmse_h30_fewshot.csv',  CATEGORIES_FULL, 1.0),
    ('Full-shot',  SRC_DIR / 'glycemic_rmse_h30_fullshot.csv', CATEGORIES_FULL, 1.0),
]

CELL_RE = re.compile(r'([\d.]+)\s*\(([\d.]+)\)')


def parse_cell(s: str) -> tuple[float, float]:
    m = CELL_RE.match(s.strip())
    if not m:
        raise ValueError(f'cannot parse {s!r}')
    return float(m.group(1)), float(m.group(2))


def _compute_offsets(categories) -> list[float]:
    sizes  = [len(ms) for _, ms in categories]
    widths = [(n - 1) * INTRA_STEP for n in sizes]
    total  = sum(widths) + INTER_GAP * (len(sizes) - 1)
    cursor = -total / 2
    pos: list[float] = []
    for k, n in enumerate(sizes):
        for i in range(n):
            pos.append(cursor + i * INTRA_STEP)
        cursor += widths[k] + INTER_GAP
    return pos


def _plot_panel(ax, title, csv_path, categories, zone_step, show_ylabel) -> None:
    df = pd.read_csv(csv_path).set_index('Glycemic stratum')
    methods = [m for _, ms in categories for m in ms]

    csv_col = {m: DISPLAY_TO_CSV.get(m, m) for m in methods}
    missing = [csv_col[m] for m in methods if csv_col[m] not in df.columns]
    if missing:
        raise KeyError(f'methods missing from {csv_path.name}: {missing}')

    zone_half    = zone_step / 2
    zone_centers = [i * zone_step for i in range(len(STRATA))]

    for i, color in enumerate(ZONE_BG):
        c = zone_centers[i]
        ax.axvspan(c - zone_half, c + zone_half, color=color, alpha=0.55, zorder=0)
        if i < len(zone_centers) - 1:
            ax.axvline(c + zone_half, color='white', lw=1.0, zorder=1)

    offsets = _compute_offsets(categories)
    for j, method in enumerate(methods):
        means, stds, xs = [], [], []
        for i, (stratum_key, _) in enumerate(STRATA):
            mean, std = parse_cell(df.loc[stratum_key, csv_col[method]])
            means.append(mean); stds.append(std)
            xs.append(zone_centers[i] + offsets[j])
        marker = METHOD_MARKERS[method]
        ax.errorbar(
            xs, means, yerr=stds,
            fmt=marker, color=METHOD_COLORS[method],
            ecolor=METHOD_COLORS[method],
            capsize=2.5, capthick=0.9, elinewidth=1.0,
            ms=_ms(method), mec='black', mew=0.5, zorder=3,
        )

    ax.set_xticks(zone_centers)
    ax.set_xticklabels([label for _, label in STRATA], fontsize=11)
    ax.set_xlim(zone_centers[0] - zone_half, zone_centers[-1] + zone_half)
    ax.set_title(title, fontsize=13, pad=4)
    if show_ylabel:
        ax.set_ylabel('30-min RMSE (mg/dL)', fontsize=12)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(axis='y', color='white', lw=0.8, zorder=1)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    width_ratios = [p[3] for p in PANELS]
    fig, axes = plt.subplots(
        1, 3, figsize=(11.0, 4.0),
        sharey=True,
        gridspec_kw={'width_ratios': width_ratios, 'wspace': 0.06},
    )
    # Reserve top strip for the shared legend (figure-fraction coords).
    fig.subplots_adjust(top=0.78, bottom=0.13, left=0.07, right=0.99)
    for k, (title, csv_path, categories, zone_step) in enumerate(PANELS):
        _plot_panel(axes[k], title, csv_path, categories, zone_step,
                    show_ylabel=(k == 0))

    # Single legend frame with inline category headers, so the three
    # method groups are visually separated but share one box.
    handles: list[mlines.Line2D] = []
    header_indices: list[int] = []
    for cat_name, cat_methods in CATEGORIES_FULL:
        header_indices.append(len(handles))
        handles.append(mlines.Line2D(
            [], [], color='none', marker='None', linestyle='None',
            label=cat_name + ':',
        ))
        for m in cat_methods:
            handles.append(mlines.Line2D(
                [], [], color=METHOD_COLORS[m], marker=METHOD_MARKERS[m],
                ms=_ms(m), mec='black', mew=0.5,
                linestyle='None', label=m,
            ))

    leg = fig.legend(
        handles=handles,
        loc='lower center', bbox_to_anchor=(0.5, 0.83),
        ncol=len(handles), frameon=True,
        fontsize=10, columnspacing=0.7,
        handletextpad=0.25, handlelength=0.8,
        borderpad=0.45,
    )
    # Bold the category-header entries to set them apart inside the box.
    texts = leg.get_texts()
    for i in header_indices:
        texts[i].set_fontweight('bold')
    frame = leg.get_frame()
    frame.set_linestyle('--')
    frame.set_edgecolor('#8a8a8a')
    frame.set_linewidth(0.8)
    frame.set_facecolor('white')

    fig.savefig(OUT, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT}')


if __name__ == '__main__':
    main()
