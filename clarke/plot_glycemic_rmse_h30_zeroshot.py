"""
clarke/plot_glycemic_rmse_h30_zeroshot.py

Render results/glycemic_rmse_h30_zeroshot.csv as a grouped point-plus-errorbar
figure: x = three glycemic strata (Hypo / In-range / Hyper) with low-saturation
yellow / green / pink backgrounds; y = RMSE (mg/dL); 4 pre-trained TSFM methods
plotted at small horizontal offsets within each zone, mean as a dot, std as the
error bar. Legend sits above the axes so it can't occlude any data.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SRC  = REPO / 'results' / 'glycemic_rmse_h30_zeroshot.csv'
OUT  = REPO / 'results' / 'figures' / 'glycemic_rmse_h30_zeroshot.png'

# Strata, left -> right. (CSV row key, x-axis tick label).
STRATA: list[tuple[str, str]] = [
    ('Hypo (<70)',        'Hypo'),
    ('In-range [70,180]', 'In-range'),
    ('Hyper (>180)',      'Hyper'),
]

# Place zone centres closer than 1.0 apart so the three strata sit tight
# together (avoids the wasted empty space the default integer spacing
# leaves with only 4 markers per zone).
ZONE_STEP   = 0.7
ZONE_HALF   = ZONE_STEP / 2
ZONE_CENTERS = [i * ZONE_STEP for i in range(3)]
ZONE_BG = ['#F8D7DA',# pastel yellow  (hypo)
           '#D7EFD2',   # pastel green   (in-range)
           '#FFF4B0']   # pastel pink    (hyper)

# Bright, mutually-distinct palette (Material 500-ish) that reads well
# against any of the 3 pastel zone backgrounds. Kept identical to the
# fewshot/fullshot scripts so colors stay consistent across figures.
METHOD_COLORS = {
    'Chronos':   '#FF9800',  # orange
    'Moirai':    '#9C27B0',  # purple   (CSV column "Uni2TS")
    'Timer':     '#F44336',  # red
    'TimesFM':   '#E91E63',  # pink
}
# Distinct marker shapes so methods are still identifiable in greyscale.
METHOD_MARKERS = {
    'Chronos':   '^',
    'Moirai':    'D',
    'Timer':     'v',
    'TimesFM':   'X',
}
# Display name -> CSV column name (for renames only).
DISPLAY_TO_CSV = {'Moirai': 'Uni2TS'}

# Method aggregation. Order within each category is the plot's left-to-right order.
CATEGORIES = [
    ('Pre-trained TSFMs', ['Chronos', 'Moirai', 'Timer', 'TimesFM']),
]

# Spacing within / between groups, in zone-x units (a zone spans 1.0).
INTRA_STEP = 0.085  # narrow within a category
INTER_GAP  = 0.18   # wider gap between categories (unused with one category)

CELL_RE = re.compile(r'([\d.]+)\s*\(([\d.]+)\)')


def parse_cell(s: str) -> tuple[float, float]:
    m = CELL_RE.match(s.strip())
    if not m:
        raise ValueError(f'cannot parse {s!r}')
    return float(m.group(1)), float(m.group(2))


def _compute_offsets() -> list[float]:
    """Build x-offsets for the methods so each category sits in its own
    sub-cluster, with INTRA_STEP between adjacent methods inside a group
    and INTER_GAP between groups. Centred on x=0."""
    sizes = [len(ms) for _, ms in CATEGORIES]
    widths = [(n - 1) * INTRA_STEP for n in sizes]
    total  = sum(widths) + INTER_GAP * (len(sizes) - 1)
    cursor = -total / 2
    pos: list[float] = []
    for k, n in enumerate(sizes):
        for i in range(n):
            pos.append(cursor + i * INTRA_STEP)
        cursor += widths[k] + INTER_GAP
    return pos


def main() -> None:
    df = pd.read_csv(SRC).set_index('Glycemic stratum')
    methods = [m for _, ms in CATEGORIES for m in ms]

    # CSV column resolution (display name -> column name).
    csv_col = {m: DISPLAY_TO_CSV.get(m, m) for m in methods}
    missing = [csv_col[m] for m in methods if csv_col[m] not in df.columns]
    if missing:
        raise KeyError(f'methods missing from CSV: {missing}')
    if any(m not in METHOD_COLORS for m in methods):
        bad = [m for m in methods if m not in METHOD_COLORS]
        raise KeyError(f'no color assigned for: {bad}')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5.0))   # narrower canvas — only 4 methods to plot

    for i, color in enumerate(ZONE_BG):
        c = ZONE_CENTERS[i]
        ax.axvspan(c - ZONE_HALF, c + ZONE_HALF, color=color, alpha=0.55, zorder=0)
        if i < len(ZONE_CENTERS) - 1:
            ax.axvline(c + ZONE_HALF, color='white', lw=1.0, zorder=1)

    # Tight intra-group offsets, wider inter-category gaps.
    offsets = _compute_offsets()
    assert len(offsets) == len(methods)

    for j, method in enumerate(methods):
        means, stds, xs = [], [], []
        for i, (stratum_key, _) in enumerate(STRATA):
            mean, std = parse_cell(df.loc[stratum_key, csv_col[method]])
            means.append(mean); stds.append(std)
            xs.append(ZONE_CENTERS[i] + offsets[j])
        marker = METHOD_MARKERS[method]
        ms = 15 if marker == '*' else 12
        ax.errorbar(
            xs, means, yerr=stds,
            fmt=marker, color=METHOD_COLORS[method],
            ecolor=METHOD_COLORS[method],
            capsize=3, capthick=1.0, elinewidth=1.2,
            ms=ms, mec='black', mew=0.6,
            label=method, zorder=3,
        )

    ax.set_xticks(ZONE_CENTERS)
    ax.set_xticklabels([label for _, label in STRATA], fontsize=13)
    ax.set_ylabel('30-min RMSE (mg/dL)', fontsize=13)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_xlim(ZONE_CENTERS[0] - ZONE_HALF, ZONE_CENTERS[-1] + ZONE_HALF)
    ax.grid(axis='y', color='white', lw=0.8, zorder=1)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    # Single category-grouped legend centred above the axes.
    cat_anchors = [0.5]
    legend_objs = []
    for (cat_name, cat_methods), x_anchor in zip(CATEGORIES, cat_anchors):
        handles = [
            mlines.Line2D(
                [], [],
                color=METHOD_COLORS[m], marker=METHOD_MARKERS[m],
                ms=(15 if METHOD_MARKERS[m] == '*' else 12),
                mec='black', mew=0.6, linestyle='None', label=m,
            )
            for m in cat_methods
        ]
        leg = ax.legend(
            handles=handles, title=cat_name,
            loc='lower center', bbox_to_anchor=(x_anchor, 1.02),
            ncol=len(cat_methods), frameon=True,
            fontsize=11, title_fontsize=11.5,
            columnspacing=0.6, handletextpad=0.25, handlelength=0.8,
            borderpad=0.4,
        )
        # Dashed border around the category for visual consistency with
        # the fewshot/fullshot figures.
        frame = leg.get_frame()
        frame.set_linestyle('--')
        frame.set_edgecolor('#8a8a8a')
        frame.set_linewidth(0.8)
        frame.set_facecolor('white')
        legend_objs.append(leg)
    # ax.legend() replaces previous; add_artist preserves the older ones.
    for leg in legend_objs[:-1]:
        ax.add_artist(leg)

    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT}')


if __name__ == '__main__':
    main()
