"""Shared helpers for the per-method glycemic_rmse_<method>.py scripts.

Intentionally tiny — only what every method needs (strata definitions,
participant->dataset JSON map, RMSE-per-participant aggregation, table
formatting). Each per-method script still owns its data loading.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

H_MIN = 30
STRATA = [
    ('Hyper (>180)',      lambda gt: gt > 180),
    ('In-range [70,180]', lambda gt: (gt >= 70) & (gt <= 180)),
    ('Hypo (<70)',        lambda gt: gt < 70),
]


def patient_to_dataset_map(participants_json: Path) -> dict[str, str]:
    """Reverse-map subject_id -> dataset_name from test_dataset/dataset_participants.json."""
    with open(participants_json) as f:
        ds_to_ids = json.load(f)
    rev: dict[str, str] = {}
    for ds, ids in ds_to_ids.items():
        for pid in ids:
            rev[str(pid)] = ds
    return rev


def per_participant_rmse(df: pd.DataFrame) -> pd.DataFrame:
    """For df with columns Patient, Dataset, GroundTruth, Prediction, return a
    DataFrame indexed by (Dataset, Patient) with one column per stratum
    holding that participant's RMSE inside that stratum (NaN if the
    participant has no points in the stratum)."""
    work = pd.DataFrame({
        'Dataset': df['Dataset'].astype(str),
        'Patient': df['Patient'].astype(str),
        'gt':      pd.to_numeric(df['GroundTruth'], errors='coerce'),
        'pred':    pd.to_numeric(df['Prediction'],  errors='coerce'),
    }).dropna(subset=['gt', 'pred'])
    work['err2'] = (work['gt'] - work['pred']) ** 2

    pieces = []
    for label, fn in STRATA:
        mask = fn(work['gt'].to_numpy())
        sub  = work[mask]
        if sub.empty:
            continue
        g = (sub.groupby(['Dataset', 'Patient'])['err2']
                .mean()
                .pow(0.5)
                .rename(label))
        pieces.append(g)
    return pd.concat(pieces, axis=1) if pieces else pd.DataFrame()


def format_mean_std(values: np.ndarray) -> str:
    if values.size == 0:
        return ''
    if values.size == 1:
        return f'{values[0]:.2f} (0.00)'
    return f'{values.mean():.2f} ({values.std(ddof=1):.2f})'


def build_table(per_protocol: Mapping[str, pd.DataFrame],
                protocol_order: Sequence[str]) -> pd.DataFrame:
    """rows = strata, cols = protocols (in order), cells = 'mean (std)'."""
    rows: dict[str, dict[str, str]] = {label: {} for label, _ in STRATA}
    for proto in protocol_order:
        ppdf = per_protocol.get(proto)
        for label, _ in STRATA:
            if ppdf is None or ppdf.empty or label not in ppdf.columns:
                rows[label][proto] = ''
                continue
            vals = ppdf[label].dropna().to_numpy(float)
            rows[label][proto] = format_mean_std(vals)
    return pd.DataFrame(rows).T[list(protocol_order)]
