"""
clarke/aggregate_glycemic_rmse_h30.py

Stitch per-method glycemic RMSE CSVs into three protocol-level tables —
one each for zero/few/full-shot. Run with --horizon {15,30,60,90}; the
script reads results/glycemic_rmse_h{H}_<method>.csv and writes
results/glycemic_rmse_h{H}_<protocol>.csv.

Rows: Hyper / In-range / Hypo. Columns: methods in fixed display order.
Cells: 'mean (std)' strings copied verbatim from the per-method CSVs;
empty if a method does not have that protocol.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RES  = REPO / 'results'
HORIZONS = (15, 30, 60, 90)

# Method display order per protocol — the methods that have that protocol.
PROTOCOL_METHODS: dict[str, list[str]] = {
    'zero-shot': ['Chronos', 'Uni2TS', 'Timer', 'TimesFM'],
    'few-shot':  ['LSTM', 'GPFormer', 'Chronos', 'Uni2TS', 'Timer',
                  'Time-LLM', 'TimesFM', 'CALF'],
    'full-shot': ['LSTM', 'GPFormer', 'Chronos', 'Uni2TS', 'Timer',
                  'Time-LLM', 'TimesFM', 'CALF'],
}

# Method label -> filename suffix used in glycemic_rmse_h{H}_<suffix>.csv
METHOD_SUFFIX = {
    'Chronos':  'chronos',
    'Uni2TS':   'uni2ts',
    'Timer':    'timer',
    'TimesFM':  'timesfm',
    'LSTM':     'lstm',
    'GPFormer': 'gpformer',
    'Time-LLM': 'timellm',
    'CALF':     'calf',
}

# Protocol label -> filename suffix used in glycemic_rmse_h{H}_<suffix>.csv
PROTOCOL_SUFFIX = {
    'zero-shot': 'zeroshot',
    'few-shot':  'fewshot',
    'full-shot': 'fullshot',
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', type=int, default=30, choices=HORIZONS)
    h = parser.parse_args().horizon

    # Load each per-method CSV once: rows=strata, cols=protocols, cells=str.
    per_method: dict[str, pd.DataFrame] = {}
    for method, suffix in METHOD_SUFFIX.items():
        path = RES / f'glycemic_rmse_h{h}_{suffix}.csv'
        if not path.exists():
            print(f'[skip] {method}: {path} missing')
            continue
        per_method[method] = (pd.read_csv(path, dtype=str)
                                .set_index('Glycemic stratum'))

    for protocol, methods in PROTOCOL_METHODS.items():
        any_present = next((m for m in methods if m in per_method), None)
        if any_present is None:
            print(f'[skip] {protocol}: no per-method CSVs available for h={h}')
            continue
        out = pd.DataFrame(index=per_method[any_present].index)
        for method in methods:
            df = per_method.get(method)
            if df is None or protocol not in df.columns:
                out[method] = ''
            else:
                out[method] = df[protocol].fillna('')
        out_path = RES / f'glycemic_rmse_h{h}_{PROTOCOL_SUFFIX[protocol]}.csv'
        out.to_csv(out_path, index_label='Glycemic stratum')
        print(f'\n=== {protocol} (h={h}min) ===')
        print(out.to_string())
        print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
