"""Count the forecast windows each protocol actually processes ("workload").

Runtime = throughput x workload.  This script computes the workload half on
CPU, with no model loaded, so it is cheap and can be re-run any time.

Two window-construction conventions exist in this repo and both are counted:

`multiwindow`  GPFormer / TimesFM / Timer / Moirai / Time-LLM / CALF / Chronos
               A window starting at t is valid iff the next ctx+hor-1 sampling
               intervals are all exactly 5 minutes.  Starts are taken every
               `stride` samples over the whole series.

`lstm`         2019Martinsson_et_al_LSTM
               The series is first split wherever the gap exceeds 6 minutes;
               segments shorter than ctx+hor are dropped; windows are then
               strided inside each surviving segment.  The LSTM additionally
               splits each subject's windows 90/10 into train/valid.

Usage
-----
    python bench_efficiency/count_windows.py \
        --data-root hf_cache \
        --out bench_efficiency/results/workload.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

STEP = np.timedelta64(5, "m")
LSTM_GAP = np.timedelta64(6, "m")

TIME_COLS = ["timestamp", "time", "datetime", "date_time", "date"]
VALUE_COLS = ["bgvalue", "glucose", "glucose_value", "sensor_glucose", "value"]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def _pick(columns, candidates) -> Optional[str]:
    lowered = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def _parse_timestamps(raw: pd.Series) -> pd.Series:
    """Mirror glucofm_data.parse_timestamp_series: unix epoch or datetime str."""
    if pd.api.types.is_numeric_dtype(raw):
        values = pd.to_numeric(raw, errors="coerce").dropna()
        if values.empty:
            return pd.to_datetime(raw, errors="coerce")
        scale = float(values.abs().max())
        unit = "s" if scale < 1e11 else ("ms" if scale < 1e14 else "us")
        return pd.to_datetime(raw, unit=unit, errors="coerce")
    return pd.to_datetime(raw, errors="coerce")


def load_series(csv_path: Path) -> Optional[np.ndarray]:
    """Return the subject's timestamps as datetime64[ns], or None if unusable."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001 - a bad CSV must not abort the sweep
        print(f"  ! skip {csv_path.name}: {exc}", file=sys.stderr)
        return None

    time_col = _pick(df.columns, TIME_COLS)
    value_col = _pick(df.columns, VALUE_COLS) or (df.columns[1] if len(df.columns) > 1 else None)
    if time_col is None or value_col is None:
        return None

    ts = _parse_timestamps(df[time_col])
    values = pd.to_numeric(df[value_col], errors="coerce")
    keep = ts.notna() & values.notna()
    ts = ts[keep]
    if ts.empty:
        return None
    return ts.to_numpy(dtype="datetime64[ns]")


# --------------------------------------------------------------------------
# window counting
# --------------------------------------------------------------------------
def contiguous_runs(timestamps: np.ndarray) -> np.ndarray:
    """Vectorised equivalent of the repo's `consecutive_gap_counts`.

    `runs[i]` = how many consecutive exact 5-minute steps follow index i, so a
    window of length L starting at i is valid iff `runs[i] >= L - 1`.
    """
    n = len(timestamps)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if n == 1:
        return np.zeros(1, dtype=np.int64)

    is_step = (timestamps[1:] - timestamps[:-1]) == STEP
    # Boundaries are the positions where the chain breaks, plus the series end.
    bounds = np.append(np.flatnonzero(~is_step), n - 1)
    idx = np.arange(n)
    return bounds[np.searchsorted(bounds, idx, side="left")] - idx


def count_multiwindow(timestamps: np.ndarray, ctx: int, hor: int, stride: int) -> int:
    total_len = ctx + hor
    max_start = len(timestamps) - total_len
    if max_start < 0:
        return 0
    runs = contiguous_runs(timestamps)
    starts = np.arange(0, max_start + 1, stride)
    return int(np.count_nonzero(runs[starts] >= total_len - 1))


def count_lstm(timestamps: np.ndarray, ctx: int, hor: int, stride: int) -> int:
    total_len = ctx + hor
    if len(timestamps) < total_len:
        return 0
    diffs = timestamps[1:] - timestamps[:-1]
    breaks = np.flatnonzero(diffs > LSTM_GAP) + 1
    total = 0
    for segment in np.split(np.arange(len(timestamps)), breaks):
        usable = len(segment) - total_len + 1
        if usable > 0:
            total += int(np.ceil(usable / stride))
    return total


# --------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------
def iter_subject_csvs(split_root: Path) -> List[tuple[str, Path]]:
    """Yield (dataset_name, csv_path); the split-level `mixed/` copy is skipped."""
    found: List[tuple[str, Path]] = []
    for path in sorted(split_root.rglob("*.csv")):
        if "mixed" in path.relative_to(split_root).parts:
            continue
        dataset = path.parent.name
        found.append((dataset, path))
    if not found:  # only a flat mixed/ folder exists
        for path in sorted((split_root / "mixed").glob("*.csv")):
            found.append((path.stem.split("__")[0], path))
    return found


def sweep_split(
    split_root: Path,
    *,
    ctx: int,
    hor: int,
    strides: List[int],
    convention: str,
) -> Dict[str, object]:
    counter = count_multiwindow if convention == "multiwindow" else count_lstm
    subjects = iter_subject_csvs(split_root)
    print(f"[{split_root}] {len(subjects)} subject CSVs ({convention})")

    per_dataset: Dict[str, Dict[str, int]] = {}
    totals = {str(s): 0 for s in strides}
    n_usable = 0

    for i, (dataset, path) in enumerate(subjects, 1):
        if i % 100 == 0:
            print(f"  ... {i}/{len(subjects)}")
        timestamps = load_series(path)
        if timestamps is None:
            continue
        n_usable += 1
        bucket = per_dataset.setdefault(dataset, {"subjects": 0})
        bucket["subjects"] += 1
        for stride in strides:
            n = counter(timestamps, ctx, hor, stride)
            bucket[f"stride_{stride}"] = bucket.get(f"stride_{stride}", 0) + n
            totals[str(stride)] += n

    return {
        "subjects_found": len(subjects),
        "subjects_usable": n_usable,
        "windows_by_stride": totals,
        "per_dataset": per_dataset,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "hf_cache")
    parser.add_argument("--context-steps", type=int, default=144, help="12 h of 5-min samples.")
    parser.add_argument(
        "--horizon-steps",
        type=int,
        default=6,
        help="30 min. Use 18 to also size the pred_len=18 methods (Chronos/Time-LLM/CALF).",
    )
    parser.add_argument(
        "--strides",
        type=int,
        nargs="+",
        default=[1, 10, 12, 240],
        help="1=eval, 12=full-shot train, 240=few-shot train, 10=legacy TSFM default.",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "workload.json")
    args = parser.parse_args(argv)

    if not args.data_root.exists():
        parser.error(
            f"{args.data_root} not found. Build it first:\n"
            f"    python prepare_dataset.py --output_dir hf_cache --create_mixed"
        )

    result: Dict[str, object] = {
        "context_steps": args.context_steps,
        "horizon_steps": args.horizon_steps,
        "data_root": str(args.data_root),
        "conventions": {},
    }

    for convention in ("multiwindow", "lstm"):
        per_split = {}
        for split in args.splits:
            split_root = args.data_root / split
            if not split_root.exists():
                print(f"! missing split: {split_root}", file=sys.stderr)
                continue
            per_split[split] = sweep_split(
                split_root,
                ctx=args.context_steps,
                hor=args.horizon_steps,
                strides=args.strides,
                convention=convention,
            )
        result["conventions"][convention] = per_split

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {args.out}")

    for convention, splits in result["conventions"].items():
        for split, payload in splits.items():
            print(f"  {convention:12s} {split:5s} {payload['windows_by_stride']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
