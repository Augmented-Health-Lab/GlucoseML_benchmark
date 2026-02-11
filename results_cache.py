from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd


def _read_csv_header(path: Path) -> list[str]:
    """Return the header columns of a CSV (empty if file missing/empty)."""
    if not path.exists():
        return []
    # utf-8-sig handles potential BOMs gracefully.
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            return [str(c) for c in next(reader)]
        except StopIteration:
            return []


def _boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    s = str(value).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


_NORMALIZERS: dict[str, Callable[[object], object]] = {
    "participant_id": lambda v: str(v),
    "context_hours": lambda v: int(v),
    "horizon_minutes": lambda v: int(v),
    "stride_steps": lambda v: int(v),
    "metric_mode": lambda v: str(v),
    "latest_requires_truth": _boolish,
}


def load_completed_keys(
    csv_path: Path,
    *,
    key_cols: Sequence[str],
) -> set[tuple[object, ...]]:
    """
    Load the set of completed keys from an on-disk CSV.

    This is intentionally lightweight: it only reads the `key_cols` columns.
    If the file is missing or doesn't contain the required key columns, an
    empty set is returned (meaning "treat as uncached").
    """
    if not csv_path.exists():
        return set()

    header = _read_csv_header(csv_path)
    if not header:
        return set()

    header_set = set(header)
    if any(col not in header_set for col in key_cols):
        return set()

    df = pd.read_csv(csv_path, usecols=list(key_cols))
    if df.empty:
        return set()

    # Drop rows missing any part of the key.
    df = df.dropna(subset=list(key_cols))
    if df.empty:
        return set()

    # Normalize types for stable membership checks.
    for col in key_cols:
        normalizer = _NORMALIZERS.get(col)
        if normalizer is None:
            continue
        df[col] = df[col].map(normalizer)

    # itertuples() is faster than iterrows() and doesn't allocate Series per row.
    keys: set[tuple[object, ...]] = set()
    for row in df.itertuples(index=False, name=None):
        keys.add(tuple(row))
    return keys


def _apply_desired_order(df: pd.DataFrame, desired_order: Sequence[str] | None) -> pd.DataFrame:
    if df.empty:
        return df
    if not desired_order:
        return df
    ordered_cols = [c for c in desired_order if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered_cols]
    return df[ordered_cols + remaining]


def upsert_csv_rows(
    csv_path: Path,
    new_rows: pd.DataFrame,
    *,
    key_cols: Sequence[str],
    desired_order: Sequence[str] | None = None,
    sort_by: Sequence[str] | None = None,
) -> int:
    """
    Append new rows to a CSV if possible; otherwise rewrite the whole file.

    - If `csv_path` doesn't exist, writes it.
    - If it exists and schema is compatible (existing columns are a superset of new columns),
      appends `new_rows` aligned to existing columns.
    - If schema is incompatible, loads the existing CSV and rewrites a merged, de-duplicated file.

    Returns the number of rows *attempted* to write (after dropping duplicates within `new_rows`).
    """
    if new_rows is None or new_rows.empty:
        return 0

    missing = [c for c in key_cols if c not in new_rows.columns]
    if missing:
        raise ValueError(f"new_rows missing key columns: {missing}")

    # Avoid writing duplicate keys from within a single run.
    new_rows = new_rows.drop_duplicates(subset=list(key_cols), keep="last").reset_index(drop=True)
    if new_rows.empty:
        return 0

    new_rows = _apply_desired_order(new_rows, desired_order)
    if sort_by:
        sortable = [c for c in sort_by if c in new_rows.columns]
        if sortable:
            new_rows = new_rows.sort_values(sortable).reset_index(drop=True)

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        new_rows.to_csv(csv_path, index=False)
        return int(new_rows.shape[0])

    existing_cols = _read_csv_header(csv_path)
    if not existing_cols:
        new_rows.to_csv(csv_path, index=False)
        return int(new_rows.shape[0])

    existing_set = set(existing_cols)
    new_set = set(new_rows.columns)

    if existing_set.issuperset(new_set):
        aligned = new_rows.reindex(columns=existing_cols, fill_value=pd.NA)
        aligned.to_csv(csv_path, mode="a", header=False, index=False)
        return int(aligned.shape[0])

    # Schema mismatch: rewrite with union columns so we don't corrupt the CSV by appending.
    existing_df = pd.read_csv(csv_path)
    merged = pd.concat([existing_df, new_rows], ignore_index=True)
    # If the existing CSV predates these key columns, de-duplicating could incorrectly
    # collapse many rows (e.g., all-NA keys). In that case, keep all existing rows.
    if all(col in existing_df.columns for col in key_cols):
        merged = merged.drop_duplicates(subset=list(key_cols), keep="last").reset_index(drop=True)
    else:
        merged = merged.reset_index(drop=True)
    merged = _apply_desired_order(merged, desired_order)
    if sort_by:
        sortable = [c for c in sort_by if c in merged.columns]
        if sortable:
            merged = merged.sort_values(sortable).reset_index(drop=True)
    merged.to_csv(csv_path, index=False)
    return int(new_rows.shape[0])
