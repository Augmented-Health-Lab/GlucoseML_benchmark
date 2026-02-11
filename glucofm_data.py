from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

DEFAULT_HF_NAME = "byluuu/gluco-tsfm-benchmark"


def _require_datasets():  # pragma: no cover
    try:
        from datasets import load_dataset  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing optional dependency 'datasets'. Install it to use --data-source hf:\n"
            "  pip install datasets\n"
        ) from exc
    return load_dataset


def _infer_unix_unit(values: pd.Series) -> str:
    """Infer epoch unit for numeric timestamps (s/ms/ns) based on magnitude.

    Assumptions:
    - epoch seconds are ~1e9 for modern data
    - epoch milliseconds are ~1e12
    - epoch nanoseconds are ~1e18
    """

    if values.empty:
        return "s"
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[~numeric.isna()]
    if numeric.empty:
        return "s"
    max_abs = float(np.nanmax(np.abs(numeric.to_numpy(dtype="float64"))))
    if max_abs >= 1e15:
        return "ns"
    if max_abs >= 1e12:
        return "ms"
    return "s"


def parse_timestamp_series(timestamps: pd.Series) -> pd.Series:
    """Parse timestamps into tz-naive pandas datetime64[ns] series.

    Supports:
    - datetime-like series
    - ISO strings
    - numeric epoch seconds/milliseconds/nanoseconds
    """

    if pd.api.types.is_datetime64_any_dtype(timestamps):
        ts = pd.to_datetime(timestamps, errors="coerce", utc=True)
    elif pd.api.types.is_numeric_dtype(timestamps):
        unit = _infer_unix_unit(timestamps)
        ts = pd.to_datetime(timestamps, errors="coerce", utc=True, unit=unit)
    else:
        ts = pd.to_datetime(timestamps, errors="coerce", utc=True)

    # Convert to tz-naive timestamps for downstream numpy datetime64 arithmetic.
    out = ts.dt.tz_convert(None)
    return out


def hf_row_to_subject_df(
    row: Dict[str, object],
    *,
    timestamp_unit: str = "s",
    value_key: str = "BGvalue",
) -> pd.DataFrame:
    ts_raw = row.get("timestamp")
    values_raw = row.get(value_key)
    if ts_raw is None or values_raw is None:
        raise KeyError(f"HF row missing required keys: timestamp and {value_key!r}")

    ts = pd.to_datetime(pd.Series(ts_raw), unit=timestamp_unit, errors="coerce", utc=True).dt.tz_convert(None)
    values = pd.to_numeric(pd.Series(values_raw), errors="coerce").astype("float32")
    df = pd.DataFrame({"timestamp": ts, "value": values}).dropna(subset=["timestamp", "value"])
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
    return df


def iter_glucofm_subjects_from_hf(
    hf_name: str,
    split: str,
    *,
    datasets: Optional[Set[str]] = None,
    timestamp_unit: str = "s",
) -> Iterable[Tuple[str, str, pd.DataFrame]]:
    """Yield (dataset_name, subject_id, subject_df) from the GlucoFM HF dataset."""

    load_dataset = _require_datasets()
    ds = load_dataset(hf_name, split=split)
    for row in ds:
        dataset_name = str(row["dataset"])
        if datasets is not None and dataset_name not in datasets:
            continue
        subject_id = str(row["subject_id"])
        df = hf_row_to_subject_df(row, timestamp_unit=timestamp_unit)
        yield dataset_name, subject_id, df


def list_glucofm_datasets_from_hf(
    hf_name: str,
    split: str,
) -> List[str]:
    load_dataset = _require_datasets()
    ds = load_dataset(hf_name, split=split)
    return sorted({str(x) for x in ds["dataset"]})

