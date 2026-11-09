"""Shared building blocks to run TiRex-2 on 5-minute CGM data.

Data loading (CSV or HuggingFace), gap-aware sliding windows, batched quantile forecasts,
metrics (RMSE, MAE, prediction interval coverage and width) and a CSV cache of completed results.
Each evaluation script, e.g. ``predict_glucose_multiwindow_tirex-2_zeroshot.py``, builds on this module.
"""

import argparse
import csv
import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch

from tirex2 import TimeseriesType, load_model

# GlucoseML_benchmark/ (repo root): holds glucofm_data.py and clarke/.
MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from glucofm_data import (  # noqa: E402
    DEFAULT_HF_NAME,
    iter_glucofm_subjects_from_hf,
    parse_timestamp_series,
)

MODEL_NAME = "NX-AI/TiRex-2"
# Per-subject CSVs written by prepare_dataset.py (used with --data-source csv).
DATA_SPLITS = {split: PROJECT_ROOT / "hf_cache" / split for split in ("train", "test")}

RESULTS_ROOT = MODEL_DIR / "multi_horizon_results_tirex2_zeroshot"

# Test-time augmentation used for every forecast (checkpoint defaults: differencing on, sign flip off).
TTA_DIFF = False
TTA_SIGN_FLIP = True

CONTROLLED_ACCESS_DATASETS = {"OhioT1DM", "DiaTrend", "T1DEXI"}

STEP_MINUTES = 5
FREQ = f"{STEP_MINUTES}min"
STEP_DELTA = np.timedelta64(STEP_MINUTES, "m")

INPUT_WINDOWS_HOURS = [1, 4, 8, 12, 16, 24]
HORIZONS_MINUTES = [15, 30, 60, 90]
# hf_cache/<split>/mixed/ is a pooled copy of all the other datasets: skipped unless listed in --datasets.
SKIP_DATASET_DIRS = {"mixed"}

DEFAULT_BATCH_SIZE = 256

# Quantile level used as the point forecast (median).
MEDIAN_QUANTILE = 0.5
# {"PI80": (0.1, 0.9), ...}: central prediction intervals scored as coverage / width.
PREDICTION_INTERVALS: Dict[str, Tuple[float, float]] = {"PI80": (0.1, 0.9), "PI60": (0.2, 0.8)}
PI_COLUMNS = [f"{name}_{stat}" for name in PREDICTION_INTERVALS for stat in ("coverage", "width")]

LOGGER = logging.getLogger(__name__)


# Results cache


def _read_csv_header(path: Path) -> List[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:  # utf-8-sig drops a possible BOM
        try:
            return [str(c) for c in next(csv.reader(f))]
        except StopIteration:
            return []


def _boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


_KEY_NORMALIZERS: Dict[str, Callable[[object], object]] = {
    "participant_id": str,
    "context_hours": int,
    "horizon_minutes": int,
    "stride_steps": int,
    "metric_mode": str,
    "latest_requires_truth": _boolish,
}


def _text_id_dtypes(columns: Sequence[str]) -> Dict[str, type]:
    # Read participant_id as text: letting pandas infer it turns zero-padded ids
    # like "001" into 1, so cached keys never match and rows get duplicated.
    return {"participant_id": str} if "participant_id" in columns else {}


def load_completed_keys(
    csv_path: Path,
    *,
    key_cols: Sequence[str],
    require_cols: Sequence[str] = (),
) -> Set[Tuple[object, ...]]:
    """Keys of the rows already saved in a results CSV.

    A row counts as completed only if its ``require_cols`` are non-empty (e.g. a metric added
    after the row was written). A missing file or key column gives an empty set.
    """
    header = _read_csv_header(csv_path)
    if not header or any(col not in header for col in (*key_cols, *require_cols)):
        return set()

    df = pd.read_csv(csv_path, usecols=[*key_cols, *require_cols], dtype=_text_id_dtypes(key_cols))
    df = df.dropna(subset=[*key_cols, *require_cols])[list(key_cols)]
    for col in key_cols:
        if col in _KEY_NORMALIZERS:
            df[col] = df[col].map(_KEY_NORMALIZERS[col])
    return set(df.itertuples(index=False, name=None))


def _apply_desired_order(df: pd.DataFrame, desired_order: Optional[Sequence[str]]) -> pd.DataFrame:
    if df.empty or not desired_order:
        return df
    ordered_cols = [c for c in desired_order if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered_cols]
    return df[ordered_cols + remaining]


def _sort_if_possible(df: pd.DataFrame, sort_by: Optional[Sequence[str]]) -> pd.DataFrame:
    sortable = [c for c in sort_by or () if c in df.columns]
    return df.sort_values(sortable).reset_index(drop=True) if sortable else df


def upsert_csv_rows(
    csv_path: Path,
    new_rows: pd.DataFrame,
    *,
    key_cols: Sequence[str],
    desired_order: Optional[Sequence[str]] = None,
    sort_by: Optional[Sequence[str]] = None,
    replace_existing: bool = False,
) -> int:
    """Append ``new_rows`` to a CSV, or rewrite the merged file when appending is not safe.

    Appending is used when the existing columns include all new columns and
    ``replace_existing`` is False. Otherwise the existing and new rows are merged, keeping
    the new row for each repeated key. Returns the number of rows written.
    """
    if new_rows is None or new_rows.empty:
        return 0

    missing = [c for c in key_cols if c not in new_rows.columns]
    if missing:
        raise ValueError(f"new_rows missing key columns: {missing}")

    new_rows = new_rows.drop_duplicates(subset=list(key_cols), keep="last").reset_index(drop=True)
    new_rows = _sort_if_possible(_apply_desired_order(new_rows, desired_order), sort_by)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    existing_cols = _read_csv_header(csv_path)
    if not existing_cols:
        new_rows.to_csv(csv_path, index=False)
        return int(new_rows.shape[0])

    if set(existing_cols).issuperset(new_rows.columns) and not replace_existing:
        aligned = new_rows.reindex(columns=existing_cols, fill_value=pd.NA)
        aligned.to_csv(csv_path, mode="a", header=False, index=False)
        return int(aligned.shape[0])

    existing_df = pd.read_csv(csv_path, dtype=_text_id_dtypes(existing_cols))
    merged = pd.concat([existing_df, new_rows], ignore_index=True)
    # Without the key columns in the old file, de-duplicating would collapse unrelated rows.
    if all(col in existing_df.columns for col in key_cols):
        merged = merged.drop_duplicates(subset=list(key_cols), keep="last")
    merged = _sort_if_possible(_apply_desired_order(merged.reset_index(drop=True), desired_order), sort_by)
    merged.to_csv(csv_path, index=False)
    return int(new_rows.shape[0])


def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lowered = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def load_subject_dataframe(csv_path: Path, extra_columns: Sequence[str] = ()) -> pd.DataFrame:
    """Read one subject CSV as ``timestamp``/``value`` rows, dropping rows with a missing value.

    ``extra_columns`` are kept as float32 columns. A missing one raises ``KeyError`` (not
    ``ValueError``), so the run stops instead of silently skipping the subject.
    """
    df = pd.read_csv(csv_path)
    timestamp_col = _find_column(df.columns, ["timestamp", "time", "datetime", "date_time", "date"])
    value_col = _find_column(
        df.columns,
        ["bgvalue", "glucose", "glucose_value", "sensor_glucose", "value", "glucose_value_mmol_l"],
    )

    if timestamp_col is None:
        raise ValueError(f"Timestamp column not found in {csv_path}")
    if value_col is None:
        if len(df.columns) < 2:
            raise ValueError(f"Value column not found in {csv_path}")
        value_col = df.columns[1]

    df[timestamp_col] = parse_timestamp_series(df[timestamp_col])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col, value_col])
    missing = [col for col in extra_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Columns {missing} not found in {csv_path}")
    df = df[[timestamp_col, value_col, *extra_columns]].rename(columns={timestamp_col: "timestamp", value_col: "value"})
    df["value"] = df["value"].astype("float32")
    for col in extra_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    return df


def load_subject(csv_paths: List[Path], extra_columns: Sequence[str] = ()) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in csv_paths:
        try:
            frames.append(load_subject_dataframe(path, extra_columns))
        except ValueError as exc:
            LOGGER.warning("%s: skipped while merging (%s)", path, exc)
    if not frames:
        raise ValueError(f"No valid data points in {[str(p) for p in csv_paths]}")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("timestamp").dropna(subset=["timestamp", "value"])
    df = df.drop_duplicates(subset="timestamp")
    if df.empty:
        raise ValueError(f"No valid data points after merging {[str(p) for p in csv_paths]}")
    return df.reset_index(drop=True)


def collect_subject_groups(dataset_dir: Path) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for csv_path in sorted(dataset_dir.rglob("*.csv")):
        participant_id = csv_path.stem
        groups.setdefault(participant_id, []).append(csv_path.resolve())
    return groups


def ensure_results_root(results_root: Path = RESULTS_ROOT) -> None:
    results_root.mkdir(parents=True, exist_ok=True)


def setup_logging(
    results_root: Path = RESULTS_ROOT, log_prefix: str = "predict_glucose_multiwindow_tirex2_zeroshot"
) -> Path:
    ensure_results_root(results_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = results_root / f"{log_prefix}_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
    LOGGER.setLevel(logging.INFO)
    return log_path


def context_steps_from_hours(hours: int) -> int:
    steps = int(round(hours * 60 / STEP_MINUTES))
    return max(1, steps)


def horizon_steps_from_minutes(minutes: int) -> int:
    steps = int(round(minutes / STEP_MINUTES))
    return max(1, steps)


def normalize_timestamps(timestamps: pd.Series) -> np.ndarray:
    ts = parse_timestamp_series(timestamps)
    arr = ts.to_numpy(dtype="datetime64[ns]")
    if np.isnat(arr).any():
        raise ValueError("Timestamps contain NaT values after normalization.")
    return arr


def consecutive_gap_counts(timestamps: np.ndarray) -> np.ndarray:
    n = int(timestamps.shape[0])
    counts = np.zeros(n, dtype=np.int32)
    if n < 2:
        return counts

    diffs = timestamps[1:] - timestamps[:-1]
    is_step = diffs == STEP_DELTA
    for idx in range(n - 2, -1, -1):
        counts[idx] = counts[idx + 1] + 1 if bool(is_step[idx]) else 0
    return counts


def find_latest_contiguous_start(
    timestamps: np.ndarray,
    context_steps: int,
    future_steps: int,
) -> Optional[int]:
    total_len = context_steps + future_steps
    if total_len <= 0:
        return None
    if len(timestamps) < total_len:
        return None

    runs = consecutive_gap_counts(timestamps)
    needed_gaps = total_len - 1
    latest_start = len(timestamps) - total_len
    for start in range(latest_start, -1, -1):
        if int(runs[start]) >= needed_gaps:
            return start
    return None


@dataclass(frozen=True)
class ForecastConfig:
    context_hours: int
    horizon_minutes: int
    context_steps: int
    horizon_steps: int


@dataclass
class HorizonMetrics:
    sse: float = 0.0
    sae: float = 0.0
    points: int = 0
    windows: int = 0
    raw: List[Dict[str, float]] = field(default_factory=list)  # only filled with collect_raw
    quantile_rows: List[Dict[str, float]] = field(default_factory=list)  # only filled with quantile_columns
    window_starts: List[int] = field(default_factory=list)
    pi_covered: Dict[str, int] = field(default_factory=dict)  # per interval: points inside [low, high]
    pi_width_sum: Dict[str, float] = field(default_factory=dict)

    def add_interval(self, name: str, lo: np.ndarray, hi: np.ndarray, truth: np.ndarray) -> None:
        self.pi_covered[name] = self.pi_covered.get(name, 0) + int(np.sum((lo <= truth) & (truth <= hi)))
        self.pi_width_sum[name] = self.pi_width_sum.get(name, 0.0) + float(np.sum(hi - lo))


# (past covariates [V_p, context] or None, future covariates [V_f, context + horizon] or None)
WindowCovariates = Tuple[Optional[np.ndarray], Optional[np.ndarray]]


def default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _batched_tirex_forecast(
    model,
    histories: List[np.ndarray],
    horizon_steps: int,
    tta_sign_flip: Optional[bool] = None,
    tta_diff: Optional[bool] = None,
    covariates: Optional[List[WindowCovariates]] = None,
) -> np.ndarray:
    """Quantile forecast of shape ``(batch, num_quantiles, horizon_steps)`` for equal-length histories.

    Quantiles follow ``model.quantiles``. ``tta_sign_flip`` and ``tta_diff`` set to ``None``
    keep the checkpoint defaults; ``True`` or ``False`` overrides them.
    """
    # Each window is its own series of shape (1, context_steps). Stacking windows into one
    # target would make TiRex-2 treat them as the variates of a single multivariate series.
    if covariates is None:
        covariates = [(None, None)] * len(histories)
    series = [
        TimeseriesType(
            target=torch.tensor(history).unsqueeze(0),
            past_covariates=None if past is None else torch.tensor(past),
            future_covariates=None if future is None else torch.tensor(future),
        )
        for history, (past, future) in zip(histories, covariates)
    ]
    extra_kwargs = {}
    if tta_sign_flip is not None:
        extra_kwargs["tta_sign_flip"] = tta_sign_flip
    if tta_diff is not None:
        extra_kwargs["tta_diff"] = tta_diff
    forecasts = model.forecast(
        series,
        prediction_length=horizon_steps,
        output_type="numpy",
        batch_size=max(1, len(series)),
        **extra_kwargs,
    )
    return np.stack([forecast[0] for forecast in forecasts]).astype("float32")


def evaluate_subject_all_horizons(
    model,
    participant_id: str,
    df: pd.DataFrame,
    context_steps: int,
    horizon_steps_list: List[int],
    stride_steps: int,
    batch_size: int,
    metric_mode: str,
    median_idx: int,
    pi_indices: Dict[str, Tuple[int, int]],
    collect_raw: bool = False,
    quantile_columns: Optional[Sequence[str]] = None,
    tta_sign_flip: Optional[bool] = None,
    tta_diff: Optional[bool] = None,
    window_covariates: Optional[Callable[[int, int, int], WindowCovariates]] = None,
    forecast_horizon_steps: Optional[int] = None,
) -> Dict[int, HorizonMetrics]:
    """Score one context length on several horizons with one forecast per window.

    Each window is forecast once for the longest horizon, and horizon ``h`` is scored on the
    first ``h`` steps. Only windows with enough contiguous data for ``h`` count for ``h``,
    so the result equals scoring each horizon on its own.

    ``pi_indices`` maps each interval name to its (low, high) quantile rows.
    ``quantile_columns`` names the quantile rows of the forecast, in order (e.g. ``q10``, ``q50``).
    When given, every scored point is also stored with all its quantiles in ``quantile_rows``.
    ``forecast_horizon_steps`` fixes the forecast length when covariates are used, because
    the future covariates span it; a resumed run must forecast as far as the full run.
    """
    metrics = {h: HorizonMetrics() for h in horizon_steps_list}
    h_min, h_max = min(horizon_steps_list), max(horizon_steps_list)
    if forecast_horizon_steps is not None:
        h_max = max(h_max, forecast_horizon_steps)
    n = len(df)
    if n < context_steps + h_min:
        return metrics

    values = df["value"].to_numpy(dtype="float32")
    timestamps = normalize_timestamps(df["timestamp"])
    contiguous_runs = consecutive_gap_counts(timestamps)

    batch_starts: List[int] = []

    def quantile_row(q_pred: np.ndarray, start: int, step: int, prediction: float, truth: float) -> Dict[str, float]:
        # ``step`` is the 1-based horizon step; its quantiles are in column ``step - 1`` of ``q_pred``.
        return {
            "window_start": start,
            "horizon_step": step,
            "prediction": prediction,
            "ground_truth": truth,
            **{name: float(q_pred[i, step - 1]) for i, name in enumerate(quantile_columns)},
        }

    def flush_batch() -> None:
        nonlocal batch_starts
        if not batch_starts:
            return
        histories = [values[start : start + context_steps] for start in batch_starts]
        covariates = (
            None
            if window_covariates is None
            else [window_covariates(start, context_steps, h_max) for start in batch_starts]
        )
        try:
            quantile_preds = _batched_tirex_forecast(
                model, histories, h_max, tta_sign_flip=tta_sign_flip, tta_diff=tta_diff, covariates=covariates
            )
        except Exception as exc:
            # With covariates a failure means wrong inputs; skipping the batch would hide it.
            if window_covariates is not None and not isinstance(exc, torch.OutOfMemoryError):
                raise
            LOGGER.warning("%s: prediction failure for %d windows (%s)", participant_id, len(batch_starts), exc)
            batch_starts = []
            return

        for start, q_pred in zip(batch_starts, quantile_preds):
            pred_vec = q_pred[median_idx]
            end = start + context_steps
            for h in horizon_steps_list:
                if start > n - context_steps - h or int(contiguous_runs[start]) < context_steps + h - 1:
                    continue
                targets = values[end : end + h]
                if np.isnan(targets).any():
                    continue
                m = metrics[h]
                if metric_mode == "final":
                    err = float(pred_vec[h - 1]) - float(targets[h - 1])
                    if math.isnan(err):
                        continue
                    m.sse += err * err
                    m.sae += abs(err)
                    m.points += 1
                    for name, (lo_idx, hi_idx) in pi_indices.items():
                        m.add_interval(name, q_pred[lo_idx, h - 1], q_pred[hi_idx, h - 1], targets[h - 1])
                    if collect_raw:
                        m.raw.append(
                            {
                                "window_start": start,
                                "horizon_step": h,
                                "prediction": float(pred_vec[h - 1]),
                                "ground_truth": float(targets[h - 1]),
                            }
                        )
                    if quantile_columns is not None:
                        m.quantile_rows.append(
                            quantile_row(q_pred, start, h, float(pred_vec[h - 1]), float(targets[h - 1]))
                        )
                else:
                    pred_arr = pred_vec[:h]
                    if np.isnan(pred_arr).any():
                        continue
                    errors = pred_arr - targets
                    m.sse += float(np.sum(errors**2))
                    m.sae += float(np.sum(np.abs(errors)))
                    m.points += h
                    for name, (lo_idx, hi_idx) in pi_indices.items():
                        m.add_interval(name, q_pred[lo_idx, :h], q_pred[hi_idx, :h], targets)
                    if collect_raw:
                        m.raw.extend(
                            {
                                "window_start": start,
                                "horizon_step": step + 1,
                                "prediction": float(pred_arr[step]),
                                "ground_truth": float(targets[step]),
                            }
                            for step in range(h)
                        )
                    if quantile_columns is not None:
                        m.quantile_rows.extend(
                            quantile_row(q_pred, start, step + 1, float(pred_arr[step]), float(targets[step]))
                            for step in range(h)
                        )
                m.windows += 1
                m.window_starts.append(start)
        batch_starts = []

    for start in range(0, n - context_steps - h_min + 1, stride_steps):
        if int(contiguous_runs[start]) < context_steps + h_min - 1:
            continue
        if np.isnan(values[start : start + context_steps]).any():
            continue
        batch_starts.append(start)
        if len(batch_starts) >= batch_size:
            flush_batch()

    flush_batch()
    return metrics


def load_existing_metrics(output_path: Path) -> pd.DataFrame:
    if not output_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(output_path)
    except Exception as exc:
        LOGGER.warning("Failed to load existing metrics from %s (%s)", output_path, exc)
        return pd.DataFrame()


RAW_DESIRED_ORDER = [
    "dataset",
    "split",
    "participant_id",
    "context_hours",
    "horizon_minutes",
    "stride_steps",
    "metric_mode",
    "window_start",
    "horizon_step",
    "prediction",
    "ground_truth",
]


EVAL_CACHE_KEY_COLS = ("participant_id", "context_hours", "horizon_minutes", "stride_steps", "metric_mode")
RAW_SORT = ["participant_id", "context_hours", "horizon_minutes", "window_start", "horizon_step"]
METRICS_DESIRED_ORDER = [
    "dataset",
    "split",
    "participant_id",
    "context_hours",
    "horizon_minutes",
    "context_steps",
    "model_context_steps",
    "horizon_steps",
    "step_minutes",
    "freq",
    "task",
    "stride_steps",
    "metric_mode",
    "latest_requires_truth",
    "prediction_timestamp",
    "prediction_value",
    "truth_value",
    "rmse",
    "mae",
    *PI_COLUMNS,
    "windows",
]


def add_common_arguments(parser: argparse.ArgumentParser, data_splits: Dict[str, Path] = DATA_SPLITS) -> None:
    """Arguments for the model, data roots, windows and outputs."""
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help="HuggingFace model id or local directory to load.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_device(),
        help="Device to load TiRex-2 on (e.g. 'mps', 'cuda:0', 'cpu'). Default: mps, else cuda, else cpu.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        choices=["train", "test"],
        help="Which dataset splits to evaluate (default: test).",
    )
    for split in ("train", "test"):
        parser.add_argument(
            f"--data-root-{split}",
            type=Path,
            default=data_splits[split],
            help=(
                f"Root directory containing dataset subfolders for the {split} split "
                f"(default: {data_splits[split].relative_to(PROJECT_ROOT)})."
            ),
        )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help=(
            "Optional dataset folder names to include (default: all except "
            f"{sorted(SKIP_DATASET_DIRS)}, which are only run when listed here)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Inference batch size (number of windows per forecast call).",
    )
    parser.add_argument(
        "--stride-steps",
        type=int,
        default=1,
        help="Sliding window stride in steps (default: 1 = 5 minutes; 0 = context_steps).",
    )
    parser.add_argument(
        "--metric-mode",
        choices=["final", "all"],
        default="final",
        help="Use only the final horizon step (final) or all horizon steps (all) for error metrics.",
    )
    parser.add_argument(
        "--context-hours",
        type=int,
        nargs="+",
        default=INPUT_WINDOWS_HOURS,
        help=f"Context window(s) in hours (default: {INPUT_WINDOWS_HOURS}).",
    )
    parser.add_argument(
        "--horizons-minutes",
        type=int,
        nargs="+",
        default=HORIZONS_MINUTES,
        help=f"Forecast horizon(s) in minutes (default: {HORIZONS_MINUTES}).",
    )
    parser.add_argument(
        "--save-raw-predictions",
        action="store_true",
        help=(
            "Eval task only: also write every scored (prediction, ground_truth) pair to "
            "raw_predictions/<dataset>_<split>_raw_predictions.csv (needed for CEG/SEG)."
        ),
    )
    parser.add_argument(
        "--save-quantile-predictions",
        action="store_true",
        help=(
            "Eval task only: also write every scored point with all the model quantiles to "
            "quantile_predictions/<dataset>_<split>_quantile_predictions.csv (columns q10 ... q90)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and overwrite existing results instead of skipping completed rows.",
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TiRex-2 on 5-minute CGM data for multiple context windows and horizons."
    )
    parser.add_argument(
        "--task",
        choices=["eval", "latest"],
        default="eval",
        help="Run rolling-window evaluation (eval) or produce one latest forecast per participant (latest).",
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--data-source",
        choices=["csv", "hf"],
        default="csv",
        help=(
            "Data source: 'csv' reads per-subject CSVs from --data-root-*, "
            "'hf' loads GlucoFM directly from HuggingFace (requires `datasets`)."
        ),
    )
    parser.add_argument(
        "--hf-name",
        type=str,
        default=DEFAULT_HF_NAME,
        help=f"HuggingFace dataset repo (default: {DEFAULT_HF_NAME}).",
    )
    parser.add_argument("--hf-train-split", type=str, default="train")
    parser.add_argument("--hf-test-split", type=str, default="test")
    parser.add_argument(
        "--latest-requires-truth",
        action="store_true",
        help="For --task latest, pick the latest window that still has ground truth for the horizon.",
    )
    return parser.parse_args(argv)


def quantile_indices(model) -> Tuple[int, Dict[str, Tuple[int, int]]]:
    """Row of the median and (low, high) rows of each prediction interval in the model's quantiles."""
    quantile_levels = [round(q, 6) for q in model.quantiles.tolist()]
    if MEDIAN_QUANTILE not in quantile_levels:
        raise RuntimeError(
            f"Median quantile {MEDIAN_QUANTILE} not in model quantiles {quantile_levels}"
        )
    median_idx = quantile_levels.index(MEDIAN_QUANTILE)
    LOGGER.info("TiRex-2 quantiles=%s, median index=%d", quantile_levels, median_idx)
    missing_q = {q for bounds in PREDICTION_INTERVALS.values() for q in bounds} - set(quantile_levels)
    if missing_q:
        raise RuntimeError(f"Prediction-interval quantiles {sorted(missing_q)} not in model quantiles {quantile_levels}")
    pi_indices = {
        name: (quantile_levels.index(lo), quantile_levels.index(hi)) for name, (lo, hi) in PREDICTION_INTERVALS.items()
    }
    return median_idx, pi_indices


def quantile_column_names(model) -> List[str]:
    """Column name of each quantile row of the model, in order: level 0.1 gives ``q10``."""
    return [f"q{int(round(q * 100))}" for q in model.quantiles.tolist()]


def build_configs(context_hours_list: List[int], horizons_minutes_list: List[int]) -> List[ForecastConfig]:
    configs: List[ForecastConfig] = []
    for context_hours in context_hours_list:
        for horizon_minutes in horizons_minutes_list:
            configs.append(
                ForecastConfig(
                    context_hours=context_hours,
                    horizon_minutes=horizon_minutes,
                    context_steps=context_steps_from_hours(context_hours),
                    horizon_steps=horizon_steps_from_minutes(horizon_minutes),
                )
            )
    return configs


def select_dataset_dirs(data_root: Path, allowed: Optional[Set[str]]) -> List[Path]:
    dataset_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
    if allowed is not None:
        return [p for p in dataset_dirs if p.name in allowed]
    return [p for p in dataset_dirs if p.name not in SKIP_DATASET_DIRS]


def load_eval_cache(
    dataset_name: str,
    split_name: str,
    output_path: Path,
    raw_path: Path,
    save_raw: bool,
    quantile_path: Path,
    save_quantiles: bool,
) -> Tuple[Set[Tuple[object, ...]], Set[Tuple[object, ...]], Set[Tuple[object, ...]]]:
    """Completed eval keys of the metrics file and, when saved, of the raw and quantile prediction files."""
    # Rows saved before the PI metrics existed count as pending.
    completed_keys = load_completed_keys(output_path, key_cols=EVAL_CACHE_KEY_COLS, require_cols=("PI80_coverage",))
    raw_completed_keys: Set[Tuple[object, ...]] = set()
    if save_raw:
        raw_completed_keys = load_completed_keys(raw_path, key_cols=EVAL_CACHE_KEY_COLS)
    quantile_completed_keys: Set[Tuple[object, ...]] = set()
    if save_quantiles:
        quantile_completed_keys = load_completed_keys(quantile_path, key_cols=EVAL_CACHE_KEY_COLS)
    if completed_keys:
        LOGGER.info(
            "%s (%s split): found %d existing rows, skipping unless --overwrite is set.",
            dataset_name,
            split_name,
            len(completed_keys),
        )
    return completed_keys, raw_completed_keys, quantile_completed_keys


def evaluate_subject_contexts(
    model,
    args: argparse.Namespace,
    dataset_name: str,
    split_name: str,
    participant_id: str,
    df: pd.DataFrame,
    configs: List[ForecastConfig],
    completed_keys: Set[Tuple[object, ...]],
    raw_completed_keys: Set[Tuple[object, ...]],
    save_raw: bool,
    quantile_completed_keys: Set[Tuple[object, ...]],
    save_quantiles: bool,
    quantile_columns: Sequence[str],
    median_idx: int,
    pi_indices: Dict[str, Tuple[int, int]],
    records: List[Dict[str, object]],
    raw_records: List[Dict[str, object]],
    quantile_records: List[Dict[str, object]],
    on_evaluated: Optional[Callable[[int, int, List[ForecastConfig], Dict[int, HorizonMetrics]], None]] = None,
    **evaluate_kwargs,
) -> None:
    """Score every pending (context, horizon) config of one subject.

    Appends one metrics row per config to ``records``. With ``save_raw``, its scored points go to
    ``raw_records``; with ``save_quantiles``, the same points with all their quantiles go to
    ``quantile_records``. ``evaluate_kwargs`` go to ``evaluate_subject_all_horizons``;
    ``on_evaluated`` is called after each context length is scored.
    """
    for context_hours in args.context_hours:
        context_steps = context_steps_from_hours(context_hours)
        stride_steps = context_steps if args.stride_steps <= 0 else args.stride_steps

        def cfg_key(cfg: ForecastConfig) -> Tuple[object, ...]:
            return (
                participant_id,
                cfg.context_hours,
                cfg.horizon_minutes,
                int(stride_steps),
                str(args.metric_mode),
            )

        # Each output file only gets the rows it lacks, since appending does not de-duplicate.
        pending = [
            cfg
            for cfg in configs
            if cfg.context_hours == context_hours
            and (
                cfg_key(cfg) not in completed_keys
                or (save_raw and cfg_key(cfg) not in raw_completed_keys)
                or (save_quantiles and cfg_key(cfg) not in quantile_completed_keys)
            )
        ]
        if not pending:
            continue

        metrics = evaluate_subject_all_horizons(
            model,
            participant_id,
            df,
            context_steps,
            [cfg.horizon_steps for cfg in pending],
            stride_steps=stride_steps,
            batch_size=args.batch_size,
            metric_mode=args.metric_mode,
            median_idx=median_idx,
            pi_indices=pi_indices,
            collect_raw=save_raw,
            quantile_columns=quantile_columns if save_quantiles else None,
            forecast_horizon_steps=(
                None
                if evaluate_kwargs.get("window_covariates") is None
                else max(cfg.horizon_steps for cfg in configs)
            ),
            **evaluate_kwargs,
        )
        if on_evaluated is not None:
            on_evaluated(context_hours, stride_steps, pending, metrics)
        for cfg in pending:
            m = metrics[cfg.horizon_steps]
            if m.points == 0 or m.windows == 0:
                continue
            if cfg_key(cfg) not in completed_keys:
                records.append(
                    {
                        "dataset": dataset_name,
                        "split": split_name,
                        "participant_id": participant_id,
                        "context_hours": cfg.context_hours,
                        "horizon_minutes": cfg.horizon_minutes,
                        "context_steps": cfg.context_steps,
                        "model_context_steps": cfg.context_steps,
                        "horizon_steps": cfg.horizon_steps,
                        "step_minutes": STEP_MINUTES,
                        "freq": FREQ,
                        "stride_steps": stride_steps,
                        "metric_mode": args.metric_mode,
                        "rmse": math.sqrt(m.sse / m.points),
                        "mae": m.sae / m.points,
                        **{
                            f"{name}_{stat}": total / m.points
                            for name in PREDICTION_INTERVALS
                            for stat, total in (
                                ("coverage", m.pi_covered[name]),
                                ("width", m.pi_width_sum[name]),
                            )
                        },
                        "windows": m.windows,
                        "task": "eval",
                    }
                )
            key_fields = {
                "dataset": dataset_name,
                "split": split_name,
                "participant_id": participant_id,
                "context_hours": cfg.context_hours,
                "horizon_minutes": cfg.horizon_minutes,
                "stride_steps": stride_steps,
                "metric_mode": args.metric_mode,
            }
            if save_raw and cfg_key(cfg) not in raw_completed_keys:
                raw_records.extend({**key_fields, **row} for row in m.raw)
            if save_quantiles and cfg_key(cfg) not in quantile_completed_keys:
                quantile_records.extend({**key_fields, **row} for row in m.quantile_rows)


def write_prediction_rows(
    records: List[Dict[str, object]],
    path: Path,
    columns: Sequence[str],
    overwrite: bool,
    label: str,
    cache_key_cols: Tuple[str, ...] = EVAL_CACHE_KEY_COLS,
) -> None:
    """Write per-point rows (raw predictions or quantile predictions) with the given ``columns``."""
    if not records:
        return
    df = pd.DataFrame(records, columns=list(columns))
    if overwrite:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.sort_values(RAW_SORT).to_csv(path, index=False)
        written = len(df)
    else:
        written = upsert_csv_rows(
            path,
            df,
            key_cols=cache_key_cols + ("window_start", "horizon_step"),
            desired_order=columns,
            sort_by=RAW_SORT,
        )
    LOGGER.info("Saved %s to %s (+%d rows)", label, path, written)


def write_metrics(
    records: List[Dict[str, object]],
    output_path: Path,
    overwrite: bool,
    cache_key_cols: Tuple[str, ...],
    dataset_name: str,
    split_name: str,
) -> None:
    new_df = pd.DataFrame(records)
    if new_df.empty:
        if output_path.exists():
            LOGGER.info("%s (%s split): no new rows to write.", dataset_name, split_name)
        else:
            LOGGER.warning("%s (%s split): no metrics to write.", dataset_name, split_name)
        return

    if overwrite:
        final_df = new_df
        ordered_cols = [col for col in METRICS_DESIRED_ORDER if col in final_df.columns]
        remaining_cols = [col for col in final_df.columns if col not in ordered_cols]
        final_df = final_df[ordered_cols + remaining_cols]
        final_df = final_df.sort_values(
            ["participant_id", "context_hours", "horizon_minutes"]
        ).reset_index(drop=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_path, index=False)
        LOGGER.info("Saved metrics to %s (%d rows)", output_path, len(final_df))
    else:
        written = upsert_csv_rows(
            output_path,
            new_df,
            key_cols=cache_key_cols,
            desired_order=METRICS_DESIRED_ORDER,
            sort_by=["participant_id", "context_hours", "horizon_minutes"],
            replace_existing=True,  # refreshed rows must replace the old ones, not duplicate them
        )
        LOGGER.info("Saved metrics to %s (+%d rows)", output_path, written)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    log_path = setup_logging()
    LOGGER.info("Logging to %s", log_path)

    split_roots = {"train": args.data_root_train, "test": args.data_root_test}

    LOGGER.info("Loading TiRex-2 model '%s' (device=%s)", args.model_name, args.device)
    model = load_model(args.model_name, device=args.device)
    LOGGER.info("Test-time augmentation: tta_diff=%s, tta_sign_flip=%s", TTA_DIFF, TTA_SIGN_FLIP)
    median_idx, pi_indices = quantile_indices(model)
    quantile_columns = quantile_column_names(model)
    quantile_file_columns = [*RAW_DESIRED_ORDER, *quantile_columns]

    configs = build_configs(args.context_hours, args.horizons_minutes)

    allowed = set(args.datasets) if args.datasets is not None else None
    save_raw = args.save_raw_predictions and args.task == "eval"
    save_quantiles = args.save_quantile_predictions and args.task == "eval"
    raw_dir = RESULTS_ROOT / "raw_predictions"
    quantile_dir = RESULTS_ROOT / "quantile_predictions"

    for split_name in args.splits:
        output_suffix = "metrics" if args.task == "eval" else "latest_predictions"

        if args.data_source == "hf":
            hf_split = args.hf_train_split if split_name == "train" else args.hf_test_split
            subjects_by_dataset: Dict[str, Dict[str, pd.DataFrame]] = {}
            for dataset_name, subject_id, df in iter_glucofm_subjects_from_hf(
                args.hf_name,
                hf_split,
                datasets=allowed,
            ):
                subjects_by_dataset.setdefault(dataset_name, {})[subject_id] = df

            if not subjects_by_dataset:
                LOGGER.warning("No HF datasets found for split '%s' (skipping).", split_name)
                continue

            if allowed is None:
                for name in SKIP_DATASET_DIRS & set(subjects_by_dataset):
                    del subjects_by_dataset[name]
            dataset_items = sorted(subjects_by_dataset.items(), key=lambda kv: kv[0])
            dataset_iter = ((name, subjects_by_dataset[name]) for name, _ in dataset_items)
        else:
            data_root = split_roots[split_name]
            if not data_root.exists():
                LOGGER.warning("Split '%s' not found at %s (skipping)", split_name, data_root)
                continue

            dataset_dirs = select_dataset_dirs(data_root, allowed)
            if not dataset_dirs:
                LOGGER.warning("No dataset folders found in %s for split '%s'", data_root, split_name)
                continue

            dataset_iter = ((p.name, p) for p in dataset_dirs)

        for dataset_name, dataset_payload in dataset_iter:
            output_path = RESULTS_ROOT / f"{dataset_name}_{split_name}_{output_suffix}.csv"
            cache_key_cols = (
                EVAL_CACHE_KEY_COLS
                if args.task == "eval"
                else ("participant_id", "context_hours", "horizon_minutes", "latest_requires_truth")
            )
            raw_path = raw_dir / f"{dataset_name}_{split_name}_raw_predictions.csv"
            quantile_path = quantile_dir / f"{dataset_name}_{split_name}_quantile_predictions.csv"
            completed_keys: Set[Tuple[object, ...]] = set()
            raw_completed_keys: Set[Tuple[object, ...]] = set()
            quantile_completed_keys: Set[Tuple[object, ...]] = set()
            if not args.overwrite:
                if args.task == "eval":
                    completed_keys, raw_completed_keys, quantile_completed_keys = load_eval_cache(
                        dataset_name, split_name, output_path, raw_path, save_raw, quantile_path, save_quantiles
                    )
                else:
                    completed_keys = load_completed_keys(output_path, key_cols=cache_key_cols)
                    if completed_keys:
                        LOGGER.info(
                            "%s (%s split): found %d existing rows, skipping unless --overwrite is set.",
                            dataset_name,
                            split_name,
                            len(completed_keys),
                        )

            records: List[Dict[str, object]] = []
            raw_records: List[Dict[str, object]] = []
            quantile_records: List[Dict[str, object]] = []
            if args.data_source == "hf":
                subject_groups = dataset_payload
            else:
                subject_groups = collect_subject_groups(dataset_payload)

            LOGGER.info("%s (%s split): %d participants", dataset_name, split_name, len(subject_groups))
            for participant_id, subject_payload in subject_groups.items():
                if args.data_source == "hf":
                    df = subject_payload
                else:
                    try:
                        df = load_subject(subject_payload)
                    except (ValueError, pd.errors.EmptyDataError) as exc:
                        LOGGER.warning("%s / %s: skipped (%s)", dataset_name, participant_id, exc)
                        continue

                if args.task == "latest":
                    for cfg in configs:
                        model_context_steps = cfg.context_steps
                        key = (
                            participant_id,
                            cfg.context_hours,
                            cfg.horizon_minutes,
                            bool(args.latest_requires_truth),
                        )
                        if key in completed_keys:
                            continue

                        required_future = cfg.horizon_steps if args.latest_requires_truth else 0
                        try:
                            timestamps = normalize_timestamps(df["timestamp"])
                        except ValueError as exc:
                            LOGGER.warning("%s: skipped due to invalid timestamps (%s)", participant_id, exc)
                            continue
                        start = find_latest_contiguous_start(timestamps, cfg.context_steps, required_future)
                        if start is None:
                            continue
                        end = start + cfg.context_steps

                        history = df["value"].iloc[start:end].to_numpy(dtype="float32")
                        if np.isnan(history).any() or int(history.shape[0]) != int(cfg.context_steps):
                            continue

                        try:
                            preds = _batched_tirex_forecast(
                                model,
                                [history],
                                cfg.horizon_steps,
                                tta_sign_flip=TTA_SIGN_FLIP,
                                tta_diff=TTA_DIFF,
                            )[0, median_idx]
                        except Exception as exc:
                            LOGGER.warning("%s: latest prediction failure (%s)", participant_id, exc)
                            continue

                        pred_final = float(preds[cfg.horizon_steps - 1])
                        pred_ts = pd.to_datetime(timestamps[end - 1]) + pd.to_timedelta(cfg.horizon_minutes, unit="m")
                        true_final: Optional[float] = None
                        if args.latest_requires_truth:
                            true_final = float(df["value"].iloc[end + cfg.horizon_steps - 1])

                        records.append(
                            {
                                "dataset": dataset_name,
                                "split": split_name,
                                "participant_id": participant_id,
                                "context_hours": cfg.context_hours,
                                "horizon_minutes": cfg.horizon_minutes,
                                "context_steps": cfg.context_steps,
                                "model_context_steps": model_context_steps,
                                "horizon_steps": cfg.horizon_steps,
                                "step_minutes": STEP_MINUTES,
                                "freq": FREQ,
                                "prediction_timestamp": pred_ts,
                                "prediction_value": pred_final,
                                "truth_value": true_final,
                                "task": "latest",
                                "latest_requires_truth": bool(args.latest_requires_truth),
                            }
                        )
                    continue

                evaluate_subject_contexts(
                    model,
                    args,
                    dataset_name,
                    split_name,
                    participant_id,
                    df,
                    configs,
                    completed_keys,
                    raw_completed_keys,
                    save_raw,
                    quantile_completed_keys,
                    save_quantiles,
                    quantile_columns,
                    median_idx,
                    pi_indices,
                    records,
                    raw_records,
                    quantile_records,
                    tta_sign_flip=TTA_SIGN_FLIP,
                    tta_diff=TTA_DIFF,
                )

            write_prediction_rows(
                raw_records, raw_path, RAW_DESIRED_ORDER, args.overwrite, "raw predictions", cache_key_cols
            )
            write_prediction_rows(
                quantile_records,
                quantile_path,
                quantile_file_columns,
                args.overwrite,
                "quantile predictions",
                cache_key_cols,
            )
            write_metrics(records, output_path, args.overwrite, cache_key_cols, dataset_name, split_name)


if __name__ == "__main__":
    main()
