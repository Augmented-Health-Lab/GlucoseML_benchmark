import argparse
import csv
import logging
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from gluonts.dataset.common import ListDataset
from gluonts.torch import PyTorchPredictor

try:
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

# ── Inlined from glucofm_data.py ───────────────────────────────────────────

DEFAULT_HF_NAME = "byluuu/gluco-tsfm-benchmark"


def _require_datasets():
    try:
        from datasets import load_dataset  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing optional dependency 'datasets'. Install it with: pip install datasets"
        ) from exc
    return load_dataset


def _infer_unix_unit(values: pd.Series) -> str:
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
    if pd.api.types.is_datetime64_any_dtype(timestamps):
        ts = pd.to_datetime(timestamps, errors="coerce", utc=True)
    elif pd.api.types.is_numeric_dtype(timestamps):
        unit = _infer_unix_unit(timestamps)
        ts = pd.to_datetime(timestamps, errors="coerce", utc=True, unit=unit)
    else:
        ts = pd.to_datetime(timestamps, errors="coerce", utc=True)
    return ts.dt.tz_convert(None)


def hf_row_to_subject_df(row: Dict, *, timestamp_unit: str = "s", value_key: str = "BGvalue") -> pd.DataFrame:
    ts = pd.to_datetime(pd.Series(row["timestamp"]), unit=timestamp_unit, errors="coerce", utc=True).dt.tz_convert(None)
    values = pd.to_numeric(pd.Series(row[value_key]), errors="coerce").astype("float32")
    df = pd.DataFrame({"timestamp": ts, "value": values}).dropna(subset=["timestamp", "value"])
    return df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)


def iter_glucofm_subjects_from_hf(
    hf_name: str, split: str, *, datasets: Optional[Set[str]] = None, timestamp_unit: str = "s"
) -> Iterable[Tuple[str, str, pd.DataFrame]]:
    load_dataset = _require_datasets()
    ds = load_dataset(hf_name, split=split)
    for row in ds:
        dataset_name = str(row["dataset"])
        if datasets is not None and dataset_name not in datasets:
            continue
        yield dataset_name, str(row["subject_id"]), hf_row_to_subject_df(row, timestamp_unit=timestamp_unit)


# ── Inlined from results_cache.py ─────────────────────────────────────────

def _read_csv_header(path: Path) -> List[str]:
    if not path.exists():
        return []
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
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


_NORMALIZERS: Dict[str, Callable] = {
    "participant_id":    lambda v: str(v),
    "context_hours":     lambda v: int(v),
    "horizon_minutes":   lambda v: int(v),
    "stride_steps":      lambda v: int(v),
    "metric_mode":       lambda v: str(v),
    "latest_requires_truth": _boolish,
}


def load_completed_keys(csv_path: Path, *, key_cols: Sequence[str]) -> Set[Tuple]:
    if not csv_path.exists():
        return set()
    header = _read_csv_header(csv_path)
    if not header or any(col not in set(header) for col in key_cols):
        return set()
    df = pd.read_csv(csv_path, usecols=list(key_cols)).dropna(subset=list(key_cols))
    if df.empty:
        return set()
    for col in key_cols:
        if col in _NORMALIZERS:
            df[col] = df[col].map(_NORMALIZERS[col])
    return {tuple(row) for row in df.itertuples(index=False, name=None)}


def _apply_desired_order(df: pd.DataFrame, desired_order: Optional[Sequence[str]]) -> pd.DataFrame:
    if df.empty or not desired_order:
        return df
    ordered = [c for c in desired_order if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered]
    return df[ordered + remaining]


def upsert_csv_rows(
    csv_path: Path,
    new_rows: pd.DataFrame,
    *,
    key_cols: Sequence[str],
    desired_order: Optional[Sequence[str]] = None,
    sort_by: Optional[Sequence[str]] = None,
) -> int:
    if new_rows is None or new_rows.empty:
        return 0
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
        return len(new_rows)
    existing_cols = _read_csv_header(csv_path)
    if not existing_cols:
        new_rows.to_csv(csv_path, index=False)
        return len(new_rows)
    if set(existing_cols).issuperset(set(new_rows.columns)):
        new_rows.reindex(columns=existing_cols, fill_value=pd.NA).to_csv(csv_path, mode="a", header=False, index=False)
        return len(new_rows)
    existing_df = pd.read_csv(csv_path)
    merged = pd.concat([existing_df, new_rows], ignore_index=True)
    if all(c in existing_df.columns for c in key_cols):
        merged = merged.drop_duplicates(subset=list(key_cols), keep="last").reset_index(drop=True)
    merged = _apply_desired_order(merged, desired_order)
    if sort_by:
        sortable = [c for c in sort_by if c in merged.columns]
        if sortable:
            merged = merged.sort_values(sortable).reset_index(drop=True)
    merged.to_csv(csv_path, index=False)
    return len(new_rows)

MODEL_NAME = "Salesforce/moirai-2.0-R-small"
DATA_SPLITS = {
    "train": Path(__file__).resolve().parent.parent / "hf_cache" / "train",
    "test":  Path(__file__).resolve().parent.parent / "hf_cache" / "test",
}

# ── Output goes to ./quantile ──────────────────────────────────────────────
RESULTS_ROOT = Path(__file__).resolve().parent / "quantile"

STEP_MINUTES = 5
FREQ = "5min"
STEP_DELTA = np.timedelta64(STEP_MINUTES, "m")

INPUT_WINDOWS_HOURS = [1, 4, 8, 12, 16, 24]
HORIZONS_MINUTES    = [15, 30, 60, 90]

DEFAULT_BATCH_SIZE = 8

# Quantile levels to extract from the sample distribution
QUANTILE_LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]

LOGGER = logging.getLogger(__name__)


# ── Unchanged helpers ──────────────────────────────────────────────────────

def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lowered = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def load_subject_dataframe(csv_path: Path) -> pd.DataFrame:
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
    df = df[[timestamp_col, value_col]].rename(columns={timestamp_col: "timestamp", value_col: "value"})
    df["value"] = df["value"].astype("float32")
    return df


def load_subject(csv_paths: List[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in csv_paths:
        try:
            frames.append(load_subject_dataframe(path))
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


def ensure_results_root() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


def setup_logging() -> Path:
    ensure_results_root()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = RESULTS_ROOT / f"predict_glucose_multiwindow_uni2ts_zeroshot_quantile_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
    LOGGER.setLevel(logging.INFO)
    return log_path


def context_steps_from_hours(hours: int) -> int:
    return max(1, int(round(hours * 60 / STEP_MINUTES)))


def horizon_steps_from_minutes(minutes: int) -> int:
    return max(1, int(round(minutes / STEP_MINUTES)))


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
    if total_len <= 0 or len(timestamps) < total_len:
        return None
    runs = consecutive_gap_counts(timestamps)
    needed_gaps = total_len - 1
    for start in range(len(timestamps) - total_len, -1, -1):
        if int(runs[start]) >= needed_gaps:
            return start
    return None


@dataclass(frozen=True)
class ForecastConfig:
    context_hours:   int
    horizon_minutes: int
    context_steps:   int
    horizon_steps:   int


def build_predictor(
    module: Moirai2Module,
    context_steps: int,
    horizon_steps: int,
    device: str,
    batch_size: int,
) -> PyTorchPredictor:
    forecast = Moirai2Forecast(
        module=module,
        prediction_length=horizon_steps,
        context_length=context_steps,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    ).to(device)
    forecast.eval()
    return forecast.create_predictor(batch_size=batch_size, device=device)


# ── Point-forecast extractor (unchanged logic) ─────────────────────────────

def _forecast_to_array(forecast, horizon_steps: int) -> np.ndarray:
    preds: Optional[np.ndarray] = None
    if hasattr(forecast, "quantile"):
        try:
            preds = np.asarray(forecast.quantile(0.5), dtype="float32")
        except Exception:
            preds = None
    if (preds is None or preds.size == 0) and hasattr(forecast, "samples"):
        samples = np.asarray(forecast.samples, dtype="float32")
        preds = samples.mean(axis=0) if samples.ndim > 1 else samples
    if (preds is None or preds.size == 0) and hasattr(forecast, "mean") and forecast.mean is not None:
        try:
            preds = np.asarray(forecast.mean, dtype="float32")
        except Exception:
            preds = None
    if preds is None or preds.size == 0:
        raise ValueError("Unable to extract predictions from forecast object.")
    return preds[:horizon_steps]


# ── NEW: quantile extractor ────────────────────────────────────────────────

def _forecast_to_quantiles(forecast, horizon_steps: int, quantile_levels: List[float]) -> Optional[Dict[float, np.ndarray]]:
    """
    Extract per-quantile arrays from a GluonTS SampleForecast.
    Returns dict {q_level: array of shape (horizon_steps,)} or None if unavailable.
    """
    if not hasattr(forecast, "quantile"):
        return None
    result = {}
    try:
        for q in quantile_levels:
            arr = np.asarray(forecast.quantile(q), dtype="float32")
            result[q] = arr[:horizon_steps]
    except Exception:
        return None
    return result


# ── evaluate_subject: now also returns PI metrics ──────────────────────────

def evaluate_subject(
    predictor: PyTorchPredictor,
    participant_id: str,
    df: pd.DataFrame,
    cfg: ForecastConfig,
    stride_steps: int,
    batch_size: int,
    metric_mode: str,
) -> Tuple[float, float, int, int, Dict[float, float], Dict[float, float], int]:
    """
    Returns:
        total_sse, total_ae, total_points, total_windows,
        pi_coverage  {(q_lo, q_hi): coverage_rate},
        pi_width     {(q_lo, q_hi): mean_width},
        pi_windows   (number of windows with valid quantiles)
    """
    if len(df) < cfg.context_steps + cfg.horizon_steps:
        return 0.0, 0.0, 0, 0, {}, {}, 0

    values     = df["value"].to_numpy(dtype="float32")
    timestamps = normalize_timestamps(df["timestamp"])
    contiguous_runs = consecutive_gap_counts(timestamps)
    required_gaps   = cfg.context_steps + cfg.horizon_steps - 1

    total_sse    = 0.0
    total_ae     = 0.0
    total_points = 0
    total_windows = 0

    # PI accumulators: keyed by (q_lo, q_hi) tuple
    PI_PAIRS = [(0.1, 0.9), (0.25, 0.75)]   # 80% and 50% PIs
    pi_covered: Dict[Tuple[float, float], int]   = {p: 0 for p in PI_PAIRS}
    pi_width_sum: Dict[Tuple[float, float], float] = {p: 0.0 for p in PI_PAIRS}
    pi_windows = 0

    batch_entries:  List[Dict[str, object]] = []
    batch_targets:  List[np.ndarray]        = []

    def flush_batch() -> None:
        nonlocal total_sse, total_ae, total_points, total_windows
        nonlocal pi_covered, pi_width_sum, pi_windows
        nonlocal batch_entries, batch_targets
        if not batch_entries:
            return
        dataset = ListDataset(batch_entries, freq=FREQ)
        try:
            with torch.inference_mode():
                forecasts = list(predictor.predict(dataset))
        except Exception as exc:
            LOGGER.warning("%s: prediction failure for %d windows (%s)", participant_id, len(batch_entries), exc)
            batch_entries = []
            batch_targets = []
            return

        for forecast, target_vals in zip(forecasts, batch_targets):
            # ── point metrics ──────────────────────────────────────────
            try:
                preds = _forecast_to_array(forecast, cfg.horizon_steps)
            except ValueError as exc:
                LOGGER.warning("%s: failed to parse forecast (%s)", participant_id, exc)
                continue

            if metric_mode == "final":
                step     = cfg.horizon_steps - 1
                pred_val = float(preds[step])
                gt_val   = float(target_vals[step])
                if math.isnan(pred_val) or math.isnan(gt_val):
                    continue
                err = pred_val - gt_val
                total_sse    += float(err * err)
                total_ae     += float(abs(err))
                total_points += 1
                total_windows += 1
            else:
                valid_len = min(len(preds), len(target_vals), cfg.horizon_steps)
                if valid_len == 0:
                    continue
                p = preds[:valid_len]
                t = target_vals[:valid_len]
                if np.isnan(p).any() or np.isnan(t).any():
                    continue
                errors = p - t
                total_sse    += float(np.sum(errors ** 2))
                total_ae     += float(np.sum(np.abs(errors)))
                total_points += int(valid_len)
                total_windows += 1

            # ── quantile / PI metrics ──────────────────────────────────
            quantiles = _forecast_to_quantiles(forecast, cfg.horizon_steps, QUANTILE_LEVELS)
            if quantiles is None:
                continue

            if metric_mode == "final":
                step   = cfg.horizon_steps - 1
                gt_val = float(target_vals[step])
                for (q_lo, q_hi) in PI_PAIRS:
                    lo = float(quantiles[q_lo][step])
                    hi = float(quantiles[q_hi][step])
                    pi_covered[  (q_lo, q_hi)] += int(lo <= gt_val <= hi)
                    pi_width_sum[(q_lo, q_hi)] += hi - lo
            else:
                valid_len = min(cfg.horizon_steps, len(target_vals))
                gt_arr = target_vals[:valid_len]
                for (q_lo, q_hi) in PI_PAIRS:
                    lo_arr = quantiles[q_lo][:valid_len]
                    hi_arr = quantiles[q_hi][:valid_len]
                    pi_covered[  (q_lo, q_hi)] += int(np.mean((lo_arr <= gt_arr) & (gt_arr <= hi_arr)) >= 0.5)
                    pi_width_sum[(q_lo, q_hi)] += float(np.mean(hi_arr - lo_arr))

            pi_windows += 1

        batch_entries = []
        batch_targets = []

    for start in range(0, len(df) - cfg.context_steps - cfg.horizon_steps + 1, stride_steps):
        if int(contiguous_runs[start]) < required_gaps:
            continue
        end        = start + cfg.context_steps
        target_end = end   + cfg.horizon_steps

        history_values = values[start:end].astype("float32")
        target_values  = values[end:target_end].astype("float32")

        window_id  = f"{participant_id}__ctx{cfg.context_hours}h__hor{cfg.horizon_minutes}m__{start}"
        start_ts   = pd.to_datetime(timestamps[start])
        if getattr(start_ts, "tzinfo", None) is not None:
            start_ts = start_ts.tz_localize(None)

        batch_entries.append({"item_id": window_id, "start": start_ts, "target": history_values})
        batch_targets.append(target_values)
        if len(batch_entries) >= batch_size:
            flush_batch()

    flush_batch()

    # Compute rates / means
    pi_coverage = {}
    pi_width    = {}
    for pair in PI_PAIRS:
        if pi_windows > 0:
            pi_coverage[pair] = pi_covered[pair]   / pi_windows
            pi_width[pair]    = pi_width_sum[pair]  / pi_windows
        else:
            pi_coverage[pair] = float("nan")
            pi_width[pair]    = float("nan")

    return total_sse, total_ae, total_points, total_windows, pi_coverage, pi_width, pi_windows


# ── Unchanged helpers ──────────────────────────────────────────────────────

def load_existing_metrics(output_path: Path) -> pd.DataFrame:
    if not output_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(output_path)
    except Exception as exc:
        LOGGER.warning("Failed to load existing metrics from %s (%s)", output_path, exc)
        return pd.DataFrame()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Zero-shot: evaluate Moirai2 with probabilistic (quantile) output on 5-minute CGM data. "
            "Results saved to ./quantile."
        )
    )
    parser.add_argument("--task", choices=["eval", "latest"], default="eval")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "test"])
    parser.add_argument("--data-root-train", type=Path, default=DATA_SPLITS["train"])
    parser.add_argument("--data-root-test",  type=Path, default=DATA_SPLITS["test"])
    parser.add_argument("--data-source", choices=["csv", "hf"], default="csv")
    parser.add_argument("--hf-name", type=str, default=DEFAULT_HF_NAME)
    parser.add_argument("--hf-train-split", type=str, default="train")
    parser.add_argument("--hf-test-split",  type=str, default="test")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--batch-size",   type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--stride-steps", type=int, default=1)
    parser.add_argument("--metric-mode",  choices=["final", "all"], default="final")
    parser.add_argument("--latest-requires-truth", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args     = parse_args()
    log_path = setup_logging()
    LOGGER.info("Logging to %s", log_path)
    LOGGER.info("Results root: %s", RESULTS_ROOT)
    LOGGER.info("Quantile levels: %s", QUANTILE_LEVELS)
    ensure_results_root()

    split_roots = {
        "train": args.data_root_train,
        "test":  args.data_root_test,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info("Loading Moirai2 module '%s' (%s)", args.model_name, device)
    module = Moirai2Module.from_pretrained(args.model_name)
    module = module.to(device)
    module.eval()

    predictor_cache: Dict[Tuple[int, int], PyTorchPredictor] = {}

    configs: List[ForecastConfig] = []
    for context_hours in INPUT_WINDOWS_HOURS:
        for horizon_minutes in HORIZONS_MINUTES:
            configs.append(ForecastConfig(
                context_hours   = context_hours,
                horizon_minutes = horizon_minutes,
                context_steps   = context_steps_from_hours(context_hours),
                horizon_steps   = horizon_steps_from_minutes(horizon_minutes),
            ))

    allowed = set(args.datasets) if args.datasets is not None else None

    for split_name in args.splits:
        output_suffix = "metrics" if args.task == "eval" else "latest_predictions"

        if args.data_source == "hf":
            hf_split = args.hf_train_split if split_name == "train" else args.hf_test_split
            subjects_by_dataset: Dict[str, Dict[str, pd.DataFrame]] = {}
            for dataset_name, subject_id, df in iter_glucofm_subjects_from_hf(
                args.hf_name, hf_split, datasets=allowed,
            ):
                subjects_by_dataset.setdefault(dataset_name, {})[subject_id] = df

            if not subjects_by_dataset:
                LOGGER.warning("No HF datasets found for split '%s' (skipping).", split_name)
                continue

            dataset_iter = ((name, subjects) for name, subjects in sorted(subjects_by_dataset.items()))
        else:
            data_root = split_roots[split_name]
            if not data_root.exists():
                LOGGER.warning("Split '%s' not found at %s (skipping)", split_name, data_root)
                continue
            dataset_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
            if allowed is not None:
                dataset_dirs = [p for p in dataset_dirs if p.name in allowed]
            if not dataset_dirs:
                LOGGER.warning("No dataset folders found in %s for split '%s'", data_root, split_name)
                continue
            dataset_iter = ((p.name, p) for p in dataset_dirs)

        for dataset_name, dataset_payload in dataset_iter:
            output_path    = RESULTS_ROOT / f"{dataset_name}_{split_name}_{output_suffix}.csv"
            cache_key_cols = (
                ("participant_id", "context_hours", "horizon_minutes", "stride_steps", "metric_mode")
                if args.task == "eval"
                else ("participant_id", "context_hours", "horizon_minutes", "latest_requires_truth")
            )
            completed_keys: Set[Tuple[object, ...]] = set()
            if not args.overwrite:
                completed_keys = load_completed_keys(output_path, key_cols=cache_key_cols)
                if completed_keys:
                    LOGGER.info(
                        "%s (%s): found %d existing rows, skipping unless --overwrite.",
                        dataset_name, split_name, len(completed_keys),
                    )

            records: List[Dict[str, object]] = []

            subject_groups = dataset_payload if args.data_source == "hf" else collect_subject_groups(dataset_payload)
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

                for cfg in configs:
                    predictor_key = (cfg.context_steps, cfg.horizon_steps)
                    predictor = predictor_cache.get(predictor_key)
                    if predictor is None:
                        predictor = build_predictor(
                            module, cfg.context_steps, cfg.horizon_steps,
                            device, batch_size=args.batch_size,
                        )
                        predictor_cache[predictor_key] = predictor

                    if args.task == "latest":
                        key = (participant_id, cfg.context_hours, cfg.horizon_minutes, bool(args.latest_requires_truth))
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
                        end     = start + cfg.context_steps
                        history = df["value"].iloc[start:end].to_numpy(dtype="float32")
                        history_start_ts = pd.to_datetime(timestamps[start])

                        entry = ListDataset(
                            [{"item_id": participant_id, "start": history_start_ts, "target": history}],
                            freq=FREQ,
                        )
                        try:
                            with torch.inference_mode():
                                forecast = next(iter(predictor.predict(entry)))
                            preds     = _forecast_to_array(forecast, cfg.horizon_steps)
                            quantiles = _forecast_to_quantiles(forecast, cfg.horizon_steps, QUANTILE_LEVELS)
                        except Exception as exc:
                            LOGGER.warning("%s: latest prediction failure (%s)", participant_id, exc)
                            continue

                        pred_final = float(preds[cfg.horizon_steps - 1])
                        pred_ts    = pd.to_datetime(timestamps[end - 1]) + pd.to_timedelta(cfg.horizon_minutes, unit="m")
                        true_final: Optional[float] = None
                        if args.latest_requires_truth:
                            true_final = float(df["value"].iloc[end + cfg.horizon_steps - 1])

                        rec = {
                            "dataset": dataset_name, "split": split_name,
                            "participant_id": participant_id,
                            "context_hours": cfg.context_hours, "horizon_minutes": cfg.horizon_minutes,
                            "context_steps": cfg.context_steps, "horizon_steps": cfg.horizon_steps,
                            "step_minutes": STEP_MINUTES, "freq": FREQ,
                            "prediction_timestamp": pred_ts,
                            "prediction_value": pred_final,
                            "truth_value": true_final,
                            "task": "latest",
                            "latest_requires_truth": bool(args.latest_requires_truth),
                        }
                        # Attach quantile values at the final horizon step
                        if quantiles is not None:
                            step = cfg.horizon_steps - 1
                            for q in QUANTILE_LEVELS:
                                rec[f"q{int(q*100)}"] = float(quantiles[q][step])
                        records.append(rec)

                    else:  # eval
                        stride_steps = cfg.context_steps if args.stride_steps <= 0 else args.stride_steps
                        key = (participant_id, cfg.context_hours, cfg.horizon_minutes,
                               int(stride_steps), str(args.metric_mode))
                        if key in completed_keys:
                            continue

                        sse, sae, points, windows, pi_coverage, pi_width, pi_windows = evaluate_subject(
                            predictor, participant_id, df, cfg,
                            stride_steps=stride_steps,
                            batch_size=args.batch_size,
                            metric_mode=args.metric_mode,
                        )
                        if points == 0 or windows == 0:
                            continue

                        rmse = math.sqrt(sse / points)
                        mae  = sae / points

                        rec = {
                            "dataset": dataset_name, "split": split_name,
                            "participant_id": participant_id,
                            "context_hours": cfg.context_hours, "horizon_minutes": cfg.horizon_minutes,
                            "context_steps": cfg.context_steps, "horizon_steps": cfg.horizon_steps,
                            "step_minutes": STEP_MINUTES, "freq": FREQ,
                            "stride_steps": stride_steps, "metric_mode": args.metric_mode,
                            "rmse": rmse, "mae": mae, "windows": windows,
                            "task": "eval",
                            # ── PI metrics ─────────────────────────────
                            "PI80_coverage": pi_coverage.get((0.1,  0.9),  float("nan")),
                            "PI80_width":    pi_width.get(   (0.1,  0.9),  float("nan")),
                            "PI50_coverage": pi_coverage.get((0.25, 0.75), float("nan")),
                            "PI50_width":    pi_width.get(   (0.25, 0.75), float("nan")),
                            "pi_windows":    pi_windows,
                        }
                        records.append(rec)

            new_df = pd.DataFrame(records)
            if new_df.empty:
                LOGGER.info("%s (%s): no new rows to write.", dataset_name, split_name) if output_path.exists() \
                    else LOGGER.warning("%s (%s): no metrics to write.", dataset_name, split_name)
                continue

            desired_order = [
                "dataset", "split", "participant_id",
                "context_hours", "horizon_minutes", "context_steps", "horizon_steps",
                "step_minutes", "freq", "task",
                "stride_steps", "metric_mode",
                "latest_requires_truth", "prediction_timestamp", "prediction_value", "truth_value",
                "rmse", "mae", "windows",
                "PI80_coverage", "PI80_width",
                "PI50_coverage", "PI50_width",
                "pi_windows",
                "q10", "q25", "q50", "q75", "q90",
            ]

            if args.overwrite:
                final_df      = new_df
                ordered_cols  = [c for c in desired_order  if c in final_df.columns]
                remaining     = [c for c in final_df.columns if c not in ordered_cols]
                final_df      = final_df[ordered_cols + remaining]
                final_df      = final_df.sort_values(["participant_id", "context_hours", "horizon_minutes"]).reset_index(drop=True)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                final_df.to_csv(output_path, index=False)
                LOGGER.info("Saved to %s (%d rows)", output_path, len(final_df))
            else:
                written = upsert_csv_rows(
                    output_path, new_df,
                    key_cols=cache_key_cols,
                    desired_order=desired_order,
                    sort_by=["participant_id", "context_hours", "horizon_minutes"],
                )
                LOGGER.info("Saved to %s (+%d rows)", output_path, written)


if __name__ == "__main__":
    main()
