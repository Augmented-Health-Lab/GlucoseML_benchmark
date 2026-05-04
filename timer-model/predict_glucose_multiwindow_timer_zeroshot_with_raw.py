import argparse
import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from results_cache import load_completed_keys, upsert_csv_rows  # noqa: E402
from glucofm_data import (  # noqa: E402
    DEFAULT_HF_NAME,
    iter_glucofm_subjects_from_hf,
    parse_timestamp_series,
)


MODEL_NAME = str(Path(__file__).resolve().parent)
DATA_SPLITS = {
    "train": Path(__file__).resolve().parent.parent / "hf_cache" / "train",
    "test": Path(__file__).resolve().parent.parent / "hf_cache" / "test",
}

# Timer produces point forecasts only (no quantile head), so only raw + metrics are written.
RESULTS_ROOT = Path(__file__).resolve().parent / "multi_horizon_results_timer_zeroshot_with_raw"

STEP_MINUTES = 5
FREQ = "5min"
STEP_DELTA = np.timedelta64(STEP_MINUTES, "m")

INPUT_WINDOWS_HOURS = [1, 4, 8, 12, 16, 24]
HORIZONS_MINUTES = [15, 30, 60, 90]

DEFAULT_BATCH_SIZE = 8

LOGGER = logging.getLogger(__name__)


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
    log_path = RESULTS_ROOT / f"predict_glucose_multiwindow_timer_zeroshot_with_raw_{timestamp}.log"
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
    context_hours: int
    horizon_minutes: int
    context_steps: int
    horizon_steps: int


def _effective_context_steps(context_steps: int, patch_len: int) -> int:
    if patch_len <= 0:
        return 0
    return (context_steps // patch_len) * patch_len


def _batched_timer_forecast(
    model: AutoModelForCausalLM,
    histories: List[torch.Tensor],
    horizon_steps: int,
    device: torch.device,
) -> torch.Tensor:
    inputs = torch.stack(histories).to(device)
    with torch.inference_mode():
        outputs = model(
            input_ids=inputs,
            max_output_length=horizon_steps,
            revin=True,
            return_dict=True,
            use_cache=False,
        )
    preds = outputs.logits
    return preds.detach().cpu()


@dataclass
class SubjectEvalResult:
    sse: float = 0.0
    ae: float = 0.0
    points: int = 0
    windows: int = 0
    model_context_steps: int = 0
    raw_predictions: List[Dict] = field(default_factory=list)


def evaluate_subject(
    model: AutoModelForCausalLM,
    participant_id: str,
    df: pd.DataFrame,
    cfg: ForecastConfig,
    stride_steps: int,
    batch_size: int,
    metric_mode: str,
    patch_len: int,
    device: torch.device,
) -> SubjectEvalResult:
    result = SubjectEvalResult()
    if len(df) < cfg.context_steps + cfg.horizon_steps:
        return result

    model_context_steps = _effective_context_steps(cfg.context_steps, patch_len)
    result.model_context_steps = model_context_steps
    if model_context_steps < patch_len:
        return result

    values = df["value"].to_numpy(dtype="float32")
    timestamps = normalize_timestamps(df["timestamp"])
    contiguous_runs = consecutive_gap_counts(timestamps)
    total_len = cfg.context_steps + cfg.horizon_steps
    required_gaps = total_len - 1

    batch_histories: List[torch.Tensor] = []
    batch_targets: List[np.ndarray] = []
    batch_starts: List[int] = []

    def flush_batch() -> None:
        nonlocal batch_histories, batch_targets, batch_starts
        if not batch_histories:
            return
        try:
            preds = _batched_timer_forecast(model, batch_histories, cfg.horizon_steps, device)
        except Exception as exc:
            LOGGER.warning("%s: prediction failure for %d windows (%s)", participant_id, len(batch_histories), exc)
            batch_histories = []
            batch_targets = []
            batch_starts = []
            return

        for pred_vec, target_vals, start_idx in zip(preds, batch_targets, batch_starts):
            pred_vec = pred_vec.to(torch.float32)

            if metric_mode == "final":
                step = cfg.horizon_steps - 1
                pred_val = float(pred_vec[step].item())
                target_val = float(target_vals[step])
                if math.isnan(pred_val) or math.isnan(target_val):
                    continue
                err = pred_val - target_val
                result.sse += float(err * err)
                result.ae += float(abs(err))
                result.points += 1
                result.windows += 1
                result.raw_predictions.append({
                    "window_start": int(start_idx),
                    "horizon_step": int(cfg.horizon_steps),
                    "prediction":   pred_val,
                    "ground_truth": target_val,
                })
                continue

            valid_len = min(int(pred_vec.numel()), int(len(target_vals)), int(cfg.horizon_steps))
            if valid_len == 0:
                continue
            pred_arr = pred_vec[:valid_len].cpu().numpy().astype("float32")
            targets = target_vals[:valid_len]
            if np.isnan(pred_arr).any() or np.isnan(targets).any():
                continue
            errors = pred_arr - targets
            result.sse += float(np.sum(errors ** 2))
            result.ae += float(np.sum(np.abs(errors)))
            result.points += int(valid_len)
            result.windows += 1

            for step_idx in range(valid_len):
                result.raw_predictions.append({
                    "window_start": int(start_idx),
                    "horizon_step": int(step_idx + 1),
                    "prediction":   float(pred_arr[step_idx]),
                    "ground_truth": float(targets[step_idx]),
                })

        batch_histories = []
        batch_targets = []
        batch_starts = []

    for start in range(0, len(df) - cfg.context_steps - cfg.horizon_steps + 1, stride_steps):
        if int(contiguous_runs[start]) < required_gaps:
            continue
        end = start + cfg.context_steps
        target_end = end + cfg.horizon_steps

        history_start = end - model_context_steps
        if history_start < start:
            history_start = start
        history_values = values[history_start:end].astype("float32")
        if int(history_values.shape[0]) != int(model_context_steps):
            continue

        target_values = values[end:target_end].astype("float32")
        if np.isnan(history_values).any() or np.isnan(target_values).any():
            continue

        batch_histories.append(torch.tensor(history_values, dtype=torch.float32))
        batch_targets.append(target_values)
        batch_starts.append(start)
        if len(batch_histories) >= batch_size:
            flush_batch()

    flush_batch()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Zero-shot: evaluate Timer on 5-minute CGM data for multiple context windows and horizons. "
            "Writes aggregate metrics and per-datapoint raw predictions (Timer is point-forecast only, "
            "no quantile output)."
        )
    )
    parser.add_argument("--task", choices=["eval", "latest"], default="eval")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--splits", nargs="+", default=["test"], choices=["train", "test"])
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
    parser.add_argument(
        "--context-hours",
        type=int,
        nargs="*",
        default=None,
        help=f"Context window(s) in hours. Default: all of {INPUT_WINDOWS_HOURS}.",
    )
    parser.add_argument(
        "--horizons-minutes",
        type=int,
        nargs="*",
        default=None,
        help=f"Forecast horizon(s) in minutes. Default: all of {HORIZONS_MINUTES}.",
    )
    parser.add_argument("--latest-requires-truth", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--save-raw-predictions",
        dest="save_raw_predictions",
        action="store_true",
        default=True,
        help="Write per-datapoint predictions to raw_predictions/ (default: enabled, eval task only).",
    )
    parser.add_argument(
        "--no-save-raw-predictions",
        dest="save_raw_predictions",
        action="store_false",
    )
    return parser.parse_args()


METRICS_DESIRED_ORDER = [
    "dataset", "split", "participant_id",
    "context_hours", "horizon_minutes", "context_steps", "model_context_steps",
    "horizon_steps", "step_minutes", "freq", "task",
    "stride_steps", "metric_mode",
    "latest_requires_truth", "prediction_timestamp", "prediction_value", "truth_value",
    "rmse", "mae", "windows",
]

RAW_DESIRED_ORDER = [
    "dataset", "split", "participant_id",
    "context_hours", "horizon_minutes", "stride_steps", "metric_mode",
    "window_start", "horizon_step", "prediction", "ground_truth",
]


def main() -> None:
    args = parse_args()
    log_path = setup_logging()
    LOGGER.info("Logging to %s", log_path)
    LOGGER.info("Results root: %s", RESULTS_ROOT)
    ensure_results_root()

    raw_dir = RESULTS_ROOT / "raw_predictions"
    if args.save_raw_predictions and args.task == "eval":
        raw_dir.mkdir(parents=True, exist_ok=True)

    split_roots = {"train": args.data_root_train, "test": args.data_root_test}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Loading Timer model '%s' (%s)", args.model_name, device)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True).to(device)
    model.eval()
    patch_len = int(getattr(model.config, "input_token_len", 0))
    LOGGER.info("Timer input_token_len=%d", patch_len)

    context_hours_list   = args.context_hours   if args.context_hours   else INPUT_WINDOWS_HOURS
    horizons_minutes_list = args.horizons_minutes if args.horizons_minutes else HORIZONS_MINUTES
    LOGGER.info("Context hours: %s", context_hours_list)
    LOGGER.info("Horizons minutes: %s", horizons_minutes_list)

    configs: List[ForecastConfig] = []
    for context_hours in context_hours_list:
        for horizon_minutes in horizons_minutes_list:
            configs.append(ForecastConfig(
                context_hours=context_hours,
                horizon_minutes=horizon_minutes,
                context_steps=context_steps_from_hours(context_hours),
                horizon_steps=horizon_steps_from_minutes(horizon_minutes),
            ))

    allowed = set(args.datasets) if args.datasets is not None else None

    metrics_key_cols = (
        ("participant_id", "context_hours", "horizon_minutes", "stride_steps", "metric_mode")
        if args.task == "eval"
        else ("participant_id", "context_hours", "horizon_minutes", "latest_requires_truth")
    )

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
            metrics_path = RESULTS_ROOT / f"{dataset_name}_{split_name}_{output_suffix}.csv"
            raw_path     = raw_dir / f"{dataset_name}_{split_name}_raw_predictions.csv"

            completed_metrics: Set[Tuple[object, ...]] = set()
            completed_raw:     Set[Tuple[object, ...]] = set()
            if args.overwrite:
                for path in (metrics_path, raw_path):
                    if path.exists():
                        try:
                            path.unlink()
                        except Exception as exc:
                            LOGGER.warning("Failed to remove %s (%s)", path, exc)
            else:
                completed_metrics = load_completed_keys(metrics_path, key_cols=metrics_key_cols)
                if completed_metrics:
                    LOGGER.info(
                        "%s (%s): found %d existing metrics rows.",
                        dataset_name, split_name, len(completed_metrics),
                    )
                if args.task == "eval" and args.save_raw_predictions:
                    completed_raw = load_completed_keys(
                        raw_path,
                        key_cols=("participant_id", "context_hours", "horizon_minutes", "stride_steps", "metric_mode"),
                    )

            metrics_records: List[Dict[str, object]] = []
            raw_records:     List[Dict[str, object]] = []

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
                    if args.task == "latest":
                        key = (participant_id, cfg.context_hours, cfg.horizon_minutes, bool(args.latest_requires_truth))
                        if key in completed_metrics:
                            continue

                        model_context_steps = _effective_context_steps(cfg.context_steps, patch_len)
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
                        if model_context_steps < patch_len:
                            continue
                        history_start = end - model_context_steps
                        if history_start < start:
                            history_start = start
                        history = df["value"].iloc[history_start:end].to_numpy(dtype="float32")
                        if np.isnan(history).any() or int(history.shape[0]) != int(model_context_steps):
                            continue
                        try:
                            preds = _batched_timer_forecast(
                                model,
                                [torch.tensor(history, dtype=torch.float32)],
                                cfg.horizon_steps, device,
                            )[0].numpy()
                        except Exception as exc:
                            LOGGER.warning("%s: latest prediction failure (%s)", participant_id, exc)
                            continue

                        pred_final = float(preds[cfg.horizon_steps - 1])
                        pred_ts = pd.to_datetime(timestamps[end - 1]) + pd.to_timedelta(cfg.horizon_minutes, unit="m")
                        true_final: Optional[float] = None
                        if args.latest_requires_truth:
                            true_final = float(df["value"].iloc[end + cfg.horizon_steps - 1])

                        metrics_records.append({
                            "dataset": dataset_name, "split": split_name,
                            "participant_id": participant_id,
                            "context_hours": cfg.context_hours, "horizon_minutes": cfg.horizon_minutes,
                            "context_steps": cfg.context_steps, "model_context_steps": model_context_steps,
                            "horizon_steps": cfg.horizon_steps,
                            "step_minutes": STEP_MINUTES, "freq": FREQ,
                            "prediction_timestamp": pred_ts,
                            "prediction_value": pred_final,
                            "truth_value": true_final,
                            "task": "latest",
                            "latest_requires_truth": bool(args.latest_requires_truth),
                        })
                        continue

                    # task == "eval"
                    stride_steps = cfg.context_steps if args.stride_steps <= 0 else args.stride_steps
                    key = (participant_id, cfg.context_hours, cfg.horizon_minutes,
                           int(stride_steps), str(args.metric_mode))
                    metrics_done = key in completed_metrics
                    raw_done     = (not args.save_raw_predictions) or (key in completed_raw)
                    if metrics_done and raw_done:
                        continue

                    result = evaluate_subject(
                        model, participant_id, df, cfg,
                        stride_steps=stride_steps,
                        batch_size=args.batch_size,
                        metric_mode=args.metric_mode,
                        patch_len=patch_len,
                        device=device,
                    )
                    if result.points == 0 or result.windows == 0:
                        continue

                    rmse = math.sqrt(result.sse / result.points)
                    mae  = result.ae / result.points

                    common_key_cols = {
                        "dataset": dataset_name, "split": split_name,
                        "participant_id": participant_id,
                        "context_hours": cfg.context_hours, "horizon_minutes": cfg.horizon_minutes,
                        "stride_steps": stride_steps, "metric_mode": args.metric_mode,
                    }

                    if not metrics_done:
                        metrics_records.append({
                            **common_key_cols,
                            "context_steps": cfg.context_steps,
                            "model_context_steps": result.model_context_steps,
                            "horizon_steps": cfg.horizon_steps,
                            "step_minutes": STEP_MINUTES, "freq": FREQ,
                            "rmse": rmse, "mae": mae, "windows": result.windows,
                            "task": "eval",
                        })

                    if args.save_raw_predictions and not raw_done:
                        for rp in result.raw_predictions:
                            raw_records.append({**common_key_cols, **rp})

            # --- flush per-dataset ---
            if metrics_records:
                new_df = pd.DataFrame(metrics_records)
                written = upsert_csv_rows(
                    metrics_path, new_df,
                    key_cols=metrics_key_cols,
                    desired_order=METRICS_DESIRED_ORDER,
                    sort_by=["participant_id", "context_hours", "horizon_minutes"],
                )
                LOGGER.info("Saved metrics to %s (+%d rows)", metrics_path, written)
            elif metrics_path.exists():
                LOGGER.info("%s (%s): no new metrics rows.", dataset_name, split_name)
            else:
                LOGGER.warning("%s (%s): no metrics to write.", dataset_name, split_name)

            if args.task == "eval" and args.save_raw_predictions and raw_records:
                new_raw_df = pd.DataFrame(raw_records)
                raw_key_cols = (
                    "participant_id", "context_hours", "horizon_minutes",
                    "stride_steps", "metric_mode", "window_start", "horizon_step",
                )
                written = upsert_csv_rows(
                    raw_path, new_raw_df,
                    key_cols=raw_key_cols,
                    desired_order=RAW_DESIRED_ORDER,
                    sort_by=["participant_id", "context_hours", "horizon_minutes", "window_start", "horizon_step"],
                )
                LOGGER.info("Saved raw predictions to %s (+%d rows)", raw_path, written)


if __name__ == "__main__":
    main()
