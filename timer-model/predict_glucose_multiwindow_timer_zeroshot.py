import argparse
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM


MODEL_NAME = str(Path(__file__).resolve().parent)
DATA_SPLITS = {
    "train": Path(__file__).resolve().parent.parent / "training_dataset",
    "test": Path(__file__).resolve().parent.parent / "test_dataset",
}

RESULTS_ROOT = Path(__file__).resolve().parent / "multi_horizon_results_timer_zeroshot"

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

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
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
    log_path = RESULTS_ROOT / f"predict_glucose_multiwindow_timer_zeroshot_{timestamp}.log"
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
    ts = pd.to_datetime(timestamps, errors="coerce", utc=True).dt.tz_convert(None)
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
) -> Tuple[float, float, int, int, int]:
    if len(df) < cfg.context_steps + cfg.horizon_steps:
        return 0.0, 0.0, 0, 0, 0

    model_context_steps = _effective_context_steps(cfg.context_steps, patch_len)
    if model_context_steps < patch_len:
        return 0.0, 0.0, 0, 0, model_context_steps

    values = df["value"].to_numpy(dtype="float32")
    timestamps = normalize_timestamps(df["timestamp"])
    contiguous_runs = consecutive_gap_counts(timestamps)
    total_len = cfg.context_steps + cfg.horizon_steps
    required_gaps = total_len - 1

    total_sse = 0.0
    total_ae = 0.0
    total_points = 0
    total_windows = 0

    batch_histories: List[torch.Tensor] = []
    batch_targets: List[np.ndarray] = []

    def flush_batch() -> None:
        nonlocal total_sse, total_ae, total_points, total_windows, batch_histories, batch_targets
        if not batch_histories:
            return
        try:
            preds = _batched_timer_forecast(model, batch_histories, cfg.horizon_steps, device)
        except Exception as exc:
            LOGGER.warning("%s: prediction failure for %d windows (%s)", participant_id, len(batch_histories), exc)
            batch_histories = []
            batch_targets = []
            return

        for pred_vec, target_vals in zip(preds, batch_targets):
            pred_vec = pred_vec.to(torch.float32)
            if metric_mode == "final":
                pred_val = float(pred_vec[cfg.horizon_steps - 1].item())
                target_val = float(target_vals[cfg.horizon_steps - 1])
                if math.isnan(pred_val) or math.isnan(target_val):
                    continue
                err = pred_val - target_val
                total_sse += float(err * err)
                total_ae += float(abs(err))
                total_points += 1
                total_windows += 1
                continue

            valid_len = min(int(pred_vec.numel()), int(len(target_vals)), int(cfg.horizon_steps))
            if valid_len == 0:
                continue
            pred_arr = pred_vec[:valid_len].cpu().numpy().astype("float32")
            targets = target_vals[:valid_len]
            if np.isnan(pred_arr).any() or np.isnan(targets).any():
                continue
            errors = pred_arr - targets
            total_sse += float(np.sum(errors**2))
            total_ae += float(np.sum(np.abs(errors)))
            total_points += int(valid_len)
            total_windows += 1

        batch_histories = []
        batch_targets = []

    for start in range(0, len(df) - cfg.context_steps - cfg.horizon_steps + 1, stride_steps):
        if int(contiguous_runs[start]) < required_gaps:
            continue
        end = start + cfg.context_steps
        target_end = end + cfg.horizon_steps

        # Timer requires sequence length to be a multiple of `input_token_len`.
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
        if len(batch_histories) >= batch_size:
            flush_batch()

    flush_batch()
    return total_sse, total_ae, total_points, total_windows, model_context_steps


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
        description="Zero-shot: evaluate Timer on 5-minute CGM data for multiple context windows and horizons."
    )
    parser.add_argument(
        "--task",
        choices=["eval", "latest"],
        default="eval",
        help="Run rolling-window evaluation (eval) or produce one latest forecast per participant (latest).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help="HuggingFace model id or local directory to load.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        choices=["train", "test"],
        help="Which dataset splits to evaluate.",
    )
    parser.add_argument(
        "--data-root-train",
        type=Path,
        default=DATA_SPLITS["train"],
        help="Root directory containing dataset subfolders for the train split.",
    )
    parser.add_argument(
        "--data-root-test",
        type=Path,
        default=DATA_SPLITS["test"],
        help="Root directory containing dataset subfolders for the test split.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset folder names to include (default: all).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Inference batch size.",
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
        "--latest-requires-truth",
        action="store_true",
        help="For --task latest, pick the latest window that still has ground truth for the horizon.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and overwrite existing results instead of skipping completed rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = setup_logging()
    LOGGER.info("Logging to %s", log_path)
    ensure_results_root()

    split_roots = {
        "train": args.data_root_train,
        "test": args.data_root_test,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Loading Timer model '%s' (%s)", args.model_name, device)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True).to(device)
    model.eval()
    patch_len = int(getattr(model.config, "input_token_len", 0))
    LOGGER.info("Timer input_token_len=%d", patch_len)

    configs: List[ForecastConfig] = []
    for context_hours in INPUT_WINDOWS_HOURS:
        for horizon_minutes in HORIZONS_MINUTES:
            configs.append(
                ForecastConfig(
                    context_hours=context_hours,
                    horizon_minutes=horizon_minutes,
                    context_steps=context_steps_from_hours(context_hours),
                    horizon_steps=horizon_steps_from_minutes(horizon_minutes),
                )
            )

    for split_name in args.splits:
        data_root = split_roots[split_name]
        if not data_root.exists():
            LOGGER.warning("Split '%s' not found at %s (skipping)", split_name, data_root)
            continue

        dataset_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
        if args.datasets is not None:
            allowed = set(args.datasets)
            dataset_dirs = [p for p in dataset_dirs if p.name in allowed]
        if not dataset_dirs:
            LOGGER.warning("No dataset folders found in %s for split '%s'", data_root, split_name)
            continue

        for dataset_dir in dataset_dirs:
            output_suffix = "metrics" if args.task == "eval" else "latest_predictions"
            output_path = RESULTS_ROOT / f"{dataset_dir.name}_{split_name}_{output_suffix}.csv"
            existing_df = pd.DataFrame()
            completed_keys: Set[Tuple[str, int, int]] = set()
            if not args.overwrite:
                existing_df = load_existing_metrics(output_path)
                required_cols = {"participant_id", "context_hours", "horizon_minutes"}
                if not existing_df.empty and required_cols.issubset(existing_df.columns):
                    completed_keys = set(
                        (
                            str(row["participant_id"]),
                            int(row["context_hours"]),
                            int(row["horizon_minutes"]),
                        )
                        for _, row in existing_df.iterrows()
                    )
                    LOGGER.info(
                        "%s (%s split): found %d existing rows, skipping unless --overwrite is set.",
                        dataset_dir.name,
                        split_name,
                        len(completed_keys),
                    )
                elif not existing_df.empty:
                    LOGGER.warning(
                        "%s (%s split): existing results missing key columns, ignoring cached content.",
                        dataset_dir.name,
                        split_name,
                    )
                    existing_df = pd.DataFrame()

            subject_groups = collect_subject_groups(dataset_dir)
            records: List[Dict[str, object]] = []

            LOGGER.info("%s (%s split): %d participants", dataset_dir.name, split_name, len(subject_groups))
            for participant_id, csv_paths in subject_groups.items():
                try:
                    df = load_subject(csv_paths)
                except (ValueError, pd.errors.EmptyDataError) as exc:
                    LOGGER.warning("%s / %s: skipped (%s)", dataset_dir.name, participant_id, exc)
                    continue

                for cfg in configs:
                    key = (participant_id, cfg.context_hours, cfg.horizon_minutes)
                    if key in completed_keys:
                        continue

                    model_context_steps = _effective_context_steps(cfg.context_steps, patch_len)

                    if args.task == "latest":
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
                                cfg.horizon_steps,
                                device,
                            )[0].numpy()
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
                                "dataset": dataset_dir.name,
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
                    else:
                        stride_steps = cfg.context_steps if args.stride_steps <= 0 else args.stride_steps
                        sse, sae, points, windows, model_context_steps = evaluate_subject(
                            model,
                            participant_id,
                            df,
                            cfg,
                            stride_steps=stride_steps,
                            batch_size=args.batch_size,
                            metric_mode=args.metric_mode,
                            patch_len=patch_len,
                            device=device,
                        )
                        if points == 0 or windows == 0:
                            continue

                        rmse = math.sqrt(sse / points)
                        mae = sae / points
                        records.append(
                            {
                                "dataset": dataset_dir.name,
                                "split": split_name,
                                "participant_id": participant_id,
                                "context_hours": cfg.context_hours,
                                "horizon_minutes": cfg.horizon_minutes,
                                "context_steps": cfg.context_steps,
                                "model_context_steps": model_context_steps,
                                "horizon_steps": cfg.horizon_steps,
                                "step_minutes": STEP_MINUTES,
                                "freq": FREQ,
                                "stride_steps": stride_steps,
                                "metric_mode": args.metric_mode,
                                "rmse": rmse,
                                "mae": mae,
                                "windows": windows,
                                "task": "eval",
                            }
                        )

            new_df = pd.DataFrame(records)
            if args.overwrite or existing_df.empty:
                final_df = new_df
            else:
                final_df = pd.concat([existing_df, new_df], ignore_index=True)
                final_df = final_df.drop_duplicates(
                    subset=["participant_id", "context_hours", "horizon_minutes"], keep="last"
                ).reset_index(drop=True)

            if final_df.empty:
                LOGGER.warning("%s (%s split): no metrics to write.", dataset_dir.name, split_name)
                continue

            desired_order = [
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
                "windows",
            ]
            ordered_cols = [col for col in desired_order if col in final_df.columns]
            remaining_cols = [col for col in final_df.columns if col not in ordered_cols]
            final_df = final_df[ordered_cols + remaining_cols]
            final_df = final_df.sort_values(["participant_id", "context_hours", "horizon_minutes"]).reset_index(drop=True)
            final_df.to_csv(output_path, index=False)
            LOGGER.info("Saved metrics to %s (%d rows)", output_path, len(final_df))


if __name__ == "__main__":
    main()
