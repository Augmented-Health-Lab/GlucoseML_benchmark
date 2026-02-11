from __future__ import annotations

import argparse
import logging
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
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
    # Train on the mixed training pool by default.
    "train": Path(__file__).resolve().parent.parent / "hf_cache" / "train" / "mixed",
    "test": Path(__file__).resolve().parent.parent / "hf_cache" / "test",
}

DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parent / "multi_horizon_results_timer_fullshot"
DEFAULT_LOG_STEM = "predict_glucose_multiwindow_timer_fullshot"
DEFAULT_SHOT_LABEL = "full-shot"

STEP_MINUTES = 5
FREQ = "5min"
STEP_DELTA = np.timedelta64(STEP_MINUTES, "m")

DEFAULT_EVAL_BATCH_SIZE = 8
DEFAULT_TRAIN_BATCH_SIZE = 16

DEFAULT_EVAL_STRIDE_STEPS = 1
DEFAULT_TRAIN_EPOCHS = 10
DEFAULT_TRAIN_STRIDE_STEPS = 10

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


def ensure_results_root(results_root: Path) -> None:
    results_root.mkdir(parents=True, exist_ok=True)


def save_model_dir(results_root: Path, *, shot_label: str, cfg: ForecastConfig) -> Path:
    return results_root / "saved_models" / f"{shot_label}_ctx{cfg.context_hours}h_hor{cfg.horizon_minutes}m"


def save_trained_model(
    model: AutoModelForCausalLM,
    *,
    results_root: Path,
    shot_label: str,
    cfg: ForecastConfig,
) -> Path:
    out_dir = save_model_dir(results_root, shot_label=shot_label, cfg=cfg)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    # HF-style model; includes config + weights.
    model.save_pretrained(out_dir)
    return out_dir


def setup_logging(results_root: Path, log_stem: str) -> Path:
    ensure_results_root(results_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = results_root / f"{log_stem}_{timestamp}.log"
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


def _effective_context_steps(context_steps: int, patch_len: int) -> int:
    if patch_len <= 0:
        return 0
    return (context_steps // patch_len) * patch_len


@dataclass(frozen=True)
class ForecastConfig:
    context_hours: int
    horizon_minutes: int
    context_steps: int
    horizon_steps: int


@dataclass(frozen=True)
class SeriesCache:
    values: np.ndarray
    timestamps: np.ndarray
    contiguous_runs: np.ndarray


def load_series_cache(df: pd.DataFrame) -> SeriesCache:
    values = df["value"].to_numpy(dtype="float32")
    timestamps = normalize_timestamps(df["timestamp"])
    contiguous_runs = consecutive_gap_counts(timestamps)
    return SeriesCache(values=values, timestamps=timestamps, contiguous_runs=contiguous_runs)


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
    return outputs.logits.detach().to(torch.float32).cpu()


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
            pred_arr = pred_vec[:valid_len].numpy().astype("float32")
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


def build_train_index(
    *,
    series: Dict[str, SeriesCache],
    context_steps: int,
    horizon_steps: int,
    stride_steps: int,
    max_windows: Optional[int],
    seed: int,
) -> List[Tuple[str, int]]:
    total_len = context_steps + horizon_steps
    if total_len <= 0:
        return []
    required_gaps = total_len - 1

    index: List[Tuple[str, int]] = []
    for participant_id, cache in series.items():
        max_start = int(cache.values.shape[0]) - total_len
        if max_start < 0:
            continue
        for start in range(0, max_start + 1, stride_steps):
            if int(cache.contiguous_runs[start]) < required_gaps:
                continue
            index.append((participant_id, start))

    if max_windows is not None and max_windows > 0 and len(index) > max_windows:
        rng = random.Random(seed)
        rng.shuffle(index)
        index = index[: max_windows]
    return index


class WindowDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        series: Dict[str, SeriesCache],
        index: List[Tuple[str, int]],
        *,
        context_steps: int,
        model_context_steps: int,
        horizon_steps: int,
    ) -> None:
        self._series = series
        self._index = index
        self._context_steps = int(context_steps)
        self._model_context_steps = int(model_context_steps)
        self._horizon_steps = int(horizon_steps)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        participant_id, start = self._index[idx]
        cache = self._series[participant_id]
        end = start + self._context_steps
        history_start = end - self._model_context_steps
        past = cache.values[history_start:end]
        future = cache.values[end : end + self._horizon_steps]
        return torch.from_numpy(past.copy()), torch.from_numpy(future.copy())


def finetune_fullshot(
    *,
    model_name: str,
    train_series: Dict[str, SeriesCache],
    cfg: ForecastConfig,
    device: torch.device,
    patch_len: int,
    seed: int,
    train_stride_steps: int,
    train_batch_size: int,
    train_epochs: int,
    max_train_steps: Optional[int],
    max_train_windows: Optional[int],
    lr: float,
    weight_decay: float,
    loss_mode: str,
    log_every: int,
) -> AutoModelForCausalLM:
    if not train_series:
        raise ValueError("No training participants loaded.")
    if train_epochs <= 0:
        raise ValueError("train_epochs must be > 0")
    if train_batch_size <= 0:
        raise ValueError("train_batch_size must be > 0")
    if train_stride_steps <= 0:
        raise ValueError("train_stride_steps must be > 0")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.train()

    model_context_steps = _effective_context_steps(cfg.context_steps, patch_len)
    if model_context_steps < patch_len:
        raise ValueError(f"Timer model_context_steps too small: {model_context_steps} (patch_len={patch_len})")

    train_index = build_train_index(
        series=train_series,
        context_steps=cfg.context_steps,
        horizon_steps=cfg.horizon_steps,
        stride_steps=train_stride_steps,
        max_windows=max_train_windows,
        seed=seed,
    )
    if not train_index:
        raise ValueError("No valid training windows found for this config.")

    dataset = WindowDataset(
        train_series,
        train_index,
        context_steps=cfg.context_steps,
        model_context_steps=model_context_steps,
        horizon_steps=cfg.horizon_steps,
    )
    loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters.")
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    total_steps = 0
    for epoch in range(train_epochs):
        for past, future in loader:
            if max_train_steps is not None and total_steps >= max_train_steps:
                break
            past = past.to(device)
            future = future.to(device)

            outputs = model(
                input_ids=past,
                max_output_length=cfg.horizon_steps,
                revin=True,
                return_dict=True,
                use_cache=False,
            )
            preds = outputs.logits.to(torch.float32)
            if loss_mode == "final":
                preds = preds[:, -1:]
                future = future[:, -1:]
            loss = torch.mean((preds - future) ** 2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            total_steps += 1
            if log_every > 0 and total_steps % log_every == 0:
                LOGGER.info(
                    "finetune step=%d epoch=%d/%d loss=%.6f windows=%d",
                    total_steps,
                    epoch + 1,
                    train_epochs,
                    float(loss.detach().cpu()),
                    len(train_index),
                )

        if max_train_steps is not None and total_steps >= max_train_steps:
            break

    model.eval()
    LOGGER.info("finetune done: total_steps=%d", total_steps)
    return model


def parse_args(
    argv: Optional[Sequence[str]] = None,
    *,
    shot_label: str = DEFAULT_SHOT_LABEL,
    default_eval_stride_steps: int = DEFAULT_EVAL_STRIDE_STEPS,
    default_train_epochs: int = DEFAULT_TRAIN_EPOCHS,
    default_train_stride_steps: int = DEFAULT_TRAIN_STRIDE_STEPS,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"{shot_label.capitalize()}: fine-tune Timer on the mixed training pool (sliding windows) "
            "and evaluate on each dataset's test split."
        )
    )
    parser.add_argument("--model-name", type=str, default=MODEL_NAME, help="HuggingFace model id or local directory.")
    parser.add_argument(
        "--data-root-train",
        type=Path,
        default=DATA_SPLITS["train"],
        help="Directory containing training CSV files (default: hf_cache/train/mixed).",
    )
    parser.add_argument(
        "--data-root-test",
        type=Path,
        default=DATA_SPLITS["test"],
        help="Root directory containing dataset subfolders for the test split (default: hf_cache/test).",
    )
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
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset folder names to include.")
    parser.add_argument("--context-hours", type=int, nargs="*", default=[12], help="Context window hours (default: 12).")
    parser.add_argument(
        "--horizons-minutes",
        type=int,
        nargs="*",
        default=[30],
        help="Forecast horizon in minutes (default: 30).",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device for training/eval: auto/cpu/cuda.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metric-mode", choices=["final", "all"], default="final")
    parser.add_argument(
        "--eval-stride-steps",
        type=int,
        default=default_eval_stride_steps,
        help="Evaluation sliding window stride in steps (0 = context_steps; 1 = 5 minutes).",
    )
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE, help="Inference batch size.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing metrics CSVs.")
    parser.add_argument(
        "--save-trained-model",
        dest="save_trained_model",
        action="store_true",
        default=True,
        help="Save a fine-tuned model per (context,horizon) under results_root/saved_models/ (default: enabled).",
    )
    parser.add_argument(
        "--no-save-trained-model",
        dest="save_trained_model",
        action="store_false",
        help="Disable saving fine-tuned models.",
    )

    parser.add_argument("--train-batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--train-epochs", type=int, default=default_train_epochs)
    parser.add_argument("--train-stride-steps", type=int, default=default_train_stride_steps)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument(
        "--max-train-windows",
        type=int,
        default=None,
        help="Optional cap on number of training windows (randomly sampled).",
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument(
        "--train-loss-mode",
        choices=["final", "all"],
        default="all",
        help="Optimize loss on final horizon point or all horizon points.",
    )
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args(argv)


def resolve_device(arg: str) -> torch.device:
    arg = str(arg).lower()
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arg in {"cpu", "cuda"}:
        if arg == "cuda" and not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but not available; falling back to CPU.")
            return torch.device("cpu")
        return torch.device(arg)
    raise ValueError(f"Unsupported device: {arg!r}")


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    log_stem: str = DEFAULT_LOG_STEM,
    shot_label: str = DEFAULT_SHOT_LABEL,
    default_eval_stride_steps: int = DEFAULT_EVAL_STRIDE_STEPS,
    default_train_epochs: int = DEFAULT_TRAIN_EPOCHS,
    default_train_stride_steps: int = DEFAULT_TRAIN_STRIDE_STEPS,
) -> None:
    args = parse_args(
        argv,
        shot_label=shot_label,
        default_eval_stride_steps=default_eval_stride_steps,
        default_train_epochs=default_train_epochs,
        default_train_stride_steps=default_train_stride_steps,
    )
    log_path = setup_logging(results_root, log_stem)
    LOGGER.info("Logging to %s", log_path)

    device = resolve_device(args.device)
    LOGGER.info("Device: %s", device)

    context_hours = args.context_hours
    horizons_minutes = args.horizons_minutes
    configs = [
        ForecastConfig(
            context_hours=ctx,
            horizon_minutes=hor,
            context_steps=context_steps_from_hours(ctx),
            horizon_steps=horizon_steps_from_minutes(hor),
        )
        for ctx in context_hours
        for hor in horizons_minutes
    ]

    # Load patch_len from the base model config (needed for training/eval windowing).
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    patch_len = int(getattr(base_model.config, "input_token_len", 0))
    del base_model
    LOGGER.info("Timer input_token_len=%d", patch_len)
    train_series: Dict[str, SeriesCache] = {}
    if args.data_source == "hf":
        LOGGER.info("Training pool: HF %s split=%s", args.hf_name, args.hf_train_split)
        for dataset_name, subject_id, df in iter_glucofm_subjects_from_hf(
            args.hf_name,
            args.hf_train_split,
        ):
            participant_id = f"{dataset_name}__{subject_id}"
            try:
                train_series[participant_id] = load_series_cache(df)
            except Exception as exc:
                LOGGER.warning("train/%s: skipped (%s)", participant_id, exc)
    else:
        train_root: Path = args.data_root_train
        if not train_root.exists():
            raise FileNotFoundError(
                f"Train root not found: {train_root}\n"
                "Tip: prepare the dataset with:\n"
                "  python prepare_dataset.py --create-mixed\n"
            )

        train_groups = collect_subject_groups(train_root)
        if not train_groups:
            raise FileNotFoundError(f"No training CSV files found under {train_root}")

        LOGGER.info("Training pool: %s (%d participants)", train_root, len(train_groups))
        for participant_id, csv_paths in train_groups.items():
            try:
                df = load_subject(csv_paths)
                train_series[participant_id] = load_series_cache(df)
            except Exception as exc:
                LOGGER.warning("train/%s: skipped (%s)", participant_id, exc)

    if not train_series:
        raise ValueError("No valid training series loaded.")

    dataset_state: Dict[str, Dict[str, object]] = {}
    if args.data_source == "hf":
        allowed = set(args.datasets) if args.datasets is not None else None
        test_subjects: Dict[str, Dict[str, pd.DataFrame]] = {}
        for dataset_name, subject_id, df in iter_glucofm_subjects_from_hf(
            args.hf_name,
            args.hf_test_split,
            datasets=allowed,
        ):
            test_subjects.setdefault(dataset_name, {})[subject_id] = df
        if not test_subjects:
            raise FileNotFoundError(
                f"No HF test subjects found (hf_name={args.hf_name!r}, split={args.hf_test_split!r})."
            )
        dataset_pairs = sorted(test_subjects.items(), key=lambda kv: kv[0])
    else:
        test_root: Path = args.data_root_test
        if not test_root.exists():
            raise FileNotFoundError(
                f"Test root not found: {test_root}\n"
                "Tip: prepare the dataset with:\n"
                "  python prepare_dataset.py --create-mixed\n"
            )
        dataset_dirs = sorted(p for p in test_root.iterdir() if p.is_dir())
        if args.datasets is not None:
            allowed = set(args.datasets)
            dataset_dirs = [p for p in dataset_dirs if p.name in allowed]
        if not dataset_dirs:
            raise FileNotFoundError(f"No dataset folders found in {test_root}")
        dataset_pairs = [(p.name, p) for p in dataset_dirs]

    cache_key_cols = ("participant_id", "context_hours", "horizon_minutes", "stride_steps", "metric_mode")
    desired_order = [
        "dataset",
        "split",
        "shot",
        "model_name",
        "participant_id",
        "context_hours",
        "horizon_minutes",
        "context_steps",
        "model_context_steps",
        "horizon_steps",
        "step_minutes",
        "freq",
        "stride_steps",
        "metric_mode",
        "rmse",
        "mae",
        "windows",
        "train_epochs",
        "train_batch_size",
        "train_stride_steps",
        "max_train_steps",
        "max_train_windows",
        "lr",
        "weight_decay",
        "train_loss_mode",
    ]
    if args.data_source == "hf":
        for dataset_name, subjects in dataset_pairs:
            output_path = results_root / f"{dataset_name}_test_metrics.csv"

            completed_keys: Set[Tuple[object, ...]] = set()
            if args.overwrite:
                if output_path.exists():
                    try:
                        output_path.unlink()
                    except Exception as exc:
                        LOGGER.warning("Failed to remove existing metrics %s (%s)", output_path, exc)
            else:
                completed_keys = load_completed_keys(output_path, key_cols=cache_key_cols)
                if completed_keys:
                    LOGGER.info(
                        "%s: found %d existing rows, skipping unless --overwrite is set.",
                        dataset_name,
                        len(completed_keys),
                    )

            participant_ids = list(subjects.keys())
            if not participant_ids:
                LOGGER.warning("%s: empty HF test participants, skipping", dataset_name)
                continue

            LOGGER.info("%s: test_participants=%d (HF)", dataset_name, len(participant_ids))
            dataset_state[dataset_name] = {
                "output_path": output_path,
                "completed_keys": completed_keys,
                "participant_ids": participant_ids,
                "test_subjects": subjects,
                "records": [],
            }
    else:
        for dataset_name, dataset_dir in dataset_pairs:
            output_path = results_root / f"{dataset_name}_test_metrics.csv"

            completed_keys: Set[Tuple[object, ...]] = set()
            if args.overwrite:
                # Clear once so incremental writes produce a fresh file for this run.
                if output_path.exists():
                    try:
                        output_path.unlink()
                    except Exception as exc:
                        LOGGER.warning("Failed to remove existing metrics %s (%s)", output_path, exc)
            else:
                completed_keys = load_completed_keys(output_path, key_cols=cache_key_cols)
                if completed_keys:
                    LOGGER.info(
                        "%s: found %d existing rows, skipping unless --overwrite is set.",
                        dataset_name,
                        len(completed_keys),
                    )

            test_groups = collect_subject_groups(dataset_dir)
            if not test_groups:
                LOGGER.warning("%s: empty test participants, skipping", dataset_name)
                continue

            participant_ids = list(test_groups.keys())
            LOGGER.info("%s: test_participants=%d", dataset_name, len(participant_ids))
            dataset_state[dataset_name] = {
                "output_path": output_path,
                "completed_keys": completed_keys,
                "participant_ids": participant_ids,
                "test_groups": test_groups,
                "records": [],
            }

    if not dataset_state:
        raise FileNotFoundError("No usable test datasets found.")

    for cfg in configs:
        missing_any = args.overwrite
        if not missing_any:
            stride_steps = cfg.context_steps if args.eval_stride_steps <= 0 else args.eval_stride_steps
            for info in dataset_state.values():
                completed = info["completed_keys"]
                participant_ids = info["participant_ids"]
                if any(
                    (
                        participant_id,
                        cfg.context_hours,
                        cfg.horizon_minutes,
                        int(stride_steps),
                        str(args.metric_mode),
                    )
                    not in completed
                    for participant_id in participant_ids
                ):
                    missing_any = True
                    break
        if not missing_any:
            continue

        LOGGER.info("Finetune mixed: ctx=%dh hor=%dm", cfg.context_hours, cfg.horizon_minutes)
        try:
            finetuned_model = finetune_fullshot(
                model_name=args.model_name,
                train_series=train_series,
                cfg=cfg,
                device=device,
                patch_len=patch_len,
                seed=args.seed,
                train_stride_steps=args.train_stride_steps,
                train_batch_size=args.train_batch_size,
                train_epochs=args.train_epochs,
                max_train_steps=args.max_train_steps,
                max_train_windows=args.max_train_windows,
                lr=args.lr,
                weight_decay=args.weight_decay,
                loss_mode=args.train_loss_mode,
                log_every=args.log_every,
            )
        except Exception as exc:
            LOGGER.warning("Finetune failed ctx=%dh hor=%dm (%s)", cfg.context_hours, cfg.horizon_minutes, exc)
            continue

        if args.save_trained_model:
            try:
                saved_path = save_trained_model(
                    finetuned_model,
                    results_root=results_root,
                    shot_label=shot_label,
                    cfg=cfg,
                )
                LOGGER.info("Saved trained model to %s", saved_path)
            except Exception as exc:
                LOGGER.warning(
                    "Failed to save trained model ctx=%dh hor=%dm (%s)",
                    cfg.context_hours,
                    cfg.horizon_minutes,
                    exc,
                )

        stride_steps = cfg.context_steps if args.eval_stride_steps <= 0 else args.eval_stride_steps
        for dataset_name, info in dataset_state.items():
            completed = info["completed_keys"]
            if args.data_source == "hf":
                test_subjects = info["test_subjects"]
                subject_iter = test_subjects.items()
            else:
                test_groups = info["test_groups"]
                subject_iter = test_groups.items()
            records = info["records"]

            for participant_id, subject_payload in subject_iter:
                key = (
                    participant_id,
                    cfg.context_hours,
                    cfg.horizon_minutes,
                    int(stride_steps),
                    str(args.metric_mode),
                )
                if key in completed:
                    continue
                try:
                    if args.data_source == "hf":
                        df = subject_payload
                    else:
                        df = load_subject(subject_payload)
                except Exception as exc:
                    LOGGER.warning("%s/test/%s: skipped (%s)", dataset_name, participant_id, exc)
                    continue

                sse, sae, points, windows, model_context_steps = evaluate_subject(
                    finetuned_model,
                    participant_id,
                    df,
                    cfg,
                    stride_steps=stride_steps,
                    batch_size=args.eval_batch_size,
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
                        "dataset": dataset_name,
                        "split": "test",
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
                        "shot": shot_label,
                        "model_name": args.model_name,
                        "train_epochs": args.train_epochs,
                        "train_batch_size": args.train_batch_size,
                        "train_stride_steps": args.train_stride_steps,
                        "max_train_steps": args.max_train_steps,
                        "max_train_windows": args.max_train_windows,
                        "lr": args.lr,
                        "weight_decay": args.weight_decay,
                        "train_loss_mode": args.train_loss_mode,
                    }
                )

        # Flush results to disk after each (context,horizon) so partial progress isn't lost.
        for dataset_name, info in dataset_state.items():
            output_path: Path = info["output_path"]
            records: List[Dict[str, object]] = info["records"]
            if not records:
                continue

            new_df = pd.DataFrame(records)
            written = upsert_csv_rows(
                output_path,
                new_df,
                key_cols=cache_key_cols,
                desired_order=desired_order,
                sort_by=["participant_id", "context_hours", "horizon_minutes"],
            )
            LOGGER.info("Saved metrics to %s (+%d rows)", output_path, written)

            completed = info["completed_keys"]
            for row in new_df.itertuples(index=False):
                completed.add(
                    (
                        str(getattr(row, "participant_id")),
                        int(getattr(row, "context_hours")),
                        int(getattr(row, "horizon_minutes")),
                        int(getattr(row, "stride_steps")),
                        str(getattr(row, "metric_mode")),
                    )
                )

            records.clear()

        # Free memory between configs.
        del finetuned_model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

