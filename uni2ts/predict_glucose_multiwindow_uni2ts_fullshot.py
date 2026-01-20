from __future__ import annotations

import argparse
import logging
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from gluonts.dataset.common import ListDataset
from gluonts.torch import PyTorchPredictor
from torch.utils.data import DataLoader, Dataset

try:
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
except ModuleNotFoundError:  # pragma: no cover
    import sys

    # Local (repo) install support: use Uni2TS in ./src without requiring pip install -e .
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

MODEL_NAME = "Salesforce/moirai-2.0-R-small"
DATA_SPLITS = {
    # Train on the mixed training pool by default.
    "train": Path(__file__).resolve().parent.parent / "training_dataset" / "mixed",
    "test": Path(__file__).resolve().parent.parent / "test_dataset",
}

DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parent / "multi_horizon_results_uni2ts_fullshot"
DEFAULT_LOG_STEM = "predict_glucose_multiwindow_uni2ts_fullshot"
DEFAULT_SHOT_LABEL = "full-shot"
DEFAULT_EVAL_STRIDE_STEPS = 1
DEFAULT_TRAIN_EPOCHS = 10
DEFAULT_TRAIN_STRIDE_STEPS = 10

STEP_MINUTES = 5
FREQ = "5min"
STEP_DELTA = np.timedelta64(STEP_MINUTES, "m")

INPUT_WINDOWS_HOURS = [1, 4, 8, 12, 16, 24]
HORIZONS_MINUTES = [15, 30, 60, 90]

DEFAULT_EVAL_BATCH_SIZE = 8
DEFAULT_TRAIN_BATCH_SIZE = 16

LOGGER = logging.getLogger(__name__)


def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lowered = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def load_subject_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    timestamp_col = _find_column(
        df.columns, ["timestamp", "time", "datetime", "date_time", "date"]
    )
    value_col = _find_column(
        df.columns,
        ["bgvalue", "glucose", "glucose_value", "sensor_glucose", "value"],
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
    df = df[[timestamp_col, value_col]].rename(
        columns={timestamp_col: "timestamp", value_col: "value"}
    )
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
        raise ValueError(
            f"No valid data points after merging {[str(p) for p in csv_paths]}"
        )
    return df.reset_index(drop=True)


def collect_subject_groups(dataset_dir: Path) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for csv_path in sorted(dataset_dir.rglob("*.csv")):
        participant_id = csv_path.stem
        groups.setdefault(participant_id, []).append(csv_path.resolve())
    return groups


def ensure_results_root(results_root: Path) -> None:
    results_root.mkdir(parents=True, exist_ok=True)


def setup_logging(results_root: Path, log_stem: str) -> Path:
    ensure_results_root(results_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = results_root / f"{log_stem}_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
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
    if ts.isna().any():
        raise ValueError("Found invalid timestamps after normalization.")
    return ts.to_numpy(dtype="datetime64[ns]")


def consecutive_gap_counts(timestamps: np.ndarray) -> np.ndarray:
    n = len(timestamps)
    if n == 0:
        return np.array([], dtype=int)
    diffs = timestamps[1:] - timestamps[:-1]
    is_step = diffs == STEP_DELTA
    counts = np.zeros(n, dtype=int)
    for idx in range(n - 2, -1, -1):
        counts[idx] = counts[idx + 1] + 1 if bool(is_step[idx]) else 0
    return counts


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
    return SeriesCache(
        values=values, timestamps=timestamps, contiguous_runs=contiguous_runs
    )


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


def evaluate_subject(
    predictor: PyTorchPredictor,
    participant_id: str,
    df: pd.DataFrame,
    cfg: ForecastConfig,
    stride_steps: int,
    batch_size: int,
    metric_mode: str,
) -> Tuple[float, float, int, int]:
    if len(df) < cfg.context_steps + cfg.horizon_steps:
        return 0.0, 0.0, 0, 0

    cache = load_series_cache(df)
    values = cache.values
    timestamps = cache.timestamps
    contiguous_runs = cache.contiguous_runs
    total_len = cfg.context_steps + cfg.horizon_steps
    required_gaps = total_len - 1

    total_sse = 0.0
    total_ae = 0.0
    total_points = 0
    total_windows = 0

    batch_entries: List[Dict[str, object]] = []
    batch_targets: List[np.ndarray] = []

    def flush_batch() -> None:
        nonlocal total_sse, total_ae, total_points, total_windows, batch_entries, batch_targets
        if not batch_entries:
            return
        dataset = ListDataset(batch_entries, freq=FREQ)
        try:
            with torch.inference_mode():
                forecasts = list(predictor.predict(dataset))
        except Exception as exc:
            LOGGER.warning(
                "%s: prediction failure for %d windows (%s)",
                participant_id,
                len(batch_entries),
                exc,
            )
            batch_entries = []
            batch_targets = []
            return

        for forecast, target_vals in zip(forecasts, batch_targets):
            try:
                preds = _forecast_to_array(forecast, cfg.horizon_steps)
            except ValueError as exc:
                LOGGER.warning("%s: failed to parse forecast (%s)", participant_id, exc)
                continue

            if metric_mode == "final":
                pred_val = float(preds[cfg.horizon_steps - 1])
                target_val = float(target_vals[cfg.horizon_steps - 1])
                if math.isnan(pred_val) or math.isnan(target_val):
                    continue
                err = pred_val - target_val
                total_sse += float(err * err)
                total_ae += float(abs(err))
                total_points += 1
                total_windows += 1
                continue

            valid_len = min(len(preds), len(target_vals), cfg.horizon_steps)
            if valid_len == 0:
                continue
            preds = preds[:valid_len]
            targets = target_vals[:valid_len]
            if np.isnan(preds).any() or np.isnan(targets).any():
                continue
            errors = preds - targets
            total_sse += float(np.sum(errors**2))
            total_ae += float(np.sum(np.abs(errors)))
            total_points += int(valid_len)
            total_windows += 1

        batch_entries = []
        batch_targets = []

    for start in range(
        0,
        len(df) - cfg.context_steps - cfg.horizon_steps + 1,
        stride_steps,
    ):
        if int(contiguous_runs[start]) < required_gaps:
            continue
        end = start + cfg.context_steps
        target_end = end + cfg.horizon_steps

        history_values = values[start:end].astype("float32")
        target_values = values[end:target_end].astype("float32")

        window_id = (
            f"{participant_id}__ctx{cfg.context_hours}h__hor{cfg.horizon_minutes}m__{start}"
        )
        start_ts = pd.to_datetime(timestamps[start])
        if getattr(start_ts, "tzinfo", None) is not None:
            start_ts = start_ts.tz_localize(None)

        batch_entries.append(
            {"item_id": window_id, "start": start_ts, "target": history_values}
        )
        batch_targets.append(target_values)
        if len(batch_entries) >= batch_size:
            flush_batch()

    flush_batch()
    return total_sse, total_ae, total_points, total_windows


def load_existing_metrics(output_path: Path) -> pd.DataFrame:
    if not output_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(output_path)
    except Exception as exc:
        LOGGER.warning("Failed to load existing metrics from %s (%s)", output_path, exc)
        return pd.DataFrame()


class WindowDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        series: Dict[str, SeriesCache],
        index: Sequence[Tuple[str, int]],
        context_steps: int,
        horizon_steps: int,
    ):
        self._series = series
        self._index = list(index)
        self._context_steps = context_steps
        self._horizon_steps = horizon_steps

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        participant_id, start = self._index[idx]
        cache = self._series[participant_id]
        end = start + self._context_steps
        future_end = end + self._horizon_steps
        past = cache.values[start:end]
        future = cache.values[end:future_end]
        return (
            torch.from_numpy(past.astype("float32")),
            torch.from_numpy(future.astype("float32")),
        )


def pinball_loss(
    quantile_preds: torch.Tensor,  # [batch, num_q, horizon]
    target: torch.Tensor,  # [batch, horizon]
    quantiles: Sequence[float],
) -> torch.Tensor:
    if quantile_preds.ndim != 3:
        raise ValueError(
            f"Expected quantile_preds with shape [B,Q,T], got {quantile_preds.shape}"
        )
    if target.ndim != 2:
        raise ValueError(f"Expected target with shape [B,T], got {target.shape}")
    if quantile_preds.shape[0] != target.shape[0] or quantile_preds.shape[2] != target.shape[1]:
        raise ValueError(f"Shape mismatch: preds={quantile_preds.shape}, target={target.shape}")

    q = torch.tensor(
        list(quantiles),
        device=quantile_preds.device,
        dtype=quantile_preds.dtype,
    ).view(1, -1, 1)
    errors = target.unsqueeze(1) - quantile_preds
    return torch.maximum(q * errors, (q - 1.0) * errors).mean()


def set_trainable_params(module: torch.nn.Module, finetune_pattern: str) -> None:
    finetune_pattern = str(finetune_pattern).lower()
    if finetune_pattern == "full":
        for p in module.parameters():
            p.requires_grad = True
        return

    if finetune_pattern == "freeze_ffn":
        for name, p in module.named_parameters():
            p.requires_grad = True
            if "ffn" in name:
                p.requires_grad = False
        return

    if finetune_pattern == "head_only":
        for name, p in module.named_parameters():
            p.requires_grad = name.startswith("out_proj.")
        return

    raise ValueError(f"Unsupported finetune_pattern: {finetune_pattern!r}")


def build_train_index(
    series: Dict[str, SeriesCache],
    context_steps: int,
    horizon_steps: int,
    stride_steps: int,
    max_windows: Optional[int],
    seed: int,
) -> List[Tuple[str, int]]:
    index: List[Tuple[str, int]] = []
    total_len = context_steps + horizon_steps
    required_gaps = total_len - 1
    for participant_id, cache in series.items():
        max_start = len(cache.values) - total_len
        if max_start < 0:
            continue
        for start in range(0, max_start + 1, stride_steps):
            if int(cache.contiguous_runs[start]) < required_gaps:
                continue
            index.append((participant_id, start))

    if max_windows is not None and len(index) > max_windows:
        rng = random.Random(seed)
        index = rng.sample(index, k=max_windows)
    return index


def finetune_fullshot(
    *,
    model_name: str,
    train_series: Dict[str, SeriesCache],
    cfg: ForecastConfig,
    device: str,
    seed: int,
    train_stride_steps: int,
    train_batch_size: int,
    train_epochs: int,
    max_train_steps: Optional[int],
    max_train_windows: Optional[int],
    lr: float,
    weight_decay: float,
    finetune_pattern: str,
    loss_mode: str,
    log_every: int,
) -> Moirai2Module:
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

    module: Moirai2Module = Moirai2Module.from_pretrained(model_name)
    set_trainable_params(module, finetune_pattern)
    module.to(device)
    module.train()

    forecast = Moirai2Forecast(
        module=module,
        prediction_length=cfg.horizon_steps,
        context_length=cfg.context_steps,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    ).to(device)
    forecast.train()

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

    dataset = WindowDataset(train_series, train_index, cfg.context_steps, cfg.horizon_steps)
    loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)
    steps_per_epoch = len(loader)
    LOGGER.info(
        "finetune setup: windows=%d batch_size=%d steps_per_epoch=%d epochs=%d max_steps=%s",
        len(train_index),
        train_batch_size,
        steps_per_epoch,
        train_epochs,
        str(max_train_steps),
    )

    trainable_params = [p for p in module.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError(f"No trainable parameters with finetune_pattern={finetune_pattern!r}")
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    quantiles = list(getattr(module, "quantile_levels", (0.5,)))
    if not quantiles:
        raise ValueError("Model quantile_levels is empty.")

    total_steps = 0
    for epoch in range(train_epochs):
        for past, future in loader:
            if max_train_steps is not None and total_steps >= max_train_steps:
                break
            past = past.to(device).unsqueeze(-1)  # [B, context, 1]
            future = future.to(device)  # [B, horizon]
            past_obs = torch.ones_like(past, dtype=torch.bool)
            past_is_pad = torch.zeros(
                past.shape[0], past.shape[1], device=device, dtype=torch.bool
            )

            quantile_preds = forecast(past, past_obs, past_is_pad)  # [B, Q, horizon]
            if loss_mode == "final":
                quantile_preds = quantile_preds[..., -1:].contiguous()
                future = future[..., -1:].contiguous()
            loss = pinball_loss(quantile_preds, future, quantiles)

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

    module.eval()
    LOGGER.info("finetune done: total_steps=%d", total_steps)
    return module


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
            f"{shot_label.capitalize()}: fine-tune Moirai2 on the mixed training pool (sliding windows) "
            "and evaluate on each dataset's test split for multiple context windows and horizons."
        )
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help="HuggingFace model id to load.",
    )
    parser.add_argument(
        "--data-root-train",
        type=Path,
        default=DATA_SPLITS["train"],
        help="Directory containing training CSV files (default: training_dataset/mixed).",
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
        "--context-hours",
        type=int,
        nargs="*",
        default=[12],
        help="Context window hours (default: 12).",
    )
    parser.add_argument(
        "--horizons-minutes",
        type=int,
        nargs="*",
        default=[30],
        help="Forecast horizon in minutes (default: 30).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for training/eval: auto/cpu/cuda.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metric-mode", choices=["final", "all"], default="final")
    parser.add_argument(
        "--eval-stride-steps",
        type=int,
        default=default_eval_stride_steps,
        help="Evaluation sliding window stride in steps (0 = context_steps; 1 = 5 minutes).",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=DEFAULT_EVAL_BATCH_SIZE,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing metrics CSVs.",
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
        "--finetune-pattern",
        choices=["full", "freeze_ffn", "head_only"],
        default="full",
    )
    parser.add_argument(
        "--train-loss-mode",
        choices=["final", "all"],
        default="all",
        help="Optimize pinball loss on final horizon point or all horizon points.",
    )
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args(argv)


def resolve_device(arg: str) -> str:
    arg = str(arg).lower()
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if arg in {"cpu", "cuda"}:
        if arg == "cuda" and not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but not available; falling back to CPU.")
            return "cpu"
        return arg
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

    train_root: Path = args.data_root_train
    test_root: Path = args.data_root_test
    if not train_root.exists():
        raise FileNotFoundError(f"Train root not found: {train_root}")
    if not test_root.exists():
        raise FileNotFoundError(f"Test root not found: {test_root}")

    context_hours = args.context_hours if args.context_hours is not None else INPUT_WINDOWS_HOURS
    horizons_minutes = args.horizons_minutes if args.horizons_minutes is not None else HORIZONS_MINUTES
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

    train_groups = collect_subject_groups(train_root)
    if not train_groups:
        raise FileNotFoundError(f"No training CSV files found under {train_root}")

    LOGGER.info("Training pool: %s (%d participants)", train_root, len(train_groups))
    train_series: Dict[str, SeriesCache] = {}
    for participant_id, csv_paths in train_groups.items():
        try:
            df = load_subject(csv_paths)
            train_series[participant_id] = load_series_cache(df)
        except Exception as exc:
            LOGGER.warning("train/%s: skipped (%s)", participant_id, exc)

    if not train_series:
        raise ValueError(f"No valid training series loaded from {train_root}")

    dataset_dirs = sorted(p for p in test_root.iterdir() if p.is_dir())
    if args.datasets is not None:
        allowed = set(args.datasets)
        dataset_dirs = [p for p in dataset_dirs if p.name in allowed]
    if not dataset_dirs:
        raise FileNotFoundError(f"No dataset folders found in {test_root}")

    dataset_state: Dict[str, Dict[str, object]] = {}
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        output_path = results_root / f"{dataset_name}_test_metrics.csv"

        existing_df = pd.DataFrame()
        completed_keys: Set[Tuple[str, int, int]] = set()
        if not args.overwrite:
            existing_df = load_existing_metrics(output_path)
            required_cols = {"participant_id", "context_hours", "horizon_minutes"}
            if not existing_df.empty and required_cols.issubset(existing_df.columns):
                completed_keys = set(
                    (str(r["participant_id"]), int(r["context_hours"]), int(r["horizon_minutes"]))
                    for _, r in existing_df.iterrows()
                )
            elif not existing_df.empty:
                LOGGER.warning(
                    "%s: existing metrics missing key columns, ignoring cached content.",
                    dataset_name,
                )
                existing_df = pd.DataFrame()

        test_groups = collect_subject_groups(dataset_dir)
        if not test_groups:
            LOGGER.warning("%s: empty test participants, skipping", dataset_name)
            continue

        LOGGER.info("%s: test_participants=%d", dataset_name, len(test_groups))
        dataset_state[dataset_name] = {
            "dataset_dir": dataset_dir,
            "output_path": output_path,
            "existing_df": existing_df,
            "completed_keys": completed_keys,
            "test_groups": test_groups,
            "records": [],
        }

    if not dataset_state:
        raise FileNotFoundError(f"No usable test datasets found in {test_root}")

    for cfg in configs:
        missing_any = args.overwrite
        if not missing_any:
            for info in dataset_state.values():
                completed = info["completed_keys"]
                test_groups = info["test_groups"]
                if any(
                    (participant_id, cfg.context_hours, cfg.horizon_minutes) not in completed
                    for participant_id in test_groups.keys()
                ):
                    missing_any = True
                    break
        if not missing_any:
            continue

        LOGGER.info("Finetune mixed: ctx=%dh hor=%dm", cfg.context_hours, cfg.horizon_minutes)
        try:
            finetuned_module = finetune_fullshot(
                model_name=args.model_name,
                train_series=train_series,
                cfg=cfg,
                device=device,
                seed=args.seed,
                train_stride_steps=args.train_stride_steps,
                train_batch_size=args.train_batch_size,
                train_epochs=args.train_epochs,
                max_train_steps=args.max_train_steps,
                max_train_windows=args.max_train_windows,
                lr=args.lr,
                weight_decay=args.weight_decay,
                finetune_pattern=args.finetune_pattern,
                loss_mode=args.train_loss_mode,
                log_every=args.log_every,
            )
        except Exception as exc:
            LOGGER.warning(
                "Finetune failed ctx=%dh hor=%dm (%s)",
                cfg.context_hours,
                cfg.horizon_minutes,
                exc,
            )
            continue

        predictor = build_predictor(
            finetuned_module,
            cfg.context_steps,
            cfg.horizon_steps,
            device,
            batch_size=args.eval_batch_size,
        )
        stride_steps = cfg.context_steps if args.eval_stride_steps <= 0 else args.eval_stride_steps

        for dataset_name, info in dataset_state.items():
            completed = info["completed_keys"]
            test_groups = info["test_groups"]
            records = info["records"]

            for participant_id, csv_paths in test_groups.items():
                key = (participant_id, cfg.context_hours, cfg.horizon_minutes)
                if key in completed:
                    continue
                try:
                    df = load_subject(csv_paths)
                except Exception as exc:
                    LOGGER.warning("%s/test/%s: skipped (%s)", dataset_name, participant_id, exc)
                    continue

                sse, sae, points, windows = evaluate_subject(
                    predictor,
                    participant_id,
                    df,
                    cfg,
                    stride_steps=stride_steps,
                    batch_size=args.eval_batch_size,
                    metric_mode=args.metric_mode,
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
                        "finetune_pattern": args.finetune_pattern,
                        "train_loss_mode": args.train_loss_mode,
                    }
                )

    for dataset_name, info in dataset_state.items():
        output_path: Path = info["output_path"]
        existing_df: pd.DataFrame = info["existing_df"]
        records: List[Dict[str, object]] = info["records"]

        new_df = pd.DataFrame(records)
        if args.overwrite or existing_df.empty:
            final_df = new_df
        else:
            final_df = pd.concat([existing_df, new_df], ignore_index=True)
            final_df = final_df.drop_duplicates(
                subset=["participant_id", "context_hours", "horizon_minutes"],
                keep="last",
            ).reset_index(drop=True)

        if final_df.empty:
            LOGGER.warning("%s: no metrics to write.", dataset_name)
            continue

        desired_order = [
            "dataset",
            "split",
            "shot",
            "model_name",
            "participant_id",
            "context_hours",
            "horizon_minutes",
            "context_steps",
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
            "finetune_pattern",
            "train_loss_mode",
        ]
        ordered_cols = [col for col in desired_order if col in final_df.columns]
        remaining_cols = [col for col in final_df.columns if col not in ordered_cols]
        final_df = final_df[ordered_cols + remaining_cols]
        final_df = final_df.sort_values(["participant_id", "context_hours", "horizon_minutes"]).reset_index(
            drop=True
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_path, index=False)
        LOGGER.info("Saved metrics to %s (%d rows)", output_path, len(final_df))


if __name__ == "__main__":
    main()
