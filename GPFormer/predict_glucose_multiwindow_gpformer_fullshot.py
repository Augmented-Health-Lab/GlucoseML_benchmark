from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# GPFormer is vendored as a folder (not a Python package). Add it to sys.path so
# its internal absolute imports (e.g. `from layers...`) resolve when running from
# the project root.
GPFORMER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(GPFORMER_ROOT))

from models import GPFormer  # noqa: E402

MODEL_NAME = "GPFormer"
DATA_SPLITS = {
    # Train on the mixed training pool by default.
    "train": Path(__file__).resolve().parent.parent / "training_dataset" / "mixed",
    "test": Path(__file__).resolve().parent.parent / "test_dataset",
}

DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parent / "multi_horizon_results_gpformer_fullshot"
DEFAULT_LOG_STEM = "predict_glucose_multiwindow_gpformer_fullshot"
DEFAULT_SHOT_LABEL = "full-shot"
DEFAULT_EVAL_STRIDE_STEPS = 1
DEFAULT_TRAIN_EPOCHS = 10
DEFAULT_TRAIN_STRIDE_STEPS = 12

STEP_MINUTES = 5
FREQ = "5min"
STEP_DELTA = np.timedelta64(STEP_MINUTES, "m")

INPUT_WINDOWS_HOURS = [1, 4, 8, 12, 16, 24]
HORIZONS_MINUTES = [15, 30, 60, 90]

DEFAULT_EVAL_BATCH_SIZE = 8
DEFAULT_TRAIN_BATCH_SIZE = 16

LOGGER = logging.getLogger(__name__)

# Torch may be installed in an environment where NumPy interop is unavailable
# (e.g. torch built against NumPy 1.x running with NumPy 2.x). We still want the
# benchmark script to run, so we provide a slow-but-safe fallback that avoids
# `torch.from_numpy` / `.numpy()`.
try:
    torch.from_numpy(np.zeros(1, dtype=np.float32))
    TORCH_NUMPY_AVAILABLE = True
except Exception:
    TORCH_NUMPY_AVAILABLE = False


def torch_tensor_from_array(arr: np.ndarray, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if TORCH_NUMPY_AVAILABLE:
        t = torch.from_numpy(arr)
        return t if t.dtype == dtype else t.to(dtype)
    return torch.tensor(arr.tolist(), dtype=dtype)


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


def iter_test_dataset_dirs(test_root: Path) -> Iterable[Tuple[str, Path]]:
    """Yield (dataset_name, dataset_dir) pairs under test_root.

    Special-cases `test_dataset/controlled_dataset` which is a container holding
    datasets as subfolders (e.g. 5_T1DEXI, 8_DiaTrend, OhioT1DM).
    """

    if not test_root.exists():
        return []

    out: List[Tuple[str, Path]] = []
    for p in sorted(test_root.iterdir()):
        if not p.is_dir():
            continue
        if p.name == "controlled_dataset":
            for child in sorted(p.iterdir()):
                if child.is_dir():
                    out.append((child.name, child))
            continue
        out.append((p.name, p))
    return out


def ensure_results_root(results_root: Path) -> None:
    results_root.mkdir(parents=True, exist_ok=True)


def _safe_path_component(value: object, *, max_len: int = 120) -> str:
    text = str(value)
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)[:max_len].strip("._") or "model"


def save_trained_model(
    model: torch.nn.Module,
    *,
    results_root: Path,
    shot_label: str,
    cfg: ForecastConfig,
    args: argparse.Namespace,
    label_len: int,
    mean: float,
    std: float,
    train_steps: int,
) -> Path:
    save_dir = (
        results_root
        / "saved_models"
        / f"{_safe_path_component(shot_label)}_ctx{cfg.context_hours}h_hor{cfg.horizon_minutes}m"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "shot": str(shot_label),
        "model_name": str(args.model_name),
        "context_hours": int(cfg.context_hours),
        "horizon_minutes": int(cfg.horizon_minutes),
        "context_steps": int(cfg.context_steps),
        "horizon_steps": int(cfg.horizon_steps),
        "train_epochs": int(args.train_epochs),
        "train_batch_size": int(args.train_batch_size),
        "train_stride_steps": int(args.train_stride_steps),
        "max_train_steps": args.max_train_steps,
        "max_train_windows": args.max_train_windows,
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "train_loss_mode": str(args.train_loss_mode),
        "seed": int(args.seed),
        "label_len": int(label_len),
        "d_model": int(args.d_model),
        "n_heads": int(args.n_heads),
        "e_layers": int(args.e_layers),
        "d_layers": int(args.d_layers),
        "d_ff": int(args.d_ff),
        "dropout": float(args.dropout),
        "scaler_mean": float(mean),
        "scaler_std": float(std),
        "train_steps": int(train_steps),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    (save_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "meta": meta,
            "scaler": {"mean": float(mean), "std": float(std)},
        },
        save_dir / "model.pt",
    )
    return save_dir


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


def cyclical_time_marks(timestamps: np.ndarray) -> np.ndarray:
    """Compute cyclical (sin, cos) marks for time-of-day.

    Returns shape [len(timestamps), 2].
    """

    idx = pd.DatetimeIndex(pd.to_datetime(timestamps))
    minutes = (idx.hour * 60 + idx.minute).to_numpy(dtype="float32")
    angle = 2.0 * np.pi * minutes / 1439.0
    sin_t = np.sin(angle).astype("float32")
    cos_t = np.cos(angle).astype("float32")
    return np.stack([sin_t, cos_t], axis=-1).astype("float32")


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
    time_marks: np.ndarray  # [T, 2]


def load_series_cache(df: pd.DataFrame) -> SeriesCache:
    values = df["value"].to_numpy(dtype="float32")
    timestamps = normalize_timestamps(df["timestamp"])
    contiguous_runs = consecutive_gap_counts(timestamps)
    time_marks = cyclical_time_marks(timestamps)
    return SeriesCache(
        values=values,
        timestamps=timestamps,
        contiguous_runs=contiguous_runs,
        time_marks=time_marks,
    )


def fit_standard_scaler(series: Dict[str, SeriesCache]) -> Tuple[float, float]:
    total = 0
    total_sum = 0.0
    total_sumsq = 0.0
    for cache in series.values():
        x = cache.values.astype("float64")
        total += int(x.size)
        total_sum += float(np.sum(x))
        total_sumsq += float(np.sum(x * x))

    if total <= 0:
        raise ValueError("Cannot fit scaler: empty training series.")

    mean = total_sum / total
    var = total_sumsq / total - mean * mean
    var = max(var, 1e-12)
    std = float(math.sqrt(var))
    return float(mean), std


def scale_values(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((values.astype("float32") - float(mean)) / float(std)).astype("float32")


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


class GPFormerWindowDataset(
    Dataset[
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ]
):
    def __init__(
        self,
        series: Dict[str, SeriesCache],
        index: Sequence[Tuple[str, int]],
        *,
        context_steps: int,
        horizon_steps: int,
        label_len: int,
        mean: float,
        std: float,
    ):
        self._series = series
        self._index = list(index)
        self._context_steps = int(context_steps)
        self._horizon_steps = int(horizon_steps)
        self._label_len = int(label_len)
        self._mean = float(mean)
        self._std = float(std)

        if self._label_len <= 0:
            raise ValueError("label_len must be > 0")
        if self._label_len > self._context_steps:
            raise ValueError("label_len must be <= context_steps")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        participant_id, start = self._index[idx]
        cache = self._series[participant_id]

        end = start + self._context_steps
        future_end = end + self._horizon_steps
        label_start = end - self._label_len

        past = cache.values[start:end]
        past_mark = cache.time_marks[start:end]

        label_vals = cache.values[label_start:end]
        dec_marks = cache.time_marks[label_start:future_end]
        future = cache.values[end:future_end]

        past_scaled = scale_values(past, self._mean, self._std)
        label_scaled = scale_values(label_vals, self._mean, self._std)
        future_scaled = scale_values(future, self._mean, self._std)

        # Decoder input: last `label_len` known values + zeros for the future horizon.
        dec_inp = np.concatenate(
            [label_scaled, np.zeros(self._horizon_steps, dtype="float32")], axis=0
        )

        return (
            torch_tensor_from_array(past_scaled, dtype=torch.float32).unsqueeze(-1),  # [context, 1]
            torch_tensor_from_array(past_mark.astype("float32"), dtype=torch.float32),  # [context, 2]
            torch_tensor_from_array(dec_inp, dtype=torch.float32).unsqueeze(-1),  # [label+pred, 1]
            torch_tensor_from_array(dec_marks.astype("float32"), dtype=torch.float32),  # [label+pred, 2]
            torch_tensor_from_array(future_scaled, dtype=torch.float32).unsqueeze(-1),  # [pred, 1]
        )


def train_fullshot(
    *,
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
    d_model: int,
    n_heads: int,
    e_layers: int,
    d_layers: int,
    d_ff: int,
    dropout: float,
    train_loss_mode: str,
    label_len: int,
    log_every: int,
) -> Tuple[torch.nn.Module, float, float, int]:
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

    mean, std = fit_standard_scaler(train_series)
    LOGGER.info("Scaler (train): mean=%.4f std=%.4f", mean, std)

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

    dataset = GPFormerWindowDataset(
        train_series,
        train_index,
        context_steps=cfg.context_steps,
        horizon_steps=cfg.horizon_steps,
        label_len=label_len,
        mean=mean,
        std=std,
    )
    loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)
    steps_per_epoch = len(loader)
    LOGGER.info(
        "train setup: windows=%d batch_size=%d steps_per_epoch=%d epochs=%d max_steps=%s",
        len(train_index),
        train_batch_size,
        steps_per_epoch,
        train_epochs,
        str(max_train_steps),
    )

    # Minimal config object expected by GPFormer.Model
    model_cfg = SimpleNamespace(
        pred_len=cfg.horizon_steps,
        output_attention=False,
        enc_in=1,
        dec_in=1,
        c_out=1,
        d_model=int(d_model),
        n_heads=int(n_heads),
        e_layers=int(e_layers),
        d_layers=int(d_layers),
        d_ff=int(d_ff),
        factor=1,
        dropout=float(dropout),
        embed="cycF",
        freq="cyc",
        activation="gelu",
    )

    model = GPFormer.Model(model_cfg).float().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    total_steps = 0
    for epoch in range(train_epochs):
        model.train()
        for batch in loader:
            if max_train_steps is not None and total_steps >= max_train_steps:
                break
            x_enc, x_mark_enc, x_dec, x_mark_dec, y = batch
            x_enc = x_enc.to(device)
            x_mark_enc = x_mark_enc.to(device)
            x_dec = x_dec.to(device)
            x_mark_dec = x_mark_dec.to(device)
            y = y.to(device)

            pred = model(x_enc, x_mark_enc, x_dec, x_mark_dec)  # [B, pred_len, 1]

            if train_loss_mode == "final":
                pred = pred[:, -1:, :]
                y = y[:, -1:, :]

            loss = loss_fn(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_steps += 1
            if log_every > 0 and total_steps % log_every == 0:
                LOGGER.info(
                    "train step=%d epoch=%d/%d loss=%.6f windows=%d",
                    total_steps,
                    epoch + 1,
                    train_epochs,
                    float(loss.detach().cpu()),
                    len(train_index),
                )

        if max_train_steps is not None and total_steps >= max_train_steps:
            break

    model.eval()
    LOGGER.info("train done: total_steps=%d", total_steps)
    return model, mean, std, total_steps


def evaluate_subject(
    model: torch.nn.Module,
    participant_id: str,
    df: pd.DataFrame,
    cfg: ForecastConfig,
    *,
    label_len: int,
    mean: float,
    std: float,
    stride_steps: int,
    batch_size: int,
    metric_mode: str,
    device: str,
) -> Tuple[float, float, int, int]:
    if len(df) < cfg.context_steps + cfg.horizon_steps:
        return 0.0, 0.0, 0, 0

    values = df["value"].to_numpy(dtype="float32")
    timestamps = normalize_timestamps(df["timestamp"])
    contiguous_runs = consecutive_gap_counts(timestamps)
    time_marks = cyclical_time_marks(timestamps)

    values_scaled = scale_values(values, mean, std)

    total_len = cfg.context_steps + cfg.horizon_steps
    required_gaps = total_len - 1

    total_sse_scaled = 0.0
    total_ae_scaled = 0.0
    total_points = 0
    total_windows = 0

    batch_x_enc: List[np.ndarray] = []
    batch_x_mark_enc: List[np.ndarray] = []
    batch_x_dec: List[np.ndarray] = []
    batch_x_mark_dec: List[np.ndarray] = []
    batch_y: List[np.ndarray] = []

    def flush_batch() -> None:
        nonlocal total_sse_scaled, total_ae_scaled, total_points, total_windows
        nonlocal batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y
        if not batch_x_enc:
            return

        try:
            with torch.inference_mode():
                x_enc_t = (
                    torch.stack(
                        [torch_tensor_from_array(a, dtype=torch.float32) for a in batch_x_enc],
                        dim=0,
                    )
                    .unsqueeze(-1)
                    .to(device)
                )
                x_mark_enc_t = torch.stack(
                    [
                        torch_tensor_from_array(a, dtype=torch.float32)
                        for a in batch_x_mark_enc
                    ],
                    dim=0,
                ).to(device)
                x_dec_t = (
                    torch.stack(
                        [torch_tensor_from_array(a, dtype=torch.float32) for a in batch_x_dec],
                        dim=0,
                    )
                    .unsqueeze(-1)
                    .to(device)
                )
                x_mark_dec_t = torch.stack(
                    [
                        torch_tensor_from_array(a, dtype=torch.float32)
                        for a in batch_x_mark_dec
                    ],
                    dim=0,
                ).to(device)
                y_t = (
                    torch.stack(
                        [torch_tensor_from_array(a, dtype=torch.float32) for a in batch_y],
                        dim=0,
                    )
                    .unsqueeze(-1)
                    .to(device)
                )
                preds = model(
                    x_enc_t.float(),
                    x_mark_enc_t.float(),
                    x_dec_t.float(),
                    x_mark_dec_t.float(),
                )

                if metric_mode == "final":
                    err = preds[:, -1, 0] - y_t[:, -1, 0]  # [B]
                    valid = torch.isfinite(err)
                    err = err[valid]
                    total_sse_scaled += float(torch.sum(err * err).cpu())
                    total_ae_scaled += float(torch.sum(torch.abs(err)).cpu())
                    n = int(err.numel())
                    total_points += n
                    total_windows += n
                else:
                    err = preds[..., 0] - y_t[..., 0]  # [B, pred_len]
                    valid = torch.isfinite(err).all(dim=1)
                    if bool(torch.any(valid)):
                        err = err[valid]
                        total_sse_scaled += float(torch.sum(err * err).cpu())
                        total_ae_scaled += float(torch.sum(torch.abs(err)).cpu())
                        total_points += int(err.numel())
                        total_windows += int(torch.sum(valid).item())
        except Exception as exc:
            LOGGER.warning(
                "%s: prediction failure for %d windows (%s)",
                participant_id,
                len(batch_x_enc),
                exc,
            )
            batch_x_enc = []
            batch_x_mark_enc = []
            batch_x_dec = []
            batch_x_mark_dec = []
            batch_y = []
            return

        batch_x_enc = []
        batch_x_mark_enc = []
        batch_x_dec = []
        batch_x_mark_dec = []
        batch_y = []

    max_start = len(values_scaled) - total_len
    if max_start < 0:
        return 0.0, 0.0, 0, 0

    for start in range(0, max_start + 1, stride_steps):
        if int(contiguous_runs[start]) < required_gaps:
            continue

        end = start + cfg.context_steps
        future_end = end + cfg.horizon_steps
        label_start = end - label_len

        past = values_scaled[start:end]
        past_mark = time_marks[start:end]
        label_vals = values_scaled[label_start:end]
        dec_marks = time_marks[label_start:future_end]
        future = values_scaled[end:future_end]

        dec_inp = np.concatenate(
            [label_vals, np.zeros(cfg.horizon_steps, dtype="float32")], axis=0
        )

        batch_x_enc.append(past.astype("float32"))
        batch_x_mark_enc.append(past_mark.astype("float32"))
        batch_x_dec.append(dec_inp.astype("float32"))
        batch_x_mark_dec.append(dec_marks.astype("float32"))
        batch_y.append(future.astype("float32"))

        if len(batch_x_enc) >= batch_size:
            flush_batch()

    flush_batch()

    # Convert scaled metrics back to original units.
    sse = float(total_sse_scaled) * float(std * std)
    sae = float(total_ae_scaled) * float(std)
    return sse, sae, total_points, total_windows


def load_existing_metrics(output_path: Path) -> pd.DataFrame:
    if not output_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(output_path)
    except Exception as exc:
        LOGGER.warning("Failed to load existing metrics from %s (%s)", output_path, exc)
        return pd.DataFrame()


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
            f"{shot_label.capitalize()}: train GPFormer on the mixed training pool (sliding windows) "
            "and evaluate on each dataset's test split for multiple context windows and horizons."
        )
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help="Model name to write to the output CSV.",
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
        help=(
            "Optional dataset folder names to include (default: all). "
            "Note: controlled datasets are selected by their subfolder name (e.g. 5_T1DEXI)."
        ),
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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument(
        "--train-loss-mode",
        choices=["final", "all"],
        default="all",
        help="Optimize MSE on final horizon point or all horizon points.",
    )

    parser.add_argument(
        "--label-len",
        type=int,
        default=None,
        help="Decoder warm-start length. Default: min(horizon_steps, context_steps).",
    )

    # GPFormer architecture knobs (kept minimal; defaults tuned for practicality).
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--e-layers", type=int, default=3)
    parser.add_argument("--d-layers", type=int, default=1)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.05)

    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args(argv)


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

    dataset_pairs = list(iter_test_dataset_dirs(test_root))
    if args.datasets is not None:
        allowed = set(args.datasets)
        dataset_pairs = [(name, p) for (name, p) in dataset_pairs if name in allowed]
    if not dataset_pairs:
        raise FileNotFoundError(f"No dataset folders found in {test_root}")

    dataset_state: Dict[str, Dict[str, object]] = {}
    for dataset_name, dataset_dir in dataset_pairs:
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

        label_len = (
            int(args.label_len)
            if args.label_len is not None
            else min(cfg.horizon_steps, cfg.context_steps)
        )
        LOGGER.info(
            "Train mixed: ctx=%dh hor=%dm (label_len=%d)",
            cfg.context_hours,
            cfg.horizon_minutes,
            label_len,
        )

        model, mean, std, total_steps = train_fullshot(
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
            d_model=args.d_model,
            n_heads=args.n_heads,
            e_layers=args.e_layers,
            d_layers=args.d_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            train_loss_mode=args.train_loss_mode,
            label_len=label_len,
            log_every=args.log_every,
        )

        try:
            saved_path = save_trained_model(
                model,
                results_root=results_root,
                shot_label=shot_label,
                cfg=cfg,
                args=args,
                label_len=label_len,
                mean=mean,
                std=std,
                train_steps=total_steps,
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
            test_groups = info["test_groups"]
            records = info["records"]

            for participant_id, csv_paths in test_groups.items():
                key = (participant_id, cfg.context_hours, cfg.horizon_minutes)
                if key in completed:
                    continue
                try:
                    df = load_subject(csv_paths)
                except Exception as exc:
                    LOGGER.warning(
                        "%s/test/%s: skipped (%s)", dataset_name, participant_id, exc
                    )
                    continue

                sse, sae, points, windows = evaluate_subject(
                    model,
                    participant_id,
                    df,
                    cfg,
                    label_len=label_len,
                    mean=mean,
                    std=std,
                    stride_steps=stride_steps,
                    batch_size=args.eval_batch_size,
                    metric_mode=args.metric_mode,
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
                        "finetune_pattern": "full",
                        "train_loss_mode": args.train_loss_mode,
                        # GPFormer-specific knobs (kept as extra columns at the end).
                        "label_len": label_len,
                        "d_model": args.d_model,
                        "n_heads": args.n_heads,
                        "e_layers": args.e_layers,
                        "d_layers": args.d_layers,
                        "d_ff": args.d_ff,
                        "dropout": args.dropout,
                        "train_steps": total_steps,
                    }
                )

        # Free GPU memory between configs.
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

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
        final_df = final_df.sort_values(
            ["participant_id", "context_hours", "horizon_minutes"]
        ).reset_index(drop=True)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_path, index=False)
        LOGGER.info("Saved metrics to %s (%d rows)", output_path, len(final_df))


if __name__ == "__main__":
    main()
