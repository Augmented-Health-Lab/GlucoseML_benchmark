from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch

GPFORMER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(GPFORMER_ROOT))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from results_cache import load_completed_keys, upsert_csv_rows  # noqa: E402
from glucofm_data import (  # noqa: E402
    DEFAULT_HF_NAME,
    iter_glucofm_subjects_from_hf,
    parse_timestamp_series,
)

from models import GPFormer  # noqa: E402

DEFAULT_DATA_TEST = Path(__file__).resolve().parent.parent / "hf_cache" / "test"
DEFAULT_LOG_STEM = "gpformer_test_only"
DEFAULT_EVAL_STRIDE_STEPS = 1
DEFAULT_EVAL_BATCH_SIZE = 8

STEP_MINUTES = 5
FREQ = "5min"
STEP_DELTA = np.timedelta64(STEP_MINUTES, "m")

LOGGER = logging.getLogger(__name__)

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


def _find_column(columns, candidates):
    lowered = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def load_subject_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    timestamp_col = _find_column(df.columns, ["timestamp", "time", "datetime", "date_time", "date"])
    value_col = _find_column(df.columns, ["bgvalue", "glucose", "glucose_value", "sensor_glucose", "value"])
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


def iter_test_dataset_dirs(test_root: Path):
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


def setup_logging(results_root: Path, log_stem: str) -> Path:
    results_root.mkdir(parents=True, exist_ok=True)
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


def normalize_timestamps(timestamps: pd.Series) -> np.ndarray:
    ts = parse_timestamp_series(timestamps)
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


def scale_values(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((values.astype("float32") - float(mean)) / float(std)).astype("float32")


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
) -> Tuple[float, float, int, int, List[Dict[str, object]]]:
    if len(df) < cfg.context_steps + cfg.horizon_steps:
        return 0.0, 0.0, 0, 0, []

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
    batch_starts: List[int] = []

    prediction_rows: List[Dict[str, object]] = []

    def flush_batch() -> None:
        nonlocal total_sse_scaled, total_ae_scaled, total_points, total_windows
        nonlocal batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y, batch_starts
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
                    [torch_tensor_from_array(a, dtype=torch.float32) for a in batch_x_mark_enc],
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
                    [torch_tensor_from_array(a, dtype=torch.float32) for a in batch_x_mark_dec],
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
                    err = preds[:, -1, 0] - y_t[:, -1, 0]
                    valid_mask = torch.isfinite(err)
                    err_v = err[valid_mask]
                    total_sse_scaled += float(torch.sum(err_v * err_v).cpu())
                    total_ae_scaled += float(torch.sum(torch.abs(err_v)).cpu())
                    n = int(err_v.numel())
                    total_points += n
                    total_windows += n
                else:
                    err = preds[..., 0] - y_t[..., 0]
                    valid_mask = torch.isfinite(err).all(dim=1)
                    if bool(torch.any(valid_mask)):
                        err_v = err[valid_mask]
                        total_sse_scaled += float(torch.sum(err_v * err_v).cpu())
                        total_ae_scaled += float(torch.sum(torch.abs(err_v)).cpu())
                        total_points += int(err_v.numel())
                        total_windows += int(torch.sum(valid_mask).item())

                preds_orig = (
                    preds[..., 0].detach().cpu().numpy().astype("float64") * float(std)
                ) + float(mean)
                truth_orig = (
                    y_t[..., 0].detach().cpu().numpy().astype("float64") * float(std)
                ) + float(mean)

                for b, start in enumerate(batch_starts):
                    end = start + cfg.context_steps
                    for h in range(cfg.horizon_steps):
                        prediction_rows.append(
                            {
                                "participant_id": participant_id,
                                "context_hours": cfg.context_hours,
                                "horizon_minutes": cfg.horizon_minutes,
                                "window_start_ts": str(timestamps[start]),
                                "context_end_ts": str(timestamps[end - 1]),
                                "forecast_ts": str(timestamps[end + h]),
                                "horizon_step": h + 1,
                                "horizon_offset_min": (h + 1) * STEP_MINUTES,
                                "prediction": float(preds_orig[b, h]),
                                "ground_truth": float(truth_orig[b, h]),
                            }
                        )
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
            batch_starts = []
            return

        batch_x_enc = []
        batch_x_mark_enc = []
        batch_x_dec = []
        batch_x_mark_dec = []
        batch_y = []
        batch_starts = []

    max_start = len(values_scaled) - total_len
    if max_start < 0:
        return 0.0, 0.0, 0, 0, []

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
        batch_starts.append(int(start))

        if len(batch_x_enc) >= batch_size:
            flush_batch()

    flush_batch()

    sse = float(total_sse_scaled) * float(std * std)
    sae = float(total_ae_scaled) * float(std)
    return sse, sae, total_points, total_windows, prediction_rows


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_checkpoint(checkpoint_path: Path, meta_json_path: Optional[Path]):
    obj = _torch_load(checkpoint_path)

    if isinstance(obj, dict) and "state_dict" in obj:
        state_dict = obj["state_dict"]
        meta = obj.get("meta")
        scaler = obj.get("scaler")
    else:
        state_dict = obj
        meta = None
        scaler = None

    if meta is None or scaler is None:
        meta_path = meta_json_path or (checkpoint_path.parent / "meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Checkpoint missing 'meta'/'scaler' and meta.json not found at {meta_path}"
            )
        with meta_path.open() as f:
            meta_disk = json.load(f)
        if meta is None:
            meta = meta_disk
        if scaler is None:
            scaler = {
                "mean": float(meta_disk["scaler_mean"]),
                "std": float(meta_disk["scaler_std"]),
            }

    required = (
        "context_hours", "horizon_minutes", "context_steps", "horizon_steps",
        "label_len", "d_model", "n_heads", "e_layers", "d_layers", "d_ff", "dropout",
    )
    missing = [k for k in required if k not in meta]
    if missing:
        raise ValueError(f"meta is missing required keys: {missing}")

    return state_dict, meta, scaler


def build_model(meta: Dict[str, object], device: str) -> torch.nn.Module:
    cfg = SimpleNamespace(
        pred_len=int(meta["horizon_steps"]),
        output_attention=False,
        enc_in=1,
        dec_in=1,
        c_out=1,
        d_model=int(meta["d_model"]),
        n_heads=int(meta["n_heads"]),
        e_layers=int(meta["e_layers"]),
        d_layers=int(meta["d_layers"]),
        d_ff=int(meta["d_ff"]),
        factor=1,
        dropout=float(meta["dropout"]),
        embed="cycF",
        freq="cyc",
        activation="gelu",
    )
    return GPFormer.Model(cfg).float().to(device)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved GPFormer checkpoint on a test dataset (no training)."
    )
    parser.add_argument("--model-checkpoint", type=Path, required=True,
                        help="Path to model.pt produced by gpformer_*shot_raw.py")
    parser.add_argument("--meta-json", type=Path, default=None,
                        help="Optional meta.json path (defaults to <ckpt-dir>/meta.json)")

    parser.add_argument("--data-source", choices=["csv", "hf"], default="csv")
    parser.add_argument("--data-root-test", type=Path, default=DEFAULT_DATA_TEST,
                        help="Root folder containing per-dataset test subfolders")
    parser.add_argument("--hf-name", type=str, default=DEFAULT_HF_NAME)
    parser.add_argument("--hf-test-split", type=str, default="test")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Optional dataset name filter")

    parser.add_argument("--results-root", type=Path, default=None,
                        help="Where to write metrics/predictions (default: <ckpt-dir>/test_only_results)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--metric-mode", choices=["final", "all"], default="final")
    parser.add_argument("--eval-stride-steps", type=int, default=DEFAULT_EVAL_STRIDE_STEPS)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--shot-label", type=str, default=None,
                        help="Override shot label in output rows (default: from meta.json)")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Override model_name in output rows (default: from meta.json)")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    ckpt_path: Path = args.model_checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict, meta, scaler = load_checkpoint(ckpt_path, args.meta_json)

    results_root = args.results_root or (ckpt_path.parent / "test_only_results")
    log_path = setup_logging(results_root, DEFAULT_LOG_STEM)
    LOGGER.info("Logging to %s", log_path)
    LOGGER.info("Loaded checkpoint: %s", ckpt_path)

    device = resolve_device(args.device)
    LOGGER.info("Device: %s", device)

    cfg = ForecastConfig(
        context_hours=int(meta["context_hours"]),
        horizon_minutes=int(meta["horizon_minutes"]),
        context_steps=int(meta["context_steps"]),
        horizon_steps=int(meta["horizon_steps"]),
    )
    label_len = int(meta["label_len"])
    mean = float(scaler["mean"])
    std = float(scaler["std"])
    shot_label = args.shot_label or str(meta.get("shot", "test"))
    model_name = args.model_name or str(meta.get("model_name", "GPFormer"))

    LOGGER.info(
        "Config: ctx=%dh hor=%dm label_len=%d mean=%.4f std=%.4f shot=%s",
        cfg.context_hours, cfg.horizon_minutes, label_len, mean, std, shot_label,
    )

    model = build_model(meta, device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if args.data_source == "hf":
        allowed = set(args.datasets) if args.datasets else None
        test_subjects: Dict[str, Dict[str, pd.DataFrame]] = {}
        for dataset_name, subject_id, df in iter_glucofm_subjects_from_hf(
            args.hf_name, args.hf_test_split, datasets=allowed,
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
            raise FileNotFoundError(f"Test root not found: {test_root}")
        dataset_pairs = list(iter_test_dataset_dirs(test_root))
        if args.datasets is not None:
            allowed = set(args.datasets)
            dataset_pairs = [(n, p) for (n, p) in dataset_pairs if n in allowed]
        if not dataset_pairs:
            raise FileNotFoundError(f"No dataset folders found in {test_root}")

    cache_key_cols = (
        "participant_id", "context_hours", "horizon_minutes", "stride_steps", "metric_mode",
    )
    desired_order = [
        "dataset", "split", "shot", "model_name", "participant_id",
        "context_hours", "horizon_minutes", "context_steps", "horizon_steps",
        "step_minutes", "freq", "stride_steps", "metric_mode",
        "rmse", "mae", "windows", "label_len", "checkpoint_path",
    ]
    stride_steps = cfg.context_steps if args.eval_stride_steps <= 0 else args.eval_stride_steps

    for entry in dataset_pairs:
        if args.data_source == "hf":
            dataset_name, subjects = entry
            subject_iter = list(subjects.items())
        else:
            dataset_name, dataset_dir = entry
            test_groups = collect_subject_groups(dataset_dir)
            if not test_groups:
                LOGGER.warning("%s: empty test participants, skipping", dataset_name)
                continue
            subject_iter = list(test_groups.items())

        output_path = results_root / f"{dataset_name}_test_metrics.csv"
        predictions_path = results_root / f"{dataset_name}_test_predictions.csv"

        if args.overwrite:
            for p in (output_path, predictions_path):
                if p.exists():
                    try:
                        p.unlink()
                    except Exception as exc:
                        LOGGER.warning("Failed to remove %s (%s)", p, exc)
            completed_keys: Set[Tuple[object, ...]] = set()
        else:
            completed_keys = load_completed_keys(output_path, key_cols=cache_key_cols)
            if completed_keys:
                LOGGER.info("%s: %d existing rows, skipping unless --overwrite",
                            dataset_name, len(completed_keys))

        LOGGER.info("%s: test_participants=%d", dataset_name, len(subject_iter))

        records: List[Dict[str, object]] = []
        prediction_records: List[Dict[str, object]] = []

        for participant_id, payload in subject_iter:
            key = (
                participant_id, cfg.context_hours, cfg.horizon_minutes,
                int(stride_steps), str(args.metric_mode),
            )
            if key in completed_keys:
                continue
            try:
                df = payload if args.data_source == "hf" else load_subject(payload)
            except Exception as exc:
                LOGGER.warning("%s/test/%s: skipped (%s)", dataset_name, participant_id, exc)
                continue

            sse, sae, points, windows, pred_rows = evaluate_subject(
                model, participant_id, df, cfg,
                label_len=label_len, mean=mean, std=std,
                stride_steps=stride_steps, batch_size=args.eval_batch_size,
                metric_mode=args.metric_mode, device=device,
            )
            if points == 0 or windows == 0:
                continue

            rmse = math.sqrt(sse / points)
            mae = sae / points
            records.append({
                "dataset": dataset_name,
                "split": "test",
                "participant_id": participant_id,
                "context_hours": cfg.context_hours,
                "horizon_minutes": cfg.horizon_minutes,
                "context_steps": cfg.context_steps,
                "horizon_steps": cfg.horizon_steps,
                "step_minutes": STEP_MINUTES,
                "freq": FREQ,
                "stride_steps": int(stride_steps),
                "metric_mode": args.metric_mode,
                "rmse": rmse,
                "mae": mae,
                "windows": windows,
                "shot": shot_label,
                "model_name": model_name,
                "label_len": label_len,
                "checkpoint_path": str(ckpt_path),
            })
            if pred_rows:
                for row in pred_rows:
                    row.setdefault("dataset", dataset_name)
                    row.setdefault("shot", shot_label)
                    row.setdefault("model_name", model_name)
                    row.setdefault("stride_steps", int(stride_steps))
                    row.setdefault("metric_mode", str(args.metric_mode))
                prediction_records.extend(pred_rows)

        if records:
            new_df = pd.DataFrame(records)
            written = upsert_csv_rows(
                output_path, new_df,
                key_cols=cache_key_cols, desired_order=desired_order,
                sort_by=["participant_id", "context_hours", "horizon_minutes"],
            )
            LOGGER.info("Saved metrics to %s (+%d rows)", output_path, written)

        if prediction_records:
            pred_df = pd.DataFrame(prediction_records)
            pred_cols = [
                "dataset", "shot", "model_name", "participant_id",
                "context_hours", "horizon_minutes", "stride_steps", "metric_mode",
                "window_start_ts", "context_end_ts", "forecast_ts",
                "horizon_step", "horizon_offset_min",
                "prediction", "ground_truth",
            ]
            pred_df = pred_df[[c for c in pred_cols if c in pred_df.columns]]
            header = not predictions_path.exists()
            pred_df.to_csv(predictions_path, mode="a", index=False, header=header)
            LOGGER.info("Saved predictions to %s (+%d rows)", predictions_path, len(pred_df))

    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
