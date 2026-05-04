from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from results_cache import load_completed_keys, upsert_csv_rows  # noqa: E402
from glucofm_data import (  # noqa: E402
    DEFAULT_HF_NAME,
    iter_glucofm_subjects_from_hf,
)

import predict_glucose_multiwindow_timesfm_fullshot as fullshot  # noqa: E402

try:
    import timesfm
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    import timesfm


LOGGER = logging.getLogger(__name__)

MODEL_NAME = fullshot.MODEL_NAME
STEP_MINUTES = fullshot.STEP_MINUTES
FREQ = fullshot.FREQ

DEFAULT_EVAL_BATCH_SIZE = fullshot.DEFAULT_EVAL_BATCH_SIZE
DEFAULT_TRAIN_BATCH_SIZE = fullshot.DEFAULT_TRAIN_BATCH_SIZE
DEFAULT_EVAL_STRIDE_STEPS = fullshot.DEFAULT_EVAL_STRIDE_STEPS
DEFAULT_TRAIN_EPOCHS = fullshot.DEFAULT_TRAIN_EPOCHS
DEFAULT_TRAIN_STRIDE_STEPS = fullshot.DEFAULT_TRAIN_STRIDE_STEPS

DEFAULT_RESULTS_ROOT = (
    Path(__file__).resolve().parent / "multi_horizon_results_timesfm_fullshot_with_raw_quantile"
)
DEFAULT_LOG_STEM = "predict_glucose_multiwindow_timesfm_fullshot_with_raw_quantile"
DEFAULT_SHOT_LABEL = "full-shot"

# TimesFM 2.5 continuous-quantile-head output is (batch, horizon, q=10).
# Slot 5 is the median / AR point forecast; slots 1..9 correspond to quantile
# levels [0.1, 0.2, ..., 0.9] (i.e. slot i ↔ quantiles[i-1]). Slot 0 is a
# legacy point-head output and is NOT a quantile.
# Values returned by model.forecast(...) are already in original (mg/dL) space
# when ForecastConfig.normalize_inputs=True — do not re-apply mu/sigma.
Q10_IDX = 1
Q20_IDX = 2
Q80_IDX = 8
Q90_IDX = 9


@dataclass
class SubjectEvalResult:
    sse: float = 0.0
    ae: float = 0.0
    points: int = 0
    windows: int = 0
    pi80_covered_sum: float = 0.0
    pi80_width_sum: float = 0.0
    pi60_covered_sum: float = 0.0
    pi60_width_sum: float = 0.0
    pi_count: int = 0
    raw_predictions: List[Dict] = field(default_factory=list)
    quantile_predictions: List[Dict] = field(default_factory=list)


def evaluate_subject_with_raw_quantile(
    model: "timesfm.TimesFM_2p5_200M_torch",
    participant_id: str,
    df: pd.DataFrame,
    cfg: fullshot.ForecastConfig,
    stride_steps: int,
    batch_size: int,
    metric_mode: str,
) -> SubjectEvalResult:
    """Runs sliding-window evaluation. Collects aggregate metrics + per-datapoint raw and quantile records.

    Both point and quantile outputs from `model.forecast(...)` are already denormalized to
    mg/dL internally when ForecastConfig.normalize_inputs=True (see timesfm_2p5_torch.py's
    compiled_decode), so no extra mu/sigma rescaling is applied here.
    """
    result = SubjectEvalResult()
    if len(df) < cfg.context_steps + cfg.horizon_steps:
        return result

    values = df["value"].to_numpy(dtype="float32")
    timestamps = fullshot.normalize_timestamps(df["timestamp"])
    contiguous_runs = fullshot.consecutive_gap_counts(timestamps)
    total_len = cfg.context_steps + cfg.horizon_steps
    required_gaps = total_len - 1

    batch_histories: List[np.ndarray] = []
    batch_targets: List[np.ndarray] = []
    batch_starts: List[int] = []

    def flush_batch() -> None:
        nonlocal batch_histories, batch_targets, batch_starts
        if not batch_histories:
            return
        try:
            with torch.inference_mode():
                preds, quantile_preds = model.forecast(
                    horizon=cfg.horizon_steps,
                    inputs=batch_histories.copy(),
                )
        except Exception as exc:
            LOGGER.warning(
                "%s: prediction failure for %d windows (%s)",
                participant_id, len(batch_histories), exc,
            )
            batch_histories = []
            batch_targets = []
            batch_starts = []
            return

        preds = np.asarray(preds, dtype="float32")
        quantile_preds = np.asarray(quantile_preds, dtype="float32")

        for i, (pred_vec, target_vals) in enumerate(zip(preds, batch_targets)):
            start_idx = batch_starts[i]
            q_vec = quantile_preds[i]  # (horizon_steps, num_quantiles) already in mg/dL

            if metric_mode == "final":
                step = cfg.horizon_steps - 1
                pred_val = float(pred_vec[step])
                target_val = float(target_vals[step])
                if math.isnan(pred_val) or math.isnan(target_val):
                    continue
                err = pred_val - target_val
                result.sse += float(err * err)
                result.ae += float(abs(err))
                result.points += 1
                result.windows += 1

                q10 = float(q_vec[step, Q10_IDX])
                q20 = float(q_vec[step, Q20_IDX])
                q80 = float(q_vec[step, Q80_IDX])
                q90 = float(q_vec[step, Q90_IDX])
                result.pi80_covered_sum += float(q10 <= target_val <= q90)
                result.pi80_width_sum += q90 - q10
                result.pi60_covered_sum += float(q20 <= target_val <= q80)
                result.pi60_width_sum += q80 - q20
                result.pi_count += 1

                result.raw_predictions.append(
                    {
                        "window_start": int(start_idx),
                        "horizon_step": int(cfg.horizon_steps),
                        "prediction": pred_val,
                        "ground_truth": target_val,
                    }
                )
                result.quantile_predictions.append(
                    {
                        "window_start": int(start_idx),
                        "horizon_step": int(cfg.horizon_steps),
                        "prediction": pred_val,
                        "ground_truth": target_val,
                        "q10": q10,
                        "q20": q20,
                        "q80": q80,
                        "q90": q90,
                    }
                )
                continue

            # metric_mode == "all"
            valid_len = min(len(pred_vec), len(target_vals), cfg.horizon_steps)
            if valid_len == 0:
                continue
            pred_slice = pred_vec[:valid_len]
            targets = target_vals[:valid_len]
            if np.isnan(pred_slice).any() or np.isnan(targets).any():
                continue
            errors = pred_slice - targets
            result.sse += float(np.sum(errors ** 2))
            result.ae += float(np.sum(np.abs(errors)))
            result.points += int(valid_len)
            result.windows += 1

            for step_idx in range(valid_len):
                pv = float(pred_slice[step_idx])
                tgt = float(targets[step_idx])
                q10 = float(q_vec[step_idx, Q10_IDX])
                q20 = float(q_vec[step_idx, Q20_IDX])
                q80 = float(q_vec[step_idx, Q80_IDX])
                q90 = float(q_vec[step_idx, Q90_IDX])
                result.pi80_covered_sum += float(q10 <= tgt <= q90)
                result.pi80_width_sum += q90 - q10
                result.pi60_covered_sum += float(q20 <= tgt <= q80)
                result.pi60_width_sum += q80 - q20
                result.pi_count += 1

                result.raw_predictions.append(
                    {
                        "window_start": int(start_idx),
                        "horizon_step": int(step_idx + 1),
                        "prediction": pv,
                        "ground_truth": tgt,
                    }
                )
                result.quantile_predictions.append(
                    {
                        "window_start": int(start_idx),
                        "horizon_step": int(step_idx + 1),
                        "prediction": pv,
                        "ground_truth": tgt,
                        "q10": q10,
                        "q20": q20,
                        "q80": q80,
                        "q90": q90,
                    }
                )

        batch_histories = []
        batch_targets = []
        batch_starts = []

    for start in range(0, len(df) - cfg.context_steps - cfg.horizon_steps + 1, stride_steps):
        if int(contiguous_runs[start]) < required_gaps:
            continue
        end = start + cfg.context_steps
        target_end = end + cfg.horizon_steps

        history_values = values[start:end].astype("float32")
        target_values = values[end:target_end].astype("float32")
        if np.isnan(history_values).any() or np.isnan(target_values).any():
            continue

        batch_histories.append(history_values)
        batch_targets.append(target_values)
        batch_starts.append(start)
        if len(batch_histories) >= batch_size:
            flush_batch()

    flush_batch()
    return result


def _compile_for_eval(
    model: "timesfm.TimesFM_2p5_200M_torch",
    *,
    cfg: fullshot.ForecastConfig,
    device: torch.device,
    eval_batch_size: int,
) -> None:
    module = model.model
    module.to(device)
    module.device = device
    module.device_count = 1

    p = int(getattr(module, "p", 0) or 0)
    o = int(getattr(module, "o", 0) or 0)
    max_context = fullshot._round_up(cfg.context_steps, p) if p > 0 else cfg.context_steps
    max_horizon = fullshot._round_up(cfg.horizon_steps, o) if o > 0 else cfg.horizon_steps

    model.compile(
        timesfm.ForecastConfig(
            max_context=max_context,
            max_horizon=max_horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
            per_core_batch_size=max(1, int(eval_batch_size)),
        )
    )
    module.eval()


def load_pretrained_model(
    pretrained_root: Path,
    *,
    model_name: str,
    shot_label: str,
    cfg: fullshot.ForecastConfig,
    device: torch.device,
    eval_batch_size: int,
) -> Optional["timesfm.TimesFM_2p5_200M_torch"]:
    """Load a previously fine-tuned model.

    Resolution order:
      1. `pretrained_root/{shot_label}_ctx{H}h_hor{M}m/` — matches the layout written by
         `save_trained_model` when multi-config weights live under a single saved_models/ root.
      2. `pretrained_root` itself — when the user points directly at a specific checkpoint
         directory (e.g. `.../saved_models/few-shot_ctx12h_hor30m`).

    A directory qualifies if it contains `config.json`, `pytorch_model.bin`, or `model.pt`.
    Returns None if no matching checkpoint exists.
    """
    candidates = [
        pretrained_root / f"{shot_label}_ctx{cfg.context_hours}h_hor{cfg.horizon_minutes}m",
        pretrained_root,
    ]

    chosen: Optional[Path] = None
    for candidate in candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue
        if (
            (candidate / "config.json").exists()
            or (candidate / "pytorch_model.bin").exists()
            or (candidate / "model.pt").exists()
        ):
            chosen = candidate
            break

    if chosen is None:
        return None

    LOGGER.info(
        "Loading pretrained model from %s (cfg ctx=%dh hor=%dm)",
        chosen, cfg.context_hours, cfg.horizon_minutes,
    )

    if (chosen / "config.json").exists():
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(str(chosen))
    else:
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_name)
        if (chosen / "pytorch_model.bin").exists():
            state = torch.load(chosen / "pytorch_model.bin", map_location=device)
            model.model.load_state_dict(state)
        elif (chosen / "model.pt").exists():
            loaded = torch.load(chosen / "model.pt", map_location=device)
            if isinstance(loaded, dict):
                model.model.load_state_dict(loaded)
            else:
                model = loaded  # type: ignore[assignment]

    _compile_for_eval(model, cfg=cfg, device=device, eval_batch_size=eval_batch_size)
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
            f"{shot_label.capitalize()}: fine-tune (or reuse a pretrained) TimesFM and evaluate "
            "each dataset's test split, emitting per-datapoint point and quantile predictions."
        )
    )
    parser.add_argument("--model-name", type=str, default=MODEL_NAME, help="HuggingFace model id to load.")
    parser.add_argument(
        "--data-root-train",
        type=Path,
        default=fullshot.DATA_SPLITS["train"],
        help="Directory containing training CSV files (default: hf_cache/train/mixed).",
    )
    parser.add_argument(
        "--data-root-test",
        type=Path,
        default=fullshot.DATA_SPLITS["test"],
        help="Root directory containing dataset subfolders for the test split (default: hf_cache/test).",
    )
    parser.add_argument(
        "--data-source",
        choices=["csv", "hf"],
        default="csv",
        help="Data source: 'csv' reads per-subject CSVs; 'hf' loads GlucoFM from HuggingFace.",
    )
    parser.add_argument("--hf-name", type=str, default=DEFAULT_HF_NAME)
    parser.add_argument("--hf-train-split", type=str, default="train")
    parser.add_argument("--hf-test-split", type=str, default="test")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset folder names to include.")
    parser.add_argument("--context-hours", type=int, nargs="*", default=[12], help="Context window hours.")
    parser.add_argument(
        "--horizons-minutes",
        type=int,
        nargs="*",
        default=[30],
        help="Forecast horizon in minutes.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device: auto/cpu/cuda.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metric-mode", choices=["final", "all"], default="final")
    parser.add_argument(
        "--eval-stride-steps",
        type=int,
        default=default_eval_stride_steps,
        help="Eval sliding window stride (0 = context_steps; 1 = 5 minutes).",
    )
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing CSVs.")

    parser.add_argument(
        "--pretrained-model-dir",
        type=Path,
        default=None,
        help=(
            "Absolute path to a pretrained/fine-tuned checkpoint. "
            "Can be either a saved_models/ root containing "
            "'{shot_label}_ctx{H}h_hor{M}m/' subfolders, OR a specific checkpoint directory "
            "(e.g. '.../saved_models/few-shot_ctx12h_hor30m'). "
            "If set, fine-tuning is skipped for configs whose checkpoint is found."
        ),
    )

    parser.add_argument(
        "--save-raw-predictions",
        dest="save_raw_predictions",
        action="store_true",
        default=True,
        help="Write per-datapoint point predictions to raw_predictions/ (default: enabled).",
    )
    parser.add_argument(
        "--no-save-raw-predictions",
        dest="save_raw_predictions",
        action="store_false",
    )
    parser.add_argument(
        "--save-quantile-predictions",
        dest="save_quantile_predictions",
        action="store_true",
        default=True,
        help="Write per-datapoint quantile predictions to quantile_predictions/ (default: enabled).",
    )
    parser.add_argument(
        "--no-save-quantile-predictions",
        dest="save_quantile_predictions",
        action="store_false",
    )

    parser.add_argument(
        "--save-trained-model",
        dest="save_trained_model",
        action="store_true",
        default=True,
        help="Save fine-tuned weights under results_root/saved_models/ (default: enabled).",
    )
    parser.add_argument(
        "--no-save-trained-model",
        dest="save_trained_model",
        action="store_false",
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
    parser.add_argument("--train-loss-mode", choices=["final", "all"], default="all")
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args(argv)


METRICS_KEY_COLS = ("participant_id", "context_hours", "horizon_minutes", "stride_steps", "metric_mode")
RAW_KEY_COLS = METRICS_KEY_COLS + ("window_start", "horizon_step")
QUANT_KEY_COLS = RAW_KEY_COLS

METRICS_DESIRED_ORDER = [
    "dataset", "split", "shot", "model_name", "pretrained_model_dir",
    "participant_id", "context_hours", "horizon_minutes",
    "context_steps", "horizon_steps", "step_minutes", "freq",
    "stride_steps", "metric_mode",
    "rmse", "mae", "windows",
    "pi80_coverage", "pi80_width", "pi60_coverage", "pi60_width",
    "train_epochs", "train_batch_size", "train_stride_steps",
    "max_train_steps", "max_train_windows",
    "lr", "weight_decay", "train_loss_mode",
]

RAW_DESIRED_ORDER = [
    "dataset", "split", "participant_id",
    "context_hours", "horizon_minutes", "stride_steps", "metric_mode",
    "window_start", "horizon_step", "prediction", "ground_truth",
]

QUANT_DESIRED_ORDER = [
    "dataset", "split", "participant_id",
    "context_hours", "horizon_minutes", "stride_steps", "metric_mode",
    "window_start", "horizon_step",
    "prediction", "ground_truth",
    "q10", "q20", "q80", "q90",
]


def _flush_records(
    output_path: Path,
    records: List[Dict[str, object]],
    *,
    key_cols: Sequence[str],
    desired_order: Sequence[str],
    sort_by: Sequence[str],
    completed_keys: Set[Tuple[object, ...]],
    completed_key_cols: Sequence[str],
    label: str,
) -> None:
    if not records:
        return
    new_df = pd.DataFrame(records)
    written = upsert_csv_rows(
        output_path,
        new_df,
        key_cols=key_cols,
        desired_order=desired_order,
        sort_by=list(sort_by),
    )
    LOGGER.info("Saved %s to %s (+%d rows)", label, output_path, written)

    # Track completed membership keyed by the (narrower) participant-level tuple.
    for row in new_df.itertuples(index=False):
        key = tuple(_normalize_key_value(col, getattr(row, col)) for col in completed_key_cols)
        completed_keys.add(key)
    records.clear()


def _normalize_key_value(col: str, value: object) -> object:
    if col in {"context_hours", "horizon_minutes", "stride_steps", "window_start", "horizon_step"}:
        return int(value)
    if col == "participant_id":
        return str(value)
    if col == "metric_mode":
        return str(value)
    return value


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
    log_path = fullshot.setup_logging(results_root, log_stem)
    LOGGER.info("Logging to %s", log_path)

    raw_dir = results_root / "raw_predictions"
    quant_dir = results_root / "quantile_predictions"
    if args.save_raw_predictions:
        raw_dir.mkdir(parents=True, exist_ok=True)
    if args.save_quantile_predictions:
        quant_dir.mkdir(parents=True, exist_ok=True)

    device = fullshot.resolve_device(args.device)
    LOGGER.info("Device: %s", device)

    configs = [
        fullshot.ForecastConfig(
            context_hours=ctx,
            horizon_minutes=hor,
            context_steps=fullshot.context_steps_from_hours(ctx),
            horizon_steps=fullshot.horizon_steps_from_minutes(hor),
        )
        for ctx in args.context_hours
        for hor in args.horizons_minutes
    ]

    # --- training data (skipped when we'll only use pretrained checkpoints) ---
    train_series: Dict[str, fullshot.SeriesCache] = {}
    if args.pretrained_model_dir is None:
        if args.data_source == "hf":
            LOGGER.info("Training pool: HF %s split=%s", args.hf_name, args.hf_train_split)
            for dataset_name, subject_id, df in iter_glucofm_subjects_from_hf(
                args.hf_name, args.hf_train_split
            ):
                participant_id = f"{dataset_name}__{subject_id}"
                try:
                    train_series[participant_id] = fullshot.load_series_cache(df)
                except Exception as exc:
                    LOGGER.warning("train/%s: skipped (%s)", participant_id, exc)
        else:
            train_root: Path = args.data_root_train
            if not train_root.exists():
                raise FileNotFoundError(
                    f"Train root not found: {train_root}\n"
                    "Tip: prepare the dataset with:\n  python prepare_dataset.py --create-mixed"
                )
            train_groups = fullshot.collect_subject_groups(train_root)
            if not train_groups:
                raise FileNotFoundError(f"No training CSV files found under {train_root}")
            LOGGER.info("Training pool: %s (%d participants)", train_root, len(train_groups))
            for participant_id, csv_paths in train_groups.items():
                try:
                    df = fullshot.load_subject(csv_paths)
                    train_series[participant_id] = fullshot.load_series_cache(df)
                except Exception as exc:
                    LOGGER.warning("train/%s: skipped (%s)", participant_id, exc)

        if not train_series:
            raise ValueError("No valid training series loaded.")
    else:
        LOGGER.info("--pretrained-model-dir=%s; skipping training data load.", args.pretrained_model_dir)

    # --- test data setup ---
    dataset_state: Dict[str, Dict[str, object]] = {}
    if args.data_source == "hf":
        allowed = set(args.datasets) if args.datasets is not None else None
        test_subjects: Dict[str, Dict[str, pd.DataFrame]] = {}
        for dataset_name, subject_id, df in iter_glucofm_subjects_from_hf(
            args.hf_name, args.hf_test_split, datasets=allowed
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
        dataset_dirs = sorted(p for p in test_root.iterdir() if p.is_dir())
        if args.datasets is not None:
            allowed = set(args.datasets)
            dataset_dirs = [p for p in dataset_dirs if p.name in allowed]
        if not dataset_dirs:
            raise FileNotFoundError(f"No dataset folders found in {test_root}")
        dataset_pairs = [(p.name, p) for p in dataset_dirs]

    for dataset_name, payload in dataset_pairs:
        metrics_path = results_root / f"{dataset_name}_test_metrics.csv"
        raw_path = raw_dir / f"{dataset_name}_test_raw_predictions.csv"
        quant_path = quant_dir / f"{dataset_name}_test_quantile_predictions.csv"

        completed_metrics: Set[Tuple[object, ...]] = set()
        completed_raw: Set[Tuple[object, ...]] = set()
        completed_quant: Set[Tuple[object, ...]] = set()

        if args.overwrite:
            for path in (metrics_path, raw_path, quant_path):
                if path.exists():
                    try:
                        path.unlink()
                    except Exception as exc:
                        LOGGER.warning("Failed to remove %s (%s)", path, exc)
        else:
            completed_metrics = load_completed_keys(metrics_path, key_cols=METRICS_KEY_COLS)
            if args.save_raw_predictions:
                completed_raw = load_completed_keys(raw_path, key_cols=METRICS_KEY_COLS)
            if args.save_quantile_predictions:
                completed_quant = load_completed_keys(quant_path, key_cols=METRICS_KEY_COLS)
            if completed_metrics:
                LOGGER.info(
                    "%s: found %d existing metrics rows (raw=%d, quantile=%d).",
                    dataset_name, len(completed_metrics), len(completed_raw), len(completed_quant),
                )

        if args.data_source == "hf":
            subjects = payload
            participant_ids = list(subjects.keys())
            if not participant_ids:
                LOGGER.warning("%s: empty HF test participants, skipping", dataset_name)
                continue
            LOGGER.info("%s: test_participants=%d (HF)", dataset_name, len(participant_ids))
            dataset_state[dataset_name] = {
                "metrics_path": metrics_path,
                "raw_path": raw_path,
                "quant_path": quant_path,
                "completed_metrics": completed_metrics,
                "completed_raw": completed_raw,
                "completed_quant": completed_quant,
                "participant_ids": participant_ids,
                "test_subjects": subjects,
                "metrics_records": [],
                "raw_records": [],
                "quant_records": [],
            }
        else:
            dataset_dir = payload
            test_groups = fullshot.collect_subject_groups(dataset_dir)
            if not test_groups:
                LOGGER.warning("%s: empty test participants, skipping", dataset_name)
                continue
            participant_ids = list(test_groups.keys())
            LOGGER.info("%s: test_participants=%d", dataset_name, len(participant_ids))
            dataset_state[dataset_name] = {
                "metrics_path": metrics_path,
                "raw_path": raw_path,
                "quant_path": quant_path,
                "completed_metrics": completed_metrics,
                "completed_raw": completed_raw,
                "completed_quant": completed_quant,
                "participant_ids": participant_ids,
                "test_groups": test_groups,
                "metrics_records": [],
                "raw_records": [],
                "quant_records": [],
            }

    if not dataset_state:
        raise FileNotFoundError("No usable test datasets found.")

    # --- per-config: load-or-finetune, then evaluate and emit ---
    for cfg in configs:
        stride_steps = cfg.context_steps if args.eval_stride_steps <= 0 else args.eval_stride_steps

        missing_any = args.overwrite
        if not missing_any:
            for info in dataset_state.values():
                completed_m = info["completed_metrics"]
                completed_r = info["completed_raw"]
                completed_q = info["completed_quant"]
                for participant_id in info["participant_ids"]:
                    key = (
                        str(participant_id),
                        int(cfg.context_hours),
                        int(cfg.horizon_minutes),
                        int(stride_steps),
                        str(args.metric_mode),
                    )
                    metrics_done = key in completed_m
                    raw_done = (not args.save_raw_predictions) or key in completed_r
                    quant_done = (not args.save_quantile_predictions) or key in completed_q
                    if not (metrics_done and raw_done and quant_done):
                        missing_any = True
                        break
                if missing_any:
                    break
        if not missing_any:
            LOGGER.info("All outputs already complete for ctx=%dh hor=%dm; skipping.", cfg.context_hours, cfg.horizon_minutes)
            continue

        LOGGER.info("Processing cfg: ctx=%dh hor=%dm", cfg.context_hours, cfg.horizon_minutes)

        model: Optional["timesfm.TimesFM_2p5_200M_torch"] = None
        used_pretrained = False
        if args.pretrained_model_dir is not None:
            model = load_pretrained_model(
                args.pretrained_model_dir,
                model_name=args.model_name,
                shot_label=shot_label,
                cfg=cfg,
                device=device,
                eval_batch_size=args.eval_batch_size,
            )
            if model is not None:
                used_pretrained = True
            else:
                LOGGER.warning(
                    "No pretrained checkpoint found under %s for ctx=%dh hor=%dm; skipping.",
                    args.pretrained_model_dir, cfg.context_hours, cfg.horizon_minutes,
                )
                continue

        if model is None:
            try:
                model = fullshot.finetune_fullshot(
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
                    loss_mode=args.train_loss_mode,
                    log_every=args.log_every,
                    eval_batch_size=args.eval_batch_size,
                )
            except Exception as exc:
                LOGGER.warning("Finetune failed ctx=%dh hor=%dm (%s)", cfg.context_hours, cfg.horizon_minutes, exc)
                continue

            if args.save_trained_model:
                try:
                    saved_path = fullshot.save_trained_model(
                        model,
                        results_root=results_root,
                        shot_label=shot_label,
                        cfg=cfg,
                    )
                    LOGGER.info("Saved trained model to %s", saved_path)
                except Exception as exc:
                    LOGGER.warning("Failed to save trained model (%s)", exc)

        # --- evaluate + record ---
        for dataset_name, info in dataset_state.items():
            completed_m: Set[Tuple[object, ...]] = info["completed_metrics"]
            completed_r: Set[Tuple[object, ...]] = info["completed_raw"]
            completed_q: Set[Tuple[object, ...]] = info["completed_quant"]
            metrics_records: List[Dict[str, object]] = info["metrics_records"]
            raw_records: List[Dict[str, object]] = info["raw_records"]
            quant_records: List[Dict[str, object]] = info["quant_records"]

            if args.data_source == "hf":
                subject_iter = info["test_subjects"].items()
            else:
                subject_iter = info["test_groups"].items()

            for participant_id, subject_payload in subject_iter:
                key = (
                    str(participant_id),
                    int(cfg.context_hours),
                    int(cfg.horizon_minutes),
                    int(stride_steps),
                    str(args.metric_mode),
                )
                metrics_done = key in completed_m
                raw_done = (not args.save_raw_predictions) or key in completed_r
                quant_done = (not args.save_quantile_predictions) or key in completed_q
                if metrics_done and raw_done and quant_done:
                    continue

                try:
                    if args.data_source == "hf":
                        df = subject_payload
                    else:
                        df = fullshot.load_subject(subject_payload)
                except Exception as exc:
                    LOGGER.warning("%s/test/%s: skipped (%s)", dataset_name, participant_id, exc)
                    continue

                result = evaluate_subject_with_raw_quantile(
                    model,
                    participant_id,
                    df,
                    cfg,
                    stride_steps=stride_steps,
                    batch_size=args.eval_batch_size,
                    metric_mode=args.metric_mode,
                )
                if result.points == 0 or result.windows == 0:
                    continue

                rmse = math.sqrt(result.sse / result.points)
                mae = result.ae / result.points
                if result.pi_count > 0:
                    pi80_cov = result.pi80_covered_sum / result.pi_count
                    pi80_w = result.pi80_width_sum / result.pi_count
                    pi60_cov = result.pi60_covered_sum / result.pi_count
                    pi60_w = result.pi60_width_sum / result.pi_count
                else:
                    pi80_cov = pi80_w = pi60_cov = pi60_w = float("nan")

                common_key_cols = {
                    "dataset": dataset_name,
                    "split": "test",
                    "participant_id": participant_id,
                    "context_hours": cfg.context_hours,
                    "horizon_minutes": cfg.horizon_minutes,
                    "stride_steps": stride_steps,
                    "metric_mode": args.metric_mode,
                }

                if not metrics_done:
                    metrics_records.append(
                        {
                            **common_key_cols,
                            "context_steps": cfg.context_steps,
                            "horizon_steps": cfg.horizon_steps,
                            "step_minutes": STEP_MINUTES,
                            "freq": FREQ,
                            "rmse": rmse,
                            "mae": mae,
                            "windows": result.windows,
                            "pi80_coverage": pi80_cov,
                            "pi80_width": pi80_w,
                            "pi60_coverage": pi60_cov,
                            "pi60_width": pi60_w,
                            "shot": shot_label,
                            "model_name": args.model_name,
                            "pretrained_model_dir": str(args.pretrained_model_dir) if used_pretrained else "",
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

                if args.save_raw_predictions and not raw_done:
                    for rp in result.raw_predictions:
                        raw_records.append({**common_key_cols, **rp})

                if args.save_quantile_predictions and not quant_done:
                    for qp in result.quantile_predictions:
                        quant_records.append({**common_key_cols, **qp})

        # --- flush per-dataset records after each (ctx, horizon) ---
        for dataset_name, info in dataset_state.items():
            _flush_records(
                info["metrics_path"],
                info["metrics_records"],
                key_cols=METRICS_KEY_COLS,
                desired_order=METRICS_DESIRED_ORDER,
                sort_by=["participant_id", "context_hours", "horizon_minutes"],
                completed_keys=info["completed_metrics"],
                completed_key_cols=METRICS_KEY_COLS,
                label="metrics",
            )
            if args.save_raw_predictions:
                _flush_records(
                    info["raw_path"],
                    info["raw_records"],
                    key_cols=RAW_KEY_COLS,
                    desired_order=RAW_DESIRED_ORDER,
                    sort_by=["participant_id", "context_hours", "horizon_minutes", "window_start", "horizon_step"],
                    completed_keys=info["completed_raw"],
                    completed_key_cols=METRICS_KEY_COLS,
                    label="raw predictions",
                )
            if args.save_quantile_predictions:
                _flush_records(
                    info["quant_path"],
                    info["quant_records"],
                    key_cols=QUANT_KEY_COLS,
                    desired_order=QUANT_DESIRED_ORDER,
                    sort_by=["participant_id", "context_hours", "horizon_minutes", "window_start", "horizon_step"],
                    completed_keys=info["completed_quant"],
                    completed_key_cols=METRICS_KEY_COLS,
                    label="quantile predictions",
                )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
