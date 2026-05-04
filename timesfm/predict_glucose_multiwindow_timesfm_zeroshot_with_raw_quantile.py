"""
Zero-shot TimesFM evaluation that emits per-datapoint point predictions and
quantile predictions, with configurable context windows and horizons.

Mirrors the layout of predict_glucose_multiwindow_timesfm_{full,few}shot_with_raw_quantile.py:
  results_root/                                 (default: multi_horizon_results_timesfm_zeroshot_with_raw_quantile/)
    ├── {dataset}_test_metrics.csv              # rmse/mae/PI80/PI60 per (participant, ctx, horizon)
    ├── raw_predictions/
    │     └── {dataset}_test_raw_predictions.csv
    └── quantile_predictions/
          └── {dataset}_test_quantile_predictions.csv

Combines (and replaces) the previous two scripts:
  - predict_glucose_multiwindow_timesfm_zeroshot_quantile.py   (quantile-only, fixed 12h ctx)
  - predict_glucose_multiwindow_timesfm_zeroshot_with_raw.py   (raw-only, hardcoded ctx list)

Reuses building blocks from predict_glucose_multiwindow_timesfm_fullshot{,_with_raw_quantile}.py.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from results_cache import load_completed_keys  # noqa: E402

import predict_glucose_multiwindow_timesfm_fullshot as fullshot  # noqa: E402
import predict_glucose_multiwindow_timesfm_fullshot_with_raw_quantile as fullshot_rq  # noqa: E402

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
DEFAULT_EVAL_STRIDE_STEPS = fullshot.DEFAULT_EVAL_STRIDE_STEPS

DEFAULT_RESULTS_ROOT = (
    Path(__file__).resolve().parent
    / "multi_horizon_results_timesfm_zeroshot_with_raw_quantile"
)
DEFAULT_LOG_STEM = "predict_glucose_multiwindow_timesfm_zeroshot_with_raw_quantile"
SHOT_LABEL = "zero-shot"

DEFAULT_CONTEXT_HOURS = [1, 4, 8, 12, 16, 24]
DEFAULT_HORIZONS_MINUTES = [15, 30, 60, 90]


# Reuse the column-key contracts from fullshot_rq so output schemas match exactly.
METRICS_KEY_COLS = fullshot_rq.METRICS_KEY_COLS
RAW_KEY_COLS     = fullshot_rq.RAW_KEY_COLS
QUANT_KEY_COLS   = fullshot_rq.QUANT_KEY_COLS

# Drop fine-tune-only columns from the metrics ordering — zero-shot doesn't have them.
_TRAIN_ONLY = {
    "train_epochs", "train_batch_size", "train_stride_steps",
    "max_train_steps", "max_train_windows",
    "lr", "weight_decay", "train_loss_mode",
    "pretrained_model_dir",
}
METRICS_DESIRED_ORDER = [c for c in fullshot_rq.METRICS_DESIRED_ORDER if c not in _TRAIN_ONLY]
RAW_DESIRED_ORDER     = fullshot_rq.RAW_DESIRED_ORDER
QUANT_DESIRED_ORDER   = fullshot_rq.QUANT_DESIRED_ORDER


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Zero-shot TimesFM evaluation that emits per-datapoint point and "
            "quantile predictions across configurable context windows and horizons."
        )
    )
    parser.add_argument("--model-name", type=str, default=MODEL_NAME,
                        help="HuggingFace model id to load.")
    parser.add_argument("--data-root-test", type=Path,
                        default=fullshot.DATA_SPLITS["test"],
                        help="Root directory containing dataset subfolders for the test split.")
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help=(
            "Either (a) a list of dataset subfolder names to include "
            "(filters --data-root-test), or (b) a single existing directory path "
            "that is treated as --data-root-test (all subfolders evaluated). "
            "Default: include all subfolders of --data-root-test."
        ),
    )
    parser.add_argument("--context-hours", type=int, nargs="+",
                        default=DEFAULT_CONTEXT_HOURS,
                        help="Context window(s) in hours.")
    parser.add_argument("--horizons-minutes", type=int, nargs="+",
                        default=DEFAULT_HORIZONS_MINUTES,
                        help="Forecast horizon(s) in minutes.")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto/cpu/cuda.")
    parser.add_argument("--metric-mode", choices=["final", "all"], default="final",
                        help="Use only the final horizon step or all steps for error metrics.")
    parser.add_argument("--eval-stride-steps", type=int,
                        default=DEFAULT_EVAL_STRIDE_STEPS,
                        help="Eval sliding window stride (0 = context_steps; 1 = 5 minutes).")
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE,
                        help="Inference batch size.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute and overwrite existing CSVs instead of resuming.")

    parser.add_argument("--save-raw-predictions",
                        dest="save_raw_predictions",
                        action="store_true", default=True,
                        help="Write per-datapoint point predictions (default: enabled).")
    parser.add_argument("--no-save-raw-predictions",
                        dest="save_raw_predictions", action="store_false")
    parser.add_argument("--save-quantile-predictions",
                        dest="save_quantile_predictions",
                        action="store_true", default=True,
                        help="Write per-datapoint quantile predictions (default: enabled).")
    parser.add_argument("--no-save-quantile-predictions",
                        dest="save_quantile_predictions", action="store_false")
    return parser.parse_args(argv)


def _load_zeroshot_model(
    model_name: str,
    cfg: fullshot.ForecastConfig,
    device: torch.device,
    eval_batch_size: int,
) -> "timesfm.TimesFM_2p5_200M_torch":
    """Load the pretrained TimesFM and compile it for the given (ctx, horizon)."""
    LOGGER.info(
        "Loading zero-shot TimesFM '%s' (ctx=%dh hor=%dm)",
        model_name, cfg.context_hours, cfg.horizon_minutes,
    )
    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_name)
    fullshot_rq._compile_for_eval(
        model, cfg=cfg, device=device, eval_batch_size=eval_batch_size,
    )
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved  = torch.cuda.memory_reserved() / 1024**2
        LOGGER.info("CUDA memory allocated: %.1f MB | reserved: %.1f MB", allocated, reserved)
    return model


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    log_stem: str = DEFAULT_LOG_STEM,
    shot_label: str = SHOT_LABEL,
) -> None:
    args = parse_args(argv)
    log_path = fullshot.setup_logging(results_root, log_stem)
    LOGGER.info("Logging to %s", log_path)

    # If --datasets is a single existing directory path, treat it as the test
    # root so callers can pass `--datasets ./test_dataset` instead of having to
    # remember --data-root-test.
    if args.datasets is not None and len(args.datasets) == 1:
        candidate = Path(args.datasets[0])
        if candidate.is_dir():
            LOGGER.info(
                "--datasets points at directory %s; using as --data-root-test.",
                candidate,
            )
            args.data_root_test = candidate
            args.datasets = None

    raw_dir   = results_root / "raw_predictions"
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

    test_root: Path = args.data_root_test
    if not test_root.exists():
        raise FileNotFoundError(f"Test root not found: {test_root}")
    dataset_dirs = sorted(p for p in test_root.iterdir() if p.is_dir())
    if args.datasets is not None:
        allowed = set(args.datasets)
        dataset_dirs = [p for p in dataset_dirs if p.name in allowed]
    if not dataset_dirs:
        raise FileNotFoundError(f"No dataset folders found in {test_root}")

    # ── per-dataset state (paths + completed-key sets + record buffers) ──
    dataset_state: Dict[str, Dict[str, object]] = {}
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        metrics_path = results_root / f"{dataset_name}_test_metrics.csv"
        raw_path     = raw_dir   / f"{dataset_name}_test_raw_predictions.csv"
        quant_path   = quant_dir / f"{dataset_name}_test_quantile_predictions.csv"

        completed_metrics: Set[Tuple[object, ...]] = set()
        completed_raw:     Set[Tuple[object, ...]] = set()
        completed_quant:   Set[Tuple[object, ...]] = set()

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
                    dataset_name, len(completed_metrics),
                    len(completed_raw), len(completed_quant),
                )

        test_groups = fullshot.collect_subject_groups(dataset_dir)
        if not test_groups:
            LOGGER.warning("%s: empty test participants, skipping", dataset_name)
            continue
        participant_ids = list(test_groups.keys())
        LOGGER.info("%s: test_participants=%d", dataset_name, len(participant_ids))

        dataset_state[dataset_name] = {
            "metrics_path":     metrics_path,
            "raw_path":         raw_path,
            "quant_path":       quant_path,
            "completed_metrics": completed_metrics,
            "completed_raw":     completed_raw,
            "completed_quant":   completed_quant,
            "participant_ids":  participant_ids,
            "test_groups":      test_groups,
            "metrics_records":  [],
            "raw_records":      [],
            "quant_records":    [],
        }

    if not dataset_state:
        raise FileNotFoundError("No usable test datasets found.")

    # ── per-config: load model fresh, evaluate, flush ──
    for cfg in configs:
        stride_steps = cfg.context_steps if args.eval_stride_steps <= 0 else args.eval_stride_steps

        # Skip cfg entirely if every (dataset, participant) is already complete.
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
                    raw_done   = (not args.save_raw_predictions)      or key in completed_r
                    quant_done = (not args.save_quantile_predictions) or key in completed_q
                    if not (metrics_done and raw_done and quant_done):
                        missing_any = True
                        break
                if missing_any:
                    break
        if not missing_any:
            LOGGER.info(
                "All outputs already complete for ctx=%dh hor=%dm; skipping.",
                cfg.context_hours, cfg.horizon_minutes,
            )
            continue

        LOGGER.info("Processing cfg: ctx=%dh hor=%dm", cfg.context_hours, cfg.horizon_minutes)

        try:
            model = _load_zeroshot_model(args.model_name, cfg, device, args.eval_batch_size)
        except Exception as exc:
            LOGGER.warning(
                "Model load failed ctx=%dh hor=%dm (%s)",
                cfg.context_hours, cfg.horizon_minutes, exc,
            )
            continue

        for dataset_name, info in dataset_state.items():
            completed_m: Set[Tuple[object, ...]] = info["completed_metrics"]
            completed_r: Set[Tuple[object, ...]] = info["completed_raw"]
            completed_q: Set[Tuple[object, ...]] = info["completed_quant"]
            metrics_records: List[Dict[str, object]] = info["metrics_records"]
            raw_records:     List[Dict[str, object]] = info["raw_records"]
            quant_records:   List[Dict[str, object]] = info["quant_records"]

            for participant_id, csv_paths in info["test_groups"].items():
                key = (
                    str(participant_id),
                    int(cfg.context_hours),
                    int(cfg.horizon_minutes),
                    int(stride_steps),
                    str(args.metric_mode),
                )
                metrics_done = key in completed_m
                raw_done   = (not args.save_raw_predictions)      or key in completed_r
                quant_done = (not args.save_quantile_predictions) or key in completed_q
                if metrics_done and raw_done and quant_done:
                    continue

                try:
                    df = fullshot.load_subject(csv_paths)
                except Exception as exc:
                    LOGGER.warning("%s/test/%s: skipped (%s)", dataset_name, participant_id, exc)
                    continue

                result = fullshot_rq.evaluate_subject_with_raw_quantile(
                    model, participant_id, df, cfg,
                    stride_steps=stride_steps,
                    batch_size=args.eval_batch_size,
                    metric_mode=args.metric_mode,
                )
                if result.points == 0 or result.windows == 0:
                    continue

                rmse = math.sqrt(result.sse / result.points)
                mae  = result.ae / result.points
                if result.pi_count > 0:
                    pi80_cov = result.pi80_covered_sum / result.pi_count
                    pi80_w   = result.pi80_width_sum   / result.pi_count
                    pi60_cov = result.pi60_covered_sum / result.pi_count
                    pi60_w   = result.pi60_width_sum   / result.pi_count
                else:
                    pi80_cov = pi80_w = pi60_cov = pi60_w = float("nan")

                common = {
                    "dataset": dataset_name,
                    "split": "test",
                    "participant_id": participant_id,
                    "context_hours": cfg.context_hours,
                    "horizon_minutes": cfg.horizon_minutes,
                    "stride_steps": stride_steps,
                    "metric_mode": args.metric_mode,
                }

                if not metrics_done:
                    metrics_records.append({
                        **common,
                        "context_steps": cfg.context_steps,
                        "horizon_steps": cfg.horizon_steps,
                        "step_minutes":  STEP_MINUTES,
                        "freq":          FREQ,
                        "rmse":          rmse,
                        "mae":           mae,
                        "windows":       result.windows,
                        "pi80_coverage": pi80_cov,
                        "pi80_width":    pi80_w,
                        "pi60_coverage": pi60_cov,
                        "pi60_width":    pi60_w,
                        "shot":          shot_label,
                        "model_name":    args.model_name,
                    })

                if args.save_raw_predictions and not raw_done:
                    for rp in result.raw_predictions:
                        raw_records.append({**common, **rp})

                if args.save_quantile_predictions and not quant_done:
                    for qp in result.quantile_predictions:
                        quant_records.append({**common, **qp})

        # Flush per-dataset records after each (ctx, horizon) config.
        for dataset_name, info in dataset_state.items():
            fullshot_rq._flush_records(
                info["metrics_path"], info["metrics_records"],
                key_cols=METRICS_KEY_COLS,
                desired_order=METRICS_DESIRED_ORDER,
                sort_by=["participant_id", "context_hours", "horizon_minutes"],
                completed_keys=info["completed_metrics"],
                completed_key_cols=METRICS_KEY_COLS,
                label="metrics",
            )
            if args.save_raw_predictions:
                fullshot_rq._flush_records(
                    info["raw_path"], info["raw_records"],
                    key_cols=RAW_KEY_COLS,
                    desired_order=RAW_DESIRED_ORDER,
                    sort_by=["participant_id", "context_hours", "horizon_minutes",
                             "window_start", "horizon_step"],
                    completed_keys=info["completed_raw"],
                    completed_key_cols=METRICS_KEY_COLS,
                    label="raw predictions",
                )
            if args.save_quantile_predictions:
                fullshot_rq._flush_records(
                    info["quant_path"], info["quant_records"],
                    key_cols=QUANT_KEY_COLS,
                    desired_order=QUANT_DESIRED_ORDER,
                    sort_by=["participant_id", "context_hours", "horizon_minutes",
                             "window_start", "horizon_step"],
                    completed_keys=info["completed_quant"],
                    completed_key_cols=METRICS_KEY_COLS,
                    label="quantile predictions",
                )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
