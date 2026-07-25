"""Runtime benchmark for Chronos-2 (zero-shot / LoRA few-shot / LoRA full-shot).

Chronos does not fit the `predict_glucose_multiwindow_*` harness: fine-tuning
runs through `Chronos2Pipeline.fit(..., num_steps=N)` inside `fullshot.py`, and
that script's `main()` would evaluate all 529 test subjects.  So rather than
calling `main()`, this bench calls the method's own building blocks directly:

    build_fit_inputs_from_hf  ->  pipeline.fit  ->  rolling_window_forecast

Training throughput uses the same two-point trick as the other benches: fit is
run twice with different `num_steps` and the slope gives sec/step, cancelling
model load and LoRA setup.

    python bench_efficiency/bench_chronos.py --protocol fullshot
    python bench_efficiency/bench_chronos.py --protocol zeroshot
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
CHRONOS_DIR = REPO_ROOT / "chronos-forecasting"
sys.path.insert(0, str(BENCH_DIR))

import timing  # noqa: E402

CONTEXT_LENGTH = 144
PREDICTION_LENGTH = 18  # what the Chronos scripts use; 30 min is read off this
GAP_HOURS = 1
MIN_SEQ_LEN = 200


def load_chronos_script(name: str):
    script = CHRONOS_DIR / name
    if not script.exists():
        raise SystemExit(f"Not found: {script}")
    sys.path.insert(0, str(CHRONOS_DIR))
    spec = importlib.util.spec_from_file_location(script.stem, script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[script.stem] = module
    spec.loader.exec_module(module)
    return module


def resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_pipeline(device: str, model_path: Optional[str]):
    import torch
    from chronos import BaseChronosPipeline

    return BaseChronosPipeline.from_pretrained(
        model_path or "amazon/chronos-2",
        device_map=device,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )


def measure_inference(module, pipeline, ds, args, device: str) -> Dict[str, Any]:
    """Time `rolling_window_forecast` per subject; drop the first as warm-up."""
    split = ds[args.test_split] if hasattr(ds, "keys") and args.test_split in ds else ds
    per_subject: List[Dict[str, Any]] = []

    for i, subject in enumerate(split):
        if i >= args.infer_subjects + 1:  # +1 for the dropped warm-up subject
            break
        df = module.prepare_df_from_subject(subject, f"subject_{i}") if hasattr(
            module, "prepare_df_from_subject"
        ) else module.load_and_prepare_data_from_hf(subject)
        sequences = module.split_into_sequences(df, gap_threshold_hours=GAP_HOURS)

        timing.sync()
        t0 = time.perf_counter()
        predictions, *_ = module.rolling_window_forecast(
            sequences,
            pipeline,
            context_length=CONTEXT_LENGTH,
            prediction_length=PREDICTION_LENGTH,
            step_size=args.step_size,
        )
        timing.sync()
        per_subject.append(
            {"seconds": time.perf_counter() - t0, "windows": len(predictions)}
        )

    steady = per_subject[1:] or per_subject  # first call carries warm-up
    seconds = sum(s["seconds"] for s in steady)
    windows = sum(s["windows"] for s in steady)
    return {
        "infer_subjects_timed": len(steady),
        "infer_seconds": seconds,
        "infer_windows": windows,
        "ms_per_window": (1000.0 * seconds / windows) if windows else None,
        "per_subject": steady,
    }


def measure_training(module, ds, args, device: str) -> Dict[str, Any]:
    """Two `fit` runs with different num_steps; the slope is sec/step."""
    print("Building training windows ...")
    t0 = time.perf_counter()
    train_inputs = module.build_fit_inputs_from_hf(
        ds,
        split=args.train_split,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        stride=args.train_stride,
        gap_threshold_hours=GAP_HOURS,
        min_sequence_length=MIN_SEQ_LEN,
    )
    build_seconds = time.perf_counter() - t0
    print(f"  {len(train_inputs)} training sequences in {build_seconds:.1f}s")

    runs = []
    for num_steps in sorted(args.steps):
        pipeline = load_pipeline(device, args.model_path)  # fresh LoRA each time
        timing.reset_peak_memory()
        timing.sync()
        t0 = time.perf_counter()
        pipeline.fit(
            inputs=train_inputs,
            prediction_length=PREDICTION_LENGTH,
            finetune_mode="lora",
            learning_rate=args.lr,
            num_steps=num_steps,
            batch_size=args.train_batch_size,
            logging_steps=max(1, num_steps // 2),
            min_past=CONTEXT_LENGTH,
        )
        timing.sync()
        runs.append(
            {
                "num_steps": num_steps,
                "seconds": time.perf_counter() - t0,
                "peak_memory_mb": timing.peak_memory_mb(),
            }
        )
        print(f"  num_steps={num_steps}: {runs[-1]['seconds']:.1f}s")
        del pipeline

    small, large = runs[0], runs[-1]
    d_steps = large["num_steps"] - small["num_steps"]
    sec_per_step = (large["seconds"] - small["seconds"]) / d_steps if d_steps else None

    out: Dict[str, Any] = {
        "train_window_build_seconds": build_seconds,
        "train_windows_available": len(train_inputs),
        "train_batch_size": args.train_batch_size,
        "fit_runs": runs,
        "sec_per_step": sec_per_step,
        "peak_memory_mb": max((r["peak_memory_mb"] or 0.0) for r in runs) or None,
    }
    if sec_per_step and sec_per_step > 0:
        out["sec_per_1k_train_windows"] = 1000.0 * sec_per_step / args.train_batch_size
    else:
        out["warning"] = "Non-positive slope; re-run with larger --steps."
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", choices=["zeroshot", "fewshot", "fullshot"], default="fullshot")
    parser.add_argument("--dataset", default="byluuu/gluco-tsfm-benchmark")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--train-stride", type=int, default=12, help="12 full-shot / 240 few-shot.")
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--steps", type=int, nargs=2, default=[20, 60], metavar=("SMALL", "LARGE"))
    parser.add_argument("--step-size", type=int, default=1, help="Eval sliding-window stride.")
    parser.add_argument("--infer-subjects", type=int, default=3)
    parser.add_argument("--model-path", default=None, help="Fine-tuned checkpoint; default is amazon/chronos-2.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, default=BENCH_DIR / "results")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    script = "zeroshot_probabilistic.py" if args.protocol == "zeroshot" else "fullshot.py"
    module = load_chronos_script(script)

    from datasets import load_dataset

    ds = load_dataset(args.dataset)

    payload: Dict[str, Any] = {
        "method": "chronos",
        "display_name": "Chronos-2",
        "protocol": args.protocol,
        "script": f"chronos-forecasting/{script}",
        "device": device,
        "environment": timing.environment(),
    }

    if args.protocol != "zeroshot":
        payload.update(measure_training(module, ds, args, device))

    print("\nTiming inference ...")
    pipeline = load_pipeline(device, args.model_path)
    payload.update(measure_inference(module, pipeline, ds, args, device))

    out_path = args.out_dir / f"throughput_chronos_{args.protocol}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str))

    print("\n=== summary ===")
    print(json.dumps({k: v for k, v in payload.items() if k != "per_subject"}, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
