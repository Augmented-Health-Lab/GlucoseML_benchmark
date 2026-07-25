"""Measure train/inference throughput for GPFormer, TimesFM, Timer and Moirai.

These four share the `predict_glucose_multiwindow_*` harness, so one runner
covers all of them.  We import the method's real script, monkeypatch its
`train_*` / `evaluate_subject*` functions with timing wrappers, and call its
own `main()`.  No method code is modified.

Training throughput
    `main()` is invoked twice with `--train-epochs 1` and two different
    `--max-train-steps` caps.  sec/step = (T_big - T_small) / (steps_big -
    steps_small), which cancels model loading, dataset construction and CUDA
    warm-up because both runs pay them identically.

Inference throughput
    `evaluate_subject*` is timed per subject and returns the window count at
    tuple index 3.  The first subject is dropped as warm-up.

Because throughput does not depend on the protocol, running this once with the
full-shot script is enough -- few-shot is derived in `aggregate.py` by swapping
in the few-shot workload.

    python bench_efficiency/bench_multiwindow.py --method timesfm
    python bench_efficiency/bench_multiwindow.py --method timer --zeroshot
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

import timing  # noqa: E402
from methods import MULTIWINDOW, MULTIWINDOW_ZEROSHOT, MultiWindowMethod  # noqa: E402


def load_script(script: Path):
    """Import a method script by path, with its own directory on sys.path.

    The `_with_raw*` wrappers do `import predict_glucose_multiwindow_X_fullshot`
    as a sibling, which only resolves if the script's directory is importable.
    """
    sys.path.insert(0, str(script.parent))
    spec = importlib.util.spec_from_file_location(script.stem, script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[script.stem] = module
    spec.loader.exec_module(module)
    return module


def patch_targets(module, cfg: MultiWindowMethod) -> List[Any]:
    """Modules that may hold the functions we want to time."""
    targets = [module]
    if cfg.base_module and cfg.base_module in sys.modules:
        targets.append(sys.modules[cfg.base_module])
    return targets


def apply_patches(targets, timer: timing.PhaseTimer, cfg: MultiWindowMethod) -> Dict[str, List[str]]:
    """Wrap every train/eval function we can find; report what was hit."""
    hits: Dict[str, List[str]] = {"train": [], "eval": []}
    for mod in targets:
        for fn_name in cfg.train_fns:
            fn = getattr(mod, fn_name, None)
            if callable(fn) and not getattr(fn, "__bench_wrapped__", False):
                setattr(mod, fn_name, timer.wrap("train", fn))
                hits["train"].append(f"{mod.__name__}.{fn_name}")
        for fn_name in cfg.eval_fns:
            fn = getattr(mod, fn_name, None)
            if callable(fn) and not getattr(fn, "__bench_wrapped__", False):
                setattr(mod, fn_name, timer.wrap("infer", fn, windows_at=3))
                hits["eval"].append(f"{mod.__name__}.{fn_name}")
    return hits


def build_argv(args: argparse.Namespace, cfg: MultiWindowMethod, max_steps: Optional[int]) -> List[str]:
    argv = [
        "--data-root-train", str(args.data_root / "train" / "mixed"),
        "--data-root-test", str(args.data_root / "test"),
        "--datasets", args.eval_dataset or cfg.smallest_dataset,
        "--context-hours", "12",
        "--horizons-minutes", "30",
        "--eval-stride-steps", str(args.eval_stride),
        "--device", args.device,
        "--overwrite",
    ]
    if max_steps is not None:
        argv += [
            "--train-epochs", "1",
            "--max-train-steps", str(max_steps),
            "--train-stride-steps", str(args.train_stride),
            "--train-batch-size", str(args.train_batch_size),
        ]
    argv += cfg.extra_args
    return argv


def run_once(cfg: MultiWindowMethod, args: argparse.Namespace, max_steps: Optional[int]) -> Dict[str, Any]:
    timer = timing.PhaseTimer()
    timing.reset_peak_memory()

    module = load_script(cfg.script)
    hits = apply_patches(patch_targets(module, cfg), timer, cfg)
    if max_steps is not None and not hits["train"]:
        raise RuntimeError(
            f"No training function found in {cfg.script.name}; "
            f"tried {cfg.train_fns}. Update methods.py."
        )
    if not hits["eval"]:
        raise RuntimeError(
            f"No evaluation function found in {cfg.script.name}; tried {cfg.eval_fns}."
        )

    argv = build_argv(args, cfg, max_steps)
    print(f"\n>>> {cfg.name} max_train_steps={max_steps}\n    patched: {hits}\n    argv: {' '.join(argv)}\n")

    with timing.Stopwatch(timer, "total"):
        module.main(argv)

    out = timer.to_dict()
    out["patched"] = hits
    out["max_train_steps"] = max_steps
    out["peak_memory_mb"] = timing.peak_memory_mb()
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True, choices=sorted(MULTIWINDOW))
    parser.add_argument("--zeroshot", action="store_true", help="Inference only; skip training timing.")
    parser.add_argument("--data-root", type=Path, default=Path("hf_cache"))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--steps",
        type=int,
        nargs=2,
        default=[20, 60],
        metavar=("SMALL", "LARGE"),
        help="Two --max-train-steps caps; the slope between them is sec/step.",
    )
    parser.add_argument("--train-stride", type=int, default=12, help="Normalized full-shot stride.")
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-stride", type=int, default=1)
    parser.add_argument("--eval-dataset", default=None, help="Single dataset folder to evaluate on.")
    parser.add_argument("--out-dir", type=Path, default=BENCH_DIR / "results")
    args = parser.parse_args(argv)

    registry = MULTIWINDOW_ZEROSHOT if args.zeroshot else MULTIWINDOW
    if args.method not in registry:
        parser.error(f"{args.method} has no {'zero-shot' if args.zeroshot else 'trainable'} script registered.")
    cfg = registry[args.method]
    if not cfg.script.exists():
        parser.error(f"Script not found: {cfg.script}")

    tag = f"{args.method}_{'zeroshot' if args.zeroshot else 'train'}"
    runs: List[Dict[str, Any]] = []

    if args.zeroshot:
        runs.append(run_once(cfg, args, None))
    else:
        # Two caps -> slope. Separate processes would be cleaner, but a fresh
        # module import per run is enough because the model is rebuilt inside.
        for cap in args.steps:
            runs.append(run_once(cfg, args, cap))

    summary = summarize(args, cfg, runs)
    out_path = args.out_dir / f"throughput_{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"runs": runs, "summary": summary}, indent=2, default=str))

    print("\n=== summary ===")
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


def summarize(args: argparse.Namespace, cfg: MultiWindowMethod, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "method": args.method,
        "display_name": cfg.name,
        "script": str(cfg.script.relative_to(BENCH_DIR.parent)),
        "protocol": "zeroshot" if args.zeroshot else "train",
        "train_batch_size": args.train_batch_size,
        "environment": runs[-1]["environment"],
        "peak_memory_mb": max((r.get("peak_memory_mb") or 0.0) for r in runs) or None,
    }

    # --- inference: drop the first subject as warm-up -----------------------
    infer_calls = [c for r in runs for c in r["phases"].get("infer", {}).get("calls", [])]
    if len(infer_calls) > 1:
        infer_calls = infer_calls[1:]
    seconds = sum(c["seconds"] for c in infer_calls)
    windows = sum(int(c.get("windows") or 0) for c in infer_calls)
    summary["infer_subjects_timed"] = len(infer_calls)
    summary["infer_windows"] = windows
    summary["infer_seconds"] = seconds
    summary["ms_per_window"] = (1000.0 * seconds / windows) if windows else None

    # --- training: slope between the two step caps -------------------------
    if not args.zeroshot and len(runs) == 2:
        (small, large) = sorted(runs, key=lambda r: r["max_train_steps"])
        t_small = small["phases"]["train"]["total_seconds"]
        t_large = large["phases"]["train"]["total_seconds"]
        d_steps = large["max_train_steps"] - small["max_train_steps"]
        sec_per_step = (t_large - t_small) / d_steps if d_steps else None
        summary.update(
            {
                "train_steps_small": small["max_train_steps"],
                "train_steps_large": large["max_train_steps"],
                "train_seconds_small": t_small,
                "train_seconds_large": t_large,
                "sec_per_step": sec_per_step,
                "train_setup_seconds": (t_small - sec_per_step * small["max_train_steps"])
                if sec_per_step is not None
                else None,
            }
        )
        if sec_per_step is not None and sec_per_step > 0:
            summary["sec_per_1k_train_windows"] = 1000.0 * sec_per_step / args.train_batch_size
        else:
            summary["warning"] = (
                "Non-positive slope: the two runs were too short to separate. "
                "Re-run with larger --steps."
            )
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
