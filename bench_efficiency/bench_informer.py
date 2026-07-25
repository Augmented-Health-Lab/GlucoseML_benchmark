"""Runtime benchmark for Time-LLM and CALF (Informer-style training scripts).

Both already instrument themselves:

    Time-LLM  run_main.py:422          "\tspeed: 0.1234s/iter; left time: ...s"
    CALF      exp_long_term_forecasting_raw.py:117   same format

so training throughput needs no patching at all -- run one capped epoch as a
subprocess and read the number the method itself prints.  This also keeps
Time-LLM's Accelerate/DeepSpeed launch path intact, which an in-process
monkeypatch would disturb.

Inference has no such print, so it is measured with a two-point subprocess run:
the same test command over a small and a larger dataset folder.  The slope
against the window counts from `count_windows.py` removes model-load and
startup cost.

    python bench_efficiency/bench_informer.py --method timellm \
        --train-root hf_cache/train/mixed \
        --test-small hf_cache/test/D1NAMO --test-large hf_cache/test/HUPA-UCM

    python bench_efficiency/bench_informer.py --method calf ...
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
sys.path.insert(0, str(BENCH_DIR))

import timing  # noqa: E402

SPEED_RE = re.compile(r"speed:\s*([0-9.]+)\s*s/iter")
EPOCH_COST_RE = re.compile(r"Epoch:\s*(\d+)\s*cost time:\s*([0-9.]+)")

SEQ_LEN = 144
LABEL_LEN = 72
PRED_LEN = 18


def timellm_train_cmd(args: argparse.Namespace) -> List[str]:
    return [
        sys.executable, "run_main.py",
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--model", "TimeLLM",
        "--model_id", "bench_efficiency",
        "--model_comment", "bench",
        "--llm_model", "GPT2", "--llm_layers", "4", "--llm_dim", "768",
        "--data", "Glucose",
        "--root_path", str(args.train_root),
        "--features", "S", "--target", "glucose", "--freq", "5min",
        "--seq_len", str(SEQ_LEN), "--label_len", str(LABEL_LEN), "--pred_len", str(PRED_LEN),
        "--enc_in", "1", "--dec_in", "1", "--c_out", "1",
        "--batch_size", str(args.train_batch_size),
        "--train_epochs", "1",
        "--learning_rate", "5e-4",
        "--num_workers", "2",
        "--stride", str(args.train_stride),
        "--patch_stride", "3",
        "--max_windows_per_epoch", str(args.max_windows_per_epoch),
        "--des", "bench",
    ]


def timellm_test_cmd(args: argparse.Namespace, test_root: Path) -> List[str]:
    return [
        sys.executable, "run_main.py",
        "--task_name", "long_term_forecast",
        "--is_training", "0",
        "--model", "TimeLLM",
        "--model_id", "bench_efficiency",
        "--model_comment", "bench",
        "--llm_model", "GPT2", "--llm_layers", "4", "--llm_dim", "768",
        "--data", "Glucose",
        "--test_root_path", str(test_root),
        "--features", "S", "--target", "glucose", "--freq", "5min",
        "--seq_len", str(SEQ_LEN), "--label_len", str(LABEL_LEN), "--pred_len", str(PRED_LEN),
        "--enc_in", "1", "--dec_in", "1", "--c_out", "1",
        "--batch_size", str(args.eval_batch_size),
        "--num_workers", "2",
        "--stride", "1",
        "--patch_stride", "3",
        "--des", "bench",
    ]


def calf_train_cmd(args: argparse.Namespace) -> List[str]:
    return [
        sys.executable, "run.py",
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--model", "CALF",
        "--model_id", "bench_efficiency",
        "--data", "Glucose",
        "--root_path", str(args.train_root),
        "--features", "S", "--target", "BGvalue", "--freq", "5min",
        "--d_model", "768", "--n_heads", "12",
        "--seq_len", str(SEQ_LEN), "--label_len", str(LABEL_LEN), "--pred_len", str(PRED_LEN),
        "--enc_in", "1", "--dec_in", "1", "--c_out", "1",
        "--stride", str(args.train_stride),
        "--batch_size", str(args.train_batch_size),
        "--gpt_layers", "4",
        "--train_epochs", "1",
        "--learning_rate", "1e-4",
        "--scale_value", "1.0",
        "--max_windows_per_epoch", str(args.max_windows_per_epoch),
        "--num_workers", "2",
        "--use_gpu", "1",
        "--per_subject_eval", "0",
    ]


def calf_test_cmd(args: argparse.Namespace, test_root: Path) -> List[str]:
    return [
        sys.executable, "run.py",
        "--task_name", "long_term_forecast",
        "--is_training", "0",
        "--model", "CALF",
        "--model_id", "bench_efficiency",
        "--data", "Glucose",
        "--root_path", "/unused/during/test",
        "--test_root_path", str(test_root),
        "--features", "S",
        "--checkpoints", "./checkpoints",
        "--seq_len", str(SEQ_LEN), "--label_len", str(LABEL_LEN), "--pred_len", str(PRED_LEN),
        "--d_model", "768", "--n_heads", "12",
        "--stride", "1",
        "--gpt_layers", "4",
        "--batch_size", str(args.eval_batch_size),
        "--scale_value", "1.0",
    ]


METHODS = {
    "timellm": {
        "display_name": "Time-LLM",
        "cwd": REPO_ROOT / "Time-LLM",
        "train_cmd": timellm_train_cmd,
        "test_cmd": timellm_test_cmd,
        "train_batch_size": 16,
    },
    "calf": {
        "display_name": "CALF",
        "cwd": REPO_ROOT / "CALF",
        "train_cmd": calf_train_cmd,
        "test_cmd": calf_test_cmd,
        "train_batch_size": 8,
    },
}


def run(cmd: List[str], cwd: Path, log_path: Path) -> Dict[str, Any]:
    print(f"\n$ cd {cwd} && {' '.join(cmd)}\n")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout + "\n===== STDERR =====\n" + proc.stderr)
    tail = "\n".join((proc.stdout or proc.stderr).splitlines()[-15:])
    print(tail)
    if proc.returncode != 0:
        print(f"! exit code {proc.returncode}; full log at {log_path}", file=sys.stderr)

    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "wall_seconds": elapsed,
        "stdout": proc.stdout,
        "log": str(log_path),
    }


def parse_training(stdout: str) -> Dict[str, Any]:
    """Read the method's own `speed: X s/iter` and `Epoch cost time` lines."""
    speeds = [float(m) for m in SPEED_RE.findall(stdout)]
    epochs = [(int(a), float(b)) for a, b in EPOCH_COST_RE.findall(stdout)]
    out: Dict[str, Any] = {
        "speed_samples": speeds,
        "epoch_cost_seconds": [c for _, c in epochs],
    }
    if speeds:
        # The first sample includes warm-up; prefer the median of the rest.
        steady = speeds[1:] or speeds
        out["sec_per_step"] = statistics.median(steady)
        out["sec_per_step_samples_used"] = len(steady)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--train-root", type=Path, default=REPO_ROOT / "hf_cache" / "train" / "mixed")
    parser.add_argument("--test-small", type=Path, default=REPO_ROOT / "hf_cache" / "test" / "D1NAMO")
    parser.add_argument("--test-large", type=Path, default=REPO_ROOT / "hf_cache" / "test" / "HUPA-UCM")
    parser.add_argument(
        "--test-windows",
        type=int,
        nargs=2,
        default=None,
        metavar=("SMALL", "LARGE"),
        help="Stride-1 window counts for the two folders, from workload.json. "
             "Without them only wall-clock is reported, not ms/window.",
    )
    parser.add_argument("--train-stride", type=int, default=12)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-windows-per-epoch", type=int, default=2000, help="Cap so one epoch is quick.")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=BENCH_DIR / "results")
    args = parser.parse_args(argv)

    spec = METHODS[args.method]
    if args.train_batch_size is None:
        args.train_batch_size = spec["train_batch_size"]
    cwd = spec["cwd"]
    if not cwd.exists():
        parser.error(f"Method directory not found: {cwd}")

    log_dir = args.out_dir / "logs"
    payload: Dict[str, Any] = {
        "method": args.method,
        "display_name": spec["display_name"],
        "train_batch_size": args.train_batch_size,
        "train_stride": args.train_stride,
        "environment": timing.environment(),
    }

    # --- training: one capped epoch, read the method's own s/iter -----------
    train_run = run(spec["train_cmd"](args), cwd, log_dir / f"{args.method}_train.log")
    payload["train_run"] = {k: v for k, v in train_run.items() if k != "stdout"}
    payload.update(parse_training(train_run["stdout"]))
    if "sec_per_step" in payload:
        payload["sec_per_1k_train_windows"] = (
            1000.0 * payload["sec_per_step"] / args.train_batch_size
        )
    else:
        payload["warning"] = (
            "No 'speed: X s/iter' line found -- the run probably failed, or "
            "finished in under --log-every iterations. Check the log and raise "
            "--max-windows-per-epoch."
        )

    # --- inference: two folders, slope removes startup ----------------------
    if not args.skip_inference:
        small = run(spec["test_cmd"](args, args.test_small), cwd, log_dir / f"{args.method}_test_small.log")
        large = run(spec["test_cmd"](args, args.test_large), cwd, log_dir / f"{args.method}_test_large.log")
        payload["test_runs"] = {
            "small": {k: v for k, v in small.items() if k != "stdout"},
            "large": {k: v for k, v in large.items() if k != "stdout"},
        }
        if args.test_windows:
            w_small, w_large = args.test_windows
            d_w = w_large - w_small
            if d_w > 0:
                sec_per_window = (large["wall_seconds"] - small["wall_seconds"]) / d_w
                payload["ms_per_window"] = 1000.0 * sec_per_window
                payload["inference_startup_seconds"] = (
                    small["wall_seconds"] - sec_per_window * w_small
                )
                if sec_per_window <= 0:
                    payload["warning_inference"] = (
                        "Non-positive slope; pick two test folders with a larger "
                        "window-count difference."
                    )
            else:
                payload["warning_inference"] = "--test-windows LARGE must exceed SMALL."
        else:
            payload["note_inference"] = (
                "Pass --test-windows from workload.json to convert wall-clock into ms/window."
            )

    out_path = args.out_dir / f"throughput_{args.method}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str))

    print("\n=== summary ===")
    print(json.dumps({k: v for k, v in payload.items() if k != "speed_samples"}, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
