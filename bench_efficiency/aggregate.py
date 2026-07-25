"""Combine measured throughput with counted workload into the efficiency table.

    runtime = throughput x workload

Throughput comes from the `bench_*.py` runs (`results/throughput_*.json`);
workload comes from `count_windows.py` (`results/workload.json`).  Because
throughput does not depend on the protocol, each method is benched once and
both few-shot and full-shot numbers are derived here by swapping the workload.

Two columns are produced:

  normalized      every method at epochs=10, batch=16, stride 12 / 240.
                  Comparable across methods; this is the fair-comparison column.
  as_configured   each method's own settings from `methods.py:AS_CONFIGURED`.
                  What the paper's runs actually cost; not comparable across rows.

    python bench_efficiency/aggregate.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

from methods import AS_CONFIGURED, NORMALIZED  # noqa: E402

METHOD_ORDER = ["lstm", "gpformer", "chronos", "moirai", "timesfm", "timer", "timellm", "calf"]
DISPLAY = {
    "lstm": "LSTM (Martinsson)",
    "gpformer": "GPFormer",
    "chronos": "Chronos-2",
    "moirai": "Moirai",
    "timesfm": "TimesFM",
    "timer": "Timer",
    "timellm": "Time-LLM",
    "calf": "CALF",
}


def fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def collect_throughput(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Merge every throughput_*.json into one record per method."""
    merged: Dict[str, Dict[str, Any]] = {}
    for path in sorted(results_dir.glob("throughput_*.json")):
        payload = load_json(path)
        if payload is None:
            continue
        # bench_multiwindow nests its numbers under "summary".
        record = payload.get("summary", payload)
        method = record.get("method") or path.stem.replace("throughput_", "").split("_")[0]
        slot = merged.setdefault(method, {"sources": []})
        slot["sources"].append(path.name)
        for key, value in record.items():
            if value is None:
                continue
            # Zero-shot inference lives in its own file; keep it separate.
            if record.get("protocol") == "zeroshot" and key in {"ms_per_window", "infer_windows", "infer_seconds"}:
                slot[f"zeroshot_{key}"] = value
            elif key not in slot or slot.get(key) in (None, "-"):
                slot[key] = value
    return merged


def workload_windows(workload: Dict[str, Any], convention: str, split: str, stride: int) -> Optional[int]:
    try:
        block = workload["conventions"][convention][split]["windows_by_stride"]
    except KeyError:
        return None
    return block.get(str(stride))


def estimate(
    method: str,
    throughput: Dict[str, Any],
    workload: Dict[str, Any],
    *,
    protocol: str,
    column: str,
) -> Dict[str, Any]:
    cfg = AS_CONFIGURED.get(method, {})
    convention = cfg.get("window_convention", "multiwindow")

    if column == "normalized":
        epochs = NORMALIZED["train_epochs"]
        batch = NORMALIZED["train_batch_size"]
        stride = NORMALIZED["train_stride_fullshot" if protocol == "fullshot" else "train_stride_fewshot"]
        train_steps_budget = None
        window_cap = None
    else:
        epochs = cfg.get("train_epochs")
        batch = cfg.get("train_batch_size")
        stride = cfg.get("train_stride_fullshot" if protocol == "fullshot" else "train_stride_fewshot")
        train_steps_budget = cfg.get("train_steps")
        window_cap = cfg.get("max_windows_per_epoch")
        if method == "lstm" and epochs is None:
            epochs = throughput.get("measured_epochs")  # filled in from bench_lstm fewshot-real

    row: Dict[str, Any] = {
        "epochs": epochs,
        "batch": batch,
        "train_stride": stride,
        "train_seconds": None,
        "infer_seconds": None,
    }

    # ---- training -------------------------------------------------------
    train_windows = workload_windows(workload, convention, "train", stride) if stride else None
    row["train_windows"] = train_windows
    sec_per_step = throughput.get("sec_per_step")
    sec_per_1k = throughput.get("sec_per_1k_train_windows")

    if train_steps_budget and sec_per_step:
        # Chronos: fixed step budget, epochs undefined.
        row["train_seconds"] = sec_per_step * train_steps_budget
        row["train_steps"] = train_steps_budget
    elif train_windows and sec_per_1k and epochs:
        per_epoch_windows = min(train_windows, window_cap) if window_cap else train_windows
        row["windows_per_epoch"] = per_epoch_windows
        row["train_seconds"] = (per_epoch_windows / 1000.0) * sec_per_1k * epochs
        if batch:
            row["train_steps"] = int(per_epoch_windows / batch) * epochs

    # ---- inference ------------------------------------------------------
    eval_windows = workload_windows(workload, convention, "test", NORMALIZED["eval_stride"])
    row["infer_windows"] = eval_windows
    ms_per_window = (
        throughput.get("zeroshot_ms_per_window")
        if protocol == "zeroshot"
        else throughput.get("ms_per_window")
    )
    row["ms_per_window"] = ms_per_window
    if eval_windows and ms_per_window:
        row["infer_seconds"] = eval_windows * ms_per_window / 1000.0

    return row


def build_table(results_dir: Path, workload: Dict[str, Any]) -> List[Dict[str, Any]]:
    throughputs = collect_throughput(results_dir)

    # bench_lstm's fewshot-real run supplies the measured early-stopping epoch count.
    lstm_real = load_json(results_dir / "throughput_lstm_fewshot-real.json")
    if lstm_real and "lstm" in throughputs:
        throughputs["lstm"]["measured_epochs"] = lstm_real.get("epochs_run")
        throughputs["lstm"]["measured_best_epoch"] = lstm_real.get("best_epoch")

    rows: List[Dict[str, Any]] = []
    for method in METHOD_ORDER:
        tp = throughputs.get(method)
        if tp is None:
            rows.append({"method": method, "display": DISPLAY[method], "missing": True})
            continue
        entry: Dict[str, Any] = {
            "method": method,
            "display": DISPLAY[method],
            "sources": tp.get("sources", []),
            "params_total": tp.get("params_total"),
            "params_trainable": tp.get("params_trainable"),
            "peak_memory_mb": tp.get("peak_memory_mb"),
            "sec_per_step": tp.get("sec_per_step"),
            "sec_per_1k_train_windows": tp.get("sec_per_1k_train_windows"),
            "ms_per_window": tp.get("ms_per_window"),
            "zeroshot_ms_per_window": tp.get("zeroshot_ms_per_window"),
            "note": AS_CONFIGURED.get(method, {}).get("note"),
        }
        for column in ("normalized", "as_configured"):
            for protocol in ("fewshot", "fullshot"):
                entry[f"{column}_{protocol}"] = estimate(
                    method, tp, workload, protocol=protocol, column=column
                )
        if tp.get("zeroshot_ms_per_window"):
            entry["zeroshot"] = estimate(method, tp, workload, protocol="zeroshot", column="normalized")
        rows.append(entry)
    return rows


def render_markdown(rows: List[Dict[str, Any]], column: str) -> str:
    header = (
        f"### {column} column\n\n"
        "| Method | Params (train/total) | s/1k train win | ms/win | "
        "Few-shot train | Few-shot infer | Full-shot train | Full-shot infer | Peak GPU |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for row in rows:
        if row.get("missing"):
            lines.append(f"| {row['display']} | _not benched yet_ | | | | | | | |")
            continue
        few = row[f"{column}_fewshot"]
        full = row[f"{column}_fullshot"]
        params = (
            f"{row['params_trainable']/1e6:.1f}M / {row['params_total']/1e6:.1f}M"
            if row.get("params_total")
            else "-"
        )
        lines.append(
            "| {display} | {params} | {s1k} | {msw} | {ft} | {fi} | {Ft} | {Fi} | {mem} |".format(
                display=row["display"],
                params=params,
                s1k=f"{row['sec_per_1k_train_windows']:.2f}" if row.get("sec_per_1k_train_windows") else "-",
                msw=f"{row['ms_per_window']:.2f}" if row.get("ms_per_window") else "-",
                ft=fmt_duration(few.get("train_seconds")),
                fi=fmt_duration(few.get("infer_seconds")),
                Ft=fmt_duration(full.get("train_seconds")),
                Fi=fmt_duration(full.get("infer_seconds")),
                mem=f"{row['peak_memory_mb']/1024:.1f}GB" if row.get("peak_memory_mb") else "-",
            )
        )
    return header + "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=BENCH_DIR / "results")
    parser.add_argument("--workload", type=Path, default=BENCH_DIR / "results" / "workload.json")
    parser.add_argument("--out-prefix", type=Path, default=BENCH_DIR / "results" / "efficiency_table")
    args = parser.parse_args(argv)

    workload = load_json(args.workload)
    if workload is None:
        parser.error(f"{args.workload} not found. Run count_windows.py first.")

    rows = build_table(args.results_dir, workload)

    json_path = args.out_prefix.with_suffix(".json")
    json_path.write_text(json.dumps({"workload": workload, "rows": rows}, indent=2, default=str))

    md = (
        "# Runtime efficiency\n\n"
        f"Context 144 steps (12 h), horizon {workload.get('horizon_steps')} steps, eval stride 1.\n\n"
        + render_markdown(rows, "normalized")
        + "\n"
        + render_markdown(rows, "as_configured")
    )
    md_path = args.out_prefix.with_suffix(".md")
    md_path.write_text(md)

    print(md)
    print(f"Wrote {md_path}\nWrote {json_path}")

    missing = [r["display"] for r in rows if r.get("missing")]
    if missing:
        print(f"\nStill missing throughput for: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
