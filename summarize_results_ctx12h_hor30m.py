import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    results_roots: Dict[str, Path]  # shot -> results root


def _iter_datasets_from_test_root(test_root: Path) -> Iterable[str]:
    if not test_root.exists():
        return []
    return sorted([p.name for p in test_root.iterdir() if p.is_dir()])


def _discover_metrics_files(results_root: Path, suffix: str = "_test_metrics.csv") -> Dict[str, Path]:
    if not results_root.exists():
        return {}
    out: Dict[str, Path] = {}
    for p in results_root.glob(f"*{suffix}"):
        dataset = p.name[: -len(suffix)]
        out[dataset] = p
    return out


def _display_dataset_name(raw_name: str) -> str:
    # Canonicalize dataset names across folders/files so we can reorder consistently.
    name = str(raw_name)
    name = pd.Series([name]).str.replace(r"^\d+_", "", regex=True).iloc[0]
    name = pd.Series([name]).str.replace(r"\s+Open$", "", regex=True).iloc[0]

    # Normalize common separators for a stable display label.
    name_norm = name.replace("_", " ").strip()

    # Hand-tuned aliases to match the user's desired naming.
    aliases = {
        "BIG IDEA LAB": "BIG IDEAS Lab",
        "D1NAMO": "DINAMO",
    }
    return aliases.get(name_norm, name_norm)


def _participant_mean_metrics(
    metrics_csv: Path,
    *,
    context_hours: int,
    horizon_minutes: int,
    eval_stride_steps: int = 1,
    split: str = "test",
    metric_mode: str = "final",
) -> Dict[str, Optional[float]]:
    df = pd.read_csv(metrics_csv)

    # Not all models have the exact same schema, so filter only when columns exist.
    if "split" in df.columns:
        df = df[df["split"] == split]
    if "task" in df.columns:
        # Uni2TS zeroshot writes task=eval; other folders may not have this column.
        df = df[df["task"] == "eval"]
    if "metric_mode" in df.columns and metric_mode is not None:
        df = df[df["metric_mode"] == metric_mode]
    if "context_hours" in df.columns:
        df = df[df["context_hours"] == context_hours]
    if "horizon_minutes" in df.columns:
        df = df[df["horizon_minutes"] == horizon_minutes]
    if "stride_steps" in df.columns and eval_stride_steps is not None:
        df = df[df["stride_steps"] == eval_stride_steps]

    # Coerce numeric columns (robust to older outputs).
    for col in ["rmse", "mae"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    pid_col = "participant_id" if "participant_id" in df.columns else None
    if pid_col is None:
        return {"rmse": None, "mae": None, "n_participants": 0}

    per_participant = df.groupby(pid_col, dropna=True)[["rmse", "mae"]].mean(numeric_only=True)
    per_participant = per_participant.dropna(how="all")

    n = int(per_participant.shape[0])
    rmse = float(per_participant["rmse"].mean()) if n and "rmse" in per_participant.columns else None
    mae = float(per_participant["mae"].mean()) if n and "mae" in per_participant.columns else None
    return {"rmse": rmse, "mae": mae, "n_participants": n}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize context=12h, horizon=30min multi-window results across models. "
            "For each dataset, compute the arithmetic mean over participants "
            "(i.e., unweighted across participants)."
        )
    )
    parser.add_argument("--context-hours", type=int, default=12)
    parser.add_argument("--horizon-minutes", type=int, default=30)
    parser.add_argument("--eval-stride-steps", type=int, default=1)
    parser.add_argument("--test-root", type=Path, default=Path("test_dataset"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_tables_ctx12h_hor30m"),
        help="Directory to write the 3 shot tables as CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    desired_dataset_order = [
        "BIG IDEAS Lab",
        "ShanghaiT1DM",
        "ShanghaiT2DM",
        "CGMacros",
        "UCHTT1DM",
        "HUPA-UCM",
        "T1DM-UOM",
        "Bris-T1D",
        "AZT1D",
        "Hall2018",
        "DINAMO",
        "Colas2019",
    ]

    models = [
        ModelSpec(
            key="moirai",
            display_name="Moirai (Uni2TS)",
            results_roots={
                "zeroshot": Path("uni2ts/multi_horizon_results_uni2ts_zeroshot"),
                "fullshot": Path("uni2ts/multi_horizon_results_uni2ts_fullshot"),
                "fewshot": Path("uni2ts/multi_horizon_results_uni2ts_fewshot"),
            },
        ),
        ModelSpec(
            key="timer",
            display_name="Timer",
            results_roots={
                "zeroshot": Path("timer-model/multi_horizon_results_timer_zeroshot"),
                "fullshot": Path("timer-model/multi_horizon_results_timer_fullshot"),
                "fewshot": Path("timer-model/multi_horizon_results_timer_fewshot"),
            },
        ),
        ModelSpec(
            key="timesfm",
            display_name="TimesFM",
            results_roots={
                "zeroshot": Path("timesfm/multi_horizon_results_timesfm_zeroshot"),
                "fullshot": Path("timesfm/multi_horizon_results_timesfm_fullshot"),
                "fewshot": Path("timesfm/multi_horizon_results_timesfm_fewshot"),
            },
        ),
    ]

    # Use test_dataset folder names as the canonical dataset list; add any datasets
    # that appear in metrics files but are missing from test_dataset.
    datasets = list(_iter_datasets_from_test_root(args.test_root))
    datasets_set = set(datasets)
    for m in models:
        for shot, root in m.results_roots.items():
            for ds in _discover_metrics_files(root).keys():
                if ds not in datasets_set:
                    datasets.append(ds)
                    datasets_set.add(ds)
    datasets = sorted(datasets)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for shot in ["zeroshot", "fewshot", "fullshot"]:
        # Pre-index metrics files for speed.
        metrics_paths = {m.key: _discover_metrics_files(m.results_roots[shot]) for m in models}

        rows = []
        for dataset in datasets:
            row = {"dataset": _display_dataset_name(dataset)}
            for m in models:
                path = metrics_paths[m.key].get(dataset)
                if path is None:
                    row[f"{m.key}_rmse"] = None
                    row[f"{m.key}_mae"] = None
                    row[f"{m.key}_n"] = 0
                    continue
                metrics = _participant_mean_metrics(
                    path,
                    context_hours=args.context_hours,
                    horizon_minutes=args.horizon_minutes,
                    eval_stride_steps=args.eval_stride_steps,
                )
                row[f"{m.key}_rmse"] = metrics["rmse"]
                row[f"{m.key}_mae"] = metrics["mae"]
                row[f"{m.key}_n"] = metrics["n_participants"]
            rows.append(row)

        table = pd.DataFrame(rows).set_index("dataset")

        # Enforce the requested dataset order, then append any leftovers.
        remaining = [d for d in table.index.tolist() if d not in desired_dataset_order]
        table = table.reindex(desired_dataset_order + sorted(remaining))

        out_csv = args.output_dir / f"{shot}_ctx{args.context_hours}h_hor{args.horizon_minutes}m.csv"
        table.to_csv(out_csv, index=True)

        # Print a compact view to stdout for quick inspection.
        print(f"\n[{shot}] -> {out_csv.as_posix()}")
        with pd.option_context("display.max_rows", 200, "display.max_columns", 200, "display.width", 200):
            print(table)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
