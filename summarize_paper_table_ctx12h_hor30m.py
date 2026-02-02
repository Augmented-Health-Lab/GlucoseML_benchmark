import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class ModelSpec:
    # Short key used as output column prefix.
    key: str
    # Column header displayed in the exported table.
    display_name: str
    # shot -> results root folder containing `*_test_metrics.csv`
    results_roots: Dict[str, Path]


def _iter_datasets_from_test_root(test_root: Path) -> Tuple[List[str], List[str]]:
    """Return (open_datasets, controlled_datasets) as raw folder names."""
    open_datasets: List[str] = []
    controlled_datasets: List[str] = []

    if not test_root.exists():
        return open_datasets, controlled_datasets

    controlled_container = test_root / "controlled_dataset"

    for p in test_root.iterdir():
        if not p.is_dir():
            continue
        if p.name == "controlled_dataset":
            if controlled_container.exists():
                controlled_datasets.extend([c.name for c in controlled_container.iterdir() if c.is_dir()])
            continue
        open_datasets.append(p.name)

    return sorted(open_datasets), sorted(controlled_datasets)


def _discover_metrics_files(results_root: Path, suffix: str = "_test_metrics.csv") -> Dict[str, Path]:
    if not results_root.exists():
        return {}
    out: Dict[str, Path] = {}
    for p in results_root.glob(f"*{suffix}"):
        dataset = p.name[: -len(suffix)]
        # controlled_dataset is a container folder; we summarize its children instead.
        if dataset == "controlled_dataset":
            continue
        out[dataset] = p
    return out


def _display_dataset_name(raw_name: str) -> str:
    # Canonicalize dataset names across folders/files so we can reorder consistently.
    name = str(raw_name)
    name = pd.Series([name]).str.replace(r"^\d+_", "", regex=True).iloc[0]
    name = pd.Series([name]).str.replace(r"\s+Open$", "", regex=True).iloc[0]

    # Normalize common separators for a stable display label.
    name_norm = name.replace("_", " ").strip()

    # Hand-tuned aliases to match the paper's naming.
    aliases = {
        "BIG IDEA LAB": "BIG IDEAS Lab",
        "D1NAMO": "DINAMO",
    }
    return aliases.get(name_norm, name_norm)


def _load_per_participant_metrics(
    metrics_csv: Path,
    *,
    context_hours: int,
    horizon_minutes: int,
    eval_stride_steps: int = 1,
    split: str = "test",
    metric_mode: str = "final",
) -> pd.DataFrame:
    """
    Load a metrics CSV and return one row per participant.

    Some result folders produce multiple rows per participant (e.g., multiple windows).
    We first average within each participant, then use those participant means for
    dataset and combined aggregations.
    """
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

    if "participant_id" not in df.columns:
        return pd.DataFrame(columns=["rmse", "mae"])

    metric_cols = [c for c in ["rmse", "mae"] if c in df.columns]
    if not metric_cols:
        return pd.DataFrame(columns=["rmse", "mae"])

    # Coerce numeric columns (robust to older outputs).
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only metrics we know how to aggregate.
    keep_cols = ["participant_id"] + metric_cols
    df = df[keep_cols].dropna(subset=["participant_id"])
    if df.empty:
        return pd.DataFrame(columns=["rmse", "mae"])

    per_participant = df.groupby("participant_id", dropna=True)[metric_cols].mean(numeric_only=True)
    per_participant = per_participant.dropna(how="all")
    # Ensure both columns exist for downstream `.get(...)` access.
    for col in ["rmse", "mae"]:
        if col not in per_participant.columns:
            per_participant[col] = pd.NA
    return per_participant[["rmse", "mae"]]


def _summary_mean_std(values: pd.Series) -> Tuple[Optional[float], Optional[float], int]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    n = int(values.shape[0])
    if n == 0:
        return None, None, 0
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n >= 2 else None
    return mean, std, n


def _format_mean_pm(mean: Optional[float], std: Optional[float], *, digits: int = 2) -> Optional[str]:
    if mean is None:
        return None
    fmt = f"{{:.{digits}f}}"
    if std is None:
        return fmt.format(mean)
    return f"{fmt.format(mean)} ({fmt.format(std)})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute paper-style RMSE table for context=12h, horizon=30m across datasets. "
            "Per-dataset numbers are unweighted arithmetic mean (STD) over participants. "
            "Combined Open/All are computed by pooling all participants across the relevant datasets."
        )
    )
    parser.add_argument("--shot", default="zeroshot", help="One of: zeroshot, fewshot, fullshot.")
    parser.add_argument("--metric", default="rmse", choices=["rmse", "mae"])
    parser.add_argument("--digits", type=int, default=2, help="Digits after decimal in the table cells.")
    parser.add_argument("--context-hours", type=int, default=12)
    parser.add_argument("--horizon-minutes", type=int, default=30)
    parser.add_argument("--eval-stride-steps", type=int, default=1)
    parser.add_argument("--test-root", type=Path, default=Path("test_dataset"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_tables_ctx12h_hor30m"),
        help="Directory to write the table as CSV.",
    )
    parser.add_argument(
        "--include-section-rows",
        action="store_true",
        help="Insert OpenAccess/ControlledAccess label rows like the paper table.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Display-order reference (paper order). We'll append any leftovers.
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
        # controlled_dataset (in this order)
        "OhioT1DM",
        "DiaTrend",
        "T1DEXI",
    ]

    # Default model columns. If some folders are missing, the script will emit blanks.
    models = [
        ModelSpec(
            key="moirai",
            display_name="Moirai2.0",
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
        ModelSpec(
            key="gpformer",
            display_name="GPFormer",
            results_roots={
                "fewshot": Path("GPFormer/multi_horizon_results_gpformer_fewshot"),
                "fullshot": Path("GPFormer/multi_horizon_results_gpformer_fullshot"),
            },
        ),
    ]

    open_datasets, controlled_datasets = _iter_datasets_from_test_root(args.test_root)
    open_set = set(open_datasets)
    controlled_set = set(controlled_datasets)

    # Add any datasets that appear in metrics files but are missing from test_dataset.
    for m in models:
        root = m.results_roots.get(args.shot)
        if root is None:
            continue
        for ds in _discover_metrics_files(root).keys():
            if ds not in open_set and ds not in controlled_set:
                open_datasets.append(ds)
                open_set.add(ds)

    # Pre-index metrics files for speed.
    metrics_paths: Dict[str, Dict[str, Path]] = {}
    for m in models:
        root = m.results_roots.get(args.shot)
        metrics_paths[m.key] = _discover_metrics_files(root) if root is not None else {}

    # Cache per-participant frames for combined rows: (model_key, dataset_raw) -> df
    per_participant_cache: Dict[Tuple[str, str], pd.DataFrame] = {}

    def get_participants(m: ModelSpec, dataset_raw: str) -> pd.DataFrame:
        cache_key = (m.key, dataset_raw)
        if cache_key in per_participant_cache:
            return per_participant_cache[cache_key]
        p = metrics_paths[m.key].get(dataset_raw)
        if p is None:
            per_participant_cache[cache_key] = pd.DataFrame(columns=["rmse", "mae"])
            return per_participant_cache[cache_key]
        per_participant_cache[cache_key] = _load_per_participant_metrics(
            p,
            context_hours=args.context_hours,
            horizon_minutes=args.horizon_minutes,
            eval_stride_steps=args.eval_stride_steps,
        )
        return per_participant_cache[cache_key]

    def dataset_row(dataset_raw: str) -> Dict[str, object]:
        row: Dict[str, object] = {"dataset": _display_dataset_name(dataset_raw)}
        for m in models:
            per_participant = get_participants(m, dataset_raw)
            mean, std, n = _summary_mean_std(per_participant.get(args.metric, pd.Series(dtype=float)))
            row[m.key] = _format_mean_pm(mean, std, digits=args.digits)
            row[f"{m.key}_n"] = n
        return row

    def combined_row(name: str, dataset_raws: Sequence[str]) -> Dict[str, object]:
        row: Dict[str, object] = {"dataset": name}
        for m in models:
            frames = []
            for ds in dataset_raws:
                per_participant = get_participants(m, ds)
                if per_participant.empty:
                    continue
                # Avoid participant_id collisions across datasets.
                tmp = per_participant.copy()
                tmp.index = pd.MultiIndex.from_product([[ds], tmp.index], names=["dataset", "participant_id"])
                frames.append(tmp)
            if not frames:
                row[m.key] = None
                row[f"{m.key}_n"] = 0
                continue
            pooled = pd.concat(frames, axis=0, copy=False)
            mean, std, n = _summary_mean_std(pooled[args.metric])
            row[m.key] = _format_mean_pm(mean, std, digits=args.digits)
            row[f"{m.key}_n"] = n
        return row

    # Build rows in paper-like layout.
    rows: List[Dict[str, object]] = []
    if args.include_section_rows:
        rows.append({"dataset": "OpenAccess"})

    for ds in open_datasets:
        rows.append(dataset_row(ds))

    rows.append(combined_row("Combined Open", open_datasets))

    if args.include_section_rows:
        rows.append({"dataset": "ControlledAccess"})

    for ds in controlled_datasets:
        rows.append(dataset_row(ds))

    rows.append(combined_row("Combined All", list(open_datasets) + list(controlled_datasets)))

    table = pd.DataFrame(rows).set_index("dataset")

    # Reorder datasets within each section based on desired_dataset_order display names.
    # This preserves the section label rows (if present) and keeps Combined rows at the end.
    model_cols = [m.key for m in models]
    n_cols = [f"{m.key}_n" for m in models]

    def sort_block(block: pd.DataFrame) -> pd.DataFrame:
        idx = block.index.tolist()
        remaining = [d for d in idx if d not in desired_dataset_order]
        ordered = [d for d in desired_dataset_order if d in idx]
        return block.reindex(ordered + sorted(remaining))

    if args.include_section_rows and "OpenAccess" in table.index and "ControlledAccess" in table.index:
        open_start = table.index.get_loc("OpenAccess")
        ctrl_start = table.index.get_loc("ControlledAccess")

        # Slice blocks by position (robust even if some datasets are missing).
        open_block = table.iloc[open_start + 1 : ctrl_start, :]
        ctrl_block = table.iloc[ctrl_start + 1 :, :]

        # Keep Combined rows at the end of each block.
        open_combined = open_block.loc[open_block.index == "Combined Open"]
        open_block = open_block.loc[open_block.index != "Combined Open"]
        ctrl_combined = ctrl_block.loc[ctrl_block.index == "Combined All"]
        ctrl_block = ctrl_block.loc[ctrl_block.index != "Combined All"]

        open_block = sort_block(open_block)
        ctrl_block = sort_block(ctrl_block)

        table = pd.concat(
            [
                table.iloc[open_start : open_start + 1, :],
                open_block,
                open_combined,
                table.iloc[ctrl_start : ctrl_start + 1, :],
                ctrl_block,
                ctrl_combined,
            ],
            axis=0,
        )
    else:
        # No section rows: just apply ordering, leaving combined rows last.
        combined_names = {"Combined Open", "Combined All"}
        combined = table.loc[table.index.isin(combined_names)]
        non_combined = table.loc[~table.index.isin(combined_names)]
        table = pd.concat([sort_block(non_combined), combined], axis=0)

    # Export: one CSV with formatted cells; and one debug CSV with n counts.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / f"{args.shot}_ctx{args.context_hours}h_hor{args.horizon_minutes}m_{args.metric}.csv"
    out_csv_n = args.output_dir / f"{args.shot}_ctx{args.context_hours}h_hor{args.horizon_minutes}m_{args.metric}_with_n.csv"

    display_cols = {m.key: m.display_name for m in models}
    display_n_cols = {f"{m.key}_n": f"{m.display_name}_n" for m in models}

    table[model_cols].rename(columns=display_cols).to_csv(out_csv, index=True)
    table[model_cols + n_cols].rename(columns={**display_cols, **display_n_cols}).to_csv(out_csv_n, index=True)

    print(f"\n[{args.shot}] metric={args.metric} -> {out_csv.as_posix()}")
    with pd.option_context("display.max_rows", 200, "display.max_columns", 200, "display.width", 200):
        print(table[model_cols].rename(columns=display_cols))
    print(f"\n(debug) with n -> {out_csv_n.as_posix()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
