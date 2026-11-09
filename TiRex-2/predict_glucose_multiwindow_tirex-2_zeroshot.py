"""TiRex-2 zero-shot evaluation on the GlucoFM benchmark.

Stage 1 forecasts the test split with ``tirex2_common`` (test-time differencing off, sign flip on).
Stage 2 scores the saved raw predictions for each participant and averages them over datasets.
See ``tirex-2.md`` for usage, arguments and outputs.
"""

import argparse
import logging
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Importing tirex2_common also puts the repository root on sys.path, needed by the imports below.
import tirex2_common as base

from clarke._glycemic_common import STRATA
from clarke.ceg_seg_upgrade import ceg_zone_pcts, seg_scores, seg_zone_distribution
from glucofm_data import DEFAULT_HF_NAME, iter_glucofm_subjects_from_hf

RESULTS_ROOT = base.RESULTS_ROOT
RAW_DIR = RESULTS_ROOT / "raw_predictions"
METRICS_DIR = RESULTS_ROOT / "full_metrics"
OUTPUT_PREFIX = "tirex2_zeroshot"
RAW_SUFFIX = "_test_raw_predictions.csv"

OPEN_EXCLUDED = base.CONTROLLED_ACCESS_DATASETS
STRATUM_KEYS = {"Hypo (<70)": "hypo", "In-range [70,180]": "inrange", "Hyper (>180)": "hyper"}

POINT_METRICS = ["rmse", "mse", "mae"]
STRATIFIED_METRICS = [f"{prefix}_{key}" for key in STRATUM_KEYS.values() for prefix in ("rmse", "n")]
CEG_METRICS = [f"ceg_{zone}" for zone in "ABCDE"] + ["ceg_AB"]
SEG_ZONES = {"None": "seg_none", "Slight": "seg_slight", "Moderate": "seg_moderate", "Great": "seg_great", "Extreme": "seg_extreme"}
SEG_METRICS = ["seg_no_risk", "seg_mean_risk", *SEG_ZONES.values()]
PERSISTENCE_METRICS = ["persist_rmse", "persist_mae", "skill_score"]
METRIC_COLUMNS = [
    "n_points",
    *POINT_METRICS,
    *STRATIFIED_METRICS,
    *CEG_METRICS,
    *SEG_METRICS,
    *base.PI_COLUMNS,
    *PERSISTENCE_METRICS,
]
KEY_COLUMNS = ["dataset", "participant_id", "context_hours", "horizon_minutes"]

OVERALL_VIEW = {
    "RMSE": "rmse",
    "MSE": "mse",
    "MAE": "mae",
    "RMSE hypo": "rmse_hypo",
    "RMSE in-range": "rmse_inrange",
    "RMSE hyper": "rmse_hyper",
    "CEG A (%)": "ceg_A",
    "SEG no-risk (%)": "seg_no_risk",
    "SEG |Risk|": "seg_mean_risk",
    "PI80 coverage": "PI80_coverage",
    "Skill score (%)": "skill_score",
}

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetTask:
    dataset: str
    raw_path: Path
    context_hours: List[int]
    horizons_minutes: List[int]
    data_source: str
    data_root: Path
    hf_name: str


def run_inference(args: argparse.Namespace) -> None:
    """Forecast every pending (participant, context, horizon) with ``tirex2_common``."""
    argv = [
        "--task", "eval",
        "--splits", "test",
        "--metric-mode", "final",
        "--stride-steps", "1",
        "--save-raw-predictions",
        "--save-quantile-predictions",
        "--device", args.device,
        "--batch-size", str(args.batch_size),
        "--data-source", args.data_source,
        "--data-root-test", str(args.data_root_test),
        "--hf-name", args.hf_name,
        "--context-hours", *map(str, args.context_hours),
        "--horizons-minutes", *map(str, args.horizons_minutes),
    ]  # fmt: skip
    if args.datasets is not None:
        argv += ["--datasets", *args.datasets]
    if args.overwrite:
        argv.append("--overwrite")
    base.main(argv)


def load_subject_values(task: DatasetTask, participant_ids: set) -> Dict[str, np.ndarray]:
    """Glucose arrays of the given participants, loaded exactly as at inference.

    ``window_start`` in the raw predictions indexes these arrays.
    """
    values: Dict[str, np.ndarray] = {}
    if task.data_source == "hf":
        for _, subject_id, df in iter_glucofm_subjects_from_hf(task.hf_name, "test", datasets={task.dataset}):
            if subject_id in participant_ids:
                values[subject_id] = df["value"].to_numpy(dtype="float32")
        return values
    groups = base.collect_subject_groups(task.data_root / task.dataset)
    for participant_id in participant_ids & set(groups):
        values[participant_id] = base.load_subject(groups[participant_id])["value"].to_numpy(dtype="float32")
    return values


def participant_metrics(pred: np.ndarray, truth: np.ndarray, persistence: np.ndarray) -> Dict[str, float]:
    """Every per-point metric for one participant at one (context, horizon)."""
    err = pred - truth
    row: Dict[str, float] = {
        "n_points": int(err.size),
        "rmse": math.sqrt(np.mean(err**2)),
        "mse": float(np.mean(err**2)),
        "mae": float(np.mean(np.abs(err))),
    }
    for label, in_stratum in STRATA:
        key = STRATUM_KEYS[label]
        mask = in_stratum(truth)
        row[f"n_{key}"] = int(mask.sum())
        row[f"rmse_{key}"] = math.sqrt(np.mean(err[mask] ** 2)) if mask.any() else np.nan

    ceg = ceg_zone_pcts(truth, pred)
    row.update({f"ceg_{zone}": ceg[zone] for zone in "ABCDE"})
    row["ceg_AB"] = ceg["A"] + ceg["B"]
    row["seg_mean_risk"], row["seg_no_risk"] = seg_scores(truth, pred)
    row.update({SEG_ZONES[zone]: pct for zone, pct in seg_zone_distribution(truth, pred).items()})

    persist_err = persistence - truth
    row["persist_rmse"] = math.sqrt(np.mean(persist_err**2))
    row["persist_mae"] = float(np.mean(np.abs(persist_err)))
    row["skill_score"] = (1 - row["rmse"] / row["persist_rmse"]) * 100 if row["persist_rmse"] > 0 else np.nan
    return row


def score_dataset(task: DatasetTask) -> pd.DataFrame:
    """Per-participant metrics of one dataset, for every requested context and horizon."""
    raw = pd.read_csv(
        task.raw_path,
        usecols=[
            "participant_id", "context_hours", "horizon_minutes", "stride_steps",
            "metric_mode", "window_start", "horizon_step", "prediction", "ground_truth",
        ],
        dtype={"participant_id": str, "metric_mode": "category", "prediction": "float32", "ground_truth": "float32"},
    )  # fmt: skip
    raw = raw[
        (raw["stride_steps"] == 1)
        & (raw["metric_mode"] == "final")
        & raw["context_hours"].isin(task.context_hours)
        & raw["horizon_minutes"].isin(task.horizons_minutes)
    ]
    if raw.empty:
        LOGGER.warning("%s: no raw predictions for the requested contexts / horizons", task.dataset)
        return pd.DataFrame()

    subjects = load_subject_values(task, set(raw["participant_id"].unique()))
    context_steps = {c: base.context_steps_from_hours(c) for c in raw["context_hours"].unique()}
    rows: List[Dict[str, object]] = []
    for participant_id, sub in raw.groupby("participant_id", sort=False):
        values = subjects.get(participant_id)
        if values is None:
            LOGGER.warning("%s / %s: subject data not found, skipped", task.dataset, participant_id)
            continue
        # Persistence = last context value; the ground truth must sit horizon_step later.
        last_idx = sub["window_start"].to_numpy() + sub["context_hours"].map(context_steps).to_numpy() - 1
        truth_idx = last_idx + sub["horizon_step"].to_numpy()
        if truth_idx.max() >= len(values) or not np.array_equal(values[truth_idx], sub["ground_truth"].to_numpy()):
            raise RuntimeError(
                f"{task.dataset} / {participant_id}: raw predictions do not line up with the subject data "
                "(was the data changed since the forecasts were made?)"
            )
        sub = sub.assign(persistence=values[last_idx])
        for (context_hours, horizon_minutes), g in sub.groupby(["context_hours", "horizon_minutes"]):
            rows.append(
                {
                    "dataset": task.dataset,
                    "participant_id": participant_id,
                    "context_hours": int(context_hours),
                    "horizon_minutes": int(horizon_minutes),
                    **participant_metrics(
                        g["prediction"].to_numpy(dtype="float64"),
                        g["ground_truth"].to_numpy(dtype="float64"),
                        g["persistence"].to_numpy(dtype="float64"),
                    ),
                }
            )
    per_participant = pd.DataFrame(rows)
    LOGGER.info("%s: %d points, %d participant rows", task.dataset, len(raw), len(per_participant))
    return merge_interval_metrics(per_participant, task.dataset)


def merge_interval_metrics(per_participant: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Add PI coverage / width from the metrics CSV (raw predictions hold no quantiles)."""
    if per_participant.empty:
        return per_participant
    metrics_path = RESULTS_ROOT / f"{dataset}_test_metrics.csv"
    keys = KEY_COLUMNS[1:]
    if not metrics_path.exists():
        LOGGER.warning("%s: %s missing, PI metrics left empty", dataset, metrics_path.name)
        return per_participant.reindex(columns=[*per_participant.columns, *base.PI_COLUMNS])
    metrics = pd.read_csv(metrics_path, dtype={"participant_id": str})
    metrics = metrics[(metrics["stride_steps"] == 1) & (metrics["metric_mode"] == "final")]
    metrics = metrics.reindex(columns=[*keys, "rmse", *base.PI_COLUMNS]).rename(columns={"rmse": "pipeline_rmse"})
    merged = per_participant.merge(metrics, on=keys, how="left", validate="one_to_one")
    mismatch = (merged["rmse"] - merged["pipeline_rmse"]).abs() > 1e-3
    if mismatch.any():
        LOGGER.warning("%s: %d rows where raw-prediction RMSE differs from the metrics CSV", dataset, mismatch.sum())
    if merged["pipeline_rmse"].isna().any():
        LOGGER.warning("%s: %d rows without a metrics-CSV row (no PI metrics)", dataset, merged["pipeline_rmse"].isna().sum())
    return merged.drop(columns="pipeline_rmse")


def summarize(per_participant: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over participants per dataset, Overall (Open) and Overall (Total)."""
    stacked = pd.concat(
        [
            per_participant,
            per_participant[~per_participant["dataset"].isin(OPEN_EXCLUDED)].assign(dataset="Overall (Open)"),
            per_participant.assign(dataset="Overall (Total)"),
        ],
        ignore_index=True,
    )
    grouped = stacked.groupby(["context_hours", "horizon_minutes", "dataset"])
    summary = grouped[METRIC_COLUMNS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.insert(0, "n_participants", grouped.size())
    summary["skill_score_of_means"] = (1 - summary["rmse_mean"] / summary["persist_rmse_mean"]) * 100
    summary = summary.reset_index()
    summary["_order"] = summary["dataset"].map({"Overall (Open)": 1, "Overall (Total)": 2}).fillna(0)
    return (
        summary.sort_values(["context_hours", "horizon_minutes", "_order", "dataset"])
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def overall_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Overall (Total) as paper-style ``mean (std)`` cells, one row per context and horizon."""
    total = summary[summary["dataset"] == "Overall (Total)"].set_index(["context_hours", "horizon_minutes"])
    view = pd.DataFrame(index=total.index)
    view["N"] = total["n_participants"]
    for label, col in OVERALL_VIEW.items():
        digits = 4 if col in ("seg_mean_risk", "PI80_coverage") else 2
        view[label] = [
            f"{m:.{digits}f} ({s:.{digits}f})" for m, s in zip(total[f"{col}_mean"], total[f"{col}_std"])
        ]
    view["Skill (of means, %)"] = total["skill_score_of_means"].round(2)
    return view


def compute_metrics(args: argparse.Namespace) -> None:
    raw_paths = sorted(RAW_DIR.glob(f"*{RAW_SUFFIX}"))
    if not raw_paths:
        raise FileNotFoundError(f"No raw predictions in {RAW_DIR}; run without --metrics-only first.")
    tasks = []
    for raw_path in raw_paths:
        dataset = raw_path.name[: -len(RAW_SUFFIX)]
        if args.datasets is not None and dataset not in args.datasets:
            continue
        if args.datasets is None and dataset in base.SKIP_DATASET_DIRS:
            continue
        tasks.append(
            DatasetTask(
                dataset=dataset,
                raw_path=raw_path,
                context_hours=list(args.context_hours),
                horizons_minutes=list(args.horizons_minutes),
                data_source=args.data_source,
                data_root=args.data_root_test,
                hf_name=args.hf_name,
            )
        )
    if not tasks:
        raise ValueError(f"No raw predictions for datasets {args.datasets} in {RAW_DIR}")

    LOGGER.info("Scoring %d datasets with %d worker(s)", len(tasks), args.workers)
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            frames = list(pool.map(score_dataset, tasks))
    else:
        frames = [score_dataset(task) for task in tasks]
    per_participant = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    per_participant = per_participant[[*KEY_COLUMNS, *METRIC_COLUMNS]].sort_values(KEY_COLUMNS)

    found = set(zip(per_participant["context_hours"], per_participant["horizon_minutes"]))
    missing = [(c, h) for c in args.context_hours for h in args.horizons_minutes if (c, h) not in found]
    if missing:
        LOGGER.warning("No raw predictions for (context h, horizon min) %s", missing)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    summary = summarize(per_participant)
    overall = overall_table(summary)
    outputs = {
        "per_participant": per_participant,
        "summary": summary.round(4),
        "overall": overall.reset_index(),
    }
    for name, df in outputs.items():
        path = METRICS_DIR / f"{OUTPUT_PREFIX}_{name}.csv"
        df.to_csv(path, index=False)
        LOGGER.info("Saved %s (%d rows)", path, len(df))

    with pd.option_context("display.width", 250, "display.max_columns", None):
        print("\nTiRex-2 zero-shot (tta_diff off, sign flip on), Overall (Total), mean (std) over participants:")
        print(overall.to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--device",
        type=str,
        default=base.default_device(),
        help="Device to load TiRex-2 on (e.g. 'cuda', 'cuda:0', 'mps', 'cpu').",
    )
    parser.add_argument("--batch-size", type=int, default=base.DEFAULT_BATCH_SIZE, help="Windows per forecast call.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help=f"Dataset names to run / score (default: all except {sorted(base.SKIP_DATASET_DIRS)}).",
    )
    parser.add_argument("--data-source", choices=["csv", "hf"], default="csv")
    parser.add_argument("--data-root-test", type=Path, default=base.DATA_SPLITS["test"])
    parser.add_argument("--hf-name", type=str, default=DEFAULT_HF_NAME)
    parser.add_argument(
        "--context-hours",
        type=int,
        nargs="+",
        default=base.INPUT_WINDOWS_HOURS,
        help=f"Context window(s) in hours (default: {base.INPUT_WINDOWS_HOURS}).",
    )
    parser.add_argument(
        "--horizons-minutes",
        type=int,
        nargs="+",
        default=base.HORIZONS_MINUTES,
        help=f"Forecast horizon(s) in minutes (default: {base.HORIZONS_MINUTES}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Recompute the forecasts instead of reusing cached rows. Rewrites each dataset's metrics "
            "and raw files with only this run's contexts / horizons."
        ),
    )
    parser.add_argument("--metrics-only", action="store_true", help="Skip inference; score the saved raw predictions.")
    parser.add_argument("--workers", type=int, default=4, help="Datasets scored in parallel (metrics stage).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.metrics_only:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    else:
        run_inference(args)
    LOGGER.setLevel(logging.INFO)
    compute_metrics(args)


if __name__ == "__main__":
    main()
