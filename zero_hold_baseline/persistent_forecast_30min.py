"""
Persistent (zero-order hold) baseline for glucose forecasting at multiple
horizons (15, 30, 60, 90 minutes).

For each valid time point t in a patient's CGM trace, uses the CGM value at t
as the prediction for t+H. Only pairs with a fully contiguous H-minute window
(exactly H/5 × 5-min steps) are included.

Two dataset sources are supported via --source:
  - test       → ../test_dataset            → outputs persistent_baseline_{H}min(.csv|_summary.csv)
  - controlled → ../controlled_test_dataset → outputs persistent_baseline_{H}min_controlled(.csv|_summary.csv)
  - both       → run both in one invocation

Outputs (per horizon H, written to results/persistance_baseline_{H}_raw/)
-----------------------------------------------------------------------
persistent_baseline_{H}min[_controlled].csv
    All (Patient, Dataset, Horizon, Prediction, GroundTruth) rows.

persistent_baseline_{H}min[_controlled]_summary.csv
    Per-patient RMSE, MAE, and window count, plus dataset-level aggregates.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

STEP_MINUTES = 5
HORIZONS_MINUTES = [15, 30, 60, 90]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

SOURCES = {
    "test": {
        "root": PROJECT_ROOT / "test_dataset",
        "suffix": "",
    },
    "controlled": {
        "root": PROJECT_ROOT / "controlled_test_dataset",
        "suffix": "_controlled",
    },
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column detection (same conventions as the TimesFM scripts)
# ---------------------------------------------------------------------------

def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lowered = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def load_patient_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    timestamp_col = _find_column(
        df.columns,
        ["timestamp", "time", "datetime", "date_time", "date"],
    )
    value_col = _find_column(
        df.columns,
        [
            "bgvalue",
            "glucose",
            "glucose_value",
            "glucose_value_mg_dl",
            "glucose_value_mmol_l",
            "sensor_glucose",
            "value",
        ],
    )

    if timestamp_col is None:
        raise ValueError(f"No timestamp column found in {csv_path}")
    if value_col is None:
        if len(df.columns) >= 2:
            value_col = df.columns[1]
        else:
            raise ValueError(f"No value column found in {csv_path}")

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col, value_col])
    df = (
        df[[timestamp_col, value_col]]
        .rename(columns={timestamp_col: "timestamp", value_col: "value"})
        .sort_values("timestamp")
        .drop_duplicates(subset="timestamp")
        .reset_index(drop=True)
    )
    return df


# ---------------------------------------------------------------------------
# Persistent forecast
# ---------------------------------------------------------------------------

def persistent_pairs(df: pd.DataFrame, horizon_steps: int) -> List[dict]:
    """
    Return list of {Prediction, GroundTruth} dicts for all contiguous
    t → t+(horizon_steps * 5min) pairs in the dataframe.
    """
    if len(df) < horizon_steps + 1:
        return []

    timestamps = df["timestamp"].to_numpy(dtype="datetime64[ns]")
    values = df["value"].to_numpy(dtype="float64")

    step_ns = np.timedelta64(STEP_MINUTES, "m")

    pairs = []
    for i in range(len(df) - horizon_steps):
        window_ts = timestamps[i : i + horizon_steps + 1]
        diffs = window_ts[1:] - window_ts[:-1]
        if not np.all(diffs == step_ns):
            continue

        pred_val = float(values[i])
        gt_val = float(values[i + horizon_steps])

        if np.isnan(pred_val) or np.isnan(gt_val):
            continue

        pairs.append({"Prediction": pred_val, "GroundTruth": gt_val})

    return pairs


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def write_horizon_outputs(
    horizon: int,
    all_records: List[dict],
    summary_records: List[dict],
    file_suffix: str = "",
) -> None:
    horizon_label = f"{horizon}min"
    out_dir = RESULTS_DIR / f"persistance_baseline_{horizon}_raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"persistent_baseline_{horizon_label}{file_suffix}.csv"
    summary_path = out_dir / f"persistent_baseline_{horizon_label}{file_suffix}_summary.csv"

    if not all_records:
        LOGGER.error("No records collected for horizon %s. Skipping.", horizon_label)
        return

    raw_df = pd.DataFrame(
        all_records,
        columns=["Patient", "Dataset", "Horizon", "Prediction", "GroundTruth"],
    )
    raw_df.to_csv(raw_path, index=False)
    LOGGER.info("Saved %d %s prediction rows to %s", len(raw_df), horizon_label, raw_path)

    summary_df = pd.DataFrame(
        summary_records,
        columns=["Dataset", "Patient", "Horizon", "N_windows", "RMSE", "MAE"],
    )

    agg_rows = []
    for dataset, grp in summary_df.groupby("Dataset", sort=False):
        total_windows = int(grp["N_windows"].sum())
        weighted_rmse = float(
            np.sqrt(np.average(grp["RMSE"] ** 2, weights=grp["N_windows"]))
        )
        weighted_mae = float(
            np.average(grp["MAE"], weights=grp["N_windows"])
        )
        agg_rows.append({
            "Dataset": dataset,
            "Patient": "ALL_PATIENTS",
            "Horizon": horizon_label,
            "N_windows": total_windows,
            "RMSE": round(weighted_rmse, 4),
            "MAE": round(weighted_mae, 4),
        })

    summary_df = pd.concat(
        [summary_df, pd.DataFrame(agg_rows)], ignore_index=True
    )
    summary_df.to_csv(summary_path, index=False)
    LOGGER.info("Saved summary to %s", summary_path)

    print(f"\n=== {horizon_label} Dataset-level Summary (weighted) ===")
    print(f"{'Dataset':<32} {'Patients':>8} {'Windows':>10} {'MAE':>8} {'RMSE':>8}")
    print("-" * 72)
    for row in agg_rows:
        n_patients = int(summary_df[
            (summary_df["Dataset"] == row["Dataset"]) &
            (summary_df["Patient"] != "ALL_PATIENTS")
        ].shape[0])
        print(
            f"{row['Dataset']:<32} {n_patients:>8} {row['N_windows']:>10,} "
            f"{row['MAE']:>8.2f} {row['RMSE']:>8.2f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_source(source: str) -> None:
    config = SOURCES[source]
    dataset_root: Path = config["root"]
    file_suffix: str = config["suffix"]

    if not dataset_root.exists():
        LOGGER.error("Dataset root %s does not exist. Skipping source=%s.", dataset_root, source)
        return

    dataset_dirs = sorted(p for p in dataset_root.iterdir() if p.is_dir())
    if not dataset_dirs:
        LOGGER.error("No dataset folders found in %s", dataset_root)
        return

    LOGGER.info("=== Running source=%s (root=%s, suffix='%s') ===", source, dataset_root, file_suffix)

    all_records: Dict[int, List[dict]] = {h: [] for h in HORIZONS_MINUTES}
    summary_records: Dict[int, List[dict]] = {h: [] for h in HORIZONS_MINUTES}

    for dataset_dir in dataset_dirs:
        csv_files = sorted(dataset_dir.rglob("*.csv"))
        LOGGER.info("Dataset %s: %d files", dataset_dir.name, len(csv_files))

        for csv_path in tqdm(csv_files, desc=dataset_dir.name):
            patient_id = csv_path.stem
            try:
                df = load_patient_csv(csv_path)
            except ValueError as exc:
                LOGGER.warning("Skipping %s: %s", csv_path, exc)
                continue

            for horizon in HORIZONS_MINUTES:
                horizon_steps = horizon // STEP_MINUTES
                horizon_label = f"{horizon}min"
                pairs = persistent_pairs(df, horizon_steps)
                if not pairs:
                    LOGGER.warning(
                        "No valid %s pairs for %s / %s",
                        horizon_label, dataset_dir.name, patient_id,
                    )
                    continue

                preds = np.array([p["Prediction"] for p in pairs])
                gts = np.array([p["GroundTruth"] for p in pairs])
                errors = preds - gts
                patient_rmse = float(np.sqrt(np.mean(errors ** 2)))
                patient_mae = float(np.mean(np.abs(errors)))

                summary_records[horizon].append({
                    "Dataset": dataset_dir.name,
                    "Patient": patient_id,
                    "Horizon": horizon_label,
                    "N_windows": len(pairs),
                    "RMSE": round(patient_rmse, 4),
                    "MAE": round(patient_mae, 4),
                })

                for pair in pairs:
                    all_records[horizon].append({
                        "Patient": patient_id,
                        "Dataset": dataset_dir.name,
                        "Horizon": horizon_label,
                        **pair,
                    })

        LOGGER.info(
            "Dataset %s done. Records so far: %s",
            dataset_dir.name,
            {f"{h}min": len(all_records[h]) for h in HORIZONS_MINUTES},
        )

    for horizon in HORIZONS_MINUTES:
        write_horizon_outputs(
            horizon,
            all_records[horizon],
            summary_records[horizon],
            file_suffix=file_suffix,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["test", "controlled", "both"],
        default="test",
        help=(
            "Which dataset to score: 'test' (../test_dataset), "
            "'controlled' (../controlled_test_dataset), or 'both'."
        ),
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sources = ["test", "controlled"] if args.source == "both" else [args.source]
    for source in sources:
        run_source(source)


if __name__ == "__main__":
    main()
