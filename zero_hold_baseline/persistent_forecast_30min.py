"""
Persistent (zero-order hold) baseline for 30-minute glucose forecasting.

For each valid time point t in a patient's CGM trace, uses the CGM value at t
as the prediction for t+30min. Only pairs with a contiguous 30-minute gap
(exactly 6 × 5-min steps) are included.

Outputs
-------
results/persistent_baseline_30min.csv
    All (Patient, Dataset, Horizon, Prediction, GroundTruth) rows — use this
    for CEG / SEG plotting.

results/persistent_baseline_30min_summary.csv
    Per-patient RMSE, MAE, and window count, plus dataset-level aggregates.
"""

import logging
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

STEP_MINUTES = 5
HORIZON_MINUTES = 30
HORIZON_STEPS = HORIZON_MINUTES // STEP_MINUTES  # 6

TEST_DATASET_ROOT = Path(__file__).resolve().parent.parent / "test_dataset"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

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

def persistent_pairs(df: pd.DataFrame) -> List[dict]:
    """
    Return list of {Prediction, GroundTruth} dicts for all contiguous
    t → t+30min pairs in the dataframe.
    """
    if len(df) < HORIZON_STEPS + 1:
        return []

    timestamps = df["timestamp"].to_numpy(dtype="datetime64[ns]")
    values = df["value"].to_numpy(dtype="float64")

    step_ns = np.timedelta64(STEP_MINUTES, "m")

    pairs = []
    for i in range(len(df) - HORIZON_STEPS):
        # Check the full 30-min window is contiguous (all 6 gaps are 5 min)
        window_ts = timestamps[i : i + HORIZON_STEPS + 1]
        diffs = window_ts[1:] - window_ts[:-1]
        if not np.all(diffs == step_ns):
            continue

        pred_val = float(values[i])
        gt_val = float(values[i + HORIZON_STEPS])

        if np.isnan(pred_val) or np.isnan(gt_val):
            continue

        pairs.append({"Prediction": pred_val, "GroundTruth": gt_val})

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_output_path = RESULTS_DIR / "persistent_baseline_30min.csv"
    summary_output_path = RESULTS_DIR / "persistent_baseline_30min_summary.csv"

    dataset_dirs = sorted(p for p in TEST_DATASET_ROOT.iterdir() if p.is_dir())
    if not dataset_dirs:
        LOGGER.error("No dataset folders found in %s", TEST_DATASET_ROOT)
        return

    all_records: List[dict] = []
    summary_records: List[dict] = []

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

            pairs = persistent_pairs(df)
            if not pairs:
                LOGGER.warning("No valid pairs for %s / %s", dataset_dir.name, patient_id)
                continue

            preds = np.array([p["Prediction"] for p in pairs])
            gts = np.array([p["GroundTruth"] for p in pairs])
            errors = preds - gts
            patient_rmse = float(np.sqrt(np.mean(errors ** 2)))
            patient_mae = float(np.mean(np.abs(errors)))
            n_windows = len(pairs)

            summary_records.append({
                "Dataset": dataset_dir.name,
                "Patient": patient_id,
                "Horizon": "30min",
                "N_windows": n_windows,
                "RMSE": round(patient_rmse, 4),
                "MAE": round(patient_mae, 4),
            })

            for pair in pairs:
                all_records.append({
                    "Patient": patient_id,
                    "Dataset": dataset_dir.name,
                    "Horizon": "30min",
                    **pair,
                })

        LOGGER.info("Dataset %s done. Records so far: %d", dataset_dir.name, len(all_records))

    if not all_records:
        LOGGER.error("No records collected. Check your data paths.")
        return

    # --- save raw predictions ---
    raw_df = pd.DataFrame(all_records, columns=["Patient", "Dataset", "Horizon", "Prediction", "GroundTruth"])
    raw_df.to_csv(raw_output_path, index=False)
    LOGGER.info("Saved %d prediction rows to %s", len(raw_df), raw_output_path)

    # --- build and save summary ---
    summary_df = pd.DataFrame(summary_records, columns=["Dataset", "Patient", "Horizon", "N_windows", "RMSE", "MAE"])

    # Append dataset-level aggregate rows
    agg_rows = []
    for dataset, grp in summary_df.groupby("Dataset", sort=False):
        total_windows = int(grp["N_windows"].sum())
        # Weighted (by window count) RMSE and MAE across patients
        weighted_rmse = float(
            np.sqrt(np.average(grp["RMSE"] ** 2, weights=grp["N_windows"]))
        )
        weighted_mae = float(
            np.average(grp["MAE"], weights=grp["N_windows"])
        )
        agg_rows.append({
            "Dataset": dataset,
            "Patient": "ALL_PATIENTS",
            "Horizon": "30min",
            "N_windows": total_windows,
            "RMSE": round(weighted_rmse, 4),
            "MAE": round(weighted_mae, 4),
        })

    summary_df = pd.concat(
        [summary_df, pd.DataFrame(agg_rows)], ignore_index=True
    )
    summary_df.to_csv(summary_output_path, index=False)
    LOGGER.info("Saved summary to %s", summary_output_path)

    # --- print console summary ---
    print("\n=== Dataset-level Summary (weighted) ===")
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


if __name__ == "__main__":
    main()
