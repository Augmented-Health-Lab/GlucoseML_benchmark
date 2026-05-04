import argparse
from pathlib import Path

import pandas as pd


DATASET_FOLDERS = {
    "fewshot": {
        "open": "test_metrics_open",
        "OhioT1DM": "test_metrics_OhioT1DM",
        "DiaTrend": "test_metrics_DiaTrend",
        "T1DEXI": "test_metrics_T1DEXI",
    },
    "fullshot": {
        "open": "test_metrics_open",
        "OhioT1DM": "test_metrics_ohio",
        "DiaTrend": "test_metrics_diatrend",
        "T1DEXI": "test_metrics_t1dexi",
    },
}

MODEL_DIRS = {
    "fewshot": "timellm_fewshot_raw",
    "fullshot": "timellm_fullshot_raw",
}

RMSE_COLUMN = "RMSE_30min_mgdl"
KEEP_COLUMNS = ["model", "dataset", "subject_id", "num_windows", RMSE_COLUMN]
ALL_DATASETS = ["open", "OhioT1DM", "DiaTrend", "T1DEXI"]


def read_per_subject_tables(results_dir: Path) -> pd.DataFrame:
    tables = []

    for model_name, dataset_folders in DATASET_FOLDERS.items():
        model_dir = results_dir / MODEL_DIRS[model_name]

        for dataset_name, folder_name in dataset_folders.items():
            csv_path = model_dir / folder_name / "per_subject_horizons.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing expected CSV: {csv_path}")

            table = pd.read_csv(csv_path)
            if RMSE_COLUMN not in table.columns:
                raise ValueError(f"{csv_path} does not contain {RMSE_COLUMN}")

            table.insert(0, "dataset", dataset_name)
            table.insert(0, "model", model_name)
            tables.append(table[KEEP_COLUMNS])

    combined = pd.concat(tables, ignore_index=True)
    combined[RMSE_COLUMN] = pd.to_numeric(combined[RMSE_COLUMN], errors="coerce")
    return combined.dropna(subset=[RMSE_COLUMN])


def summarize_scope(table: pd.DataFrame, scope: str, datasets: list[str]) -> list[dict]:
    rows = []

    for model_name in MODEL_DIRS:
        subset = table[(table["model"] == model_name) & (table["dataset"].isin(datasets))]
        rows.append(
            {
                "model": model_name,
                "scope": scope,
                "datasets": ";".join(datasets),
                "n_participants": int(len(subset)),
                "mean_RMSE_30min_mgdl": float(subset[RMSE_COLUMN].mean()),
                "std_RMSE_30min_mgdl": float(subset[RMSE_COLUMN].std(ddof=1)),
            }
        )

    return rows


def build_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.extend(summarize_scope(table, "open", ["open"]))
    rows.extend(summarize_scope(table, "all", ALL_DATASETS))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize participant-average TimeLLM 30-minute RMSE from per-subject CSVs."
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results",
        help="Path to the repository results directory.",
    )
    parser.add_argument(
        "--summary_out",
        type=Path,
        default=None,
        help="Output CSV path for compact summary.",
    )
    parser.add_argument(
        "--combined_out",
        type=Path,
        default=None,
        help="Output CSV path for combined per-subject RMSE table.",
    )
    args = parser.parse_args()

    summary_out = args.summary_out or args.results_dir / "timellm_rmse30_summary.csv"
    combined_out = args.combined_out or args.results_dir / "timellm_rmse30_per_subject_combined.csv"

    combined = read_per_subject_tables(args.results_dir)
    summary = build_summary(combined)

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    combined_out.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(summary_out, index=False)
    combined.to_csv(combined_out, index=False)

    print(f"Saved summary: {summary_out}")
    print(f"Saved combined table: {combined_out}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
