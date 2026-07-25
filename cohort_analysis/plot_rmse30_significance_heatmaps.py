"""Plot pairwise RMSE_30min significance heatmaps for cohort_analysis results.

For each shot folder (zero, few, full), this script:
1. Loads every method CSV in the folder.
2. Normalizes dataset names and pairs rows by (Dataset, Patient).
3. Runs two-sided paired Wilcoxon signed-rank tests for every method pair.
4. Applies Benjamini-Hochberg FDR correction within each shot folder.
5. Saves a binary significance heatmap plus CSV tables of p-values and tests.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon


ALPHA = 0.05
RMSE_COLUMNS = ["RMSE_30min", "RMSE_30min_mgdl"]
SHOT_ORDER = ["zero", "few", "full"]
METHOD_ORDER = [
    "Chronos2",
    "CALF",
    "LSTM",
    "GPFormer",
    "TimeLLM",
    "Timer",
    "TimesFM",
    "Uni2TS",
]

DATASET_NAME_MAP = {
    "14_HUPA-UCM": "HUPA-UCM",
    "17_T1DM-UOM": "T1DM-UOM",
    "18_Bris-T1D Open": "Bris-T1D",
    "19_AZT1D": "AZT1D",
    "1_Hall2018": "hall",
    "2_D1NAMO": "d1namo",
    "5_T1DEXI": "T1DEXI",
    "8_DiaTrend": "DiaTrend",
    "BIG_IDEA_LAB": "BIG_IDEAs",
    "CGMacros": "CGMacros",
    "OhioT1DM": "OhioT1DM",
    "ShanghaiT1DM": "shanghait1dm",
    "ShanghaiT2DM": "shanghait2dm",
    "UCHTT1DM": "UCHTT1DM",
    "HUPA-UCM": "HUPA-UCM",
    "T1DM-UOM": "T1DM-UOM",
    "Bris-T1D Open": "Bris-T1D",
    "Bris-T1D": "Bris-T1D",
    "AZT1D": "AZT1D",
    "Hall2018": "hall",
    "hall": "hall",
    "D1NAMO": "d1namo",
    "d1namo": "d1namo",
    "T1DEXI": "T1DEXI",
    "t1dexi": "T1DEXI",
    "DiaTrend": "DiaTrend",
    "diatrend": "DiaTrend",
    "BIG_IDEAs": "BIG_IDEAs",
    "ShanghaiT1DM": "shanghait1dm",
    "shanghait1dm": "shanghait1dm",
    "ShanghaiT2DM": "shanghait2dm",
    "shanghait2dm": "shanghait2dm",
}


def method_name_from_path(path: Path) -> str:
    name = path.name.lower()

    if name == "all_datasets_result_with_phenotype.csv":
        return "Chronos2"
    if "chronos" in name:
        return "Chronos2"
    if "calf" in name:
        return "CALF"
    if "lstm" in name:
        return "LSTM"
    if "gpformer" in name:
        return "GPFormer"
    if "timellm" in name:
        return "TimeLLM"
    if "timer" in name:
        return "Timer"
    if "timesfm" in name:
        return "TimesFM"
    if "uni2ts" in name:
        return "Uni2TS"

    raise ValueError(f"Cannot infer method name from {path}")


def normalize_dataset(dataset: object) -> str:
    value = str(dataset).strip()
    value = re.sub(r"^\d+_", "", value)
    return DATASET_NAME_MAP.get(value, value)


def patient_key(patient: object) -> str:
    return str(patient).strip()


def read_method_csv(path: Path) -> pd.DataFrame:
    method = method_name_from_path(path)
    table = pd.read_csv(path, dtype={"Patient": str, "subject_id": str})

    rmse_col = next((col for col in RMSE_COLUMNS if col in table.columns), None)
    if rmse_col is None:
        raise ValueError(f"{path} does not contain any of {RMSE_COLUMNS}")

    if "Patient" not in table.columns and "subject_id" in table.columns:
        table = table.rename(columns={"subject_id": "Patient"})

    required = {"Dataset", "Patient", rmse_col}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    if "Context_Length" in table.columns:
        context_lengths = pd.to_numeric(table["Context_Length"], errors="coerce")
        if (context_lengths == 144).any():
            table = table[context_lengths == 144]

    clean = pd.DataFrame(
        {
            "method": method,
            "dataset": table["Dataset"].map(normalize_dataset),
            "patient": table["Patient"].map(patient_key),
            "RMSE_30min": pd.to_numeric(table[rmse_col], errors="coerce"),
        }
    ).dropna(subset=["RMSE_30min"])

    clean["participant_key"] = clean["dataset"] + "||" + clean["patient"]
    return (
        clean.groupby(["method", "participant_key"], as_index=False)["RMSE_30min"]
        .mean()
        .sort_values(["method", "participant_key"])
    )


def load_shot_folder(folder: Path) -> pd.DataFrame:
    csv_paths = sorted(folder.glob("*.csv"))
    tables = []

    for path in csv_paths:
        if path.name.startswith("rmse30_pairwise_") or path.name.startswith("rmse30_significance_"):
            continue
        tables.append(read_method_csv(path))

    if not tables:
        raise ValueError(f"No method CSVs found in {folder}")

    return pd.concat(tables, ignore_index=True)


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    adjusted = [math.nan] * len(p_values)
    valid = [(idx, p) for idx, p in enumerate(p_values) if not math.isnan(p)]
    if not valid:
        return adjusted

    order = sorted(valid, key=lambda item: item[1])
    total = len(order)
    running_min = 1.0

    for rank_from_end, (idx, p_value) in enumerate(reversed(order), start=1):
        rank = total - rank_from_end + 1
        running_min = min(running_min, p_value * total / rank)
        adjusted[idx] = min(running_min, 1.0)

    return adjusted


def paired_wilcoxon(method_a: pd.DataFrame, method_b: pd.DataFrame) -> dict:
    paired = method_a.merge(method_b, on="participant_key", suffixes=("_a", "_b"))
    n_pairs = len(paired)

    if n_pairs < 2:
        return {
            "n_pairs": n_pairs,
            "p_value": math.nan,
            "mean_rmse_diff_a_minus_b": math.nan,
        }

    differences = paired["RMSE_30min_a"] - paired["RMSE_30min_b"]
    mean_diff = float(differences.mean())

    if np.allclose(differences.to_numpy(), 0):
        return {
            "n_pairs": n_pairs,
            "p_value": 1.0,
            "mean_rmse_diff_a_minus_b": mean_diff,
        }

    p_value = float(wilcoxon(differences, zero_method="wilcox", alternative="two-sided").pvalue)
    return {
        "n_pairs": n_pairs,
        "p_value": p_value,
        "mean_rmse_diff_a_minus_b": mean_diff,
    }


def pairwise_tests(table: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    methods = [m for m in METHOD_ORDER if m in set(table["method"])]
    methods.extend(sorted(set(table["method"]) - set(methods)))

    by_method = {
        method: table[table["method"] == method][["participant_key", "RMSE_30min"]]
        for method in methods
    }

    rows = []
    for idx_a, method_a in enumerate(methods):
        for method_b in methods[idx_a + 1 :]:
            result = paired_wilcoxon(by_method[method_a], by_method[method_b])
            rows.append(
                {
                    "method_a": method_a,
                    "method_b": method_b,
                    **result,
                }
            )

    tests = pd.DataFrame(rows)
    tests["p_adj_bh"] = benjamini_hochberg(tests["p_value"].tolist())
    tests["significant_fdr_0.05"] = tests["p_adj_bh"] < ALPHA
    return tests, methods


def matrix_from_tests(tests: pd.DataFrame, methods: list[str], value_col: str, diagonal_value: float) -> pd.DataFrame:
    matrix = pd.DataFrame(np.nan, index=methods, columns=methods, dtype=float)
    for method in methods:
        matrix.loc[method, method] = diagonal_value

    for row in tests.itertuples(index=False):
        value = getattr(row, value_col)
        matrix.loc[row.method_a, row.method_b] = value
        matrix.loc[row.method_b, row.method_a] = value

    return matrix


def format_p_value(p_value: float) -> str:
    if math.isnan(p_value):
        return "NA"
    if p_value < 0.001:
        return f"{p_value:.1e}"
    return f"{p_value:.3f}"


def significance_stars(q_value: float) -> str:
    if math.isnan(q_value):
        return "NA"
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return "ns"


def annotation_matrix(tests: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    annotations = pd.DataFrame("", index=methods, columns=methods, dtype=object)
    for method in methods:
        annotations.loc[method, method] = "—"

    for _, row in tests.iterrows():
        stars = significance_stars(row["p_adj_bh"])
        text = (
            f"{stars}\n"
            f"p={format_p_value(row['p_value'])}\n"
            f"q={format_p_value(row['p_adj_bh'])}\n"
            f"n={int(row['n_pairs'])}"
        )
        annotations.loc[row["method_a"], row["method_b"]] = text
        annotations.loc[row["method_b"], row["method_a"]] = text

    return annotations


def save_heatmap(folder: Path, shot: str, tests: pd.DataFrame, methods: list[str]) -> None:
    significance = matrix_from_tests(
        tests.assign(significant_numeric=tests["significant_fdr_0.05"].astype(float)),
        methods,
        "significant_numeric",
        diagonal_value=np.nan,
    )
    annotations = annotation_matrix(tests, methods)

    size = max(7, 0.75 * len(methods) + 2)
    plt.figure(figsize=(size, size))
    sns.heatmap(
        significance,
        vmin=0,
        vmax=1,
        cmap=sns.color_palette(["#f3f4f6", "#ef4444"]),
        cbar_kws={"ticks": [0.25, 0.75], "label": "FDR-significant"},
        linewidths=0.8,
        linecolor="white",
        square=True,
        annot=annotations,
        fmt="",
        annot_kws={"fontsize": 8},
    )
    colorbar = plt.gcf().axes[-1]
    colorbar.set_yticklabels(["No", "Yes"])

    plt.title(
        f"{shot.capitalize()}-shot RMSE_30min pairwise significance\n"
        "Paired Wilcoxon tests with BH-FDR correction"
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    out_path = folder / "rmse30_significance_heatmap.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_outputs(folder: Path, tests: pd.DataFrame, methods: list[str]) -> None:
    p_values = matrix_from_tests(tests, methods, "p_value", diagonal_value=0.0)
    q_values = matrix_from_tests(tests, methods, "p_adj_bh", diagonal_value=0.0)
    significance = matrix_from_tests(
        tests.assign(significant_numeric=tests["significant_fdr_0.05"].astype(int)),
        methods,
        "significant_numeric",
        diagonal_value=0.0,
    )

    tests.to_csv(folder / "rmse30_pairwise_tests.csv", index=False)
    p_values.to_csv(folder / "rmse30_pairwise_pvalues.csv")
    q_values.to_csv(folder / "rmse30_pairwise_fdr_qvalues.csv")
    significance.to_csv(folder / "rmse30_significance_matrix.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort_dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Path to cohort_analysis directory.",
    )
    parser.add_argument(
        "--shots",
        nargs="+",
        default=SHOT_ORDER,
        choices=SHOT_ORDER,
        help="Shot folders to process.",
    )
    args = parser.parse_args()

    for shot in args.shots:
        folder = args.cohort_dir / shot
        table = load_shot_folder(folder)
        tests, methods = pairwise_tests(table)
        save_outputs(folder, tests, methods)
        save_heatmap(folder, shot, tests, methods)
        print(
            f"{shot}: {len(methods)} methods, {len(tests)} pairwise tests -> "
            f"{folder / 'rmse30_significance_heatmap.png'}"
        )


if __name__ == "__main__":
    main()
