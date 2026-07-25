"""Aggregate per-participant 30-min RMSEs across datasets and merge with diabetes_type.csv.

Produces one CSV per (model, shot) under cohort_analysis/{zero,few,full}/.
"""

import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DIABETES_TYPE_CSV = REPO / "results" / "diabetes_type.csv"

# Map model dataset name -> diabetes_type.csv "Dataset" column.
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
}

# {model_key: {shot: metrics_dir}}
MODELS = {
    "timesfm": {
        "zero": REPO / "timesfm" / "multi_horizon_results_timesfm_zeroshot",
        "few": REPO / "timesfm" / "multi_horizon_results_timesfm_fewshot",
        "full": REPO / "timesfm" / "multi_horizon_results_timesfm_fullshot",
    },
    "uni2ts": {
        "zero": REPO / "uni2ts" / "multi_horizon_results_uni2ts_zeroshot",
        "few": REPO / "uni2ts" / "multi_horizon_results_uni2ts_fewshot",
        "full": REPO / "uni2ts" / "multi_horizon_results_uni2ts_fullshot",
    },
    "timer": {
        "zero": REPO / "timer-model" / "multi_horizon_results_timer_zeroshot",
        "few": REPO / "timer-model" / "multi_horizon_results_timer_fewshot",
        "full": REPO / "timer-model" / "multi_horizon_results_timer_fullshot",
    },
    "gpformer": {
        "few": REPO / "GPFormer" / "multi_horizon_results_gpformer_fewshot",
        "full": REPO / "GPFormer" / "multi_horizon_results_gpformer_fullshot",
    },
}

# TimeLLM has a different layout: results/timellm_<shot>shot_raw/test_metrics_<group>/per_subject_horizons.csv
TIMELLM_SHOTS = {
    "few": REPO / "results" / "timellm_fewshot_raw",
    "full": REPO / "results" / "timellm_fullshot_raw",
}

# Dedicated single-dataset subdirs (case differs between fewshot and fullshot).
TIMELLM_DEDICATED = {
    "DiaTrend": "8_DiaTrend",
    "diatrend": "8_DiaTrend",
    "OhioT1DM": "OhioT1DM",
    "ohio": "OhioT1DM",
    "T1DEXI": "5_T1DEXI",
    "t1dexi": "5_T1DEXI",
}


def route_open_subject(sid: str) -> str:
    """Return timesfm-style dataset name for a subject in the 'open' bundle."""
    if re.fullmatch(r"00[1-8]", sid):
        return "2_D1NAMO"
    if re.fullmatch(r"10\d{2}", sid):
        return "ShanghaiT1DM"
    if re.fullmatch(r"20\d{2}", sid):
        return "ShanghaiT2DM"
    if sid.startswith(("1636-", "2133-")):
        return "1_Hall2018"
    if sid.startswith("Subject "):
        return "19_AZT1D"
    if sid.startswith("HUPA"):
        return "14_HUPA-UCM"
    if sid.startswith("UoMGlucose"):
        return "17_T1DM-UOM"
    if sid.startswith("Dexcom_"):
        return "BIG_IDEA_LAB"
    if re.fullmatch(r"P\d{2}", sid):
        return "18_Bris-T1D Open"
    if sid.startswith(("HT_", "T1DM_")):
        return "UCHTT1DM"
    if sid.startswith("CGMacros-"):
        return "CGMacros"
    raise ValueError(f"Unrouted TimeLLM subject_id: {sid!r}")


def aggregate_timellm_shot(shot_root: Path) -> pd.DataFrame:
    rows = []
    for subdir in sorted(shot_root.glob("test_metrics_*")):
        if not subdir.is_dir():
            continue
        csv_path = subdir / "per_subject_horizons.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, dtype={"subject_id": str})
        df = df[["subject_id", "RMSE_30min_mgdl"]].rename(
            columns={"subject_id": "Patient", "RMSE_30min_mgdl": "RMSE_30min"}
        )
        group = subdir.name.removeprefix("test_metrics_")
        if group == "open":
            df["Dataset"] = df["Patient"].map(route_open_subject)
        elif group in TIMELLM_DEDICATED:
            df["Dataset"] = TIMELLM_DEDICATED[group]
        else:
            raise ValueError(f"Unknown TimeLLM subdir: {subdir.name!r}")
        rows.append(df[["Dataset", "Patient", "RMSE_30min"]])
    return pd.concat(rows, ignore_index=True)


def aggregate_shot(metrics_dir: Path) -> pd.DataFrame:
    rows = []
    for csv_path in sorted(metrics_dir.glob("*_test_metrics.csv")):
        df = pd.read_csv(csv_path, dtype={"participant_id": str})
        h30 = df[df["horizon_minutes"] == 30]
        if h30.empty:
            continue
        rows.append(
            h30[["dataset", "participant_id", "rmse"]].rename(
                columns={
                    "dataset": "Dataset",
                    "participant_id": "Patient",
                    "rmse": "RMSE_30min",
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def merge_with_diabetes_type(agg: pd.DataFrame, dtype_df: pd.DataFrame) -> pd.DataFrame:
    agg = agg.copy()
    agg["_dtype_dataset"] = agg["Dataset"].map(DATASET_NAME_MAP)
    unmapped = agg[agg["_dtype_dataset"].isna()]["Dataset"].unique()
    if len(unmapped) > 0:
        print(f"  WARNING: unmapped datasets: {list(unmapped)}")

    dtype_subset = dtype_df[["Dataset", "Patient", "diabetes_type"]].rename(
        columns={"Dataset": "_dtype_dataset"}
    )
    merged = agg.merge(
        dtype_subset, on=["_dtype_dataset", "Patient"], how="left"
    )

    missing = merged[merged["diabetes_type"].isna()]
    if not missing.empty:
        print(f"  WARNING: {len(missing)} rows missing diabetes_type")
        for ds, sub in missing.groupby("Dataset"):
            print(f"    {ds}: {len(sub)} (e.g. {sub['Patient'].head(3).tolist()})")

    return merged.drop(columns=["_dtype_dataset"])[
        ["Dataset", "Patient", "RMSE_30min", "diabetes_type"]
    ]


def main() -> None:
    dtype_df = pd.read_csv(DIABETES_TYPE_CSV, dtype={"Patient": str})
    for model, shots in MODELS.items():
        for shot, metrics_dir in shots.items():
            print(f"[{model}/{shot}] reading from {metrics_dir.relative_to(REPO)}")
            agg = aggregate_shot(metrics_dir)
            merged = merge_with_diabetes_type(agg, dtype_df)
            out_dir = REPO / "cohort_analysis" / shot
            out_path = out_dir / f"{model}_{shot}shot_30min_rmse_by_patient.csv"
            merged.to_csv(out_path, index=False)
            print(f"  wrote {len(merged)} rows -> {out_path.relative_to(REPO)}")

    for shot, shot_root in TIMELLM_SHOTS.items():
        print(f"[timellm/{shot}] reading from {shot_root.relative_to(REPO)}")
        agg = aggregate_timellm_shot(shot_root)
        merged = merge_with_diabetes_type(agg, dtype_df)
        out_dir = REPO / "cohort_analysis" / shot
        out_path = out_dir / f"timellm_{shot}shot_30min_rmse_by_patient.csv"
        merged.to_csv(out_path, index=False)
        print(f"  wrote {len(merged)} rows -> {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
