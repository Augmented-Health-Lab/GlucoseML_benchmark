import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error


# =====================================================
# DEVICE
# =====================================================

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


# =====================================================
# PREPROCESS
# =====================================================

def prepare_df_from_csv(csv_path):
    patient_name = csv_path.stem
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["item_id"] = patient_name
    return df[["item_id", "timestamp", "bg", "insulin", "carbs"]], patient_name


def split_into_sequences(df, gap_threshold_hours=1):
    df = df.copy()
    df["time_diff"] = df["timestamp"].diff()
    gap_threshold = pd.Timedelta(hours=gap_threshold_hours)

    df["new_sequence"] = (df["time_diff"] > gap_threshold) | df["time_diff"].isna()
    df["sequence_id"] = df["new_sequence"].cumsum()

    sequences = []
    for _, g in df.groupby("sequence_id"):
        g = g.drop(columns=["time_diff", "new_sequence", "sequence_id"])
        sequences.append(g.reset_index(drop=True))

    return sequences


def generate_windows_from_sequence(seq_df, context_length, prediction_length, stride):
    windows = []
    max_start = len(seq_df) - context_length - prediction_length
    if max_start < 0:
        return windows

    bg = seq_df["bg"].astype("float32").values
    insulin = seq_df["insulin"].astype("float32").values
    carbs = seq_df["carbs"].astype("float32").values

    for i in range(0, max_start + 1, stride):
        end = i + context_length + prediction_length
        windows.append({
            "target": bg[i:end],
            "past_covariates": {
                "insulin": insulin[i:end],
                "carbs": carbs[i:end],
            },
            "future_covariates": {},
        })

    return windows


# =====================================================
# BUILD FEW-SHOT TRAIN INPUTS (local CSVs)
# =====================================================

def build_fit_inputs_from_csvs(
    data_dir,
    context_length,
    prediction_length,
    stride,
    gap_threshold_hours,
    min_sequence_length,
):
    inputs = []

    for csv_path in sorted(Path(data_dir).glob("*.csv")):
        df, patient_name = prepare_df_from_csv(csv_path)
        sequences = split_into_sequences(df, gap_threshold_hours)

        for seq in sequences:
            if len(seq) < min_sequence_length:
                continue

            windows = generate_windows_from_sequence(
                seq,
                context_length=context_length,
                prediction_length=prediction_length,
                stride=stride,
            )
            inputs.extend(windows)

    return inputs


# =====================================================
# EVALUATION (local CSVs)
# =====================================================

def evaluate_csv_split(
    data_dir,
    pipeline,
    context_lengths,
    prediction_length,
    step_size,
    gap_threshold_hours,
):
    horizon_steps = {
        "15min": 3,
        "30min": 6,
        "60min": 12,
        "90min": 18,
    }

    records = []

    for context_length in context_lengths:
        print("\n" + "=" * 70)
        print(f"Context length = {context_length}")
        print("=" * 70)

        for csv_path in sorted(Path(data_dir).glob("*.csv")):
            df, patient_name = prepare_df_from_csv(csv_path)
            sequences = split_into_sequences(df, gap_threshold_hours)

            preds_all, gts_all = [], []

            for seq in sequences:
                if len(seq) < context_length + prediction_length:
                    continue

                max_start = len(seq) - context_length - prediction_length

                for start in range(0, max_start + 1, step_size):
                    context = seq.iloc[start:start + context_length].copy()
                    future = seq.iloc[start + context_length:
                                      start + context_length + prediction_length]

                    try:
                        pred_df = pipeline.predict_df(
                            context,
                            future_df=None,
                            prediction_length=prediction_length,
                            quantile_levels=[0.5],
                            target="bg",
                        )

                        mask = pred_df["target_name"].astype(str) == "bg"
                        preds = pred_df.loc[mask, "predictions"].values.astype(float)

                        if len(preds) == 0:
                            continue

                        preds_all.append(preds)
                        gts_all.append(future["bg"].values)

                    except Exception as e:
                        print(f"    Error: {patient_name} window {start}: {e}")
                        continue

            if len(preds_all) == 0:
                print(f"  {patient_name}: no valid windows")
                continue

            preds_all = np.array(preds_all)
            gts_all = np.array(gts_all)

            print(f"  {patient_name}: {len(preds_all)} windows")

            for name, h in horizon_steps.items():
                if h > prediction_length:
                    continue

                rmse = np.sqrt(
                    mean_squared_error(gts_all[:, h - 1], preds_all[:, h - 1])
                )
                mae = mean_absolute_error(
                    gts_all[:, h - 1], preds_all[:, h - 1]
                )

                records.append({
                    "Patient": patient_name,
                    "Context_Length": context_length,
                    "Horizon": name,
                    "RMSE": rmse,
                    "MAE": mae,
                    "N_Windows": len(preds_all),
                })

    detailed_df = pd.DataFrame(records)

    if len(detailed_df) == 0:
        return pd.DataFrame(), detailed_df

    summary_df = (
        detailed_df
        .groupby(["Context_Length", "Horizon"])
        .agg(
            RMSE_Mean=("RMSE", "mean"),
            RMSE_Std=("RMSE", "std"),
            MAE_Mean=("MAE", "mean"),
            MAE_Std=("MAE", "std"),
            N_Patients=("Patient", "nunique"),
        )
        .reset_index()
    )

    return summary_df, detailed_df


# =====================================================
# MAIN
# =====================================================

def main(args):
    base_dir = Path(__file__).resolve().parent.parent / "multivariate_exploration"

    datasets = {
        "Bris": {
            "train": base_dir / "Bris" / "clean_train",
            "test": base_dir / "Bris" / "clean_test",
        },
        "HUPA-UOM": {
            "train": base_dir / "HUPA-UOM" / "clean_train",
            "test": base_dir / "HUPA-UOM" / "clean_test",
        },
    }

    print("Loading Chronos-2 on", device)

    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=device,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )

    for dataset_name, paths in datasets.items():
        print("\n" + "=" * 80)
        print(f"DATASET: {dataset_name}")
        print("=" * 80)

        print("\nBuilding few-shot training windows...")

        train_inputs = build_fit_inputs_from_csvs(
            data_dir=paths["train"],
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            stride=args.train_stride,
            gap_threshold_hours=args.gap_hours,
            min_sequence_length=args.min_seq_len,
        )

        print(f"Total training windows: {len(train_inputs)}")

        print("\nFew-shot LoRA fine-tuning...")

        finetuned_pipeline = pipeline.fit(
            inputs=train_inputs,
            prediction_length=args.prediction_length,
            finetune_mode="lora",
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            min_past=args.min_past,
        )

        model_dir = Path(f"./chronos2_lora_fewshot_covariate_{dataset_name}")
        finetuned_pipeline.save_pretrained(model_dir)
        print(f"Saved model to {model_dir}")

        print("\nEvaluating few-shot model...")

        summary_df, detailed_df = evaluate_csv_split(
            data_dir=paths["test"],
            pipeline=finetuned_pipeline,
            context_lengths=args.context_lengths,
            prediction_length=args.prediction_length,
            step_size=args.step_size,
            gap_threshold_hours=args.gap_hours,
        )

        results_dir = Path(f"./results_covariate/{dataset_name}/fewshot")
        results_dir.mkdir(parents=True, exist_ok=True)

        summary_df.to_csv(results_dir / "summary.csv", index=False)
        detailed_df.to_csv(results_dir / "detailed.csv", index=False)

        print(f"Saved results to {results_dir}")

    print("\nALL DATASETS COMPLETE")


# =====================================================
# CLI
# =====================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--context_length", type=int, default=144)
    parser.add_argument("--train_stride", type=int, default=240)
    parser.add_argument("--prediction_length", type=int, default=18)
    parser.add_argument("--context_lengths", type=int, nargs="+", default=[144])
    parser.add_argument("--step_size", type=int, default=1)
    parser.add_argument("--gap_hours", type=int, default=1)
    parser.add_argument("--min_seq_len", type=int, default=200)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_steps", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--min_past", type=int, default=144)

    args = parser.parse_args()

    main(args)
