import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
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

def prepare_df_from_subject(subject, patient_name):
    df = pd.DataFrame({
        "timestamp": subject["timestamp"],
        "target": subject["BGvalue"]
    })

    df["item_id"] = patient_name
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df[["item_id", "timestamp", "target"]]


def split_into_sequences(df, gap_threshold_hours=1):

    df["time_diff"] = df["timestamp"].diff()
    gap_threshold = pd.Timedelta(hours=gap_threshold_hours)

    df["new_sequence"] = (df["time_diff"] > gap_threshold) | df["time_diff"].isna()
    df["sequence_id"] = df["new_sequence"].cumsum()

    sequences = []
    for _, g in df.groupby("sequence_id"):
        g = g.drop(columns=["time_diff", "new_sequence", "sequence_id"])
        sequences.append(g.reset_index(drop=True))

    return sequences


def sequence_to_fit_input(seq_df):
    return {
        "target": seq_df["target"].astype("float32").values,
        "past_covariates": {},
        "future_covariates": {},
    }

def generate_windows_from_sequence(
    seq_df,
    context_length,
    prediction_length,
    stride,
):
    windows = []

    max_start = len(seq_df) - context_length - prediction_length
    if max_start < 0:
        return windows

    target = seq_df["target"].values

    for i in range(0, max_start + 1, stride):
        windows.append({
            "target": target[i : i + context_length + prediction_length],
            "past_covariates": {},
            "future_covariates": {},
        })

    return windows



# =====================================================
# BUILD TRAINING INPUTS (HF)
# =====================================================

def build_fit_inputs_from_hf(
    ds,
    split,
    context_length,
    prediction_length,
    stride,
    gap_threshold_hours,
    min_sequence_length,
):

    inputs = []

    for subject in ds[split]:

        patient_name = subject["subject_id"]
        df = prepare_df_from_subject(subject, patient_name)

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
# EVALUATION (HF)
# =====================================================

def evaluate_hf_split(
    ds,
    split,
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

        for subject in ds[split]:

            patient_name = subject["subject_id"]
            df = prepare_df_from_subject(subject, patient_name)
            sequences = split_into_sequences(df, gap_threshold_hours)

            preds_all = []
            gts_all = []

            for seq in sequences:
                if len(seq) < context_length + prediction_length:
                    continue

                max_start = len(seq) - context_length - prediction_length

                for start in range(0, max_start + 1, step_size):
                    context = seq.iloc[start : start + context_length]
                    future = seq.iloc[start + context_length :
                                     start + context_length + prediction_length]

                    try:
                        pred_df = pipeline.predict_df(
                            context,
                            prediction_length=prediction_length,
                            quantile_levels=[0.5],
                        )

                        preds_all.append(pred_df["predictions"].values)
                        gts_all.append(future["target"].values)

                    except Exception:
                        continue

            if len(preds_all) == 0:
                continue

            preds_all = np.array(preds_all)
            gts_all = np.array(gts_all)

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

    ds = load_dataset(args.dataset)

    print("Loading Chronos-2 on", device)

    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=device,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )

    print("\nBuilding training windows...")

    train_inputs = build_fit_inputs_from_hf(
        ds,
        split=args.train_split,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        stride=args.train_stride,
        gap_threshold_hours=args.gap_hours,
        min_sequence_length=args.min_seq_len,
    )

    print("Total training sequences:", len(train_inputs))

    print("\nFine-tuning (LoRA)...")

    finetuned_pipeline = pipeline.fit(
        inputs=train_inputs,
        prediction_length=args.prediction_length,
        finetune_mode="lora",
        learning_rate=args.lr,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        logging_steps=200,
        min_past=args.min_past,
    )

    finetuned_pipeline.save_pretrained(args.output_model)

    print("\nEvaluating full-shot model...")

    summary_df, detailed_df = evaluate_hf_split(
        ds,
        split=args.test_split,
        pipeline=finetuned_pipeline,
        context_lengths=args.context_lengths,
        prediction_length=args.prediction_length,
        step_size=args.step_size,
        gap_threshold_hours=args.gap_hours,
    )

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    summary_path = results_dir / args.summary_csv
    detailed_path = results_dir / args.detailed_csv

    summary_df.to_csv(summary_path, index=False)
    detailed_df.to_csv(detailed_path, index=False)

    print(f"Saved summary to: {summary_path}")
    print(f"Saved detailed to: {detailed_path}")


    print("\nSaved results.")


# =====================================================
# CLI
# =====================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="byluuu/gluco-tsfm-benchmark")
    parser.add_argument("--context_length", type=int, default=144)
    parser.add_argument("--train_stride", type=int, default=12)


    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--test_split", type=str, default="test")

    parser.add_argument("--prediction_length", type=int, default=18)
    parser.add_argument("--context_lengths", type=int, nargs="+", default=[144])

    parser.add_argument("--step_size", type=int, default=1)
    parser.add_argument("--gap_hours", type=int, default=1)

    parser.add_argument("--min_seq_len", type=int, default=200)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_steps", type=int, default=16000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--min_past", type=int, default=144)

    parser.add_argument("--output_model", type=str,
                        default="./chronos2_lora_fullshot")

    parser.add_argument("--summary_csv", type=str,
                        default="./fullshot_summary.csv")

    parser.add_argument("--detailed_csv", type=str,
                        default="./fullshot_detailed.csv")
    

    args = parser.parse_args()

    main(args)
