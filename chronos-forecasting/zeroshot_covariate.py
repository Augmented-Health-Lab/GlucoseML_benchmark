import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from chronos import BaseChronosPipeline, Chronos2Pipeline

# Device selection
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-2", device_map=device
)


def split_into_sequences(df, gap_threshold_hours=1):
    """Split data into continuous sequences based on time gaps."""
    df = df.copy()
    df["time_diff"] = df["timestamp"].diff()
    gap_threshold = pd.Timedelta(hours=gap_threshold_hours)

    df["new_sequence"] = (df["time_diff"] > gap_threshold) | (df["time_diff"].isna())
    df["sequence_id"] = df["new_sequence"].cumsum()

    sequences = []
    for _, group in df.groupby("sequence_id"):
        group = group.drop(columns=["time_diff", "new_sequence", "sequence_id"]).reset_index(drop=True)
        sequences.append(group)

    return sequences


def rolling_window_forecast(sequences, pipeline, context_length, prediction_length=18, step_size=1, verbose=False):
    """
    Perform rolling window forecasting with covariates across all sequences.
    Insulin and carbs are passed as past-only covariates in the context window.
    """
    all_predictions = []
    all_ground_truth = []
    all_timestamps = []
    all_sequence_ids = []

    for seq_idx, seq_df in enumerate(sequences):
        seq_length = len(seq_df)
        max_start_idx = seq_length - context_length - prediction_length

        if max_start_idx < 0:
            if verbose:
                print(f"    Seq {seq_idx+1}: Too short ({seq_length} points), skipping...")
            continue

        for start_idx in range(0, max_start_idx + 1, step_size):
            end_idx = start_idx + context_length
            pred_end_idx = end_idx + prediction_length

            # Context window includes target + covariates
            context_window = seq_df.iloc[start_idx:end_idx][["item_id", "timestamp", "bg", "insulin", "carbs"]].copy()
            ground_truth_window = seq_df.iloc[end_idx:pred_end_idx].copy()

            try:
                # predict_df: columns in df but NOT in future_df are past-only covariates
                # We pass future_df=None so insulin and carbs are treated as past-only covariates
                pred_df = pipeline.predict_df(
                    context_window,
                    future_df=None,
                    prediction_length=prediction_length,
                    quantile_levels=[0.1, 0.5, 0.9],
                    target="bg",
                )

                # Use str cast for robust Arrow/pandas string comparison
                mask = pred_df["target_name"].astype(str) == "bg"
                predictions = pred_df.loc[mask, "predictions"].values.astype(float)
                ground_truth = ground_truth_window["bg"].values

                if len(predictions) == 0:
                    if verbose:
                        print(f"    Empty predictions at window {start_idx} in seq {seq_idx+1}")
                        print(f"    target_name values: {pred_df['target_name'].unique()}")
                    continue

                timestamps = ground_truth_window["timestamp"].values

                all_predictions.append(predictions)
                all_ground_truth.append(ground_truth)
                all_timestamps.append(timestamps)
                all_sequence_ids.append(seq_idx)

            except Exception as e:
                print(f"    Error at window {start_idx} in seq {seq_idx+1}: {e}")
                continue

    return all_predictions, all_ground_truth, all_timestamps, all_sequence_ids


def calculate_metrics(all_predictions, all_ground_truth, horizon_steps):
    """Calculate RMSE and MAE for different prediction horizons."""
    results = {}

    for horizon_name, horizon_step in horizon_steps.items():
        rmse_values = []
        mae_values = []

        for pred, gt in zip(all_predictions, all_ground_truth):
            if len(pred) >= horizon_step and len(gt) >= horizon_step:
                pred_value = pred[horizon_step - 1]
                gt_value = gt[horizon_step - 1]

                squared_error = (pred_value - gt_value) ** 2
                absolute_error = abs(pred_value - gt_value)

                rmse_values.append(squared_error)
                mae_values.append(absolute_error)

        if len(rmse_values) > 0:
            avg_rmse = np.sqrt(np.mean(rmse_values))
            avg_mae = np.mean(mae_values)
            results[horizon_name] = {
                "RMSE": avg_rmse,
                "MAE": avg_mae,
                "n_samples": len(rmse_values),
            }
        else:
            results[horizon_name] = {"RMSE": np.nan, "MAE": np.nan, "n_samples": 0}

    return results


def evaluate_single_patient(df, patient_name, pipeline, context_lengths, prediction_length=18, step_size=1):
    horizon_steps = {"15min": 3, "30min": 6, "60min": 12, "90min": 18}

    try:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["item_id"] = patient_name

        sequences = split_into_sequences(df, gap_threshold_hours=1)

        print(f"  Patient: {patient_name}")
        print(f"    Total points: {len(df)}, Sequences: {len(sequences)}")

        patient_results = {}

        for context_length in context_lengths:
            all_predictions, all_ground_truth, all_timestamps, all_sequence_ids = rolling_window_forecast(
                sequences,
                pipeline,
                context_length,
                prediction_length,
                step_size,
                verbose=True,
            )

            if len(all_predictions) == 0:
                print(f"    Context {context_length}: NO valid predictions!")
                patient_results[context_length] = None
                continue

            results = calculate_metrics(all_predictions, all_ground_truth, horizon_steps)
            patient_results[context_length] = results

            print(f"    Context {context_length}: {len(all_predictions)} windows")

        return patient_results

    except Exception as e:
        print(f"  Error processing {patient_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def aggregate_results_across_patients(all_patient_results, context_lengths):
    """Aggregate results across all patients."""
    horizons = ["15min", "30min", "60min", "90min"]
    summary_data = []

    for context_length in context_lengths:
        for horizon in horizons:
            rmse_values = []
            mae_values = []

            for patient_name, patient_results in all_patient_results.items():
                if patient_results is not None and context_length in patient_results:
                    if patient_results[context_length] is not None:
                        if horizon in patient_results[context_length]:
                            r = patient_results[context_length][horizon]
                            if not np.isnan(r["RMSE"]):
                                rmse_values.append(r["RMSE"])
                                mae_values.append(r["MAE"])

            if len(rmse_values) > 0:
                avg_rmse = np.mean(rmse_values)
                std_rmse = np.std(rmse_values)
                avg_mae = np.mean(mae_values)
                std_mae = np.std(mae_values)
                n_patients = len(rmse_values)
            else:
                avg_rmse = std_rmse = avg_mae = std_mae = np.nan
                n_patients = 0

            summary_data.append({
                "Context_Length": context_length,
                "Context_Hours": context_length * 5 / 60,
                "Horizon": horizon,
                "RMSE_Mean": avg_rmse,
                "RMSE_Std": std_rmse,
                "MAE_Mean": avg_mae,
                "MAE_Std": std_mae,
                "N_Patients": n_patients,
            })

    return pd.DataFrame(summary_data)


def save_detailed_results(all_patient_results, context_lengths, output_dir):
    """Save detailed per-patient results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    horizons = ["15min", "30min", "60min", "90min"]

    for context_length in context_lengths:
        patient_data = []

        for patient_name, patient_results in all_patient_results.items():
            if patient_results is not None and context_length in patient_results:
                if patient_results[context_length] is not None:
                    for horizon in horizons:
                        if horizon in patient_results[context_length]:
                            r = patient_results[context_length][horizon]
                            patient_data.append({
                                "Patient": patient_name,
                                "Context_Length": context_length,
                                "Horizon": horizon,
                                "RMSE": r["RMSE"],
                                "MAE": r["MAE"],
                                "N_Samples": r["n_samples"],
                            })

        if patient_data:
            detail_df = pd.DataFrame(patient_data)
            output_file = f"{output_dir}/context_{context_length}_detailed.csv"
            detail_df.to_csv(output_file, index=False)
            print(f"Saved: {output_file}")


def main(args):
    base_dir = Path(__file__).resolve().parent.parent / "multivariate_exploration"

    datasets = {
        "Bris": base_dir / "Bris" / "clean_test",
        "HUPA-UOM": base_dir / "HUPA-UOM" / "clean_test",
    }

    context_lengths = [144]

    print(f"Running covariate zeroshot with step_size = {args.step_size}")
    print(f"Prediction length = {args.prediction_length}")

    # === Quick sanity check on first file of first dataset ===
    first_dir = list(datasets.values())[0]
    first_csv = sorted(first_dir.glob("*.csv"))[0]
    test_df = pd.read_csv(first_csv)
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
    test_df["item_id"] = "test"
    print(f"Sanity check on {first_csv.name}:")
    print(f"  Missing bg: {test_df['bg'].isna().sum()}, insulin: {test_df['insulin'].isna().sum()}, carbs: {test_df['carbs'].isna().sum()}")
    ctx = test_df.iloc[:144][["item_id", "timestamp", "bg", "insulin", "carbs"]].copy()
    try:
        test_pred = pipeline.predict_df(ctx, future_df=None, prediction_length=18, quantile_levels=[0.1, 0.5, 0.9], target="bg")
        print(f"  predict_df returned shape: {test_pred.shape}")
        print(f"  target_name values: {test_pred['target_name'].unique()}")
        print(f"  target_name dtype: {test_pred['target_name'].dtype}")
        mask = test_pred["target_name"].astype(str) == "bg"
        preds = test_pred.loc[mask, "predictions"].values
        print(f"  Filtered predictions len: {len(preds)}, values: {preds}")
    except Exception as e:
        print(f"  SANITY CHECK FAILED: {e}")
        import traceback
        traceback.print_exc()
    print()

    for dataset_name, data_dir in datasets.items():
        print("\n" + "=" * 80)
        print(f"PROCESSING DATASET: {dataset_name}")
        print("=" * 80)

        csv_files = sorted(data_dir.glob("*.csv"))
        print(f"Found {len(csv_files)} patients\n")

        all_patient_results = {}

        for i, csv_path in enumerate(csv_files, 1):
            patient_name = csv_path.stem
            print(f"[{i}/{len(csv_files)}] {patient_name}")

            df = pd.read_csv(csv_path)

            patient_results = evaluate_single_patient(
                df=df,
                patient_name=patient_name,
                pipeline=pipeline,
                context_lengths=context_lengths,
                prediction_length=args.prediction_length,
                step_size=args.step_size,
            )

            all_patient_results[patient_name] = patient_results

        summary_df = aggregate_results_across_patients(all_patient_results, context_lengths)

        out_dir = Path(f"./results_covariate/{dataset_name}/step_{args.step_size}")
        out_dir.mkdir(parents=True, exist_ok=True)

        summary_df.to_csv(out_dir / "summary.csv", index=False)

        save_detailed_results(
            all_patient_results,
            context_lengths,
            output_dir=str(out_dir / "patient_results"),
        )

        print(f"Saved results to {out_dir}")

    print("\nALL DATASETS COMPLETE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_size", type=int, default=1)
    parser.add_argument("--prediction_length", type=int, default=18)
    args = parser.parse_args()
    main(args)
