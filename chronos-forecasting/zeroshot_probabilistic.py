import os
import argparse
from datasets import load_dataset
import torch
# Use only 1 GPU if available
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chronos import BaseChronosPipeline, Chronos2Pipeline
import os
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the Chronos-2 pipeline
# GPU recommended for faster inference, but CPU is also supported using device_map="cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-2",
    device_map=device
)
pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map=device)

# Quantile levels to request from the model.
# Column names in pred_df will be the string versions: "0.1", "0.25", "0.5", "0.75", "0.9"
QUANTILE_LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]


def load_and_prepare_data_from_hf(subject):
    """Load and prepare glucose data from HuggingFace row."""

    df = pd.DataFrame({
        "timestamp": subject["timestamp"],
        "target": subject["BGvalue"]
    })

    df["item_id"] = subject["subject_id"]

    # Convert numeric timestamp back to datetime (if needed)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    df = df.sort_values("timestamp").reset_index(drop=True)

    df = df[["item_id", "timestamp", "target"]]

    return df

def split_into_sequences(df, gap_threshold_hours=1):
    """Split data into continuous sequences based on time gaps."""
    df['time_diff'] = df['timestamp'].diff()
    gap_threshold = pd.Timedelta(hours=gap_threshold_hours)

    df['new_sequence'] = (df['time_diff'] > gap_threshold) | (df['time_diff'].isna())
    df['sequence_id'] = df['new_sequence'].cumsum()

    sequences = []
    for seq_id, group in df.groupby('sequence_id'):
        group = group.drop(columns=['time_diff', 'new_sequence', 'sequence_id']).reset_index(drop=True)
        sequences.append(group)

    return sequences

def rolling_window_forecast(sequences, pipeline, context_length, prediction_length=18, step_size=1, verbose=False):
    """
    Perform rolling window probabilistic forecasting across all sequences.

    Parameters:
    -----------
    sequences : list of DataFrames
        List of continuous data sequences
    pipeline : Chronos2Pipeline
        The forecasting pipeline
    context_length : int
        Number of historical time steps to use as context
    prediction_length : int
        Number of future time steps to predict
    step_size : int
        Step size for rolling window
    verbose : bool
        Whether to print progress

    Returns:
    --------
    all_predictions : list
        List of mean prediction arrays, shape (prediction_length,) each
    all_quantiles : dict of list
        Keys are quantile level strings ("0.1", "0.25", etc.);
        values are lists of arrays, shape (prediction_length,) each
    all_ground_truth : list
        List of ground truth arrays
    all_timestamps : list
        List of timestamp arrays
    all_sequence_ids : list
        List of sequence IDs for each prediction
    """
    all_predictions  = []
    all_ground_truth = []
    all_timestamps   = []
    all_sequence_ids = []

    # One list per quantile level
    all_quantiles = {str(q): [] for q in QUANTILE_LEVELS}

    total_windows = 0

    for seq_idx, seq_df in enumerate(sequences):
        seq_length = len(seq_df)
        max_start_idx = seq_length - context_length - prediction_length

        if max_start_idx < 0:
            if verbose:
                print(f"    Seq {seq_idx+1}: Too short ({seq_length} points), skipping...")
            continue

        num_windows = max_start_idx + 1
        total_windows += num_windows

        for start_idx in range(0, max_start_idx + 1, step_size):
            end_idx = start_idx + context_length
            pred_end_idx = end_idx + prediction_length

            context_window      = seq_df.iloc[start_idx:end_idx].copy()
            ground_truth_window = seq_df.iloc[end_idx:pred_end_idx].copy()

            try:
                pred_df = pipeline.predict_df(
                    context_window,
                    prediction_length=prediction_length,
                    quantile_levels=QUANTILE_LEVELS
                )

                # pred_df is long-format: one row per future timestep.
                # Filter to the target variate and sort by timestamp to be safe.
                rows = pred_df[pred_df['target_name'] == 'target'].sort_values('timestamp')

                # Point forecast (mean)
                predictions  = rows['predictions'].values           # (prediction_length,)
                ground_truth = ground_truth_window['target'].values
                timestamps   = ground_truth_window['timestamp'].values

                all_predictions.append(predictions)
                all_ground_truth.append(ground_truth)
                all_timestamps.append(timestamps)
                all_sequence_ids.append(seq_idx)

                # Quantile forecasts — column name is str(quantile_level)
                for q in QUANTILE_LEVELS:
                    col = str(q)
                    all_quantiles[col].append(rows[col].values)   # (prediction_length,)

            except Exception as e:
                if verbose:
                    print(f"    Error at window {start_idx} in seq {seq_idx+1}: {e}")
                continue

    return all_predictions, all_quantiles, all_ground_truth, all_timestamps, all_sequence_ids


def calculate_metrics(all_predictions, all_quantiles, all_ground_truth, horizon_steps):
    """
    Calculate point metrics (RMSE, MAE) and probabilistic metrics
    (prediction interval coverage and width) for different horizons.

    Probabilistic metrics use the 10th–90th percentile as the 80% PI
    and the 25th–75th percentile as the 50% PI.
    """
    results = {}

    for horizon_name, horizon_step in horizon_steps.items():
        rmse_values  = []
        mae_values   = []

        # 80% PI  (Q10 – Q90)
        covered_80   = []
        width_80     = []

        # 50% PI  (Q25 – Q75)
        covered_50   = []
        width_50     = []

        for idx, (pred, gt) in enumerate(zip(all_predictions, all_ground_truth)):
            if len(pred) >= horizon_step and len(gt) >= horizon_step:
                step = horizon_step - 1          # 0-indexed
                pred_val = pred[step]
                gt_val   = gt[step]

                rmse_values.append((pred_val - gt_val) ** 2)
                mae_values.append(abs(pred_val - gt_val))

                # 80% PI
                if '0.1' in all_quantiles and '0.9' in all_quantiles:
                    q10 = all_quantiles['0.1'][idx][step]
                    q90 = all_quantiles['0.9'][idx][step]
                    covered_80.append(int(q10 <= gt_val <= q90))
                    width_80.append(q90 - q10)

                # 50% PI
                if '0.25' in all_quantiles and '0.75' in all_quantiles:
                    q25 = all_quantiles['0.25'][idx][step]
                    q75 = all_quantiles['0.75'][idx][step]
                    covered_50.append(int(q25 <= gt_val <= q75))
                    width_50.append(q75 - q25)

        if len(rmse_values) > 0:
            results[horizon_name] = {
                'RMSE':            float(np.sqrt(np.mean(rmse_values))),
                'MAE':             float(np.mean(mae_values)),
                'n_samples':       len(rmse_values),
                'PI80_coverage':   float(np.mean(covered_80))  if covered_80  else np.nan,
                'PI80_width':      float(np.mean(width_80))    if width_80    else np.nan,
                'PI50_coverage':   float(np.mean(covered_50))  if covered_50  else np.nan,
                'PI50_width':      float(np.mean(width_50))    if width_50    else np.nan,
            }
        else:
            results[horizon_name] = {
                'RMSE': np.nan, 'MAE': np.nan, 'n_samples': 0,
                'PI80_coverage': np.nan, 'PI80_width': np.nan,
                'PI50_coverage': np.nan, 'PI50_width': np.nan,
            }

    return results


def evaluate_single_patient(df, patient_name, pipeline, context_lengths,
                            prediction_length=18, step_size=1):

    horizon_steps = {
        '15min': 3,
        '30min': 6,
        '60min': 12,
        '90min': 18
    }

    try:
        df = df.copy()
        df = df.rename(columns={"BGvalue": "target"})
        df["item_id"] = patient_name
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = df[["item_id", "timestamp", "target"]]

        sequences = split_into_sequences(df, gap_threshold_hours=1)

        print(f"  Patient: {patient_name}")
        print(f"    Total points: {len(df)}, Sequences: {len(sequences)}")

        patient_results = {}

        for context_length in context_lengths:

            all_predictions, all_quantiles, all_ground_truth, all_timestamps, all_sequence_ids = \
                rolling_window_forecast(
                    sequences,
                    pipeline,
                    context_length,
                    prediction_length,
                    step_size,
                    verbose=False
                )

            if len(all_predictions) == 0:
                patient_results[context_length] = None
                continue

            results = calculate_metrics(
                all_predictions,
                all_quantiles,
                all_ground_truth,
                horizon_steps
            )

            patient_results[context_length] = results

            print(f"    Context {context_length}: {len(all_predictions)} windows")

        return patient_results

    except Exception as e:
        print(f"  Error processing {patient_name}: {e}")
        return None


def aggregate_results_across_patients(all_patient_results, context_lengths):
    """
    Aggregate results across all patients.

    Returns a DataFrame with average point and probabilistic metrics per
    (context_length, horizon).
    """
    horizons = ['15min', '30min', '60min', '90min']

    summary_data = []

    for context_length in context_lengths:
        for horizon in horizons:
            rmse_values    = []
            mae_values     = []
            pi80_coverages = []
            pi80_widths    = []
            pi50_coverages = []
            pi50_widths    = []

            for patient_name, patient_results in all_patient_results.items():
                if patient_results is None:
                    continue
                if context_length not in patient_results:
                    continue
                r = patient_results[context_length]
                if r is None or horizon not in r:
                    continue
                h = r[horizon]
                if np.isnan(h['RMSE']):
                    continue

                rmse_values.append(h['RMSE'])
                mae_values.append(h['MAE'])
                if not np.isnan(h['PI80_coverage']):
                    pi80_coverages.append(h['PI80_coverage'])
                    pi80_widths.append(h['PI80_width'])
                if not np.isnan(h['PI50_coverage']):
                    pi50_coverages.append(h['PI50_coverage'])
                    pi50_widths.append(h['PI50_width'])

            def _mean(lst):  return float(np.mean(lst))  if lst else np.nan
            def _std(lst):   return float(np.std(lst))   if lst else np.nan

            summary_data.append({
                'Context_Length':    context_length,
                'Context_Hours':     context_length * 5 / 60,
                'Horizon':           horizon,
                'RMSE_Mean':         _mean(rmse_values),
                'RMSE_Std':          _std(rmse_values),
                'MAE_Mean':          _mean(mae_values),
                'MAE_Std':           _std(mae_values),
                'N_Patients':        len(rmse_values),
                'PI80_Coverage_Mean': _mean(pi80_coverages),
                'PI80_Width_Mean':   _mean(pi80_widths),
                'PI50_Coverage_Mean': _mean(pi50_coverages),
                'PI50_Width_Mean':   _mean(pi50_widths),
            })

    return pd.DataFrame(summary_data)


def save_detailed_results(all_patient_results, context_lengths, output_dir='./results'):
    """Save detailed per-patient results (point + probabilistic) to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    horizons = ['15min', '30min', '60min', '90min']

    for context_length in context_lengths:
        patient_data = []

        for patient_name, patient_results in all_patient_results.items():
            if patient_results is None:
                continue
            if context_length not in patient_results or patient_results[context_length] is None:
                continue
            for horizon in horizons:
                if horizon not in patient_results[context_length]:
                    continue
                r = patient_results[context_length][horizon]
                patient_data.append({
                    'Patient':          patient_name,
                    'Context_Length':   context_length,
                    'Horizon':          horizon,
                    'RMSE':             r['RMSE'],
                    'MAE':              r['MAE'],
                    'N_Samples':        r['n_samples'],
                    'PI80_Coverage':    r['PI80_coverage'],
                    'PI80_Width':       r['PI80_width'],
                    'PI50_Coverage':    r['PI50_coverage'],
                    'PI50_Width':       r['PI50_width'],
                })

        if patient_data:
            detail_df = pd.DataFrame(patient_data)
            output_file = f"{output_dir}/context_{context_length}_detailed.csv"
            detail_df.to_csv(output_file, index=False)
            print(f"Saved: {output_file}")


def main(args):

    ds = load_dataset("byluuu/gluco-tsfm-benchmark")   # change to yours
    split = args.split

    context_lengths = [144]

    print(f"\nRunning with step_size = {args.step_size}")
    print(f"Split = {split}")
    print(f"Quantile levels = {QUANTILE_LEVELS}")

    dataset_names = sorted(set(ds[split]["dataset"]))

    for dataset_name in dataset_names:

        print("\n" + "=" * 80)
        print(f"PROCESSING DATASET: {dataset_name}")
        print("=" * 80)

        subset = ds[split].filter(
            lambda x: x["dataset"] == dataset_name
        )

        print(f"Found {len(subset)} patients\n")

        all_patient_results = {}

        for i, subject in enumerate(subset, 1):

            patient_name = subject["subject_id"]
            print(f"[{i}/{len(subset)}] {patient_name}")

            df = pd.DataFrame({
                "timestamp": subject["timestamp"],
                "BGvalue":   subject["BGvalue"]
            })

            patient_results = evaluate_single_patient(
                df=df,
                patient_name=patient_name,
                pipeline=pipeline,
                context_lengths=context_lengths,
                prediction_length=args.prediction_length,
                step_size=args.step_size
            )

            all_patient_results[patient_name] = patient_results

        summary_df = aggregate_results_across_patients(
            all_patient_results,
            context_lengths
        )

        out_dir = Path(args.output_dir) / dataset_name / f"step_{args.step_size}"
        out_dir.mkdir(parents=True, exist_ok=True)

        summary_df.to_csv(out_dir / "summary.csv", index=False)

        save_detailed_results(
            all_patient_results,
            context_lengths,
            output_dir=out_dir / "patient_results"
        )

        print(f"Saved results to {out_dir}")

    print("\nALL DATASETS COMPLETE")


# ============================================================
# ARGUMENTS
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--step_size",         type=int, default=1)
    parser.add_argument("--prediction_length", type=int, default=18)
    parser.add_argument("--split",             type=str, default="test")
    parser.add_argument("--output_dir",        type=str, default="./results",
                        help="Root folder for all saved results.")

    args = parser.parse_args()

    main(args)
