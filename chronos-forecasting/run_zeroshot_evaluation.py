"""
Zero-shot evaluation script for Chronos-2 on glucose monitoring datasets.
Supports multiple datasets and context lengths with configurable parameters.
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch

# Set environment variable for GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from chronos import BaseChronosPipeline, Chronos2Pipeline


def load_and_prepare_data(file_path):
    """Load and prepare the glucose monitoring data."""
    context_df = pd.read_csv(file_path)
    df = context_df.copy()
    df = df.rename(columns={'BGvalue': 'target'})
    df['item_id'] = 'patient_1'
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df[['item_id', 'timestamp', 'target']]
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
    Perform rolling window forecasting across all sequences.

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
        List of prediction arrays
    all_ground_truth : list
        List of ground truth arrays
    all_timestamps : list
        List of timestamp arrays
    all_sequence_ids : list
        List of sequence IDs for each prediction
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

            context_window = seq_df.iloc[start_idx:end_idx].copy()
            ground_truth_window = seq_df.iloc[end_idx:pred_end_idx].copy()

            try:
                pred_df = pipeline.predict_df(
                    context_window,
                    prediction_length=prediction_length,
                    quantile_levels=[0.1, 0.5, 0.9]
                )

                predictions = pred_df[pred_df['target_name'] == 'target']['predictions'].values
                ground_truth = ground_truth_window['target'].values
                timestamps = ground_truth_window['timestamp'].values

                all_predictions.append(predictions)
                all_ground_truth.append(ground_truth)
                all_timestamps.append(timestamps)
                all_sequence_ids.append(seq_idx)

            except Exception as e:
                if verbose:
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
                'RMSE': avg_rmse,
                'MAE': avg_mae,
                'n_samples': len(rmse_values)
            }
        else:
            results[horizon_name] = {
                'RMSE': np.nan,
                'MAE': np.nan,
                'n_samples': 0
            }

    return results


def evaluate_single_patient(file_path, pipeline, context_lengths, prediction_length=18, step_size=1):
    """
    Evaluate a single patient across multiple context lengths.

    Returns:
    --------
    patient_results : dict
        Dictionary with results for each context length
    """
    horizon_steps = {
        '15min': 3,
        '30min': 6,
        '60min': 12,
        '90min': 18
    }

    patient_name = Path(file_path).stem

    try:
        # Load and prepare data
        df = load_and_prepare_data(file_path)

        # Split into sequences
        sequences = split_into_sequences(df, gap_threshold_hours=1)

        print(f"  Patient: {patient_name}")
        print(f"    Total points: {len(df)}, Sequences: {len(sequences)}")

        patient_results = {}

        # Evaluate for each context length
        for context_length in context_lengths:
            # Perform rolling window forecasting
            all_predictions, all_ground_truth, all_timestamps, all_sequence_ids = rolling_window_forecast(
                sequences, pipeline, context_length, prediction_length, step_size, verbose=False
            )

            if len(all_predictions) == 0:
                print(f"    Context {context_length}: No predictions (insufficient data)")
                patient_results[context_length] = None
                continue

            # Calculate metrics
            results = calculate_metrics(all_predictions, all_ground_truth, horizon_steps)
            patient_results[context_length] = results

            print(f"    Context {context_length}: {len(all_predictions)} predictions")

        return patient_results

    except Exception as e:
        print(f"  Error processing {patient_name}: {e}")
        return None


def aggregate_results_across_patients(all_patient_results, context_lengths):
    """
    Aggregate results across all patients.

    Returns:
    --------
    aggregated_results : DataFrame
        Summary table with average metrics across all patients
    """
    horizons = ['15min', '30min', '60min', '90min']

    summary_data = []

    for context_length in context_lengths:
        for horizon in horizons:
            rmse_values = []
            mae_values = []

            # Collect metrics from all patients for this context length and horizon
            for patient_name, patient_results in all_patient_results.items():
                if patient_results is not None and context_length in patient_results:
                    if patient_results[context_length] is not None:
                        if horizon in patient_results[context_length]:
                            r = patient_results[context_length][horizon]
                            if not np.isnan(r['RMSE']):
                                rmse_values.append(r['RMSE'])
                                mae_values.append(r['MAE'])

            # Calculate average across patients
            if len(rmse_values) > 0:
                avg_rmse = np.mean(rmse_values)
                std_rmse = np.std(rmse_values)
                avg_mae = np.mean(mae_values)
                std_mae = np.std(mae_values)
                n_patients = len(rmse_values)
            else:
                avg_rmse = np.nan
                std_rmse = np.nan
                avg_mae = np.nan
                std_mae = np.nan
                n_patients = 0

            summary_data.append({
                'Context_Length': context_length,
                'Context_Hours': context_length * 5 / 60,
                'Horizon': horizon,
                'RMSE_Mean': avg_rmse,
                'RMSE_Std': std_rmse,
                'MAE_Mean': avg_mae,
                'MAE_Std': std_mae,
                'N_Patients': n_patients
            })

    return pd.DataFrame(summary_data)


def save_detailed_results(all_patient_results, context_lengths, output_dir='./results'):
    """Save detailed per-patient results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    horizons = ['15min', '30min', '60min', '90min']

    for context_length in context_lengths:
        patient_data = []

        for patient_name, patient_results in all_patient_results.items():
            if patient_results is not None and context_length in patient_results:
                if patient_results[context_length] is not None:
                    for horizon in horizons:
                        if horizon in patient_results[context_length]:
                            r = patient_results[context_length][horizon]
                            patient_data.append({
                                'Patient': patient_name,
                                'Context_Length': context_length,
                                'Horizon': horizon,
                                'RMSE': r['RMSE'],
                                'MAE': r['MAE'],
                                'N_Samples': r['n_samples']
                            })

        if patient_data:
            detail_df = pd.DataFrame(patient_data)
            output_file = f"{output_dir}/context_{context_length}_detailed.csv"
            detail_df.to_csv(output_file, index=False)
            print(f"Saved: {output_file}")


def print_summary(summary_df):
    """Print formatted summary of results."""
    print("\n" + "="*90)
    print("SUMMARY: AVERAGE PERFORMANCE ACROSS ALL PATIENTS")
    print("="*90)

    # Print by horizon
    for horizon in ['15min', '30min', '60min', '90min']:
        print(f"\n{horizon.upper()} Prediction Horizon:")
        print("-"*90)
        print(f"{'Context':<10} {'Hours':<10} {'RMSE (Mean±Std)':<25} {'MAE (Mean±Std)':<25} {'N Patients':<15}")
        print("-"*90)

        horizon_data = summary_df[summary_df['Horizon'] == horizon]
        for _, row in horizon_data.iterrows():
            rmse_str = f"{row['RMSE_Mean']:.2f}±{row['RMSE_Std']:.2f}" if not np.isnan(row['RMSE_Mean']) else "N/A"
            mae_str = f"{row['MAE_Mean']:.2f}±{row['MAE_Std']:.2f}" if not np.isnan(row['MAE_Mean']) else "N/A"
            print(f"{int(row['Context_Length']):<10} {row['Context_Hours']:<10.1f} {rmse_str:<25} {mae_str:<25} {int(row['N_Patients']):<15}")

    print("\n" + "="*90)


def main():
    parser = argparse.ArgumentParser(description='Zero-shot evaluation of Chronos-2 on glucose monitoring datasets')
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing patient CSV files')
    parser.add_argument('--output_dir', type=str, default='./zeroshot_results',
                        help='Directory to save results (default: ./zeroshot_results)')
    parser.add_argument('--model_name', type=str, default='amazon/chronos-2',
                        help='Model name or path (default: amazon/chronos-2)')
    parser.add_argument('--context_lengths', type=int, nargs='+', default=[12, 48, 96, 144, 192, 288],
                        help='Context lengths to evaluate (default: 12 48 96 144 192 288)')
    parser.add_argument('--prediction_length', type=int, default=18,
                        help='Prediction length (default: 18)')
    parser.add_argument('--step_size', type=int, default=1,
                        help='Step size for rolling window (default: 1)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='Optional dataset name for output files')
    
    args = parser.parse_args()

    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist!")
        return

    # Get dataset name from directory if not provided
    dataset_name = args.dataset_name if args.dataset_name else Path(args.data_dir).name

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("CHRONOS-2 ZERO-SHOT EVALUATION")
    print("="*70)
    print(f"Dataset: {dataset_name}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model: {args.model_name}")
    print(f"Context lengths: {args.context_lengths}")
    print(f"Prediction length: {args.prediction_length}")
    print(f"Device: {args.device}")
    print("="*70)

    # Load the Chronos-2 pipeline
    print(f"\nLoading model: {args.model_name}...")
    pipeline = BaseChronosPipeline.from_pretrained(
        args.model_name, 
        device_map=args.device
    )
    print("Model loaded successfully!")

    # Get all CSV files
    csv_files = sorted(Path(args.data_dir).glob("*.csv"))
    print(f"\nFound {len(csv_files)} patient files")

    if len(csv_files) == 0:
        print("No CSV files found in the specified directory!")
        return

    # Storage for all patient results
    all_patient_results = {}

    # Process each patient
    print("\n" + "="*70)
    print("PROCESSING ALL PATIENTS")
    print("="*70)

    for i, file_path in enumerate(csv_files, 1):
        patient_name = file_path.stem
        print(f"\n[{i}/{len(csv_files)}] Processing {patient_name}...")

        patient_results = evaluate_single_patient(
            file_path=str(file_path),
            pipeline=pipeline,
            context_lengths=args.context_lengths,
            prediction_length=args.prediction_length,
            step_size=args.step_size
        )

        all_patient_results[patient_name] = patient_results

    # Aggregate results
    print("\n" + "="*70)
    print("AGGREGATING RESULTS")
    print("="*70)

    summary_df = aggregate_results_across_patients(all_patient_results, args.context_lengths)

    # Print summary
    print_summary(summary_df)

    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    summary_file = f"{args.output_dir}/chronos2_summary_{dataset_name}.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved to: {summary_file}")

    save_detailed_results(all_patient_results, args.context_lengths, 
                         output_dir=f"{args.output_dir}/detailed_{dataset_name}")

    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
