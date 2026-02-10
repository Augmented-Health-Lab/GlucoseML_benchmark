# CALF for Glucose Forecasting

This directory contains the implementation and experiments for CALF (Context-Aware Language Foundation) applied to glucose monitoring data.

## Overview

CALF is a foundation model for time series forecasting that uses a GPT-based architecture with specialized attention mechanisms. This implementation adapts CALF for glucose prediction tasks.

## Installation

```bash
cd CALF
pip install -r new_requirement.txt
```

## PCA Preparation

Before training, generate the PCA embeddings required by CALF:

```bash
python pca.py
```

This creates `wte_pca_500.pt` which contains the pre-computed word token embeddings.

## Dataset Preparation

### Option 1: Prepare from HuggingFace Dataset

Use the provided script to download and organize the dataset:

```bash
# Basic preparation
python prepare_dataset.py \
    --hf_name byluuu/gluco-tsfm-benchmark \
    --output_dir ./hf_cache

# With mixed dataset (combines all subdatasets)
python prepare_dataset.py \
    --hf_name byluuu/gluco-tsfm-benchmark \
    --output_dir ./hf_cache \
    --create_mixed
```

This will create:
```
hf_cache/
├── train/
│   ├── BIG_IDEA_LAB/
│   ├── ShanghaiT1DM/
│   ├── ...
│   └── mixed/          # All training data combined
└── test/
    ├── BIG_IDEA_LAB/
    ├── ShanghaiT1DM/
    ├── ...
    └── mixed/          # All test data combined
```

### Option 2: Use Existing CSV Files

If you already have CSV files organized in the required format, ensure they follow this structure:
- Each file: `{subject_id}.csv`
- Columns: `timestamp`, `BGvalue`

## Training

Train the model on the training dataset:

```bash
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model CALF \
  --model_id glucose_train \
  --data Glucose \
  --root_path ./hf_cache/train/mixed \
  --features S \
  --target BGvalue \
  --freq 5min \
  --d_model 768 \
  --n_heads 12 \
  --seq_len 144 \
  --label_len 72 \
  --pred_len 18 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --stride 12 \
  --batch_size 8 \
  --gpt_layers 4 \
  --train_epochs 30 \
  --patience 8 \
  --learning_rate 1e-4 \
  --scale_value 1.0 \
  --max_windows_per_epoch 30000 \
  --num_workers 2 \
  --use_gpu 1 \
  --per_subject_eval 0
```

### Key Training Parameters

- `--seq_len 144`: Context length (144 × 5min = 12 hours)
- `--label_len 72`: Label length for decoder (half of seq_len)
- `--pred_len 18`: Prediction length (18 × 5min = 90 minutes)
- `--stride 12`: Stride for sliding window (12 × 5min = 1 hour)
- `--batch_size 8`: Batch size (adjust based on GPU memory)
- `--train_epochs 30`: Number of training epochs
- `--patience 8`: Early stopping patience
- `--d_model 768`: Model dimension
- `--n_heads 12`: Number of attention heads
- `--gpt_layers 4`: Number of GPT layers
- `--scale_value 1.0`: Scaling factor for input normalization
- `--max_windows_per_epoch 30000`: Max windows per epoch (prevents OOM)

## Evaluation

### Standard Evaluation (Mixed Test Set)

Evaluate on all test subjects:

```bash
python run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --model CALF \
  --model_id glucose_train \
  --data Glucose \
  --root_path ./unused/during/test \
  --test_root_path ./hf_cache/test/mixed \
  --features S \
  --target BGvalue \
  --freq 5min \
  --checkpoints ./checkpoints \
  --seq_len 144 \
  --label_len 72 \
  --pred_len 18 \
  --d_model 768 \
  --n_heads 12 \
  --gpt_layers 4 \
  --batch_size 32 \
  --stride 1 \
  --scale_value 1.0
```

### Evaluate Specific Dataset

To evaluate on a specific dataset (e.g., T1DEXI):

```bash
python run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --model CALF \
  --model_id glucose_train \
  --data Glucose \
  --root_path ./unused/during/test \
  --test_root_path ./hf_cache/test/T1DEXI \
  --features S \
  --target BGvalue \
  --freq 5min \
  --checkpoints ./checkpoints \
  --seq_len 144 \
  --label_len 72 \
  --pred_len 18 \
  --d_model 768 \
  --n_heads 12 \
  --gpt_layers 4 \
  --batch_size 32 \
  --stride 1 \
  --scale_value 1.0
```

## Multi-Horizon Evaluation

Evaluate at different prediction horizons by changing `--pred_len`:

### 15-Minute Prediction (pred_len=3)
```bash
python run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --model CALF \
  --model_id glucose_train \
  --data Glucose \
  --root_path ./unused/during/test \
  --test_root_path ./hf_cache/test/mixed \
  --features S \
  --target BGvalue \
  --freq 5min \
  --checkpoints ./checkpoints \
  --seq_len 144 \
  --label_len 72 \
  --pred_len 3 \
  --d_model 768 \
  --n_heads 12 \
  --gpt_layers 4 \
  --batch_size 32 \
  --stride 1 \
  --scale_value 1.0
```

### 30-Minute Prediction (pred_len=6)
```bash
python run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --model CALF \
  --model_id glucose_train \
  --data Glucose \
  --root_path ./unused/during/test \
  --test_root_path ./hf_cache/test/mixed \
  --features S \
  --target BGvalue \
  --freq 5min \
  --checkpoints ./checkpoints \
  --seq_len 144 \
  --label_len 72 \
  --pred_len 6 \
  --d_model 768 \
  --n_heads 12 \
  --gpt_layers 4 \
  --batch_size 32 \
  --stride 1 \
  --scale_value 1.0
```

### 60-Minute Prediction (pred_len=12)
```bash
python run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --model CALF \
  --model_id glucose_train \
  --data Glucose \
  --root_path ./unused/during/test \
  --test_root_path ./hf_cache/test/mixed \
  --features S \
  --target BGvalue \
  --freq 5min \
  --checkpoints ./checkpoints \
  --seq_len 144 \
  --label_len 72 \
  --pred_len 12 \
  --d_model 768 \
  --n_heads 12 \
  --gpt_layers 4 \
  --batch_size 32 \
  --stride 1 \
  --scale_value 1.0
```

### 90-Minute Prediction (pred_len=18)
```bash
python run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --model CALF \
  --model_id glucose_train \
  --data Glucose \
  --root_path ./unused/during/test \
  --test_root_path ./hf_cache/test/mixed \
  --features S \
  --target BGvalue \
  --freq 5min \
  --checkpoints ./checkpoints \
  --seq_len 144 \
  --label_len 72 \
  --pred_len 18 \
  --d_model 768 \
  --n_heads 12 \
  --gpt_layers 4 \
  --batch_size 32 \
  --stride 1 \
  --scale_value 1.0
```

## Prediction Horizon Options

| Horizon | pred_len | Time | Description |
|---------|----------|------|-------------|
| 15min   | 3        | 3 × 5min | Short-term prediction |
| 30min   | 6        | 6 × 5min | Medium-term prediction |
| 60min   | 12       | 12 × 5min | Long-term prediction |
| 90min   | 18       | 18 × 5min | Extended prediction |

**Note:** You can train separate models for each horizon or use the same model with different `pred_len` during evaluation.

## Results

Results are saved in:
```
./results/
└── long_term_forecast_Glucose_144_72_{pred_len}_CALF_{model_id}/
    ├── metrics.csv          # Evaluation metrics (MSE, MAE)
    ├── pred.npy            # Predictions
    └── true.npy            # Ground truth
```

## Parameter Guide

### Model Architecture
- `--model CALF`: Use CALF model
- `--d_model 768`: Model dimension
- `--n_heads 12`: Number of attention heads
- `--gpt_layers 4`: Number of GPT layers

### Data Processing
- `--seq_len`: Context window length (timesteps)
- `--label_len`: Label length for decoder
- `--pred_len`: Prediction horizon (timesteps)
- `--freq`: Data frequency (5min for glucose)
- `--stride`: Sliding window stride (12 for training, 1 for testing)
- `--scale_value`: Scaling factor for normalization

### Training
- `--batch_size`: Batch size (8 for training, 32 for testing)
- `--train_epochs`: Number of epochs
- `--learning_rate`: Learning rate
- `--patience`: Early stopping patience
- `--max_windows_per_epoch`: Max windows per epoch

### Paths
- `--root_path`: Training data directory
- `--test_root_path`: Test data directory (for evaluation)
- `--checkpoints`: Model checkpoint directory (default: ./checkpoints/)

### Other
- `--use_gpu`: Use GPU (1 for yes, 0 for no)
- `--per_subject_eval`: Per-subject evaluation (0 for aggregate, 1 for per-subject)

## Tips

1. **PCA Preparation**: Always run `python pca.py` before training
2. **Memory Management**: Reduce `--batch_size` or `--max_windows_per_epoch` if encountering OOM errors
3. **Multiple Horizons**: Train once, evaluate at multiple horizons by changing `pred_len`
4. **Dataset-Specific**: Use `--test_root_path` to evaluate on specific datasets
5. **Evaluation Stride**: Use `stride=1` during evaluation for denser predictions
6. **Batch Size**: Use smaller batch size (8) for training, larger (32) for evaluation

## Citation

If you use this implementation, please cite the original CALF paper:
```bibtex
@article{liu2024calf,
  title={CALF: Aligning LLMs for Time Series Forecasting via Cross-modal Fine-Tuning},
  author={Liu, Peiyuan and Zhao, Hang and Li, Tao and others},
  journal={arXiv preprint arXiv:2403.07300},
  year={2024}
}
```
