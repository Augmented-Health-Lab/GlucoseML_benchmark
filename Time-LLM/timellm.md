# Time-LLM for Glucose Forecasting

This directory contains the implementation and experiments for Time-LLM applied to glucose monitoring data.

## Overview

Time-LLM leverages large language models (LLMs) for time series forecasting by reprogramming the LLM to understand temporal patterns. This implementation adapts Time-LLM for glucose prediction tasks.

## Installation

```bash
cd Time-LLM
pip install -r requirements.txt
```

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
python run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model TimeLLM \
  --model_id Glucose_train_main \
  --model_comment "GlucoseTrain" \
  --llm_model GPT2 \
  --llm_layers 4 \
  --llm_dim 768 \
  --data Glucose \
  --root_path ./hf_cache/train/mixed \
  --features S \
  --target glucose \
  --freq 5min \
  --seq_len 144 \
  --label_len 72 \
  --pred_len 18 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --train_epochs 40 \
  --learning_rate 5e-4 \
  --num_workers 2 \
  --stride 12 \
  --max_windows_per_epoch 30000 \
  --des GlucoseTrain
```

### Key Training Parameters

- `--seq_len 144`: Context length (144 × 5min = 12 hours)
- `--label_len 72`: Label length for decoder (using half of the seq_len)
- `--pred_len 18`: Prediction length (18 × 5min = 90 minutes. Other options: 3, 6, 12)
- `--stride 12/240`: Fullshot: Stride for sliding window (12 × 5min = 1 hour). Fewshot: Stride for sliding window (240 × 5min = 20 hour)
- `--batch_size 16`: Batch size (adjust based on GPU memory)
- `--train_epochs 40`: Number of training epochs
- `--max_windows_per_epoch 30000`: Max windows per epoch (prevents OOM)

## Evaluation

### Standard Evaluation (Mixed Test Set)

Evaluate on all test subjects:

```bash
python run_main.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --model TimeLLM \
  --model_id Glucose_train_main \
  --model_comment "GlucoseTrain" \
  --llm_model GPT2 \
  --llm_layers 4 \
  --llm_dim 768 \
  --data Glucose \
  --test_root_path ./hf_cache/test/mixed \
  --features S \
  --target glucose \
  --freq 5min \
  --seq_len 144 \
  --label_len 72 \
  --pred_len 18 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --num_workers 2 \
  --stride 3 \
  --des GlucoseTrain
```

### Evaluate Specific Dataset

To evaluate on a specific dataset (e.g., BIG_IDEA_LAB):

```bash
python run_main.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --model TimeLLM \
  --model_id Glucose_train_main \
  --model_comment "GlucoseTrain_BIG_IDEA_LAB" \
  --llm_model GPT2 \
  --llm_layers 4 \
  --llm_dim 768 \
  --data Glucose \
  --test_root_path ./hf_cache/test/BIG_IDEA_LAB \
  --features S \
  --target glucose \
  --freq 5min \
  --seq_len 144 \
  --label_len 72 \
  --pred_len 18 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --num_workers 2 \
  --stride 3 \
  --des GlucoseTrain
```

## Multi-Horizon Evaluation

Evaluate at different prediction horizons by changing `--pred_len`:

### 15-Minute Prediction (pred_len=3)
```bash
python run_main.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --model TimeLLM \
  --model_id Glucose_train_main \
  --model_comment "GlucoseTrain_15min" \
  --llm_model GPT2 \
  --llm_layers 4 \
  --llm_dim 768 \
  --data Glucose \
  --test_root_path ./hf_cache/test/mixed \
  --features S \
  --target glucose \
  --freq 5min \
  --seq_len 144 \
  --label_len 72 \
  --pred_len 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --num_workers 2 \
  --stride 3 \
  --des GlucoseTrain
```

### 30-Minute Prediction (pred_len=6)
```bash
python run_main.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --model TimeLLM \
  --model_id Glucose_train_main \
  --model_comment "GlucoseTrain_30min" \
  --llm_model GPT2 \
  --llm_layers 4 \
  --llm_dim 768 \
  --data Glucose \
  --test_root_path ./hf_cache/test/mixed \
  --features S \
  --target glucose \
  --freq 5min \
  --seq_len 144 \
  --label_len 72 \
  --pred_len 6 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --num_workers 2 \
  --stride 3 \
  --des GlucoseTrain
```

### 60-Minute Prediction (pred_len=12)
```bash
python run_main.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --model TimeLLM \
  --model_id Glucose_train_main \
  --model_comment "GlucoseTrain_60min" \
  --llm_model GPT2 \
  --llm_layers 4 \
  --llm_dim 768 \
  --data Glucose \
  --test_root_path ./hf_cache/test/mixed \
  --features S \
  --target glucose \
  --freq 5min \
  --seq_len 144 \
  --label_len 72 \
  --pred_len 12 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --num_workers 2 \
  --stride 3 \
  --des GlucoseTrain
```

### 90-Minute Prediction (pred_len=18)
```bash
python run_main.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --model TimeLLM \
  --model_id Glucose_train_main \
  --model_comment "GlucoseTrain_90min" \
  --llm_model GPT2 \
  --llm_layers 4 \
  --llm_dim 768 \
  --data Glucose \
  --test_root_path ./hf_cache/test/mixed \
  --features S \
  --target glucose \
  --freq 5min \
  --seq_len 144 \
  --label_len 72 \
  --pred_len 18 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --num_workers 2 \
  --stride 3 \
  --des GlucoseTrain
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
└── long_term_forecast_Glucose_144_72_{pred_len}_TimeLLM_{model_comment}/
    ├── metrics.csv          # Evaluation metrics (MSE, MAE)
    ├── pred.npy            # Predictions
    └── true.npy            # Ground truth
```

## Parameter Guide

### Model Architecture
- `--llm_model GPT2`: Base LLM (GPT2, LLAMA, etc.)
- `--llm_layers 4`: Number of LLM layers to use
- `--llm_dim 768`: LLM hidden dimension

### Data Processing
- `--seq_len`: Context window length (timesteps)
- `--label_len`: Label length for decoder
- `--pred_len`: Prediction horizon (timesteps)
- `--freq`: Data frequency (5min for glucose)
- `--stride`: Sliding window stride for training/testing

### Training
- `--batch_size`: Batch size
- `--train_epochs`: Number of epochs
- `--learning_rate`: Learning rate
- `--patience`: Early stopping patience

### Paths
- `--root_path`: Training data directory
- `--test_root_path`: Test data directory (for evaluation)
- `--checkpoints`: Model checkpoint directory (default: ./checkpoints/)

## Tips

1. **Memory Management**: Reduce `--batch_size` or `--max_windows_per_epoch` if encountering OOM errors
2. **Multiple Horizons**: Train once, evaluate at multiple horizons by changing `pred_len`
3. **Dataset-Specific**: Use `--test_root_path` to evaluate on specific datasets
4. **Evaluation Stride**: Use smaller `--stride` (e.g., 3) during evaluation for denser predictions

## Citation

If you use this implementation, please cite the original Time-LLM paper:
```bibtex
@article{jin2023time,
  title={Time-LLM: Time Series Forecasting by Reprogramming Large Language Models},
  author={Jin, Ming and Wang, Shiyu and Ma, Lintao and Chu, Zhixuan and Zhang, James Y and Shi, Xiaoming and Chen, Pin-Yu and Liang, Yuxuan and Li, Yuan-Fang and Pan, Shirui and others},
  journal={arXiv preprint arXiv:2310.01728},
  year={2023}
}
```
