# Replicate the Model Proposed by Martionsson et al., 2019

This replication work is based on the code from the original study's repository: https://github.com/johnmartinsson/blood-glucose-prediction. However, not all files or functions from the original repository were used in this study.

## Structure

```
2019Martinsson_et_al_LSTM/
├── Original_Martinsson/           # Original code from Martinsson et al[1].
│   ├── datasets/
│   │   └── ohio.py                # OhioT1DM dataset handler
│   ├── loss_functions/           # Loss function implementations
│   │   ├── gmse_keras.py        # GMSE loss function
│   │   ├── mse_keras.py         # MSE loss function
│   │   ├── nll_keras.py         # NLL loss function (main)
│   │   └── nll_mse_keras.py     # Combined NLL-MSE loss
│   ├── models/                   # Model architectures
│   │   └── lstm_experiment_keras.py  # Main LSTM model implementation
│   ├── optimizers/              # Optimization algorithms
│   │   └── adam_keras.py        # Adam optimizer
│   ├── train/                   # Training utilities
│   |    └── train_keras.py       # Main training loop
│   └── utils.py                   # Utility functions
├── datasets_loader/             # Dataset loading and processing
│   ├── __init__.py
│   ├── diatrend.py             # DiaTrend dataset handler
│   ├── glucofm_bench.py        # GlucoFM benchmark dataset handler (local & HF)
│   ├── mixed.py                # Mixed dataset handler
│   ├── ohio.py                 # OhioT1DM dataset handler
│   └── t1dexi.py               # T1DEXI dataset handler
├── glucofm_fullshot_open_144/   # GlucoFM fullshot experiment configs
│   ├── full_shot_open_dataset_train_*.yaml  # Training configs
│   └── full_shot_open_dataset_eval_*/       # Evaluation configs per subject
├── result_tables/              # Performance results with different sampling horizon
├── fullshot_lstm.py           # GlucoFM fullshot training/evaluation (HF + Local)
├── fewshot_lstm.py            # GlucoFM fewshot training/evaluation (HF + Local)
├── fullshot_oncontrol.py      # GlucoFM fullshot on controlled datasets
├── fewshot_oncontrol.py       # GlucoFM fewshot on controlled datasets
├── training_evaluation_functions.py  # Core training functions
├── metrics.py                 # Evaluation metrics
├── utils.py                   # Utility functions
├── requirements.txt           # Python dependencies
├── generate_new_yaml.ipynb    # Notebook to generate new yaml config files
└── README.md                  # Project documentation
```

## Installation

Install dependencies:
```bash
cd 2019Martinsson_et_al_LSTM
pip install -r requirements.txt
```

## How to run this method

### GlucoFM Benchmark Scripts
- **Dual Data Loading**: Supports both HuggingFace datasets and local CSV files
- **Flexible Configuration**: YAML-based configuration system for reproducible experiments
- **Multi-Horizon Evaluation**: Evaluate at 15min, 30min, 60min, and 90min prediction horizons
- **Fullshot vs Fewshot**: 
  - Fullshot: Train on all available training data
  - Fewshot: Limited training data scenarios
- **Automated Workflows**: Single script handles training, evaluation, and result saving

### Dataset Support
- **HuggingFace**: `byluuu/gluco-tsfm-benchmark` with 80/20 train/test split
- **Local CSV**: Custom glucose monitoring datasets with timestamp and BGvalue columns
- **Original Datasets**: OhioT1DM, DiaTrend, T1DEXI from Martinsson et al

#### Fewshot Training & Evaluation

Run fewshot LSTM with HuggingFace dataset:
```bash
python fewshot_lstm.py
```

Similar to fullshot, you can switch between HuggingFace and local modes by editing the `mode` variable and paths in `fewshot_lstm.py`.

#### Configuration Options

Both scripts support:
- **Context lengths**: `nb_past_steps` (default: 144 timesteps = 12 hours)
- **Prediction horizons**: `param_nb_future_steps` (e.g., [3, 6, 12, 18] for 15min, 30min, 60min, 90min)
- **HuggingFace dataset**: `hf_dataset="byluuu/gluco-tsfm-benchmark"`
- **Dataset filtering**: `filter_dataset="BIG_IDEA_LAB"` (for specific dataset evaluation)

Results are saved in:
```
artifacts/martinsson_kdd_experiment_<context>sh_<shot>_<horizon>/
└── <dataset_name>/
    ├── <subject_id>_rmse.txt
    └── <subject_id>_mae.txt
```



## Reference
- [1] Martinsson, J., Schliep, A., Eliasson, B. et al. Blood Glucose Prediction with Variance Estimation Using Recurrent Neural Networks. J Healthc Inform Res 4, 1–18 (2020). https://doi.org/10.1007/s41666-019-00059-y. Open-source code: https://github.com/johnmartinsson/blood-glucose-prediction 