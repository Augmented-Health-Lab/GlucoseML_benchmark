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
├── datasets/                    # Dataset loading and processing
│   ├── __init__.py
│   ├── diatrend.py             # DiaTrend dataset handler
│   ├── ohio.py                 # OhioT1DM dataset handler
│   └── t1dexi.py               # T1DEXI dataset handler
├── original_diatrend_experiments_60min/  # DiaTrend configs
│   ├── fold1_eval/             # Evaluation configs for fold 1
│   └── all_final_experiment_fold1.yaml  # Training configs for fold 1
├── original_ohio_experiments_60min/      # Ohio configs
│   ├── *_all_final_evaluation.yaml      # Subject-specific evaluations
│   └── all_final_experiment.yaml         # Training config
├── original_t1dexi_experiments_60min/    # T1DEXI configs
│   ├── fold1_eval/                      # Evaluation configs for fold 1
│   └── all_final_experiment_fold1.yaml     # Training configs for fold 1
├── result_tables/              # Performance results with different sampling horizon
├── diatrend_main.py           # DiaTrend training/evaluation script
├── ohio_main.py               # Ohio training/evaluation script
├── t1dexi_main.py            # T1DEXI training/evaluation script
├── training_evaluation_functions.py  # Core training functions
├── generate_new_yaml.ipynb         # Notebook to generate new yaml config files
├── run_diatrend.sh           # DiaTrend execution script
├── run_ohio.sh               # Ohio execution script
├── run_t1dexi.sh            # T1DEXI execution script
└── README.md                 # Project documentation
```

## How to run this method

The code execution relies on the configuration yaml files included in the following three folders:
```
original_diatrend_experiments_60min
original_ohio_experiments_60min
original_t1dexi_experiments_60min
```
Please ensure that the ```csv_path``` and ```xml_path``` parameters within each YAML file are updated with the correct absolute paths before running the execution code below.

### OhioT1DM
```bash
chmod +x run_ohio.sh
./run_ohio.sh
```
### DiaTrend
```bash
chmod +x run_diatrend.sh
./run_diatrend.sh 1 # Fold 1
...
```
This repository provides the training and evaluation code and configuration for Fold 1 only. To run training and evaluation for Folds 2 to 5, please use [generate_new_yaml.ipynb](./generate_new_yaml.ipynb) to generate new yaml files with your own data path.

### T1DEXI
```bash
chmod +x run_t1dexi.sh
./run_t1dexi.sh 1 # Fold 1
...
```
Similar with the DiaTrend code, this repository also only provides the training and evaluation code and configuration for Fold 1 only. To run training and evaluation for Folds 2 to 5, please use [generate_new_yaml.ipynb](./generate_new_yaml.ipynb) to generate new yaml files with your own data path.

## Reference
- [1] Martinsson, J., Schliep, A., Eliasson, B. et al. Blood Glucose Prediction with Variance Estimation Using Recurrent Neural Networks. J Healthc Inform Res 4, 1–18 (2020). https://doi.org/10.1007/s41666-019-00059-y. Open-source code: https://github.com/johnmartinsson/blood-glucose-prediction 