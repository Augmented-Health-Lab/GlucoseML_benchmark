import os
import sys
import glob
import copy
import yaml
import pprint
import logging
import itertools
import importlib.util

import numpy as np
import tensorflow as tf
import pandas as pd

import metrics  # your local metrics.py must exist

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)

# ============================================================
# YAML GENERATORS
# ============================================================

def generate_yaml_configs(
    dataset_dir,
    base_yaml_root,
    nb_past_steps=144,
    param_nb_future_steps=[6],
    dataset="mix",
    stride=1,
    seed=[20],
):
    """Generate per-subject evaluation YAMLs for all CSVs in dataset_dir."""
    horizon = param_nb_future_steps[0]

    new_yaml_dir = os.path.join(
        base_yaml_root,
        f"few_shot_open_dataset_eval_{horizon}_{dataset}"
    )
    os.makedirs(new_yaml_dir, exist_ok=True)

    csv_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith(".csv")])

    logging.info(f"Generating eval YAMLs in: {new_yaml_dir}")
    logging.info(f"Found {len(csv_files)} CSV files")

    for csv_file in csv_files:
        csv_path = os.path.join(dataset_dir, csv_file)

        yaml_content = {
            "dataset": {
                "csv_path": csv_path,
                "nb_past_steps": nb_past_steps,
                "param_nb_future_steps": param_nb_future_steps,
                "scale": 0.01,
                "script_path": "../../datasets/glucofm_bench.py",
                "test_fraction": 1.0,
                "train_fraction": 0.0,
                "valid_fraction": 0.0,
                "stride": stride,
            },
            "loss_function": {
                "script_path": "../../Original_Martinsson/loss_functions/nll_keras.py"
            },
            "model": {
                "activation_function": "exp",
                "nb_lstm_states": 256,
                "script_path": "../../Original_Martinsson/models/lstm_experiment_keras.py",
            },
            "optimizer": {
                "learning_rate": 0.001,
                "script_path": "../../Original_Martinsson/optimizers/adam_keras.py",
            },
            "train": {
                # NOTE: this is a BASE dir; load_cfgs() will append config_name
                "artifacts_path": f"../../artifacts/martinsson_kdd_experiment_{nb_past_steps}sh_few_{horizon}/",
                "batch_size": 1024,
                "epochs": 10000,
                "param_seed": seed,
                "patience": 50,
                "script_path": "../../Original_Martinsson/train/train_keras.py",
                "shuffle": False,
            },
        }

        yaml_filename = os.path.splitext(csv_file)[0] + ".yaml"
        yaml_path = os.path.join(new_yaml_dir, yaml_filename)

        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f, sort_keys=False)

    logging.info("Eval YAML generation complete.")
    return new_yaml_dir


def generate_fewshot_train_yaml(
    output_root,
    training_data_path,
    nb_past_steps=144,
    param_nb_future_steps=[6],
):
    """Generate ONE YAML for training on all data under training_data_path."""
    horizon = param_nb_future_steps[0]

    yaml_filename = f"few_shot_open_dataset_train_{horizon}.yaml"
    yaml_path = os.path.join(output_root, yaml_filename)

    yaml_content = {
        "dataset": {
            "script_path": "../datasets/glucofm_bench.py",
            "csv_path": training_data_path,
            "nb_past_steps": nb_past_steps,
            "param_nb_future_steps": param_nb_future_steps,
            "train_fraction": 0.9,
            "valid_fraction": 0.1,
            "test_fraction": 0.0,
            "scale": 0.01,
            "stride": 12,
        },
        "model": {
            "script_path": "../Original_Martinsson/models/lstm_experiment_keras.py",
            "nb_lstm_states": 256,
            "activation_function": "exp",
        },
        "optimizer": {
            "script_path": "../Original_Martinsson/optimizers/adam_keras.py",
            "learning_rate": 1e-3,
        },
        "loss_function": {
            "script_path": "../Original_Martinsson/loss_functions/nll_keras.py",
        },
        "train": {
            "script_path": "../Original_Martinsson/train/train_keras.py",
            # BASE dir; load_cfgs() will append config_name
            "artifacts_path": f"../artifacts/martinsson_kdd_experiment_{nb_past_steps}sh_few_{horizon}/",
            "batch_size": 1024,
            "epochs": 10000,
            "patience": 50,
            "shuffle": False,
            "param_seed": [20],
        },
    }

    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    logging.info(f"Generated train YAML: {yaml_path}")
    return yaml_path


# ============================================================
# CONFIG + MODULE LOADING
# ============================================================

def load_module(script_path):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location("loaded_module", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_paths_absolute(dir_, cfg):
    """Make all *_path entries absolute relative to dir_."""
    for key in list(cfg.keys()):
        if key.endswith("_path"):
            cfg[key] = os.path.abspath(os.path.join(dir_, cfg[key]))
            if not os.path.exists(cfg[key]):
                logging.error("Path does not exist: %s", cfg[key])
        if isinstance(cfg[key], dict):
            cfg[key] = make_paths_absolute(dir_, cfg[key])
    return cfg


def load_cfgs(yaml_filepath):
    """Load YAML and expand param_* hyperparameters into a list of cfg dicts."""
    with open(yaml_filepath, "r") as stream:
        cfg = yaml.load(stream, Loader=yaml.SafeLoader)

    cfg = make_paths_absolute(os.path.dirname(yaml_filepath), cfg)

    hyperparameters = []
    hyperparameter_names = []
    hyperparameter_values = []

    # find param_* entries
    for k1 in cfg.keys():
        for k2 in cfg[k1].keys():
            if k2.startswith("param_"):
                hyperparameters.append((k1, k2))
                hyperparameter_names.append((k1, k2[6:]))  # remove 'param_'
                hyperparameter_values.append(cfg[k1][k2])

    # cartesian product of hyperparams
    hyperparameter_valuess = itertools.product(*hyperparameter_values) if hyperparameter_values else [()]

    base_artifacts_path = cfg["train"]["artifacts_path"]

    cfgs = []
    for values in hyperparameter_valuess:
        cfg_i = copy.deepcopy(cfg)
        configuration_name = ""

        for ((k1, name), value) in zip(hyperparameter_names, values):
            cfg_i[k1][name] = value
            configuration_name += f"{name}_{value}_"

        # store the chosen seed under cfg['train']['seed'] for convenience
        if "seed" not in cfg_i["train"] and "param_seed" in cfg_i["train"]:
            # if param_seed wasn't expanded, take first
            cfg_i["train"]["seed"] = int(cfg_i["train"]["param_seed"][0])

        # artifacts path per configuration
        cfg_i["train"]["artifacts_path"] = os.path.join(base_artifacts_path, configuration_name)

        cfgs.append(cfg_i)

    return cfgs


# ============================================================
# TRAIN / EVAL CORE
# ============================================================

def train(model, module_train, x_train, y_train, x_valid, y_valid, cfg):
    os.makedirs(cfg["train"]["artifacts_path"], exist_ok=True)
    return module_train.train(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        batch_size=int(cfg["train"]["batch_size"]),
        epochs=int(cfg["train"]["epochs"]),
        patience=int(cfg["train"]["patience"]),
        shuffle=bool(cfg["train"]["shuffle"]),
        artifacts_path=cfg["train"]["artifacts_path"],
    )


def evaluate(model, x_test, y_test, cfg):
    # patient id
    basename = os.path.basename(cfg["dataset"]["csv_path"])
    patient_id = os.path.splitext(basename)[0]

    # dataset name from path: .../<dataset_name>/<patient>.csv
    dataset_name = os.path.basename(os.path.dirname(cfg["dataset"]["csv_path"]))

    scale = float(cfg["dataset"].get("scale", 1.0))

    print(f"Evaluating: {dataset_name} / {patient_id}")

    # load trained weights
    weights_path = os.path.join(cfg["train"]["artifacts_path"], "model.keras")
    print("Loading weights:", weights_path)
    model.load_weights(weights_path)

    pred = model.predict(x_test)
    y_pred = pred[:, 1].flatten() / scale
    y_test = y_test.flatten() / scale

    # output folder: artifacts/<exp>/DATASET_NAME/
    dataset_output_dir = os.path.join(cfg["train"]["artifacts_path"], dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)

    rmse = metrics.root_mean_squared_error(y_test, y_pred)
    mae = float(np.mean(np.abs(y_test - y_pred)))

    with open(os.path.join(dataset_output_dir, f"{patient_id}_rmse.txt"), "w") as f:
        f.write(f"{rmse}\n")

    with open(os.path.join(dataset_output_dir, f"{patient_id}_mae.txt"), "w") as f:
        f.write(f"{mae}\n")

    print(f"Saved metrics to: {dataset_output_dir}")


# ============================================================
# RUNNERS
# ============================================================

def run_training_from_yaml(yaml_filepath):
    cfgs = load_cfgs(yaml_filepath)
    logging.info(f"Running {len(cfgs)} train experiment(s) from: {yaml_filepath}")

    for exp_id, cfg in enumerate(cfgs, 1):
        print("=" * 80)
        print(f"TRAIN EXPERIMENT {exp_id}/{len(cfgs)}")
        print("=" * 80)

        seed = int(cfg["train"].get("seed", 20))
        np.random.seed(seed)
        tf.random.set_seed(seed)

        module_dataset = load_module(cfg["dataset"]["script_path"])
        module_model = load_module(cfg["model"]["script_path"])
        module_optimizer = load_module(cfg["optimizer"]["script_path"])
        module_loss_function = load_module(cfg["loss_function"]["script_path"])
        module_train = load_module(cfg["train"]["script_path"])

        pprint.PrettyPrinter(indent=4).pprint(cfg)

        result = module_dataset.load_dataset(cfg["dataset"])
        if result is None:
            print("Training dataset loader returned None — skipping.")
            continue
        x_train, y_train, x_valid, y_valid, x_test, y_test = result

        optimizer = module_optimizer.load(cfg["optimizer"])
        loss_function = module_loss_function.load()

        out_dim = y_train.shape[1] * 2 if "tf_nll" in loss_function.__name__ else y_train.shape[1]
        model = module_model.load(x_train.shape[1:], out_dim, cfg["model"])

        if "initial_weights_path" in cfg["train"]:
            model.load_weights(cfg["train"]["initial_weights_path"])

        model.compile(optimizer=optimizer, loss=loss_function)

        train(model, module_train, x_train, y_train, x_valid, y_valid, cfg)


def run_evaluation_from_yaml_folder(yaml_dir):
    yaml_files = sorted(glob.glob(os.path.join(yaml_dir, "*.yaml")))
    logging.info(f"Found {len(yaml_files)} eval YAML(s) in: {yaml_dir}")

    for y_idx, yaml_fp in enumerate(yaml_files, 1):
        print("=" * 80)
        print(f"EVAL YAML {y_idx}/{len(yaml_files)}: {yaml_fp}")
        print("=" * 80)

        cfgs = load_cfgs(yaml_fp)
        for exp_id, cfg in enumerate(cfgs, 1):

            seed = int(cfg["train"].get("seed", 20))
            np.random.seed(seed)
            tf.random.set_seed(seed)

            module_dataset = load_module(cfg["dataset"]["script_path"])
            module_model = load_module(cfg["model"]["script_path"])
            module_optimizer = load_module(cfg["optimizer"]["script_path"])
            module_loss_function = load_module(cfg["loss_function"]["script_path"])

            result = module_dataset.load_dataset(cfg["dataset"])
            if result is None:
                print(f"Skipping {cfg['dataset']['csv_path']} (too short)")
                continue

            x_train, y_train, x_valid, y_valid, x_test, y_test = result

            optimizer = module_optimizer.load(cfg["optimizer"])
            loss_function = module_loss_function.load()

            out_dim = y_train.shape[1] * 2 if "tf_nll" in loss_function.__name__ else y_train.shape[1]
            model = module_model.load(x_train.shape[1:], out_dim, cfg["model"])

            # IMPORTANT: evaluation needs the trained weights path.
            # You must set initial_weights_path in eval cfg OR ensure model.keras exists in artifacts_path.
            if "initial_weights_path" in cfg["train"]:
                model.load_weights(cfg["train"]["initial_weights_path"])

            model.compile(optimizer=optimizer, loss=loss_function)

            evaluate(model, x_test, y_test, cfg)


# ============================================================
# MAIN FLOW
# ============================================================

def main():
    base_root = "/content/drive/Shareddrives/Baiying/2019Martinsson_et_al_LSTM/glucofm_fewshot_open_144"
    nb_past_steps = 144
    horizon = 3
    dataset_tag = "mix"

    # 1) Generate eval YAMLs (per subject)
    # eval_dir = generate_yaml_configs(
    #     dataset_dir="/content/drive/Shareddrives/Baiying/preprocessed_dataset/test_dataset/mixed",
    #     base_yaml_root=base_root,
    #     nb_past_steps=nb_past_steps,
    #     param_nb_future_steps=[horizon],
    #     dataset=dataset_tag,
    #     stride=1,
    # )

    eval_dir_ohio = generate_yaml_configs(
        dataset_dir="/content/drive/Shareddrives/Baiying/preprocessed_dataset/test_dataset/controlled_datasets/OhioT1DM",
        base_yaml_root=base_root,
        nb_past_steps=nb_past_steps,
        param_nb_future_steps=[horizon],
        dataset="ohio",
        stride=1,
    )
    run_evaluation_from_yaml_folder(eval_dir_ohio)

    eval_dir_diatrend = generate_yaml_configs(
        dataset_dir="/content/drive/Shareddrives/Baiying/preprocessed_dataset/test_dataset/controlled_datasets/8_DiaTrend",
        base_yaml_root=base_root,
        nb_past_steps=nb_past_steps,
        param_nb_future_steps=[horizon],
        dataset="diatrend",
        stride=3,
    )
    run_evaluation_from_yaml_folder(eval_dir_diatrend)

    eval_dir_t1dexi = generate_yaml_configs(
        dataset_dir="/content/drive/Shareddrives/Baiying/preprocessed_dataset/test_dataset/controlled_datasets/5_T1DEXI",
        base_yaml_root=base_root,
        param_nb_future_steps=[horizon],
        dataset="t1dexi",
        stride=1,
    )
    run_evaluation_from_yaml_folder(eval_dir_t1dexi)

    
    
    
if __name__ == "__main__":
    main()
