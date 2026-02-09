import logging
import sys
import os
import yaml
import pprint
import importlib.util
import tensorflow as tf
import itertools
import copy
import datetime
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)

import numpy as np
import Original_Martinsson.metrics as metrics
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt



def load_module(script_path):
    spec = importlib.util.spec_from_file_location("module.name", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_cfg(yaml_filepath):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    yaml_filepath : str

    Returns
    -------
    cfg : dict
    """
    # Read YAML experiment definition file
    with open(yaml_filepath, 'r') as stream:
        cfg = yaml.load(stream)
    cfg = make_paths_absolute(os.path.dirname(yaml_filepath), cfg)
    return cfg

def load_cfgs(yaml_filepath):
    """
    Load YAML configuration files.

    Parameters
    ----------
    yaml_filepath : str

    Returns
    -------
    cfgs : [dict]
    """
    # Read YAML experiment definition file
    with open(yaml_filepath, 'r') as stream:
        cfg = yaml.load(stream, Loader=yaml.SafeLoader)

    cfg = make_paths_absolute(os.path.dirname(yaml_filepath), cfg)

    hyperparameters = []
    hyperparameter_names = []
    hyperparameter_values = []
    # TODO: ugly, should handle arbitrary depth
    for k1 in cfg.keys():
        for k2 in cfg[k1].keys():
            if k2.startswith("param_"):
                hyperparameters.append((k1, k2))
                hyperparameter_names.append((k1, k2[6:]))
                hyperparameter_values.append(cfg[k1][k2])

    hyperparameter_valuess = itertools.product(*hyperparameter_values)


    artifacts_path = cfg['train']['artifacts_path']

    cfgs = []
    for hyperparameter_values in hyperparameter_valuess:
        configuration_name = ""
        for ((k1, k2), value) in zip(hyperparameter_names, hyperparameter_values):
            #print(k1, k2, value)
            cfg[k1][k2] = value
            configuration_name += "{}_{}_".format(k2, str(value))

        cfg['train']['artifacts_path'] = os.path.join(artifacts_path, configuration_name)

        cfgs.append(copy.deepcopy(cfg))

    return cfgs



def make_paths_absolute(dir_, cfg):
    """
    Make all values for keys ending with `_path` absolute to dir_.

    Parameters
    ----------
    dir_ : str
    cfg : dict

    Returns
    -------
    cfg : dict
    """
    for key in cfg.keys():
        if key.endswith("_path"):
            cfg[key] = os.path.join(dir_, cfg[key])
            cfg[key] = os.path.abspath(cfg[key])
            if not os.path.exists(cfg[key]):
                logging.error("%s does not exist.", cfg[key])
        if type(cfg[key]) is dict:
            cfg[key] = make_paths_absolute(dir_, cfg[key])
    return cfg

def evaluate_ohio(model, x_test, y_test, cfg):
    """
    Evaluate the performance of a trained model on the OhioT1DM dataset.

    This function calculates various performance metrics (RMSE, MAE, MSE, MAPE) 
    for the model's predictions and compares them to baseline metrics using the 
    last observed value (t0). The results are saved to files in the specified 
    artifacts directory and printed to the console.

    Parameters
    ----------
    model : keras.Model
        The trained model to evaluate.
    x_test : numpy.ndarray
        The input test data.
    y_test : numpy.ndarray
        The ground truth target values for the test data.
    cfg : dict
        Configuration dictionary containing paths and scaling information.

    Returns
    -------
    None
    """
    if 'xml_path' in cfg['dataset']:
        basename = os.path.basename(cfg['dataset']['xml_path'])
        patient_id = basename.split('-')[0]
    else:
        patient_id = ""
    if 'scale' in cfg['dataset']:
        scale = float(cfg['dataset']['scale'])
    else:
        scale = 1.0

    # load the trained weights
    weights_path = os.path.join(cfg['train']['artifacts_path'], "model.hdf5")
    print("loading weights: {}".format(weights_path))
    model.load_weights(weights_path)

    y_pred = model.predict(x_test)[:,1].flatten()/scale
    y_std  = model.predict(x_test)[:,0].flatten()/scale
    y_test = y_test.flatten()/scale
    t0 = x_test[:,-1,0]/scale

    # Calculate RMSE
    rmse = metrics.root_mean_squared_error(y_test, y_pred)
    print("patient id: ", patient_id)
    with open(os.path.join(cfg['train']['artifacts_path'], "{}_rmse.txt".format(patient_id)), "w") as outfile:
        outfile.write("{}\n".format(rmse))

    # Calculate MAE
    mae = np.mean(np.abs(y_test - y_pred))
    with open(os.path.join(cfg['train']['artifacts_path'], "{}_mae.txt".format(patient_id)), "w") as outfile:
        outfile.write("{}\n".format(mae))

    # Calculate MSE
    mse = np.mean((y_test - y_pred) ** 2)
    with open(os.path.join(cfg['train']['artifacts_path'], "{}_mse.txt".format(patient_id)), "w") as outfile:
        outfile.write("{}\n".format(mse))

    # Calculate MAPE
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Multiply by 100 for percentage
    with open(os.path.join(cfg['train']['artifacts_path'], "{}_mape.txt".format(patient_id)), "w") as outfile:
        outfile.write("{}\n".format(mape))


    # Calculate baseline (t0) metrics
    t0_rmse = metrics.root_mean_squared_error(y_test, t0)
    t0_mse = np.mean((y_test - t0) ** 2)
    t0_mape = np.mean(np.abs((y_test - t0) / y_test)) * 100
    t0_mae = np.mean(np.abs(y_test - t0))
    
    with open(os.path.join(cfg['train']['artifacts_path'], "{}_t0_rmse.txt".format(patient_id)), "w") as outfile:
        outfile.write("{}\n".format(t0_rmse))
    with open(os.path.join(cfg['train']['artifacts_path'], "{}_t0_mse.txt".format(patient_id)), "w") as outfile:
        outfile.write("{}\n".format(t0_mse))
    with open(os.path.join(cfg['train']['artifacts_path'], "{}_t0_mape.txt".format(patient_id)), "w") as outfile:
        outfile.write("{}\n".format(t0_mape))

    with open(os.path.join(cfg['train']['artifacts_path'], "{}_mean_std.txt".format(patient_id)), "w") as outfile:
        outfile.write("{}\n".format(np.mean(y_std)))
    # Calculate MAE
    with open(os.path.join(cfg['train']['artifacts_path'], "{}_t0_mae.txt".format(patient_id)), "w") as outfile:
        outfile.write("{}\n".format(t0_mae))

    # Print all metrics
    print("Model Performance Metrics:")
    print("-" * 25)
    print(f"RMSE: {rmse:.2f}")
    print(f"MSE:  {mse:.2f}")
    print(f"MAPE: {mape:.2f}%")

    print("\nBaseline (t0) Performance:")
    print("-" * 25)
    print(f"t0 RMSE: {t0_rmse:.2f}")
    print(f"t0 MSE:  {t0_mse:.2f}")
    print(f"t0 MAPE: {t0_mape:.2f}%")


def evaluate_diatrend(model, x_test, y_test, cfg):
    """
    Evaluate the performance of a trained model on the DiaTrend dataset.

    This function calculates various performance metrics (RMSE, MAE, MSE, MAPE) 
    for the model's predictions and compares them to baseline metrics using the 
    last observed value (t0). The results are saved to files in the specified 
    artifacts directory and printed to the console.

    Parameters
    ----------
    model : keras.Model
        The trained model to evaluate.
    x_test : numpy.ndarray
        The input test data.
    y_test : numpy.ndarray
        The ground truth target values for the test data.
    cfg : dict
        Configuration dictionary containing paths and scaling information.

    Returns
    -------
    None
    """
    if 'csv_path' in cfg['dataset']:
        basename = os.path.basename(cfg['dataset']['csv_path'])
        patient_id = basename.split('_')[-1].split('.')[0]
    else:
        patient_id = ""
    if 'scale' in cfg['dataset']:
        scale = float(cfg['dataset']['scale'])
    else:
        scale = 1.0
    print(f"Evaluating for patient_id: {patient_id}")
    # load the trained weights
    weights_path = os.path.join(cfg['train']['artifacts_path'], "model.hdf5")
    print("loading weights: {}".format(weights_path))
    model.load_weights(weights_path)

    y_pred = model.predict(x_test)[:,1].flatten()/scale
    y_std  = model.predict(x_test)[:,0].flatten()/scale
    y_test = y_test.flatten()/scale
    t0 = x_test[:,-1,0]/scale

    rmse = metrics.root_mean_squared_error(y_test, y_pred)
    print("patient id: ", patient_id)
    with open(os.path.join(cfg['train']['artifacts_path'], "{}_rmse.txt".format(patient_id)), "w") as outfile:
        outfile.write("{}\n".format(rmse))
    
    # Calculate MAE
    mae = np.mean(np.abs(y_test - y_pred))
    with open(os.path.join(cfg['train']['artifacts_path'], "{}_mae.txt".format(patient_id)), "w") as outfile:
        outfile.write("{}\n".format(mae))

    # Calculate MSE
    mse = np.mean((y_test - y_pred) ** 2)
    with open(os.path.join(cfg['train']['artifacts_path'], "{}_mse.txt".format(patient_id)), "w") as outfile:
        outfile.write("{}\n".format(mse))

    # Calculate MAPE
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Multiply by 100 for percentage
    with open(os.path.join(cfg['train']['artifacts_path'], "{}_mape.txt".format(patient_id)), "w") as outfile:
        outfile.write("{}\n".format(mape))


def evaluate_t1dexi(model, x_test, y_test, cfg):
    """
    Evaluate the performance of a trained model on the T1DEXI dataset.
    The same as the DiaTrend evaluation function.

    This function calculates various performance metrics (RMSE, MAE, MSE, MAPE) 
    for the model's predictions and compares them to baseline metrics using the 
    last observed value (t0). The results are saved to files in the specified 
    artifacts directory and printed to the console.

    Parameters
    ----------
    model : keras.Model
        The trained model to evaluate.
    x_test : numpy.ndarray
        The input test data.
    y_test : numpy.ndarray
        The ground truth target values for the test data.
    cfg : dict
        Configuration dictionary containing paths and scaling information.

    Returns
    -------
    None
    """
    if 'csv_path' in cfg['dataset']:
        basename = os.path.basename(cfg['dataset']['csv_path'])
        patient_id = basename.split('-')[0]
    else:
        patient_id = ""
    if 'scale' in cfg['dataset']:
        scale = float(cfg['dataset']['scale'])
    else:
        scale = 1.0

    # load the trained weights
    weights_path = os.path.join(cfg['train']['artifacts_path'], "model.hdf5")
    print("loading weights: {}".format(weights_path))
    model.load_weights(weights_path)

    y_pred = model.predict(x_test)[:,1].flatten()/scale
    y_std  = model.predict(x_test)[:,0].flatten()/scale
    y_test = y_test.flatten()/scale
    t0 = x_test[:,-1,0]/scale

    rmse = metrics.root_mean_squared_error(y_test, y_pred)
    print("patient id: ", patient_id)
    with open(os.path.join(cfg['train']['artifacts_path'], "{}_rmse.txt".format(patient_id)), "w") as outfile:
        outfile.write("{}\n".format(rmse))
    
    # Calculate MAE
    mae = np.mean(np.abs(y_test - y_pred))
    with open(os.path.join(cfg['train']['artifacts_path'], "{}_mae.txt".format(patient_id)), "w") as outfile:
        outfile.write("{}\n".format(mae))

    # Calculate MSE
    mse = np.mean((y_test - y_pred) ** 2)
    with open(os.path.join(cfg['train']['artifacts_path'], "{}_mse.txt".format(patient_id)), "w") as outfile:
        outfile.write("{}\n".format(mse))

    # Calculate MAPE
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Multiply by 100 for percentage
    with open(os.path.join(cfg['train']['artifacts_path'], "{}_mape.txt".format(patient_id)), "w") as outfile:
        outfile.write("{}\n".format(mape))



def train(model, module_train, x_train, y_train, x_valid, y_valid, cfg):
    model = module_train.train(
        model          = model,
        x_train        = x_train,
        y_train        = y_train,
        x_valid        = x_valid,
        y_valid        = y_valid,
        batch_size     = int(cfg['train']['batch_size']),
        epochs         = int(cfg['train']['epochs']),
        patience       = int(cfg['train']['patience']),
        shuffle        = cfg['train']['shuffle'],
        artifacts_path = cfg['train']['artifacts_path']
    )

    return model
