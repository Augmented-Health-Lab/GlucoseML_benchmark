import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import utils
import os
import glob
# "C:\Users\baiyi\OneDrive\Desktop\Modify_GenBG\OhioT1DM 2020\fold1_training\552-ws-combined.xml"
# "C:\Users\baiyi\OneDrive\Desktop\Modify_GenBG\modified_diatrend_subset\fold1_training\processed_cgm_data_Subject12.csv"
# "C:\Users\baiyi\OneDrive\Desktop\Modify_GenBG\modified_t1dexi_subset\T1DEXI_cgm_processed\fold1_training\252.csv"
# def load_dataset(cfg):
#     if os.path.basename(cfg['mix_file_path']) == 'all':
#         print("loading training data for all patients ...")
#         ohio_files = glob.glob(os.path.join(cfg['ohio_path'], "*.xml"))
#         diatrend_files = glob.glob(os.path.join(cfg['diatrend_path'], "*.csv"))
#         t1dexi_files = glob.glob(os.path.join(cfg['t1dexi_path'], "*.csv"))
#         all_files = ohio_files + diatrend_files + t1dexi_files
#         tups = []
#         for file_path in all_files:
#             if file_path in ohio_files:
#                 dataset_type = 'ohio'
#             elif file_path in diatrend_files:
#                 dataset_type = 'diatrend'
#             elif file_path in t1dexi_files:
#                 dataset_type = 't1dexi'
            
#             cfg['mix_file_path'] = file_path
#             tups.append(load_data(cfg, dataset_type))

#         # for mix_file_path in all_files:
#         #     cfg['mix_file_path'] = mix_file_path
#         #     tups.append(load_data(cfg))
#         x_train = np.concatenate([t[0] for t in tups], axis=0)
#         y_train = np.concatenate([t[1] for t in tups], axis=0)
#         x_valid = np.concatenate([t[2] for t in tups], axis=0)
#         y_valid = np.concatenate([t[3] for t in tups], axis=0)
#         x_test = np.concatenate([t[4] for t in tups], axis=0)
#         y_test = np.concatenate([t[5] for t in tups], axis=0)

#         cfg['mix_file_path'] = 'all'
#         return x_train, y_train, x_valid, y_valid, x_test, y_test
#     else:
#         x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(cfg, dataset_type)
#         return x_train, y_train, x_valid, y_valid, x_test, y_test
def load_dataset(cfg):
    print("loading training data for all patients ...")
    
    # Collect all file paths from the three datasets
    ohio_files = glob.glob(os.path.join(cfg['ohio_path'], "*.xml"))
    diatrend_files = glob.glob(os.path.join(cfg['diatrend_path'], "*.csv"))
    t1dexi_files = glob.glob(os.path.join(cfg['t1dexi_path'], "*.csv"))
    
    all_files = ohio_files + diatrend_files + t1dexi_files
    tups = []
    
    for file_path in all_files:
        if file_path in ohio_files:
            dataset_type = 'ohio'
        elif file_path in diatrend_files:
            dataset_type = 'diatrend'
        elif file_path in t1dexi_files:
            dataset_type = 't1dexi'
        
        cfg['mix_file_path'] = file_path
        tups.append(load_data(cfg, dataset_type))
    
    x_train = np.concatenate([t[0] for t in tups], axis=0)
    y_train = np.concatenate([t[1] for t in tups], axis=0)
    x_valid = np.concatenate([t[2] for t in tups], axis=0)
    y_valid = np.concatenate([t[3] for t in tups], axis=0)
    x_test = np.concatenate([t[4] for t in tups], axis=0)
    y_test = np.concatenate([t[5] for t in tups], axis=0)

    return x_train, y_train, x_valid, y_valid, x_test, y_test

def load_data(cfg, dataset_type):
    mix_file_path        = cfg['mix_file_path']
    nb_past_steps   = int(cfg['nb_past_steps'])
    nb_future_steps = int(cfg['nb_future_steps'])
    train_fraction  = float(cfg['train_fraction'])
    valid_fraction  = float(cfg['valid_fraction'])
    test_fraction   = float(cfg['test_fraction'])
    

    xs, ys = load_glucose_data(mix_file_path, nb_past_steps, nb_future_steps, dataset_type)
    ys = np.expand_dims(ys, axis=1)

    x_train, x_valid, x_test = utils.split_data(xs, train_fraction,
            valid_fraction, test_fraction)
    y_train, y_valid, y_test = utils.split_data(ys, train_fraction,
            valid_fraction, test_fraction)

    # scale data
    scale = float(cfg['scale'])
    x_train *= scale
    y_train *= scale
    x_valid *= scale
    y_valid *= scale
    x_test  *= scale
    y_test  *= scale

    return x_train, y_train, x_valid, y_valid, x_test, y_test

def load_glucose_data(mix_file_path, nb_past_steps, nb_future_steps, dataset_type):
    if dataset_type == 'ohio':
        df_glucose_level = load_ohio_series(mix_file_path, "glucose_level", "value")
    elif dataset_type == 'diatrend':
        df_glucose_level = load_diatrend_series(mix_file_path)
    elif dataset_type == 't1dexi':
        df_glucose_level = load_T1DEXI_series(mix_file_path)
    else:
        raise ValueError("Unsupported dataset type")

    dt = df_glucose_level.index.to_series().diff().dropna()
    idx_breaks = np.argwhere(dt > pd.Timedelta(6, 'm'))

    # It would be possible to load more features here
    nd_glucose_level = df_glucose_level.values
    consecutive_segments = np.split(nd_glucose_level, idx_breaks.flatten())

    consecutive_segments = [c for c in consecutive_segments if len(c) >=
                            nb_past_steps + nb_future_steps]

    sups = [utils.sequence_to_supervised(c, nb_past_steps, nb_future_steps) for
            c in consecutive_segments]

    xss = [sup[0] for sup in sups]
    yss = [sup[1] for sup in sups]

    xs = np.concatenate(xss)
    ys = np.concatenate(yss)

    return np.expand_dims(xs, axis=2), ys



# def load_glucose_data(mix_file_path, nb_past_steps, nb_future_steps):
#     df_glucose_level = load_ohio_series(mix_file_path, "glucose_level", "value")
#     dt = df_glucose_level.index.to_series().diff().dropna()
#     idx_breaks = np.argwhere(dt>pd.Timedelta(6, 'm'))

#     # It would be possible to load more features here
#     nd_glucose_level = df_glucose_level.values
#     consecutive_segments = np.split(nd_glucose_level, idx_breaks.flatten())

#     consecutive_segments = [c for c in consecutive_segments if len(c) >=
#             nb_past_steps+nb_future_steps]

#     sups = [utils.sequence_to_supervised(c, nb_past_steps, nb_future_steps) for
#             c in consecutive_segments]

#     xss = [sup[0] for sup in sups]
#     yss = [sup[1] for sup in sups]

#     xs = np.concatenate(xss)
#     ys = np.concatenate(yss)

#     return np.expand_dims(xs, axis=2), ys

def load_ohio_series(mix_file_path, variate_name, attribute, time_attribue="ts"):
    tree = ET.parse(mix_file_path)
    root = tree.getroot()
    for child in root:
        if child.tag == variate_name:
            dates = []
            values = []
            for event in child:
                ts = event.attrib[time_attribue]
                date = pd.to_datetime(ts, format='%d-%m-%Y %H:%M:%S')
                date = date.replace(second=0)
                value = float(event.attrib[attribute])
                dates.append(date)
                values.append(value)
            index = pd.DatetimeIndex(dates)
            series = pd.Series(values, index=index)
            return series
        
def parse_date(date_str):
    try:
        return pd.to_datetime(date_str, format="%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        try:
            return pd.to_datetime(date_str, format="%Y-%m-%d %H:%M:%S")
        except ValueError:
            return pd.NaT

def load_diatrend_series(path):
    subject = pd.read_csv(path)

    # Lists to store the results
    parsed_dates = []
    values = []

    # Iterate through each row in the DataFrame
    for index, row in subject.iterrows():
        # Parse the date using the custom function
        parsed_date = parse_date(row['date'])
        
        # Append the parsed date and corresponding value to the lists
        parsed_dates.append(parsed_date)
        values.append(float(row['mg/dl']))

    # Now 'parsed_dates' and 'values' contain your data
    # print(parsed_dates)
    # print(values)
    index = pd.DatetimeIndex(parsed_dates)
    series = pd.Series(values, index=index)
    
    return series

def load_T1DEXI_series(path):
    subject = pd.read_csv(path)

    # Lists to store the results
    parsed_dates = []
    values = []

    # Iterate through each row in the DataFrame
    for index, row in subject.iterrows():
        # Parse the date using the custom function
        parsed_date = parse_date(row['LBDTC'])
        
        # Append the parsed date and corresponding value to the lists
        parsed_dates.append(parsed_date)
        values.append(float(row['LBORRES']))

    # Now 'parsed_dates' and 'values' contain your data
    # print(parsed_dates)
    # print(values)
    index = pd.DatetimeIndex(parsed_dates)
    series = pd.Series(values, index=index)
    
    return series
