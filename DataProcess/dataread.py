# -*- coding: utf-8 -*-
# @Time    : 2021/8/12 16:39
# @Author  : SSK
# @Email   : skshao@bjtu.edu.cn
# @File    : dataread.py
# @Software: PyCharm
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
import pandas as pd

# dataset path
prefix = "./data/nasa-smd_processed/"


def swat_multi_get_data(dataset):
    """
        get data from csv files
    """
    # Normal
    normal = pd.read_csv("data/swat/SWaT_Dataset_Normal_v1.csv")  # , nrows=1000)
    normal = normal.drop(["Timestamp", "Normal/Attack"], axis=1)

    # Transform all columns into float64
    for i in list(normal):
        normal[i] = normal[i].apply(lambda x: str(x).replace(",", "."))
    normal = normal.astype(float)

    # Normalization
    min_max_scaler = MinMaxScaler()
    x = normal.values
    x_scaled = min_max_scaler.fit_transform(x)
    normal = pd.DataFrame(x_scaled)

    # Attack
    # Read data
    attack = pd.read_csv("data/swat/SWaT_Dataset_Attack_v0.csv", sep=";")  # , nrows=1000)
    labels = [float(label != 'Normal') for label in attack["Normal/Attack"].values]
    attack = attack.drop(["Timestamp", "Normal/Attack"], axis=1)

    # Transform all columns into float64
    for i in list(attack):
        attack[i] = attack[i].apply(lambda x: str(x).replace(",", "."))
    attack = attack.astype(float)

    # Normalization
    x = attack.values
    x_scaled = min_max_scaler.transform(x)
    attack = pd.DataFrame(x_scaled)
    print("----------------------------------------------------------------")
    print("Data Read Success!")
    return (normal.values, None), (attack.values, np.array(labels))


def multi_get_data(dataset, max_train_size=None, max_test_size=None, print_log=True, do_preprocess=True, train_start=0,
             test_start=0):
    """
        get data from pkl files
        return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    """
    if max_train_size is None:
        train_end = None
    else:
        train_end = train_start + max_train_size
    if max_test_size is None:
        test_end = None
    else:
        test_end = test_start + max_test_size
    print('load data of:', dataset)
    print("train: ", train_start, train_end)
    print("test: ", test_start, test_end)
    x_dim = multi_get_data_dim(dataset)
    f = open(os.path.join(prefix, dataset + '_train.pkl'), "rb")
    train_data = pickle.load(f).reshape((-1, x_dim))[train_start:train_end, :]
    f.close()
    try:
        f = open(os.path.join(prefix, dataset + '_test.pkl'), "rb")
        test_data = pickle.load(f).reshape((-1, x_dim))[test_start:test_end, :]
        f.close()
    except (KeyError, FileNotFoundError):
        test_data = None
    try:
        f = open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb")
        test_label = pickle.load(f).reshape((-1))[test_start:test_end]
        f.close()
    except (KeyError, FileNotFoundError):
        test_label = None
    if do_preprocess:
        train_data = multi_preprocess(train_data)
        test_data = multi_preprocess(test_data)
    print("----------------------------------------------------------------")
    print("train set shape: ", train_data.shape)
    print("test set shape: ", test_data.shape)
    print("test set label shape: ", test_label.shape)
    print("Data Read Success!")
    print("----------------------------------------------------------------")
    return (train_data, None), (test_data, test_label)


def multi_get_data_dim(dataset):
    """
        Return data dimension
    """
    if dataset == 'SMAP':
        return 25
    elif dataset == 'MSL':
        return 55
    elif str(dataset).startswith('machine'):
        return 38
    else:
        raise ValueError('unknown dataset '+str(dataset))


def multi_preprocess(df):
    """
        returns normalized and standardized data.
    """

    df = np.asarray(df, dtype=np.float32)

    if len(df.shape) == 1:
        raise ValueError('Data must be a 2-D array')

    if np.any(sum(np.isnan(df)) != 0):
        print('Data contains null values. Will be replaced with 0')
        df = np.nan_to_num()

    # normalize data
    df = MinMaxScaler().fit_transform(df)
    print('Data normalized')

    return df