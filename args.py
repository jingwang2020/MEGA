# -*- coding: utf-8 -*-
# @Time    : 2021/11/24 16:51
# @Author  : SSK
# @Email   : skshao@bjtu.edu.cn
# @File    : args.py
# @Software: PyCharm
import argparse
import warnings
import random
import torch
import numpy as np


def setup_seed(seed):
    warnings.warn(f'You have chosen to seed ({seed}) training. This will turn on the CUDNN deterministic setting, '
                  f'which can slow down your training considerably! You may see unexpected behavior when restarting '
                  f'from checkpoints.')

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args(verbose=True):
    parser = argparse.ArgumentParser(description='ASTMAD: Adversarial ST-GCN based Multivariate Time Series Anomaly Detection')

    # Dataset
    group_dataset = parser.add_argument_group('Dataset')
    group_dataset.add_argument("--data-category", dest='data_category',
                               choices=['MSL', 'SMAP', 'Swat', 'machine-1-1', 'machine-1-2',
                                        'machine-1-3', 'machine-1-4', 'machine-1-5',
                                        'machine-1-6', 'machine-1-7', 'machine-1-8',
                                        'machine-2-1', 'machine-2-2', 'machine-2-3',
                                        'machine-2-4', 'machine-2-5', 'machine-2-6',
                                        'machine-2-7', 'machine-2-8', 'machine-2-9',
                                        'machine-3-1', 'machine-3-2', 'machine-3-3',
                                        'machine-3-4', 'machine-3-5', 'machine-3-6',
                                        'machine-3-7', 'machine-3-8', 'machine-3-9',
                                        'machine-3-10', 'machine-3-11'], type=str, default='MSL')
    group_dataset.add_argument("--val-split", dest='val_split', type=float, default=0.1)
    group_dataset.add_argument("--shuffle-dataset", dest='shuffle_dataset', action='store_true')


    # Model
    group_model = parser.add_argument_group('Model')
    group_model.add_argument("--print-model", dest='print_model', action='store_true')
    group_model.add_argument("--window", dest='window_size', type=int, default=128)
    group_model.add_argument("--hidden", dest='hidden_size', type=int, default=100)
    group_model.add_argument("--latent", dest='latent_size', type=int, default=16)
    group_model.add_argument("--top-k", dest='top_k', type=int, default=20)
    group_model.add_argument("--embedding-dim", dest='embedding_dim', type=int, default=64)
    group_model.add_argument("--gc-depth", dest='gc_depth', type=int, default=1)

    # Discriminator
    group_model.add_argument("--use-dis", dest='use_dis', action='store_true')
    group_model.add_argument("--a", dest='a', type=float, default=3.0)
    group_model.add_argument("--b", dest='b', type=float, default=3.0)
    group_model.add_argument("--c", dest='c', type=float, default=3.0)
    group_model.add_argument("--d", dest='d', type=float, default=3.0)
    group_model.add_argument("--e", dest='e', type=float, default=3.0)

    # Save and load
    group_save_load = parser.add_argument_group('Save and Load')
    group_save_load.add_argument("--resume", dest='resume', action='store_true')
    group_save_load.add_argument("--load-path", type=str, default=None)
    group_save_load.add_argument("--save-path", dest='save_path', type=str, default='./cache/uncategorized/')
    group_save_load.add_argument("--interval", dest='save_interval', type=int, default=10)
    # Whether save the data after the training is completed for outputting graphic visualization
    group_save_load.add_argument("--save-plot-data", dest='save_plot_data', action='store_true')

    # Devices
    group_device = parser.add_argument_group('Device')
    group_device.add_argument("--ngpu", dest='num_gpu', help="The number of gpu to use", default=0, type=int)
    group_device.add_argument("--seed", dest='seed', type=int, default=2021, help="The random seed")
    group_device.add_argument("--device", dest='device', help="device to use", default='cuda:3', type=str)

    # Training
    group_training = parser.add_argument_group('Training')
    group_training.add_argument("--epochs", dest="epochs", type=int, default=1, help="The number of epochs to run")
    group_training.add_argument("--batch", dest="batch_size", type=int, default=200, help="The batch size")
    group_model.add_argument("--model-lr", dest='model_lr', type=float, default=1e-3)
    group_model.add_argument("--critic", dest='critic_iter', type=int, default=2)

    # Detection
    group_detection = parser.add_argument_group('Detection')

    args_parsed = parser.parse_args()

    # Display parser information
    if verbose:
        message = ''
        message += '-------------------------------- Args ------------------------------\n'
        for k, v in sorted(vars(args_parsed).items()):
            comment = ''
            default = parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>35}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '-------------------------------- End ----------------------------------'
        print(message)
    return args_parsed