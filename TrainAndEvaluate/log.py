# -*- coding: utf-8 -*-
# @Time    : 2021/8/19 10:11
# @Author  : SSK
# @Email   : skshao@bjtu.edu.cn
# @File    : log.py
# @Software: PyCharm
import os
import numpy as np


class Log(object):
    """Logger class to log training metadata.

    Args:
        log_file_path (type): Log file name.
        op (type): Read or write.

    Examples
        Examples should be written in doctest format, and
        should illustrate how to use the function/class.
        >>>

    Attributes:
        log (type): Description of parameter `log`.
        op

    """
    def __init__(self, log_file_path, log_file, op='r'):
        if not os.path.exists(log_file_path):
            os.mkdir(log_file_path)
        self.log = open(log_file_path+log_file, op)
        self.op = op

    def write_model(self, model):
        self.log.write('\n##MODEL START##\n')
        self.log.write(model)
        self.log.write('\n##MODEL END##\n')

        self.log.write('\n##MODEL SIZE##\n')
        self.log.write(str(sum(p.numel() for p in model.parameters())))
        self.log.write('\n##MODEL SIZE##\n')

    def log_train_metrics(self, metrics, epoch):
        self.log.write('\n##TRAIN METRICS##\n')
        self.log.write('@epoch:' + str(epoch) + '\n')
        for k, v in metrics.items():
            self.log.write(k + '=' + str(v) + '\n')
        self.log.write('\n##TRAIN METRICS##\n')

    def log_val_metrics(self, metrics, epoch):
        self.log.write('\n##VAL METRICS##\n')
        self.log.write('@epoch:' + str(epoch) + '\n')
        for k, v in metrics.items():
            self.log.write(k + '=' + str(v) + '\n')
        self.log.write('\n##VAL METRICS##\n')

    def log_test_metrics(self, metrics, epoch):
        self.log.write('\n##TEST METRICS##\n')
        self.log.write('@epoch:' + str(epoch) + '\n')

        for k, v in metrics.items():
            self.log.write(k + '=' + str(v) + '\n')
        self.log.write('\n##TEST METRICS##\n')

    def close(self):
        self.log.close()


def initialize_train_metrics():

    metrics = {
        'rec_loss': []
    }

    return metrics


def get_mean_metrics(metrics_dict):

    return {k: np.mean(v) for k, v in metrics_dict.items()}


def set_train_metrics(metrics_dict, loss):

    metrics_dict['rec_loss'].append(loss.item())

    return metrics_dict


def set_val_metrics(metrics_dict, loss):

    metrics_dict['rec_loss'].append(loss.item())

    return metrics_dict


def initialize_test_metrics():
    metrics = {

    }

    return metrics


def initialize_val_metrics():
    metrics = {
        'rec_loss': []
    }

    return metrics


def set_test_metrics(metrics_dict, performance):
    # metrics_dict['PR_ORI'] = performance['PR_ORI'][0]
    # metrics_dict['RECALL_ORI'] = performance['REC_ORI'][0]
    # metrics_dict['F1_ORI'] = performance['F1_ORI'][0]
    metrics_dict['PR'] = performance['PR'][0]
    metrics_dict['RECALL'] = performance['REC'][0]
    metrics_dict['F1'] = performance['F1'][0]

    return metrics_dict
