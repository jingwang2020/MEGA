# -*- coding: utf-8 -*-
# @Time    : 2021/10/23 16:39
# @Author  : SSK
# @Email   : skshao@bjtu.edu.cn
# @File    : emd_experiment.py
# @Software: PyCharm

# Reference Documents：https://bitbucket.org/luukko/pyeemd/src/master/examples/eemd_example.py

from pyeemd import eemd
from pylab import plot, show, title, figure
import sys
sys.path.append('/data/Zhongchao/MEGA')
from Visualize.wavelet_experiment import smd_test_data_read, smd_train_data_read
import numpy as np

def time_series_emd(machine_id='1-1', test_data=True, output_folder="./OutputHtml/"):
    if test_data:
        test_data, test_label = smd_test_data_read(machine_id)
        print("test_data.shape:", test_data.shape)
        print("test_label.shape:", test_label.shape)
        # Intercept abnormal fragments related to several selected sequences
        test_data = test_data[13710:20248, :]
        emd_test_data = test_data[:, 22].reshape(-1)
        print("time series.shape:", emd_test_data.shape)

        imfs = eemd(emd_test_data, num_siftings=10)
        print(imfs.shape)
        figure()
        title("Original signal")
        plot([i for i in range(len(emd_test_data))], emd_test_data)

        highfreq_sum = np.sum([imfs[i] for i in range(0, 4)], axis=0)
        midfreq_sum = np.sum([imfs[i] for i in range(4, 8)], axis=0)
        lwo_sum = np.sum([imfs[i] for i in range(8, imfs.shape[0])], axis=0)

        freq = [highfreq_sum, midfreq_sum, lwo_sum]
        for i in range(3):
            figure()
            title("imfs:")
            plot([i for i in range(imfs.shape[1])], freq[i])
    else:
        train_data = smd_train_data_read(machine_id)
        print("test_data.shape:", train_data.shape)
        train_data = train_data[10114:15919, :]
        emd_train_data = train_data[:, 22].reshape(-1)
        print("time series.shape:", emd_train_data.shape)

        imfs = eemd(emd_train_data, num_siftings=10)
        print(imfs.shape)
        figure()
        title("Original signal")
        plot([i for i in range(len(emd_train_data))], emd_train_data)

        highfreq_sum = np.sum([imfs[i] for i in range(0, 4)], axis=0)
        midfreq_sum = np.sum([imfs[i] for i in range(4, 8)], axis=0)
        lwo_sum = np.sum([imfs[i] for i in range(8, imfs.shape[0])], axis=0)

        freq = [highfreq_sum, midfreq_sum, lwo_sum]
        for i in range(3):
            figure()
            title("imfs:")
            plot([i for i in range(imfs.shape[1])], freq[i])

    show()


if __name__ == '__main__':
    time_series_emd(machine_id='2-5', test_data=False, output_folder="./OutputHtml/")
