# -*- coding: utf-8 -*-
# @Time    : 2021/10/13 16:52
# @Author  : SSK
# @Email   : skshao@bjtu.edu.cn
# @File    : wavelet_experiment.py
# @Software: PyCharm
import os
import pickle
from plotly import subplots
import matplotlib.pyplot as plt
from plotly.graph_objs import *
import plotly.offline
from pytorch_wavelets import DWT1DForward, DWT1DInverse
import numpy as np
import torch
import sys
sys.path.append('/data/Zhongchao/MEGA')
from DataProcess.dataread import multi_preprocess, multi_get_data_dim


def smd_test_data_read(machine_id='1-1'):
    """
    :param machine_id: machine ID
    :return: preprocessed multivariate time series and labels
    """
    x_dim = multi_get_data_dim('machine-' + machine_id)
    try:
        f = open(os.path.join("../data/nasa-smd_processed/", 'machine-' + machine_id +'_test.pkl'), "rb")
        test_data = pickle.load(f).reshape((-1, x_dim))[0:None, :]
        f.close()
    except (KeyError, FileNotFoundError):
        test_data = None
    try:
        f = open(os.path.join("../data/nasa-smd_processed/", 'machine-' + machine_id + "_test_label.pkl"), "rb")
        test_label = pickle.load(f).reshape((-1))[0:None]
        f.close()
    except (KeyError, FileNotFoundError):
        test_label = None
    test_data = multi_preprocess(test_data)

    return test_data, test_label


def smd_train_data_read(machine_id='1-1'):
    """
    :param machine_id: machine ID
    :return: The data corresponding to the machine ID training set
    """
    x_dim = multi_get_data_dim('machine-' + machine_id)
    f = open(os.path.join("../data/nasa-smd_processed/", 'machine-' + machine_id + '_train.pkl'), "rb")
    train_data = pickle.load(f).reshape((-1, x_dim))[0:None, :]
    f.close()
    train_data = multi_preprocess(train_data)
    print("train set shape: ", train_data.shape)
    return train_data


def smd_offline_plot(test_data, test_label, machine_id='1-1', output_path="./OutputHtml/", output_file_name="smd_5metric_seg.html"):
    """
    Draw the global data and labels of the smd dataset, draw the training set or test set, the test set has labeled data, and the test set has no labeled data
    :param test_data: Preprocessed test_data
    :param test_label: Preprocessed test_label
    :param machine_id: machine ID
    :param output_path:
    :param output_file_name:
    :return:
    """
    fig = subplots.make_subplots(rows=5, cols=1, shared_xaxes=True)
    plot_data2 = Scatter(
        x=[i for i in range(test_data.shape[0])],
        y=test_data[:, 6],
    )
    plot_data4 = Scatter(
        x=[i for i in range(test_data.shape[0])],
        y=test_data[:, 19],
    )
    plot_data5 = Scatter(
        x=[i for i in range(test_data.shape[0])],
        y=test_data[:, 21],
    )
    plot_data6 = Scatter(
        x=[i for i in range(test_data.shape[0])],
        y=test_data[:, 22],
    )
    plot_data8 = Scatter(
        x=[i for i in range(test_data.shape[0])],
        y=test_data[:, 34],
    )
    if test_label is not None:
        plot_label = Scatter(
            x=[i for i in range(len(test_label))],
            y=test_label,
            # mode='markers',
            fillcolor='red'
        )
        fig.append_trace(plot_label, 1, 1)
    fig.append_trace(plot_data2, 1, 1)
    fig.append_trace(plot_data4, 2, 1)
    fig.append_trace(plot_data5, 3, 1)
    fig.append_trace(plot_data6, 4, 1)
    fig.append_trace(plot_data8, 5, 1)

    with plt.style.context(['science']):
        plotly.offline.plot(fig,
                            filename=output_path + "machine" + machine_id + output_file_name)


def msl_smap_test_data_read(dataset='MSL'):
    """
        :param dataset: MSL or SMAP
        :return: Preprocessed test data and labels
        """
    x_dim = multi_get_data_dim(dataset)
    try:
        f = open(os.path.join("../data/nasa-smd_processed/", dataset + '_test.pkl'), "rb")
        test_data = pickle.load(f).reshape((-1, x_dim))[0:None, :]
        f.close()
    except (KeyError, FileNotFoundError):
        test_data = None
    try:
        f = open(os.path.join("../data/nasa-smd_processed/", dataset + "_test_label.pkl"), "rb")
        test_label = pickle.load(f).reshape((-1))[0:None]
        f.close()
    except (KeyError, FileNotFoundError):
        test_label = None
    test_data = multi_preprocess(test_data)

    return test_data, test_label


def msl_smap_offline_plot(dataset='MSL', output_folder="./OutputHtml/", one_dim=True):
    test_data, test_label = msl_smap_test_data_read(dataset)
    test_label = test_label+0
    if dataset == "MSL":
        if one_dim == True:
            fig = subplots.make_subplots(1, cols=1, shared_xaxes=True)
            plot_series = Scatter(
                x=[i for i in range(len(test_label))],
                y=test_data[0:37000, 0],
            )
            fig.append_trace(plot_series, 1, 1)
            with plt.style.context(['science']):
                plotly.offline.plot(fig, filename=output_folder + "MSL_One_Dim.html")
        else:
            fig = subplots.make_subplots(11, cols=1, shared_xaxes=True)
            for i in range(11):
                plot_series = Scatter(
                    x=[i for i in range(len(test_label))],
                    y=test_data[:,i],
                    # name='data1'
                )
                fig.append_trace(plot_series, i+1, 1)
            plot_series_label = Scatter(
                x=[i for i in range(len(test_label))],
                y=test_label,
                # name='data1'
            )
            fig.append_trace(plot_series_label, 1, 1)
            with plt.style.context(['science']):
                plotly.offline.plot(fig, filename=output_folder + "MSL_1.html")

            fig2 = subplots.make_subplots(11, cols=1, shared_xaxes=True)
            for i in range(11, 22):
                plot_series = Scatter(
                    x=[i for i in range(len(test_label))],
                    y=test_data[:, i],
                    # name='data1'
                )
                fig2.append_trace(plot_series, i-10, 1)
            fig2.append_trace(plot_series_label, 1, 1)
            with plt.style.context(['science']):
                plotly.offline.plot(fig2, filename=output_folder + "MSL_2.html")

            fig3 = subplots.make_subplots(11, cols=1, shared_xaxes=True)
            for i in range(22, 33):
                plot_series = Scatter(
                    x=[i for i in range(len(test_label))],
                    y=test_data[:, i],
                    # name='data1'
                )
                fig3.append_trace(plot_series, i - 21, 1)
            fig3.append_trace(plot_series_label, 1, 1)
            with plt.style.context(['science']):
                plotly.offline.plot(fig3, filename=output_folder + "MSL_3.html")

            fig4 = subplots.make_subplots(11, cols=1, shared_xaxes=True)
            for i in range(33, 44):
                plot_series = Scatter(
                    x=[i for i in range(len(test_label))],
                    y=test_data[:, i],
                    # name='data1'
                )
                fig4.append_trace(plot_series, i - 32, 1)
            fig4.append_trace(plot_series_label, 1, 1)
            with plt.style.context(['science']):
                plotly.offline.plot(fig4, filename=output_folder + "MSL_4.html")

            fig5 = subplots.make_subplots(11, cols=1, shared_xaxes=True)
            for i in range(44, 55):
                plot_series = Scatter(
                    x=[i for i in range(len(test_label))],
                    y=test_data[:, i],
                    # name='data1'
                )
                fig5.append_trace(plot_series, i - 43, 1)
            fig5.append_trace(plot_series_label, 1, 1)
            with plt.style.context(['science']):
                plotly.offline.plot(fig5, filename=output_folder + "MSL_5.html")
    else:
        if one_dim == True:
            fig = subplots.make_subplots(1, cols=1, shared_xaxes=True)
            plot_series = Scatter(
                x=[i for i in range(len(test_label))],
                y=test_data[354500:360000, 0],
            )
            fig.append_trace(plot_series, 1, 1)
            with plt.style.context(['science']):
                plotly.offline.plot(fig, filename=output_folder + "SMAP_One_Dim.html")
        else:
            fig = subplots.make_subplots(12, cols=1, shared_xaxes=True)
            for i in range(12):
                plot_series = Scatter(
                    x=[i for i in range(len(test_label))],
                    y=test_data[:, i],
                    # name='data1'
                )
                fig.append_trace(plot_series, i + 1, 1)
            plot_series_label = Scatter(
                x=[i for i in range(len(test_label))],
                y=test_label,
                # name='data1'
            )
            fig.append_trace(plot_series_label, 1, 1)
            with plt.style.context(['science']):
                plotly.offline.plot(fig, filename=output_folder + "SMAP_1.html")

            fig2 = subplots.make_subplots(13, cols=1, shared_xaxes=True)
            for i in range(12, 25):
                plot_series = Scatter(
                    x=[i for i in range(len(test_label))],
                    y=test_data[:, i],
                    # name='data1'
                )
                fig2.append_trace(plot_series, i - 11, 1)
            fig2.append_trace(plot_series_label, 1, 1)
            with plt.style.context(['science']):
                plotly.offline.plot(fig2, filename=output_folder + "SMAP_2.html")


def wavelet_decompose_virtual(wave, machine_id, yl, yh, decomposition_level, low_iteration, output_folder="./OutputHtml/"):
    """
    :param machine_id: machine ID
    :param yl: Decomposed low frequency components
    :param yh: Decomposed high-frequency components，list
    :param decomposition_level: length of yh
    :param low_iteration: How many decompositions have been iterated on yl in total
    :param output_folder: Folder for output images
    :return:
    """
    fig = subplots.make_subplots(decomposition_level + 1, cols=1, shared_xaxes=True)

    plot_wave_low = Scatter(
        x=[i for i in range(len(yl))],
        y=yl,
        # name='data1'
    )
    fig.append_trace(plot_wave_low, 1, 1)

    for i in range(decomposition_level):
        plot_wave_high = Scatter(
            x=[i for i in range(len(yl))],
            y=yh[i],
            # name='data1'
        )
        fig.append_trace(plot_wave_high, i + 2, 1)

    with plt.style.context(['science']):
        plotly.offline.plot(fig, filename=output_folder + "machine" + machine_id + "wave_" + wave
                                          + '_' +str(decomposition_level) + "decompose_iteration" + str(
            low_iteration) + "_symmetric.html")


def smd_wavelet_transform(machine_id='1-1', wave='haar', decomposition_level=3, low_iteration=1, output_folder="./OutputHtml/"):
    """
    :param machine_id: machine ID
    :param wave: The type of wavelet basis used，'haar' 'db1' 'db2'
    :param decomposition_level: The number of high-frequency components decomposed
    :param low_iteration: The decomposed low-frequency components are iteratively decomposed several times
    :return:
    """
    test_data, test_label = smd_test_data_read(machine_id)
    print("test_data.shape:", test_data.shape)
    print("test_label.shape:", test_label.shape)

    # Intercept abnormal fragments related to several selected sequences
    test_data = test_data[13710:20248, :]
    test_label = test_label[13710:20248]
    # Before performing pytorch_wavelet, you need to convert numpy data into tensor
    wave_test_data = test_data[:,22]
    wave_test_data = torch.from_numpy(wave_test_data)
    print("time series.shape:", wave_test_data.shape)

    wave_test_data = wave_test_data.unsqueeze(0)
    wave_test_data = wave_test_data.unsqueeze(0)
    print("after extend dim test_data.shape:", wave_test_data.shape)

    for i in range(low_iteration):
        dwt = DWT1DForward(J=decomposition_level, wave=wave, mode='zero')
        if i==0:
            yl, yh = dwt(wave_test_data)
        else:
            yl, yh = dwt(yl)

    yl = yl.squeeze()
    yl = yl.numpy()
    for i in range(decomposition_level):
        yh[i] = yh[i].squeeze()
        yh[i] = yh[i].numpy()
    print(yl.shape)
    print(len(yh))

    wavelet_decompose_virtual(wave, machine_id, yl, yh, decomposition_level, low_iteration, output_folder)


if __name__ == '__main__':
    ####################################################
    # Draw the smd test set image
    test_data, test_label = smd_test_data_read(machine_id = '2-5')
    # smd_offline_plot(test_data, test_label, machine_id, "./OutputHtml/", "smd_5metric_seg2.html")
    # Draw the smd training set image
    # train_data = smd_train_data_read('2-5')
    # smd_offline_plot(train_data, None, '2-5', "./OutputHtml/", "smd_5metric_train_seg.html")
    ####################################################
    # Draw wavelet transformed smd image
    smd_wavelet_transform('2-5', wave='db1', decomposition_level=4, low_iteration=1)
    ####################################################
    # draw msl image
    # msl_smap_offline_plot(dataset='MSL', output_folder="./OutputHtml/", one_dim=True)
    ####################################################
    # draw smap image
    # msl_smap_offline_plot(dataset='SMAP', output_folder="./OutputHtml/", one_dim=True)
    ####################################################
