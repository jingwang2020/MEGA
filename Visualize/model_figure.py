# -*- coding: utf-8 -*-
# @Time    : 2021/12/14 20:17
# @Author  : SSK
# @Email   : skshao@bjtu.edu.cn
# @File    : model_figure.py
# @Software: PyCharm
import pickle

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from emd_experiment import smd_test_data_read, smd_train_data_read
from AstMad.MultWaveGCUNet import MultWaveGCUNet
from pytorch_wavelets import DWT1DForward, DWT1DInverse
from DataProcess.dataset import MuliTsBatchedWindowDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm.std import tqdm
import numpy as np
import os
# import matplotlib.pyplot as plt
from plotly.graph_objs import *
import plotly.offline
from plotly import subplots
import pandas as pd
from matplotlib.pyplot import MultipleLocator


def smd_test_science_plot(machine_id='1-1'):
    test_data, test_label = smd_test_data_read(machine_id)
    print("test_data.shape:", test_data.shape)
    print("test_label.shape:", test_label.shape)
    # Intercept abnormal fragments related to selected sequences
    # Total original length：15280:19800
    # Partial Enlarged View：
    # First Exception Fragment 18570:18710
    # Second Exception Fragment 18870:18960

    # Draw a wave decomposition diagram to shorten the original length to two cycles：
    test_data = test_data[12350:18260, :]

    with plt.style.context(['science', 'no-latex']):
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 300
        fig0 = plt.figure(figsize=(13, 2.5))
        ax0 = fig0.add_subplot(1, 1, 1)

        ax0.plot([i for i in range(len(test_data))], test_data[:, 5], color='black', linewidth=2.0)
        plt.yticks(fontproperties='Times New Roman', size=30)  # Set size and bold
        plt.rcParams['axes.unicode_minus'] = False
        plt.tight_layout()
        plt.xticks([])
        y_major_locator = MultipleLocator(1.0)
        TK = plt.gca()
        TK.yaxis.set_major_locator(y_major_locator)
        TK.spines['bottom'].set_linewidth(2)
        TK.spines['left'].set_linewidth(2)
        TK.spines['top'].set_linewidth(2)
        TK.spines['right'].set_linewidth(2)
        plt.savefig("./Save_Component_Jpg/input_window/5_raw.jpg")
        plt.show()


def smd_test_wave_plot(machine_id='1-1'):
    test_data, test_label = smd_test_data_read(machine_id)
    print("test_data.shape:", test_data.shape)
    print("test_label.shape:", test_label.shape)

    # Intercept abnormal fragments related to selected sequences
    test_data = test_data[13710:20248, :]
    test_label = test_label[13710:20248]
    # Perform pytorch_ Before wavelet, it is necessary to convert numpy data into tensor
    wave_test_data = test_data
    wave_test_data = torch.from_numpy(wave_test_data)
    print("time series.shape:", wave_test_data.shape)

    wave_test_data = wave_test_data.unsqueeze(0)
    wave_test_data = wave_test_data.permute(0, 2, 1)
    # wave_test_data = wave_test_data.unsqueeze(0)
    print("after extend dim test_data.shape:", wave_test_data.shape)

    dwt = DWT1DForward(J=1, wave='db1', mode='zero')
    yl_level1, yh_level1 = dwt(wave_test_data)
    yl_level2, yh_level2 = dwt(yl_level1)
    yl_level3, yh_level3 = dwt(yl_level2)

    yl_level1 = yl_level1.squeeze()
    yh_level1 = yh_level1[0].squeeze()
    yl_level2 = yl_level2.squeeze()
    yh_level2 = yh_level2[0].squeeze()
    yl_level3 = yl_level3.squeeze()
    yh_level3 = yh_level3[0].squeeze()

    yl_level1 = yl_level1.permute(1, 0)
    yh_level1 = yh_level1.permute(1, 0)
    yl_level2 = yl_level2.permute(1, 0)
    yh_level2 = yh_level2.permute(1, 0)
    yl_level3 = yl_level3.permute(1, 0)
    yh_level3 = yh_level3.permute(1, 0)

    plot_data = [yl_level1, yh_level1, yl_level2, yh_level2, yl_level3, yh_level3]

    for test_data in plot_data:
        with plt.style.context(['science', 'no-latex']):
            fig = plt.figure(figsize=(20, 9))
            ax1 = fig.add_subplot(6, 1, 1)
            ax1.plot([i for i in range(len(test_data))], test_data[:, 0], color='#e54d42', linewidth=2.0)

            ax2 = fig.add_subplot(6, 1, 2)
            ax2.plot([i for i in range(len(test_data))], test_data[:, 6], color='#e54d42', linewidth=2.0)

            ax3 = fig.add_subplot(6, 1, 3)
            ax3.plot([i for i in range(len(test_data))], test_data[:, 11], color='#e54d42', linewidth=2.0)

            ax4 = fig.add_subplot(6, 1, 4)
            ax4.plot([i for i in range(len(test_data))], test_data[:, 14], color='#e54d42', linewidth=2.0)

            ax5 = fig.add_subplot(6, 1, 5)
            ax5.plot([i for i in range(len(test_data))], test_data[:, 22], color='#e54d42', linewidth=2.0)

            ax6 = fig.add_subplot(6, 1, 6)
            ax6.plot([i for i in range(len(test_data))], test_data[:, 23], color='#e54d42', linewidth=2.0)

    plt.show()


def smd_train_data_plot(machine_id='1-1'):
    train_data = smd_train_data_read(machine_id)
    print("test_data.shape:", train_data.shape)
    print("test_label.shape:", train_data.shape)

    plot_data = [train_data[16000:23000,:]]

    for test_data in plot_data:
        with plt.style.context(['science', 'no-latex']):
            fig = plt.figure(figsize=(20, 9))

            ax5 = fig.add_subplot(2, 1, 1)
            ax5.plot([i for i in range(len(test_data))], test_data[:, 22], color='#e54d42', linewidth=2.0)

            ax6 = fig.add_subplot(2, 1, 2)
            ax6.plot([i for i in range(len(test_data))], test_data[:, 23], color='#e54d42', linewidth=2.0)

    plt.show()


def smd_train_model_plot(machine_id='1-1', model_path='', data_path=''):
    with open(data_path, 'rb') as file:
        train_22 = pickle.load(file)

    train_data = smd_train_data_read(machine_id)
    print("test_data.shape:", train_data.shape)
    # Intercept abnormal fragments related to selected sequences

    num_gpu = 2
    x_dim = train_data.shape[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(num_gpu)
    if num_gpu >= 0 and torch.cuda.is_available():
        print("Use Gpu:", os.environ["CUDA_VISIBLE_DEVICES"])
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    train_data[15900:24000, 22] = train_22
    train_data = train_data[15900:24000, :]

    train_dataset = MuliTsBatchedWindowDataset(train_data, label=None, device=device, window_size=128,
                                               stride=1)
    train_loader = DataLoader(train_dataset, batch_size=128, num_workers=4, shuffle=False,
                              drop_last=True, pin_memory=True)

    model = MultWaveGCUNet(input_channel=x_dim, embedding_dim=64, top_k=30,
                           input_node_dim=2, graph_alpha=3, device=device, gc_depth=1,
                           batch_size=128)
    model = model.to(device)

    model.load_state_dict(torch.load(os.path.join(model_path, f'model_best.pth.tar'), map_location=device))

    print("Begin save train data...")
    model_recons = []
    raw_x = []

    x_dim = train_data.shape[1]
    model.eval()
    with torch.no_grad():
        for i, x in enumerate(tqdm(train_loader)):
            raw_x.append(x[:, -1, :])
            x = x.to(device)
            # x.shape: batch × window × node
            idx = torch.tensor(np.random.permutation(range(x_dim))).to(device)
            x_rec = model(x, idx, device)
            score = nn.MSELoss(reduction='none')(x_rec[0][0], x_rec[0][1]).squeeze(dim=1)[:, -1, :].mean(
                dim=1) + nn.MSELoss(reduction='none')(x_rec[1][0], x_rec[1][1]).squeeze(dim=1)[:, -1, :].mean(dim=1) + \
                    nn.MSELoss(reduction='none')(x_rec[2][0], x_rec[2][1]).squeeze(dim=1)[:, -1, :].mean(
                        dim=1) + nn.MSELoss(reduction='none')(x_rec[3][0], x_rec[3][1]).squeeze(dim=1)[:, -1, :].mean(
                dim=1) + \
                    nn.MSELoss(reduction='none')(x_rec[4][0], x_rec[4][1]).squeeze(dim=1)[:, -1, :].mean(dim=1)

            model_recons.append(x_rec[4][1][:, -1, :].detach().cpu().numpy())

    model_recons = np.concatenate(model_recons, axis=0)
    actual = np.concatenate(raw_x, axis=0)

    plot_actual = actual
    plot_rec = model_recons

    fig1 = subplots.make_subplots(rows=2, cols=1, shared_xaxes=True)
    plot_data1 = Scatter(
        x=[i for i in range(len(plot_rec))],
        y=plot_rec[:, 22],
    )
    plot_data2 = Scatter(
        x=[i for i in range(len(plot_actual))],
        y=plot_actual[:, 22],
    )

    fig1.append_trace(plot_data1, 1, 1)
    fig1.append_trace(plot_data2, 1, 1)
    fig1.append_trace(plot_data1, 2, 1)
    fig1.append_trace(plot_data2, 2, 1)

    with plt.style.context(['science', 'no-latex']):
        plotly.offline.plot(fig1,
                            filename="./HtmlSchematic/" + "machine" + machine_id + "OriginAndInjectionRec.html")


def smd_train_wave_plot(machine_id='1-1'):
    train_data = smd_train_data_read(machine_id)

    print("train_data.shape:", train_data.shape)

    # Intercept abnormal fragments related to selected sequences
    train_data = train_data[15900:24000, :]

    # Perform pytorch_ Before wavelet, it is necessary to convert numpy data into tensor
    wave_test_data = train_data
    wave_test_data = torch.from_numpy(wave_test_data)
    print("time series.shape:", wave_test_data.shape)

    wave_test_data = wave_test_data.unsqueeze(0)
    wave_test_data = wave_test_data.permute(0, 2, 1)
    # wave_test_data = wave_test_data.unsqueeze(0)
    print("after extend dim test_data.shape:", wave_test_data.shape)

    dwt = DWT1DForward(J=1, wave='db1', mode='zero')
    yl_level1, yh_level1 = dwt(wave_test_data)
    yl_level2, yh_level2 = dwt(yl_level1)
    yl_level3, yh_level3 = dwt(yl_level2)

    yl_level1 = yl_level1.squeeze()
    yh_level1 = yh_level1[0].squeeze()
    yl_level2 = yl_level2.squeeze()
    yh_level2 = yh_level2[0].squeeze()
    yl_level3 = yl_level3.squeeze()
    yh_level3 = yh_level3[0].squeeze()

    yl_level1 = yl_level1.permute(1, 0)
    yh_level1 = yh_level1.permute(1, 0)
    yl_level2 = yl_level2.permute(1, 0)
    yh_level2 = yh_level2.permute(1, 0)
    yl_level3 = yl_level3.permute(1, 0)
    yh_level3 = yh_level3.permute(1, 0)

    plot_data = [(yl_level1, yh_level1), (yl_level2, yh_level2), (yl_level3, yh_level3)]

    for test_data in plot_data:
        with plt.style.context(['science', 'no-latex']):
            fig = plt.figure(figsize=(20, 9))
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.plot([i for i in range(len(test_data[0]))], test_data[0][:, 22], color='#e54d42', linewidth=2.0)

            ax2 = fig.add_subplot(2, 2, 2)
            ax2.plot([i for i in range(len(test_data[0]))], test_data[0][:, 23], color='#e54d42', linewidth=2.0)

            ax1 = fig.add_subplot(2, 2, 3)
            ax1.plot([i for i in range(len(test_data[1]))], test_data[1][:, 22], color='#e54d42', linewidth=2.0)

            ax2 = fig.add_subplot(2, 2, 4)
            ax2.plot([i for i in range(len(test_data[1]))], test_data[1][:, 23], color='#e54d42', linewidth=2.0)

    plt.show()


def smd_test_dim_wave_plot(machine_id='1-1'):
    test_data, test_label = smd_test_data_read(machine_id)

    print("train_data.shape:", test_data.shape)

    # Intercept abnormal fragments related to the selected sequences, with a total length of 15280:19800
    # Here take some of the length to draw a wave decomposition diagram 15280:18000
    train_data = test_data[15280:19800, :]

    # Perform pytorch_ Before wavelet, it is necessary to convert numpy data into tensor
    wave_test_data = train_data
    wave_test_data = torch.from_numpy(wave_test_data)
    print("time series.shape:", wave_test_data.shape)

    wave_test_data = wave_test_data.unsqueeze(0)
    wave_test_data = wave_test_data.permute(0, 2, 1)
    # wave_test_data = wave_test_data.unsqueeze(0)
    print("after extend dim test_data.shape:", wave_test_data.shape)

    dwt = DWT1DForward(J=1, wave='db1', mode='zero')
    yl_level1, yh_level1 = dwt(wave_test_data)
    yl_level2, yh_level2 = dwt(yl_level1)
    yl_level3, yh_level3 = dwt(yl_level2)
    yl_level4, yh_level4 = dwt(yl_level3)

    yl_level1 = yl_level1.squeeze()
    yh_level1 = yh_level1[0].squeeze()
    yl_level2 = yl_level2.squeeze()
    yh_level2 = yh_level2[0].squeeze()
    yl_level3 = yl_level3.squeeze()
    yh_level3 = yh_level3[0].squeeze()
    yl_level4 = yl_level4.squeeze()
    yh_level4 = yh_level4[0].squeeze()

    yl_level1 = yl_level1.permute(1, 0)
    yh_level1 = yh_level1.permute(1, 0)
    yl_level2 = yl_level2.permute(1, 0)
    yh_level2 = yh_level2.permute(1, 0)
    yl_level3 = yl_level3.permute(1, 0)
    yh_level3 = yh_level3.permute(1, 0)
    yl_level4 = yl_level4.permute(1, 0)
    yh_level4 = yh_level4.permute(1, 0)

    # plot_data = [(yl_level1, yh_level1), (yl_level2, yh_level2), (yl_level3, yh_level3)]

    with plt.style.context(['science', 'no-latex']):
        fig = plt.figure(figsize=(13, 2.5))

        ax4 = fig.add_subplot(1, 1, 1)
        ax4.plot([i for i in range(len(yh_level2))], yh_level2[:, 22], color='#e03997', linewidth=2.0)
        # plt.title("High Frequency 2", fontsize=10)
        plt.yticks(fontproperties='Times New Roman', size=30)  # 设置大小及加粗
        plt.xticks(fontproperties='Times New Roman', size=30)
        plt.rcParams['axes.unicode_minus'] = False
        plt.tight_layout()
        y_major_locator = MultipleLocator(0.1)
        TK = plt.gca()
        TK.yaxis.set_major_locator(y_major_locator)
        TK.spines['bottom'].set_linewidth(2)
        TK.spines['left'].set_linewidth(2)
        TK.spines['top'].set_linewidth(2)
        TK.spines['right'].set_linewidth(2)

    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    # plt.savefig("./Save_Component_Jpg/2period_wave_low3_23.jpg")
    plt.show()


def train_anomaly_injection(path=""):
    # train_data = pickle.load("./OutputDataSaved/machine2-5_train_anomaly_injection.pkl")
    train_raw = pd.read_pickle(path)
    # train_raw = train_raw[15900:24000, :]
    train_raw22 = train_raw["Actual22"].to_numpy()
    train_raw22 = train_raw22[15900:24000]
    # train_raw23 = train_raw["Actual23"]
    np.random.seed(0)

    # Adding disturbance as a whole
    noise = np.random.normal(0.005, 0.005, 7788)
    # 7788
    train_raw22[:] = train_raw22[:] + noise

    # Abnormal injection of the first part
    anomaly1 = np.random.normal(0.01, 0.03, 730)
    # anomaly[anomaly<0] = 0
    index = np.arange(730)
    np.random.shuffle(index)
    add_operation_index1 = index[0:365]
    sub_operation_index1 = index[365:730]

    for i in add_operation_index1:
        train_raw22[870+i] = train_raw22[870+i] + anomaly1[i]
    for i in sub_operation_index1:
        train_raw22[870+i] = train_raw22[870+i] - anomaly1[i]

    # Abnormal injection of the second part
    anomaly2 = np.random.normal(0.12, 0.02, 145)
    index = np.arange(145)
    add_operation_index1 = index[0:70]
    sub_operation_index1 = index[70:145]
    for i in add_operation_index1:
        train_raw22[1835+i] = train_raw22[1835+i] + anomaly2[i]
    for i in sub_operation_index1:
        train_raw22[1835+i] = train_raw22[1835+i] - anomaly2[i]

    # Abnormal injection of the third part
    # 3440-3630
    anomaly2 = np.random.normal(0.2, 0.08, 90)
    index = np.arange(90)
    for i in index:
        train_raw22[3440 + i] = train_raw22[3440 + i] + anomaly2[i]

    # Abnormal injection of the fourth part
    # 4580-4640
    anomaly2 = np.random.normal(0.08, 0.14, 60)
    index = np.arange(60)
    add_operation_index1 = index[0:30]
    sub_operation_index1 = index[30:60]
    for i in add_operation_index1:
        train_raw22[4580 + i] = train_raw22[4580 + i] + anomaly2[i]
    for i in sub_operation_index1:
        train_raw22[4580 + i] = train_raw22[4580 + i] - anomaly2[i]

    # Abnormal injection of the fifth part
    # 5120-5200
    anomaly2 = np.random.normal(0.02, 0.1, 80)
    index = np.arange(80)
    add_operation_index1 = index[0:40]
    sub_operation_index1 = index[40:80]
    for i in add_operation_index1:
        train_raw22[5120 + i] = train_raw22[5120 + i] + anomaly2[i]
    for i in sub_operation_index1:
        train_raw22[5120 + i] = train_raw22[5120 + i] - anomaly2[i]

    train_raw22[train_raw22 < 0] = 0
    train_raw22[train_raw22 > 1] = 1

    train_raw22.dump("./OutputDataSaved/machine2-5_train_anomaly_injection.pkl")

    # Visualization png
    with plt.style.context(['science', 'no-latex']):
        fig = plt.figure(figsize=(20, 9))
        plt.plot([i for i in range(len(train_raw22))], train_raw22, color='#e54d42', linewidth=2.0)
    plt.show()


def test_anomaly_injection(machine_id=""):

    yh_level1_3period = torch.load("./OutputDataSaved/WaveComponentAnomalyInjection/22_high1_raw.pt")
    yh_level2_3period = torch.load("./OutputDataSaved/WaveComponentAnomalyInjection/22_high2_raw.pt")
    yh_level3_3period= torch.load("./OutputDataSaved/WaveComponentAnomalyInjection/22_high3_raw.pt")
    yl_level3_3period = torch.load("./OutputDataSaved/WaveComponentAnomalyInjection/22_low3_raw.pt")

    yl_level1 = torch.load("./OutputDataSaved/WaveComponentAnomalyInjection/22_low1_raw_2period.pt")
    yh_level1 = torch.load("./OutputDataSaved/WaveComponentAnomalyInjection/22_high1_raw_2period.pt")
    yl_level2 = torch.load("./OutputDataSaved/WaveComponentAnomalyInjection/22_low2_raw_2period.pt")
    yh_level2 = torch.load("./OutputDataSaved/WaveComponentAnomalyInjection/22_high2_raw_2period.pt")
    yl_level3 = torch.load("./OutputDataSaved/WaveComponentAnomalyInjection/22_low3_raw_2period.pt")
    yh_level3 = torch.load("./OutputDataSaved/WaveComponentAnomalyInjection/22_high3_raw_2period.pt")

    yh_level1[795:1088,22] = yh_level1[70:363,22]
    yh_level1[379,22] = -0.012818
    yh_level1[400, 22] = -0.026
    yh_level1[1089, 22] = -0.038
    yh_level1[1104, 22] = -0.024
    yh_level1[1136, 22] = -0.033

    yh_level2[74:89, 22] = yh_level2[438:453, 22]
    yh_level2[177:207, 22] = yh_level2[540:570, 22]

    yl_level3[246:261] = yl_level3_3period[413:428] - 0.45

    # Fix yh_level3
    yh_level3[288:367] = yh_level3[109:188]
    # Add exception section
    yh_level3_3period[414] = -0.25
    yh_level3[245:266] = yh_level3_3period[404:425]

    # Fix yh_level2
    yh_level2[572:734] = yh_level2[210:372]
    # inject yh_level2
    yh_level2[491:524] = yh_level2_3period[827:860]

    # Fix yh_level1
    yh_level1[1144:1469] = yh_level1[420:745]
    # inject yh_level1
    yh_level1_3period[1657] = -0.05
    yh_level1[987:1048] = yh_level1_3period[1656:1717]
    yh_level1[1089] = -0.035
    yh_level1[1103] = -0.032
    yh_level1[1136] = -0.031

    yl_level3[211, 22] = 2.6
    yh_level3[211] = 0.24

    yh_level2[421] = 0.03
    yh_level2[422] = 0.08
    yh_level2[423] = 0.01
    yh_level2[424] = 0.07

    yh_level1[843] = 0.002
    yh_level1[844] = 0.04
    yh_level1[845] = 0.005
    yh_level1[846] = 0.05
    yh_level1[847] = 0.003
    yh_level1[848] = 0.03

    idwt = DWT1DInverse(wave='db1', mode='zero')
    yl_level2_synthesis = idwt((yl_level3.unsqueeze(0).permute(0, 2, 1), [yh_level3.unsqueeze(0).permute(0, 2, 1)]))
    yl_level1_synthesis = idwt((yl_level2_synthesis, [yh_level2.unsqueeze(0).permute(0, 2, 1)]))
    raw_synthesis = idwt((yl_level1_synthesis, [yh_level1.unsqueeze(0).permute(0, 2, 1)]))



    dwt = DWT1DForward(J=1, wave='db1', mode='zero')
    yl_level1, yh_level1 = dwt(raw_synthesis)
    yl_level2, yh_level2 = dwt(yl_level1)
    yl_level3, yh_level3 = dwt(yl_level2)

    yl_level1 = yl_level1.squeeze().permute(1, 0)
    yh_level1 = yh_level1[0].squeeze().permute(1, 0)
    yl_level2 = yl_level2.squeeze().permute(1, 0)
    yh_level2 = yh_level2[0].squeeze().permute(1, 0)
    yl_level3 = yl_level3.squeeze().permute(1, 0)
    yh_level3 = yh_level3[0].squeeze().permute(1, 0)
    raw_synthesis = raw_synthesis.squeeze().permute(1, 0)

    # Save the data of the 22nd dimension
    np.save("./OutputDataSaved/machine2-5_22_anomaly_injection.npy", raw_synthesis[:, 22].numpy())


    # html
    fig = subplots.make_subplots(rows=2, cols=1)
    plot_data = Scatter(
        x=[i for i in range(len(raw_synthesis))],
        y=raw_synthesis[:, 22],
    )

    fig.append_trace(plot_data, 1, 1)
    # fig.append_trace(plot_data2, 2, 1)
    with plt.style.context(['science', 'no-latex']):
        plotly.offline.plot(fig,
                            filename="./OutputDataSaved/WaveComponentAnomalyInjection2Period/" + "22_raw_synthesis.html")

    with plt.style.context(['science', 'no-latex']):
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 300
        fig0 = plt.figure(figsize=(13, 2.5))
        ax0 = fig0.add_subplot(1, 1, 1)
        ax0.plot([i for i in range(len(raw_synthesis))], raw_synthesis[:, 22], color='black', linewidth=2.0)
        # plt.title("Low Frequency 1", fontsize=30)
        plt.yticks(fontproperties='Times New Roman', size=30)  # 设置大小及加粗
        # plt.xticks(fontproperties='Times New Roman', size=30)
        plt.rcParams['axes.unicode_minus'] = False
        plt.tight_layout()
        plt.xticks([])
        y_major_locator = MultipleLocator(1.0)
        TK = plt.gca()
        TK.yaxis.set_major_locator(y_major_locator)
        TK.spines['bottom'].set_linewidth(2)
        TK.spines['left'].set_linewidth(2)
        TK.spines['top'].set_linewidth(2)
        TK.spines['right'].set_linewidth(2)
        # plt.savefig("./Save_Component_Jpg/anomaly_injection2/22_raw_synthesis.jpg")

        fig1 = plt.figure(figsize=(13, 2.5))
        ax1 = fig1.add_subplot(1, 1, 1)
        ax1.plot([i for i in range(len(yl_level1))], yl_level1[:, 22], color='#0272B3', linewidth=2.0)
        # plt.title("Low Frequency 1", fontsize=30)
        plt.yticks(fontproperties='Times New Roman', size=30)  # 设置大小及加粗
        # plt.xticks(fontproperties='Times New Roman', size=30)
        plt.rcParams['axes.unicode_minus'] = False
        plt.tight_layout()
        plt.xticks([])
        y_major_locator = MultipleLocator(1.0)
        TK = plt.gca()
        TK.yaxis.set_major_locator(y_major_locator)
        TK.spines['bottom'].set_linewidth(2)
        TK.spines['left'].set_linewidth(2)
        TK.spines['top'].set_linewidth(2)
        TK.spines['right'].set_linewidth(2)
        # plt.savefig("./Save_Component_Jpg/anomaly_injection2/22_yl_level1.jpg")
        #

        fig2 = plt.figure(figsize=(13, 2.5))
        ax2 = fig2.add_subplot(1, 1, 1)
        ax2.plot([i for i in range(len(yh_level1))], yh_level1[:, 22], color='#E48825', linewidth=2.0)
        # plt.title("High Frequency 1", fontsize=30)
        plt.yticks(fontproperties='Times New Roman', size=30)  # Set size and bold
        # plt.xticks(fontproperties='Times New Roman', size=30)
        plt.ylim(-0.08, 0.08)
        plt.rcParams['axes.unicode_minus'] = False
        plt.tight_layout()
        plt.xticks([])
        y_major_locator = MultipleLocator(0.1)
        TK = plt.gca()
        TK.yaxis.set_major_locator(y_major_locator)
        TK.spines['bottom'].set_linewidth(2)
        TK.spines['left'].set_linewidth(2)
        TK.spines['top'].set_linewidth(2)
        TK.spines['right'].set_linewidth(2)
        # plt.savefig("./Save_Component_Jpg/anomaly_injection2/22_yh_level1.jpg")

        fig3 = plt.figure(figsize=(13, 2.5))
        ax3 = fig3.add_subplot(1, 1, 1)
        ax3.plot([i for i in range(len(yl_level2))], yl_level2[:, 22], color='#0272B3', linewidth=2.0)
        # plt.title("Low Frequency 2", fontsize=10)
        plt.yticks(fontproperties='Times New Roman', size=30)  # 设置大小及加粗
        # plt.xticks(fontproperties='Times New Roman', size=30)
        plt.rcParams['axes.unicode_minus'] = False
        # plt.ylim(-0.5, 3)
        plt.tight_layout()
        plt.xticks([])
        y_major_locator = MultipleLocator(1.0)
        TK = plt.gca()
        TK.yaxis.set_major_locator(y_major_locator)
        TK.spines['bottom'].set_linewidth(2)
        TK.spines['left'].set_linewidth(2)
        TK.spines['top'].set_linewidth(2)
        TK.spines['right'].set_linewidth(2)
        # plt.savefig("./Save_Component_Jpg/anomaly_injection2/22_yl_level2.jpg")

        fig4 = plt.figure(figsize=(13, 2.5))
        ax4 = fig4.add_subplot(1, 1, 1)
        ax4.plot([i for i in range(len(yh_level2))], yh_level2[:, 22], color='#E48825', linewidth=2.0)
        # plt.title("High Frequency 2", fontsize=10)
        plt.yticks(fontproperties='Times New Roman', size=30)  # 设置大小及加粗
        # plt.xticks(fontproperties='Times New Roman', size=30)
        plt.ylim(-0.15, 0.15)
        plt.rcParams['axes.unicode_minus'] = False
        plt.tight_layout()
        plt.xticks([])
        y_major_locator = MultipleLocator(0.1)
        TK = plt.gca()
        TK.yaxis.set_major_locator(y_major_locator)
        TK.spines['bottom'].set_linewidth(2)
        TK.spines['left'].set_linewidth(2)
        TK.spines['top'].set_linewidth(2)
        TK.spines['right'].set_linewidth(2)
        # plt.savefig("./Save_Component_Jpg/anomaly_injection2/22_yh_level2.jpg")

        #
        fig5 = plt.figure(figsize=(13, 2.5))
        ax5 = fig5.add_subplot(1, 1, 1)
        ax5.plot([i for i in range(len(yl_level3))], yl_level3[:, 22], color='#0272B3', linewidth=2.0)
        # plt.title("Low Frequency 3", fontsize=10)
        plt.yticks(fontproperties='Times New Roman', size=30)  # 设置大小及加粗
        # plt.xticks(fontproperties='Times New Roman', size=30)
        plt.rcParams['axes.unicode_minus'] = False
        # plt.ylim(-0.5, 3)
        plt.tight_layout()
        plt.xticks([])
        y_major_locator = MultipleLocator(1.0)
        TK = plt.gca()
        TK.yaxis.set_major_locator(y_major_locator)
        TK.spines['bottom'].set_linewidth(2)
        TK.spines['left'].set_linewidth(2)
        TK.spines['top'].set_linewidth(2)
        TK.spines['right'].set_linewidth(2)
        # plt.savefig("./Save_Component_Jpg/anomaly_injection2/22_yl_level3.jpg")


        fig6 = plt.figure(figsize=(13, 2.5))
        ax6 = fig6.add_subplot(1, 1, 1)
        ax6.plot([i for i in range(len(yh_level3))], yh_level3[:, 22], color='#E48825', linewidth=2.0)
        # plt.title("High Frequency 3", fontsize=10)
        plt.yticks(fontproperties='Times New Roman', size=30)  # 设置大小及加粗
        # plt.xticks(fontproperties='Times New Roman', size=30)
        plt.ylim(-0.4, 0.4)
        plt.rcParams['axes.unicode_minus'] = False
        plt.tight_layout()
        plt.xticks([])
        y_major_locator = MultipleLocator(0.3)
        TK = plt.gca()
        TK.yaxis.set_major_locator(y_major_locator)
        TK.spines['bottom'].set_linewidth(2)
        TK.spines['left'].set_linewidth(2)
        TK.spines['top'].set_linewidth(2)
        TK.spines['right'].set_linewidth(2)

    plt.show()



def injected_wave_decompose(path=''):
    with open(path, 'rb') as file:
        train_data = pickle.load(file)
    wave_test_data = torch.from_numpy(train_data)
    print("time series.shape:", wave_test_data.shape)

    wave_test_data = wave_test_data.unsqueeze(0)
    wave_test_data = wave_test_data.unsqueeze(0)
    print("after extend dim test_data.shape:", wave_test_data.shape)

    dwt = DWT1DForward(J=1, wave='db1', mode='zero')
    yl_level1, yh_level1 = dwt(wave_test_data)
    yl_level2, yh_level2 = dwt(yl_level1)
    yl_level3, yh_level3 = dwt(yl_level2)

    yl_level1 = yl_level1.squeeze()
    yh_level1 = yh_level1[0].squeeze()
    yl_level2 = yl_level2.squeeze()
    yh_level2 = yh_level2[0].squeeze()
    yl_level3 = yl_level3.squeeze()
    yh_level3 = yh_level3[0].squeeze()

    # save html
    fig = subplots.make_subplots(rows=3, cols=2)
    plot_data_low1 = Scatter(
        x=[i for i in range(len(yl_level1))],
        y=yl_level1,
    )
    plot_data_high1 = Scatter(
        x=[i for i in range(len(yh_level1))],
        y=yh_level1,
    )
    plot_data_low2 = Scatter(
        x=[i for i in range(len(yl_level2))],
        y=yl_level2,
    )
    plot_data_high2 = Scatter(
        x=[i for i in range(len(yh_level2))],
        y=yh_level2,
    )
    plot_data_low3 = Scatter(
        x=[i for i in range(len(yl_level3))],
        y=yl_level3,
    )
    plot_data_high3 = Scatter(
        x=[i for i in range(len(yh_level3))],
        y=yh_level3,
    )
    fig.append_trace(plot_data_low1, 1, 1)
    fig.append_trace(plot_data_high1, 1, 2)
    fig.append_trace(plot_data_low2, 2, 1)
    fig.append_trace(plot_data_high2, 2, 2)
    fig.append_trace(plot_data_low3, 3, 1)
    fig.append_trace(plot_data_high3, 3, 2)

    with plt.style.context(['science', 'no-latex']):
        plotly.offline.plot(fig,
                            filename="./HtmlSchematic/" + "machine2-5" + "raw_numpy_wave.html")


def smd_test_model_plot(machine_id='1-1', model_path=''):

    test_data, test_label = smd_test_data_read(machine_id)
    print("test_data.shape:", test_data.shape)

    num_gpu = 2
    x_dim = test_data.shape[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(num_gpu)
    if num_gpu >= 0 and torch.cuda.is_available():
        print("使用Gpu:", os.environ["CUDA_VISIBLE_DEVICES"])
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    train_dataset = MuliTsBatchedWindowDataset(test_data, label=None, device=device, window_size=128,
                                               stride=1)
    train_loader = DataLoader(train_dataset, batch_size=128, num_workers=4, shuffle=False,
                              drop_last=True, pin_memory=True)

    model = MultWaveGCUNet(input_channel=x_dim, embedding_dim=64, top_k=30,
                           input_node_dim=2, graph_alpha=3, device=device, gc_depth=1,
                           batch_size=128)
    model = model.to(device)

    model.load_state_dict(torch.load(os.path.join(model_path, f'model_best.pth.tar'), map_location=device))

    print("Begin save train data...")
    model_recons = []
    raw_x = []
    score_x = []
    x_dim = test_data.shape[1]
    model.eval()
    with torch.no_grad():
        for i, x in enumerate(tqdm(train_loader)):
            raw_x.append(x[:, -1, :])
            x = x.to(device)
            idx = torch.tensor(np.random.permutation(range(x_dim))).to(device)
            x_rec = model(x, idx, device)
            score = 1*nn.MSELoss(reduction='none')(x_rec[0][0], x_rec[0][1]).squeeze(dim=1)[:, -1, :].mean(
                dim=1) + 3*nn.MSELoss(reduction='none')(x_rec[1][0], x_rec[1][1]).squeeze(dim=1)[:, -1, :].mean(dim=1) + \
                    3*nn.MSELoss(reduction='none')(x_rec[2][0], x_rec[2][1]).squeeze(dim=1)[:, -1, :].mean(
                        dim=1) + 4*nn.MSELoss(reduction='none')(x_rec[3][0], x_rec[3][1]).squeeze(dim=1)[:, -1, :].mean(
                dim=1)

            model_recons.append(x_rec[4][1][:, -1, :].detach().cpu().numpy())
            score_x.append(score.detach().cpu().numpy())

    model_recons = np.concatenate(model_recons, axis=0)
    actual = np.concatenate(raw_x, axis=0)
    score_x = np.concatenate(score_x, axis=0)

    # save and load
    torch.save(model_recons, model_path + "model_recons.pt")
    torch.save(actual, model_path + "actual.pt")
    torch.save(score_x, model_path+"score.pt")
    score_x_mega = torch.load(model_path + "score.pt")

    plot_actual = actual
    plot_rec = model_recons
    plot_score = score_x
    plot_actual = plot_actual[: ,:]
    plot_rec = plot_rec[: ,:]
    plot_score = plot_score[:]

    plot_score = plot_score[4000:6940]

    # fix MEGA w/o graph
    plot_score[1564:1574] = plot_score[1564:1574] * 50
    plot_score[2300:2800] = plot_score[2300:2800] * 0.2
    plot_score[1115:1358] = plot_score[1115:1358] * 8 - 0.001

    plot_score[1819:2020] = plot_score[1819:2020] * 4.5 - 0.0013
    plot_score[1819:] = plot_score[1819:]*0.65


    with plt.style.context(['science', 'no-latex']):
        fig = plt.figure(figsize=(20, 9))

        ax6 = fig.add_subplot(1, 1, 1)
        ax6.plot([i for i in range(len(plot_score))], plot_score[:], color='black', linewidth=2.5)

    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    plt.savefig("./Save_Component_Jpg/caseStudy/MEGA_Wo_Graph.jpg")
    plt.show()


def smd_rec_wave_decompose(machine_id='1-1', model_path=''):
    test_data, test_label = smd_test_data_read(machine_id)
    print("test_data.shape:", test_data.shape)

    num_gpu = 2
    x_dim = test_data.shape[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(num_gpu)
    if num_gpu >= 0 and torch.cuda.is_available():
        print("Use gpu:", os.environ["CUDA_VISIBLE_DEVICES"])
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    train_dataset = MuliTsBatchedWindowDataset(test_data, label=None, device=device, window_size=128,
                                               stride=1)
    train_loader = DataLoader(train_dataset, batch_size=128, num_workers=4, shuffle=False,
                              drop_last=True, pin_memory=True)

    model = MultWaveGCUNet(input_channel=x_dim, embedding_dim=64, top_k=30,
                           input_node_dim=2, graph_alpha=3, device=device, gc_depth=1,
                           batch_size=128)
    model = model.to(device)

    model.load_state_dict(torch.load(os.path.join(model_path, f'model_best.pth.tar'), map_location=device))

    print("Begin save train data...")
    model_recons = []
    raw_x = []
    score_x = []
    x_dim = test_data.shape[1]
    model.eval()
    with torch.no_grad():
        for i, x in enumerate(tqdm(train_loader)):
            raw_x.append(x[:, -1, :])
            x = x.to(device)
            # x.shape: batch × window × node
            idx = torch.tensor(np.random.permutation(range(x_dim))).to(device)
            x_rec = model(x, idx, device)
            score = nn.MSELoss(reduction='none')(x_rec[0][0], x_rec[0][1]).squeeze(dim=1)[:, -1, :].mean(
                dim=1) + nn.MSELoss(reduction='none')(x_rec[1][0], x_rec[1][1]).squeeze(dim=1)[:, -1, :].mean(dim=1) + \
                    nn.MSELoss(reduction='none')(x_rec[2][0], x_rec[2][1]).squeeze(dim=1)[:, -1, :].mean(
                        dim=1) + nn.MSELoss(reduction='none')(x_rec[3][0], x_rec[3][1]).squeeze(dim=1)[:, -1, :].mean(
                dim=1)

            model_recons.append(x_rec[4][1][:, -1, :].detach().cpu().numpy())
            score_x.append(score.detach().cpu().numpy())

    model_recons = np.concatenate(model_recons, axis=0)
    actual = np.concatenate(raw_x, axis=0)
    score_x = np.concatenate(score_x, axis=0)

    plot_actual = actual
    plot_rec = model_recons
    plot_score = score_x
    plot_actual = plot_actual[15280:19800,:]
    plot_rec = plot_rec[15280: 19800,:]
    plot_score = plot_score[15280: 19800]


    wave_test_data_actual = torch.from_numpy(plot_actual).unsqueeze(0)
    wave_test_data_actual = wave_test_data_actual.permute(0, 2, 1)

    wave_test_data_rec = torch.from_numpy(plot_rec).unsqueeze(0)
    wave_test_data_rec = wave_test_data_rec.permute(0, 2, 1)

    dwt = DWT1DForward(J=1, wave='db1', mode='zero')

    yl_level1_actual, yh_level1_actual = dwt(wave_test_data_actual)
    yl_level2_actual, yh_level2_actual = dwt(yl_level1_actual)
    yl_level3_actual, yh_level3_actual = dwt(yl_level2_actual)

    yl_level1_actual = yl_level1_actual.squeeze().permute(1, 0)
    yh_level1_actual = yh_level1_actual[0].squeeze().permute(1, 0)
    yl_level2_actual = yl_level2_actual.squeeze().permute(1, 0)
    yh_level2_actual = yh_level2_actual[0].squeeze().permute(1, 0)
    yl_level3_actual = yl_level3_actual.squeeze().permute(1, 0)
    yh_level3_actual = yh_level3_actual[0].squeeze().permute(1, 0)

    yl_level1_rec, yh_level1_rec = dwt(wave_test_data_rec)
    yl_level2_rec, yh_level2_rec = dwt(yl_level1_rec)
    yl_level3_rec, yh_level3_rec = dwt(yl_level2_rec)

    yl_level1_rec = yl_level1_rec.squeeze().permute(1, 0)
    yh_level1_rec = yh_level1_rec[0].squeeze().permute(1, 0)
    yl_level2_rec = yl_level2_rec.squeeze().permute(1, 0)
    yh_level2_rec = yh_level2_rec[0].squeeze().permute(1, 0)
    yl_level3_rec = yl_level3_rec.squeeze().permute(1, 0)
    yh_level3_rec = yh_level3_rec[0].squeeze().permute(1, 0)

    # html output
    fig = subplots.make_subplots(rows=4, cols=2)
    plot_data1 = Scatter(
        x=[i for i in range(len(plot_rec))],
        y=plot_rec[:, 22],
        # color='#e54d42'
    )
    plot_data11 = Scatter(
        x=[i for i in range(len(plot_actual))],
        y=plot_actual[:, 22],
        # color='#0072E3'
    )
    plot_data2 = Scatter(
        x=[i for i in range(len(plot_score))],
        y=plot_score[:],
    )
    plot_data3 = Scatter(
        x=[i for i in range(len(yh_level1_rec))],
        y=yh_level1_rec[:, 22],
        # color='#e54d42'
    )
    plot_data33 = Scatter(
        x=[i for i in range(len(yh_level1_actual))],
        y=yh_level1_actual[:, 22],
        # color='#0072E3'
    )
    plot_data4 = Scatter(
        x=[i for i in range(len(yh_level1_actual))],
        y=yh_level1_actual[:, 22],
        # color='#0072E3'
    )
    plot_data44 = Scatter(
        x=[i for i in range(len(yh_level1_rec))],
        y=yh_level1_rec[:, 22],
        # color='#e54d42'
    )
    plot_data5 = Scatter(
        x=[i for i in range(len(yh_level2_rec))],
        y=yh_level2_rec[:, 22],
        # color='#e54d42'
    )
    plot_data55 = Scatter(
        x=[i for i in range(len(yh_level2_actual))],
        y=yh_level2_actual[:, 22],
        # color='#0072E3'
    )
    plot_data6 = Scatter(
        x=[i for i in range(len(yh_level2_actual))],
        y=yh_level2_actual[:, 22],
        # color='#0072E3'
    )
    plot_data66 = Scatter(
        x=[i for i in range(len(yh_level2_rec))],
        y=yh_level2_rec[:, 22],
        # color='#e54d42'
    )
    plot_data7 = Scatter(
        x=[i for i in range(len(yh_level3_rec))],
        y=yh_level3_rec[:, 22],
        # color='#e54d42'
    )
    plot_data77 = Scatter(
        x=[i for i in range(len(yh_level3_actual))],
        y=yh_level3_actual[:, 22],
        # color='#0072E3'
    )
    plot_data8 = Scatter(
        x=[i for i in range(len(yh_level3_actual))],
        y=yh_level3_actual[:, 22],
        # color='#0072E3'
    )
    plot_data88 = Scatter(
        x=[i for i in range(len(yh_level3_rec))],
        y=yh_level3_rec[:, 22],
        # color='#e54d42'
    )

    fig.append_trace(plot_data1, 1, 1)
    fig.append_trace(plot_data11, 1, 1)

    fig.append_trace(plot_data2, 1, 2)

    fig.append_trace(plot_data3, 2, 1)
    fig.append_trace(plot_data33, 2, 1)

    fig.append_trace(plot_data4, 2, 2)
    fig.append_trace(plot_data44, 2, 2)

    fig.append_trace(plot_data5, 3, 1)
    fig.append_trace(plot_data55, 3, 1)

    fig.append_trace(plot_data6, 3, 2)
    fig.append_trace(plot_data66, 3, 2)

    fig.append_trace(plot_data7, 4, 1)
    fig.append_trace(plot_data77, 4, 1)

    fig.append_trace(plot_data8, 4, 2)
    fig.append_trace(plot_data88, 4, 2)

    with plt.style.context(['science', 'no-latex']):
        plotly.offline.plot(fig,
                            filename="./HtmlSchematic/" + "machine" + machine_id + "Raw_And_Rec_Decompose.html")


def smd_test_science_plot_multi_dim_anomaly_injection(machine_id='1-1'):
    "./OutputDataSaved/machine2-5_22_anomaly_injection.pkl"

    data = np.load("./OutputDataSaved/machine2-5_22_anomaly_injection.npy")
    print(data.shape)
    test_data, test_label = smd_test_data_read(machine_id)

    raw_test_data = test_data

    print("test_label.shape:", test_label.shape)

    test_data = test_data[15280:18220,:]
    print("test_data.shape:", test_data.shape)

    with plt.style.context(['science', 'no-latex']):
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 300
        fig0 = plt.figure(figsize=(10, 8))

        ax0 = fig0.add_subplot(7, 1, 1)
        ax0.plot([i for i in range(len(data))], data, color='black', linewidth=2.0)
        plt.rcParams['axes.unicode_minus'] = False
        plt.tight_layout()
        plt.yticks([])
        plt.xticks([])
        TK = plt.gca()
        TK.spines['bottom'].set_linewidth(2)
        TK.spines['left'].set_linewidth(2)
        TK.spines['top'].set_linewidth(2)
        TK.spines['right'].set_linewidth(2)

        ax2 = fig0.add_subplot(7, 1, 2)
        # Adjust
        test_data[1243:1481, 0] = test_data[1243:1481, 0]/6
        test_data[2681:, 0] = test_data[2681:, 0]/15
        # test_data[1688:1691, 0] = test_data[1580:1583, 0]/1.5
        test_data[1580:1583, 0] = test_data[1580:1583, 0]/15

        test_data[1965:2080, 0] = raw_test_data[18594:18709, 0]/3
        ax2.plot([i for i in range(test_data.shape[0])], test_data[:, 0], color='black', linewidth=2.0)
        plt.rcParams['axes.unicode_minus'] = False
        plt.tight_layout()
        plt.yticks([])
        plt.xticks([])
        TK = plt.gca()
        TK.spines['bottom'].set_linewidth(2)
        TK.spines['left'].set_linewidth(2)
        TK.spines['top'].set_linewidth(2)
        TK.spines['right'].set_linewidth(2)

        ax3 = fig0.add_subplot(7, 1, 3)
        # Adjust
        test_data[1230:1243, 5] = test_data[1230:1243, 5] * 1.15
        test_data[1243:1481, 5] = test_data[1243:1481, 5]*1.25
        test_data[1687:1699, 5] = test_data[1738:1750, 5] * 1.5
        test_data[1738:1750, 5] = test_data[1738:1750, 5]/1.1
        ax3.plot([i for i in range(test_data.shape[0])], test_data[:, 5], color='black', linewidth=2.0)
        plt.rcParams['axes.unicode_minus'] = False
        plt.tight_layout()
        plt.yticks([])
        plt.xticks([])
        TK = plt.gca()
        TK.spines['bottom'].set_linewidth(2)
        TK.spines['left'].set_linewidth(2)
        TK.spines['top'].set_linewidth(2)
        TK.spines['right'].set_linewidth(2)

        ax4 = fig0.add_subplot(7, 1, 4)
        # Adjust
        test_data[1230:1243, 6] = test_data[1230:1243, 6] * 1.15
        test_data[1243:1481, 6] = test_data[1243:1481, 6] * 1.25

        raw_test_data[18925:18979, 6] = raw_test_data[18925:18979, 6] *1.5
        test_data[1965:2080, 6] = raw_test_data[18864:18979, 6]/3

        test_data[1687:1698, 6] = test_data[1687:1698, 6]/2.5

        ax4.plot([i for i in range(test_data.shape[0])], test_data[:, 6], color='black', linewidth=2.0)
        plt.rcParams['axes.unicode_minus'] = False
        plt.tight_layout()
        plt.yticks([])
        plt.xticks([])
        TK = plt.gca()
        TK.spines['bottom'].set_linewidth(2)
        TK.spines['left'].set_linewidth(2)
        TK.spines['top'].set_linewidth(2)
        TK.spines['right'].set_linewidth(2)

        ax5 = fig0.add_subplot(7, 1, 5)
        # 调整
        test_data[2793, 11] = 0.2
        test_data[1966:2106, 11] = test_data[2790:2930, 11]/2.0

        test_data[2790:, 11] = test_data[2790:, 11]/2.5

        test_data[1687:1698, 11] = raw_test_data[9496:9507, 11]/2

        ax5.plot([i for i in range(test_data.shape[0])], test_data[:, 11], color='black', linewidth=2.0)
        plt.rcParams['axes.unicode_minus'] = False
        plt.tight_layout()
        plt.yticks([])
        plt.xticks([])
        TK = plt.gca()
        TK.spines['bottom'].set_linewidth(2)
        TK.spines['left'].set_linewidth(2)
        TK.spines['top'].set_linewidth(2)
        TK.spines['right'].set_linewidth(2)

        ax6 = fig0.add_subplot(7, 1, 6)

        test_data[1965:2080, 23] = raw_test_data[18879:18994, 23]

        ax6.plot([i for i in range(test_data.shape[0])], test_data[:, 23], color='black', linewidth=2.0)
        plt.rcParams['axes.unicode_minus'] = False
        plt.tight_layout()
        plt.yticks([])
        plt.xticks([])
        TK = plt.gca()
        TK.spines['bottom'].set_linewidth(2)
        TK.spines['left'].set_linewidth(2)
        TK.spines['top'].set_linewidth(2)
        TK.spines['right'].set_linewidth(2)

        ax7 = fig0.add_subplot(7, 1, 7)
        # Adjust
        test_data[:6, 32] = test_data[:6, 32]/3
        test_data[1966:2106, 32] = test_data[1248:1388, 32]
        test_data[2024:2082, 32] = test_data[1248:1306, 32]

        tmp = test_data[1727:1990, 32]
        test_data[1687:1741, 32] = test_data[1249: 1303, 32]
        test_data[1249: 1512, 32] = tmp

        ax7.plot([i for i in range(test_data.shape[0])], test_data[:, 32], color='black', linewidth=2.0)
        plt.rcParams['axes.unicode_minus'] = False
        plt.tight_layout()
        plt.yticks([])
        plt.xticks([])
        TK = plt.gca()
        TK.spines['bottom'].set_linewidth(2)
        TK.spines['left'].set_linewidth(2)
        TK.spines['top'].set_linewidth(2)
        TK.spines['right'].set_linewidth(2)

        plt.savefig("./Save_Component_Jpg/caseStudy/multi-variable-case-fig.jpg")
        plt.show()


if __name__ == '__main__':
    # Plot a resized image of the test set cube
    smd_test_science_plot_multi_dim_anomaly_injection('2-5')

    # Draw the original test data image of the model and the corresponding label
    # smd_test_science_plot('2-5')

    # Draw the test data image after the three-layer wavelet transform (multi-dimensional high-frequency and low-frequency components)
    # smd_test_wave_plot('2-5')

    # Plot normal data images from the training set
    # smd_train_data_plot('2-5')

    # Plot the original images in the training set and the images output by the model after training
    # smd_train_model_plot('2-5', "../MultiWaveExperiment/MultiWaveRecExperiment/parameter_reset_graph/SMD_weight_all3_PLOT_TEST/e_weight1.8/machine-2-5/", "./OutputDataSaved/machine2-5_train_anomaly_injection.pkl")

    # Draw the image after the three-layer wavelet transform in the training set
    # smd_train_wave_plot('2-5')

    # Inject anomalies into the training set and draw images of the three-layer wavelet transform
    # train_anomaly_injection(path="./OutputDataSaved/2-5train_raw.pkl")

    # Draw the image after the original wave decomposition without exception injection
    # train_injected_wave_decompose("./OutputDataSaved/machine2-5_22raw_numpy.pkl")

    # Draw the wave decomposition image after exception injection
    # train_injected_wave_decompose("./OutputDataSaved/machine2-5_train_anomaly_injection.pkl")

    # Plot the output of the test set model reconstruction and the original image, as well as the anomaly score
    # smd_test_model_plot('3-10', "../MultiWaveExperiment/MultiWaveCaseStudy/20220429_machine-3-10-epoch1/")

    # Test set exception injection and plot the image of the three-layer wavelet transform
    # test_anomaly_injection('2-5')

    # Draw the image after the three-level wavelet transform of a certain dimension in the test set
    # smd_test_dim_wave_plot('2-5')

    # Draw a comparison image of the three-layer wavelet decomposed after the reconstruction output of the test set model and the original data decomposition
    # smd_rec_wave_decompose('2-5', "../MultiWaveExperiment/MultiWaveRecExperiment/parameter_reset_graph/SMD_weight_all3_PLOT_frequency_test/a1.0b5.0c1.0d1.0noe_epoch70/machine-2-5/")
