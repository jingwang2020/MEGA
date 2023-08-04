# -*- coding: utf-8 -*-
# @Time    : 2021/11/25 15:36
# @Author  : SSK
# @Email   : skshao@bjtu.edu.cn
# @File    : wavemodule.py
# @Software: PyCharm
import torch.nn as nn
from pytorch_wavelets import DWT1DForward, DWT1DInverse
import torch


class Dwtconv(nn.Module):

    def __init__(self, node_feature, outC1, outC2):
        super(Dwtconv, self).__init__()

        self.dwt1 = DWT1DForward(J=1, wave='haar', mode='symmetric')
        outChannel_conv1 = outC1 // 2
        nIn_conv1 = node_feature * 2
        self.conv1_1 = nn.Conv1d(nIn_conv1, outChannel_conv1, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm1d(outChannel_conv1)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv1d(outChannel_conv1, outC1, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm1d(outC1)
        self.relu1_2 = nn.ReLU()

        nIn_conv2 = node_feature * 2
        self.dwt2 = DWT1DForward(J=1, wave='haar', mode='symmetric')
        outChannel_conv2 = outC2 // 2
        self.conv2_1 = nn.Conv1d(nIn_conv2, outChannel_conv2, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm1d(outChannel_conv2)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv1d(outChannel_conv2, outC2, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm1d(outC2)
        self.relu2_2 = nn.ReLU()

    def forward(self, x):
        dwt1_1_l, dwt1_1_h = self.dwt1(x)
        dwt1_1 = torch.cat((dwt1_1_l, dwt1_1_h[0][:,:,:]), dim=1)
        conv1_1 = self.conv1_1(dwt1_1)
        bn1_1 = self.bn1_1(conv1_1)
        relu1_1 = self.relu1_1(bn1_1)
        conv1_2 = self.conv1_2(relu1_1)
        bn1_2 = self.bn1_2(conv1_2)
        relu1_2 = self.relu1_2(bn1_2)

        dwt2_1_l, dwt2_1_h = self.dwt2(dwt1_1_l)
        dwt2_1 = torch.cat((dwt2_1_l, dwt2_1_h[0][:, :, :]), dim=1)
        conv2_1 = self.conv2_1(dwt2_1)
        bn2_1 = self.bn2_1(conv2_1)
        relu2_1 = self.relu2_1(bn2_1)
        conv2_2 = self.conv2_2(relu2_1)
        bn2_2 = self.bn2_2(conv2_2)
        relu2_2 = self.relu2_2(bn2_2)

        return relu1_2, relu2_2


class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm1d(nChannels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(nChannels, nOutChannels, kernel_size=1)
        self.avg_pool = nn.AvgPool1d(2)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.avg_pool(out)
        return out


class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4 *growthRate
        self.bn1 = nn.BatchNorm1d(nChannels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(nChannels, interChannels, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(interChannels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv1d(interChannels, growthRate, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out


class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm1d(nChannels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(nChannels, growthRate, kernel_size=3,
                               padding=1)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


class DWTCompose(nn.Module):
    def __init__(self, node):
        super(DWTCompose, self).__init__()
        self.dwt1 = DWT1DForward(J=1, wave='db1', mode='zero')
        self.dwt2 = DWT1DForward(J=1, wave='db1', mode='zero')
        # self.dwt3 = DWT1DForward(J=1, wave='db1', mode='zero')
        self.bn = nn.BatchNorm1d(node)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        dwt1_low, dwt1_high = self.dwt1(x)
        dwt2_low, dwt2_high = self.dwt1(dwt1_low)

        dwt1_low = dwt1_low
        dwt1_high = dwt1_high[0]

        dwt2_low = dwt2_low
        dwt2_high = dwt2_high[0]

        dwt1_low = dwt1_low.permute(0, 2, 1)
        dwt1_high = dwt1_high.permute(0, 2, 1)
        dwt2_low = dwt2_low.permute(0, 2, 1)
        dwt2_high = dwt2_high.permute(0, 2, 1)

        return [(dwt1_low, dwt1_high), (dwt2_low, dwt2_high)]


class DWTCompose1Layer(nn.Module):
    def __init__(self, node):
        super(DWTCompose1Layer, self).__init__()
        self.dwt1 = DWT1DForward(J=1, wave='db1', mode='zero')
        self.bn = nn.BatchNorm1d(node)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        dwt1_low, dwt1_high = self.dwt1(x)

        dwt1_low = dwt1_low
        dwt1_high = dwt1_high[0]

        dwt1_low = dwt1_low.permute(0, 2, 1)
        dwt1_high = dwt1_high.permute(0, 2, 1)

        return [(dwt1_low, dwt1_high)]


class IDWT(nn.Module):
    def __init__(self):
        super(IDWT, self).__init__()
        self.idwt1 = DWT1DInverse(wave='db1', mode='zero')

    def forward(self, low, high):
        return self.idwt1((low, high))
