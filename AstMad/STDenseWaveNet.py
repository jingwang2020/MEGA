# -*- coding: utf-8 -*-
# @Time    : 2021/11/15 16:36
# @Author  : SSK
# @Email   : skshao@bjtu.edu.cn
# @File    : STDenseWaveNet.py
# @Software: PyCharm
from torch import nn
import torch
from .net import EmbeddingGraph, DenseWave


class STDenseWaveNet(nn.Module):
    """
        The simple Dense Wave network used for reconstruction,
        the reconstruction effect is not good, but the evaluation index of anomaly detection is very high
    """
    def __init__(self, input_channel, embedding_dim=64, top_k=30, input_node_dim=1, conv_node_dim=16, graph_alpha=3,
                 wave_channel1=64, wave_channel2=32, device=torch.device('cuda:1')):
        super(STDenseWaveNet, self).__init__()
        self.graph = EmbeddingGraph(input_channel, embedding_dim, top_k, graph_alpha, device)
        self.conv_start = nn.Conv2d(in_channels=input_node_dim, out_channels=conv_node_dim, kernel_size=(1, 1))
        self.dense_wave = DenseWave(input_channel=input_channel, wave_channel1=wave_channel1, wave_channel2=wave_channel2,
                                    growthRate=4, nDenseBlocks=3)

    def forward(self, x, idx):
        adj = self.graph(idx)
        output = self.dense_wave(x)
        return output
