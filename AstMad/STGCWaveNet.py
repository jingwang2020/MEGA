# -*- coding: utf-8 -*-
# @Time    : 2021/12/3 10:24
# @Author  : SSK
# @Email   : skshao@bjtu.edu.cn
# @File    : STGCWaveNet.py
# @Software: PyCharm
from torch import nn
import torch
from .net import EmbeddingGraph, WaveDecompose, WaveGCN


class STGCWaveNet(nn.Module):
    """
        DWT decomposition
        decompose the features of different frequency classifications, connect to GCN, aggregate node features
        use 3 different W weights to adaptively get the importance of different frequencies,
        finally connect a gru and linear to complete the reconstruction, using only the final refactoring structure loss for training
    """
    def __init__(self, input_channel, embedding_dim=64, top_k=30, input_node_dim=2, conv_node_dim=16, graph_alpha=3,
                 device=torch.device('cuda:1')):
        super(STGCWaveNet, self).__init__()
        self.device = device
        self.batch = 200
        self.graph = EmbeddingGraph(input_channel, embedding_dim, top_k, graph_alpha, device)
        self.wave = WaveDecompose(input_channel)

        wave_conv_out_dim = [8, 16, 32]

        self.wave_conv1 = nn.Conv2d(in_channels=input_node_dim, out_channels=wave_conv_out_dim[0], kernel_size=(1, 1))
        self.wave_conv2 = nn.Conv2d(in_channels=input_node_dim, out_channels=wave_conv_out_dim[1], kernel_size=(1, 1))
        self.wave_conv3 = nn.Conv2d(in_channels=input_node_dim, out_channels=wave_conv_out_dim[2], kernel_size=(1, 1))

        self.wave_gcn = WaveGCN()

        self.fuse_weights1 = nn.Parameter(torch.Tensor(200, 38, 16), requires_grad=True).to(device)
        self.fuse_weights2 = nn.Parameter(torch.Tensor(200, 38, 16), requires_grad=True).to(device)
        self.fuse_weights3 = nn.Parameter(torch.Tensor(200, 38, 16), requires_grad=True).to(device)

        self.bn = nn.BatchNorm1d(38)
        self.act = nn.ReLU()

        self.gru = nn.GRU(input_size=38, hidden_size=64, num_layers=2, batch_first=True)
        self.linear_layer = nn.Linear(16 * 64, 38 * 128)

    def forward(self, x, idx):
        adj = self.graph(idx)
        wave_feature = self.wave(x)
        wave_low1 = torch.unsqueeze(wave_feature[0][0], dim=1)
        wave_high1 = torch.unsqueeze(wave_feature[0][1], dim=1)
        wave_feature1 = torch.cat((wave_low1, wave_high1), dim=1)
        wave_feature1 = wave_feature1.transpose(2, 3)
        wave_feature1 = self.wave_conv1(wave_feature1)

        wave_low2 = torch.unsqueeze(wave_feature[1][0], dim=1)
        wave_high2 = torch.unsqueeze(wave_feature[1][1], dim=1)
        wave_feature2 = torch.cat((wave_low2, wave_high2), dim=1)
        wave_feature2 = wave_feature2.transpose(2, 3)
        wave_feature2 = self.wave_conv2(wave_feature2)

        wave_low3 = torch.unsqueeze(wave_feature[2][0], dim=1)
        wave_high3 = torch.unsqueeze(wave_feature[2][1], dim=1)
        wave_feature3 = torch.cat((wave_low3, wave_high3), dim=1)
        wave_feature3 = wave_feature3.transpose(2, 3)
        wave_feature3 = self.wave_conv3(wave_feature3)

        out_put = self.wave_gcn(wave_feature1, wave_feature2, wave_feature3, adj)

        fuse_latent = self.act(self.bn(
            self.fuse_weights1 * out_put[0] + self.fuse_weights2 * out_put[1] + self.fuse_weights3 * out_put[2]))

        fuse_latent = fuse_latent.transpose(1, 2)
        # h0：D∗num_layers, Batch, H_out
        h0 = torch.zeros(2, 200, 64).to(self.device)
        out_put, _ = self.gru(fuse_latent, h0)

        out_put = out_put.contiguous().view(self.batch, -1)
        out_put = self.linear_layer(out_put)
        out_put = out_put.view(200, 128, 38)

        return out_put
