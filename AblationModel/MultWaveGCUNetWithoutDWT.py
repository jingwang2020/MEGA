# -*- coding: utf-8 -*-
# @Time    : 2021/12/6 17:10
# @Author  : SSK
# @Email   : skshao@bjtu.edu.cn
# @File    : MultWaveGC-UNet.py
# @Software: PyCharm
from torch import nn
import torch
from .net import WaveDecompose, IDWTLayers, EmbeddingGraph, MutiLevelWaveGCN
# from .net import NodeGraph


class MultWaveGCUNetWithoutDWT(nn.Module):
    """
        Discard DWT layer
    """
    def __init__(self, input_channel, embedding_dim=64, top_k=30, input_node_dim=1, graph_alpha=3, device=torch.device('cuda:1'), gc_depth=1, batch_size=128):
        super(MultWaveGCUNetWithoutDWT, self).__init__()
        self.wave_decompose = WaveDecompose(input_channel)
        self.graph = EmbeddingGraph(input_channel, embedding_dim, top_k, graph_alpha, device)

        self.input_channel = input_channel
        wave_conv_out_dim = [32, 64, 128]

        self.conv1_gcn_input1 = nn.Sequential(
            nn.Conv2d(in_channels=input_node_dim, out_channels=wave_conv_out_dim[0], kernel_size=(1, 1)),
            nn.LeakyReLU(),
        )
        self.conv1_gcn_input2 = nn.Sequential(
            nn.Conv2d(in_channels=input_node_dim, out_channels=wave_conv_out_dim[1], kernel_size=(1, 1)),
            nn.LeakyReLU(),
        )

        self.conv1_gcn_input3 = nn.Sequential(
            nn.Conv2d(in_channels=input_node_dim, out_channels=wave_conv_out_dim[2], kernel_size=(1, 1)),
            nn.LeakyReLU(),
        )

        self.wave_gcn_layer1 = MutiLevelWaveGCN(input_channel=wave_conv_out_dim[0], gcn_depth=gc_depth)
        self.output_latent1 = nn.Sequential(
            nn.Conv2d(in_channels=wave_conv_out_dim[0], out_channels=wave_conv_out_dim[0], kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(wave_conv_out_dim[0])
        )
        self.wave_gcn_layer2 = MutiLevelWaveGCN(input_channel=wave_conv_out_dim[1], gcn_depth=gc_depth)
        self.output_latent2 = nn.Sequential(
            nn.Conv2d(in_channels=wave_conv_out_dim[1], out_channels=wave_conv_out_dim[1], kernel_size=(3, 3), stride=1,padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(wave_conv_out_dim[1])
        )
        self.wave_gcn_layer3 = MutiLevelWaveGCN(input_channel=wave_conv_out_dim[2], gcn_depth=gc_depth)
        self.output_latent3 = nn.Sequential(
            nn.Conv2d(in_channels=wave_conv_out_dim[2], out_channels=wave_conv_out_dim[2], kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(wave_conv_out_dim[2])
        )

        self.wave_generate_high_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=wave_conv_out_dim[0], out_channels=1, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU()
        )
        self.wave_generate_high_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=wave_conv_out_dim[1], out_channels=1, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU()
        )
        self.wave_generate_low_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=wave_conv_out_dim[1], out_channels=1, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.idwt_layer_1 = IDWTLayers()
        self.idwt_layer_2 = IDWTLayers()
        # self.idwt_layer_3 = IDWTLayers()

    def forward(self, x, idx, device):
        adj = self.graph(idx)
        x = torch.unsqueeze(x, dim=1)
        x = x.transpose(2,3)
        x = self.conv1_gcn_input1(x)
        wave_latent1 = self.wave_gcn_layer1(x, adj)
        rec = self.wave_generate_high_layer1(wave_latent1).squeeze(dim=1).transpose(1, 2)
        return rec
