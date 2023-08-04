# -*- coding: utf-8 -*-
# @Time    : 2021/12/6 17:10
# @Author  : SSK
# @Email   : skshao@bjtu.edu.cn
# @File    : MultWaveGC-UNet.py
# @Software: PyCharm
from torch import nn
import torch
from .net import WaveDecompose, WaveDecompose1Layer,IDWTLayers, EmbeddingGraph, MutiLevelWaveGCN
# from .net import NodeGraph


class MultWaveGCUNet1DecomposeLayerWithOutGCN(nn.Module):
    """
        Only use one Decompose Layer without GCN
    """
    def __init__(self, input_channel, embedding_dim=64, top_k=30, input_node_dim=2, graph_alpha=3, device=torch.device('cuda:1'), gc_depth=1, batch_size=128):
        super(MultWaveGCUNet1DecomposeLayerWithOutGCN, self).__init__()
        self.wave_decompose = WaveDecompose1Layer(input_channel)
        # self.graph = EmbeddingGraph(input_channel, embedding_dim, top_k, graph_alpha, device)

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

        self.wave_generate_high_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=wave_conv_out_dim[0], out_channels=1, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU()
        )
        self.wave_generate_low_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=wave_conv_out_dim[0], out_channels=1, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU()
        )
        self.idwt_layer_1 = IDWTLayers()
        # self.idwt_layer_2 = IDWTLayers()
        # self.idwt_layer_3 = IDWTLayers()

    def forward(self, x, idx, device):
        # adj = self.graph(idx)
        wave_feature = self.wave_decompose(x)
        # reconstruction loss 0：x
        # reconstruction loss 1：wave_high1
        wave_low1 = wave_feature[0][0]
        wave_high1 = wave_feature[0][1]
        # adj1 = self.graph_layer1(wave_low1, wave_high1)

        wave_feature1 = torch.cat((torch.unsqueeze(wave_feature[0][0], dim=1), torch.unsqueeze(wave_feature[0][1], dim=1)), dim=1)
        wave_feature1 = wave_feature1.transpose(2, 3)
        wave_feature1 = self.conv1_gcn_input1(wave_feature1)
        # gcn input feature：wave_feature1

        # wave_latent1 = self.wave_gcn_layer1(wave_feature1, adj)
        wave_latent1 = wave_feature1
        # wave_latent1 = wave_feature1

        # wave_latent1 = self.output_latent1(wave_latent1) + wave_latent1
        # latent_representation1：wave_latent1
        generated_high1 = self.wave_generate_high_layer1(wave_latent1).squeeze(dim=1)
        generated_low1 = self.wave_generate_low_layer1(wave_latent1).squeeze(dim=1)

        generated_recons = self.idwt_layer_1(generated_low1, [generated_high1])

        generated_loss_1 = (wave_low1, generated_low1.transpose(1, 2))
        generated_loss_2 = (wave_high1, generated_high1.transpose(1, 2))
        generated_loss_3 = (x, generated_recons.transpose(1, 2))

        return [generated_loss_1, generated_loss_2, generated_loss_3]
