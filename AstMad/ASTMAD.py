# -*- coding: utf-8 -*-
# @Time    : 2021/8/13 15:05
# @Author  : SSK
# @Email   : skshao@bjtu.edu.cn
# @File    : ASTMAD.py
# @Software: PyCharm
from torch import nn
from .net import EmbeddingGraph, STGCNEncoder, STGCNDecoder
import torch


class ASTMAD(nn.Module):
    def __init__(self, seq_len, input_channel, embedding_dim=64, top_k=30, input_node_dim=1, conv_node_dim=16, conv_layers=3, graph_alpha=3, device=torch.device('cuda:1')):
        super(ASTMAD, self).__init__()

        self.layers = conv_layers
        self.graph = EmbeddingGraph(input_channel, embedding_dim, top_k, graph_alpha, device)
        self.conv_start = nn.Conv2d(in_channels=input_node_dim, out_channels=conv_node_dim, kernel_size=(1, 1))
        self.stencoder = STGCNEncoder(st_layers=1, input_node_channel=16, conv_node_channel=16, end_node_channel=32, output_node_channel=64, gcn_depth=1, hopalpha=0.05, dropout=0.2)
        self.stdecoder = STGCNDecoder(st_layers=1, input_node_channel=64, conv_node_channel=64, end_node_channel=32, output_node_channel=16, gcn_depth=1, hopalpha=0.05, dropout=0.2)
        self.conv_end = nn.Conv2d(in_channels=conv_node_dim, out_channels=input_node_dim, kernel_size=(1, 1))

    def forward(self, x, idx):
        x = torch.unsqueeze(x, dim=1)
        x = x.transpose(2, 3)
        adj = self.graph(idx)
        x = self.conv_start(x) # batch × 1 × node × window -> batch × 16 × node × window
        z = self.stencoder(x, adj, training=True)  # 512*16*55*128 -> 512*64*55*16 batch×node_feature×node×window
        x = self.stdecoder(z, adj, training=True)
        x = self.conv_end(x)
        x = torch.squeeze(x, dim=1)
        x = x.transpose(1, 2)
        return x


class DataDiscriminator(nn.Module):
    def __init__(self, window_size, input_channel, hidden_size):
        super(DataDiscriminator, self).__init__()

        self.start_dis_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_channel, out_channels=1, kernel_size=1),
            nn.BatchNorm1d(1),
            nn.ReLU(True)
        )
        self.pred_layers = nn.Sequential(
            nn.Linear(window_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):

        x = x.squeeze(dim=1)
        x = self.start_dis_layers(x)
        x = x.squeeze(dim=1)
        out = self.pred_layers(x)
        return out


