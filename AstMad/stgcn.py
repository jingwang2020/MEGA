# -*- coding: utf-8 -*-
# @Time    : 2021/8/13 15:26
# @Author  : SSK
# @Email   : skshao@bjtu.edu.cn
# @File    : stgcn.py
# @Software: PyCharm
from torch import nn
import torch


class mix(nn.Module):
    def __init__(self, input_channel, output_channel, bias=True):
        super(mix, self).__init__()
        self.mlp = torch.nn.Conv2d(input_channel, output_channel, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self):
        super(gcn, self).__init__()

    def forward(self, x, adj):
        x = torch.einsum('ncwl,vw->ncvl', (x, adj))  # batch × node_feature × node × window
        return x.contiguous()


class sgcn(nn.Module):
    def __init__(self, input_channel, output_channel, gcn_depth, hop_alpha):
        super(sgcn, self).__init__()
        self.mix = mix((gcn_depth+1)*input_channel, output_channel)
        self.gcn = gcn()
        self.depth = gcn_depth
        self.alpha = hop_alpha
        self.batch_norm = nn.BatchNorm2d(output_channel)
        self.activation = nn.LeakyReLU()

    def forward(self, input, adj):
        adj = adj + torch.eye(adj.size(0)).to(input.device)
        d = adj.sum(1)
        h = input
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.depth):
            h = self.alpha * input + (1 - self.alpha) * self.activation(self.batch_norm(self.gcn(h, a)))
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mix(ho)
        return ho


class en_tcn(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(en_tcn, self).__init__()
        # self.tcn_block = nn.Sequential(
        #     nn.Conv2d(input_channel, output_channel, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4)),
        #     nn.BatchNorm2d(output_channel),
        #     nn.ReLU(inplace=True)
        # )
        self.tcn_block = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.tcn_block(x)


class de_tcn(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(de_tcn, self).__init__()
        self.tcn_block = nn.Sequential(
            nn.ConvTranspose2d(input_channel, output_channel, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.tcn_block(x)
