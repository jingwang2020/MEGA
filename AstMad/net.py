# -*- coding: utf-8 -*-
# @Time    : 2021/8/13 10:50
# @Author  : SSK
# @Email   : skshao@bjtu.edu.cn
# @File    : net.py
# @Software: PyCharm
from torch import nn
import torch
import torch.nn.functional as F
from .stgcn import sgcn, en_tcn, de_tcn
from .wavemodule import Dwtconv, Transition, Bottleneck, SingleLayer, DWTCompose, IDWT


class NodeGraph(nn.Module):
    def __init__(self, node_size, node_dim, sequence_len, batch_size, top_k, alpha, device):
        super(NodeGraph, self).__init__()
        # self.linear1 = nn.Linear(node_dim, node_dim)
        # self.linear2 = nn.Linear(node_dim, node_dim)
        self.node_size = node_size
        self.embedding1 = NodeEmbedding(node_size, node_dim, sequence_len, batch_size)
        self.embedding2 = NodeEmbedding(node_size, node_dim, sequence_len, batch_size)
        self.top_k = top_k
        self.alpha = alpha
        self.device = device
        # self.LeReLu = nn.LeakyReLU()

    def forward(self, low_frequency, high_frequency):
        vec1 = self.embedding1(low_frequency, high_frequency)
        vec2 = self.embedding2(low_frequency, high_frequency)

        matrix = torch.mm(vec1, vec2.transpose(1, 0)) - torch.mm(vec2, vec1.transpose(1, 0))

        adj = F.relu(torch.tanh(self.alpha * matrix))
        mask = torch.zeros(self.node_size, self.node_size).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.top_k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj


class NodeEmbedding(nn.Module):
    def __init__(self, node_size, node_dim=64, sequence_len=64, batch_size=128):
        """
            By entering a frequency query as the key of the query dictionary, the corresponding node vector representation in the dictionary is obtained. 25*64
        """
        super(NodeEmbedding, self).__init__()
        self.node_size = node_size
        self.low_memory_bank = nn.Parameter(torch.randn(node_size, node_dim))
        self.high_memory_bank = nn.Parameter(torch.randn(node_size, node_dim))

        self.query_net_low = DenseQuery(batch_size=batch_size, sequence_len=sequence_len, hidden_size=64, latent_size=node_size)
        self.query_net_high = DenseQuery(batch_size=batch_size, sequence_len=sequence_len, hidden_size=64, latent_size=node_size)

    def forward(self, low_frequency, high_frequency):
        # frequency input shape batch*length*node 128*64*25
        low_frequency = low_frequency.permute(2, 0, 1)
        low_frequency = low_frequency.contiguous().view(self.node_size, -1)
        high_frequency = high_frequency.permute(2, 0, 1)
        high_frequency = high_frequency.contiguous().view(self.node_size, -1)

        query_low = self.query_net_low(low_frequency)
        query_low = F.softmax(query_low, dim=-1)
        query_high = self.query_net_high(high_frequency)
        query_high = F.softmax(query_high, dim=-1)

        node_low = F.leaky_relu(torch.mm(query_low, self.low_memory_bank))
        # print("node-low::::::::::::")
        # print(node_low)
        node_high = F.leaky_relu(torch.mm(query_high, self.high_memory_bank))
        # print("node-high::::::::::::")
        # print(node_high)
        node_represent = 3*node_low + 3*node_high
        # print("")
        # node_represent = F.linear(query_low, (self.low_memory_bank).T) + F.linear(query_high, (self.high_memory_bank).T)
        # node_represent shape 25*64
        return node_represent


class DenseQuery(nn.Module):
    def __init__(self, batch_size, sequence_len, hidden_size, latent_size):
        super(DenseQuery, self).__init__()
        input_size = batch_size*sequence_len
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(inplace=True),
            # nn.Linear(hidden_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, latent_size)
        )

    def forward(self, x):
        out = self.layers(x)
        return out


class EmbeddingGraph(nn.Module):
    """
        Return an asymmetric adjacency matrix
    """
    def __init__(self, nodes, dim, top_k, alpha, device):
        super(EmbeddingGraph, self).__init__()
        self.embedding1 = nn.Embedding(nodes, dim)
        self.embedding2 = nn.Embedding(nodes, dim)
        self.top_k = top_k
        self.alpha = alpha
        self.device = device
        self.LeReLu = nn.LeakyReLU()

    def forward(self, index_input):
        vec1 = self.embedding1(index_input)
        vec2 = self.embedding2(index_input)

        # matrix 1:
        matrix = torch.mm(vec1, vec2.transpose(1, 0)) - torch.mm(vec2, vec1.transpose(1, 0))
        adj = self.LeReLu(torch.tanh(self.alpha*matrix))
        # matrix 2:
        mask = torch.zeros(index_input.size(0), index_input.size(0)).to(self.device)
        mask.fill_(float('0'))
        # rand_like filled with random numbers uniformly distributed on the interval [0,1)
        # topk：Returns the k largest values of the input tensor input along the given dim dimension, returning a tuple (values,indices)
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.top_k, 1)
        # scatter_ modifying the mask matrix simply means modifying another tensor through a tensor src. Which element needs to be modified and which element in src is used to modify is determined by dim and index
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj

class STGCNEncoder(nn.Module):
    def __init__(self, st_layers=2, input_node_channel=16, conv_node_channel=16, end_node_channel=32, output_node_channel=64, gcn_depth=2, hopalpha=0.05, dropout=0.2):
        """
            Dimensions of latent space--the number of layers of ST convolution
        """
        super(STGCNEncoder, self).__init__()
        self.layers = st_layers

        self.en_tc = nn.ModuleList()
        self.en_gc = nn.ModuleList()
        self.dropout = dropout
        # self.norm = nn.ModuleList()

        for i in range(st_layers):

            self.en_tc.append(en_tcn(input_node_channel, conv_node_channel))
            self.en_gc.append(sgcn(conv_node_channel, input_node_channel, gcn_depth, hopalpha))

        self.en_output_conv1 = nn.Conv2d(input_node_channel, end_node_channel, (1,1))
        self.en_output_conv2 = nn.Conv2d(end_node_channel, output_node_channel, (1,1))

    def forward(self, x, adj, training=False):
        for i in range(self.layers):
            residual = x
            tc_output = self.en_tc[i](x)
            x = F.dropout(tc_output, self.dropout, training=training)
            gc_output = self.en_gc[i](x, adj)
            x = torch.relu(gc_output)
            x = x+residual[:, :, :, -x.size(3):]
        stc_output = F.relu(self.en_output_conv1(x))
        return self.en_output_conv2(stc_output)


class STGCNDecoder(nn.Module):
    def __init__(self, st_layers=3, input_node_channel=64, conv_node_channel=64, end_node_channel=32, output_node_channel=16, gcn_depth=2, hopalpha=0.05, dropout=0.2):
        super(STGCNDecoder, self).__init__()
        self.layers = st_layers
        self.de_tc = nn.ModuleList()
        self.de_gc = nn.ModuleList()
        self.dropout = dropout

        self.upsampling = nn.Upsample(scale_factor=(1, ), mode='bilinear')
        for i in range(st_layers):
            self.de_tc.append(de_tcn(input_node_channel, conv_node_channel))
            self.de_gc.append(sgcn(conv_node_channel, input_node_channel, gcn_depth, hopalpha))

        self.de_output_conv1 = nn.Conv2d(input_node_channel, end_node_channel, (1,1))
        self.de_output_conv2 = nn.Conv2d(end_node_channel, output_node_channel, (1,1))

    def forward(self, z, adj, training=False):
        for i in range(self.layers):
            residual = z
            tc_output = self.de_tc[i](z)
            z = F.dropout(tc_output, self.dropout, training=training)
            gc_output = self.de_gc[i](z, adj)
            z = torch.relu(gc_output)
            m = nn.Upsample(scale_factor=(1, z.shape[3]/residual.shape[3]), mode='bilinear')
            residual = m(residual)
            z = z+residual

        stc_output = F.relu(self.de_output_conv1(z))
        return self.de_output_conv2(stc_output)


class DenseWave(nn.Module):
    def __init__(self, input_channel, wave_channel1, wave_channel2, growthRate, nDenseBlocks):
        super(DenseWave, self).__init__()
        self.dwt = Dwtconv(input_channel, wave_channel1, wave_channel2)
        self.conv1 = nn.Conv1d(input_channel, wave_channel1, kernel_size=3, bias=True, padding=1, stride=1)
        self.dense1 = self._make_dense(wave_channel1, growthRate=growthRate, nDenseBlocks=nDenseBlocks, bottleneck=True)
        self.trans1 = Transition(wave_channel1+growthRate*nDenseBlocks, wave_channel1)
        self.dense2 = self._make_dense(wave_channel1, growthRate=growthRate, nDenseBlocks=nDenseBlocks, bottleneck=True)
        self.trans2 = Transition(wave_channel1+growthRate*nDenseBlocks, wave_channel2)

        self.bn1 = nn.BatchNorm1d(wave_channel2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(wave_channel2, input_channel, kernel_size=1)
        self.bn_out = nn.BatchNorm1d(input_channel)
        self.relu_out = nn.ReLU()

        self.out_conv = nn.ConvTranspose1d(input_channel, input_channel, kernel_size=6, padding=1, stride=4)

    def _make_dense(self, nChannels, growthRate=2, nDenseBlocks=6, bottleneck=True):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):  # x.dim：batch × window × node
        x = x.permute(0, 2, 1)
        out = self.conv1(x)
        out_dwt0, out_dwt1 = self.dwt(x)
        out = self.dense1(out)
        out = self.trans1(out)
        out = self.dense2(torch.add(out_dwt0, out))
        out = self.trans2(out)

        out = self.conv2(self.relu1(self.bn1(torch.add(out_dwt1, out))))
        out = self.relu_out(self.bn_out(out))

        out = self.out_conv(out)

        out = out.permute(0, 2, 1)
        return out


class WaveDecompose(nn.Module):
    """
        Multilevel DWT
        return frequency components
    """
    def __init__(self, input_channel):
        super(WaveDecompose, self).__init__()
        self.dwt = DWTCompose(input_channel)

    def forward(self, x):
        dwt = self.dwt(x)
        return dwt


class IDWTLayers(nn.Module):
    def __init__(self):
        super(IDWTLayers, self).__init__()
        self.idwt = IDWT()

    def forward(self, low, high):
        return self.idwt(low, high)


class WaveGCN(nn.Module):
    """
        Input three different levels of Wave input,
        return three GCN outputs
    """
    def __init__(self, gcn_depth=2, hopalpha=0.05):
        super(WaveGCN, self).__init__()
        self.gcn1 = sgcn(8, 8, gcn_depth, hopalpha)
        self.gcn2 = sgcn(16, 16, gcn_depth, hopalpha)
        self.gcn3 = sgcn(32, 32, gcn_depth, hopalpha)

        self.gc1_conv_end = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=(1, 1))
        self.gc2_conv_end = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(1, 1))
        self.gc3_conv_end = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1, 1))

        self.gc1_out1 = nn.Conv1d(in_channels=64, out_channels=16, kernel_size=1)
        self.gc1_out2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=1)

    def forward(self, wave_feature1, wave_feature2, wave_feature3, adj):
        gc_output1 = self.gcn1(wave_feature1, adj)
        gc_output2 = self.gcn2(wave_feature2, adj)
        gc_output3 = self.gcn3(wave_feature3, adj)

        gc_output1 = torch.relu(gc_output1)
        gc_output2 = torch.relu(gc_output2)
        gc_output3 = torch.relu(gc_output3)

        gc_output1 = self.gc1_conv_end(gc_output1)
        gc_output2 = self.gc2_conv_end(gc_output2)
        gc_output3 = self.gc3_conv_end(gc_output3)

        gc_output1 = torch.squeeze(gc_output1, dim=1)
        gc_output2 = torch.squeeze(gc_output2, dim=1)
        gc_output3 = torch.squeeze(gc_output3, dim=1)

        gc_output1 = gc_output1.transpose(1, 2)
        gc_output2 = gc_output2.transpose(1, 2)

        gc_output1 = self.gc1_out1(gc_output1)
        gc_output2 = self.gc1_out2(gc_output2)

        gc_output1 = gc_output1.transpose(1, 2)
        gc_output2 = gc_output2.transpose(1, 2)

        return [gc_output1, gc_output2, gc_output3]


class MutiLevelWaveGCN(nn.Module):
    """
        Perform GCN by layer, and the results are expressed as Latent
        Subsequent Latent will be reconstructed back to different frequency components for reconstruction loss training
        return: the result of GCN
    """
    def __init__(self, input_channel=8, gcn_depth=1, hopalpha=0.05):
        super(MutiLevelWaveGCN, self).__init__()
        self.gcn = sgcn(input_channel, input_channel, gcn_depth, hopalpha)

    def forward(self, wave_feature, adj):
        gc_output = self.gcn(wave_feature, adj)
        return gc_output
