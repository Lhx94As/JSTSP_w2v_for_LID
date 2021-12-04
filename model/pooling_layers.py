import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
import argparse


class LDEPooling(torch.nn.Module):
    """A novel learnable dictionary encoding layer.
    Reference: Weicheng Cai, etc., "A NOVEL LEARNABLE DICTIONARY ENCODING LAYER FOR END-TO-END
               LANGUAGE IDENTIFICATION", icassp, 2018
    """
    def __init__(self, input_dim, c_num=64, eps=1.0e-10):
        super(LDEPooling, self).__init__()

        self.input_dim = input_dim
        self.output_dim = input_dim * c_num
        self.eps = eps

        self.mu = torch.nn.Parameter(torch.randn(input_dim, c_num))
        self.s = torch.nn.Parameter(torch.ones(c_num))

        self.softmax_for_w = torch.nn.Softmax(dim=3)

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        r = inputs.transpose(1,2).unsqueeze(3) - self.mu
        # Make sure beta=self.s**2+self.eps > 0
        w = self.softmax_for_w(- (self.s**2 + self.eps) * torch.sum(r**2, dim=2, keepdim=True))
        e = torch.mean(w * r, dim=1)

        return e.reshape(-1, self.output_dim)

    def get_output_dim(self):
        return self.output_dim

# Attention Statistics Pooling
class Attensive_statistics_pooling(nn.Module):
    def __init__(self, inputdim, outputdim, attn_dropout=0.0):
        super(Attensive_statistics_pooling, self).__init__()
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.attn_dropout = attn_dropout
        self.linear_projection = nn.Linear(inputdim, outputdim)
        self.v = torch.nn.Parameter(torch.randn(outputdim))

    def weighted_sd(self, inputs, attention_weights, mean):
        el_mat_prod = torch.mul(inputs, attention_weights.unsqueeze(2).expand(-1, -1, inputs.shape[-1]))
        hadmard_prod = torch.mul(inputs, el_mat_prod)
        variance = torch.sum(hadmard_prod, 1) - torch.mul(mean, mean)
        return variance

    def stat_attn_pool(self, inputs, attention_weights):
        el_mat_prod = torch.mul(inputs, attention_weights.unsqueeze(2).expand(-1, -1, inputs.shape[-1]))
        mean = torch.mean(el_mat_prod, dim=1)
        variance = self.weighted_sd(inputs, attention_weights, mean)
        stat_pooling = torch.cat((mean, variance), 1)
        return stat_pooling

    def forward(self,inputs):
        inputs = inputs.transpose(1,2)
        # print("input shape: {}".format(inputs.shape))
        lin_out = self.linear_projection(inputs)
        # print('lin_out shape:',lin_out.shape)
        v_view = self.v.unsqueeze(0).expand(lin_out.size(0), len(self.v)).unsqueeze(2)
        # print("v's shape after expand:",v_view.shape)
        attention_weights = F.relu(lin_out.bmm(v_view).squeeze(2))
        # print("attention weight shape:",attention_weights.shape)
        attention_weights = F.softmax(attention_weights, dim=1)
        statistics_pooling_out = self.stat_attn_pool(inputs, attention_weights)
        # print(statistics_pooling_out.shape)
        return statistics_pooling_out

    def get_output_dim(self):
        return self.inputdim*2



class SElayer(nn.Module):
    def __init__(self, dim, seq_len, reduction = 16):
        super(SElayer, self).__init__()
        self.pooling = torch.nn.AvgPool1d(kernel_size=seq_len)
        self.squeeze_excitation = nn.Sequential(nn.Linear(dim, dim//reduction, bias=False),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(dim//reduction, dim, bias=False),
                                                nn.Sigmoid())
    def forward(self, x):
        weight = self.pooling(x.transpose(1,2)).squeeze(-1)
        weight = self.squeeze_excitation(weight).unsqueeze(1)
        out = x*weight
        return out, weight.squeeze(1)


class SElayer_random(nn.Module):
    def __init__(self, dim, reduction = 16):
        super(SElayer_random, self).__init__()
        self.squeeze_excitation_1 = nn.Sequential(nn.Linear(dim, dim // reduction, bias=False),
                                                  nn.ReLU(inplace=True))
        self.squeeze_excitation_2 = nn.Sequential(nn.Linear(dim // reduction, dim, bias=False),
                                                  nn.Sigmoid())

    def forward(self, x, seq_weights=1):
        # print(x.size(), seq_weights.size())
        weight = x.mean(dim=1).transpose(0,1)*seq_weights
        weight = weight.transpose(0,1)
        weight_mid = self.squeeze_excitation_1(weight)
        weight = self.squeeze_excitation_2(weight_mid).unsqueeze(1)
        out = x*weight
        return out, weight.squeeze(1), weight_mid

if __name__ == "__main__":
    asp = Attensive_statistics_pooling(inputdim=4, outputdim=5)
