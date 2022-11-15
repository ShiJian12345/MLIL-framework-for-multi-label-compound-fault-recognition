#!/usr/bin/env python
import random
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pdb

## 借鉴D:\code\DANet-master\encoding\nn\da_att.py
## 借鉴D:\code\Attention-mechanism-implementation-main\models\DaNet.py

class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)
        feat_b = x.view(batch_size, -1, height * width)
        feat_e = torch.bmm(attention, feat_b).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out

class Discriminator(nn.Module):
    def __init__(self, input_dim=64, output_dim=6):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        inter_channels_ = 2
        inter_channels = 8
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(inter_channels_, inter_channels, 1, padding=0, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True)
        )
        self.pam = _PositionAttentionModule(inter_channels)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 1, padding=0, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(inter_channels_, inter_channels, 1, padding=0,bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True)
        )
        self.cam = _ChannelAttentionModule()
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 1, padding=0,bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True)
        )
        same_padding = (5 - 1) // 2
        self.conv1 = nn.Conv2d(8, 16, (3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, (3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, (3, 3), padding=(1, 1))

        self.fc = nn.Sequential(
                nn.Linear(512, 64),
                nn.ReLU(),
                # nn.Dropout(0.6),
                nn.Linear(64, self.output_dim),
                # nn.LogSoftmax(dim=1)
            )

    def forward(self, x, tsne=False):
        x = x.view(-1, 2, self.input_dim, self.input_dim)  # (batch_size, 1, 4096, 1)

        # if cuda:
        #     x = x.cuda()

        position_fusion_0 = self.conv_p1(x)
        position_fusion_1 = self.pam(position_fusion_0)
        position_fusion_2 = self.conv_p2(position_fusion_1)
        channel_fusion_0 = self.conv_c1(x)
        channel_fusion_1 = self.cam(channel_fusion_0)
        channel_fusion_2 = self.conv_c2(channel_fusion_1)
        da_fusion = position_fusion_2+channel_fusion_2


        feature_conv1 = self.conv1(da_fusion)
        feature_bn1 = self.bn1(feature_conv1)
        feature_relu1 = F.relu(feature_bn1)

        # feature_pool1 = feature_relu1
        feature_pool1 = nn.MaxPool2d((2, 2))(feature_relu1)


        feature_conv2 = self.conv2(feature_pool1)
        feature_bn2 = self.bn2(feature_conv2)
        feature_relu2 = F.relu(feature_bn2)

        feature_pool2 = nn.MaxPool2d((4, 4))(feature_relu2)
        # print(x.size())

        feature_conv3 = self.conv3(feature_pool2)
        feature_bn3 = self.bn3(feature_conv3)
        feature_relu3 = F.relu(feature_bn3)

        feature_pool3 = nn.MaxPool2d((2, 2))(feature_relu3)

        feature_conv4 = self.conv4(feature_pool3)
        # x = self.bn4(x)
        feature_bn4 = F.relu(feature_conv4)

        feature_pool4 = nn.MaxPool2d((2, 2))(feature_bn4)

        # print(x.size())
        feature_input_fc = feature_pool4.view(-1, 512)


        pi1 = self.fc(feature_input_fc)
        if tsne:
            return pi1, da_fusion
        else:
            return pi1