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

class Discriminator(nn.Module):
    def __init__(self, input_dim=64, output_dim=6):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        inter_channels_ = 2


        same_padding = (5 - 1) // 2
        self.conv1 = nn.Conv2d(1, 16, (3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, (3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, (3, 3), padding=(1, 1))
        # self.lin1 = nn.Linear(512, 64)
        # # self.lin2 = nn.Linear(512, 64)
        # self.out_digit = nn.Linear(64, self.output_dim)

        self.fc = nn.Sequential(
                nn.Linear(512, 64),
                nn.ReLU(),
                # nn.Dropout(0.6),
                nn.Linear(64, self.output_dim),
                # nn.LogSoftmax(dim=1)
            )

    def forward(self, x, tsne=False):
        x = x.view(-1, 1, self.input_dim, self.input_dim)  # (batch_size, 1, 4096, 1)
        # print(x.size())

        # feature_relu1 = da_fusion
        feature_conv1 = self.conv1(x)
        feature_bn1 = self.bn1(feature_conv1)
        feature_relu1 = F.relu(feature_bn1)

        # feature_pool1 = feature_relu1
        feature_pool1 = nn.MaxPool2d((2, 2))(feature_relu1)
        # feature_input2 = feature_pool1.view(-1, 8 * 1 * 512)
        # print(feature_input2.size())

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

        # x = self.lin1(x4)
        # x = F.relu(x)
        #
        # pi1 = self.out_digit(x)

        pi1 = self.fc(feature_input_fc)

        if tsne:
            temp = self.fc[0](feature_input_fc)
            fc_1 = self.fc[1](temp)
            return pi1, fc_1
        else:
            return pi1