# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch.common.losses import *
from collections import OrderedDict


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, data):
        x = data
        for name, module in self.submodule._modules.items():
            if len(module._modules.items()) != 0:
                for name2, module2 in module._modules.items():
                    x = module2(x)
            else:
                x = module(x)
        return x

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1) # x.size(0) - размер батча

class CNNNet(nn.Module):
    def __init__(self, num_classes, depth, data_size, emb_name=[], pretrain_weight=None):
        super(CNNNet, self).__init__()
        sample_size = data_size['width']
        sample_duration = data_size['depth']

        # TODO: Реализуйте архитектуру нейронной сети
        module = nn.Sequential()

        module.add_module('conv1', nn.Conv3d(3, 16, 5, 1, 2))
        #module.add_module('pool1', nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        module.add_module('pool1', nn.AdaptiveAvgPool3d(1))
        module.add_module('conv2', nn.Conv3d(16, 32, 5, 1, 2))
        module.add_module('pool2', nn.AdaptiveAvgPool3d(1))
        #module.add_module('pool2', nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        #module.add_module('conv3', nn.Conv3d(32, 48, 3, 1, 1))
        #module.add_module('pool3', nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        module.add_module('flatten', Flatten())
        #module.add_module('linear1', nn.Linear(16, 16))
        module.add_module('linear', nn.Linear(32, num_classes))

        self.net = module
        #net = []
        #self.net = FeatureExtractor(net, emb_name)

    def forward(self, data):
        # output = self.net(torch.squeeze(data, 2))
        output = self.net(data)
        return output


import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms

class Resnet152(nn.Module):
    def __init__(self, embedding_dim=512, pretrained=False):
        super(Resnet101, self).__init__()
        self.embedding_dim = embedding_dim
        self.resnet152 = models.resnet152(pretrained=pretrained)
        self.linear = nn.Linear(self.resnet152.fc.in_features, embedding_dim)
        self.resnet152.fc = self.linear
        # self.batch_norm = nn.BatchNorm1d(embedding_dim, momentum=0.01)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, images):
        embed = self.resnet152(torch.squeeze(images, 2))
        # embed = self.batch_norm(embed)
        return embed

import torch
import torchvision
import torch.nn as nn
import torchvision.models as models

class Alexnet(nn.Module):
	def __init__(self, embedding_dim = 32, pretrained = False):
		super(Alexnet, self).__init__()
		self.embedding_dim = embedding_dim
		self.alexnet = models.alexnet(pretrained=pretrained)
		in_features = self.alexnet.classifier[6].in_features
		self.linear = nn.Linear(in_features, embedding_dim)
		self.alexnet.classifier[6] = self.linear
		self.init_weights()

	def init_weights(self):
		self.linear.weight.data.normal_(0.0, 0.02)
		self.linear.bias.data.fill_(0)

	def forward(self, images):
		embed = self.alexnet(torch.squeeze(images, 2))
		return embed

