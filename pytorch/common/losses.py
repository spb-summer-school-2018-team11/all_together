# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


class TotalLoss(nn.Module):
    def __init__(self, loss_param, num_samples_per_classes, cuda_id):
        super(TotalLoss, self).__init__()
        self.loss_param = loss_param
        self.loss_types = list(loss_param.keys())

    def forward(self, logits, targets, emb=None, emb_norm=None, step=None, summary_writer=None):
        total_loss = 0

        if 'MSE' in self.loss_types:
            total_loss += self.loss_param['MSE']['w'] * nn.MSELoss()(logits, targets)

        if 'CCC' in self.loss_types:
            mean_true = targets.mean()
            mean_pred = logits.mean()
            cov = torch.mean((logit - mean_pred)*(target - mean_true))
            var_true = torch.var(targets)
            var_pred = torch.var(logits)
            total_loss += self.loss_param['CCC']['w'] * (1. - 2.*cov/(var_true + var_pred + (mean_true - mean_pred).pow(2)))
        # TODO: реализуйте objective function из статьи https://arxiv.org/pdf/1704.08619.pdf

        return total_loss
