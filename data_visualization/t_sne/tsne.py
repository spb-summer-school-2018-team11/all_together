import torch
import torch.autograd
from torch import nn

class TSNE(nn.Module):
    def __init__(self, n_points, n_dim):
        self.n_points = n_points
        self.n_dim = n_dim
        super(TSNE, self).__init__()
        # Logit of datapoint-to-topic weight
        self.logits = nn.Embedding(n_points, n_dim)

    def forward(self, pij, i, j):
        # TODO: реализуйте вычисление матрицы сходства для точек отображения и расстояние Кульбака-Лейблера
        # pij - значения сходства между точками данных
        # i, j - индексы точек
        '''
        y = self.logits.weight
        y_2 = y.pow(2)
        y_2_v = y_2.sum(dim=1, keepdim=True)
        y_2_h = y_2_v.t()
        dij = y_2_v + y_2_h - 2*torch.matmul(y, y.t())
        dij = (1. + dij).pow(-1)
        D = dij.sum() - dij.shape[0]
        qij = dij[i.long(), j.long()]/D
        loss_kld = (pij * (torch.log(pij) - torch.log(qij)))
        return loss_kld.sum()
        '''
        x = self.logits.weight
        A = x[i.long()]
        B = x[j.long()]
        num = (1. + (A - B).pow(2).sum(1)).pow(-1.0)
        MA = x.expand(self.n_points, self.n_points, self.n_dim)
        # MB = MA.t()
        MB = torch.transpose(MA, 0, 1)
        denom = (1. + (MA - MB).pow(2).sum(2)).pow(-1.0).view(-1).sum()
        denom -= pij.shape[0]
        qij = num / denom
        loss_kld = (pij * (torch.log(pij) - torch.log(qij))).sum()
        return loss_kld

    def __call__(self, *args):
        return self.forward(*args)