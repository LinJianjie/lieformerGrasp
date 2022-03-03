import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))
import torch.nn as nn
from pytorch_utils.components.externs_tools.chamfer_distance_topnet import ChamferDistance
from pytorch_utils.components.externs_tools.earth_mover_distance import EarthMoverDistance
from pytorch_utils.components.externs_tools.msn_emd import emdModule


class EMDLoss(nn.Module):
    def __init__(self, use_msn=False):
        super(EMDLoss, self).__init__()
        if use_msn:
            print("using use_msn emd_loss: ", use_msn)
            self.emd_dist = emdModule()
        else:
            self.emd_dist = EarthMoverDistance()

    def forward(self, pcs1, pcs2):
        # assert pcs1.shape == pcs2.shape
        assert pcs1.shape[2] == 3
        return self.emd_dist(pcs1, pcs2)


class CDLoss(nn.Module):
    """
    Chamfer Distance Batch Loss - only available for CUDA, no cpu support given
    Computes CD based average loss for batch of pcs
    Adapted from https://github.com/chrdiller/pyTorchChamferDistance
    """

    def __init__(self):
        super(CDLoss, self).__init__()
        self.chamfer_dist = ChamferDistance()

    def forward(self, pcs1, pcs2):
        """
        :param pcs1: expects [batch_size x num_points x 3] PyTorch tensor
        :param pcs2: expects [batch_size x num_points x 3] PyTorch tensor, must be same shape as pcs1
        :return: loss as float tensor
        """
        # Squared distance between each point in pcs1 to its nearest neighbour in pcs2 and vice versa
        # print("pcs1: ",pcs1.shape)
        # print("pcs2: ", pcs2.shape)
        # pcs1 = pcs1.permute(0, 2, 1)
        # pcs2 = pcs2.permute(0, 2, 1)
        # assert pcs1.shape == pcs2.shape
        assert pcs1.shape[2] == 3
        mean_dist, dist1, dist2 = self.chamfer_dist(pcs1, pcs2)
        return mean_dist


class SmoothL1Loss(nn.Module):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()
        self.loss = nn.SmoothL1Loss()

    def forward(self, x1, x2):
        return self.loss(x1, x2)


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, x1, x2):
        return self.loss(x1, x2)


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, x1, x2):
        return self.loss(x1, x2)


class NegativeLogLoss(nn.Module):
    def __init__(self):
        super(NegativeLogLoss, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.loss = nn.NLLLoss()

    def forward(self, x1, x2):
        """

        :param x1: has (N,C), C is the number of class
        :param x2: has N, where 0<x2[i]<C-1, i in [0,N-1]
        :return:
        """
        return self.loss(self.log_softmax(x1), x2)


class Metrics(object):
    def __init__(self):
        self.__loss = []

    @property
    def loss(self):
        return sum(self.__loss) / len(self.__loss)

    @loss.setter
    def loss(self, value):
        self.__loss.append(value)
