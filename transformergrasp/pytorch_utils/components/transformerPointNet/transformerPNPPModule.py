import sys
from pathlib import Path

import torch
import torch.nn as nn

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[3]
sys.path.append(str(root))
from pytorch_utils.components.transformerPointNet.utils import TransformerPointNetSetAbstractionMsg, \
    TransformerPointNetSetAbstractionMsg3D
from pytorch_utils.components.netUtils import NetUtil


class TransformerPointNetPPHead(nn.Module):
    def __init__(self, config):
        super(TransformerPointNetPPHead, self).__init__()
        self.config = config
        self.SA_modules = nn.ModuleList()
        self.num_sa_modules = len(self.config["PNConfig"])
        self.first_layer = NetUtil.SeqPointNetConv1d(channels=self.config["PointNet"]["channels"])
        for i in range(self.num_sa_modules):
            sa1 = TransformerPointNetSetAbstractionMsg(PNConfig=self.config["PNConfig"][i],
                                                       AttentionConfig=self.config["AttentionConfig"][i])

            self.SA_modules.append(sa1)
        self.down_sample_xyz = None

    def forward(self, xyz):
        features = self.first_layer(xyz.permute(0, 2, 1)).permute(0, 2, 1)
        for i in range(self.num_sa_modules):
            new_xyz, new_features = self.SA_modules[i](xyz, features)
            features = new_features
            xyz = new_xyz
            # print("new_xyz: ", xyz.shape)
            # print("new_features: ", new_features.shape)
        self.down_sample_xyz = xyz
        if self.config["group_all"]:
            return torch.cat([xyz, features], dim=-1)
        else:
            return features


class TransformerPointNetPPHead3D(nn.Module):
    def __init__(self, config):
        super(TransformerPointNetPPHead3D, self).__init__()
        self.config = config
        self.SA_modules = nn.ModuleList()
        self.num_sa_modules = len(self.config["PNConfig"])
        self.first_layer = NetUtil.SeqPointNetConv1d(channels=self.config["PointNet"]["channels"])
        for i in range(self.num_sa_modules):
            sa1 = TransformerPointNetSetAbstractionMsg3D(PNConfig=self.config["PNConfig"][i],
                                                         AttentionConfig=self.config["AttentionConfig"][i])

            self.SA_modules.append(sa1)
        self.down_sample_xyz = None

    def forward(self, xyz):
        features = self.first_layer(xyz.permute(0, 2, 1)).permute(0, 2, 1)
        for i in range(self.num_sa_modules):
            new_xyz, new_features = self.SA_modules[i](xyz, features)
            features = new_features
            xyz = new_xyz
            # print("new_xyz: ", xyz.shape)
            # print("new_features: ", new_features.shape)
        self.down_sample_xyz = xyz
        if self.config["group_all"]:
            return torch.cat([xyz, features], dim=-1)
        else:
            return features
