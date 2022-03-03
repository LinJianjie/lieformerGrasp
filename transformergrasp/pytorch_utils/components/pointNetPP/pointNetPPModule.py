import sys

from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[3]
sys.path.append(str(root))
from pytorch_utils.components.pointNetPP.pointnet_util import *
import torch.nn.functional as F


class PointNetPPClassification(nn.Module):
    def __init__(self, num_class, config):
        super(PointNetPPClassification).__init__()
        self.num_class = num_class
        self.config = config
        self.sa1 = PointNetSetAbstractionMsg(ratio=self.config["sa1"]["ratio"],
                                             radius_list=self.config["sa1"]["radius_list"],
                                             max_sample_list=self.config["sa1"]["max_sample_list"],
                                             mlp_list=self.config["sa1"]["mlp_list"])

        self.sa2 = PointNetSetAbstractionMsg(ratio=self.config["sa2"]["ratio"],
                                             radius_list=self.config["sa2"]["radius_list"],
                                             max_sample_list=self.config["sa2"]["max_sample_list"],
                                             mlp_list=self.config["sa2"]["mlp_list"])

        self.sa3 = PointNetSetAbstractionMsg(ratio=self.config["sa3"]["ratio"],
                                             radius_list=self.config["sa3"]["radius_list"],
                                             max_sample_list=self.config["sa3"]["max_sample_list"],
                                             mlp_list=self.config["sa3"]["mlp_list"])
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, self.num_class)

    def forward(self, xyz, features):
        B, _, _ = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz, features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        return x, l3_points


class PointNetPPCHead(nn.Module):
    def __init__(self, config):
        super(PointNetPPCHead, self).__init__()
        self.config = config
        self.SA_modules = nn.ModuleList()
        self.num_sa_modules = len(self.config["SA_modules"])
        for i in range(self.num_sa_modules):
            sa1 = PointNetSetAbstractionMsg(ratio=self.config["SA_modules"][i]["ratio"],
                                            radius_list=self.config["SA_modules"][i]["radius_list"],
                                            max_sample_list=self.config["SA_modules"][i]["max_sample_list"],
                                            in_channel=self.config["SA_modules"][i]["in_channel"],
                                            mlp_list=self.config["SA_modules"][i]["mlp_list"])

            self.SA_modules.append(sa1)
        self.down_sample_xyz = None

    def forward(self, xyz):
        features = None
        for i in range(self.num_sa_modules):
            new_xyz, new_features = self.SA_modules[i](xyz, features)
            features = new_features
            xyz = new_xyz
        self.down_sample_xyz = xyz
        if self.config["group_all"]:
            return torch.cat([xyz, features], dim=-1)
        else:
            return features
