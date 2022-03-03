import sys
from pathlib import Path

import torch.nn as nn

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[3]
sys.path.append(str(root))

from pytorch_utils.components.pointNetPP.pc_common import *
from pytorch_utils.components.torch_cluster_sampling import *
from pytorch_utils.components.netUtils import NetUtil
from pytorch_utils.components.transformerNet2D.encoder_decoder import PointEncoderLayer, PointEncoder, PointEncoderXD, \
    PointEncoderLayerXD


class TransformerPointNetSetAbstractionMsg(nn.Module):
    def __init__(self, PNConfig, AttentionConfig):
        super(TransformerPointNetSetAbstractionMsg, self).__init__()
        self.AttentionConfig = AttentionConfig
        self.PNConfig = PNConfig
        self.ratio = self.PNConfig["ratio"]
        self.radius_list = self.PNConfig["radius_list"]
        self.max_sample_list = self.PNConfig["max_sample_list"]
        self.mlp_list = self.PNConfig["mlp_list"]
        self.in_channel = self.PNConfig["in_channel"]
        self.msgPointNet = nn.ModuleList()
        self.transformerPointEncoder = nn.ModuleList()
        assert len(self.radius_list) == len(self.max_sample_list)
        assert len(self.mlp_list) == len(self.max_sample_list)
        # self.conv_blocks = nn.ModuleList()
        # self.bn_blocks = nn.ModuleList()
        # for i in range(len(mlp_list)):
        #     convs = nn.ModuleList()
        #     bns = nn.ModuleList()
        #     last_channel = in_channel + 3
        #     for out_channel in mlp_list[i]:
        #         convs.append(nn.Conv2d(last_channel, out_channel, 1))
        #         bns.append(nn.BatchNorm2d(out_channel))
        #         last_channel = out_channel
        #     self.conv_blocks.append(convs)
        #     self.bn_blocks.append(bns)
        for channels in self.mlp_list:
            channels.insert(0, self.in_channel + 3)
            pN = NetUtil.SeqPointNetConv2d(channels=channels)
            self.msgPointNet.append(pN)
            encoder = PointEncoder(
                encode_layer=PointEncoderLayer(d_model=self.AttentionConfig["d_model"],
                                               d_ff=self.AttentionConfig["d_ff"],
                                               num_heads=self.AttentionConfig["num_head"],
                                               dropout=self.AttentionConfig["dropout"]),
                input_linear_transform=None,
                N=self.AttentionConfig["N"],
                use_cat=True)
            self.transformerPointEncoder.append(encoder)

    def forward(self, xyz, features):
        """
        Input:
            xyz: input points position data, [B, N, C] --> C is feature size, N ist the number of way
            points: input points data, [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B, N, C = xyz.shape
        xyz_fps, _, _ = farthest_point_sampling(x=xyz, ratio=self.ratio)
        _, S, _ = xyz_fps.shape
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.max_sample_list[i]
            # group_idx, grouped_points = query_ball_point(radius, K, xyz, xyz_fps)
            # # grouped_points = DataUtils.index_points(features, group_idx)
            # grouped_points = grouped_points - xyz_fps.view(B, grouped_points, 1, C)
            group_points = sampling_and_group(xyz=xyz, xyz_fps=xyz_fps, features=features, radius_=radius, n_sample=K)
            group_points = group_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            # print("group_points: ", group_points.shape)
            # for j in range(len(self.conv_blocks[i])):
            #     conv = self.conv_blocks[i][j]
            #     bn = self.bn_blocks[i][j]
            #     grouped_points = F.relu(bn(conv(grouped_points)))
            # print("before group_points: ", group_points.shape)
            group_points = self.msgPointNet[i](group_points)
            # print("after group_points: ", group_points.shape)
            max_features = torch.max(group_points, 2)[0]  # [B, D', S]
            new_points = self.transformerPointEncoder[i](max_features)
            # print("new_points: ", new_points.shape)
            new_points_list.append(new_points)
        # new_xyz = xyz_fps.permute(0, 2, 1)
        new_xyz = xyz_fps
        new_points_concat = torch.cat(new_points_list, dim=1).permute(0, 2, 1)
        return new_xyz, new_points_concat


class TransformerPointNetSetAbstractionMsg3D(nn.Module):
    def __init__(self, PNConfig, AttentionConfig):
        super(TransformerPointNetSetAbstractionMsg3D, self).__init__()
        self.AttentionConfig = AttentionConfig
        self.PNConfig = PNConfig
        self.ratio = self.PNConfig["ratio"]
        self.radius_list = self.PNConfig["radius_list"]
        self.max_sample_list = self.PNConfig["max_sample_list"]
        self.mlp_list = self.PNConfig["mlp_list"]
        self.in_channel = self.PNConfig["in_channel"]
        self.msgPointNet = nn.ModuleList()
        self.transformerPointEncoder = nn.ModuleList()
        assert len(self.radius_list) == len(self.max_sample_list)
        assert len(self.mlp_list) == len(self.max_sample_list)
        # self.conv_blocks = nn.ModuleList()
        # self.bn_blocks = nn.ModuleList()
        # for i in range(len(mlp_list)):
        #     convs = nn.ModuleList()
        #     bns = nn.ModuleList()
        #     last_channel = in_channel + 3
        #     for out_channel in mlp_list[i]:
        #         convs.append(nn.Conv2d(last_channel, out_channel, 1))
        #         bns.append(nn.BatchNorm2d(out_channel))
        #         last_channel = out_channel
        #     self.conv_blocks.append(convs)
        #     self.bn_blocks.append(bns)
        for channels in self.mlp_list:
            channels.insert(0, self.in_channel + 3)
            pN = NetUtil.SeqPointNetConv2d(channels=channels)
            self.msgPointNet.append(pN)
            encoder = PointEncoderXD(
                encode_layer=PointEncoderLayerXD(d_model=self.AttentionConfig["d_model"],
                                                 d_ff=self.AttentionConfig["d_ff"],
                                                 num_heads=self.AttentionConfig["num_head"],
                                                 dropout=self.AttentionConfig["dropout"],
                                                 choose_XD=3),
                input_linear_transform=None,
                N=self.AttentionConfig["N"],
                use_cat=True,
                choose_XD=3)
            self.transformerPointEncoder.append(encoder)

    def forward(self, xyz, features):
        """
        Input:
            xyz: input points position data, [B, N, C] --> C is feature size, N ist the number of way
            points: input points data, [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B, N, C = xyz.shape
        xyz_fps, _, _ = farthest_point_sampling(x=xyz, ratio=self.ratio)
        _, S, _ = xyz_fps.shape
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.max_sample_list[i]
            # group_idx, grouped_points = query_ball_point(radius, K, xyz, xyz_fps)
            # # grouped_points = DataUtils.index_points(features, group_idx)
            # grouped_points = grouped_points - xyz_fps.view(B, grouped_points, 1, C)
            group_points = sampling_and_group(xyz=xyz, xyz_fps=xyz_fps, features=features, radius_=radius, n_sample=K)
            group_points = group_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            # print("group_points: ", group_points.shape)
            # for j in range(len(self.conv_blocks[i])):
            #     conv = self.conv_blocks[i][j]
            #     bn = self.bn_blocks[i][j]
            #     grouped_points = F.relu(bn(conv(grouped_points)))
            # print("before group_points: ", group_points.shape)
            group_points = self.msgPointNet[i](group_points)
            new_points = self.transformerPointEncoder[i](group_points)
            max_features = torch.max(new_points, 2)[0]  # [B, D', S]
            # print("new_points: ", new_points.shape)
            new_points_list.append(max_features)
        # new_xyz = xyz_fps.permute(0, 2, 1)
        new_xyz = xyz_fps
        new_points_concat = torch.cat(new_points_list, dim=1).permute(0, 2, 1)
        return new_xyz, new_points_concat
