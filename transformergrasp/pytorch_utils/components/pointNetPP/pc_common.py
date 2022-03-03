import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[3]
sys.path.append(str(root))
import torch.nn.parallel
from pytorch_utils.components.dataUtils import *
from pytorch_utils.components.torch_cluster_sampling import *


def query_ball_point(radius_, n_sample, xyz, xyz_fps):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample], and [B,S, nsample,3]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = xyz_fps.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    # sqrdists = DataUtils.square_distance(xyz_fps, xyz)
    c_dists = torch.cdist(x1=xyz_fps, x2=xyz, p=2)
    group_idx[c_dists > radius_] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :n_sample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, n_sample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]  # repeat the same points

    # idx_base = torch.arange(0, B, device=xyz.device).view(-1, 1, 1) * N
    # idx = group_idx + idx_base
    # idx = idx.view(-1)
    # new_x = x.view(B * N, -1)[idx, :]
    # select_x = new_x.view(B, S, n_sample, C)
    return group_idx


def get_index_points(xyz, xyz_fps, group_idx, n_sample):
    B, N, C = xyz.shape
    _, S, _ = xyz_fps.shape
    idx_base = torch.arange(0, B, device=xyz.device).view(-1, 1, 1) * N
    idx = group_idx + idx_base
    idx = idx.reshape(-1)
    new_x = xyz.reshape(B * N, C)[idx, :]
    select_x = new_x.view(B, S, n_sample, C)
    return select_x


# radius_, n_sample,
def sampling_and_group(xyz, xyz_fps, features, radius_, n_sample):
    group_idx = query_ball_point(radius_, n_sample, xyz, xyz_fps)
    fps_ball_points = get_index_points(xyz, xyz_fps, group_idx, n_sample)
    relative_points = fps_ball_points - xyz_fps.view(xyz.shape[0], xyz_fps.shape[1], 1, xyz.shape[2])

    if features is not None:
        fps_ball_features = get_index_points(features, xyz_fps, group_idx, n_sample)
        group_points = torch.cat([relative_points, fps_ball_features], dim=-1)
    else:
        group_points = relative_points
    return group_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

# def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
#     """
#     Input:
#         npoint:
#         radius:
#         nsample:
#         xyz: input points position data, [B, N, 3]
#         points: input points data, [B, N, D]
#     Return:
#         new_xyz: sampled points position data, [B, npoint, nsample, 3]
#         new_points: sampled points data, [B, npoint, nsample, 3+D]
#     """
#     B, N, C = xyz.shape
#     S = npoint
#     # TODO remove to fps
#     fps_idx = farthest_point_sampling(xyz, npoint)  # [B, npoint, C] --> npoint is the number of fps value
#     new_xyz = DataUtils.index_points(xyz, fps_idx)
#
#     idx = query_ball_point(radius, nsample, xyz, new_xyz)  # nsample --> number of knn
#     grouped_xyz = DataUtils.index_points(xyz, idx)  # [B, npoint, nsample, C]
#     grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
#
#     if points is not None:
#         grouped_points = DataUtils.index_points(points, idx)
#         new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
#     else:
#         new_points = grouped_xyz_norm
#     if returnfps:
#         return new_xyz, new_points, grouped_xyz, fps_idx
#     else:
#         return new_xyz, new_points
