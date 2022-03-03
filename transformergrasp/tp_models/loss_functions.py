import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
from pytorch_utils.components.lie_groups.se3 import SE3Matrix as SE3
from pytorch_utils.components.lie_groups.myse3 import SE3Matrix as My_SE3
import torch.nn as nn
import torch
import numpy as np


class TransformationLoss(nn.Module):
    def __init__(self):
        super(TransformationLoss, self).__init__()
        self.rotation_loss = nn.MSELoss()
        self.translation_loss = nn.MSELoss()
        self.transformation_loss = nn.MSELoss()
        self.T = None

    def forward(self, points_1, epsilon_1, T2_1, use_Transformation=True):
        epsilon = epsilon_1.reshape(-1, epsilon_1.shape[-1])
        points = points_1.reshape(-1, points_1.shape[-1])
        T2 = T2_1.reshape(-1, 4, 4)
        self.T = T2
        if use_Transformation:
            gt_epsilon = SE3.log(SE3.from_matrix(T2))
            loss = self.transformation_loss(epsilon, gt_epsilon)
        else:
            # pred_translation_offset = epsilon[:, :3]
            # pred_translation = points + pred_translation_offset
            #
            # pred_rotation_epsilon = epsilon[:, 3:]
            # gt_rotation_epsilon = SO3.log(SO3.from_matrix(T2[:, :3, :3]))
            #
            # loss = self.rotation_loss(pred_rotation_epsilon, gt_rotation_epsilon) + \
            #        self.translation_loss(pred_translation, T2[:, :3, 3])
            T = My_SE3.exp(epsilon)
            I = torch.eye(3, device=T.device, dtype=T.dtype).unsqueeze(dim=0).repeat(epsilon.shape[0], 1, 1)
            rotation_distance = torch.norm(I - torch.bmm(T[:, :3, :3].transpose(2, 1), T2[:, :3, :3]), p="fro")
            # delta_epsilon = My_SE3.log(torch.bmm(torch.inverse(T), T2))
            translation_distance = torch.norm(T[:, :3, 3] - T2[:, :3, 3], p="fro")
            distance = rotation_distance + translation_distance
            loss = distance
        # loss = torch.mean(gt_rotation_epsilon) + self.translation_loss(pred_translation, T2[:, :3, 3])
        # loss = torch.mean((epsilon - delta_epsilon) ** 2)
        return loss

        # T_pred = SE3.exp(epsilon).as_matrix()
        # T2 = T2_1.reshape(-1, 4, 4)
        # self.T = T2
        # print("T_pred: ", T_pred)
        # delta_T = torch.matmul(torch.inverse(T2), T_pred)
        # print("delta_T: ", delta_T.shape)
        # delta_good = SE3.from_matrix(delta_T)
        # delta_epsilon = SE3.log(delta_good)
        # return torch.mean(delta_epsilon * delta_epsilon)


class ApproachAngleLoss(nn.Module):
    def __init__(self):
        super(ApproachAngleLoss, self).__init__()
        handle_middle = torch.from_numpy(np.asarray([0, 0, 6.59999996e-02, 1])).unsqueeze(0).unsqueeze(2).float()
        self.register_buffer('handle_middle', handle_middle)
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    def get_approach_direction(self, T):
        # T has the shape (BN* 4*4)
        self.handle_middle = self.handle_middle.to(T.device)
        self.handle_middle_2 = self.handle_middle.repeat(T.shape[0], 1, 1)
        handle_middle1 = torch.bmm(T, self.handle_middle_2).squeeze(-1)[:, :3]
        start_points = T[:, :3, 3]
        # direction = (handle_middle1 - start_points) / torch.norm(handle_middle1 - start_points, dim=-1, keepdim=True)
        direction = (handle_middle1 - start_points)
        return direction, handle_middle1

    def forward(self, predict_T, selected_points):
        approach_direction, handle_middle_points = self.get_approach_direction(predict_T)
        selected_points = selected_points.reshape(-1, 3)
        connected_approach = selected_points - handle_middle_points
        # connected_approach = connected_two_points / torch.norm(connected_two_points, dim=-1, keepdim=True)
        out = 1 - self.cosine_similarity(approach_direction, connected_approach)
        return torch.mean(out)
        # cos_angle = 1 - torch.sum(approach_direction * connected_approach, dim=-1)
        # cos_angle2 = cos_angle * cos_angle
        # return torch.mean(cos_angle2)


class GraspGaussianFocalLoss(nn.Module):
    def __init__(self, sigma=0.1, alpha=1, gamma=1, reduction="mean"):
        super(GraspGaussianFocalLoss, self).__init__()
        self.sigma = sigma
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.eps = 1e-6
        self.l1_loss = nn.L1Loss(reduction='none')
        self.use_focal_loss = False
        self.L2_Loss = nn.MSELoss()
        self.L1_smooth = nn.SmoothL1Loss(beta=0.1)

    def forward(self, gt_grasp_success, point_pred_score, min_dis):
        # dis = torch.cdist(select_points, gt_grasp_points, p=2)
        # min_dis, index = torch.min(dis, dim=-1)

        gaussian_kernel_GT = torch.exp(-min_dis / self.sigma)
        gaussian_kernel_GT = gaussian_kernel_GT

        if self.use_focal_loss:
            # dis_soft = self.l1_loss(point_pred_score, gaussian_kernel_GT)
            dis_soft = torch.abs(point_pred_score - gaussian_kernel_GT)
            # print(point_pred_score)
            # if torch.sum(torch.gt(dis_soft, 1)) > 1:
            #    print("it has a value bigger than 1")
            focal_loss = -1 * self.alpha * dis_soft ** self.gamma * torch.log((1.0 - dis_soft) + self.eps)
            # focal_loss = -1 * (torch.log(1 - dis_soft + self.eps))
            # focal_loss = dis_soft
            # focal_loss = focal_loss * gt_grasp_success
            if self.reduction == 'none':
                loss = focal_loss
            elif self.reduction == 'mean':
                loss = torch.mean(focal_loss)
            elif self.reduction == 'sum':
                loss = torch.sum(focal_loss)
            else:
                raise NotImplementedError("Invalid reduction mode: {}".format(self.reduction))
            return loss
        else:
            return self.L2_Loss(point_pred_score, gaussian_kernel_GT)


#        return self.mse_loss(point_pred_score, gaussian_kernel_GT)


if __name__ == '__main__':
    pass
