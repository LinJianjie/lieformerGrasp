import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
from pytorch_utils.components.transformerPointNet.transformerPNPPModule import TransformerPointNetPPHead
from pytorch_utils.components.pytorchLightModule import TorchLightModule
from pytorch_utils.components.netUtils import NetUtil
from pytorch_utils.components.metrics import Metrics
from pytorch_utils.components.torch_utils import batchIndexing
from tp_models.loss_functions import TransformationLoss, ApproachAngleLoss, GraspGaussianFocalLoss
import torch.nn as nn
import torch
import h5py


class PTNetGrasp(nn.Module):
    def __init__(self, config):
        super(PTNetGrasp, self).__init__()
        self.config = config
        self.use_2D = self.config["TransformerConfig"]["use_2D"]
        self.PTNetEncoder = TransformerPointNetPPHead(config=config)

        if self.use_2D:
            self.pn_layers = NetUtil.SeqPointNetConv1d(
                channels=self.config["PTConfig"]["head"],
                active="relu")
        else:
            self.pn_layers = NetUtil.SeqLinear(
                channels=[self.encoder.last_layer_size, self.encoder.last_layer_size],
                activation="LeakyReLU")

        if self.use_2D:
            self.qualityNet = nn.Sequential(
                NetUtil.SeqPointNetConv1d(channels=self.config["TransformerConfig"]["quality_channels2D"]),
                nn.Conv1d(in_channels=self.config["TransformerConfig"]["quality_channels2D"][-1],
                          out_channels=1,
                          kernel_size=1),
                nn.Sigmoid())

            self.lieAlgebraNet = nn.Sequential(
                NetUtil.SeqPointNetConv1d(channels=self.config["TransformerConfig"]["lie_channels2D"]),
                nn.Conv1d(in_channels=self.config["TransformerConfig"]["lie_channels2D"][-1],
                          out_channels=6,
                          kernel_size=1))
        else:
            self.qualityNet = nn.Sequential(
                NetUtil.SeqLinear(channels=self.config["TransformerConfig"]["quality_channels1D"]),
                nn.Linear(in_features=self.config["TransformerConfig"]["quality_channels1D"][-1], out_features=256),
                nn.Sigmoid())
            self.lieAlgebraNet = nn.Sequential(
                NetUtil.SeqLinear(channels=self.config["TransformerConfig"]["lie_channels1D"]),
                nn.Linear(in_features=self.config["TransformerConfig"]["lie_channels1D"][-1], out_features=256 * 6))
        # self.quality_activation = nn.Sigmoid()
        # self.graspNet = nn.Sequential(
        #     NetUtil.SeqLinear(channels=self.config["TransformerConfig"]["lie_channels"]),
        #     nn.Linear(self.config["TransformerConfig"]["lie_channels"][-1], 7)
        # )

    def forward(self, x):
        point_feature_ = self.PTNetEncoder(x).permute(0, 2, 1)
        point_feature = self.pn_layers(point_feature_)
        down_sample_points = self.PTNetEncoder.down_sample_xyz
        # down_sample_points = x
        quality = self.qualityNet(point_feature)
        epsilon = self.lieAlgebraNet(point_feature)
        if self.use_2D:
            quality = quality.permute(0, 2, 1)
            epsilon = epsilon.permute(0, 2, 1)
            quality = quality.squeeze(dim=-1)
            return down_sample_points, quality, epsilon
        else:
            quality = quality.squeeze(dim=-1)
            epsilon = epsilon.reshape(epsilon.shape[0], -1, 6)
            return down_sample_points, quality, epsilon
        # point_feature = point_feature.permute(0, 2, 1)
        # grasp_feature = self.graspNet(point_feature)
        # epsilon = grasp_feature[:, :, 1:]
        # quality = self.quality_activation(grasp_feature[:, :, 0])
        # # quality = self.qualityNet(point_feature)
        # # epsilon = self.lieAlgebraNet(point_feature)
        # # quality = quality.permute(0, 2, 1)
        # # epsilon = epsilon.permute(0, 2, 1)
        # quality = quality.squeeze(dim=-1)
        # return down_sample_points, quality, epsilon

    @staticmethod
    def search_correspond_grasp(selects_points, grasp_points):
        dis = torch.cdist(selects_points, grasp_points, p=2)
        min_dis, index = torch.min(dis, dim=-1)
        return min_dis, index


class PTNetGraspModel(TorchLightModule):
    def __init__(self, config, checkpoint_name, best_name, checkpoint_path, logger_file_name):
        super(PTNetGraspModel, self).__init__(config, checkpoint_name, best_name, checkpoint_path, logger_file_name)
        self.model = PTNetGrasp(config=config)
        self.loss_transformation, self.loss_grasp_gaussian, self.loss_approach_angle = \
            self.configure_loss_function(sigma=self.config["LossConfig"]["sigma"],
                                         alpha=self.config["LossConfig"]["alpha"],
                                         gamma=self.config["LossConfig"]["gamma"])
        self.configure_optimizers()

    def forward(self, x):
        down_sample_points, quality, epsilon = self.model(x)
        return down_sample_points, quality, epsilon

    def backward_model(self, gt_grasp_points, gt_point_transformation, gt_grasp_success, select_points, pred_quality,
                       pred_epsilon):
        # Here has a bug, since the selects points will be chosen by knn
        # corresponding search
        min_dis, index = PTNetGrasp.search_correspond_grasp(selects_points=select_points,
                                                            grasp_points=gt_grasp_points)

        selected_gt_transform = batchIndexing(input_xyz_query=select_points,
                                              input_xyz_list=[gt_point_transformation],
                                              batch_index=index, n_sample=1)

        selected_gt_success = batchIndexing(input_xyz_query=select_points,
                                            input_xyz_list=[gt_grasp_success],
                                            batch_index=index, n_sample=1)

        loss_transformation = self.loss_transformation(points_1=select_points,
                                                       epsilon_1=pred_epsilon,
                                                       T2_1=selected_gt_transform[0])
        loss = self.config["LossConfig"]["lambda_transformation"] * loss_transformation
        if self.config["LossConfig"]["lambda_grasp_gaussian"] > 0:
            loss_grasp_gaussian = self.loss_grasp_gaussian(gt_grasp_success=selected_gt_success[0],
                                                           point_pred_score=pred_quality,
                                                           min_dis=min_dis)
            loss += self.config["LossConfig"]["lambda_grasp_gaussian"] * loss_grasp_gaussian
        if self.config["LossConfig"]["lambda_approach_angle"] > 0:
            loss_approach_angle = self.loss_approach_angle(predict_T=self.loss_transformation.T,
                                                           selected_points=select_points)
            loss += self.config["LossConfig"]["lambda_approach_angle"] * loss_approach_angle
        # loss = self.config["LossConfig"]["lambda_transformation"] * loss_transformation + \
        #        self.config["LossConfig"]["lambda_approach_angle"] * loss_approach_angle + \
        #        self.config["LossConfig"]["lambda_grasp_gaussian"] * loss_grasp_gaussian

        # loss = self.config["LossConfig"]["lambda_transformation"] * loss_transformation + \
        #        self.config["LossConfig"]["lambda_grasp_gaussian"] * loss_grasp_gaussian
        return loss, loss_transformation, loss_grasp_gaussian

    def backward_model_update(self, loss):
        self.model_optimizer["opt"].zero_grad()
        loss.backward()
        self.model_optimizer["opt"].step()
        self.model_optimizer["scheduler"].step()

    @staticmethod
    def configure_loss_function(sigma=1, alpha=1, gamma=1):
        transformation_loss = TransformationLoss()
        grasp_gaussian_loss = GraspGaussianFocalLoss(sigma=sigma, alpha=alpha, gamma=gamma, reduction="mean")
        approach_loss = ApproachAngleLoss()
        return transformation_loss, grasp_gaussian_loss, approach_loss

    # def toCuda(self, device):
    #     self.device = device
    #     if torch.cuda.device_count() > 1:
    #         print("====> use data parallel")
    #         self.model = nn.DataParallel(self.model)
    #         self.model.to(device)
    #     else:
    #         print("====> use only one cuda")
    #         self.model.to(device)

    def compute_step(self, dataset, compute_loss: bool = True, normalized=False):
        input_point_cloud, gt_grasp_points, gt_transform, gt_grasp_success = dataset
        input_point_cloud = input_point_cloud.to(self.device)
        gt_grasp_points = gt_grasp_points.to(self.device)
        gt_transform = gt_transform.to(self.device)
        gt_grasp_success = gt_grasp_success.to(self.device)
        down_sampled_points, pred_quality, pred_epsilon = self(input_point_cloud)

        # if normalized:
        #     normalized_inputs, center_points = DataUtils.zeroCenter(input_point_cloud)
        #     gt_grasp_points = gt_grasp_points - center_points
        #     gt_transform[:, :, :3, 3] = (gt_transform[:, :, :3, 3] - center_points)
        # if normalized:
        #     down_sampled_points, pred_quality, pred_epsilon = self(normalized_inputs)
        #
        # else:
        #     down_sampled_points, pred_quality, pred_epsilon = self(input_point_cloud)
        if compute_loss:
            loss, loss_transform, loss_gaussian = self.backward_model(gt_point_transformation=gt_transform,
                                                                      gt_grasp_points=gt_grasp_points,
                                                                      gt_grasp_success=gt_grasp_success,
                                                                      pred_epsilon=pred_epsilon,
                                                                      select_points=down_sampled_points,
                                                                      pred_quality=pred_quality)
            return loss, loss_transform, loss_gaussian
        else:
            return down_sampled_points, pred_quality, pred_epsilon

    def train_one_epoch(self, train_loader, at_epoch, n_epochs, batch_size):
        self.train()
        metric_train_loss = Metrics()
        metric_train_trans_loss = Metrics()
        metric_train_gaussian_loss = Metrics()
        for i, dataset in enumerate(train_loader):
            loss, loss_transform, loss_gaussian = self.compute_step(dataset, normalized=True)
            self.backward_model_update(loss)
            metric_train_loss.loss = loss.item()
            metric_train_trans_loss.loss = loss_transform.item()
            metric_train_gaussian_loss.loss = loss_gaussian.item()
        print('Train loss transform %.4f, train gaussian loss %.4f.' % (
            metric_train_trans_loss.loss, metric_train_gaussian_loss.loss))
        return metric_train_loss.loss

    def evaluation_step(self, test_loader, check_point_name=None):
        self.eval()
        metric_evaluation_loss = Metrics()
        for i, dataset in enumerate(test_loader):
            with torch.no_grad():
                loss = self.compute_step(dataset)
                # self.Logger.evaluation_loss(loss.item())
                metric_evaluation_loss.loss = loss.item()
        return metric_evaluation_loss.loss

    def predict(self, test_loader, evaluation_prefix: str = None):
        self.eval()
        if evaluation_prefix is None:
            h5_filename = "checkpoints/evaluation.h5"
            grasp_data = h5py.File(h5_filename, "w")
        else:
            h5_filename = evaluation_prefix + "evaluation.h5"
            grasp_data = h5py.File(h5_filename, "w")
            print("save to:  ", h5_filename)
        quality = []
        epsilon = []
        point_cloud = []
        for i, dataset in enumerate(test_loader):
            with torch.no_grad():
                input_point_cloud, gt_grasp_points, gt_transform, gt_grasp_success = dataset
                input_point_cloud = input_point_cloud.to(self.device)
                down_sampled_points, pred_quality, pred_epsilon = self(input_point_cloud)
                point_cloud.append(input_point_cloud)
                quality.append(pred_quality)
                epsilon.append(pred_epsilon)
        quality = torch.stack(quality, dim=0).cpu().detach().numpy()
        epsilon = torch.stack(epsilon, dim=0).cpu().detach().numpy()
        point_cloud = torch.stack(point_cloud, dim=0).cpu().detach().numpy()
        grasp_data.create_dataset("pred_quality", data=quality, dtype=float)
        grasp_data.create_dataset('pred_epsilon', data=epsilon, dtype=float)
        grasp_data.create_dataset("point_cloud", data=point_cloud, dtype=float)

        grasp_data.close()
        print("===> Finished to evaluation", h5_filename)
