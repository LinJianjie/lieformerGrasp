import sys
from pathlib import Path

import torch

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
from pytorch_utils.components.nvidia_grasp_dataset import NvidiaGraspDataSet
from pytorch_utils.components.mainTrainer import MainTrainer
from tp_models.Tnet_grasp import TNetGraspModel
from pytorch_utils.components.yaml_utils import YAMLConfig

if __name__ == '__main__':
    print("===>")
    yaml_file = "../tp_models/tnetgrasp.yaml"
    yaml_config = YAMLConfig(yaml_file)
    trainer = MainTrainer(yaml_config=yaml_config.config)
    trainer.from_argparse_args(args=None)
    train_dataset = NvidiaGraspDataSet(train=True)
    print("===>")
    test_dataset = NvidiaGraspDataSet(train=False)

    tNetGraspModel = TNetGraspModel(config=yaml_config.config, checkpoint_name="checkpoints.pth",
                                    best_name="best_name.pth",
                                    checkpoint_path=".", logger_file_name="log")
    trainer.load_data(train_dataset=train_dataset, test_dataset=test_dataset)
    trainer.set_model(tNetGraspModel, use_cuda=False)
    # gt_grasp_points, gt_point_transformation, select_points, pred_quality, pred_epsilon
    gt_grasp_points = torch.rand(2, 2000, 3)
    select_points = torch.rand(2, 1000, 3)
    pred_quality = torch.rand(2, 1000)
    pred_epsilon = torch.rand(2, 1000, 6)
    T = torch.eye(4)
    T = T.unsqueeze(0).unsqueeze(1).repeat(2, 2000, 1, 1)
    tNetGraspModel.backward_model(gt_grasp_points=gt_grasp_points, gt_point_transformation=T,
                                  select_points=select_points, pred_quality=pred_quality,
                                  pred_epsilon=pred_epsilon)
