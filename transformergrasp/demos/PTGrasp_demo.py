import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
import torch
from tp_models.PTnet_grasp import PTNetGrasp, PTNetGraspModel
from pytorch_utils.components.yaml_utils import YAMLConfig

if __name__ == '__main__':
    x = torch.rand(2, 1024, 3)
    yaml_file = "../tp_models/tnetgrasp.yaml"
    yaml_config = YAMLConfig(yaml_file)
    tnetGrasp = PTNetGrasp(config=yaml_config.config)
    cuda_index = "cuda"
    device = torch.device(cuda_index if (torch.cuda.is_available()) else "cpu")
    print(device)
    tnetGrasp.to(device)
    x = x.to(device)
    down_sample_points, quality, epsilon = tnetGrasp(x)
    print("down_sample_points:", down_sample_points.shape)
    print("quality: ", quality.shape)
    print("epsilon: ", epsilon.shape)
    tNetGraspModel = PTNetGraspModel(config=yaml_config.config, checkpoint_name="checkpoints.pth",
                                     best_name="best_name.pth",
                                     checkpoint_path=".", logger_file_name="log")
