import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[3]
sys.path.append(str(root))
from pytorch_utils.components.torch_cluster_sampling import farthest_point_sampling
import torch

if __name__ == '__main__':
    x = torch.rand(8, 100, 3)
    fps_center_, x, fps_idx_ = farthest_point_sampling(x, ratio=0.5)
    print(fps_center_.shape)
