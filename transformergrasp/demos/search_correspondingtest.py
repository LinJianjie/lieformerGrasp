import sys
from pathlib import Path

import torch

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
from tp_models.Tnet_grasp import TNetGrasp
from pytorch_utils.components.torch_utils import batchIndexing

if __name__ == '__main__':
    x = torch.rand(2, 3, 3)
    y = torch.rand(2, 6, 3)
    T1 = torch.rand(2, 6, 4, 4)
    T2 = torch.rand(2, 6)
    dis, index = TNetGrasp.search_correspond_grasp(selects_points=x, grasp_points=y)
    print(index)
    selected_gt = batchIndexing(input_xyz_query=x,
                                input_xyz_list=[T1, T2],
                                batch_index=index, n_sample=1)
    print(T1[1])
    print(selected_gt[0][1])
