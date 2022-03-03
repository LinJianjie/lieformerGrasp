import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[3]
sys.path.append(str(root))
from pytorch_utils.components.torch_utils import batchIndexing
import torch

if __name__ == '__main__':
    x = torch.rand(8, 1024, 3)
    y = torch.rand(8, 2048, 3)
    z = torch.rand(8, 2048)
    T = torch.rand(8, 2048, 4, 4)
    dis = torch.cdist(x, y, p=2)
    min_dis, index = torch.min(dis, dim=-1)
    results = batchIndexing(input_xyz_query=x, input_xyz_list=[z, T], batch_index=index, n_sample=1)
    print(results[0].shape)
    print(results[1].shape)
