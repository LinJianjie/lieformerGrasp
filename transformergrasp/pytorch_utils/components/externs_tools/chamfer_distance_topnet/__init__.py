import os
import sys

import torch
from torch import nn
from torch.autograd import Function

from pytorch_utils.components.externs_tools.externtools_path import get_project_root

project_root = get_project_root()
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
# chamferFunction_build = "lib/python3.6/site-packages/chamfer-0.0.0-py3.6-linux-x86_64.egg/"

# for root, dirs, files in os.walk(os.path.join(os.path.dirname(__file__)),
#                                  "../../../extensions_build/chamfer_distance_topnet"):
_build = None
for root, dirs, files in os.walk(os.path.join(project_root, "extensions_build/chamfer_distance_topnet")):
    for file in files:
        if ".so" in file:
            _build = root
            break
# _path = os.path.join(project_root, "extensions_build/chamfer_distance_topnet", _build)
_path = _build
sys.path.append(_path)

import chamfer


# Chamfer's distance module @thibaultgroueix
# GPU tensors only
class chamferFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()

        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)

        dist1 = dist1.cuda()
        dist2 = dist2.cuda()
        idx1 = idx1.cuda()
        idx2 = idx2.cuda()

        chamfer.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2

    @staticmethod
    def backward(ctx, graddist1, graddist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        gradxyz1 = gradxyz1.cuda()
        gradxyz2 = gradxyz2.cuda()
        chamfer.backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        return gradxyz1, gradxyz2


class ChamferDistance(nn.Module):
    def __init__(self):
        super(ChamferDistance, self).__init__()

    def forward(self, input1, input2):
        dist1, dist2 = chamferFunction.apply(input1, input2)
        return torch.mean(dist1) + torch.mean(dist2), dist1, dist2
