import sys
from pathlib import Path

import numpy as np
import torch
from torch.autograd import Variable

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
import pytorch_utils.components.lie_groups.se3 as se3
from tp_models.loss_functions import ApproachAngleLoss

if __name__ == '__main__':
    R = [[0.4396997, 0.8511918, -0.2865950],
         [-0.6534387, 0.5220968, 0.5481174],
         [0.6161833, -0.0537348, 0.7857676]]
    R = torch.from_numpy(np.asarray(R))
    t = torch.from_numpy(np.asarray([1.0, 2.0, 3.0]))
    T = torch.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    T_good = se3.SE3Matrix.from_matrix(T)
    h = se3.SE3Matrix.log(T_good)
    h = h.unsqueeze(0).repeat(2, 1)
    T = T.unsqueeze(0).repeat(2, 1, 1)
    approach_loss = ApproachAngleLoss()
    cuda_index = "cuda"
    device = torch.device(cuda_index if (torch.cuda.is_available()) else "cpu")
    # h = Variable(torch.rand(2, 6).type(dtype).to(device), requires_grad=True)
    with torch.autograd.set_detect_anomaly(True):
        dtype = torch.FloatTensor
        h = Variable(torch.rand(2, 10, 6).type(dtype).to(device), requires_grad=True)
        h1 = h.reshape(-1, 6)
        T_pred = se3.SE3Matrix.exp(h1).as_matrix()
        T_pred_2 = T_pred.reshape(-1, 4, 4)
        select_points = torch.rand(2, 10, 3)
        select_points = select_points.to(device)
        loss = approach_loss(predict_T=T_pred_2, selected_points=select_points)
        print(loss)
        loss.backward()
        print(h.grad.shape)
