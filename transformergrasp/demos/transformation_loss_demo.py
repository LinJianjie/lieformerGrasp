import sys
from pathlib import Path
from torch.autograd import Variable
import numpy as np
import torch

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
from tp_models.loss_functions import TransformationLoss
import pytorch_utils.components.lie_groups.se3 as se3

if __name__ == '__main__':
    R = [[0.4396997, 0.8511918, -0.2865950],
         [-0.6534387, 0.5220968, 0.5481174],
         [0.6161833, -0.0537348, 0.7857676]]
    R = torch.from_numpy(np.asarray(R))
    t = torch.from_numpy(np.asarray([1.0, 2.0, 3.0]))
    T = torch.eye(4)
    transformation_loss = TransformationLoss()
    T[:3, :3] = R
    T[:3, 3] = t
    T_good = se3.SE3Matrix.from_matrix(T)
    h = se3.SE3Matrix.log(T_good)
    h = h.unsqueeze(0).unsqueeze(1).repeat(2, 2, 1)
    T = T.unsqueeze(0).unsqueeze(1).repeat(2, 2, 1, 1)
    print("h: ", h.shape)
    print("T: ", T.shape)
    cuda_index = "cuda"
    device = torch.device(cuda_index if (torch.cuda.is_available()) else "cpu")

    T = T.to(device)
    transformation_loss.to(device)
    dtype = torch.FloatTensor
    with torch.autograd.set_detect_anomaly(True):
        h = Variable(torch.rand(2, 2, 6).type(dtype).to(device), requires_grad=True)
        # h = h.cuda()
        print("input:", h)
        transformation_loss.zero_grad()
        loss = transformation_loss(h, T)
        print("myloss: ", loss)
        # tt.retain_grad()
        loss.backward()

        print("h:", h.grad)
