import sys
from pathlib import Path

import torch

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[3]
sys.path.append(str(root))
import pytorch_utils.components.lie_groups.se3 as se3
from pytorch_utils.components.lie_groups.myse3 import SE3Matrix as MySE3

if __name__ == '__main__':
    # R = [[0.4396997, 0.8511918, -0.2865950],
    #      [-0.6534387, 0.5220968, 0.5481174],
    #      [0.6161833, -0.0537348, 0.7857676]]
    # R = torch.from_numpy(np.asarray(R))
    # t = torch.from_numpy(np.asarray([1.0, 2.0, 3.0]))
    # # SE3_tRANSFORATION = se3.SE3Matrix(rot=R.unsqueeze(0), trans=t.unsqueeze(0))
    # T = torch.eye(4)
    # T[:3, :3] = R
    # T[:3, 3] = t
    # T_good = se3.SE3Matrix.from_matrix(T)
    # # epislon = SE3_tRANSFORATION.log()
    # # print(epislon)
    # print(T_good)
    # h = se3.SE3Matrix.log(T_good)
    # print(h)
    # M = se3.SE3Matrix.exp(h).as_matrix()
    # print(M)
    #
    # T2 = T.unsqueeze(0)
    # T3 = T2.repeat(6, 1, 1)
    # T33 = T3.reshape(2, 3, 4, 4)
    # T33_1 = T33.reshape(-1, 4, 4)
    # print("T33_1:", T33_1.shape)
    #
    # T3_good = se3.SE3Matrix.from_matrix(T3)
    # h3 = se3.SE3Matrix.log(T3_good)
    # print("h3:", h3.shape)
    # M3 = se3.SE3Matrix.exp(h3).as_matrix()
    # print("M3: ", M3.shape)
    # T5 = torch.inverse(M3)
    # print("T5:", T5.shape)
    # TT4 = torch.bmm(T5, T33_1)
    # print(TT4.shape)
    # TT4_good = se3.SE3Matrix.from_matrix(TT4)
    # h4 = se3.SE3Matrix.log(TT4_good)
    # print(torch.mean(h4))
    # # print("h4:", h4.shape)
    # gt_rotation_epsilon = SO3.log(SO3.from_matrix(T[:3, :3]))
    # print("gt_rotation_epsilon:", gt_rotation_epsilon)
    # gt_epsilon = SE3.log(SE3.from_matrix(T))
    # print("gt_epsilon: ", gt_epsilon)
    x = torch.rand(6)
    M = se3.SE3Matrix.exp(x).as_matrix()
    print(M)
    rotaion, tranlstion = MySE3.exp(x)
    print("rotaion: ", rotaion)
    print(rotaion.permute(1,0))
    print("R*R^T: ", rotaion*rotaion.permute(1,0))
    print("tranlstion: ", tranlstion)
