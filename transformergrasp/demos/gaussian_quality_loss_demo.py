import sys
from pathlib import Path

import torch
from torch.autograd import Variable

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
from tp_models.loss_functions import GraspGaussianFocalLoss

if __name__ == '__main__':
    Gaussian_Loss = GraspGaussianFocalLoss(reduction="mean")
    cuda_index = "cuda"
    device = torch.device(cuda_index if (torch.cuda.is_available()) else "cpu")
    with torch.autograd.set_detect_anomaly(True):
        dtype = torch.FloatTensor
        # predict_score = Variable(torch.rand(2, 10).type(dtype).to(device), requires_grad=True)
        predict_scores = Variable(torch.rand(2, 10), requires_grad=True)
        gt_grasp_success = torch.rand(2, 10)
        min_dis = torch.rand(2, 10)
        loss = Gaussian_Loss(point_pred_score=predict_scores, min_dis=min_dis,
                             gt_grasp_success=gt_grasp_success)
        print(loss)
        loss.backward()
        print("preditc: ", predict_scores.grad)
