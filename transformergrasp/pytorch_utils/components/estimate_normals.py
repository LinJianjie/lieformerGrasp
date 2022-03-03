import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))
import open3d as o3d
import torch
import numpy as np


class BatchEstimateNormals:
    def __init__(self, points_, maxKnn, includeSelf=True):  # [batch,N,3]
        points = points_.numpy()
        self.normal = torch.zeros_like(points_)
        for i in range(points.shape[0]):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[i, :, :])
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=maxKnn))
            self.normal[i, :, :] = torch.from_numpy(np.asarray(pcd.normals))
