import sys
from pathlib import Path

import torch

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))
import h5py
from torch.utils.data import Dataset
from pytorch_utils.components.constant import *
from pytorch_utils.components.dataUtils import DataUtils
from pytorch_utils.components.torch_cluster_sampling import farthest_point_sampling


# evaluate the success to score
class NvidiaGraspDataSet(Dataset):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    foldName = "NvidiaGrasp"
    DATA_DIR = os.path.join(BASE_DIR, dataFile, foldName)

    def __init__(self, train=True, normalized=True, load=True, down_sampled=True, ycb=False):
        super(NvidiaGraspDataSet, self).__init__()
        self.train = train
        self.normalized = normalized
        self.down_sampled = down_sampled
        self.ycb = ycb
        if load:
            if train:
                self.ycb = False
            if self.ycb:
                print("==> use YCB dataset")
            self.points, self.grasp_points, self.transform, self.grasp_success_quality = self.load(train=self.train,
                                                                                                   ycb=self.ycb)

    def __getitem__(self, item):
        shift_points, center = DataUtils.zeroCenter(self.points[item])
        if self.ycb:
            shift_grasp_points = self.grasp_points[item]
            shift_transform = self.transform[item]
        else:
            shift_grasp_points = self.grasp_points[item] - center
            shift_transform = self.transform[item]
            shift_transform[:, :3, 3] = shift_transform[:, :3, 3] - center
        # return self.points[item], self.grasp_points[item], self.transform[item], self.grasp_success_quality[item]
        return shift_points, shift_grasp_points, shift_transform, self.grasp_success_quality[item]

    def __len__(self):
        return self.points.shape[0]

    def load(self, train, ycb=False):
        if train:
            if self.down_sampled:
                filename = "train_data_part_0_down_sampled.h5"
            else:
                filename = "train_data_part_0.h5"
        else:
            if ycb:
                filename = "ycb_dataset.h5"
            else:
                if self.down_sampled:
                    filename = "test_data_part_0_down_sampled.h5"
                else:
                    filename = "test_data_part_0.h5"
        grasp_file = os.path.join(self.DATA_DIR, filename)
        grasp_data = h5py.File(grasp_file, "r")
        if ycb:
            points = torch.from_numpy(grasp_data["points"][:].astype('float32'))
            return points, points, points, points
        else:
            T = torch.from_numpy(grasp_data["Transform"][:].astype('float32'))
            points = torch.from_numpy(grasp_data["points"][:].astype('float32'))
            grasp_points = torch.from_numpy(grasp_data["grasp_points"][:].astype('float32'))
            grasp_success_quality = torch.from_numpy(grasp_data["quality"][:].astype('int64'))
            return points, grasp_points, T, grasp_success_quality

    def down_sampling(self, train):
        if train:
            new_file_name = "train_data_part_0_down_sampled.h5"
        else:
            new_file_name = "test_data_part_0_down_sampled.h5"
        if train:
            filename = "train_data_part_0.h5"
        else:
            filename = "test_data_part_0.h5"
        grasp_file = os.path.join(self.DATA_DIR, filename)
        grasp_data = h5py.File(grasp_file, "r+")
        points = torch.from_numpy(grasp_data["points"][:].astype('float32'))
        points = points.split(32, dim=0)
        down_sampled = []
        for point in points:
            down_sampled_points, _, _ = farthest_point_sampling(point, ratio=0.4)
            print("down_sample_points: ", down_sampled_points.shape)
            down_sampled.append(down_sampled_points)
        new_down_sampled = torch.cat(down_sampled, dim=0)
        print(new_down_sampled.shape)
        new_down_sampled = new_down_sampled.numpy()
        hf_file = h5py.File(new_file_name, 'w')
        hf_file.create_dataset("quality", data=grasp_data["quality"], dtype=int)
        hf_file.create_dataset("labels", data=grasp_data["labels"], dtype=int)
        hf_file.create_dataset("scale", data=grasp_data["scale"], dtype=float)
        hf_file.create_dataset('points', data=new_down_sampled, dtype=float)
        hf_file.create_dataset("Transform", data=grasp_data["Transform"], dtype=float)
        hf_file.create_dataset('grasp_points', data=grasp_data["grasp_points"], dtype=float)
        hf_file.close()
        grasp_data.close()
