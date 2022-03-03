import os
import sys
from pathlib import Path

import open3d as o3d

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))
import shutil


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def readText(path, sep):
    f = open(path, 'r')
    data = []
    for line in f.readlines():
        ll = line.replace('\n', '').split(sep)
        data.append(ll)
    return data


def save_ply(pc, path):
    """
        Save numpy tensor as .ply file
        :param pc -> numpy ndarry with shape [num_points, 3] expected
        :param path -> saving path including .ply name
        """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.io.write_point_cloud(path, pcd)


def save_batch_ply(pc, path):
    for i in range(pc.shape[0]):
        save_ply(pc[0], path)


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_dirs_checkout():
    checkout_path = os.path.join(os.getcwd(), "checkpoints")
    make_dirs(path=checkout_path)
    return checkout_path


def make_dirs_log():
    log_dirs = os.path.join(os.getcwd(), "logs")
    make_dirs(path=log_dirs)
    return log_dirs


def copy_file(source, des, filename, rename=None):
    shutil.copy(source, des)
    old_name = os.path.join(des, filename)
    new_name = os.path.join(des, rename)
    os.rename(old_name, new_name)
