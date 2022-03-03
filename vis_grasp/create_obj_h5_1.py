import argparse
import os

import h5py
import numpy as np
import trimesh as tm

from acronym import LoadData


def make_parser():
    parser = argparse.ArgumentParser(
        description="Visualize grasps from the dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_h5", nargs="+", help="HDF5 or JSON Grasp file(s).")
    parser.add_argument(
        "--num_grasps", type=int, default=60, help="Number of grasps to show."
    )
    parser.add_argument(
        "--mesh_root", default=".", help="Directory used for loading meshes."
    )
    return parser.parse_args()


def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, tm.Scene):
        mesh = tm.util.concatenate([tm.Trimesh(vertices=m.vertices, faces=m.faces)
                                    for m in scene_or_mesh.geometry.values()])
    else:
        mesh = scene_or_mesh
    return mesh


def get_sample(grasp_data, mesh_root):
    obj_mesh = grasp_data.load_mesh(mesh_root_dir=mesh_root)
    # mesh = as_mesh(scene_or_mesh=obj_mesh)
    samples = obj_mesh.sample(5000)
    return samples


class GraspData:
    def __init__(self):
        self._T = []
        self._quality = []
        self._points = []
        self._scales = []
        self._label = []

    def reinit(self):
        self._T = []
        self._scales = []
        self._points = []
        self._quality = []
        self._label = []

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label.append(value)

    @property
    def quality(self):
        return self._quality

    @quality.setter
    def quality(self, value):
        self._quality.append(value)

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value):
        self._T.append(value)

    @property
    def scale(self):
        return self._scales

    @scale.setter
    def scale(self, value):
        self._scales.append(value)

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, value):
        self._points.append(value)


def main(args):
    class_directory = dict()
    for (dirpath, dirnames, grasp_filenames) in os.walk("../grasps"):

        i = 0
        for k, h5_file in enumerate(grasp_filenames):
            h5_file_str = str(h5_file)
            class_name = h5_file_str.split("_")[0]
            if str(class_name) not in class_directory.keys():
                class_directory[str(class_name)] = []

            class_directory[str(class_name)].append(h5_file)
    train_index = 0
    train_part_index = 0
    train_data_h5 = "train_data_part_" + str(train_part_index) + ".h5"
    test_part_index = 0
    test_data_h5 = "test_data_part_" + str(test_part_index) + ".h5"
    test_index = 0
    graspTrainData = GraspData()
    graspTestData = GraspData()
    key_to_index = dict
    file1 = open("class2index.csv", "w")
    str_name = "label name\n"
    file1.write(str_name)
    for i, (key, value) in enumerate(class_directory.items()):
        # print("at: i", i, "key: ", key)
        print("at: i", i, "key: ", key)
        l = list(range(len(value)))
        if len(value) > 1:
            str_name = "%d, %s\n" % (i, key)
            file1.write(str_name)
            train_size = int((len(value) * 0.8))
            np.random.shuffle(l)
            train_file = [value[index] for index in l[:train_size]]
            test_file = [value[index] for index in l[train_size:]]
            for k, train_h5_file in enumerate(train_file):
                grasp_filenames = os.path.join("../grasps", train_h5_file)
                grasp_train_data = LoadData(filename=grasp_filenames)
                graspTrainData.T = grasp_train_data.grasp_data.grasp_transformer
                graspTrainData.quality = grasp_train_data.grasp_data.quality_success
                assert len(grasp_train_data.grasp_data.quality_success) == 2000
                graspTrainData.scale = grasp_train_data.grasp_data.object_scale
                graspTrainData.points = get_sample(grasp_data=grasp_train_data, mesh_root=args.mesh_root)
                graspTrainData.label = i

            for k, test_h5_file in enumerate(test_file):
                grasp_filenames = os.path.join("../grasps", test_h5_file)
                grasp_test_data = LoadData(filename=grasp_filenames)
                graspTestData.T = grasp_test_data.grasp_data.grasp_transformer
                graspTestData.quality = grasp_test_data.grasp_data.quality_success
                assert len(grasp_test_data.grasp_data.quality_success) == 2000
                graspTestData.scale = grasp_test_data.grasp_data.object_scale
                graspTestData.points = get_sample(grasp_data=grasp_test_data, mesh_root=args.mesh_root)
                graspTestData.label = i
                # if test_index == 2000:
                #     T = np.stack(graspTestData.T, axis=0)
                #     quality = np.stack(graspTestData.quality, axis=0)
                #     scale = np.stack(graspTestData.scale, axis=0)
                #     points = np.stack(graspTestData.points, axis=0)
                #     hf_file = h5py.File(train_data_h5, 'w')
                #     hf_file.create_dataset("quality", data=quality, dtype=int)
                #     hf_file.create_dataset("scale", data=scale, dtype=float)
                #     hf_file.create_dataset('points', data=points, dtype=float)
                #     hf_file.create_dataset("Transform", data=T, dtype=float)
                #     hf_file.close()
                #     print(T.shape[0])
                #     graspTestData.reinit()
                #     test_part_index = test_part_index + 1
                #     train_index = 0
                #     train_data_h5 = "train_data_part_" + str(test_part_index) + ".h5"
                # test_index = test_index + 1
    file1.close()
    T = np.stack(graspTrainData.T, axis=0)
    quality = np.stack(graspTrainData.quality, axis=0)
    scale = np.stack(graspTrainData.scale, axis=0)
    points = np.stack(graspTrainData.points, axis=0)
    labels = np.stack(graspTrainData.label, axis=0)
    hf_file = h5py.File(train_data_h5, 'w')
    hf_file.create_dataset("quality", data=quality, dtype=int)
    hf_file.create_dataset("labels", data=labels, dtype=int)
    hf_file.create_dataset("scale", data=scale, dtype=float)
    hf_file.create_dataset('points', data=points, dtype=float)
    hf_file.create_dataset("Transform", data=T, dtype=float)
    hf_file.close()
    print("Train: T: ", T.shape)

    T = np.stack(graspTestData.T, axis=0)
    quality = np.stack(graspTestData.quality, axis=0)
    scale = np.stack(graspTestData.scale, axis=0)
    points = np.stack(graspTestData.points, axis=0)
    labels = np.stack(graspTestData.label, axis=0)
    hf_file = h5py.File(test_data_h5, 'w')
    hf_file.create_dataset("quality", data=quality, dtype=int)
    hf_file.create_dataset("labels", data=labels, dtype=int)
    hf_file.create_dataset("scale", data=scale, dtype=float)
    hf_file.create_dataset('points', data=points, dtype=float)
    hf_file.create_dataset("Transform", data=T, dtype=float)
    hf_file.close()
    print("Test: T: ", T.shape)

    # vertices = mesh.vertices
    # pcd_new = o3d.geometry.PointCloud()
    # pcd_new.points = o3d.utility.Vector3dVector(samples)
    # o3d.visualization.draw_geometries([pcd_new])


if __name__ == '__main__':
    args = make_parser()
    main(args=args)
