import os

import h5py
import numpy as np
import trimesh


def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
                                         for m in scene_or_mesh.geometry.values()])
    else:
        mesh = scene_or_mesh
    return mesh


def get_sample(mesh_root, num_of_points=2000):
    obj_mesh = trimesh.load_mesh(file_obj=mesh_root)
    mesh = as_mesh(scene_or_mesh=obj_mesh)
    samples = mesh.sample(num_of_points)
    return samples


if __name__ == '__main__':
    obj_dir = "path/to/ycb_dataset"
    points = []
    filename_list = []
    # obj_data = h5py.File("ycb_dataset.h5", "r+")
    # points=obj_data["points"][:].astype('float32')
    # print(points.shape)
    # obj_data.close()
    file1 = open("ycb_dataset_class.csv", "w")
    for root, dirs, files in os.walk(obj_dir):
        if files:
            obj_filename = None
            collision_filename = None
            for file in files:
                if file.endswith(".obj"):
                    if "tsdf" in root:
                        # o3d.read
                        filename = root + "/" + file
                        filename_list.append(filename)
                        file1.write(filename)
                        sample = get_sample(mesh_root=filename)
                        points.append(sample)
    file1.close()
    points = np.stack(points, axis=0)
    filename_list = np.stack(filename_list, axis=0)
    # print("filename_list: ",filename_list.shape)
    np.save("ycb_dataset_class", filename_list)
    # print(filename_list)
    # print(points.shape)
    # obj_data.create_dataset("points", data=points, dtype=float)
    # obj_data.create_dataset("filename", data=filename_list, dtype=str)
    # obj_data.close()
