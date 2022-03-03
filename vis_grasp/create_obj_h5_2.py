import h5py
import numpy as np
import argparse
from scipy.spatial import distance
import open3d as o3d
from acronym import GripperMesh, NormalizeObjectPose


# This is script is used to further improve the features of the dataset
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


def compute_nn_distance(grasp_points, points):
    y = distance.cdist(points, grasp_points)
    index = np.argmin(y, axis=0)
    return points[index], index


def get_corresponding_points(T, points):
    gripper_geometry = GripperMesh()
    grasp_points = gripper_geometry.get_graspPoints(T)
    corresponds_points, index = compute_nn_distance(grasp_points, points)
    return corresponds_points, index


def get_grasp_points(grasp_data_):
    grasp_points = []
    grasp_points_index_ = []
    for i in range(len(grasp_data_["Transform"])):
        corresponds_grasp_points, grasp_points_index = get_corresponding_points(T=grasp_data_["Transform"][i],
                                                                                points=grasp_data_["points"][i])
        grasp_points.append(corresponds_grasp_points)
        grasp_points_index_.append(grasp_points_index)
    grasp_points = np.stack(grasp_points, axis=0)
    grasp_points_index_ = np.stack(grasp_points_index_, axis=0)
    return grasp_points, grasp_points_index_


def get_estimate_normal(points_):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=50))
    return np.asarray(pcd.normals)


def get_grasp_and_normal(grasp_data_):
    grasp_points = []
    grasp_normal = []
    for i in range(len(grasp_data_["Transform"])):
        corresponds_grasp_points, grasp_points_index = get_corresponding_points(T=grasp_data_["Transform"][i],
                                                                                points=grasp_data_["points"][i])
        normal = get_estimate_normal(points_=grasp_data_["points"][i])
        grasp_normal_ = normal[grasp_points_index]
        grasp_points.append(corresponds_grasp_points)
        grasp_normal.append(grasp_normal_)
    grasp_points = np.stack(grasp_points, axis=0)
    grasp_normal = np.stack(grasp_normal, axis=0)
    return grasp_points, grasp_normal


def estimate_normal(points_):
    surface_normal = []
    for i in range(points_.shape[0]):
        normal = get_estimate_normal(points_[i])
        surface_normal.append(normal)
    surface_normal = np.stack(surface_normal, axis=0)
    return surface_normal


def normalized_objects(points_):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_)
    pcd.translate(-1 * pcd.get_center())
    print("old max_bound: ", pcd.get_max_bound())
    print("old min_bound: ", pcd.get_min_bound())
    bbox = pcd.get_axis_aligned_bounding_box()
    box_points = np.asarray(bbox.get_box_points())
    cd = distance.cdist(box_points, box_points)
    print(np.max(cd))
    pcd.scale(center=pcd.get_center(), scale=1 / np.max(cd))
    bbox = pcd.get_axis_aligned_bounding_box()
    # hh = pcd.get_oriented_bounding_box().get_box_points()
    # new_points = NormalizeObjectPose(points_, np.asarray(hh))
    # # print(np.asarray(hh))
    # # print(pcd.get_max_bound())
    # # print(pcd.get_min_bound())
    # new_pcd = o3d.geometry.PointCloud()
    # new_pcd.points = o3d.utility.Vector3dVector(new_points)
    # print("center: ", new_pcd.get_center())
    # new_pcd.paint_uniform_color([1, 0, 0])
    # pcd.paint_uniform_color([0, 0, 1])
    # print(np.min(points_, axis=0), np.max(points_, axis=0))
    print("new max_bound: ", pcd.get_max_bound())
    print("new min_bound: ", pcd.get_min_bound())
    # print(np.min(np.asarray(pcd.points), axis=0), np.max(np.asarray(pcd.points), axis=0))
    o3d.visualization.draw_geometries([pcd, bbox])


if __name__ == '__main__':
    args = make_parser()
    grasp_data = h5py.File(args.input_h5[0], "r+")
    #normalized_objects(grasp_data["points"][100])
    
    # del grasp_data["grasp_points"]
    # del grasp_data["grasp_points_normal"]
    # grasp_points_, grasp_normal_ = get_grasp_and_normal(grasp_data_=grasp_data)
    # print(grasp_points_.shape)
    # print(grasp_normal_.shape)
    # grasp_data.create_dataset('grasp_points', data=grasp_points_, dtype=float)
    # grasp_data.create_dataset('grasp_points_normal', data=grasp_normal_, dtype=float)
    # points = grasp_data["points"]
    # print(points.shape)
    # estimate_normal(points_=points)
    grasp_data.close()
    # print(grasp_data["grasp_points"].shape)
    # grasp_data.create_dataset('grasp_points', data=grasp_data["points"], dtype=float)
    # grasp_data.close()
    # grasp_points = []
    # for i in range(len(grasp_data["Transform"])):
    #     print(i)
    #     grasp_points.append(get_corresponding_points(T=grasp_data["Transform"][i], points=grasp_data["points"][i]))
    #
    # grasp_points = np.stack(grasp_points, axis=0)
    # grasp_data.create_dataset('grasp_points', data=grasp_points, dtype=float)
    # grasp_data.close()
    # print(correpontd_points.shape)

    # gripper_geometry = GripperMesh()
    # grasp_points = gripper_geometry.get_graspPoints(T)
    # print(grasp_points.shape)

    # corresponds_points = compute_nn_distance(grasp_points, points)

    # pcd_points = []
    #
    # pcd_new = o3d.geometry.PointCloud()
    # pcd_new.points = o3d.utility.Vector3dVector(points)
    # pcd_new.paint_uniform_color([0, 0, 1])
    # pcd_points.append(pcd_new)
    #
    # my_pcd = o3d.geometry.PointCloud()
    # my_pcd.points = o3d.utility.Vector3dVector(np.asarray([grasp_points[0]]))
    # my_pcd.paint_uniform_color([1, 0, 1])
    # pcd_points.append(my_pcd)
    #
    # my_pcd2 = o3d.geometry.PointCloud()
    # my_pcd2.points = o3d.utility.Vector3dVector(np.asarray([corresponds_points[0]]))
    # my_pcd2.paint_uniform_color([1, 0, 1])
    # pcd_points.append(my_pcd2)
    #
    # gripper = gripper_geometry.create_gripper_marker_open3d(T=grasp_data["Transform"][0][0])
    # pcd_points.append(gripper)
    # o3d.visualization.draw_geometries(pcd_points)
