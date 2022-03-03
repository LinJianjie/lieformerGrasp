import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))
import open3d as o3d
import h5py
import argparse
import numpy as np
from acronym import GripperMesh


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
    parser.add_argument(
        "--vis", type=lambda x: not (str(x).lower() == 'false'), default=False, help="show the visualization."
    )
    return parser.parse_args()


def main(args):
    grasp_data = h5py.File(args.input_h5[0], "r")
    points = grasp_data["points"]
    quality = grasp_data["quality"]
    print("the same of whole train dataset:", quality.shape)
    success_grasp_num = []
    no_success_grasp = []
    grasp_distribution = []
    for i in range(quality.shape[0]):
        success_grasp = np.where(quality[i] == 1)[0]
        # print("quality[i]: ", quality[i].shape, "success_grasp: ", success_grasp.shape[0])
        if success_grasp.shape[0] <= 500:
            no_success_grasp.append(i)
        else:
            success_grasp_num.append(success_grasp.shape[0])

        grasp_distribution.append(success_grasp.shape[0])

    print("min:", min(success_grasp_num))
    print("shape: ", len(success_grasp_num))
    print("the number of no succes: ", len(no_success_grasp))
    # print(points.shape)
    #
    # quality = grasp_data["quality"]
    # print("quality:", quality.shape)
    # success = np.where(quality[0] == 1)
    # fall = np.where(quality[0] == 0)
    # print("success: ", success[0].shape)
    # print("fall: ", fall[0].shape)
    #
    # success = np.where(quality[1] == 1)
    # fall = np.where(quality[1] == 0)
    # print("success: ", success[0].shape)
    # print("fall: ", fall[0].shape)
    #
    # index = 600
    # print("labels:", grasp_data["labels"][index])
    # print("T: ", grasp_data["Transform"].shape)
    print(args.vis)
    if args.vis:
        index = np.random.choice(no_success_grasp, 1)[0]
        success_grasp = np.where(quality[index] == 1)[0]
        pcd_new = o3d.geometry.PointCloud()
        pcd_new.points = o3d.utility.Vector3dVector(points[index])
        pcd_new.paint_uniform_color([0, 0, 1])
        pcd_points = [pcd_new]
        gripper_finger = GripperMesh()
        success_grasp = [0, 1, 2, 3, 4, 5]
        choose = np.random.choice(success_grasp, 5)
        for i in choose:
            T = grasp_data["Transform"][index][i]
            gripper = gripper_finger.create_gripper_marker_open3d(T=T)
            pcd_points.append(gripper)

            start_points = gripper_finger.get_graspPoint(T).reshape(1, 3)
            select_grasp_end_points = grasp_data["grasp_points"][index][i].reshape(1, 3)
            # select_grasp_points_normal = grasp_data["grasp_points_normal"][index][i].reshape(1, 3)
            approach_direction = (start_points[0] - T[:3, 3]) / np.linalg.norm((start_points[0] - T[:3, 3]))
            approach_direction = approach_direction.reshape(1, 3)
            extend_grasp_points = start_points[0] + approach_direction * 0.03
            # T = grasp_data["Transform"][index][i]
            # x = np.asarray([0, 0, 6.59999996e-02, 1]).reshape(4, 1)
            # x = np.matmul(T, x)
            # middle_points = x[:3].reshape(1, 3)
            # y = distance.cdist(middle_points, points[index])
            # print(y.shape)
            # hh = np.argmin(y)
            # print(hh)
            # my_selected = [points[index][hh]]
            # new_points = [T[:3, 3], x[:3].reshape(3)]
            my_pcd = o3d.geometry.PointCloud()
            my_pcd.points = o3d.utility.Vector3dVector(start_points)
            my_pcd.paint_uniform_color([1, 0, 1])
            pcd_points.append(my_pcd)

            my_pcd2 = o3d.geometry.PointCloud()
            my_pcd2.points = o3d.utility.Vector3dVector(select_grasp_end_points)
            my_pcd2.paint_uniform_color([1, 0, 1])
            pcd_points.append(my_pcd2)

            my_pcd3 = o3d.geometry.PointCloud()
            my_pcd3.points = o3d.utility.Vector3dVector(extend_grasp_points)
            my_pcd3.paint_uniform_color([1, 0, 1])
            pcd_points.append(my_pcd3)

            points_line = [start_points[0], select_grasp_end_points[0], extend_grasp_points[0]]
            lines = [
                [0, 1],
                [0, 2]
            ]
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(points_line),
                lines=o3d.utility.Vector2iVector(lines),
            )
            colors = [[1, 0, 0] for i in range(len(lines))]
            line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
            pcd_points.append(line_set)
            # my_pcd = o3d.geometry.PointCloud()
            # my_pcd.points = o3d.utility.Vector3dVector(np.asarray(new_points))
            # my_pcd.paint_uniform_color([0, 0, 1])
            # pcd_points.append(my_pcd)

        o3d.visualization.draw_geometries(pcd_points)
    # grasp_data = LoadData(filename=args.input_h5[0])
    # obj_mesh = grasp_data.load_mesh(mesh_root_dir=args.mesh_root)
    # T, success = grasp_data.load_grasp()
    # successful_grasps = [
    #     create_gripper_marker(color=[0, 255, 0]).apply_transform(t)
    #     for t in T[np.random.choice(np.where(success == 1)[0], args.num_grasps)]
    # ]
    # print(len(successful_grasps))
    # failed_grasps = [
    #     create_gripper_marker(color=[255, 0, 0]).apply_transform(t)
    #     for t in T[np.random.choice(np.where(success == 0)[0], args.num_grasps)]
    # ]
    # trimesh.Scene([obj_mesh] + successful_grasps).show()


if __name__ == '__main__':
    parser = make_parser()
    main(args=parser)
