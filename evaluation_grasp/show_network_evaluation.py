import sys
from pathlib import Path

import numpy as np
import trimesh.collision

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
import h5py
from evaluation_grasp.acronym import GripperMesh
import torch
from transformergrasp.pytorch_utils.components.lie_groups.myse3 import SE3Matrix as SE3
from transformergrasp.pytorch_utils.components.dataUtils import DataUtils
from transformergrasp.pytorch_utils.components.wapper_transformation3d import Transform3D
from transformergrasp.pytorch_utils.components.math import Math


# todo Collision checking
# todo remove the wrong grasp

def in_collision_with(mesh, transform, min_distance=0.0, epsilon=1.0 / 1e3):
    """Check whether the scene is in collision with mesh. Optional: Define a minimum distance.

    Args:
        mesh (trimesh.Trimesh): Object mesh to test with scene.
        transform (np.ndarray): Pose of the object mesh as a 4x4 homogenous matrix.
        min_distance (float, optional): Minimum distance that is considered in collision. Defaults to 0.0.
        epsilon (float, optional): Epsilon for minimum distance check. Defaults to 1.0/1e3.

    Returns:
        bool: Whether the object mesh is colliding with anything in the scene.
    """
    collision_manager = trimesh.collision.CollisionManager()
    colliding = collision_manager.in_collision_single(mesh=mesh, transform=transform)
    if not colliding and min_distance > 0.0:
        distance = collision_manager.min_distance_single(
            mesh=mesh, transform=transform
        )
        if distance < min_distance - epsilon:
            colliding = True
    return colliding


def coarse_to_fine(approach_direction, angle, T_base, T_Pred):
    T_1 = Transform3D.euler2Hom(Rx=0, Ry=0, Rz=0, tx=-T_base[0], ty=-T_base[1], tz=-T_base[2])
    T_angle = Transform3D.axisangle2Hom(axis=approach_direction, angle=Math.Deg2Rad(angle))
    T_2 = Transform3D.euler2Hom(Rx=0, Ry=0, Rz=0, tx=T_base[0], ty=T_base[1], tz=T_base[2])
    T_fine = np.matmul(T_2, np.matmul(T_angle, np.matmul(T_1, T_Pred)))
    return T_fine


def move_against_approach(approach_direction, scale, T_Pred):
    T_1 = Transform3D.euler2Hom(Rx=0, Ry=0, Rz=0,
                                tx=-approach_direction[0] * scale,
                                ty=-approach_direction[1] * scale,
                                tz=-approach_direction[2] * scale)
    T_fine = np.matmul(T_1, T_Pred)
    return T_fine


def compute_distance(p1, p2):
    dis = torch.cdist(torch.from_numpy(p2.reshape(1, 3)), torch.from_numpy(p1), p=2)
    min_dis, index = torch.min(dis, dim=-1)
    return p1[index]


def check_approach_angle(approach_direction, corr_p1, T):
    approach_2 = corr_p1 - T[:3, 3]
    approach_2 /= np.linalg.norm(approach_2)
    theta = Math.Rad2Deg(np.arccos(np.matmul(approach_direction, approach_2.transpose())))
    return theta


def approach_to_contact(approach_direction, scale, obj_mesh, T_current):
    gripper_finger_template = GripperMesh()
    gripper_mesh = gripper_finger_template.create_gripper_marker_trimesh(T=T_current)
    gripper_collision_ = trimesh.collision.CollisionManager()
    gripper_collision_.add_object(name="gripper", mesh=gripper_mesh)
    object_collision = trimesh.collision.CollisionManager()
    object_collision.add_object(name="object", mesh=obj_mesh)
    has_object_collision = object_collision.in_collision_other(gripper_collision_)
    k = 1
    T_fine = T_current
    while not has_object_collision:
        T_1 = Transform3D.euler2Hom(Rx=0, Ry=0, Rz=0,
                                    tx=approach_direction[0] * scale * k,
                                    ty=approach_direction[1] * scale * k,
                                    tz=approach_direction[2] * scale * k)
        T_fine = np.matmul(T_1, T_current)
        gripper_mesh = gripper_finger_template.create_gripper_marker_trimesh(T=T_fine)
        gripper_collision_ = trimesh.collision.CollisionManager()
        gripper_collision_.add_object(name="gripper", mesh=gripper_mesh)
        has_object_collision = object_collision.in_collision_other(gripper_collision_)
        k = k + 1
        if k > 100:
            T_fine = T_current
            break

    return T_fine


def show_results(i, num_grasp=256, vis=True, approach=False):
    trimesh_scence = []
    class_index = i
    ycb_points = ycb_original_data["points"]
    shift_points, center = DataUtils.zeroCenter(torch.from_numpy(ycb_points[class_index]))
    points = grasp_data["point_cloud"]

    pre_quality = grasp_data["pred_quality"]
    pre_epsilon = grasp_data["pred_epsilon"]

    # object_points = o3d.geometry.PointCloud()
    # object_points.points = o3d.utility.Vector3dVector(points[class_index][0])
    # object_points.paint_uniform_color([0, 0, 1])

    object_mesh_path = class_file[class_index]
    object_mesh = trimesh.load(object_mesh_path)
    center = center.squeeze()
    object_mesh.apply_translation(-center.numpy())
    object_collision = trimesh.collision.CollisionManager()
    object_collision.add_object(name="object", mesh=object_mesh)
    trimesh_scence.append(object_mesh)
    pcd_points = list()
    # pcd_points.append(object_points)

    transform = np.eye(4)
    min_coordinate = np.min(points[class_index][0], axis=0)
    transform[:3, 3] = [0, 0, min_coordinate[2]]
    table_mesh = GripperMesh.create_a_table(bwh=[0.5, 0.5, 0.02], transform=transform)
    table_collision = trimesh.collision.CollisionManager()
    table_collision.add_object(name="table", mesh=table_mesh)
    # pcd_points.append(table_mesh.as_open3d)
    trimesh_scence = trimesh_scence + [table_mesh]
    for index in np.arange(0, num_grasp):
        # print("==> the quality", pre_quality[class_index][0][index])
        if pre_quality[class_index][0][index] > 0:
            # print("==> the quality", pre_quality[class_index][0][index])
            # print("==> load the grippers")
            gripper_finger = GripperMesh()
            T = SE3.exp(torch.from_numpy(pre_epsilon[class_index][0][index])).squeeze().numpy()
            start_points = gripper_finger.get_graspPoint(T).reshape(1, 3)
            # select_grasp_points_normal = grasp_data["grasp_points_normal"][index][i].reshape(1, 3)
            approach_direction = (start_points[0] - T[:3, 3]) / np.linalg.norm((start_points[0] - T[:3, 3]))

            or_x, or_y, or_z = Math.orthorgonal(vec=approach_direction)
            pred_collision = False
            corre_p1 = compute_distance(p1=points[class_index][0], p2=T[:3, 3])
            approach_angle = check_approach_angle(approach_direction=approach_direction, corr_p1=corre_p1, T=T)
            gripper_mesh = gripper_finger.create_gripper_marker_trimesh(T=T)

            gripper_collision_OR = trimesh.collision.CollisionManager()
            gripper_collision_OR.add_object(name="gripper_or", mesh=gripper_mesh)
            has_table_collision = table_collision.in_collision_other(gripper_collision_OR)
            if approach_angle < 15:
                for k in range(10):
                    # T_fine = coarse_to_fine(approach_direction=or_z, angle=-10 * k, T_base=start_points[0], T_Pred=T)
                    T_fine = move_against_approach(approach_direction=or_z, scale=0.01 * k, T_Pred=T)
                    # T_angle = Transform3D.axisangle2Hom(axis=-1 * approach_direction, angle=Math.Deg2Rad(10 * k))
                    # T_euler = Transform3D.euler2Hom(Rx=Math.Deg2Rad(10 * k),
                    #                                 Ry=Math.Deg2Rad(0),
                    #                                 Rz=Math.Deg2Rad(0),
                    #                                 tx=approach_direction[0] * k * 0.1,
                    #                                 ty=approach_direction[1] * k * 0.1,
                    #                                 tz=approach_direction[2] * k * 0.1)
                    # T_new = np.matmul(T_2, np.matmul(T_angle, np.matmul(T_1, T)))
                    # gripper = gripper_finger.create_gripper_marker_open3d(T=T_fine)
                    gripper_mesh = gripper_finger.create_gripper_marker_trimesh(T=T_fine)

                    gripper_collision = trimesh.collision.CollisionManager()
                    gripper_collision.add_object(name="gripper", mesh=gripper_mesh)

                    has_object_collision = object_collision.in_collision_other(gripper_collision)
                    # print("is_collision: ", has_object_collision)
                    if has_object_collision and not pred_collision:
                        pred_collision = has_object_collision

                    if not has_object_collision:
                        has_table_collision = table_collision.in_collision_other(gripper_collision)
                        start_points = gripper_finger.get_graspPoint(T_fine).reshape(1, 3)
                        if not has_table_collision and min_coordinate[2] < T_fine[2, 3]:
                            pred_collision = False
                            if approach:
                                T_contact = approach_to_contact(approach_direction=or_z, scale=0.001,
                                                                obj_mesh=object_mesh,
                                                                T_current=T_fine)
                                gripper_mesh = gripper_finger.create_gripper_marker_trimesh(T=T_contact)

                            # pcd_points.append(gripper_mesh.as_open3d)
                            # trimesh_scence.append(gripper_mesh)
                            trimesh_scence = trimesh_scence + [gripper_mesh]
                            break

    #

    # pcd_points.append(object_mesh.as_open3d)
    # print("at i: ", i, "the class name: ", class_file[i].split("/")[-3], " ==> the size of success: ",
    #       len(pcd_points))

    if vis:
        # o3d.visualization.draw_geometries(pcd_points)
        trimesh.Scene(trimesh_scence).show()
    if len(trimesh_scence) > 2:
        return True
    else:
        return False


def show_one_case(name, num_grasp=500):
    for i in range(86):
        if name in class_file[i]:
            show_results(i, vis=True, num_grasp=num_grasp)


def compute_success(num_grasp=200):
    has_success = 0
    for i in range(86):

        has_grasp = show_results(i, num_grasp=num_grasp, vis=False, approach=False)
        print("at: ", i, " ==> has_grasp: ", has_grasp)
        if has_grasp:
            has_success = has_success + 1
    # o3d.visualization.draw_geometries(pcd_points)
    print("the number of sucess_grasp: ", has_success, " the sucess rate: ", has_success / 86.0)


if __name__ == '__main__':
    grasp_data = h5py.File("../scripts/checkpoints/20210513_163145_best_model_PTgrasp_YCB_256_evaluation.h5", "r")
    ycb_original_data = h5py.File("../data/NvidiaGrasp/ycb_dataset.h5", "r")
    class_file = np.load("ycb_dataset_class.npy")
    quality = grasp_data["pred_quality"]
    show_one_case(name="025_mug", num_grasp=quality.shape[-1])
    #compute_success(num_grasp=quality.shape[-1])
