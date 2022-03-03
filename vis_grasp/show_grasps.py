import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))
import sys
import json
import trimesh
import argparse
import numpy as np
from acronym import LoadData, create_gripper_marker


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


def main(args):
    grasp_data = LoadData(filename=args.input_h5[0])
    obj_mesh = grasp_data.load_mesh(mesh_root_dir=args.mesh_root)
    T, success = grasp_data.load_grasp()
    successful_grasps = [
        create_gripper_marker(color=[0, 255, 0]).apply_transform(t)
        for t in T[np.random.choice(np.where(success == 1)[0], args.num_grasps)]
    ]
    print(len(successful_grasps))
    failed_grasps = [
        create_gripper_marker(color=[255, 0, 0]).apply_transform(t)
        for t in T[np.random.choice(np.where(success == 0)[0], args.num_grasps)]
    ]
    trimesh.Scene([obj_mesh] + successful_grasps).show()


if __name__ == '__main__':
    parser = make_parser()
    main(args=parser)
