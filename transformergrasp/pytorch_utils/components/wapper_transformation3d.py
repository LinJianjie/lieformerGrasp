import sys

from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))
import transforms3d
import numpy as np


class Transform3D:
    @staticmethod
    def quaternions2mat(w, x, y, z):
        q = [w, x, y, z]
        return transforms3d.quaternions.quat2mat(q)

    @staticmethod
    def mat2quaternion(M):
        return transforms3d.quaternions.mat2quat(M)

    @staticmethod
    def affine_composed(translation, Rotation, Z=np.ones(3), S=np.zeros(3)):
        A = transforms3d.affines.compose(translation, Rotation, Z, S)
        return A

    @staticmethod
    def affine_decomposed(A):
        trans, rotation, Z, S = transforms3d.affines.decompose44(A)
        return trans, rotation, Z, S

    @staticmethod
    def euler2mat(x=0, y=0, z=0):
        return transforms3d.euler.euler2mat(ai=x, aj=y, ak=z)

    @staticmethod
    def euler2Hom(Rx=0, Ry=0, Rz=0, tx=0, ty=0, tz=0):
        R = transforms3d.euler.euler2mat(ai=Rx, aj=Ry, ak=Rz)
        T = np.eye(4)
        T[:3, :3] = R
        T[0, 3] = tx
        T[1, 3] = ty
        T[2, 3] = tz
        return T

    @staticmethod
    def axisangle2Hom(axis, angle, tx=0, ty=0, tz=0):
        R = transforms3d.axangles.axangle2mat(axis=axis, angle=angle)
        T = np.eye(4)
        T[:3, :3] = R
        T[0, 3] = tx
        T[1, 3] = ty
        T[2, 3] = tz
        return T
