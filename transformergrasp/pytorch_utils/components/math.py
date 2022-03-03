import numpy as np


class Math:
    @staticmethod
    def Deg2Rad(deg):
        return deg * np.pi / 180

    @staticmethod
    def Rad2Deg(rad):
        return rad * 180 / np.pi

    @staticmethod
    def orthorgonal(vec):
        vec /= np.linalg.norm(vec)
        x = np.random.randn(3)
        x -= x.dot(vec) * vec
        x /= np.linalg.norm(x)
        y = np.cross(vec, x)
        z = vec
        return x, y, z
