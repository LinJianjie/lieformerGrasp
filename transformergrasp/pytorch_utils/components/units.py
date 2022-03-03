import numpy as np


class Math:
    @staticmethod
    def Deg2Rad(deg):
        return deg * np.pi / 180

    @staticmethod
    def Rad2Deg(rad):
        return rad * 180 / np.pi
