import h5py


class GraspAcronym:
    def __init__(self, filename):
        self.data = h5py.File(filename, "r")

    @property
    def object_file(self):
        return self.data["object/file"][()]

    @property
    def object_com(self):
        return self.data["/object/com"][()]

    @property
    def object_friction(self):
        return self.data["/object/friction"][()]

    @property
    def object_scale(self):
        return self.data["/object/scale"][()]

    @property
    def grasp_transformer(self):
        return self.data["grasps/transforms"][()]

    @property
    def grasp_configuration(self):
        return self.data["/gripper/configuration"][()]

    @property
    def grasp_type(self):
        return self.data["/gripper/type"][()]

    @property
    def quality_success(self):
        return self.data["grasps/qualities/flex/object_in_gripper"][()]

    @property
    def object_motion_during_closing_angular(self):
        return self.data["grasps/qualities/flex/object_motion_during_closing_angular"][()]

    @property
    def object_motion_during_closing_linear(self):
        return self.data["grasps/qualities/flex/object_motion_during_closing_linear"][()]

    @property
    def object_motion_during_shaking_angular(self):
        return self.data["grasps/qualities/flex/object_motion_during_shaking_angular"][()]

    @property
    def object_motion_during_shaking_linear(self):
        return self.data["grasps/qualities/flex/object_motion_during_shaking_linear"][()]


if __name__ == '__main__':
    graspacronym = GraspAcronym(filename="../grasps/1Shelves_a9c2bcc286b68ee217a3b9ca1765e2a4_0.007691275299398608.h5")
    print(graspacronym.object_file)
    print(graspacronym.quality_success, graspacronym.quality_success.shape)
    print(graspacronym.object_motion_during_closing_angular, graspacronym.object_motion_during_closing_angular.shape)
    print(graspacronym.object_motion_during_closing_linear, graspacronym.object_motion_during_closing_linear.shape)
    print(graspacronym.object_motion_during_shaking_angular, graspacronym.object_motion_during_shaking_angular.shape)
    print(graspacronym.object_motion_during_shaking_linear, graspacronym.object_motion_during_shaking_linear.shape)
    print(graspacronym.grasp_transformer.shape)
    print(graspacronym.grasp_configuration)
    print(graspacronym.grasp_type)
    print(graspacronym.object_com)
    print(graspacronym.object_scale)
    print(graspacronym.object_friction)
