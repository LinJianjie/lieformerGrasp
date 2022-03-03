this is a transformer grasp

## The process for training

    1. Dataset is using the nvidia dataset with 2000 grasps for each object,
    2. the graps can be success or failed due to the occlusion, in this work, we can consider every grasp is sucess
    3. we using the Lie Algebra for preresenting the transformation, and use the Lie algebra for optimizing the grasp
    4. The grasp points are searched with KNN, and we use the Gaussian distribution to determin the predicted grasp quality
    5. We use the grasp approach angles to minimize the optimization


