import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[3]
sys.path.append(str(root))
from pytorch_utils.components.nvidia_grasp_dataset import NvidiaGraspDataSet

if __name__ == '__main__':
    dataset = NvidiaGraspDataSet(train=False, load=False)
    dataset.down_sampling(train=False)
