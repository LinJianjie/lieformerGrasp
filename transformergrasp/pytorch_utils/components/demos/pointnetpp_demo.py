import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[3]
sys.path.append(str(root))
from pytorch_utils.components.pointNetPP.pointnet_util import *
from pytorch_utils.components.pointNetPP.pointNetPPModule import PointNetPPCHead
from pytorch_utils.components.yaml_utils import YAMLConfig

if __name__ == '__main__':
    x = torch.rand(2, 1024, 3)
    features = None
    # PS = PointNetSetAbstractionMsg(ratio=0.5, radius_list=[0.5], max_sample_list=[64], in_channel=3,
    #                                mlp_list=[[64, 128, 128]])
    # PS(x, features)
    # yaml_file = "../../components/pointNetmodel/demo.yaml"
    # with open(yaml_file) as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    yaml_file = "tnetgrasp.yaml"
    yaml_config = YAMLConfig(yaml_file)
    PPhead = PointNetPPCHead(yaml_config.config)
    PPhead(x)
