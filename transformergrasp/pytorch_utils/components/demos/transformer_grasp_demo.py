import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[3]
sys.path.append(str(root))
from pytorch_utils.components.pointNetPP.pointnet_util import *
from pytorch_utils.components.pointNetPP.pointNetPPModule import PointNetPPCHead
from pytorch_utils.components.transformerNet2D.encoder_decoder import TransformerEncoder
from pytorch_utils.components.yaml_utils import YAMLConfig

if __name__ == '__main__':
    yaml_file = "tnetgrasp.yaml"
    yaml_config = YAMLConfig(yaml_file)

    # printYaml(config)
    x = torch.rand(2, 1024, 3)
    pointNet = PointNetPPCHead(config=yaml_config.config)
    TNetEncoder = TransformerEncoder(N=yaml_config.config["TransformerConfig"]["N"],
                                     d_model=yaml_config.config["TransformerConfig"]["d_model"],
                                     d_ff=yaml_config.config["TransformerConfig"]["d_ff"],
                                     num_head=yaml_config.config["TransformerConfig"]["num_head"],
                                     dropout=yaml_config.config["TransformerConfig"]["dropout"],
                                     use_cMLP=yaml_config.config["TransformerConfig"]["use_cMLP"],
                                     pointNet=pointNet)
    z_hat = TNetEncoder(x)
    print("z_hat: ", z_hat.shape)
