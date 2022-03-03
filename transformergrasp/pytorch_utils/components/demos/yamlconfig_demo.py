import sys
from pathlib import Path
import yaml

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[3]
sys.path.append(str(root))
from pytorch_utils.components.yaml_utils import YAMLConfig

if __name__ == '__main__':
    yamlconfig = YAMLConfig(config="tnetgrasp.yaml")
    yamlconfig.get_all_keys()
