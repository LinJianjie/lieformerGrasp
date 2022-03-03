import sys
from pathlib import Path

import torch.nn as nn

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[3]
sys.path.append(str(root))


class TransformerPointNetPP(nn.Module):
    def __init__(self):
        super(TransformerPointNetPP, self).__init__()
        pass

    def forward(self):
        pass
