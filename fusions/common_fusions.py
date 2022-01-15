import torch
from torch import nn

# Simple concatenation on dim 1
class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, modalities):
        flattened = []
        for modality in modalities:
            flattened.append(torch.flatten(modality, start_dim=1))
        return torch.cat(flattened, dim=1)