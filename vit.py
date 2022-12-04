import torch
from torch import nn
from utils.encoder import Encoder

class ViT(nn.Module):
    def __init__(self, dim, num_heads, dim_heads, num_patches, n_classes):
        super().__init__()

        self.position_emb = nn.Parameter(torch.rand(1, num_patches + 1, dim))
        self.encoder = Encoder(dim, num_heads, dim_heads)

        self.fc_out = nn.Linear(dim, n_classes)

    def forward(self, img):
        x = self.encoder(img)
        out = self.fc_out(x)
        return out
