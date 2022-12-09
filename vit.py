import torch
from torch import nn
from utils.model.encoder import Encoder

class ViT(nn.Module):
    def __init__(self, n_layer, n_classes, dim, num_heads, dim_heads, num_patches, dropout):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)
        self.pos_emb = nn.Parameter(torch.rand(1, num_patches + 1, dim))
        self.encoder = Encoder(n_layer, dim, num_heads, dim_heads)
        self.fc_out = nn.Linear(dim, n_classes)
        self.norm = nn.LayerNorm(dim)

    def forward(self, img):
        x = torch.cat([self.cls_token, img], dim = 1)
        x += self.pos_emb[:, : (n + 1)]
        x = self.dropout(x)
        x = self.encoder(x)[:, 0]
        x = self.norm(x)
        out = self.fc_out(x)

        return out
