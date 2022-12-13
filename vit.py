import torch
from torch import nn
from einops import repeat
from .utils.model.encoder import Encoder
from einops.layers.torch import Rearrange

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, n_layer, n_classes, dim, num_heads,
                 dim_heads, dropout, channels = 3):
        super().__init__()

        n_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        patch_dim = channels * patch_size[0] * patch_size[1]

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size[0], p2 = patch_size[1]),
            nn.Linear(patch_dim, dim),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)
        self.pos_emb = nn.Parameter(torch.rand(1, n_patches + 1, dim))
        self.encoder = Encoder(n_layer, dim, num_heads, dim_heads)
        self.to_latent = nn.Identity()
        self.fc_out = nn.Linear(dim, n_classes)
        self.norm = nn.LayerNorm(dim)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat([cls_tokens, img], dim = 1)
        x += self.pos_emb[:, : (n + 1)]
        x = self.dropout(x)
        x = self.encoder(x)[:, 0]
        x = self.to_latent(x)
        x = self.norm(x)
        out = self.fc_out(x)

        return out
