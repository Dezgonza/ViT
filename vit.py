import torch
from torch import nn
from einops import repeat
from model.encoder import Encoder
from einops.layers.torch import Rearrange

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, n_classes, n_layers, dim, num_heads,
                 linear_dim, dim_head=64, channels=3, dropout=0.):
        super().__init__()

        self.n_classes = n_classes
        n_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        patch_dim = channels * patch_size[0] * patch_size[1]

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1 = patch_size[0], p2 = patch_size[1]),
            nn.Linear(patch_dim, dim),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)
        self.pos_emb = nn.Parameter(torch.rand(1, n_patches + 1, dim))
        self.encoder = Encoder(n_layers, dim=dim, num_heads=num_heads,
                               dim_head=dim_head, linear_dim=linear_dim, dropout=dropout)
        self.to_latent = nn.Identity()
        self.norm = nn.LayerNorm(dim)
        self.fc_out = nn.Linear(dim, n_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_emb[:, :(n + 1)]
        x = self.dropout(x)
        x = self.encoder(x)[:, 0]
        x = self.to_latent(x)
        x = self.norm(x)
        out = self.fc_out(x)

        return out
