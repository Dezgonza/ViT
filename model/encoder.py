from torch import nn
from .linear import LinearNet
from .attention import Attention

class EncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, linear_dim, dim_head=64, dropout=0.):
        super().__init__()

        self.att = Attention(dim, num_heads, dim_head, dropout)
        self.linear = LinearNet(dim, linear_dim, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, img):
        att_out = self.att(self.norm1(img))
        x = att_out + img
        mlp_out = self.linear(self.norm2(x))
        out = mlp_out + x

        return out

class Encoder(nn.Module):
    def __init__(self, n_layers, **layer_args):
        super().__init__()
        
        self.layers = nn.ModuleList([EncoderLayer(**layer_args) \
                                     for _ in range(n_layers)])
                                    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            
        return x
