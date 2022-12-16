from torch import nn
from .linear import LinearNet
from .attention import Attention

class EncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, dim_heads, dim_linear, dropout):
        super().__init__()

        self.att = Attention(dim, num_heads, dim_heads, dropout)
        self.linear = LinearNet(dim, dim_linear, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, img):
        att_out = self.att(self.norm1(img))
        x = self.dropout(att_out) + img
        mlp_out = self.linear(self.norm2(x))
        out = self.dropout(mlp_out) + x

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
