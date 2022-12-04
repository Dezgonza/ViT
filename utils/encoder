from torch import nn
from attention import Attention

class Encoder(nn.Module):
    def __init__(self, dim, num_heads, dim_heads):
        super().__init__()

        self.att_1 = Attention(dim, num_heads, dim_heads)
        self.att_2 = Attention(dim, num_heads, dim_heads)

    def forward(self, img):
        x = self.att_1(img)
        out = self.att_2(x)
        return out