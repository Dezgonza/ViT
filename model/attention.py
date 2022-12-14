import torch, math
from torch import nn
from einops import rearrange

class Attention(nn.Module):
    def __init__(self, dim, num_heads, dim_heads, dropout):
        super().__init__()

        self.num_heads = num_heads
        self.dim_heads = dim_heads
        self.D_h = num_heads * dim_heads

        self.split_to_qkv = nn.Linear(dim, 3 * self.D_h, bias = False)
        self.softmax = nn.Softmax(dim = -1)
        self.fc = nn.Linear(self.D_h, dim)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, img):
        qkv = self.split_to_qkv(img).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        #qkv = qkv.reshape(img.size()[0], img.size()[1], self.num_heads, 3 * self.dim_heads)
        #qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        #q, k, v = qkv.chunk(3, dim=-1)

        attention = torch.matmul(q, k.transpose(-1, -2))
        attention *= math.sqrt(self.dim_heads)
        attention = self.softmax(attention)
        attention = torch.matmul(attention, v)
        attention = self.fc(attention)
        attention = self.drop(attention)

        return attention
