import torch, math
from torch import nn
from einops import rearrange

class Attention(nn.Module):
    def __init__(self, dim, num_heads, dim_head=64, dropout=0.):
        super().__init__()

        self.dim_head = dim_head
        self.num_heads = num_heads
        self.inner_dim = num_heads * dim_head

        self.split_to_qkv = nn.Linear(dim, 3 * self.inner_dim, bias = False)
        self.softmax = nn.Softmax(dim = -1)
        self.fc = nn.Linear(self.inner_dim, dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        
    def forward(self, img):
        qkv = self.split_to_qkv(img).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), qkv)
        qk = torch.matmul(q, k.transpose(-1, -2))
        qk /= math.sqrt(self.dim_head)
        att = self.drop1(self.softmax(qk))
        att = torch.matmul(att, v)
        att = rearrange(att, 'b h n d -> b n (h d)')
        att = self.fc(att)
        out = self.drop2(att)

        return out
