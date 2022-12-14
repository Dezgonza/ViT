from torch import nn
import torch.nn.functional as F

class LinearNet(nn.Module):
    def __init__(self, init_dim, inner_dim, dropout):
        super().__init__()
        self.fn_init = nn.Linear(init_dim, inner_dim)
        self.fn_out = nn.Linear(inner_dim, inner_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.drop(F.gelu(self.fn_init(x)))
        out = self.drop(self.fn_out(x))
        
        return out
