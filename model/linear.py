from torch import nn
import torch.nn.functional as F

class LinearNet(nn.Module):
    def __init__(self, init_dim, inner_dim, dropout):
        super().__init__()
        
        self.fn_init = nn.Linear(init_dim, inner_dim)
        self.fn_out = nn.Linear(inner_dim, init_dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        out = self.drop1(F.gelu(self.fn_init(x)))
        out = self.drop2(self.fn_out(out))
        
        return out
