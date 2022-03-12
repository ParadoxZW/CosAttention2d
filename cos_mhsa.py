# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: It shows how to use the cosine attention module (`cos_attn2d.py`) 
# in a convolutional MHSA module (with LayerNorm and Residual).
# ------------------------------------------------------------------------------ #

import math
import torch
from torch import nn
from torch.nn import functional as F
from cos_attn2d import CosAttn2d

class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(size, 1, 1))
        self.bias = nn.Parameter(torch.zeros(size, 1, 1))

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        std = x.std(1, keepdim=True)
        normed = (x - mean) / (std + self.eps)

        return self.weight * normed + self.bias



class CosMHSA(nn.Module):
    def __init__(self, n_head, external_dim, size, dropout=0.1):
        """
        A multi-head attention module using cosine attention mechanism.
        
        Args:
            n_head (int): number of head
            external_dim (int): dimension of external feature
            size (int): size of 2d feature map
            dropout (float): dropout rate
        """
        super().__init__()
        self.n_head = n_head
        self.external_dim = external_dim
        self.internal_dim = external_dim // self.n_head

        self.linear_v = nn.Conv2d(self.external_dim, self.external_dim, (1, 1), bias=False)
        self.linear_k = nn.Conv2d(self.external_dim, self.external_dim, (1, 1), bias=False)
        self.linear_q = nn.Conv2d(self.external_dim, self.external_dim, (1, 1), bias=False)
        self.linear_merge = nn.Conv2d(self.external_dim, self.external_dim, (1, 1))
        self.cos_att = CosAttn2d(size, M=size+2) 

        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(external_dim)

    def forward(self, x):
        """
        Args:
            x:   [batch_size, channel, height, width], 2d feature map
        Returns: 
            out: [batch_size, channel, height, width]
        """

        v = self.linear_v(x)
        k = self.linear_k(x)
        q = self.linear_q(x)
        atted = self.cos_att(q, k, v)

        atted = self.linear_merge(atted)
        atted = self.norm(
            x + self.dropout(atted)
        )

        return atted

if __name__ == '__main__':
    x = torch.randn(32, 512, 14, 14).cuda()
    model = CosMHSA(n_head=8, external_dim=512, size=14)
    model.cuda()
    out = model(x)
    print(out.shape)