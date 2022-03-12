# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: This is a 2D Cosine Attention Module inspired by
# [cosFormer: Rethinking Softmax in Attention](https://arxiv.org/abs/2202.08791).
# ------------------------------------------------------------------------------ #

from math import pi
import torch
from torch import nn
from torch.nn import functional as F


def linear_attention(q, k, v):
    k_cumsum = k.sum(dim=-2)
    D = torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd->...ne', context, q)
    return out, D

class CosAttn2d(nn.Module):
    def __init__(self, N, n_head=8, M=16):
        """
        This module implements a cosine attention mechanism designed for
        grid features, e.g. feature maps of images. Note that there is no 
        learnable weights in this module. It's only responsible for computing
        KQV with linear time complexity.
        
        Args:
            N: edge length of 2d grid
            n_head: number of heads
            M: a constant, M > N
        """
        super().__init__()
        self.N = N
        self.M = M
        self.n_head = n_head
        idx = torch.arange(0, N)
        freq = pi / (2 * M)
        _cos = torch.cos(idx * freq)
        _sin = torch.sin(idx * freq)
        icos_jcos = (_cos.view(-1, 1) * _cos.view(1, -1)).unsqueeze(0)
        icos_jsin = (_cos.view(-1, 1) * _sin.view(1, -1)).unsqueeze(0)
        isin_jcos = (_sin.view(-1, 1) * _cos.view(1, -1)).unsqueeze(0)
        isin_jsin = (_sin.view(-1, 1) * _sin.view(1, -1)).unsqueeze(0)
        attn_coef = torch.cat([icos_jcos, icos_jsin, isin_jcos, isin_jsin], dim=0)
        self.register_buffer('attn_coef', attn_coef)
    
    def flatten(self, x):
        b, _, H, W = x.shape
        x = x.view(b, self.n_head, -1, H, W)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(b, self.n_head, H*W, -1)
        return x
    
    def forward(self, q, k, v):
        """
        Args:
            q:   [batch_size, channel, height, width], transformed Q
            k:   [batch_size, channel, height, width], transformed K
            v:   [batch_size, channel, height, width], transformed V
        Returns:
            out: [batch_size, channel, height, width]
        """
        BS, C, H, W = q.shape
        data_normalizer = (C ** -0.25)
        v = self.flatten(v)
        v = v.unsqueeze(1).repeat(1, 4, 1, 1, 1).view(BS * 4, self.n_head, H*W, -1)
        q = F.relu(q * data_normalizer, True) + 1e-5  # (BS, C, H, W)
        k = F.relu(k * data_normalizer, True) + 1e-5  # (BS, C, H, W)
        q = q[:, None, :, :, :] * self.attn_coef[None, :, None, :, :]  # (BS, 4, C, H, W)
        k = k[:, None, :, :, :] * self.attn_coef[None, :, None, :, :]  # (BS, 4, C, H, W)
        q = self.flatten(q.view(BS * 4, C, H, W))  # (BS*4, head, H*W, C//head)
        k = self.flatten(k.view(BS * 4, C, H, W))  # (BS*4, head, H*W, C//head)

        unnormed, D = linear_attention(q, k, v)
        unnormed = unnormed.view(BS, 4, self.n_head, H*W, -1).sum(dim=1)
        D_inv = 1. / D.view(BS, 4, self.n_head, -1).sum(dim=1)
        out = torch.einsum('...ne,...n->...ne', unnormed, D_inv)
        out = out.permute(0, 1, 3, 2).contiguous().view(BS, C, H, W)
        return out

if __name__ == '__main__':
    attn = CosAttn2d(14).cuda()
    x = torch.rand(32, 512, 14, 14).cuda()
    y = attn(x, x, x)
    print(y.shape)
