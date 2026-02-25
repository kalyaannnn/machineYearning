

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # learnable scale γ

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        # compute RMS over last dimension, keep dims for broadcasting
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms) * self.weight