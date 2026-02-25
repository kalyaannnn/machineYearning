

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # gate projection
        self.w1 = nn.Linear(config.d_model, config.ffn_dim, bias=False)
        # down projection (maps back to residual stream)
        self.w2 = nn.Linear(config.ffn_dim, config.d_model, bias=False)
        # up projection (value branch of the gate)
        self.w3 = nn.Linear(config.d_model, config.ffn_dim, bias=False)

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, d_model]

        Computes:
            gate   = SiLU(x W1)    — gating signal
            value  = x W3          — value branch
            hidden = gate ⊙ value  — gated activation
            out    = hidden W2     — project back down
        """
        gate   = F.silu(self.w1(x))    # [B, T, ffn_dim]
        value  = self.w3(x)             # [B, T, ffn_dim]
        hidden = gate * value           # elementwise gating
        return self.dropout(self.w2(hidden))   # [B, T, d_model]