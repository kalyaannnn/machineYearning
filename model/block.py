
import torch
import torch.nn as nn
from typing import Optional

from .config import ModelConfig
from .norm import RMSNorm
from .attention import GroupedQueryAttention
from .ffn import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # norms applied BEFORE each sub-layer (pre-norm)
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn_norm  = RMSNorm(config.d_model, config.norm_eps)

        self.attn = GroupedQueryAttention(config)
        self.ffn  = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:         [B, T, d_model]   — residual stream
            freqs_cis: [T, head_dim//2]  — RoPE frequencies
            mask:      [T, T] or None    — causal mask

        Returns:
            x: [B, T, d_model]           — updated residual stream
        """
        # attention sub-layer: normalise → attend → add residual
        x = x + self.attn(self.attn_norm(x), freqs_cis, mask)
        # FFN sub-layer: normalise → transform → add residual
        x = x + self.ffn(self.ffn_norm(x))
        return x
