

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import ModelConfig
from .rope import apply_rope


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads    = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_groups   = config.n_groups    # Q heads per KV head
        self.head_dim   = config.head_dim
        self.d_model    = config.d_model

        # ── Projections (no bias anywhere) ─────────────────────────────────
        # Q projects to n_heads heads
        self.wq = nn.Linear(config.d_model, config.n_heads    * config.head_dim, bias=False)
        # K, V project to n_kv_heads heads (fewer)
        self.wk = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        # output projection
        self.wo = nn.Linear(config.n_heads  * config.head_dim, config.d_model,   bias=False)

        self.dropout_p = config.dropout   # passed to sdpa during training

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:         [B, T, d_model]
            freqs_cis: [T, head_dim // 2]   — precomputed RoPE frequencies
            mask:      [T, T] additive causal mask (0 or -inf), or None

        Returns:
            out: [B, T, d_model]
        """
        B, T, _ = x.shape

        # ── Project ────────────────────────────────────────────────────────
        q = self.wq(x).view(B, T, self.n_heads,    self.head_dim)  # [B,T,Hq,D]
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)  # [B,T,Hk,D]
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)  # [B,T,Hk,D]

        # ── Apply RoPE to Q and K ──────────────────────────────────────────
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        # ── Expand K and V to match Q head count ──────────────────────────
        # [B, T, n_kv_heads, D] → [B, T, n_heads, D]
        # each KV head is repeated n_groups times to pair with its Q heads
        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=2)
            v = v.repeat_interleave(self.n_groups, dim=2)

        # ── Transpose to [B, n_heads, T, head_dim] for SDPA ───────────────
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ── Scaled Dot-Product Attention ───────────────────────────────────
        # F.scaled_dot_product_attention:
        #   - uses Flash Attention 2 kernel if available (CUDA + flash-attn)
        #   - falls back to memory-efficient math attention otherwise
        #   - is_causal=True generates the causal mask internally (faster)
        dropout_p = self.dropout_p if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=dropout_p,
            is_causal=(mask is None),   # if no mask passed, use causal mode
        )                               # [B, n_heads, T, head_dim]

        # ── Merge heads and project out ────────────────────────────────────
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(attn_out)        # [B, T, d_model]