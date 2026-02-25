

import torch
from typing import Optional


def precompute_rope_freqs(
    head_dim: int,
    seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Precompute complex RoPE frequency tensor.

    Args:
        head_dim: dimension of each attention head (must be even)
        seq_len:  maximum sequence length to precompute for
        theta:    RoPE base — controls the frequency range
        device:   torch device

    Returns:
        freqs_cis: complex tensor of shape [seq_len, head_dim // 2]
                   freqs_cis[t, i] = e^(i * t * theta^(-2i/d))
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    # inverse frequencies — one per pair of dimensions: [head_dim // 2]
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )

    # positions: [seq_len]
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)

    # outer product → [seq_len, head_dim // 2]
    freqs = torch.outer(positions, inv_freq)

    # convert to complex: magnitude = 1, angle = freq → e^(i·freq)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings to a query or key tensor.

    Rotation is applied per head, per position, in the complex plane.
    Each consecutive pair of dimensions (d, d+1) forms a 2D subspace
    that gets rotated by the corresponding frequency.

    Args:
        x:         [B, T, n_heads, head_dim]   — real tensor
        freqs_cis: [T, head_dim // 2]          — complex tensor

    Returns:
        rotated tensor of same shape as x
    """
    # view last dim as pairs → [..., head_dim // 2, 2] then complex
    x_complex = torch.view_as_complex(
        x.float().reshape(*x.shape[:-1], -1, 2)
    )                               # [B, T, n_heads, head_dim // 2]

    # broadcast freqs over batch and head dims
    # freqs_cis: [T, head_dim // 2] → [1, T, 1, head_dim // 2]
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)

    # complex multiply = rotation in 2D subspace
    x_rotated = x_complex * freqs_cis  # [B, T, n_heads, head_dim // 2]

    # back to real, reshape to original dims
    x_out = torch.view_as_real(x_rotated).reshape(*x.shape)
    return x_out.type_as(x)          # preserve original dtype (bf16/fp16)