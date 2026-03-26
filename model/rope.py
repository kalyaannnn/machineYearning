import torch
from typing import Optional


def precompute_rope_freqs(
    head_dim: int,
    seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> tuple:
    """
    Precompute RoPE cos/sin tables as real tensors.

    Returns real-valued (cos, sin) instead of complex freqs_cis so that
    torch.compile / TorchInductor can fully fuse the rotation kernel.

    Args:
        head_dim: dimension of each attention head (must be even)
        seq_len:  maximum sequence length to precompute for
        theta:    RoPE base frequency
        device:   torch device

    Returns:
        (cos, sin): both [seq_len, head_dim // 2], float32
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)   # [seq_len, head_dim // 2]
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, freqs_cis: tuple) -> torch.Tensor:
    """
    Apply rotary embeddings using real-valued cos/sin rotation.

    Equivalent to complex multiplication but uses only real ops so
    torch.compile can fully optimise the kernel without falling back
    to eager for complex number support.

    Rotation formula for each consecutive pair (x0, x1) at angle θ:
        x0' = x0 * cos(θ) - x1 * sin(θ)
        x1' = x0 * sin(θ) + x1 * cos(θ)

    Args:
        x:         [B, T, n_heads, head_dim]  — real tensor (bf16 or fp16)
        freqs_cis: (cos, sin) each [T, head_dim // 2], float32

    Returns:
        rotated tensor of same shape and dtype as x
    """
    cos, sin = freqs_cis

    # split x into consecutive pairs along head_dim
    # x_: [B, T, n_heads, head_dim // 2, 2]
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)
    x0, x1 = x_[..., 0], x_[..., 1]   # each [B, T, n_heads, head_dim // 2]

    # broadcast cos/sin over batch and head dims
    # [T, head_dim // 2] → [1, T, 1, head_dim // 2]
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    # apply 2D rotation
    r0 = x0 * cos - x1 * sin
    r1 = x0 * sin + x1 * cos

    # interleave back: [B, T, n_heads, head_dim // 2, 2] → [B, T, n_heads, head_dim]
    rotated = torch.stack([r0, r1], dim=-1).reshape(*x.shape)
    return rotated.type_as(x)
