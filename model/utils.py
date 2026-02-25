

import math
import time
import torch
from pathlib import Path
from typing import Optional

from .config import ModelConfig
from .transformer import Transformer


# ── Device Helper ──────────────────────────────────────────────────────────

def get_device() -> str:
    """
    Auto-detect best available device.
        cuda -> A100 / any NVIDIA GPU (Colab)
        mps  -> Apple Silicon M1/M2/M3/M4 (local MacBook)
        cpu  -> fallback
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Checkpoint I/O ─────────────────────────────────────────────────────────

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    path: str,
    extra: Optional[dict] = None,
):
    """
    Save model + optimizer + training state to a single .pt file.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step":      step,
        "loss":      loss,
        "config":    model.config.to_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"[ckpt] Saved -> {path}  (step={step}, loss={loss:.4f})")


def load_checkpoint(
    path: str,
    device: Optional[str] = None,
) -> tuple:
    """
    Load a checkpoint saved with save_checkpoint().

    Args:
        path:   path to .pt checkpoint file
        device: target device. If None, auto-detects (cuda -> mps -> cpu)

    Returns:
        (model, ckpt) where ckpt contains optimizer state, step, loss, etc.

    Usage:
        model, ckpt = load_checkpoint("checkpoints/step_5000.pt")
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
    """
    if device is None:
        device = get_device()

    ckpt   = torch.load(path, map_location=device)
    config = ModelConfig.from_dict(ckpt["config"])
    model  = Transformer(config).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"[ckpt] Loaded <- {path}  (step={ckpt['step']}, loss={ckpt['loss']:.4f})")
    return model, ckpt


# ── Smoke Test ─────────────────────────────────────────────────────────────

def smoke_test(config: Optional[ModelConfig] = None, verbose: bool = True):
    """
    Full sanity check. Works on cuda (A100), mps (Apple Silicon), and cpu.

    Run before pretraining:
        python -c "from model.utils import smoke_test; smoke_test()"

    Checks:
      1. Parameter count
      2. Forward pass + loss magnitude
      3. Backward pass (no NaN gradients)
      4. Throughput estimate
      5. Generation (greedy + sampled)
      6. Memory usage
    """
    device = get_device()
    if config is None:
        config = ModelConfig()

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Model Smoke Test  [{device.upper()}]")
    print(sep)

    # 1. Parameter count
    model  = Transformer(config).to(device)
    params = model.get_num_params()
    print(f"\n[1] Parameters")
    print(f"    Total:           {params['total'] / 1e6:.2f}M")
    print(f"    Non-embedding:   {params['non_embedding'] / 1e6:.2f}M")
    print(f"    Embedding:       {params['embedding'] / 1e6:.2f}M")
    print(f"    Tied:            {config.tie_embeddings}")

    # 2. Forward pass
    B, T = 2, 256
    x = torch.randint(0, config.vocab_size, (B, T), device=device)
    y = torch.randint(0, config.vocab_size, (B, T), device=device)

    logits, loss = model(x, y)
    expected_loss = math.log(config.vocab_size)
    print(f"\n[2] Forward pass (B={B}, T={T})")
    print(f"    Logits shape:    {list(logits.shape)}")
    print(f"    Loss:            {loss.item():.4f}")
    print(f"    Expected (rand): {expected_loss:.4f}  <- should be close")
    assert abs(loss.item() - expected_loss) < 2.0, \
        f"Loss {loss.item():.4f} far from expected {expected_loss:.4f}"

    # 3. Backward pass
    loss.backward()
    grad_norms = {
        name: p.grad.norm().item()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    has_nan = any(math.isnan(v) for v in grad_norms.values())
    print(f"\n[3] Backward pass")
    print(f"    Params with gradients: {len(grad_norms)}")
    print(f"    NaN gradients:         {has_nan}  <- should be False")
    assert not has_nan, "NaN gradients detected"
    model.zero_grad()

    # 4. Throughput
    B_bench, T_bench = 4, 512
    x_b = torch.randint(0, config.vocab_size, (B_bench, T_bench), device=device)
    y_b = torch.randint(0, config.vocab_size, (B_bench, T_bench), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for _ in range(3):   # warmup
        _, lb = model(x_b, y_b); lb.backward(); opt.step(); opt.zero_grad()

    if device == "cuda":  torch.cuda.synchronize()
    elif device == "mps": torch.mps.synchronize()

    t0, N = time.time(), 20
    for _ in range(N):
        _, lb = model(x_b, y_b); lb.backward(); opt.step(); opt.zero_grad()

    if device == "cuda":  torch.cuda.synchronize()
    elif device == "mps": torch.mps.synchronize()
    elapsed = time.time() - t0

    tok_bench = (N * B_bench * T_bench) / elapsed
    tok_full  = tok_bench * (16 * 4096) / (B_bench * T_bench)
    eta_hrs   = (3_500_000_000 / tok_full) / 3600

    print(f"\n[4] Throughput  (B={B_bench}, T={T_bench})")
    print(f"    Measured:                {tok_bench:>10,.0f} tok/s")
    print(f"    Projected (B=16,T=4096): {tok_full:>10,.0f} tok/s")
    print(f"    Est. pretrain time:      {eta_hrs:.1f} hrs for 3.5B tokens")
    if device == "cuda" and tok_full < 80_000:
        print("    WARNING: <80k tok/s on CUDA — check Flash Attn + torch.compile")
    elif device == "mps":
        print("    INFO: MPS (Apple Silicon) — expected ~5-15k tok/s locally.")
        print("          Full training on Colab A100 only.")
    elif device == "cpu":
        print("    INFO: CPU only — for code checking, not training.")

    # 5. Generation
    model.eval()
    prompt = torch.randint(0, config.vocab_size, (1, 8), device=device)
    with torch.no_grad():
        out_g = model.generate(prompt, max_new_tokens=10, do_sample=False)
        out_s = model.generate(prompt, max_new_tokens=10, temperature=0.8, top_p=0.9)
    print(f"\n[5] Generation")
    print(f"    Greedy:  {out_g.shape[1]} tokens")
    print(f"    Sampled: {out_s.shape[1]} tokens")

    # 6. Memory
    print(f"\n[6] Memory  ({device})")
    if device == "cuda":
        torch.cuda.empty_cache()
        alloc  = torch.cuda.memory_allocated() / 1e9
        reserv = torch.cuda.memory_reserved()  / 1e9
        total  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"    Allocated: {alloc:.2f} GB")
        print(f"    Reserved:  {reserv:.2f} GB")
        print(f"    GPU total: {total:.1f} GB  |  headroom: {total - reserv:.1f} GB")
    elif device == "mps":
        alloc = torch.mps.current_allocated_memory() / 1e9
        print(f"    MPS allocated: {alloc:.2f} GB  (shared with system RAM)")
    else:
        print(f"    CPU — no GPU memory tracking")

    print(f"\n{sep}")
    print("  All checks passed")
    print(sep)
    return model