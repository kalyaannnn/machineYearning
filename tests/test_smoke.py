"""
Smoke tests for the Transformer model.

Runs forward/backward passes with dummy tensors to catch shape mismatches,
NaN gradients, and other training-time errors before full training.

Run:
    python -m pytest tests/test_smoke.py -v
    # or
    python tests/test_smoke.py
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import ModelConfig
from model.transformer import Transformer


# Small config for fast smoke tests
def _smoke_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=1000,
        d_model=256,
        n_layers=4,
        n_heads=8,
        n_kv_heads=4,
        ffn_dim=512,
        seq_len=128,
        dropout=0.0,
        tie_embeddings=True,
    )


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Shape & forward tests ───────────────────────────────────────────────────


def test_forward_training_shapes():
    """Forward pass with targets: logits [B,T,V], loss scalar."""
    config = _smoke_config()
    model = Transformer(config).to(_get_device())
    model.train()

    B, T = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (B, T), device=model.embed.weight.device)
    targets = torch.randint(0, config.vocab_size, (B, T), device=model.embed.weight.device)

    logits, loss = model(input_ids, targets)

    assert logits.shape == (B, T, config.vocab_size), f"logits {logits.shape}"
    assert loss.dim() == 0, f"loss should be scalar, got {loss.shape}"
    assert not math.isnan(loss.item()), "loss is NaN"
    assert not math.isinf(loss.item()), "loss is inf"
    assert loss.item() > 0, "loss should be positive"


def test_forward_inference_shapes():
    """Forward pass without targets: logits [B,1,V], loss=None."""
    config = _smoke_config()
    model = Transformer(config).to(_get_device())
    model.eval()

    B, T = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (B, T), device=model.embed.weight.device)

    with torch.no_grad():
        logits, loss = model(input_ids, targets=None)

    assert logits.shape == (B, 1, config.vocab_size), f"logits {logits.shape}"
    assert loss is None


def test_forward_with_mask_ignore_index():
    """Targets with -100 (ignore) should not affect loss."""
    config = _smoke_config()
    model = Transformer(config).to(_get_device())
    model.train()

    B, T = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (B, T), device=model.embed.weight.device)
    targets = input_ids.clone()
    targets[:, :4] = -100  # mask first 4 positions

    logits, loss = model(input_ids, targets)

    assert logits.shape == (B, T, config.vocab_size)
    assert not math.isnan(loss.item())
    assert loss.item() > 0


def test_forward_edge_seq_len_one():
    """Minimal sequence length T=1."""
    config = _smoke_config()
    model = Transformer(config).to(_get_device())
    model.train()

    B = 4
    input_ids = torch.randint(0, config.vocab_size, (B, 1), device=model.embed.weight.device)
    targets = torch.randint(0, config.vocab_size, (B, 1), device=model.embed.weight.device)

    logits, loss = model(input_ids, targets)

    assert logits.shape == (B, 1, config.vocab_size)
    assert not math.isnan(loss.item())


def test_forward_max_seq_len():
    """Sequence length at config.seq_len."""
    config = _smoke_config()
    model = Transformer(config).to(_get_device())
    model.train()

    B, T = 2, config.seq_len
    input_ids = torch.randint(0, config.vocab_size, (B, T), device=model.embed.weight.device)
    targets = torch.randint(0, config.vocab_size, (B, T), device=model.embed.weight.device)

    logits, loss = model(input_ids, targets)

    assert logits.shape == (B, T, config.vocab_size)
    assert not math.isnan(loss.item())


# ── Backward tests ───────────────────────────────────────────────────────────


def test_backward_no_nan_gradients():
    """Backward pass must not produce NaN gradients."""
    config = _smoke_config()
    model = Transformer(config).to(_get_device())
    model.train()

    B, T = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (B, T), device=model.embed.weight.device)
    targets = torch.randint(0, config.vocab_size, (B, T), device=model.embed.weight.device)

    logits, loss = model(input_ids, targets)
    loss.backward()

    for name, p in model.named_parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), f"NaN in {name}"
            assert not torch.isinf(p.grad).any(), f"Inf in {name}"

    model.zero_grad()


def test_backward_all_params_receive_grads():
    """All trainable params (except possibly tied) should get gradients."""
    config = _smoke_config()
    model = Transformer(config).to(_get_device())
    model.train()

    B, T = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (B, T), device=model.embed.weight.device)
    targets = torch.randint(0, config.vocab_size, (B, T), device=model.embed.weight.device)

    logits, loss = model(input_ids, targets)
    loss.backward()

    # Embedding and head are tied, so both get grads
    grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
    for name, g in grads.items():
        assert g is not None, f"No gradient for {name}"

    model.zero_grad()


# ── Generation tests ────────────────────────────────────────────────────────


def test_generate_greedy():
    """Greedy generation produces correct shape."""
    config = _smoke_config()
    model = Transformer(config).to(_get_device())
    model.eval()

    prompt_len, max_new = 8, 10
    prompt = torch.randint(0, config.vocab_size, (1, prompt_len), device=model.embed.weight.device)

    with torch.no_grad():
        out = model.generate(prompt, max_new_tokens=max_new, do_sample=False)

    assert out.shape == (1, prompt_len + max_new)


def test_generate_sampled():
    """Sampled generation produces correct shape."""
    config = _smoke_config()
    model = Transformer(config).to(_get_device())
    model.eval()

    prompt_len, max_new = 4, 6
    prompt = torch.randint(0, config.vocab_size, (2, prompt_len), device=model.embed.weight.device)

    with torch.no_grad():
        out = model.generate(prompt, max_new_tokens=max_new, temperature=0.8, top_p=0.9)

    assert out.shape == (2, prompt_len + max_new)


def test_generate_batch():
    """Batch generation (B>1) works."""
    config = _smoke_config()
    model = Transformer(config).to(_get_device())
    model.eval()

    B, prompt_len, max_new = 3, 4, 5
    prompt = torch.randint(0, config.vocab_size, (B, prompt_len), device=model.embed.weight.device)

    with torch.no_grad():
        out = model.generate(prompt, max_new_tokens=max_new, do_sample=False)

    assert out.shape == (B, prompt_len + max_new)


# ── Training step simulation ─────────────────────────────────────────────────


def test_training_step_simulation():
    """Simulate one training step: forward, backward, optimizer step."""
    config = _smoke_config()
    model = Transformer(config).to(_get_device())
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    B, T = 4, 64
    input_ids = torch.randint(0, config.vocab_size, (B, T), device=model.embed.weight.device)
    targets = torch.randint(0, config.vocab_size, (B, T), device=model.embed.weight.device)

    for step in range(3):
        logits, loss = model(input_ids, targets)
        assert not math.isnan(loss.item()), f"NaN loss at step {step}"
        loss.backward()
        opt.step()
        opt.zero_grad()


# ── Config / param sanity ───────────────────────────────────────────────────


def test_param_count_matches_tied_embeddings():
    """With tie_embeddings, embed and head share weights."""
    config = _smoke_config()
    config.tie_embeddings = True
    model = Transformer(config)

    assert model.head.weight is model.embed.weight


def test_loss_magnitude_random_targets():
    """With random targets, loss should be ~ln(vocab_size)."""
    config = _smoke_config()
    model = Transformer(config).to(_get_device())
    model.train()

    B, T = 8, 64
    input_ids = torch.randint(0, config.vocab_size, (B, T), device=model.embed.weight.device)
    targets = torch.randint(0, config.vocab_size, (B, T), device=model.embed.weight.device)

    logits, loss = model(input_ids, targets)
    expected = math.log(config.vocab_size)
    assert abs(loss.item() - expected) < 2.0, f"loss {loss.item():.4f} far from ln(V)={expected:.4f}"


# ── Run as script ────────────────────────────────────────────────────────────


def run_all():
    """Run all smoke tests. Use with pytest or standalone."""
    tests = [
        test_forward_training_shapes,
        test_forward_inference_shapes,
        test_forward_with_mask_ignore_index,
        test_forward_edge_seq_len_one,
        test_forward_max_seq_len,
        test_backward_no_nan_gradients,
        test_backward_all_params_receive_grads,
        test_generate_greedy,
        test_generate_sampled,
        test_generate_batch,
        test_training_step_simulation,
        test_param_count_matches_tied_embeddings,
        test_loss_magnitude_random_targets,
    ]
    passed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {fn.__name__}: {e}")
            raise
    print(f"\n{passed}/{len(tests)} smoke tests passed")


if __name__ == "__main__":
    try:
        import pytest
        pytest.main([__file__, "-v"])
    except ImportError:
        print("Smoke tests (run without pytest)\n")
        run_all()
        print("\nAll checks passed.")
