"""
pretrain.py — Pretraining script for the 270M LLM.

Usage (Colab):
    !python pretrain.py
    !python pretrain.py --resume checkpoints/pretrain/step_005000.pt

Features:
    - Loads data directly from raokalyaan/codeMath on HuggingFace
    - Optional torch.compile for throughput boost
    - bf16 mixed precision
    - Gradient accumulation (effective batch = 128 seqs = ~524k tokens)
    - Cosine LR schedule with linear warmup
    - Checkpoints every 500 steps
    - Val loss every 1000 steps
    - Full W&B logging
"""

import os
import sys
import math
import time
import argparse
import gc

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from datasets import load_dataset
import wandb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Transformer, ModelConfig, save_checkpoint, load_checkpoint

# ── Config ─────────────────────────────────────────────────────────────────

REPO             = "raokalyaan/codeMath"

# Model
SEQ_LEN          = 4096

# Optimiser
PEAK_LR          = 3e-4
MIN_LR           = 3e-5
WARMUP_STEPS     = 1000
WEIGHT_DECAY     = 0.1
GRAD_CLIP        = 1.0
BETAS            = (0.9, 0.95)

# Batch
BATCH_SIZE       = 4           # sequences per device step (stable on A100 80GB @ 4096)
GRAD_ACCUM       = 32          # effective batch = 128 seqs = ~524k tokens/step

# Runtime
USE_COMPILE      = False       # enable after stable run if you want extra throughput

# Schedule
TOTAL_STEPS      = 26_000      # 3.5B tokens / 524k ≈ 26k steps (~8-10 hrs)

# Logging & checkpointing
LOG_EVERY        = 10
EVAL_EVERY       = 1000
CHECKPOINT_EVERY = 500
CHECKPOINT_DIR   = "./checkpoints/pretrain"

# DataLoader
NUM_WORKERS      = 4

# ── LR Schedule ────────────────────────────────────────────────────────────

def get_lr(step: int) -> float:
    """Linear warmup → cosine decay."""
    if step < WARMUP_STEPS:
        return PEAK_LR * (step / WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
    return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1 + math.cos(math.pi * progress))


# ── Data ───────────────────────────────────────────────────────────────────

def build_loaders():
    """
    Load pretrain_train and pretrain_val shards from HuggingFace.
    Each row is exactly SEQ_LEN=4096 token IDs — no padding, no packing needed.
    """
    print("Loading pretrain data from HuggingFace...")

    train_ds = load_dataset(
        REPO,
        data_files={"train": "pretrain_train/*.parquet"},
        split="train",
    )
    val_ds = load_dataset(
        REPO,
        data_files={"val": "pretrain_val/*.parquet"},
        split="val",
    )

    train_ds.set_format("torch", columns=["input_ids"])
    val_ds.set_format("torch", columns=["input_ids"])

    print(f"  Train: {len(train_ds):,} sequences  "
          f"({len(train_ds) * SEQ_LEN / 1e9:.2f}B tokens)")
    print(f"  Val:   {len(val_ds):,} sequences  "
          f"({len(val_ds) * SEQ_LEN / 1e9:.2f}B tokens)")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,       # keep batch sizes uniform
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )

    return train_loader, val_loader


# ── Evaluation ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_loader, max_batches: int = 50) -> float:
    """
    Compute average val loss over up to max_batches batches.
    Capped so eval doesn't take too long mid-training.
    """
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    for batch in val_loader:
        if n_batches >= max_batches:
            break
        input_ids = batch["input_ids"].to("cuda")
        with autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(input_ids, input_ids)
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(1, n_batches)


# ── Training ───────────────────────────────────────────────────────────────

def train(resume_from: str = None):

    # ── W&B ────────────────────────────────────────────────────────────────
    wandb.init(
        project="llm-270m",
        name="pretrain-run-1",
        config={
            "model":          "270M decoder-only transformer",
            "params_M":       268,
            "d_model":        1024,
            "n_layers":       24,
            "n_heads":        16,
            "n_kv_heads":     8,
            "ffn_dim":        4096,
            "vocab_size":     32000,
            "seq_len":        SEQ_LEN,
            "peak_lr":        PEAK_LR,
            "min_lr":         MIN_LR,
            "warmup_steps":   WARMUP_STEPS,
            "weight_decay":   WEIGHT_DECAY,
            "grad_clip":      GRAD_CLIP,
            "batch_size":     BATCH_SIZE,
            "grad_accum":     GRAD_ACCUM,
            "eff_batch_seqs": BATCH_SIZE * GRAD_ACCUM,
            "eff_batch_toks": BATCH_SIZE * GRAD_ACCUM * SEQ_LEN,
            "total_steps":    TOTAL_STEPS,
            "total_tokens":   TOTAL_STEPS * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN,
            "gpu":            torch.cuda.get_device_name(0),
            "data_repo":      REPO,
        },
        resume="allow",
    )

    # ── Model ──────────────────────────────────────────────────────────────
    config = ModelConfig()
    start_step = 0

    if resume_from:
        print(f"Resuming from {resume_from}...")
        model, ckpt = load_checkpoint(resume_from, device="cuda")
        start_step  = ckpt["step"] + 1
        print(f"  Resuming from step {start_step}")
    else:
        model = Transformer(config).to("cuda")

    # torch.compile can increase memory pressure; keep off by default for stability.
    if USE_COMPILE:
        print("Compiling model (takes ~60s first time)...")
        model = torch.compile(model)
        print("Compile done.")
    else:
        print("torch.compile disabled (USE_COMPILE=False).")

    # ── Optimiser ──────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PEAK_LR,
        betas=BETAS,
        eps=1e-8,
        weight_decay=WEIGHT_DECAY,
        fused=True,           # fused AdamW — ~10% faster on A100
    )

    if resume_from and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        print("  Optimizer state restored.")

    scaler = GradScaler()

    # ── Data ───────────────────────────────────────────────────────────────
    train_loader, val_loader = build_loaders()
    train_iter = iter(train_loader)

    def next_batch():
        """Infinite iterator — loops dataset when exhausted."""
        nonlocal train_iter
        try:
            return next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            return next(train_iter)

    # ── Pre-training throughput check ──────────────────────────────────────
    print("\nRunning 20-step throughput check...")
    model.train()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    t_check = time.time()
    for _ in range(20):
        ids = next_batch()["input_ids"].to("cuda")
        with autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(ids, ids)
            loss = loss / GRAD_ACCUM
        scaler.scale(loss).backward()
        if _ % GRAD_ACCUM == GRAD_ACCUM - 1:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    torch.cuda.synchronize()
    check_elapsed = time.time() - t_check
    check_tps = (20 * BATCH_SIZE * SEQ_LEN) / check_elapsed
    check_eta = (3_500_000_000 / (check_tps * GRAD_ACCUM)) / 3600
    print(f"  Throughput: {check_tps * GRAD_ACCUM:,.0f} effective tok/s")
    print(f"  ETA for 3.5B tokens: {check_eta:.1f} hours")
    if check_tps * GRAD_ACCUM < 80_000:
        print("  WARNING: <80k tok/s — check Flash Attention + torch.compile")
    else:
        print("  Throughput OK. Starting full training run.")

    # ── Main Training Loop ─────────────────────────────────────────────────
    print(f"\nTraining from step {start_step} to {TOTAL_STEPS}")
    print(f"Checkpoints → {CHECKPOINT_DIR}/\n")

    model.train()
    optimizer.zero_grad()

    for step in range(start_step, TOTAL_STEPS):
        t0 = time.time()

        # update lr
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # gradient accumulation loop
        accum_loss = 0.0
        for micro_step in range(GRAD_ACCUM):
            input_ids = next_batch()["input_ids"].to("cuda")

            with autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(input_ids, input_ids)
                loss = loss / GRAD_ACCUM

            scaler.scale(loss).backward()
            accum_loss += loss.item()

        # unscale → clip → step → zero
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), GRAD_CLIP
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        step_time = time.time() - t0
        tok_per_sec = (BATCH_SIZE * GRAD_ACCUM * SEQ_LEN) / step_time
        tokens_seen = (step + 1) * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN

        # ── Logging ────────────────────────────────────────────────────────
        if step % LOG_EVERY == 0:
            print(
                f"step {step:6d}/{TOTAL_STEPS} | "
                f"loss {accum_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"grad {grad_norm:.3f} | "
                f"{tok_per_sec/1e3:.1f}k tok/s | "
                f"{tokens_seen/1e9:.3f}B tokens"
            )
            wandb.log({
                "train/loss":           accum_loss,
                "train/lr":             lr,
                "train/grad_norm":      grad_norm.item(),
                "train/tokens_per_sec": tok_per_sec,
                "train/tokens_seen":    tokens_seen,
                "train/step_time_ms":   step_time * 1000,
            }, step=step)

        # ── Val loss ───────────────────────────────────────────────────────
        if step % EVAL_EVERY == 0 and step > 0:
            val_loss = evaluate(model, val_loader)
            print(f"  [eval] step {step} | val_loss {val_loss:.4f}")
            wandb.log({"val/loss": val_loss}, step=step)
            model.train()

        # ── Checkpoint ─────────────────────────────────────────────────────
        if step % CHECKPOINT_EVERY == 0 and step > 0:
            save_checkpoint(
                model, optimizer, step, accum_loss,
                f"{CHECKPOINT_DIR}/step_{step:06d}.pt",
                extra={"tokens_seen": tokens_seen},
            )

        # ── NaN guard ──────────────────────────────────────────────────────
        if not math.isfinite(accum_loss):
            print(f"  FATAL: loss is {accum_loss} at step {step}. Stopping.")
            save_checkpoint(
                model, optimizer, step, accum_loss,
                f"{CHECKPOINT_DIR}/crash_step_{step:06d}.pt",
            )
            wandb.finish(exit_code=1)
            sys.exit(1)

    # ── Final checkpoint ───────────────────────────────────────────────────
    print("\nTraining complete.")
    save_checkpoint(
        model, optimizer, TOTAL_STEPS, accum_loss,
        f"{CHECKPOINT_DIR}/final.pt",
        extra={"tokens_seen": TOTAL_STEPS * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN},
    )

    # final val loss
    val_loss = evaluate(model, val_loader, max_batches=200)
    print(f"Final val loss: {val_loss:.4f}")
    wandb.log({"val/loss": val_loss}, step=TOTAL_STEPS)
    wandb.finish()

    return model


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain the 270M LLM")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from, e.g. checkpoints/pretrain/step_005000.pt"
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required for training"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    train(resume_from=args.resume)
