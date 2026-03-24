"""
sft.py — deterministic supervised fine-tuning stage.

Usage:
    python sft.py --pretrain-ckpt checkpoints/pretrain/<run_id>/final.pt
    python sft.py --pretrain-ckpt ... --run-id myrun --seed 42
"""

import argparse
import math
import os
import sys
import time

import torch
from datasets import load_dataset
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import wandb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import load_checkpoint, save_checkpoint
from pipeline_utils import (
    FIXED_PROMPT_BATTERY,
    PROMPT_TEMPLATE,
    append_jsonl,
    decode_generated_text,
    load_codemath_tokenizer,
    make_run_id,
    prepare_stage_paths,
    score_prompt_output,
    set_seed,
    write_json,
)


REPO = "raokalyaan/codeMath"
SEQ_LEN = 4096
SEED = 42

SFT_LR = 1e-5
MIN_LR = 1e-6
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
BETAS = (0.9, 0.95)

BATCH_SIZE = 4
GRAD_ACCUM = 8
TOTAL_STEPS = 2_000
EVAL_EVERY = 200
CHECKPOINT_EVERY = 200
NUM_WORKERS = 4
VAL_MAX_BATCHES = 100
VAL_SPLIT = 0.02
USE_COMPILE = False


def get_lr(step: int) -> float:
    if step < WARMUP_STEPS:
        return SFT_LR * (step / WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
    return MIN_LR + 0.5 * (SFT_LR - MIN_LR) * (1 + math.cos(math.pi * progress))


def build_loaders(seed: int):
    print("Loading SFT data from HuggingFace...")
    ds = load_dataset(
        REPO,
        data_files={"train": "sft_train/*.parquet"},
        split="train",
    )
    splits = ds.train_test_split(test_size=VAL_SPLIT, seed=seed, shuffle=True)
    train_ds = splits["train"]
    val_ds = splits["test"]

    train_ds.set_format("torch", columns=["input_ids", "labels"])
    val_ds.set_format("torch", columns=["input_ids", "labels"])
    print(f"  Train: {len(train_ds):,} rows")
    print(f"  Val:   {len(val_ds):,} rows")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
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


@torch.no_grad()
def evaluate(model, val_loader, max_batches: int = VAL_MAX_BATCHES) -> dict:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    n_batches = 0
    for batch in val_loader:
        if n_batches >= max_batches:
            break
        input_ids = batch["input_ids"].to("cuda")
        labels = batch["labels"].to("cuda")
        with autocast("cuda", dtype=torch.bfloat16):
            logits, loss = model(input_ids, labels)
        preds = logits.argmax(dim=-1)
        valid = labels != -100
        total_correct += (preds[valid] == labels[valid]).sum().item()
        total_count += valid.sum().item()
        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    ppl = math.exp(min(avg_loss, 20.0))
    acc = total_correct / max(1, total_count)
    return {"loss": avg_loss, "perplexity": ppl, "next_token_acc": acc}


@torch.no_grad()
def generate_prompt_samples(
    model,
    tokenizer,
    device: str,
    stage: str,
    run_id: str,
    step: int,
    samples_jsonl: str,
    max_new_tokens: int = 96,
) -> dict:
    model.eval()
    total_coherence = 0.0
    rows = 0
    for prompt_spec in FIXED_PROMPT_BATTERY:
        prompt = PROMPT_TEMPLATE.format(instruction=prompt_spec["instruction"])
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        x = torch.tensor([prompt_ids[-SEQ_LEN:]], dtype=torch.long, device=device)
        out = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen_ids = out[0][x.shape[1]:].tolist()
        decoded = decode_generated_text(tokenizer, gen_ids)
        score = score_prompt_output(prompt_spec, decoded["clean"])
        total_coherence += score["coherence"]
        rows += 1
        append_jsonl(samples_jsonl, {
            "stage": stage,
            "run_id": run_id,
            "step": step,
            "prompt_id": prompt_spec["id"],
            "instruction": prompt_spec["instruction"],
            "prompt": prompt,
            "output_clean": decoded["clean"],
            "output_raw": decoded["raw"],
            "scores": score,
        })
    return {"coherence_score": total_coherence / max(1, rows)}


def train_sft(pretrain_ckpt: str, run_id: str = None, seed: int = SEED, resume_from: str = None):
    assert torch.cuda.is_available(), "CUDA required for SFT"
    set_seed(seed)

    run_id = run_id or make_run_id("sft")
    paths = prepare_stage_paths(stage="sft", run_id=run_id)
    tokenizer = load_codemath_tokenizer(REPO)
    metrics_events = []

    if resume_from:
        model, ckpt = load_checkpoint(resume_from, device="cuda")
        start_step = ckpt["step"] + 1
    else:
        model, _ = load_checkpoint(pretrain_ckpt, device="cuda")
        start_step = 0

    if USE_COMPILE:
        print("Compiling model (SFT)...")
        model = torch.compile(model)
        print("Compile done.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=SFT_LR,
        betas=BETAS,
        eps=1e-8,
        weight_decay=WEIGHT_DECAY,
        fused=True,
    )
    scaler = GradScaler()

    train_loader, val_loader = build_loaders(seed=seed)
    train_iter = iter(train_loader)

    def next_batch():
        nonlocal train_iter
        try:
            return next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            return next(train_iter)

    wandb.init(
        project="llm-270m",
        name=f"sft-{run_id}",
        config={
            "stage": "sft",
            "run_id": run_id,
            "seed": seed,
            "repo": REPO,
            "pretrain_ckpt": pretrain_ckpt,
            "seq_len": SEQ_LEN,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "total_steps": TOTAL_STEPS,
            "eval_every": EVAL_EVERY,
            "checkpoint_every": CHECKPOINT_EVERY,
            "val_split": VAL_SPLIT,
            "prompt_template": PROMPT_TEMPLATE,
        },
        resume="allow",
    )

    write_json(
        os.path.join(paths.checkpoint_dir, "config_snapshot.json"),
        {
            "stage": "sft",
            "run_id": run_id,
            "seed": seed,
            "pretrain_ckpt": pretrain_ckpt,
            "resume_from": resume_from,
            "train_config": {
                "sft_lr": SFT_LR,
                "min_lr": MIN_LR,
                "warmup_steps": WARMUP_STEPS,
                "weight_decay": WEIGHT_DECAY,
                "grad_clip": GRAD_CLIP,
                "batch_size": BATCH_SIZE,
                "grad_accum": GRAD_ACCUM,
                "total_steps": TOTAL_STEPS,
                "eval_every": EVAL_EVERY,
                "checkpoint_every": CHECKPOINT_EVERY,
                "val_split": VAL_SPLIT,
            },
        },
    )

    print(f"\nSFT from step {start_step} to {TOTAL_STEPS}")
    print(f"Run ID: {run_id}")
    print(f"Checkpoints -> {paths.checkpoint_dir}\n")

    best_val = float("inf")
    best_ckpt_path = os.path.join(paths.checkpoint_dir, "best.pt")

    model.train()
    optimizer.zero_grad()
    for step in range(start_step, TOTAL_STEPS):
        t0 = time.time()
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        accum_loss = 0.0
        for _ in range(GRAD_ACCUM):
            batch = next_batch()
            input_ids = batch["input_ids"].to("cuda")
            labels = batch["labels"].to("cuda")
            with autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(input_ids, labels)
                raw_loss = loss.detach().item()
                loss = loss / GRAD_ACCUM
            scaler.scale(loss).backward()
            accum_loss += raw_loss

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        step_time = time.time() - t0
        loss_mean = accum_loss / GRAD_ACCUM
        tok_per_sec = (BATCH_SIZE * GRAD_ACCUM * SEQ_LEN) / step_time

        if step % 10 == 0:
            print(
                f"step {step:6d}/{TOTAL_STEPS} | "
                f"loss_mean {loss_mean:.4f} | "
                f"lr {lr:.2e} | "
                f"grad {grad_norm:.3f} | "
                f"{tok_per_sec/1e3:.1f}k tok/s"
            )
            wandb.log({
                "train/loss_mean": loss_mean,
                "train/loss_sum": accum_loss,
                "train/lr": lr,
                "train/grad_norm": grad_norm.item(),
                "train/tokens_per_sec": tok_per_sec,
                "train/step_time_ms": step_time * 1000,
            }, step=step)
            metrics_events.append({
                "kind": "train",
                "step": step,
                "loss_mean": loss_mean,
                "loss_sum": accum_loss,
                "lr": lr,
                "grad_norm": grad_norm.item(),
                "tokens_per_sec": tok_per_sec,
            })

        if step % EVAL_EVERY == 0 and step > 0:
            val_metrics = evaluate(model, val_loader)
            sample_metrics = generate_prompt_samples(
                model=model,
                tokenizer=tokenizer,
                device="cuda",
                stage="sft",
                run_id=run_id,
                step=step,
                samples_jsonl=paths.samples_jsonl,
            )
            print(
                f"  [eval] step {step} | "
                f"val_loss {val_metrics['loss']:.4f} | "
                f"val_ppl {val_metrics['perplexity']:.2f} | "
                f"val_acc {val_metrics['next_token_acc']:.4f} | "
                f"coherence {sample_metrics['coherence_score']:.4f}"
            )
            wandb.log({
                "val/loss": val_metrics["loss"],
                "val/perplexity": val_metrics["perplexity"],
                "val/next_token_acc": val_metrics["next_token_acc"],
                "val/coherence_score": sample_metrics["coherence_score"],
            }, step=step)
            metrics_events.append({
                "kind": "eval",
                "step": step,
                "val_loss": val_metrics["loss"],
                "val_perplexity": val_metrics["perplexity"],
                "val_next_token_acc": val_metrics["next_token_acc"],
                "coherence_score": sample_metrics["coherence_score"],
            })
            model.train()

            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                save_checkpoint(
                    model,
                    optimizer,
                    step,
                    loss_mean,
                    best_ckpt_path,
                    extra={"run_id": run_id, "stage": "sft", "best_by": "val/loss"},
                )
                print(f"  [best] updated -> {best_ckpt_path}")

        if step % CHECKPOINT_EVERY == 0 and step > 0:
            save_checkpoint(
                model,
                optimizer,
                step,
                loss_mean,
                f"{paths.checkpoint_dir}/step_{step:06d}.pt",
                extra={"run_id": run_id, "stage": "sft"},
            )

        if not math.isfinite(loss_mean):
            print(f"FATAL: loss_mean={loss_mean} at step {step}")
            save_checkpoint(
                model,
                optimizer,
                step,
                loss_mean,
                f"{paths.checkpoint_dir}/crash_step_{step:06d}.pt",
                extra={"run_id": run_id, "stage": "sft"},
            )
            wandb.finish(exit_code=1)
            sys.exit(1)

    save_checkpoint(
        model,
        optimizer,
        TOTAL_STEPS,
        loss_mean,
        f"{paths.checkpoint_dir}/last.pt",
        extra={"run_id": run_id, "stage": "sft"},
    )
    final_val = evaluate(model, val_loader, max_batches=200)
    final_samples = generate_prompt_samples(
        model=model,
        tokenizer=tokenizer,
        device="cuda",
        stage="sft",
        run_id=run_id,
        step=TOTAL_STEPS,
        samples_jsonl=paths.samples_jsonl,
    )
    wandb.log({
        "val/loss": final_val["loss"],
        "val/perplexity": final_val["perplexity"],
        "val/next_token_acc": final_val["next_token_acc"],
        "val/coherence_score": final_samples["coherence_score"],
    }, step=TOTAL_STEPS)

    best_metrics = None
    if os.path.exists(best_ckpt_path):
        best_model, _ = load_checkpoint(best_ckpt_path, device="cuda")
        best_metrics = evaluate(best_model, val_loader, max_batches=200)
        print(
            "Best checkpoint val | "
            f"loss {best_metrics['loss']:.4f} | "
            f"ppl {best_metrics['perplexity']:.2f} | "
            f"acc {best_metrics['next_token_acc']:.4f}"
        )
        wandb.log({
            "best_val/loss": best_metrics["loss"],
            "best_val/perplexity": best_metrics["perplexity"],
            "best_val/next_token_acc": best_metrics["next_token_acc"],
        }, step=TOTAL_STEPS)

    metrics_payload = {
        "stage": "sft",
        "run_id": run_id,
        "seed": seed,
        "pretrain_ckpt": pretrain_ckpt,
        "best_checkpoint": best_ckpt_path if os.path.exists(best_ckpt_path) else None,
        "last_checkpoint": f"{paths.checkpoint_dir}/last.pt",
        "events": metrics_events,
        "final": {
            "val_loss": final_val["loss"],
            "val_perplexity": final_val["perplexity"],
            "val_next_token_acc": final_val["next_token_acc"],
            "coherence_score": final_samples["coherence_score"],
            "best_checkpoint_eval": best_metrics,
        },
    }
    write_json(paths.metrics_json, metrics_payload)
    write_json(paths.metrics_json_run, metrics_payload)
    wandb.finish()
    return metrics_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic SFT stage")
    parser.add_argument("--pretrain-ckpt", type=str, required=True, help="Path to pretrain checkpoint.")
    parser.add_argument("--resume", type=str, default=None, help="Optional SFT checkpoint to resume from.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run ID.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    args = parser.parse_args()

    train_sft(
        pretrain_ckpt=args.pretrain_ckpt,
        run_id=args.run_id,
        seed=args.seed,
        resume_from=args.resume,
    )
