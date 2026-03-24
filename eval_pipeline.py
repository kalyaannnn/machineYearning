"""
eval_pipeline.py — compact benchmark/evaluation suite for pretrain vs SFT.

Usage:
  python eval_pipeline.py \
    --pretrain-ckpt checkpoints/pretrain/<run_id>/final.pt \
    --sft-ckpt checkpoints/sft/<run_id>/best.pt
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from torch.amp import autocast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import load_checkpoint
from pipeline_utils import (
    FIXED_PROMPT_BATTERY,
    PROMPT_TEMPLATE,
    append_jsonl,
    decode_generated_text,
    extract_numeric_answer,
    load_codemath_tokenizer,
    make_run_id,
    prepare_stage_paths,
    promotion_gate,
    score_prompt_output,
    write_json,
)


REPO = "raokalyaan/codeMath"
SEQ_LEN = 4096
PRETRAIN_EVAL_BATCHES = 50
CODE_SUBSET_BATCHES = 50
MATH_SUBSET_ROWS = 80


def make_lm_targets(input_ids: torch.Tensor) -> torch.Tensor:
    targets = torch.roll(input_ids, shifts=-1, dims=1)
    targets[:, -1] = -100
    return targets


@torch.no_grad()
def evaluate_pretrain_val(model, batch_size: int = 4) -> dict:
    ds = load_dataset(
        REPO,
        data_files={"val": "pretrain_val/*.parquet"},
        split="val",
    )
    ds.set_format("torch", columns=["input_ids"])
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    n_batches = 0
    for batch in loader:
        if n_batches >= PRETRAIN_EVAL_BATCHES:
            break
        input_ids = batch["input_ids"].to("cuda")
        targets = make_lm_targets(input_ids)
        with autocast("cuda", dtype=torch.bfloat16):
            logits, loss = model(input_ids, targets)
        preds = logits.argmax(dim=-1)
        valid = targets != -100
        total_correct += (preds[valid] == targets[valid]).sum().item()
        total_count += valid.sum().item()
        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    return {
        "pretrain_val_loss": avg_loss,
        "pretrain_val_ppl": math.exp(min(avg_loss, 20.0)),
        "pretrain_val_next_token_acc": total_correct / max(1, total_count),
    }


@torch.no_grad()
def evaluate_code_subset(model, batch_size: int = 4) -> dict:
    ds = load_dataset(
        REPO,
        data_files={"train": "sft_train/*.parquet"},
        split="train[:2000]",
    )
    splits = ds.train_test_split(test_size=0.2, seed=42, shuffle=True)
    val_ds = splits["test"]
    val_ds.set_format("torch", columns=["input_ids", "labels"])
    loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    n_batches = 0
    for batch in loader:
        if n_batches >= CODE_SUBSET_BATCHES:
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
    return {
        "code_subset_loss": avg_loss,
        "code_subset_ppl": math.exp(min(avg_loss, 20.0)),
        "code_token_acc": total_correct / max(1, total_count),
    }


@torch.no_grad()
def evaluate_math_subset(model, tokenizer) -> dict:
    ds = load_dataset(
        REPO,
        data_files={"train": "grpo_train/gsm8k_train.parquet"},
        split=f"train[:{MATH_SUBSET_ROWS}]",
    )
    model.eval()
    correct = 0
    total = 0
    for row in ds:
        prompt = row["prompt"]
        gold = row["answer"].strip()
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        x = torch.tensor([prompt_ids[-SEQ_LEN:]], dtype=torch.long, device="cuda")
        out = model.generate(
            x,
            max_new_tokens=64,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen_ids = out[0][x.shape[1]:].tolist()
        decoded = decode_generated_text(tokenizer, gen_ids)["clean"]
        pred_num = extract_numeric_answer(decoded)
        gold_num = extract_numeric_answer(gold)
        correct += 1 if pred_num and pred_num == gold_num else 0
        total += 1
    return {"math_exact_match": correct / max(1, total), "math_total": total}


@torch.no_grad()
def evaluate_prompt_battery(model, tokenizer, model_tag: str, run_id: str, samples_jsonl: str) -> dict:
    model.eval()
    coh = 0.0
    adh = 0.0
    rel = 0.0
    fmt = 0.0
    inc = 0.0
    n = 0
    for prompt_spec in FIXED_PROMPT_BATTERY:
        prompt = PROMPT_TEMPLATE.format(instruction=prompt_spec["instruction"])
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        x = torch.tensor([prompt_ids[-SEQ_LEN:]], dtype=torch.long, device="cuda")
        out = model.generate(
            x,
            max_new_tokens=96,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen_ids = out[0][x.shape[1]:].tolist()
        decoded = decode_generated_text(tokenizer, gen_ids)
        score = score_prompt_output(prompt_spec, decoded["clean"])
        coh += score["coherence"]
        adh += score["adherence"]
        rel += score["relevance"]
        fmt += score["format"]
        inc += score["must_include"]
        n += 1
        append_jsonl(samples_jsonl, {
            "stage": "eval",
            "run_id": run_id,
            "model_tag": model_tag,
            "prompt_id": prompt_spec["id"],
            "instruction": prompt_spec["instruction"],
            "prompt": prompt,
            "output_clean": decoded["clean"],
            "output_raw": decoded["raw"],
            "scores": score,
        })
    return {
        "coherence_score": coh / max(1, n),
        "coherence_adherence": adh / max(1, n),
        "coherence_relevance": rel / max(1, n),
        "coherence_format": fmt / max(1, n),
        "coherence_must_include": inc / max(1, n),
    }


def format_report_table(pretrain_metrics: dict, sft_metrics: dict, gate: dict) -> str:
    lines = [
        "| Metric | Pretrain | SFT | Delta (SFT-Pretrain) |",
        "|---|---:|---:|---:|",
    ]
    keys = [
        "coherence_score",
        "code_token_acc",
        "math_exact_match",
        "code_subset_ppl",
        "pretrain_val_ppl",
    ]
    for key in keys:
        p = pretrain_metrics.get(key, float("nan"))
        s = sft_metrics.get(key, float("nan"))
        d = s - p
        lines.append(f"| {key} | {p:.4f} | {s:.4f} | {d:+.4f} |")
    lines.append("")
    lines.append("### Promotion Gate")
    lines.append(f"- Passed: `{gate['passed']}`")
    for check_name, ok in gate["checks"].items():
        lines.append(f"- {check_name}: `{ok}`")
    lines.append(f"- coherence_gain: {gate['coherence_gain']:+.4f}")
    lines.append(f"- code_delta: {gate['code_delta']:+.4f}")
    lines.append(f"- math_delta: {gate['math_delta']:+.4f}")
    return "\n".join(lines)


def evaluate_model(model, tokenizer, model_tag: str, run_id: str, samples_jsonl: str) -> dict:
    metrics = {}
    metrics.update(evaluate_pretrain_val(model))
    metrics.update(evaluate_code_subset(model))
    metrics.update(evaluate_math_subset(model, tokenizer))
    metrics.update(evaluate_prompt_battery(model, tokenizer, model_tag, run_id, samples_jsonl))
    return metrics


def main(pretrain_ckpt: str, sft_ckpt: str, run_id: str = None):
    assert torch.cuda.is_available(), "CUDA required for evaluation"
    run_id = run_id or make_run_id("eval")
    paths = prepare_stage_paths(stage="eval", run_id=run_id)
    tokenizer = load_codemath_tokenizer(REPO)

    print(f"Loading pretrain checkpoint: {pretrain_ckpt}")
    pretrain_model, _ = load_checkpoint(pretrain_ckpt, device="cuda")
    print(f"Loading sft checkpoint: {sft_ckpt}")
    sft_model, _ = load_checkpoint(sft_ckpt, device="cuda")

    t0 = time.time()
    pretrain_metrics = evaluate_model(pretrain_model, tokenizer, "pretrain", run_id, paths.samples_jsonl)
    sft_metrics = evaluate_model(sft_model, tokenizer, "sft", run_id, paths.samples_jsonl)
    gate = promotion_gate(pretrain_metrics, sft_metrics)
    retry_plan = None
    if not gate["passed"]:
        retry_plan = {
            "action": "retry_once_then_diagnose",
            "max_retries": 1,
            "controlled_change": {
                "target": "sft_lr",
                "scale": 0.5,
                "reason": "Gate failed: improve coherence without task regressions.",
            },
        }
        write_json("metrics/sft_retry_plan.json", retry_plan)

    payload = {
        "stage": "eval",
        "run_id": run_id,
        "pretrain_ckpt": pretrain_ckpt,
        "sft_ckpt": sft_ckpt,
        "elapsed_sec": time.time() - t0,
        "pretrain_metrics": pretrain_metrics,
        "sft_metrics": sft_metrics,
        "promotion_gate": gate,
        "retry_plan": retry_plan,
    }
    write_json(paths.metrics_json, payload)
    write_json(paths.metrics_json_run, payload)

    report = format_report_table(pretrain_metrics, sft_metrics, gate)
    report_path = f"metrics/eval_report_{run_id}.md"
    with open(report_path, "w") as f:
        f.write(report + "\n")
    with open("metrics/eval_report.md", "w") as f:
        f.write(report + "\n")

    print("\n=== Evaluation Complete ===")
    print(report)
    print(f"\nSaved metrics: {paths.metrics_json_run}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare pretrain vs SFT checkpoints.")
    parser.add_argument("--pretrain-ckpt", type=str, required=True)
    parser.add_argument("--sft-ckpt", type=str, required=True)
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()
    main(pretrain_ckpt=args.pretrain_ckpt, sft_ckpt=args.sft_ckpt, run_id=args.run_id)
