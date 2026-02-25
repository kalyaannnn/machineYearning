"""
Colab smoke test for:
  - HF dataset: raokalyaan/codeMath
  - local model code in this repo

Run in Colab:
  !pip install -q torch datasets transformers huggingface_hub
  !python smoke_test_colab.py
"""

import math
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerFast


REPO_ID = "raokalyaan/codeMath"
DATA_FILE = "pretrain_train/starcoder_py_000.parquet"
TOKENIZER_DIR = "/tmp/codemath_tokenizer"
MAX_LEN = 512
BATCH_SIZE = 2
TRAIN_STEPS = 2


def log(msg: str) -> None:
    print(msg, flush=True)


def assert_ok(cond: bool, name: str, detail: str = "") -> None:
    if cond:
        log(f"[PASS] {name}" + (f" | {detail}" if detail else ""))
    else:
        raise RuntimeError(f"[FAIL] {name}" + (f" | {detail}" if detail else ""))


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tokenizer(repo_id: str) -> PreTrainedTokenizerFast:
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    filenames = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    for filename in filenames:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=f"tokenizer/{filename}",
        )
        # Ensure files are in TOKENIZER_DIR root for from_pretrained().
        shutil.copyfile(downloaded_path, os.path.join(TOKENIZER_DIR, filename))
    tok = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
    assert_ok(tok.vocab_size > 0, "tokenizer loads", f"vocab={tok.vocab_size}")
    return tok


def load_batch(repo_id: str, data_file: str, batch_size: int, max_len: int, device: str) -> torch.Tensor:
    ds = load_dataset(repo_id, data_files={"train": data_file}, split=f"train[:{batch_size}]")
    sample = ds[0]
    assert_ok("input_ids" in sample, "dataset has input_ids")
    seq_len = len(sample["input_ids"])
    assert_ok(seq_len > 0, "dataset row non-empty", f"row_len={seq_len}")

    rows = []
    for row in ds:
        ids = row["input_ids"][:max_len]
        rows.append(ids)

    batch = torch.tensor(rows, dtype=torch.long, device=device)
    assert_ok(batch.ndim == 2, "batch tensor shape", str(tuple(batch.shape)))
    return batch


def main() -> None:
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    from model import ModelConfig, Transformer

    device = get_device()
    log("=" * 60)
    log("codeMath Colab smoke test")
    log("=" * 60)
    log(f"torch={torch.__version__} | device={device}")
    if device == "cuda":
        log(f"gpu={torch.cuda.get_device_name(0)}")

    tokenizer = load_tokenizer(REPO_ID)
    batch = load_batch(REPO_ID, DATA_FILE, BATCH_SIZE, MAX_LEN, device)

    # Tiny config so smoke test runs quickly on any Colab GPU.
    cfg = ModelConfig()
    cfg = replace(
        cfg,
        d_model=256,
        n_layers=4,
        n_heads=8,
        n_kv_heads=4,
        ffn_dim=1024,
        seq_len=MAX_LEN,
    )

    model = Transformer(cfg).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    losses = []
    for step in range(TRAIN_STEPS):
        logits, loss = model(batch, batch)
        assert_ok(logits.shape == (BATCH_SIZE, MAX_LEN, cfg.vocab_size), "forward shape")
        assert_ok(math.isfinite(loss.item()), "finite loss", f"step={step+1} loss={loss.item():.4f}")
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        log(f"step {step+1}/{TRAIN_STEPS} loss={loss.item():.4f}")

    prompt = "<|user|>Write a Python function to add two numbers.<|assistant|>"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    x = torch.tensor([prompt_ids[-MAX_LEN:]], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        out = model.generate(
            x,
            max_new_tokens=32,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0][x.shape[1]:].tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    assert_ok(len(text.strip()) > 0, "generation non-empty")

    log("-" * 60)
    log("Smoke test complete.")
    log(f"Losses: {[round(x, 4) for x in losses]}")
    log(f"Generated: {text[:160]!r}")
    log("=" * 60)


if __name__ == "__main__":
    # Optional auth for private resources in Colab:
    #   os.environ["HF_TOKEN"] = "hf_..."
    token = os.getenv("HF_TOKEN")
    if token:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
    main()
