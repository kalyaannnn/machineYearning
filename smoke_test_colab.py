# ============================================================
# smoke_test_colab.py
# End-to-end smoke test — run this on Colab A100 BEFORE
# launching full pretraining.
#
# Tests:
#   1. Environment  — torch, cuda, flash attn
#   2. Tokenizer    — load from codeMath, roundtrip
#   3. Dataset      — load pretrain/sft/grpo shards, check shapes
#   4. Model        — init, forward, backward, param count
#   5. DataLoader   — real batch from HF dataset through model
#   6. Training step — loss goes down over 10 real steps
#   7. Throughput   — measure real tok/s, project pretrain ETA
#   8. Generation   — model generates tokens from a real prompt
#
# Usage (Colab cell):
#   !python smoke_test_colab.py
# ============================================================

# ============================================================
# CELL 1 — Install
# ============================================================
# !pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121 -q
# !pip install flash-attn --no-build-isolation -q
# !pip install transformers datasets tokenizers==0.15.2 huggingface_hub wandb -q

# ============================================================
# CELL 2 — Imports
# ============================================================
import os, sys, math, time, shutil, gc
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from huggingface_hub import hf_hub_download, login

# add model/ to path if running from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in dir() else "/content")

from google.colab import userdata
login(token=userdata.get("HF_TOKEN"))
REPO = "raokalyaan/codeMath"

PASS = "  [PASS]"
FAIL = "  [FAIL]"
results = []

def check(name, passed, msg=""):
    status = PASS if passed else FAIL
    line = f"{status}  {name}" + (f"  —  {msg}" if msg else "")
    print(line)
    results.append((name, passed, msg))
    if not passed:
        raise AssertionError(f"FAILED: {name}  —  {msg}")

print("\n" + "=" * 65)
print("  Colab Smoke Test — 270M LLM Pipeline")
print("=" * 65)

# ============================================================
# SECTION 1 — Environment
# ============================================================
print("\n[1] Environment")

check("CUDA available", torch.cuda.is_available(),
      "No GPU — make sure Colab runtime is set to A100")

gpu_name = torch.cuda.get_device_name(0)
vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
check("GPU is A100 / sufficient VRAM",
      vram_gb >= 30,
      f"{gpu_name}  {vram_gb:.1f} GB")

try:
    from flash_attn import flash_attn_func
    check("Flash Attention 2 installed", True)
except ImportError:
    check("Flash Attention 2 installed", False,
          "run: pip install flash-attn --no-build-isolation")

print(f"  torch version: {torch.__version__}")

# ============================================================
# SECTION 2 — Tokenizer
# ============================================================
print("\n[2] Tokenizer")

os.makedirs("/tmp/tokenizer", exist_ok=True)
for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
    p = hf_hub_download(
        repo_id=REPO, repo_type="dataset",
        filename=f"tokenizer/{fname}",
        local_dir="/tmp/tok_dl",
        
    )
    shutil.copy(p, f"/tmp/tokenizer/{fname}")

tokenizer = PreTrainedTokenizerFast.from_pretrained("/tmp/tokenizer")

check("tokenizer loads",         True, f"vocab_size={tokenizer.vocab_size}")
check("vocab_size == 32000",     tokenizer.vocab_size == 32000)
check("<|user|> in vocab",       "<|user|>"      in tokenizer.get_vocab())
check("<|assistant|> in vocab",  "<|assistant|>" in tokenizer.get_vocab())
check("<|eos|> in vocab",        "<|eos|>"        in tokenizer.get_vocab())

# roundtrip
test_code = "def fibonacci(n):\n    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
test_math = "The integral of x^2 dx = x^3/3 + C"
for txt in [test_code, test_math]:
    ids  = tokenizer.encode(txt, add_special_tokens=False)
    back = tokenizer.decode(ids)
    check(f"roundtrip: '{txt[:30]}...'", txt in back, f"{len(ids)} tokens")

# chat template
prompt = f"<|user|>What is 2+2?<|assistant|>The answer is 4.<|eos|>"
ids = tokenizer.encode(prompt, add_special_tokens=False)
check("chat template encodes", len(ids) > 5, f"{len(ids)} tokens")

PAD_ID = tokenizer.pad_token_id
EOS_ID = tokenizer.eos_token_id
print(f"  pad_id={PAD_ID}  eos_id={EOS_ID}")

# ============================================================
# SECTION 3 — Dataset
# ============================================================
print("\n[3] Dataset  (raokalyaan/codeMath)")

# --- Pretrain ---
print("  Loading pretrain_train shard...")
pt_ds = load_dataset(
    REPO,
    data_files={"train": "pretrain_train/starcoder_py_000.parquet"},
    split="train",
)
pt_sample = pt_ds[0]

check("pretrain shard loads",            True,               f"{len(pt_ds):,} rows")
check("pretrain has input_ids",          "input_ids" in pt_sample)
check("input_ids length == 4096",        len(pt_sample["input_ids"]) == 4096,
      f"got {len(pt_sample['input_ids'])}")
check("no -100 in pretrain",             -100 not in pt_sample["input_ids"])
check("token ids in range [0, 31999]",
      0 <= min(pt_sample["input_ids"]) and max(pt_sample["input_ids"]) < 32000,
      f"min={min(pt_sample['input_ids'])}, max={max(pt_sample['input_ids'])}")
del pt_ds; gc.collect()

# --- SFT ---
print("  Loading sft_train shard...")
sft_ds = load_dataset(
    REPO,
    data_files={"train": "sft_train/magicoder_000.parquet"},
    split="train",
)
sft_sample = sft_ds[0]

check("sft shard loads",                True,  f"{len(sft_ds):,} rows")
check("sft has input_ids + labels",     "input_ids" in sft_sample and "labels" in sft_sample)
check("sft input_ids length == 4096",   len(sft_sample["input_ids"]) == 4096)
check("sft labels length == 4096",      len(sft_sample["labels"])    == 4096)
n_masked   = sum(1 for l in sft_sample["labels"] if l == -100)
n_unmasked = sum(1 for l in sft_sample["labels"] if l != -100)
check("sft has masked prompt tokens",   n_masked   > 0, f"{n_masked} masked")
check("sft has unmasked completion",    n_unmasked > 0, f"{n_unmasked} unmasked")
mismatches = sum(1 for i, l in zip(sft_sample["input_ids"], sft_sample["labels"])
                 if l != -100 and i != l)
check("input_ids and labels agree",     mismatches == 0, f"{mismatches} mismatches")
del sft_ds; gc.collect()

# --- GRPO ---
print("  Loading grpo_train shard...")
grpo_ds = load_dataset(
    REPO,
    data_files={"train": "grpo_train/gsm8k_train.parquet"},
    split="train",
)
grpo_sample = grpo_ds[0]
check("grpo shard loads",               True, f"{len(grpo_ds):,} rows")
check("grpo has prompt + answer",       "prompt" in grpo_sample and "answer" in grpo_sample)
check("prompt has <|user|>",            "<|user|>" in grpo_sample["prompt"])
check("prompt has <|assistant|>",       "<|assistant|>" in grpo_sample["prompt"])
check("answer is numeric",
      grpo_sample["answer"].replace(".", "").replace("-", "").isdigit(),
      f"answer='{grpo_sample['answer']}'")
del grpo_ds; gc.collect()

# ============================================================
# SECTION 4 — Model Init
# ============================================================
print("\n[4] Model")

from model import Transformer, ModelConfig

config = ModelConfig()
model  = Transformer(config).to("cuda")
params = model.get_num_params()

check("model initialises",           True)
check("~268M total params",          240e6 < params["total"] < 300e6,
      f"{params['total']/1e6:.1f}M")
check("~236M non-embedding params",  200e6 < params["non_embedding"] < 270e6,
      f"{params['non_embedding']/1e6:.1f}M")
check("embeddings tied",             config.tie_embeddings)

# forward pass with random input
B, T = 2, 512
x = torch.randint(0, config.vocab_size, (B, T), device="cuda")
y = torch.randint(0, config.vocab_size, (B, T), device="cuda")
logits, loss = model(x, y)

expected = math.log(config.vocab_size)
check("forward pass runs",           True)
check("logits shape correct",        list(logits.shape) == [B, T, config.vocab_size],
      str(list(logits.shape)))
check("loss is finite",              loss.item() == loss.item(),  # NaN check
      f"{loss.item():.4f}")
check("loss near log(vocab_size)",   abs(loss.item() - expected) < 2.0,
      f"{loss.item():.4f} vs expected {expected:.4f}")

# backward pass
loss.backward()
grads = {n: p.grad.norm().item() for n, p in model.named_parameters() if p.grad is not None}
has_nan = any(math.isnan(v) for v in grads.values())
check("backward pass runs",          True)
check("no NaN gradients",            not has_nan, f"{len(grads)} params have gradients")
model.zero_grad()

torch.cuda.empty_cache(); gc.collect()

# ============================================================
# SECTION 5 — DataLoader Integration
# ============================================================
print("\n[5] DataLoader  (real data through model)")

# load a small slice of pretrain data
slice_ds = load_dataset(
    REPO,
    data_files={"train": "pretrain_train/finemath_000.parquet"},
    split="train[:100]",    # first 100 rows only — fast
)
slice_ds.set_format("torch", columns=["input_ids"])

loader = DataLoader(
    slice_ds,
    batch_size=4,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

batch = next(iter(loader))
input_ids = batch["input_ids"].to("cuda")

check("dataloader yields batches",   True, f"shape={list(input_ids.shape)}")
check("batch shape correct",
      list(input_ids.shape) == [4, 4096],
      str(list(input_ids.shape)))

with torch.no_grad():
    _, loss_real = model(input_ids, input_ids)
check("forward on real data",        True, f"loss={loss_real.item():.4f}")
check("real data loss is finite",    math.isfinite(loss_real.item()),
      f"{loss_real.item():.4f}")

del slice_ds, loader; gc.collect()

# ============================================================
# SECTION 6 — Training Step (loss decreases)
# ============================================================
print("\n[6] Training step  (10 steps, loss should decrease)")

optimizer = torch.optim.AdamW(
    model.parameters(), lr=3e-4,
    betas=(0.9, 0.95), weight_decay=0.1, fused=True,
)

# load a fresh small slice for this test
step_ds = load_dataset(
    REPO,
    data_files={"train": "pretrain_train/finemath_000.parquet"},
    split="train[:200]",
)
step_ds.set_format("torch", columns=["input_ids"])
step_loader = DataLoader(step_ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
step_iter   = iter(step_loader)

model.train()
losses = []
for i in range(10):
    batch     = next(step_iter)
    input_ids = batch["input_ids"].to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = model(input_ids, input_ids)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    losses.append(loss.item())
    print(f"    step {i+1:2d}  loss={loss.item():.4f}")

check("10 training steps complete",  True)
check("loss is finite throughout",   all(math.isfinite(l) for l in losses),
      f"losses={[f'{l:.3f}' for l in losses]}")
check("loss decreased over 10 steps",
      losses[-1] < losses[0],
      f"first={losses[0]:.4f}  last={losses[-1]:.4f}")

del step_ds, step_loader; gc.collect(); torch.cuda.empty_cache()

# ============================================================
# SECTION 7 — Throughput
# ============================================================
print("\n[7] Throughput  (real training speed)")

# Use torch.compile for accurate real-world measurement
print("  Compiling model (takes ~60s first time)...")
model_compiled = torch.compile(model)

B_bench, T_bench = 8, 4096
x_b = torch.randint(0, config.vocab_size, (B_bench, T_bench), device="cuda")
y_b = torch.randint(0, config.vocab_size, (B_bench, T_bench), device="cuda")

# warmup — let compile finish and CUDA graphs warm up
for _ in range(3):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, lb = model_compiled(x_b, y_b)
    lb.backward()
    optimizer.step()
    optimizer.zero_grad()
torch.cuda.synchronize()

N = 10
t0 = time.time()
for _ in range(N):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, lb = model_compiled(x_b, y_b)
    lb.backward()
    optimizer.step()
    optimizer.zero_grad()
torch.cuda.synchronize()
elapsed = time.time() - t0

tok_per_sec = (N * B_bench * T_bench) / elapsed
eta_hrs     = (3_500_000_000 / tok_per_sec) / 3600

print(f"  tok/s:         {tok_per_sec:>10,.0f}")
print(f"  pretrain ETA:  {eta_hrs:.1f} hours  (3.5B tokens)")

check("throughput > 80k tok/s",
      tok_per_sec > 80_000,
      f"{tok_per_sec:,.0f} tok/s  —  if failing, Flash Attn not working")

# VRAM check
torch.cuda.empty_cache()
vram_used = torch.cuda.memory_reserved() / 1e9
print(f"  VRAM used:     {vram_used:.1f} GB  /  80.0 GB")
check("VRAM under 40GB  (headroom for full batch)",
      vram_used < 40,
      f"{vram_used:.1f} GB used")

gc.collect()

# ============================================================
# SECTION 8 — Generation
# ============================================================
print("\n[8] Generation  (real prompt from dataset)")

# load one real GRPO prompt
gen_ds = load_dataset(
    REPO,
    data_files={"train": "grpo_train/gsm8k_train.parquet"},
    split="train[:1]",
)
prompt_text = gen_ds[0]["prompt"]
answer_text = gen_ds[0]["answer"]
print(f"  Prompt: {prompt_text[:80]}...")
print(f"  Answer: {answer_text}")

prompt_ids = torch.tensor(
    [tokenizer.encode(prompt_text, add_special_tokens=False)],
    device="cuda",
)

model_compiled.eval()
with torch.no_grad():
    out = model_compiled.generate(
        prompt_ids,
        max_new_tokens=64,
        temperature=0.8,
        top_p=0.95,
        eos_token_id=EOS_ID,
        do_sample=True,
    )

completion = tokenizer.decode(out[0][prompt_ids.shape[1]:], skip_special_tokens=True)
print(f"  Completion (first 100 chars): {completion[:100]}...")

check("generation runs",              True, f"{out.shape[1] - prompt_ids.shape[1]} new tokens")
check("completion is non-empty",      len(completion.strip()) > 0)

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 65)
passed = sum(1 for _, p, _ in results if p)
failed = sum(1 for _, p, _ in results if not p)
total  = len(results)
print(f"  {passed} passed  |  {failed} failed  |  {total} total checks")

if failed == 0:
    print("\n  All checks passed.")
    print("  Ready to launch pretraining.")
    print(f"\n  Projected pretraining:")
    print(f"    Throughput:   {tok_per_sec:,.0f} tok/s")
    print(f"    Total tokens: 3.5B")
    print(f"    ETA:          {eta_hrs:.1f} hours")
else:
    print("\n  Failed checks:")
    for name, p, msg in results:
        if not p:
            print(f"    [FAIL]  {name}  —  {msg}")
print("=" * 65)