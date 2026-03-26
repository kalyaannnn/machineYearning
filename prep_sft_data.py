"""
prep_sft_data.py — Build a packed SFT dataset from OpenHermes-2.5.

Downloads OpenHermes-2.5, formats conversations into your chat template,
tokenizes with the codeMath tokenizer, creates prompt-masked labels,
packs sequences to SEQ_LEN=4096, and saves train/val parquet files.

Usage:
    python prep_sft_data.py
    python prep_sft_data.py --max-samples 30000 --output-dir ./data/sft_openhermes

Output (used directly by sft.py --local-data-dir):
    {output_dir}/sft_train.parquet
    {output_dir}/sft_val.parquet

Each row has:
    input_ids: List[int]  length == SEQ_LEN
    labels:    List[int]  -100 for prompt tokens, token id for completion tokens
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_utils import load_codemath_tokenizer

REPO        = "raokalyaan/codeMath"
SEQ_LEN     = 4096
VAL_SPLIT   = 0.02
SEED        = 42


def format_conversation(convo: list, tokenizer: PreTrainedTokenizerFast) -> Tuple[List[int], List[int]]:
    """
    Convert a list of conversation turns into token ids + labels.

    Format per turn:
        human:  <|user|>{text}
        gpt:    <|assistant|>{text}<|eos|>

    Labels:
        -100 for all human/user tokens (prompt masking)
        token id for all assistant tokens (train on completions only)

    Returns:
        (input_ids, labels) — variable length, not yet packed
    """
    input_ids = []
    labels    = []

    user_tok      = tokenizer.encode("<|user|>",      add_special_tokens=False)
    assistant_tok = tokenizer.encode("<|assistant|>", add_special_tokens=False)
    eos_tok       = tokenizer.encode("<|eos|>",       add_special_tokens=False)

    for turn in convo:
        role = turn.get("from", "").lower()
        text = turn.get("value", "").strip()
        if not text:
            continue

        if role in ("human", "user"):
            toks = user_tok + tokenizer.encode(text, add_special_tokens=False)
            input_ids.extend(toks)
            labels.extend([-100] * len(toks))

        elif role in ("gpt", "assistant"):
            prefix = assistant_tok
            body   = tokenizer.encode(text, add_special_tokens=False)
            suffix = eos_tok
            toks   = prefix + body + suffix
            input_ids.extend(toks)
            # mask the <|assistant|> prefix token(s) too, only train on the response body
            labels.extend([-100] * len(prefix) + body + suffix)

    return input_ids, labels


def pack_sequences(
    all_ids:    List[List[int]],
    all_labels: List[List[int]],
    seq_len:    int,
    pad_id:     int,
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Pack variable-length tokenized conversations into fixed seq_len chunks.
    Conversations are concatenated end-to-end; chunks never split a token.
    Leftover positions in the final chunk are padded with pad_id / -100.
    """
    packed_ids    = []
    packed_labels = []

    buf_ids    = []
    buf_labels = []

    for ids, labs in zip(all_ids, all_labels):
        buf_ids.extend(ids)
        buf_labels.extend(labs)

        while len(buf_ids) >= seq_len:
            packed_ids.append(buf_ids[:seq_len])
            packed_labels.append(buf_labels[:seq_len])
            buf_ids    = buf_ids[seq_len:]
            buf_labels = buf_labels[seq_len:]

    # flush partial last chunk with padding
    if buf_ids:
        pad_len = seq_len - len(buf_ids)
        packed_ids.append(buf_ids    + [pad_id]  * pad_len)
        packed_labels.append(buf_labels + [-100] * pad_len)

    return packed_ids, packed_labels


def build(args):
    random.seed(SEED)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = load_codemath_tokenizer(REPO)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    print(f"  vocab_size={tokenizer.vocab_size}  pad_id={pad_id}")

    print(f"\nLoading OpenHermes-2.5 (max_samples={args.max_samples})...")
    ds = load_dataset("teknium/OpenHermes-2.5", split="train")
    print(f"  Total rows: {len(ds):,}")

    if args.max_samples and args.max_samples < len(ds):
        indices = random.sample(range(len(ds)), args.max_samples)
        ds = ds.select(indices)
        print(f"  Sampled:    {len(ds):,}")

    print("\nTokenizing conversations...")
    all_ids    = []
    all_labels = []
    skipped    = 0

    for row in ds:
        convo = row.get("conversations", [])
        if not convo:
            skipped += 1
            continue
        ids, labs = format_conversation(convo, tokenizer)
        # skip empty or prompt-only examples
        if not ids or all(l == -100 for l in labs):
            skipped += 1
            continue
        all_ids.append(ids)
        all_labels.append(labs)

    print(f"  Tokenized: {len(all_ids):,}  skipped: {skipped}")

    # shuffle before packing so conversations mix across chunks
    paired = list(zip(all_ids, all_labels))
    random.shuffle(paired)
    all_ids, all_labels = zip(*paired)

    print(f"\nPacking to seq_len={SEQ_LEN}...")
    packed_ids, packed_labels = pack_sequences(list(all_ids), list(all_labels), SEQ_LEN, pad_id)
    print(f"  Total packed chunks: {len(packed_ids):,}")

    # verify label diversity — catch memorization risk early
    label_tokens = [l for labs in packed_labels for l in labs if l != -100]
    unique_labels = len(set(label_tokens))
    print(f"  Unique non-masked label tokens: {unique_labels:,}  (should be >> 1000)")
    if unique_labels < 1000:
        print("  WARNING: very few unique label tokens — data may be trivially easy to memorize")

    # train / val split
    n_val   = max(1, int(len(packed_ids) * VAL_SPLIT))
    n_train = len(packed_ids) - n_val
    train_ids,    val_ids    = packed_ids[:n_train],    packed_ids[n_val:]
    train_labels, val_labels = packed_labels[:n_train], packed_labels[n_val:]

    print(f"\nSplit: train={n_train:,}  val={n_val:,}")

    def save_parquet(ids, labels, path):
        table = pa.table({
            "input_ids": pa.array(ids,    type=pa.list_(pa.int32())),
            "labels":    pa.array(labels, type=pa.list_(pa.int32())),
        })
        pq.write_table(table, path)
        print(f"  Saved {len(ids):,} rows → {path}")

    save_parquet(train_ids,    train_labels,  output_dir / "sft_train.parquet")
    save_parquet(val_ids,      val_labels,    output_dir / "sft_val.parquet")

    print(f"\nDone. Load with:")
    print(f"  python sft.py --pretrain-ckpt <ckpt> --local-data-dir {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples",  type=int,  default=50_000,
                        help="Max conversations to use (default 50k, full dataset is ~1M)")
    parser.add_argument("--output-dir",   type=str,  default="./data/sft_openhermes",
                        help="Where to write sft_train.parquet and sft_val.parquet")
    args = parser.parse_args()
    build(args)
