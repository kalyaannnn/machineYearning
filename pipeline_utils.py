import json
import os
import random
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerFast


PROMPT_TEMPLATE = "<|user|>{instruction}<|assistant|>"
TOKENIZER_CACHE_DIR = "./artifacts/tokenizer_cache"

FIXED_PROMPT_BATTERY = [
    {
        "id": "code_fn",
        "instruction": "Write a Python function named add_two that returns the sum of two integers.",
        "expected_format": "code",
        "must_include": ["def", "return"],
    },
    {
        "id": "math_steps",
        "instruction": "Solve 27 * 14 and show your reasoning in numbered steps.",
        "expected_format": "steps",
        "must_include": ["378"],
    },
    {
        "id": "json_struct",
        "instruction": "Return valid JSON with keys topic and summary about gradient descent.",
        "expected_format": "json",
        "must_include": ["topic", "summary"],
    },
    {
        "id": "bullet_list",
        "instruction": "Give three concise tips for debugging CUDA out-of-memory errors as bullet points.",
        "expected_format": "list",
        "must_include": ["memory", "batch", "checkpoint"],
    },
]


STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "your", "about",
    "show", "give", "return", "three", "named", "write", "valid", "json",
}


@dataclass
class StagePaths:
    stage: str
    run_id: str
    checkpoint_dir: str
    metrics_json: str
    metrics_json_run: str
    samples_jsonl: str


def make_run_id(prefix: str = "run") -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_stage_paths(stage: str, run_id: str) -> StagePaths:
    ckpt_dir = Path("checkpoints") / stage / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    Path("metrics").mkdir(parents=True, exist_ok=True)
    Path("samples").mkdir(parents=True, exist_ok=True)

    return StagePaths(
        stage=stage,
        run_id=run_id,
        checkpoint_dir=str(ckpt_dir),
        metrics_json=f"metrics/{stage}_metrics.json",
        metrics_json_run=f"metrics/{stage}_metrics_{run_id}.json",
        samples_jsonl=f"samples/{stage}_prompts_and_outputs.jsonl",
    )


def write_json(path: str, payload: Dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def append_jsonl(path: str, record: Dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_codemath_tokenizer(repo_id: str, local_dir: str = TOKENIZER_CACHE_DIR) -> PreTrainedTokenizerFast:
    os.makedirs(local_dir, exist_ok=True)
    filenames = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
    for filename in filenames:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=f"tokenizer/{filename}",
        )
        shutil.copyfile(downloaded_path, os.path.join(local_dir, filename))
    return PreTrainedTokenizerFast.from_pretrained(local_dir)


def decode_generated_text(tokenizer: PreTrainedTokenizerFast, token_ids: List[int]) -> Dict[str, str]:
    clean = tokenizer.decode(token_ids, skip_special_tokens=True)
    raw = tokenizer.decode(token_ids, skip_special_tokens=False)
    return {"clean": clean, "raw": raw}


def extract_numeric_answer(text: str) -> str:
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else ""


def _score_adherence(output: str) -> float:
    text = output.strip()
    if not text:
        return 0.0
    if text.count("<|assistant|>") > 4:
        return 0.0
    return 1.0 if len(text) >= 12 else 0.4


def _score_format(expected_format: str, output: str) -> float:
    text = output.strip()
    if expected_format == "json":
        return 1.0 if text.startswith("{") and text.endswith("}") else 0.0
    if expected_format == "code":
        return 1.0 if ("def " in text or "```" in text) else 0.0
    if expected_format == "steps":
        return 1.0 if ("1." in text and "2." in text) else 0.3
    if expected_format == "list":
        bullet_count = text.count("- ") + text.count("* ")
        return 1.0 if bullet_count >= 3 else 0.3
    return 0.5


def _score_relevance(prompt: str, output: str) -> float:
    p_words = {
        w for w in re.findall(r"[a-zA-Z]{4,}", prompt.lower())
        if w not in STOPWORDS
    }
    if not p_words:
        return 0.5
    out_words = set(re.findall(r"[a-zA-Z]{4,}", output.lower()))
    overlap = len(p_words & out_words)
    ratio = overlap / max(1, len(p_words))
    return max(0.0, min(1.0, ratio * 2.0))


def score_prompt_output(prompt_spec: Dict, output: str) -> Dict[str, float]:
    adherence = _score_adherence(output)
    fmt = _score_format(prompt_spec.get("expected_format", ""), output)
    relevance = _score_relevance(prompt_spec["instruction"], output)

    must_include = prompt_spec.get("must_include", [])
    include_hits = 0
    low = output.lower()
    for key in must_include:
        include_hits += 1 if key.lower() in low else 0
    include_score = include_hits / max(1, len(must_include)) if must_include else 1.0

    coherence = (adherence + fmt + relevance + include_score) / 4.0
    return {
        "adherence": adherence,
        "format": fmt,
        "relevance": relevance,
        "must_include": include_score,
        "coherence": coherence,
    }


def promotion_gate(
    pretrain_metrics: Dict[str, float],
    sft_metrics: Dict[str, float],
    min_coherence_gain: float = 0.03,
    max_code_regression: float = 0.01,
    max_math_regression: float = 0.01,
) -> Dict:
    coherence_gain = sft_metrics["coherence_score"] - pretrain_metrics["coherence_score"]
    code_delta = sft_metrics["code_token_acc"] - pretrain_metrics["code_token_acc"]
    math_delta = sft_metrics["math_exact_match"] - pretrain_metrics["math_exact_match"]

    checks = {
        "coherence_improved": coherence_gain >= min_coherence_gain,
        "code_not_regressed": code_delta >= -max_code_regression,
        "math_not_regressed": math_delta >= -max_math_regression,
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "coherence_gain": coherence_gain,
        "code_delta": code_delta,
        "math_delta": math_delta,
        "thresholds": {
            "min_coherence_gain": min_coherence_gain,
            "max_code_regression": max_code_regression,
            "max_math_regression": max_math_regression,
        },
    }
