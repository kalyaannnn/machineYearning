

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import ModelConfig
from .norm import RMSNorm
from .block import TransformerBlock
from .rope import precompute_rope_freqs


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.head.weight = self.embed.weight

        freqs_cis = precompute_rope_freqs(
            head_dim=config.head_dim,
            seq_len=config.seq_len,
            theta=config.rope_theta,
        )
        self.register_buffer("freqs_cis", freqs_cis)

        self._init_weights()

        total = sum(p.numel() for p in self.parameters())
        print(f"Transformer initialised — {total / 1e6:.1f}M parameters")


    def _init_weights(self):
        """
        Standard LLM weight init:
          • Embeddings + all Linear layers: N(0, 0.02)
          • Residual output projections (wo, w2): scaled down by
            1/sqrt(2 * n_layers) to keep residual stream variance
            stable at initialisation regardless of depth.
        """
        std = 0.02
        residual_std = std / math.sqrt(2 * self.config.n_layers)

        nn.init.normal_(self.embed.weight, mean=0.0, std=std)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for block in self.layers:
            nn.init.normal_(block.attn.wo.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.ffn.w2.weight,  mean=0.0, std=residual_std)


    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            input_ids: [B, T]  — token indices
            targets:   [B, T]  — shifted input_ids for LM loss
                                 pass None during inference
                                 use -100 to mask SFT prompt positions

        Returns:
            (logits, loss)
              logits: [B, T, vocab_size]  during training
                      [B, 1, vocab_size]  during inference (last position only)
              loss:   scalar cross-entropy, or None if targets=None
        """
        B, T = input_ids.shape
        assert T <= self.config.seq_len, (
            f"Input length {T} exceeds max sequence length {self.config.seq_len}"
        )

        # embed tokens
        x = self.embed(input_ids)              # [B, T, d_model]

        # get precomputed RoPE freqs for this sequence length
        freqs_cis = self.freqs_cis[:T]         # [T, head_dim // 2]

        # forward through all blocks
        for layer in self.layers:
            x = layer(x, freqs_cis)

        # final norm
        x = self.norm(x)                       # [B, T, d_model]

        if targets is not None:
            # ── Training ──────────────────────────────────────────────────
            logits = self.head(x)              # [B, T, vocab_size]
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100,             # masks SFT prompt tokens
            )
            return logits, loss
        else:
            # ── Inference ─────────────────────────────────────────────────
            # only compute logits for the last position — saves memory
            logits = self.head(x[:, [-1], :])  # [B, 1, vocab_size]
            return logits, None

    # ── Generation ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 0.95,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive token generation.

        Args:
            input_ids:      [B, T] — prompt token ids
            max_new_tokens: maximum tokens to generate
            temperature:    >1 = more random, <1 = more peaked
            top_p:          nucleus sampling — keeps top tokens summing to p
            do_sample:      False = greedy decoding
            eos_token_id:   stop when all sequences produce this token

        Returns:
            [B, T + n_generated] — prompt + generated tokens
        """
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # crop to max context if needed
            ctx = generated if generated.shape[1] <= self.config.seq_len \
                  else generated[:, -self.config.seq_len:]

            logits, _ = self(ctx)          # [B, 1, vocab_size]
            logits = logits[:, -1, :]      # [B, vocab_size]

            if not do_sample:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-5)
                next_token = _nucleus_sample(logits, top_p)

            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

    # ── Utilities ──────────────────────────────────────────────────────────

    def count_params(self, non_embedding: bool = True) -> int:
        """
        Count trainable parameters.
        non_embedding=True subtracts embedding table — standard convention
        since embedding params don't contribute to FLOPs.
        """
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            total -= self.embed.weight.numel()
        return total

    def get_num_params(self) -> dict:
        return {
            "total":         sum(p.numel() for p in self.parameters()),
            "non_embedding": self.count_params(non_embedding=True),
            "embedding":     self.embed.weight.numel(),
        }


# ── Nucleus Sampling Helper ────────────────────────────────────────────────

def _nucleus_sample(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Top-p (nucleus) sampling.

    Sort tokens by probability, keep the smallest set whose cumulative
    probability exceeds top_p, zero out the rest, then sample.

    Args:
        logits: [B, vocab_size]   — unnormalised logits (temperature already applied)
        top_p:  float in (0, 1]

    Returns:
        next_token: [B, 1]
    """
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = sorted_probs.cumsum(dim=-1)

    # remove tokens once cumulative prob exceeds top_p
    # shift right by one so we always keep at least the top token
    remove_mask = (cumulative_probs - sorted_probs) > top_p
    sorted_probs[remove_mask] = 0.0
    sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True))  # renormalise

    # sample from the nucleus
    sampled = torch.multinomial(sorted_probs, num_samples=1)  # [B, 1] — index into sorted
    next_token = sorted_idx.gather(dim=-1, index=sampled)     # [B, 1] — actual token id
    return next_token