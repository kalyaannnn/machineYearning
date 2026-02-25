
from dataclasses import dataclass, field, asdict
import json


@dataclass
class ModelConfig:
    # ── Dimensions ─────────────────────────────────────────────────────────
    vocab_size:  int   = 32000
    d_model:     int   = 1024    # hidden / residual stream dimension
    n_layers:    int   = 24      # number of transformer blocks
    n_heads:     int   = 16      # query attention heads
    n_kv_heads:  int   = 8       # key/value heads  (GQA: n_heads // n_kv_heads = 2 groups)
    ffn_dim:     int   = 4096    # SwiGLU intermediate dimension
    seq_len:     int   = 4096    # maximum context length

    # ── Regularisation ─────────────────────────────────────────────────────
    dropout:     float = 0.0     # 0 during pretraining; optionally 0.1 for SFT

    # ── Norm / init ────────────────────────────────────────────────────────
    norm_eps:    float = 1e-5
    rope_theta:  float = 10000.0 # RoPE base frequency

    # ── Misc ───────────────────────────────────────────────────────────────
    tie_embeddings: bool = True   # tie input embed ↔ output projection

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads

    @property
    def n_groups(self) -> int:
        """How many query heads share each KV head."""
        assert self.n_heads % self.n_kv_heads == 0
        return self.n_heads // self.n_kv_heads

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Config saved → {path}")

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def __repr__(self) -> str:
        lines = ["ModelConfig("]
        for k, v in self.to_dict().items():
            lines.append(f"  {k}={v!r},")
        lines.append(")")
        return "\n".join(lines)
