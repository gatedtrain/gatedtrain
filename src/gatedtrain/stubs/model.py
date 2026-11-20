"""skeleton-stub model: a tiny character-level causal transformer (CPU only).

MVP-SPEC section 3 "Model": a few hundred thousand parameters. The
checkpoint directory holds `model.pt` (a torch state dict) and
`model_config.json`. torch.save output is byte-deterministic for equal
tensors, so equal weights give equal manifest hashes (determinism check,
MVP-SPEC section 5).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import task

STUB = "skeleton-stub"
MODEL_FILE = "model.pt"
CONFIG_FILE = "model_config.json"

DEFAULT_CONFIG = {
    "kind": "char_transformer",
    "stub": STUB,
    "vocab_size": len(task.VOCAB),
    "seq_len": task.SEQ_LEN,
    "d_model": 96,
    "n_layers": 4,
    "n_heads": 4,
    "d_ff": 384,
}


def configure_torch() -> None:
    """Single-threaded deterministic CPU kernels: same seed, same bytes, on any CPU. oneDNN is off: it JIT-compiles
    kernels for the CPU's instruction set (GELU ran as `eltwise jit:avx2` on one host and avx512 on another), which
    ATEN_CPU_CAPABILITY does not reach (gatedtrain/stubs/__init__.py pins ATen and MKL)."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=-1)
        h = self.n_heads
        q = q.view(b, t, h, d // h).transpose(1, 2)
        k = k.view(b, t, h, d // h).transpose(1, 2)
        v = v.view(b, t, h, d // h).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        return x + self.ff2(F.gelu(self.ff1(self.ln2(x))))


class CharTransformer(nn.Module):
    """skeleton-stub causal transformer over the task's character vocabulary."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = dict(cfg)
        d = cfg["d_model"]
        self.tok = nn.Embedding(cfg["vocab_size"], d)
        self.pos = nn.Embedding(cfg["seq_len"], d)
        self.blocks = nn.ModuleList(Block(d, cfg["n_heads"], cfg["d_ff"]) for _ in range(cfg["n_layers"]))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg["vocab_size"])
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        for blk in self.blocks:
            nn.init.normal_(blk.proj.weight, std=0.02 / math.sqrt(2 * cfg["n_layers"]))
            nn.init.normal_(blk.ff2.weight, std=0.02 / math.sqrt(2 * cfg["n_layers"]))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        t = ids.shape[1]
        x = self.tok(ids) + self.pos(torch.arange(t))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln(x))


def build(seed: int, cfg: dict | None = None) -> CharTransformer:
    torch.manual_seed(seed)
    return CharTransformer(cfg or DEFAULT_CONFIG)


def save(model: CharTransformer, directory: Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().clone() for k, v in model.state_dict().items()}, directory / MODEL_FILE)
    (directory / CONFIG_FILE).write_text(json.dumps(model.cfg, sort_keys=True, indent=1) + "\n", encoding="utf-8")


def load(directory: Path) -> CharTransformer:
    directory = Path(directory)
    cfg = json.loads((directory / CONFIG_FILE).read_text(encoding="utf-8"))
    if cfg.get("kind") != "char_transformer":
        raise ValueError(f"{directory}: not a skeleton-stub char_transformer checkpoint")
    model = CharTransformer(cfg)
    model.load_state_dict(torch.load(directory / MODEL_FILE, weights_only=True))
    return model


def count_parameters(checkpoint_dir: Path) -> int:
    """Number of trainable parameters of the checkpoint (tests/README.md section 7.1)."""
    return sum(p.numel() for p in load(checkpoint_dir).parameters())
