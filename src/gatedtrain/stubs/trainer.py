"""skeleton-stub trainer backend `cpu_stub`: supervised fine-tuning on CPU.

MVP-SPEC section 3 "Trainer backend cpu_stub" and tests/README.md section 6.1.
The outputs have the names and shapes NeMo RL writes at 4f39fd8, so the
ledger writer reads them as it would read NeMo RL's:

- `logs/train_data_step<N>.jsonl`, one JSON object per consumed sample, with
  keys from the `log_data` set of nemo_rl/algorithms/grpo_sync.py:1429-1456
  plus `idx` added by Logger.log_batched_dict_as_jsonl
  (nemo_rl/utils/logger.py:1427-1436). There is no dataset row ID and no
  shard field, as in NeMo RL;
- `checkpoints/step_<N>/training_info.json` holding `consumed_samples`
  (grpo_sync.py:1323, :1353; written by nemo_rl/utils/checkpoint.py:406-412);
- `checkpoints/step_<N>/train_dataloader.pt`, the data-loader state_dict
  (grpo_sync.py:1407-1414; nemo_rl/data/utils.py:69-82).

The loader reads shards in file-name order and rows in file order, without
shuffling, with NeMo RL's drop_last epoch semantics (decision D-1;
tests/README.md section 10.1). The SFT stand-in has no reward; `rewards`
is written as 0.0 per sample only to keep NeMo RL's required key.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from . import model as model_mod
from . import task, write_marker

STUB = "skeleton-stub"

TRAINER_KEYS = {"backend", "seed", "lr", "batch_size", "max_steps", "save_period"}
WARMUP_KEYS = {"backend", "data", "seed", "lr", "batch_size", "max_steps"}


class ConfigError(ValueError):
    """A stand-in config error: (field, message, fix)."""

    def __init__(self, field: str, message: str, fix: str) -> None:
        super().__init__(f"{field}: {message}")
        self.field, self.message, self.fix = field, message, fix


def load_config(path: Path, *, kind: str) -> dict:
    """Read and check a cpu_stub trainer (`kind="trainer"`) or warm-up config."""
    allowed = TRAINER_KEYS if kind == "trainer" else WARMUP_KEYS
    try:
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(kind, f"cannot read {path}: {exc}", "point the spec at a readable YAML file") from exc
    if not isinstance(cfg, dict):
        raise ConfigError(kind, "the config is not a mapping", "write the config as a YAML mapping")
    for key in sorted(set(cfg) - allowed):
        raise ConfigError(f"{kind}.{key}", "unknown cpu_stub config key", f"use only {sorted(allowed)}")
    for key in sorted(allowed - {"save_period"}):
        if key not in cfg:
            raise ConfigError(f"{kind}.{key}", "missing cpu_stub config key", f"add `{key}`")
    if cfg["backend"] != "cpu_stub":
        raise ConfigError(f"{kind}.backend", f"backend {cfg['backend']!r} is not cpu_stub", "set `backend: cpu_stub`")
    for key in ("seed", "batch_size", "max_steps") + (("save_period",) if "save_period" in cfg else ()):
        if not isinstance(cfg[key], int) or isinstance(cfg[key], bool) or cfg[key] < (0 if key == "seed" else 1):
            raise ConfigError(f"{kind}.{key}", f"{key} must be a positive integer", f"set `{key}` to an integer")
    if not isinstance(cfg["lr"], (int, float)) or isinstance(cfg["lr"], bool) or cfg["lr"] < 0:
        raise ConfigError(f"{kind}.lr", "lr must be a number >= 0", "set `lr`, for example 0.003")
    cfg["lr"] = float(cfg["lr"])
    return cfg


class SequentialShardLoader:
    """Rows of all shards in order, no shuffling (skeleton-stub).

    Epochs follow NeMo RL's `StatefulDataLoader(..., drop_last=True)`
    (nemo_rl/algorithms/grpo.py:883-890 at 4f39fd8, `drop_last=True` on line
    888; decision D-1): a batch that does not fit in the rest of the epoch is
    dropped and the next epoch restarts at row 0. Its state_dict mirrors what
    NeMo RL saves (grpo_sync.py:1407-1414): counters, no shard or row field.
    """

    def __init__(self, dataset_dir: Path, batch_size: int) -> None:
        self.rows = task.load_dataset(dataset_dir)
        if batch_size > len(self.rows):
            raise ValueError(f"batch_size {batch_size} exceeds the {len(self.rows)}-row dataset (drop_last)")
        self.batch_size = batch_size
        self.num_yielded = 0
        self.samples_yielded = 0
        self.epoch_position = 0

    def next_batch(self) -> list[dict[str, str]]:
        if self.epoch_position + self.batch_size > len(self.rows):
            self.epoch_position = 0                          # drop the tail, start a new epoch
        start = self.epoch_position
        batch = self.rows[start : start + self.batch_size]
        self.epoch_position += self.batch_size
        self.samples_yielded += self.batch_size
        self.num_yielded += 1
        return batch

    def state_dict(self) -> dict:
        return {"_num_yielded": self.num_yielded, "samples_yielded": self.samples_yielded}


def _tensors(batch: list[dict[str, str]]):
    ids, masks = zip(*(task.encode_example(r["prompt"], r["answer"]) for r in batch))
    return torch.tensor(ids, dtype=torch.long), torch.tensor(masks, dtype=torch.float32)


def _train_loop(model, loader, *, lr, max_steps, on_step=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.0)
    model.train()
    total_valid_tokens = 0
    for step in range(1, max_steps + 1):
        batch = loader.next_batch()
        ids, mask = _tensors(batch)
        logits = model(ids[:, :-1])
        target = ids[:, 1:]
        tmask = mask[:, 1:]
        loss_tok = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none")
        loss = (loss_tok * tmask.reshape(-1)).sum() / tmask.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_valid_tokens += int(tmask.sum().item())
        if on_step is not None:
            on_step(step, batch, ids, mask, total_valid_tokens)
    model.eval()
    return total_valid_tokens


def warmup(config_path: Path, out_dir: Path) -> Path:
    """Train the baseline checkpoint from scratch (skeleton-stub; MVP-SPEC section 3 "Baseline")."""
    model_mod.configure_torch()
    cfg = load_config(config_path, kind="warmup")
    data_dir = (Path(config_path).resolve().parent / cfg["data"]).resolve()
    model = model_mod.build(cfg["seed"])
    loader = SequentialShardLoader(data_dir, cfg["batch_size"])
    tokens = _train_loop(model, loader, lr=cfg["lr"], max_steps=cfg["max_steps"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    model_mod.save(model, out_dir)
    info = {"consumed_samples": loader.samples_yielded, "current_step": cfg["max_steps"],
            "total_steps": cfg["max_steps"], "current_epoch": 0, "total_valid_tokens": tokens}
    (out_dir / "training_info.json").write_text(json.dumps(info), encoding="utf-8")
    write_marker(out_dir, "baseline checkpoint from the cpu_stub warm-up run")
    return out_dir


def run_stage(base_dir: Path, dataset_dir: Path, config_path: Path, stage_dir: Path) -> list[tuple[int, Path]]:
    """Fine-tune the baseline on the stage's shards (skeleton-stub cpu_stub backend).

    Writes `stage_dir/logs/train_data_step<N>.jsonl` for every step and
    `stage_dir/checkpoints/step_<N>/` every `save_period` steps and at the
    last step. Returns [(step, checkpoint dir)].
    """
    model_mod.configure_torch()
    cfg = load_config(config_path, kind="trainer")
    torch.manual_seed(cfg["seed"])
    model = model_mod.load(base_dir)
    loader = SequentialShardLoader(dataset_dir, cfg["batch_size"])
    logs = Path(stage_dir) / "logs"
    ckpts = Path(stage_dir) / "checkpoints"
    logs.mkdir(parents=True)
    ckpts.mkdir(parents=True)
    write_marker(logs, "cpu_stub trainer logs in NeMo RL train_data_step<N>.jsonl format")
    save_period = cfg.get("save_period") or cfg["max_steps"]
    saved: list[tuple[int, Path]] = []

    def on_step(step, batch, ids, mask, total_valid_tokens):
        # One line per consumed sample (grpo_sync.py:1441-1453 keys; logger.py:1436 adds idx).
        with (logs / f"train_data_step{step}.jsonl").open("w", encoding="utf-8") as f:
            for i in range(len(batch)):
                row_ids = ids[i].tolist()
                row_mask = mask[i].tolist()
                length = sum(1 for t in row_ids if t != task.PAD)
                sample = {
                    "rewards": 0.0,
                    "input_lengths": length,
                    "token_loss_mask": [int(m) for m in row_mask],
                    "sample_loss_mask": 1.0,
                    "token_ids": row_ids,
                    "idx": i,
                }
                f.write(json.dumps(sample) + "\n")
        if step % save_period == 0 or step == cfg["max_steps"]:
            d = ckpts / f"step_{step}"
            model.eval()
            model_mod.save(model, d)
            model.train()
            # training_info.json as NeMo RL's checkpointer writes it (checkpoint.py:406-412).
            info = {"consumed_samples": loader.samples_yielded, "current_step": step, "total_steps": step,
                    "current_epoch": 0, "total_valid_tokens": total_valid_tokens}
            (d / "training_info.json").write_text(json.dumps(info), encoding="utf-8")
            torch.save(loader.state_dict(), d / "train_dataloader.pt")
            write_marker(d, f"cpu_stub checkpoint at step {step}")
            saved.append((step, d))

    _train_loop(model, loader, lr=cfg["lr"], max_steps=cfg["max_steps"], on_step=on_step)
    return saved
