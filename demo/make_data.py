"""Generate the demo task data (test fixture, skeleton-stub task).

skeleton-stub: this is the Tier A stand-in task of MVP-SPEC section 3,
2-digit addition as character strings with an exact-match score.

The data is a test fixture written before any code
exists, because the registration must record the prompt-set hash before
the run (05 section 3.7). The output is deterministic: running this
script again rewrites byte-identical files.

Usage (from the repository root):
    .venv/bin/python demo/make_data.py

Layout written under demo/data/ (one JSON object per line, keys sorted,
compact separators, LF line ends):
    train/shard-0000{0..4}.jsonl    training shards used by the demo stage
    train-shuffled/shard-0000{0,1}.jsonl  same prompts as train shards 0-1
                                    with answers permuted (bad-training control)
    unused/shard-00000.jsonl        training-format shard no stage uses
    warmup/shard-00000.jsonl        data for the baseline warm-up run
    heldout/prompts.jsonl           500 held-out evaluation prompts
    heldout-small/prompts.jsonl     first 30 held-out prompts (too-few-prompts control)

Row format: {"answer": "<sum>", "prompt": "<a>+<b>="}. In prompt sets the
problem index is the 0-based line number.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 20261109
N_HELDOUT = 500
N_HELDOUT_SMALL = 30
SHARD_ROWS = 1000
N_TRAIN_SHARDS = 5
N_SHUFFLED_SHARDS = 2


def _row(a: int, b: int, answer: str | None = None) -> dict[str, str]:
    return {"prompt": f"{a}+{b}=", "answer": answer if answer is not None else str(a + b)}


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows)
    path.write_bytes(text.encode("utf-8"))


def main() -> None:
    root = Path(__file__).resolve().parent / "data"
    rng = random.Random(SEED)
    pairs = [(a, b) for a in range(100) for b in range(100)]
    rng.shuffle(pairs)

    cursor = 0

    def take(n: int) -> list[tuple[int, int]]:
        nonlocal cursor
        out = pairs[cursor : cursor + n]
        cursor += n
        return out

    heldout = [_row(a, b) for a, b in take(N_HELDOUT)]
    train_shards = [[_row(a, b) for a, b in take(SHARD_ROWS)] for _ in range(N_TRAIN_SHARDS)]
    unused = [_row(a, b) for a, b in take(SHARD_ROWS)]
    warmup = [_row(a, b) for a, b in take(SHARD_ROWS)]

    shuffled_shards = []
    for shard in train_shards[:N_SHUFFLED_SHARDS]:
        answers = [r["answer"] for r in shard]
        rng.shuffle(answers)
        shuffled_shards.append([{"prompt": r["prompt"], "answer": ans} for r, ans in zip(shard, answers)])

    _write(root / "heldout" / "prompts.jsonl", heldout)
    _write(root / "heldout-small" / "prompts.jsonl", heldout[:N_HELDOUT_SMALL])
    for i, shard in enumerate(train_shards):
        _write(root / "train" / f"shard-{i:05d}.jsonl", shard)
    for i, shard in enumerate(shuffled_shards):
        _write(root / "train-shuffled" / f"shard-{i:05d}.jsonl", shard)
    _write(root / "unused" / "shard-00000.jsonl", unused)
    _write(root / "warmup" / "shard-00000.jsonl", warmup)


if __name__ == "__main__":
    main()
