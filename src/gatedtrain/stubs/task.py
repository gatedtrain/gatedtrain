"""skeleton-stub task: 2-digit addition as character strings (MVP-SPEC section 3).

Rows are JSON objects {"prompt": "<a>+<b>=", "answer": "<a+b>"}. A
training dataset is a directory of `*.jsonl` shard files; a prompt set is a
directory holding `prompts.jsonl`, whose 0-based line number is the
problem index (tests/README.md section 4.3).

Encoding: the prompt is left-padded with PAD to PROMPT_LEN tokens so every
answer starts at the same position; the answer is followed by EOS. The
padding is part of the stand-in model's input format, not of the data.
"""

from __future__ import annotations

import json
from pathlib import Path

PAD, EOS = 0, 1
CHARS = "0123456789+="
VOCAB = ["<pad>", "<eos>", *CHARS]
CHAR_TO_ID = {c: i + 2 for i, c in enumerate(CHARS)}
PROMPT_LEN = 6        # "99+99="
ANSWER_LEN = 4        # up to 3 digits plus EOS
SEQ_LEN = PROMPT_LEN + ANSWER_LEN


def read_rows(path: Path) -> list[dict[str, str]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def shard_files(dataset_dir: Path) -> list[Path]:
    """Shards in file-name order (tests/README.md section 6.1)."""
    return sorted(Path(dataset_dir).glob("*.jsonl"), key=lambda p: p.name.encode("utf-8"))


def load_dataset(dataset_dir: Path) -> list[dict[str, str]]:
    """All rows of all shards, shards in file-name order, rows in file order."""
    rows: list[dict[str, str]] = []
    for shard in shard_files(dataset_dir):
        rows.extend(read_rows(shard))
    if not rows:
        raise ValueError(f"dataset {dataset_dir} has no rows")
    return rows


def load_prompts(prompt_dir: Path) -> list[dict[str, str]]:
    return read_rows(Path(prompt_dir) / "prompts.jsonl")


def encode_prompt(prompt: str) -> list[int]:
    ids = [CHAR_TO_ID[c] for c in prompt]
    if len(ids) > PROMPT_LEN:
        raise ValueError(f"prompt longer than {PROMPT_LEN} characters: {prompt!r}")
    return [PAD] * (PROMPT_LEN - len(ids)) + ids


def encode_example(prompt: str, answer: str) -> tuple[list[int], list[int]]:
    """Token ids (length SEQ_LEN) and the loss mask over predicted tokens."""
    ans = [CHAR_TO_ID[c] for c in answer] + [EOS]
    if len(ans) > ANSWER_LEN:
        raise ValueError(f"answer longer than {ANSWER_LEN - 1} characters: {answer!r}")
    ids = encode_prompt(prompt) + ans
    mask = [0] * PROMPT_LEN + [1] * len(ans)
    ids += [PAD] * (SEQ_LEN - len(ids))
    mask += [0] * (SEQ_LEN - len(mask))
    return ids, mask


def decode_answer(token_ids: list[int]) -> str:
    """Characters up to the first EOS; PAD decodes to nothing."""
    out = []
    for t in token_ids:
        if t == EOS:
            break
        if t >= 2:
            out.append(VOCAB[t])
    return "".join(out)


def exact_match(response: str, expected: str) -> float:
    return 1.0 if response == expected else 0.0
