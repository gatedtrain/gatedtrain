"""Generate the tiny GSM8K-shaped source fixture (deterministic).

Writes gsm8k-source/train.jsonl (40 rows) and gsm8k-source/test.jsonl (30 rows)
next to this file, in the row shape of Hugging Face `openai/gsm8k` config
`main`: {"question": ..., "answer": "<reasoning>\n#### <final answer>"}.
NeMo RL keeps the text after `####`, stripped (nemo_rl/data/datasets/
response_datasets/gsm8k.py:22-25 at 4f39fd8); a few rows use a thousands
comma and extra spaces after `####` to pin that rule.

Usage (from the repository root): .venv/bin/python tests/fixtures/tierb/make_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def row(i: int, split: str) -> dict[str, str]:
    a = 7 * i + (3 if split == "test" else 1)
    b = 11 * i + 5
    s = a + b
    if i % 9 == 4:
        a, b = a * 37, b * 41
        s = a + b
    shown = f"{s:,}" if s >= 1000 else str(s)
    sep = "####  " if i % 7 == 3 else "#### "
    question = f"{split.upper()} {i}: Tom has {a} apples and buys {b} more. How many apples does he have now?"
    answer = f"Tom has {a} + {b} = <<{a}+{b}={s}>>{s} apples.\n{sep}{shown}"
    return {"question": question, "answer": answer}


def write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8"))


def main() -> None:
    write(HERE / "gsm8k-source" / "train.jsonl", [row(i, "train") for i in range(40)])
    write(HERE / "gsm8k-source" / "test.jsonl", [row(i, "test") for i in range(30)])


if __name__ == "__main__":
    main()
