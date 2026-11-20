"""Regenerate `toy.jsonl`: 600 labelled rows for the synthetic scenario pack (tests/test_an_e2e.py).

Run: python tests/fixtures/ainative/make_toy.py

Inputs are `<token>-<i>` (20 tokens, 5 labels). The label follows the token except for one occurrence in four,
which carries the label "odd", so a model that learned the task still misses some items.
"""

from __future__ import annotations

import json
from pathlib import Path

N_ROWS = 600


def rows(n: int = N_ROWS) -> list[dict]:
    return [{"input": f"tok{i % 20}-{i}", "label": "odd" if (i // 20) % 4 == 3 else f"L{(i % 20) % 5}"}
            for i in range(n)]


def main() -> None:
    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows())
    Path(__file__).with_name("toy.jsonl").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
