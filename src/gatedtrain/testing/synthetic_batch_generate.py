"""The synthetic pack's batch generator (AN-25): standard library only, run as

    python -I synthetic_batch_generate.py --items ITEMS --sampler SAMPLER --model MODEL_DIR --out OUT_DIR

It is copied into the keeper's inference container, so it must not import gatedtrain. It reads the lookup table from
`MODEL_DIR/table.json`, the items (`{"item_id", "input"}` lines) and the seeds (`{"sampler", "seeds"}`), and writes
`OUT_DIR/predictions.jsonl`: one canonical JSON line `{"item_id", "prediction", "seed"}` per seed and item, the
prediction equal to `synthetic_pack.generate` for a model whose `table` is that file's content. `feature` is defined
here only and `synthetic_pack` imports it, so both forms key on the same token. The lines use the writer recipe of
`gatedtrain.ainative.batch`, canonical for the pack's text labels.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def feature(text):
    """The token a lookup model keys on: the text before the first '-'."""
    return text.split("-", 1)[0] if isinstance(text, str) else None


def _line(doc) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode() + b"\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    for name in ("--items", "--sampler", "--model", "--out"):
        ap.add_argument(name, required=True)
    args = ap.parse_args(argv)
    with open(os.path.join(args.model, "table.json"), encoding="utf-8") as f:
        table = json.load(f)
    with open(args.sampler, encoding="utf-8") as f:
        seeds = json.load(f)["seeds"]
    with open(args.items, encoding="utf-8") as f:
        items = [json.loads(line) for line in f.read().split("\n") if line]
    out = b"".join(_line({"item_id": item["item_id"], "prediction": table.get(feature(item["input"])), "seed": seed})
                   for seed in seeds for item in items)
    fd = os.open(os.path.join(args.out, "predictions.jsonl"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
