"""The synthetic pack's train job (AN-22): standard library only, run in a container of the local backend as

    python synthetic_train_job.py --table TABLE --out OUT_DIR

It must not import gatedtrain: the backend mounts it, with the table the `train` stage fitted, read-only under
`/inputs` of an image that has only Python. It writes `OUT_DIR/table.json` (the lookup table, canonical JSON, mode
0644 so the keeper user can read it after the hand-off) and prints one line with the table's size. It prints no
token and no label.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    with open(args.table, encoding="utf-8") as f:
        table = json.load(f)
    if not (isinstance(table, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in table.items())):
        print("the table is not a mapping of text to text", file=sys.stderr)
        return 2
    data = json.dumps(table, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    path = os.path.join(args.out, "table.json")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.chmod(path, 0o644)
    print(f"trained a lookup table of {len(table)} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
