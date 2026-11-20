"""Consumption ledger writer (MVP-SPEC section 2.9).

Reads only what the trainer already writes, with NeMo RL's names and shapes
at 4f39fd8, and never changes the trainer:

- `train_data_step<N>.jsonl` (grpo_sync.py:1458-1460): one line per sample,
  no dataset row ID, so only the number of samples per step is used;
- `step_<N>/training_info.json` with `consumed_samples` (grpo_sync.py:1323,
  :1353) and `step_<N>/train_dataloader.pt` (grpo_sync.py:1407-1414).

The loader reads shards in file-name order and rows in file order with
NeMo RL's drop_last epoch semantics (decision D-1, tests/README.md section
10.1), so the loader positions of each step map to shards. Rows are written at precision `iteration` (05 section
11.2, 05:341-347): one row per (step, shard) the step drew from, with
`data_ref.row = null`. This is the prototype extension of interim rule G-1:
05 section 12.2 allows only `row` and
`step_range` (05:381) and requires a row key (05:361). No row IDs are
reconstructed.

Only steps covered by a saved checkpoint are written: a record not covered
by any committed cut is orphaned and excluded (05:409). `orphaned_rows`
lists those records, one per (step, shard), for the impact query's counts
(05 section 12.4, 05:424; R10A-9); the runner keeps them in lineage, never
in the ledger. `consumed_samples`
of each checkpoint must equal the samples logged up to its step; a mismatch
is an error, not a guess.

Design revision 11 (R11-18, 05 sections 11.2 and 12.2; tests/README.md section 11.16 R11A-18): every
record carries `derivation`, `observed` (emitted by the trainer during training) or `reconstructed`
(derived afterwards). A reconstructed record claims at most `iteration` precision unless the adapter
passed a replay conformance test, after which `row` is allowed (`check_precision`). The Tier A and Tier B
rows are `reconstructed`: the shard of each step is derived from the step sample counts and the
sequential loader order (decision D-1), not emitted by the trainer (design choice 6).

Tier B note (interim rule G-7): NeMo RL's `consumed_samples` counts prompts
(grpo_sync.py:1323) while `train_data_step<N>.jsonl` counts prompts times
generations; the Tier B reader passes `samples_per_position` = generations
per prompt. Tier A writes one line per consumed sample (1).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import TAGS, __version__
from .canonical import manifest_hash

STEP_FILE_RE = re.compile(r"^train_data_step(\d+)\.jsonl$")
CKPT_DIR_RE = re.compile(r"^step_(\d+)$")
WRITER_ID = f"gatedtrain.ledger@{__version__}"

DATA_REF = pa.struct([("snapshot", pa.string()), ("shard", pa.string()), ("row", pa.string()),
                      ("mixture", pa.string()), ("component", pa.int64())])
PARENT_REF = pa.struct([("kind", pa.string()), ("id", pa.string())])
CHUNK = pa.struct([("first", pa.int64()), ("last", pa.int64())])
SEED_INPUTS = pa.struct([("seed", pa.int64())])

# 05 section 12.2 ConsumptionRecord fields (05:372-382).
SCHEMA = pa.schema([
    ("run_id", pa.string()), ("attempt_id", pa.string()), ("step", pa.int64()), ("sample_id", pa.string()),
    ("data_ref", DATA_REF), ("parent_ref", PARENT_REF),
    ("outcome", pa.string()), ("reason", pa.string()), ("rule_version", pa.string()),
    ("gen_version", pa.int64()), ("reward_epoch", pa.int64()), ("rewards", pa.list_(pa.float64())),
    ("sampler_config_hash", pa.string()), ("seed_inputs", SEED_INPUTS), ("chunk", CHUNK),
    ("precision", pa.string()), ("derivation", pa.string()), ("writer_id", pa.string()),
    # Resumed external runs (EXT-RES-1, EXT-RES-5): the attempt number and `kept` | `orphaned`. A ledger written
    # before them has neither column, and a row without them is a kept row of attempt 0.
    ("controller_epoch", pa.int64()), ("status", pa.string()),
], metadata={
    b"schema": b"gatedtrain.consumption_record",
    b"schema_version": b"1.0",
    b"note": b"prototype extension (interim rule G-1): precision 'iteration', data_ref.row null; "
             b"05 section 12.2 allows row or step_range only",
    b"tags": ",".join(TAGS).encode(),
})


PRECISIONS = ("row", "step_range", "iteration", "stage")
DERIVATIONS = ("observed", "reconstructed")
# The derivation of every row this writer emits (R11A-18, design choice 6).
LEDGER_DERIVATION = "reconstructed"


def check_precision(precision: str, derivation: str, *, replay_conformance: bool = False) -> None:
    """05 section 11.2 (R11-18): `observed` allows every precision; `reconstructed` allows `iteration` and
    `stage`, and `row` only after a replay conformance test. Raises ValueError for a disallowed pair or an
    unknown value."""
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}; one of {', '.join(PRECISIONS)}")
    if derivation not in DERIVATIONS:
        raise ValueError(f"unknown derivation {derivation!r}; one of {', '.join(DERIVATIONS)}")
    if derivation == "reconstructed" and precision not in ("iteration", "stage"):
        if not (precision == "row" and replay_conformance):
            raise ValueError(f"a reconstructed record cannot claim precision {precision!r} "
                             "(iteration or stage; row only after a replay conformance test, 05 section 11.2)")


# Orphaned records (R10A-9): (run_id, stage_id) are added by the runner's lineage writer.
ORPHAN_SCHEMA = pa.schema([
    ("run_id", pa.string()), ("stage_id", pa.string()), ("attempt_id", pa.string()), ("step", pa.int64()),
    ("snapshot", pa.string()), ("shard", pa.string()), ("precision", pa.string()),
])


def _count_lines(path: Path) -> int:
    with Path(path).open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def read_step_sizes(log_dir: Path) -> list[tuple[int, int]]:
    """[(step, samples logged at that step)] from train_data_step<N>.jsonl, steps 1..N contiguous."""
    steps = []
    for p in Path(log_dir).iterdir():
        m = STEP_FILE_RE.match(p.name)
        if m:
            steps.append((int(m.group(1)), _count_lines(p)))
    steps.sort()
    if not steps:
        raise ValueError(f"no train_data_step<N>.jsonl files in {log_dir}")
    numbers = [s for s, _ in steps]
    if numbers != list(range(1, len(numbers) + 1)):
        raise ValueError(f"train_data_step files are not contiguous from step 1: {numbers[:5]}...")
    return steps


def read_checkpoints(checkpoints_dir: Path) -> list[tuple[int, int, Path]]:
    """[(step, consumed_samples, dir)] from step_<N>/training_info.json (checkpoint.py:406-412)."""
    out = []
    for p in Path(checkpoints_dir).iterdir():
        m = CKPT_DIR_RE.match(p.name)
        if m and p.is_dir():
            info = json.loads((p / "training_info.json").read_text(encoding="utf-8"))
            out.append((int(m.group(1)), int(info["consumed_samples"]), p))
    out.sort()
    if not out:
        raise ValueError(f"no step_<N> checkpoints in {checkpoints_dir}")
    return out


def shard_table(dataset_dir: Path) -> list[tuple[str, int]]:
    """[(shard file name, row count)] in the loader's read order (file-name order)."""
    shards = sorted(Path(dataset_dir).glob("*.jsonl"), key=lambda p: p.name.encode("utf-8"))
    return [(p.name, _count_lines(p)) for p in shards]


def step_shards(step_sizes: list[tuple[int, int]], shards: list[tuple[str, int]],
                samples_per_position: int = 1) -> dict[int, list[str]]:
    """Map each step to the shards its loader positions fall in.

    Loader semantics are NeMo RL's `StatefulDataLoader(..., drop_last=True)`
    (nemo_rl/algorithms/grpo.py:883-890 at 4f39fd8). Rows are read in order. A batch that
    does not fit in the rest of the epoch is dropped, so the tail rows are
    skipped and the next epoch restarts at row 0; no batch straddles an
    epoch boundary. A step with zero samples maps to no shard.
    """
    total = sum(n for _, n in shards)
    if total == 0:
        raise ValueError("the dataset has no rows")
    bounds, start = [], 0
    for name, n in shards:
        bounds.append((name, start, start + n))
        start += n
    out: dict[int, list[str]] = {}
    cursor = 0
    for step, size in step_sizes:
        if size % samples_per_position:
            raise ValueError(f"step {step}: {size} lines is not a multiple of {samples_per_position}")
        positions = size // samples_per_position
        if positions == 0:
            out[step] = []                                   # D-2
            continue
        if positions > total:
            raise ValueError(f"step {step}: {positions} rows per step exceed the {total}-row dataset; "
                             "under drop_last such a batch is never yielded")
        if cursor + positions > total:                       # D-1: drop the tail, restart the epoch
            cursor = 0
        lo, hi = cursor, cursor + positions
        out[step] = [name for name, s, e in bounds if s < hi and lo < e]
        cursor = hi
    return out


def ledger_rows(log_dir: Path, checkpoints_dir: Path, dataset_dir: Path, *, run_id: str, attempt_id: str = "0",
                samples_per_position: int = 1) -> list[dict]:
    """The ledger rows, after every consistency check. Raises ValueError and writes nothing."""
    sizes = read_step_sizes(log_dir)
    ckpts = read_checkpoints(checkpoints_dir)
    cumulative, total = {}, 0
    for step, size in sizes:
        total += size
        cumulative[step] = total
    for step, consumed, d in ckpts:
        if step not in cumulative:
            raise ValueError(f"checkpoint {d.name} has no train_data_step{step}.jsonl")
        if cumulative[step] % samples_per_position:
            raise ValueError(f"{d.name}: {cumulative[step]} logged lines is not a multiple of "
                             f"samples_per_position {samples_per_position} (G-7)")
        logged = cumulative[step] // samples_per_position
        if logged != consumed:
            raise ValueError(f"{d.name}: consumed_samples = {consumed} but the step logs hold {logged} up to step {step} "
                             f"(samples_per_position {samples_per_position}, G-7)")
    check_precision("iteration", LEDGER_DERIVATION)
    last_cut = ckpts[-1][0]
    covered = [(s, n) for s, n in sizes if s <= last_cut]   # orphaned records excluded (05:409)
    snapshot = manifest_hash(dataset_dir)
    mapping = step_shards(covered, shard_table(dataset_dir), samples_per_position)
    rows = []
    for step, _ in covered:
        for shard in mapping[step]:                          # a zero-sample step writes no row (D-2)
            sample_id = hashlib.sha256(f"{run_id}\x00{attempt_id}\x00{step}\x00{shard}".encode()).hexdigest()
            rows.append({
                "run_id": run_id, "attempt_id": attempt_id, "step": step, "sample_id": "sha256:" + sample_id,
                "data_ref": {"snapshot": snapshot, "shard": shard, "row": None, "mixture": None, "component": None},
                "parent_ref": None, "outcome": "consumed", "reason": None, "rule_version": None,
                "gen_version": None, "reward_epoch": None, "rewards": None, "sampler_config_hash": None,
                "seed_inputs": None, "chunk": None, "precision": "iteration", "derivation": LEDGER_DERIVATION,
                "writer_id": WRITER_ID,
            })
    return rows


def orphaned_rows(log_dir: Path, checkpoints_dir: Path, dataset_dir: Path, *, attempt_id: str = "0",
                  samples_per_position: int = 1) -> list[dict]:
    """Records after the attempt's last committed cut (05:409, 05:424), at precision `iteration`: one per
    (step, shard) the step drew from, mapped with the same loader semantics as the ledger."""
    sizes = read_step_sizes(log_dir)
    last_cut = read_checkpoints(checkpoints_dir)[-1][0]
    if sizes[-1][0] <= last_cut:
        return []
    snapshot = manifest_hash(dataset_dir)
    mapping = step_shards(sizes, shard_table(dataset_dir), samples_per_position)
    return [{"attempt_id": attempt_id, "step": step, "snapshot": snapshot, "shard": shard, "precision": "iteration"}
            for step, _ in sizes if step > last_cut for shard in mapping[step]]


def write_ledger(log_dir: Path, checkpoints_dir: Path, dataset_dir: Path, out_path: Path, *, run_id: str,
                 attempt_id: str = "0", samples_per_position: int = 1) -> Path:
    """Write `ConsumptionRecord`-shaped Parquet rows at precision `iteration` (G-1)."""
    rows = ledger_rows(log_dir, checkpoints_dir, dataset_dir, run_id=run_id, attempt_id=attempt_id,
                       samples_per_position=samples_per_position)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), out_path)
    return out_path
