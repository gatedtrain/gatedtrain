"""AN-25, AN-9 (revision 25): the batch generator protocol.

A scenario pack declares a batch generator as data (`BatchGenerator`), so no pack code decides how the keeper runs
it; gatedtrain implements the call (`generate_batch`) and the keeper runs it in its inference container, one
container per arm for all registered seeds. This module holds what both sides of the container agree on, and no
container code:

- the container layout: the sealed items at `/in/items.jsonl` (one canonical JSON line `{"item_id", "input"}` per
  item, never a label), the sampler and the seeds at `/in/sampler.json`, the model files at `/in/model`, the script
  at `/gen/<name>` and one empty output directory `/out`; `container_argv` is the command, and nothing about the
  items is passed in it;
- `write_items` and `sampler_doc`, the two input files;
- `read_predictions`, the strict read-back of `/out`: exactly one regular file `predictions.jsonl` of canonical JSON
  lines `{"item_id", "prediction", "seed"}`, one per item and seed. Anything else is a crash (AN-9), and the refusal
  names counts only, never a prediction (it may quote a sealed input).
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from ..canonical import canonical_json
from ..errors import Problem, Refused

RULE = "AN-25"
CONTAINER_ITEMS = "/in/items.jsonl"
CONTAINER_SAMPLER = "/in/sampler.json"
CONTAINER_MODEL = "/in/model"
CONTAINER_SCRIPT_DIR = "/gen"
CONTAINER_OUT = "/out"
PREDICTIONS = "predictions.jsonl"
MAX_PREDICTIONS_BYTES = 256 * 1024 * 1024
DEFAULT_TIME_LIMIT_S = 1200            # design 05 §18.10 (AN-25)

_PREDICTION_KEYS = frozenset({"item_id", "seed", "prediction"})
CRASH = "the inference output is not one prediction per item and seed (crash)"
CRASH_FIX = "fix the scenario pack's batch generator so it writes only /out/predictions.jsonl, one line per item and seed"


@dataclass(frozen=True)
class BatchGenerator:
    """A pack's batch generator: `script`, the absolute path of a file inside the pack's installed package;
    `interpreter`, the absolute container path that runs it (the image's own, or inside the keeper's eval trees when
    `eval_trees`); whether it needs the GPU; and its time limit in seconds."""

    script: Path
    interpreter: str
    gpu: bool = False
    time_limit_s: int = DEFAULT_TIME_LIMIT_S
    eval_trees: bool = False

    def __post_init__(self) -> None:
        if not (isinstance(self.script, Path) and self.script.is_absolute()):
            raise ValueError("BatchGenerator.script must be an absolute Path")
        if not (isinstance(self.interpreter, str) and PurePosixPath(self.interpreter).is_absolute()):
            raise ValueError("BatchGenerator.interpreter must be an absolute container path")
        if not (isinstance(self.gpu, bool) and isinstance(self.eval_trees, bool)):
            raise ValueError("BatchGenerator.gpu and eval_trees must be booleans")
        if isinstance(self.time_limit_s, bool) or not isinstance(self.time_limit_s, int) or self.time_limit_s < 1:
            raise ValueError("BatchGenerator.time_limit_s must be a positive integer")


def container_argv(gen: BatchGenerator) -> tuple[str, ...]:
    """The command the container runs: the script in isolated mode with the fixed container paths."""
    return (gen.interpreter, "-I", f"{CONTAINER_SCRIPT_DIR}/{gen.script.name}", "--items", CONTAINER_ITEMS,
            "--sampler", CONTAINER_SAMPLER, "--model", CONTAINER_MODEL, "--out", CONTAINER_OUT)


def _refuse(rule: str, field: str, message: str, fix: str) -> Refused:
    return Refused([Problem(rule, field, message, fix)])


def write_items(path: Path, items: list[dict]) -> None:
    """One canonical JSON line `{"item_id": i, "input": ...}` per item (the label, if any, is never written), in a
    new file (`O_EXCL | O_NOFOLLOW`, mode 0600)."""
    try:
        data = b"".join(canonical_json({"item_id": i, "input": item["input"]}) + b"\n" for i, item in enumerate(items))
    except ValueError:
        raise _refuse(RULE, "items", "an item's input cannot be written as canonical JSON",
                      "seal the data again; sealed inputs are canonical JSON") from None
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        os.fchmod(fd, 0o600)
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
    finally:
        os.close(fd)


def sampler_doc(sampler: dict, seeds: list) -> dict:
    """The content of `/in/sampler.json`."""
    return {"sampler": sampler, "seeds": seeds}


def _no_constant(name: str) -> Any:
    raise ValueError("non-finite number")


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict:
    doc = dict(pairs)
    if len(doc) != len(pairs):
        raise ValueError("duplicate key")
    return doc


def read_predictions(out_dir: Path, n_items: int, seeds: list) -> list[dict]:
    """`[{"item_id", "seed", "prediction"}]` in (seed, item) order from the container's output directory, or Refused
    (AN-9) when the directory holds anything but the regular file `predictions.jsonl`, the file exceeds
    `MAX_PREDICTIONS_BYTES`, a line is not canonical JSON with exactly the keys `item_id`, `seed`, `prediction`, or a
    (item_id, seed) pair is missing, repeated or unknown. The refusal carries counts only."""
    expected = n_items * len(seeds)

    def crash(detail: str) -> Refused:
        return _refuse("AN-9", "predictions", f"{CRASH}: {detail}", CRASH_FIX)

    try:
        dfd = os.open(out_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError:
        raise crash("the output directory cannot be opened") from None
    try:
        names = os.listdir(dfd)
        if names != [PREDICTIONS]:
            raise crash(f"the output directory holds {len(names)} entries, expected only {PREDICTIONS}")
        try:
            fd = os.open(PREDICTIONS, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=dfd)
        except OSError:
            raise crash(f"{PREDICTIONS} is not a regular file") from None
    finally:
        os.close(dfd)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise crash(f"{PREDICTIONS} is not a regular file")
        if st.st_size > MAX_PREDICTIONS_BYTES:
            raise crash(f"{PREDICTIONS} has {st.st_size} bytes, more than {MAX_PREDICTIONS_BYTES}")
        chunks, total = [], 0
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_PREDICTIONS_BYTES:
                raise crash(f"{PREDICTIONS} has more than {MAX_PREDICTIONS_BYTES} bytes")
            chunks.append(chunk)
    finally:
        os.close(fd)
    data = b"".join(chunks)
    lines = data.split(b"\n")
    if lines[-1] != b"":
        raise crash("the last line has no line end")
    lines.pop()
    if len(lines) != expected:
        raise crash(f"{len(lines)} lines, expected {expected}")
    seed_index: dict[bytes, int] = {}
    for j, s in enumerate(seeds):
        seed_index.setdefault(canonical_json(s), j)
    found: dict[tuple[int, int], Any] = {}
    for n, line in enumerate(lines, 1):
        try:
            doc = json.loads(line.decode("utf-8"), object_pairs_hook=_unique_keys, parse_constant=_no_constant)
            canonical = isinstance(doc, dict) and canonical_json(doc) == line
        except (ValueError, RecursionError):
            canonical = False
        if not canonical:
            raise crash(f"line {n} of {expected} is not a canonical JSON object")
        if set(doc) != _PREDICTION_KEYS:
            raise crash(f"line {n} of {expected} does not have exactly the keys item_id, seed, prediction")
        item, seed = doc["item_id"], seed_index.get(canonical_json(doc["seed"]))
        if isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < n_items or seed is None:
            raise crash(f"line {n} of {expected} names an item or seed that is not in the evaluation")
        if (seed, item) in found:
            raise crash(f"line {n} of {expected} repeats an item and seed")
        found[(seed, item)] = doc["prediction"]
    # The count matched and no pair repeats or is unknown, so every pair is present.
    return [{"item_id": i, "seed": s, "prediction": found[(j, i)]} for j, s in enumerate(seeds) for i in range(n_items)]


class BatchRunner(Protocol):
    """What runs a batch generator (the keeper's inference container); returns predictions as `read_predictions`
    does."""

    def run_batch(self, gen: BatchGenerator, model: dict, items: list[dict], sampler: dict,
                  seeds: list) -> list[dict]: ...


def generate_batch(gen: BatchGenerator, model: dict, items: list[dict], sampler: dict, seeds: list, *,
                   runner: BatchRunner) -> list[dict]:
    """Spec §5.5's call: the predictions of `gen` for every item and seed, through `runner`. The items handed on carry
    `input` only, so a label cannot reach the runner."""
    return runner.run_batch(gen, dict(model), [{"input": item["input"]} for item in items], dict(sampler),
                            list(seeds))
