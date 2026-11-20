"""The adapter SDK: helpers an external-run adapter may use (stable names; the adapters of this package and of the
recipe packages import them from here, never from each other).

Standard library only, plus `gatedtrain.canonical` and `gatedtrain.errors` (MM-12: an adapter imports no framework). Every reader
returns (value, None) or (None, why) instead of raising, so a caller can turn a defect into a `Problem`.

Resumed runs (EXT-RES-1 to EXT-RES-5). A run directory with `attempts.jsonl` is the new layout: ext-collect builds the
segment chain (`gatedtrain.ext.attempts`) and calls the adapter's
`consumption(stage_out, data, cuts, *, run_id, chain: list[Segment] | None = None)`, which returns
`{"rows": [...], "orphaned": [...], "precision": str}`. `rows` are the kept records, `orphaned` the records after a
non-final attempt's last cut (steps a later attempt ran again); each row carries the segment's `attempt_id`
(`<run_id>.<A>`), `controller_epoch` A and `status` (`kept` | `orphaned`), and `gatedtrain.ext.attempts.split_steps`
does the split. An adapter whose `consumption` takes no `chain` keyword is called as before for a run with one
non-empty attempt, and ext-collect refuses a run with more (EXT-RES-3: "this adapter does not support resumed runs").

The chain's last segment ends at run_args.json `max_steps` when that is an integer not below the last cut, else at the
last cut (records of the final attempt after it are then ignored, as for an old-layout run). ext-collect stamps `status`,
`controller_epoch` and `parent_ref` (`<run_id>.<A-1>:<branch_step>`, none for attempt 1) on the rows itself, and refuses
(EXT-RES-4) a row whose `attempt_id` or `controller_epoch` names no segment. An adapter may define
`chain_problems(stage_out, data, cuts, chain) -> list[Problem]`, run before anything is written, and should split its
records with `split_steps` before any precision fallback so a multi-attempt run is checked at every precision.
`steps_redone` in the report is the result's own `steps_redone` integer, else the distinct (attempt, step) of its
`orphaned` rows; for more than one attempt and neither, it is null in the report data and "unknown" in its text,
never 0.
"""

from __future__ import annotations

import hashlib
import json
import math
import posixpath
import struct
from pathlib import Path, PurePosixPath
from typing import Any

from ..canonical import file_sha256, sha256_bytes, text_lines
from ..errors import Problem
from ..paths import resolve

# File names of the launcher layout every area runtime writes (MULTIMODAL-SPEC section 5).
RUN_ARGS = "run_args.json"                  # <stage-out>/run_args.json: the launcher's resolved arguments
CUT_MARKER = "cut.json"                     # <cut>/cut.json, written last
RUNTIME_FILE = "env-runtime.json"           # <stage-out>/env-runtime.json: the script the run executed
SCRIPT_KEY = "script_sha256"                # its key holding the script's own SHA-256 (EXT-83)

# The reference implementation's cap on the JSON header (safetensors `lib.rs`, `MAX_HEADER_SIZE`); a longer
# header is refused before it is read (EXT-43).
MAX_HEADER_SIZE = 100_000_000


# --- values ---

def is_int(x: Any) -> bool:
    """An integer that is not a boolean."""
    return isinstance(x, int) and not isinstance(x, bool)


def is_num(x: Any) -> bool:
    """A finite int or float that is not a boolean."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def float32(x: float) -> float:
    """The float32 value of `x` (a reward a framework logs as float32)."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def compact_sha256(obj: Any) -> str:
    """The area runtimes' own hash encoding (`canonical` in their scripts): SHA-256 of compact sorted JSON, not
    RFC 8785."""
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


MAX_SEED = 2 ** 63                          # derive_seed: 8 bytes >> 1


def derive_seed(*parts: Any) -> int:
    """The area runtimes' `derive_seed`: a 63-bit seed from SHA-256 of the parts joined by ':'."""
    return int.from_bytes(hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).digest()[:8], "big") >> 1


def norm_path(p: Any) -> Any:
    """`posixpath.normpath` of a string, anything else unchanged."""
    return posixpath.normpath(p) if isinstance(p, str) else p


# --- files ---

def plain_file(p: Path) -> bool:
    """A regular file that is not a symbolic link."""
    return p.is_file() and not p.is_symlink()


def sha_or_none(p: Path) -> str | None:
    """Return the file's SHA-256, or None so the caller can report an unreadable file (EXT-154)."""
    try:
        return file_sha256(p)
    except OSError:
        return None


def read_json(path: Path) -> tuple[Any, str | None]:
    """(the parsed JSON document, None) or (None, why the file cannot be used)."""
    if not plain_file(path):
        return None, f"{path.name} is missing"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, UnicodeDecodeError, ValueError, RecursionError) as exc:
        return None, f"{path.name} does not parse ({exc.__class__.__name__})"


def read_jsonl(path: Path) -> tuple[list[dict] | None, str | None]:
    """([line objects] in file order, None) or (None, why the file cannot be read)."""
    if not plain_file(path):
        return None, f"{path.name} is not a plain file"
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return None, f"{path.name} cannot be read ({exc.__class__.__name__})"
    out = []
    for n, line in enumerate(text_lines(text), start=1):
        if not line.strip():
            continue
        try:
            doc = json.loads(line)
        except (ValueError, RecursionError):
            return None, f"{path.name} line {n} does not parse"
        if not isinstance(doc, dict):
            return None, f"{path.name} line {n} is not a JSON object"
        out.append(doc)
    return out, None


def read_run_args(stage_out: Path) -> tuple[dict[str, Any] | None, str | None]:
    """(`<stage-out>/run_args.json` as a JSON object, None) or (None, why it cannot be used)."""
    path = Path(stage_out) / RUN_ARGS
    if not plain_file(path):
        return None, f"{RUN_ARGS} is missing"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError, RecursionError) as exc:
        return None, f"{RUN_ARGS} does not parse ({exc.__class__.__name__})"
    if not isinstance(doc, dict):
        return None, f"{RUN_ARGS} is not a JSON object"
    return doc, None


def read_cut_marker(d: Path) -> tuple[dict | None, str | None]:
    """(`<d>/cut.json` as an object with a `files` object, None) or (None, why not)."""
    doc, why = read_json(Path(d) / CUT_MARKER)
    if why:
        return None, f"{why} ({CUT_MARKER} is written last)"
    if not isinstance(doc, dict) or not isinstance(doc.get("files"), dict):
        return None, f"{CUT_MARKER} has no `files` object"
    return doc, None


def safetensors_problem(path: Path) -> str | None:
    """None when `path` is a whole safetensors file: 8-byte little-endian header length n (at most
    MAX_HEADER_SIZE), a JSON object header whose tensors have integer `data_offsets` [begin, end] with
    0 <= begin <= end (booleans are not integers), and a size of exactly 8 + n + the largest end offset.
    Read with the standard library only; a header that does not parse (deep nesting included) is a problem,
    never an exception (EXT-43)."""
    if not plain_file(path):
        return f"{path.name} is missing"
    size = path.stat().st_size
    try:
        with path.open("rb") as f:
            head = f.read(8)
            if len(head) < 8:
                return f"{path.name} is shorter than its 8-byte header length"
            (n,) = struct.unpack("<Q", head)
            if n > MAX_HEADER_SIZE:
                return f"{path.name} header length {n} exceeds the {MAX_HEADER_SIZE}-byte cap"
            if 8 + n > size:
                return f"{path.name} header length {n} runs past the end of the file"
            header = json.loads(f.read(n).decode("utf-8"))
    except (OSError, ValueError, RecursionError) as exc:
        return f"{path.name} header does not parse ({exc.__class__.__name__})"
    if not isinstance(header, dict):
        return f"{path.name} header is not a JSON object"
    end = 0
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        offs = entry.get("data_offsets") if isinstance(entry, dict) else None
        if not (isinstance(offs, list) and len(offs) == 2
                and all(isinstance(o, int) and not isinstance(o, bool) and o >= 0 for o in offs)):
            return f"{path.name} tensor {name!r} has no data_offsets"
        if offs[0] > offs[1]:
            return f"{path.name} tensor {name!r} has data_offsets {offs} with begin after end"
        end = max(end, offs[1])
    if size != 8 + n + end:
        return f"{path.name} is {size} bytes, not 8 + {n} + {end} (truncated or padded)"
    return None


# --- training data ---

def single_shard(data: Path) -> str | None:
    """The name of the one `*.jsonl` shard in `data`, or None when there is not exactly one."""
    names = sorted(p.name for p in Path(data).glob("*.jsonl"))
    return names[0] if len(names) == 1 else None


def shards_concat_sha256(data: Path) -> str:
    """SHA-256 of the training shards concatenated in the loader's file-name order (`ledger.shard_table`): the
    `train.jsonl` of a `gatedtrain export-gsm8k` export (its `concat_sha256`)."""
    from ..ledger import shard_table

    h = hashlib.sha256()
    for name, _ in shard_table(data):
        with (Path(data) / name).open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    return "sha256:" + h.hexdigest()


# --- launch records ---

def bound_mounts(mounts: Any, host: Path) -> list[dict]:
    """The launch record's read-only mounts whose host path resolves to `host`."""
    want = resolve(host)
    return [m for m in mounts or [] if isinstance(m, dict) and m.get("mode") == "ro"
            and isinstance(m.get("host"), str) and PurePosixPath(m["host"]).is_absolute()
            and resolve(m["host"]) == want]


def runtime_script_problems(stage_out: Path, launch: dict) -> list[Problem]:
    """EXT-83: `<stage-out>/env-runtime.json` `script_sha256` (the script the run executed, as the launcher hashes
    itself) equals the launch record's `script_sha256`."""
    fix = "collect the outputs of the run launched with this record's training script"
    path = Path(stage_out) / RUNTIME_FILE
    field = "--launch-record.script_sha256"
    if not plain_file(path):
        return [Problem("EXT-LAUNCH", field, f"{RUNTIME_FILE} is missing, so the script the run executed is not "
                        "shown (EXT-83)", fix)]
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError, RecursionError) as exc:
        return [Problem("EXT-LAUNCH", field, f"{RUNTIME_FILE} does not parse ({exc.__class__.__name__})", fix)]
    got = doc.get(SCRIPT_KEY) if isinstance(doc, dict) else None
    want = launch.get("script_sha256")
    if got != want:
        return [Problem("EXT-LAUNCH", field, f"{RUNTIME_FILE} `{SCRIPT_KEY}` is {got!r}, not the launch record's "
                        f"{want!r}: the run executed another training script (EXT-83)", fix)]
    return []


# --- argv ---

def resolve_option(token: Any, options: dict[str, bool]) -> tuple[str | None, str | None]:
    """(option, inline value) of an argv token as argparse reads it: an exact option or a unique prefix of one
    (allow_abbrev), optionally `=value`; (None, None) for anything else (an ambiguous prefix argparse refuses)."""
    if not (isinstance(token, str) and token.startswith("--") and len(token) > 2):
        return None, None
    name, eq, value = token.partition("=")
    if name in options:
        return name, value if eq else None
    hits = [o for o in options if o.startswith(name)]
    return (hits[0], value if eq else None) if len(hits) == 1 else (None, None)


def parse_options(args: list, options: dict[str, bool]) -> list[tuple[str, str | None, str]]:
    """[(option, value, token)] of the resolved options in `args` (a value option takes `=value` or the next token)."""
    out, i = [], 0
    while i < len(args):
        opt, value = resolve_option(args[i], options)
        tok = args[i]
        if opt is not None and options[opt] and value is None:
            value = args[i + 1] if i + 1 < len(args) and isinstance(args[i + 1], str) else ""
            i += 1
        if opt is not None:
            out.append((opt, value, tok))
        i += 1
    return out


def runtime_argv_problems(argv: Any, *, profile: str | None, options_of: dict[str, dict[str, bool]],
                          eval_script: str, default_base_rel: str,
                          rule: str) -> tuple[list[str], str | None, list[tuple[str, str | None, str]]]:
    """EXT-133, EXT-150, EXT-156, for runtimes whose argv[0] is `<runtime>/.venv/bin/python`
    (the registry's pinned interpreter): (problems, script name, its parsed options).

    - argv[1] is `<runtime>/<script>` for one of the runtime's scripts (a copy elsewhere, even byte-identical, would
      bring its own default base and items);
    - the evaluation script: `--base` and `--items` are absent or the script's defaults (`<runtime>/<default base>`,
      `<runtime>/items`): the evaluator hashes neither; `--limit` is refused for `--profile full` (a gate arm covers
      the registered item set; the smoke baseline is the first 20 items)."""
    if not (isinstance(argv, list) and len(argv) > 1 and isinstance(argv[0], str) and isinstance(argv[1], str)):
        return [], None, []
    root = posixpath.dirname(posixpath.dirname(posixpath.dirname(argv[0])))
    script = posixpath.basename(argv[1])
    allowed = sorted(posixpath.join(root, s) for s in options_of)
    if script not in options_of or argv[1] != posixpath.join(root, script):
        return [f"argv[1] {argv[1]!r} is not one of the runtime's scripts {allowed} (the directory of the pinned "
                f"interpreter {argv[0]}): a copy elsewhere brings its own default base and items ({rule})"], None, []
    parsed = parse_options(argv[2:], options_of[script])
    out = []
    if script == eval_script:
        defaults = {"--base": posixpath.join(root, default_base_rel), "--items": posixpath.join(root, "items")}
        for opt, value, _ in parsed:
            if opt in defaults and value != defaults[opt]:
                out.append(f"{eval_script} {opt} {value!r} is not its default {defaults[opt]}: the evaluator does not "
                           f"hash it, so an arm is bound to the registered {opt[2:]} only through the default inside "
                           f"the read-only runtime mount ({rule})")
            if opt == "--limit" and profile == "full":
                out.append(f"{eval_script} --limit {value!r} under --profile full: a gate arm evaluates the whole "
                           f"registered item set; --limit is for the smoke run ({rule})")
    return out, script, parsed


# --- mounts (the launch record's read-only mounts as ext-launch resolves them) ---

def ro_table(mounts: Any) -> list[tuple[str, str]]:
    """(container path, strict host realpath) of each well-formed read-only mount whose source resolves."""
    from .launch import _ro_table

    return _ro_table(mounts)


def within(path: str, root: str) -> bool:
    """Whether `path` is `root` or lies under it (lexical, POSIX paths)."""
    from .launch import _within

    return _within(path, root)


def container_file(path: Any, mounts: Any) -> tuple[str, str] | None:
    """EXT-91: (resolved container path, host file) of the absolute container path `path` as the container resolves
    it through the read-only mounts; None when it is no regular file in a read-only mount (`launch.container_file`)."""
    from .launch import container_file as _container_file

    return _container_file(path, mounts)
