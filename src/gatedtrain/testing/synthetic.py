"""A synthetic external-run adapter, for gatedtrain's own tests and for third-party adapter authors.

It drives no real framework and is never registered by an installed distribution: a test registers it explicitly,
in process with `install()`, or for a CLI subprocess with `python -m gatedtrain.testing.synthetic <gatedtrain arguments>`
(registers it, then runs `gatedtrain.cli.main`). gatedtrain's suite runs ext-launch, ext-arm-fill, ext-from-nel and ext-collect
through it, so the framework rules are tested without any recipe installed. A recipe author can read it as a complete,
small example of the adapter interface (MM-12) next to `gatedtrain.ext.conformance`.

The run it reads (`<stage-out>`, the one read-write mount `/out` of the launch):

- `checkpoint-<N>/`: a model directory (`config.json`, `tokenizer_config.json`, whole safetensors weights:
  `model.safetensors`, or `model.safetensors.index.json` and its shards) with `trainer_state.json` written last,
  `global_step` = N (the Hugging Face Trainer layout). A directory without it is an incomplete cut;
- `run_args.json`, the launcher's record of the run's arguments: `model_name_or_path` (the base mount's container
  path), `seed`, `profile`, `dry_run`, `max_steps`, `train_file` (`<export mount>/train.jsonl`), `train_file_sha256`
  (the SHA-256 of the training shards concatenated in file-name order), `rows_per_step` and `shuffle`;
- `samples.jsonl` (optional): one line `{step, row}` per training row each optimizer step consumed; in a resumed
  run (`attempts.jsonl` present) each line also carries the `attempt` that logged it;
- `env-freeze.txt` (`name==version` lines, `synthetic-trainer` among them) and `env-runtime.json` (`script_sha256`).

Consumption: with `shuffle` false and a positive integer `rows_per_step` P, optimizer step s read rows P(s-1) .. Ps-1
of the training file, an epoch dropping its last short chunk and restarting at row 0: precision `iteration`, mapped to
shards as MM-22 (`ledger.step_shards`), derivation reconstructed; a present `samples.jsonl` must log the same rows.
Otherwise precision `stage` with the reason. The evaluator `synthetic.items@0.1` reads NeMo Evaluator GSM8K results
(`ext-from-nel`), scored by `numeric_match`.

This module imports no framework (MM-12).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from ..canonical import entries_hash, file_sha256, manifest_entries, manifest_hash, text_lines
from ..errors import Problem, Refused
from ..ext.rowkit import CONTAINER_PATHS, HF_LLM_COMPONENTS
from ..ext.sdk import (bound_mounts, plain_file, read_run_args, runtime_script_problems, safetensors_problem,
                       shards_concat_sha256)

NAME = "synthetic"
TRAINER_ID = "rl.synthetic@0.1"
EVALUATOR_ID = "synthetic.items@0.1"
FRAMEWORK_PACKAGE = "synthetic-trainer"
# Placeholder host paths: a test names its own layout with the GATEDTRAIN_EXT_* overrides (input roots, interpreter and
# interpreter trees); nothing is read under these.
ROOT = "/srv/gatedtrain-synthetic"

CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")
TRAINER_STATE = "trainer_state.json"
REQUIRED_FILES = ("config.json", "tokenizer_config.json")
SINGLE_WEIGHTS, WEIGHTS_INDEX = "model.safetensors", "model.safetensors.index.json"
FREEZE_FILE, RUN_ARGS, SAMPLES = "env-freeze.txt", "run_args.json", "samples.jsonl"
EXPORT_TRAIN = "train.jsonl"     # `<export>/train.jsonl`: the shards of `<export>/train/` concatenated (export-gsm8k)

TRAINER_ROWS: dict[str, dict[str, Any]] = {NAME: {
    "uses": (TRAINER_ID,),
    "pins": (FRAMEWORK_PACKAGE,),
    "outputs": ("checkpoint-<step>/ model directories with trainer_state.json written last", "run_args.json",
                "samples.jsonl"),
    "components": HF_LLM_COMPONENTS,
    "consumption": {"target": "iteration", "fallback": "stage"},
    "gpu_h_estimate": 2.5,
    "evaluators": (EVALUATOR_ID,),
    "framework": FRAMEWORK_PACKAGE,
    "input_roots": (ROOT,),
    "spec_binding": True,                            # the spec's base and export are bound to read-only mounts
    # Names an LLM trainer with a colocated vLLM engine typically sets; none is a credential.
    "env_names": ("DO_NOT_TRACK", "FLASHINFER_WORKSPACE_BASE", "HF_HOME", "HF_HUB_OFFLINE", "HOME", "MASTER_ADDR",
                  "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED", "TOKENIZERS_PARALLELISM", "TORCHINDUCTOR_CACHE_DIR",
                  "TORCH_CUDA_ARCH_LIST", "TRANSFORMERS_OFFLINE", "TRITON_CACHE_DIR", "VLLM_CACHE_ROOT",
                  "VLLM_CONFIG_ROOT", "VLLM_NO_USAGE_STATS", "VLLM_USE_FLASHINFER_SAMPLER", "XDG_CACHE_HOME"),
    "scan_skip": (f"{ROOT}/.venv",),
    "interpreter_trees": (f"{ROOT}/python",),
    "interpreter": f"{ROOT}/.venv/bin/python",
    "container_paths": CONTAINER_PATHS,
    "env_paths": ("FLASHINFER_WORKSPACE_BASE", "HF_HOME", "HOME", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
                  "VLLM_CACHE_ROOT", "VLLM_CONFIG_ROOT", "XDG_CACHE_HOME"),
}}

# NeMo Evaluator's numeric_match at ea35ba3 (the Tier B scorer); code_sha256 is the SHA-256 of
# src/nemo_evaluator/scoring/pattern.py in that commit.
EVALUATOR_ROWS: dict[str, dict[str, Any]] = {EVALUATOR_ID: {
    "metric": "accuracy",
    "direction": "higher_is_better",
    "score_range": (0, 1),
    "scorer": {"name": "numeric_match", "version": "nemo-evaluator@ea35ba3", "weights_sha256": None,
               "code_sha256": "sha256:594c235cb9158a6d175b4f00ac6cd9db850b27c41e97e6c01257c6ffbb2b62a5"},
    "tier": "T3",
    "binary": True,
    "error_classes": ("scorer_error", "missing_score"),
    "score_outcomes": ("wrong_number", "no_number_in_response"),
    "payload_text_field": "question",
    "item_payload": ("question", "answer"),
    "normalizer": False,
    "env_version": False,
    "framework": "nemo-evaluator",
    "nel_benchmark": "gsm8k",
    "samples": "30, 2",
}}

PLUGIN = {"adapters": {NAME: "gatedtrain.testing.synthetic:SyntheticAdapter"}, "trainer_rows": TRAINER_ROWS,
          "evaluator_rows": EVALUATOR_ROWS}


def install() -> None:
    """Register the synthetic adapter in this process (idempotent). Call it before a command reads the registry; the
    spec and item tables that copy the registry at import are refreshed."""
    from .. import items, spec
    from ..ext import registry

    if registry.ADAPTERS.get(NAME) != PLUGIN["adapters"][NAME]:
        registry.register(PLUGIN, "gatedtrain.testing.synthetic.PLUGIN")
    spec._external_ids()
    items.EVALUATORS.update(items._external_evaluators())


def _pos_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x > 0


def _component(name: str) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in HF_LLM_COMPONENTS)


def weights_problem(d: Path) -> str | None:
    """None when the directory's weights are whole: model.safetensors, or an index whose every shard is whole."""
    single, index = d / SINGLE_WEIGHTS, d / WEIGHTS_INDEX
    if not plain_file(single) and not plain_file(index):
        return f"no {SINGLE_WEIGHTS} and no {WEIGHTS_INDEX}"
    if plain_file(single) and (bad := safetensors_problem(single)):
        return bad
    if plain_file(index):
        try:
            shards = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            return f"{WEIGHTS_INDEX} does not parse as {{weight_map}} ({exc.__class__.__name__})"
        if not shards:
            return f"{WEIGHTS_INDEX} names no shard"
        for name in shards:
            if not isinstance(name, str) or "/" in name or name in (".", ".."):
                return f"{WEIGHTS_INDEX} names {name!r}, not a file of the directory"
            if bad := safetensors_problem(d / name):
                return bad
    return None


def step_rows(per_step: int, total: int, last: int) -> dict[int, range]:
    """Training-file rows of optimizer steps 1 .. last: P consecutive rows each, an epoch dropping its last short chunk
    and restarting at row 0 (the cursor rule of `ledger.step_shards`)."""
    out, cursor = {}, 0
    for step in range(1, last + 1):
        if cursor + per_step > total:
            cursor = 0
        out[step] = range(cursor, cursor + per_step)
        cursor += per_step
    return out


def read_sample_records(path: Path) -> tuple[list[dict] | None, str | None]:
    """(the lines of samples.jsonl as objects with integer `step` and `row`, None), or (None, why it cannot be read).
    A resumed run's lines also carry the integer `attempt` that logged them."""
    if not plain_file(path):
        return None, f"{path.name} is not a plain file"
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return None, f"{path.name} cannot be read ({exc.__class__.__name__})"
    out: list[dict] = []
    for n, line in enumerate(text_lines(text), start=1):
        if not line.strip():
            continue
        try:
            doc = json.loads(line)
        except (ValueError, RecursionError):
            return None, f"{path.name} line {n} does not parse"
        step, row = (doc.get("step"), doc.get("row")) if isinstance(doc, dict) else (None, None)
        if not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in (step, row)):
            return None, f"{path.name} line {n} has no integer `step` and `row`"
        out.append(doc)
    return out, None


def logged_rows(records: list[dict]) -> dict[int, set[int]]:
    """{step: rows logged} of records from `read_sample_records`."""
    logged: dict[int, set[int]] = {}
    for doc in records:
        logged.setdefault(doc["step"], set()).add(doc["row"])
    return logged


def read_samples(path: Path) -> tuple[dict[int, set[int]] | None, str | None]:
    """({step: rows logged}, None) from samples.jsonl, or (None, why it cannot be read)."""
    records, why = read_sample_records(path)
    return (None, why) if records is None else (logged_rows(records), None)


class SyntheticAdapter:
    """The MM-12 interface over the run layout of the module docstring."""

    name = NAME
    trainer_id = TRAINER_ID
    evaluator_ids = (EVALUATOR_ID,)
    framework_name = FRAMEWORK_PACKAGE
    binds_spec = True
    checkpoint_completeness = {
        "library": "synthetic (Hugging Face Trainer checkpoint layout)",
        "rule": ("a checkpoint-<N> directory is a cut iff config.json, tokenizer_config.json and whole safetensors "
                 "weights exist and trainer_state.json, written last, parses with global_step = N"),
        "complete_dir_pattern": CHECKPOINT_DIR_RE.pattern,
        "sources": ["gatedtrain/testing/synthetic.py (module docstring)"],
        "verified": False,
    }
    seed_derivation = ("the training seed is read from <stage-out>/run_args.json `seed` and must equal the launch "
                       "record's training_seed (synthetic adapter)")
    BASE_FIX = ("train from the spec's base directory mounted read-only, loading it by its container path (recorded in "
                "run_args.json), and record that mount in the launch record")
    DATA_FIX = ("train on the spec's export mounted read-only, reading its train.jsonl (recorded in run_args.json), "
                "and record that mount in the launch record")

    def metric(self, evaluator_id: str) -> dict[str, Any]:
        if evaluator_id not in self.evaluator_ids:
            raise KeyError(evaluator_id)
        row = EVALUATOR_ROWS[evaluator_id]
        return {"metric": row["metric"], "direction": row["direction"], "score_range": tuple(row["score_range"]),
                "error_classes": tuple(row["error_classes"]), "score_outcomes": tuple(row["score_outcomes"]),
                "scorer": dict(row["scorer"]), "binary": row["binary"], "tier": row["tier"]}

    # --- cuts and identity (MM-17, MM-20) ---

    def model_dir_problem(self, d: Path) -> str | None:
        d = Path(d)
        if d.is_symlink() or not d.is_dir():
            return f"{d} is not a plain directory"
        for f in REQUIRED_FILES:
            if not plain_file(d / f):
                return f"{f} is missing"
        return weights_problem(d)

    def cut_problem(self, d: Path, step: int) -> str | None:
        if bad := self.model_dir_problem(d):
            return bad
        state = Path(d) / TRAINER_STATE
        if not plain_file(state):
            return f"{TRAINER_STATE} (written last) is missing"
        try:
            doc = json.loads(state.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return f"{TRAINER_STATE} does not parse ({exc.__class__.__name__})"
        gs = doc.get("global_step") if isinstance(doc, dict) else None
        if not (isinstance(gs, int) and not isinstance(gs, bool) and gs == step):
            return f"{TRAINER_STATE} global_step is {gs!r}, not {step}"
        return None

    def checkpoint_dirs(self, stage_out: Path) -> list[tuple[int, Path]]:
        stage_out = Path(stage_out)
        if not stage_out.is_dir():
            return []
        return sorted((int(m.group(1)), p) for p in stage_out.iterdir()
                      if (m := CHECKPOINT_DIR_RE.match(p.name)) and p.is_dir() and not p.is_symlink())

    def find_cuts(self, stage_out: Path) -> tuple[list[tuple[int, Path]], list[Path]]:
        complete, incomplete = [], []
        for step, d in self.checkpoint_dirs(stage_out):
            (complete.append((step, d)) if self.cut_problem(d, step) is None else incomplete.append(d))
        return complete, incomplete

    def identity(self, cut: Path, base: Any = None, *, entries: list[dict] | None = None) -> dict[str, Any]:
        """`cut_hash` the manifest hash of the whole directory, `model_hash` that of its top-level deployable
        components (`HF_LLM_COMPONENTS`); full weights, so `base_ref` is null."""
        entries = manifest_entries(Path(cut)) if entries is None else entries
        comps = [e for e in entries if "/" not in e["path"] and _component(e["path"])]
        return {"model_hash": entries_hash(comps), "cut_hash": entries_hash(entries),
                "components": [e["path"] for e in comps], "base_ref": None}

    # --- consumption (MM-22, MM-23) ---

    def consumption(self, stage_out: Path, data: Path, cuts: list, *, run_id: str = "",
                    attempt_id: str = "0", chain: list | None = None) -> dict[str, Any]:
        """The ledger rows of the run. With `chain` (the segments of a resumed run, `gatedtrain.ext.attempts`) the
        samples.jsonl lines, each tagged with its attempt, are split into kept and orphaned records (EXT-RES-4);
        the result then also lists `orphaned` rows (one per orphaned (attempt, step, shard)); `rows` are the kept
        ones, each with its segment's `attempt_id` and `controller_epoch` (EXT-RES-1, EXT-RES-5)."""
        from ..ledger import LEDGER_DERIVATION, WRITER_ID, check_precision, shard_table, step_shards

        def stage(why: str) -> dict[str, Any]:
            return {"precision": "stage", "rows": [], "orphaned": [], "derivation": None,
                    "reason": f"{why}; the row order is not shown, so every cut possibly consumed every shard (MM-23)"}

        args, why = read_run_args(stage_out)
        if args is None:
            return stage(why)
        per_step = args.get("rows_per_step")
        if args.get("shuffle") is not False:
            return stage(f"{RUN_ARGS} does not show shuffle false ({args.get('shuffle')!r})")
        if not _pos_int(per_step):
            return stage(f"{RUN_ARGS} has no positive integer `rows_per_step` ({per_step!r})")
        recorded, actual = args.get("train_file_sha256"), shards_concat_sha256(data)
        if recorded != actual:
            return stage(f"{RUN_ARGS} train_file_sha256 {recorded!r} is not the SHA-256 of the spec's training shards "
                         f"concatenated in file-name order ({actual}), so its row numbers are not this snapshot's")
        if not cuts:
            return stage("no complete cut")
        last = max(step for step, _ in cuts)
        table = shard_table(data)
        try:
            mapping = step_shards([(s, per_step) for s in range(1, last + 1)], table)
        except ValueError as exc:
            return stage(f"the rows per step do not fit the training data ({exc})")
        samples = Path(stage_out) / SAMPLES
        kept_steps: dict[int, Any] = {}               # step -> the segment that keeps it
        orphan_steps: set[tuple[int, int]] = set()     # (attempt, step)
        if chain is not None:
            from ..ext.attempts import split_steps

            by_attempt = {sg.attempt: sg for sg in chain}
            records = None
            if samples.exists() or samples.is_symlink():
                records, why = read_sample_records(samples)
                if records is None:
                    return stage(f"{why}, so the reconstructed rows cannot be cross-checked")
            elif len(chain) > 1:
                raise Refused([Problem("EXT-RES-4", SAMPLES, f"{SAMPLES} is missing, so the records of the attempts "
                                       "cannot be split into kept and orphaned",
                                       "collect a resumed run with the records of every attempt")])
            if records is not None:
                kept, orphaned, problems = split_steps(chain, records)
                if problems:
                    raise Refused(problems)
                for doc in kept:
                    kept_steps[doc["step"]] = by_attempt[doc["attempt"]]
                orphan_steps = {(doc["attempt"], doc["step"]) for doc in orphaned}
                for step, want in step_rows(per_step, sum(n for _, n in table), last).items():
                    got = {doc["row"] for doc in kept if doc["step"] == step}
                    if got != set(want):
                        return stage(f"{SAMPLES} logs rows {sorted(got)[:8]} for step {step}, not the reconstructed "
                                     f"rows {want.start} .. {want.stop - 1}")
            else:
                kept_steps = {s: chain[0] for s in range(1, last + 1)}
        elif samples.exists() or samples.is_symlink():
            logged, why = read_samples(samples)
            if logged is None:
                return stage(f"{why}, so the reconstructed rows cannot be cross-checked")
            for step, want in step_rows(per_step, sum(n for _, n in table), last).items():
                if (got := logged.get(step, set())) != set(want):
                    return stage(f"{SAMPLES} logs rows {sorted(got)[:8]} for step {step}, not the reconstructed rows "
                                 f"{want.start} .. {want.stop - 1}")
        check_precision("iteration", LEDGER_DERIVATION)
        snapshot = manifest_hash(data)

        def row(step: int, shard: str, att: str, epoch: int | None, status: str | None) -> dict[str, Any]:
            sample_id = hashlib.sha256(f"{run_id}\x00{att}\x00{step}\x00{shard}".encode()).hexdigest()
            return {
                "run_id": run_id, "attempt_id": att, "step": step, "sample_id": "sha256:" + sample_id,
                "data_ref": {"snapshot": snapshot, "shard": shard, "row": None, "mixture": None, "component": None},
                "parent_ref": None, "outcome": "consumed", "reason": None, "rule_version": None,
                "gen_version": None, "reward_epoch": None, "rewards": None, "sampler_config_hash": None,
                "seed_inputs": None, "chunk": None, "precision": "iteration", "derivation": LEDGER_DERIVATION,
                "writer_id": WRITER_ID, "controller_epoch": epoch, "status": status}

        rows, orphaned_rows = [], []
        for step in range(1, last + 1):
            for shard in mapping[step]:
                if chain is None:
                    rows.append(row(step, shard, attempt_id, None, None))
                else:
                    sg = kept_steps[step]
                    rows.append(row(step, shard, sg.attempt_id, sg.attempt, "kept"))
        for a, step in sorted(orphan_steps):
            sg = {x.attempt: x for x in chain}[a]
            orphaned_rows += [row(step, shard, sg.attempt_id, sg.attempt, "orphaned") for shard in mapping[step]]
        return {"precision": "iteration", "rows": rows, "orphaned": orphaned_rows, "derivation": LEDGER_DERIVATION,
                "reason": (f"{RUN_ARGS} shows shuffle false and {per_step} rows per optimizer step, so step s read rows "
                           "P(s-1) .. Ps-1 of the training file in file order; shards mapped as MM-22, derivation "
                           "reconstructed" + (f"; {SAMPLES} logs the same rows per step" if samples.exists() else ""))}

    # --- pins and binding checks (MM-33, MM-21) ---

    def framework_identity(self, stage_out: Path, launch: dict) -> dict[str, Any]:
        freeze = Path(stage_out) / FREEZE_FILE
        if not plain_file(freeze):
            raise ValueError(f"{freeze} is missing")
        versions = {}
        for line in text_lines(freeze.read_text(encoding="utf-8")):
            line = line.strip()
            if "==" in line and not line.startswith("#"):
                name, _, version = line.partition("==")
                versions[name.strip().lower().replace("_", "-")] = version.strip()
        version = versions.get(FRAMEWORK_PACKAGE)
        if version is None:
            raise ValueError(f"{freeze} has no `{FRAMEWORK_PACKAGE}==<version>` line")
        fw = launch.get("framework") if isinstance(launch.get("framework"), dict) else {}
        if fw.get("name") != FRAMEWORK_PACKAGE or fw.get("version") != version:
            raise ValueError(f"{freeze} pins {FRAMEWORK_PACKAGE}=={version}, but the launch record claims framework "
                             f"{fw.get('name')!r} {fw.get('version')!r}")
        return {"name": FRAMEWORK_PACKAGE, "version": version, "freeze_sha256": file_sha256(freeze)}

    def script_problems(self, stage_out: Path, launch: dict) -> list[Problem]:
        return runtime_script_problems(stage_out, launch)

    @staticmethod
    def export_dir(data: Path) -> Path:
        """The export directory: the parent of the spec's training shards (`<export>/train/`)."""
        return Path(data).parent

    def mount_binding_problems(self, mounts: Any, *, base: Path, data: Path, base_field: str, data_field: str,
                               flag: str = "--launch-record") -> list[Problem]:
        """A read-only mount whose host path is the spec's base, and one whose host path is its export directory."""
        out = []
        if not bound_mounts(mounts, base):
            out.append(Problem("EXT-BASE", base_field, f"{flag} mounts no read-only directory whose host path is the "
                               f"spec's base {base}", self.BASE_FIX))
        export = self.export_dir(data)
        if not bound_mounts(mounts, export):
            out.append(Problem("EXT-DATA", data_field, f"{flag} mounts no read-only directory whose host path is the "
                               f"spec's export directory {export} (the parent of the training shards, holding "
                               f"{EXPORT_TRAIN}) (EXT-65)", self.DATA_FIX))
        return out

    def checks(self, *, last_cut: Path, base: Path, launch: dict, base_field: str,
               stage_out: Path | None = None, data: Path | None = None, data_field: str = "data") -> list[Problem]:
        """run_args.json binds the run to the launch record: `model_name_or_path` is the base mount's container path
        (EXT-20), `seed` the record's training_seed (EXT-50), `train_file` the export mount's train.jsonl and
        `train_file_sha256` the spec's shards (EXT-65), `profile` the record's, `dry_run` false and `max_steps` the
        last cut's step (EXT-68)."""
        mounts = bound_mounts(launch.get("mounts"), base)
        out: list[Problem] = []
        if not mounts:
            out.append(Problem("EXT-BASE", base_field, f"the launch record mounts no read-only directory whose host "
                               f"path is the spec's base {base}", self.BASE_FIX))
        exports = bound_mounts(launch.get("mounts"), self.export_dir(data)) if data is not None else []
        if data is not None and not exports:
            out += [p for p in self.mount_binding_problems(launch.get("mounts"), base=base, data=data,
                                                           base_field=base_field, data_field=data_field)
                    if p.rule == "EXT-DATA"]
        args, why = read_run_args(Path(stage_out) if stage_out is not None else Path(last_cut).parent)
        if args is None:
            out.append(Problem("EXT-BASE", base_field, f"{why}, so the trained model cannot be bound to the hashed "
                               "base directory (`model_name_or_path`, EXT-20)", self.BASE_FIX))
            if data is not None:
                out.append(Problem("EXT-DATA", data_field, f"{why}, so the training file cannot be bound to the "
                                   "spec's training shards (EXT-65)", self.DATA_FIX))
            return out
        if data is not None:
            out += self._data_problems(args, data, exports, data_field)
        out += self._run_problems(args, launch, last_cut)
        got = args.get("model_name_or_path")
        want = sorted({m.get("container") for m in mounts if isinstance(m.get("container"), str)})
        if mounts and not isinstance(got, str):
            out.append(Problem("EXT-BASE", base_field, f"{RUN_ARGS} has no string `model_name_or_path` ({got!r}), so "
                               "the trained model cannot be bound to the hashed base directory", self.BASE_FIX))
        elif mounts and got not in want:
            out.append(Problem("EXT-BASE", base_field, f"{RUN_ARGS} `model_name_or_path` is {got!r}, not the base "
                               f"mount's container path {want} (a Hub id or another directory cannot be gated against "
                               "this baseline)", self.BASE_FIX))
        seed, claimed = args.get("seed"), launch.get("training_seed")
        if not (isinstance(seed, int) and not isinstance(seed, bool) and seed == claimed):
            out.append(Problem("EXT-LAUNCH", "--launch-record.training_seed", f"the launch record claims training_seed "
                               f"{claimed!r}, but {RUN_ARGS} `seed` is {seed!r} (MM-29, EXT-50)",
                               "launch with the seed the launch record names (ext-launch argv and request)"))
        return out

    def _data_problems(self, args: dict, data: Path, exports: list[dict], field: str) -> list[Problem]:
        out = []
        got = args.get("train_file")
        want = sorted({f"{m['container'].rstrip('/')}/{EXPORT_TRAIN}" for m in exports
                       if isinstance(m.get("container"), str)})
        if exports and got not in want:
            out.append(Problem("EXT-DATA", field, f"{RUN_ARGS} `train_file` is {got!r}, not the export mount's "
                               f"{want}: the run read another training file (EXT-65)", self.DATA_FIX))
        recorded = args.get("train_file_sha256")
        if not isinstance(recorded, str):
            return out + [Problem("EXT-DATA", field, f"{RUN_ARGS} has no `train_file_sha256` ({recorded!r}); the "
                                  "launcher always records the hash of the file it read (EXT-65)", self.DATA_FIX)]
        try:
            actual = shards_concat_sha256(data)
        except (OSError, ValueError) as exc:
            return out + [Problem("EXT-DATA", field, f"cannot hash the training shards ({exc.__class__.__name__})",
                                  "keep the training dataset readable")]
        if recorded != actual:
            out.append(Problem("EXT-DATA", field, f"{RUN_ARGS} `train_file_sha256` {recorded} is not the SHA-256 of "
                               f"the spec's training shards concatenated in file-name order ({actual}): the run trained "
                               "on another file, so neither the ledger nor the contamination count would be this run's "
                               "(EXT-65)",
                               "collect with the spec whose training data the run read, or train on the spec's export"))
        return out

    def _run_problems(self, args: dict, launch: dict, last_cut: Path) -> list[Problem]:
        out = []
        if args.get("profile") != launch.get("profile"):
            out.append(Problem("EXT-PROFILE", "--launch-record.profile", f"{RUN_ARGS} `profile` is "
                               f"{args.get('profile')!r}, the launch record's {launch.get('profile')!r} (EXT-68)",
                               "collect the run launched with this record's profile"))
        if args.get("dry_run") is not False:
            out.append(Problem("EXT-STAGE", "--stage-out", f"{RUN_ARGS} `dry_run` is {args.get('dry_run')!r}, not "
                               "false: a dry run trains nothing (EXT-68)", "collect a training run, not a dry run"))
        max_steps = args.get("max_steps")
        m = CHECKPOINT_DIR_RE.match(Path(last_cut).name)
        last = int(m.group(1)) if m else None
        if not (isinstance(max_steps, int) and not isinstance(max_steps, bool) and max_steps == last):
            out.append(Problem("EXT-STAGE", "--stage-out", f"the last cut is step {last}, but {RUN_ARGS} `max_steps` "
                               f"is {max_steps!r}: the run did not save its last step (EXT-68)",
                               "collect a run that finished and saved its last step"))
        return out


def main(argv: list[str] | None = None) -> int:
    """`python -m gatedtrain.testing.synthetic <gatedtrain arguments>`: the gatedtrain CLI with the synthetic adapter."""
    install()
    from ..cli import main as cli_main

    return cli_main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    from gatedtrain.testing.synthetic import main as _main     # register the importable module's rows, not __main__'s

    sys.exit(_main())
