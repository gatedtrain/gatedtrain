"""Collect external training and evaluation outputs into a run directory.

The collector validates the spec and registration, item set, evaluation arms,
launch and smoke records, and GPU log. Adapters validate checkpoints and
report checkpoint identity and data consumption. Optional adapter hooks
cover recipe-specific inputs, evaluator metadata and report details.

The collector verifies the provenance of merged arms and builds the input
manifests before writing. Invalid artifacts produce a refusal (exit 3).
Evaluator errors and item or evaluation identity mismatches that affect the
gate are recorded in its report. A failed write removes the partial run.

The output includes the gate report, consumption ledger, checkpoint lineage
and lockfile. When a registration contains multiple clauses, pass one
baseline and candidate arm per clause, in registration order. Each clause
uses its own evaluator and the report also gives the combined verdict.

See `MULTIMODAL-SPEC.md` and `tests/README.md` for the collection contract.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .. import ANALYSIS_CODE_VERSION, TAGS
from ..canonical import (canonical_json, entries_hash, file_sha256, manifest_entries, manifest_hash, sha256_bytes,
                         text_lines, write_canonical)
from ..errors import Problem, Refused
from ..paths import resolve
from . import registry
from ..gates import _cluster_value_ok as cluster_value_ok
from .arm import gate_record, meta_differences, provenance_problems, read_lines, read_meta
from .launch import first_start, gpu_seconds, gpu_step_log_problem, record_problems
from .merge import write_problem
from .smoke import smoke_problems

TRAINER_CONFIG_KEYS = ("backend", "profile")
PROFILES = ("smoke", "full")
# EXT-64: stage outputs besides the cuts that ext-collect reads; each must postdate the label's first START.
# env-runtime.json also binds the script used by the run (EXT-83).
READ_OUTPUTS = ("run_args.json", "env-freeze.txt", "samples.jsonl", "env-runtime.json",
                "rollouts.jsonl",                                       # EXT-102 (sb3_ppo)
                "train_steps.jsonl",                                    # EXT-123 (ddpo_sd15)
                "epoch_plan.jsonl",                                     # EXT-165 (flow_grpo_sd35)
                "env_calls.jsonl")                                      # EXT-186 (nemo_rl_ext)


def stale_outputs(stage_out: Path, cut_dirs: list[Path], started: float, extra: list[Path] = ()) -> list[str]:
    """EXT-64: files of the cuts, of READ_OUTPUTS and `extra` (an adapter's further logs) whose modification time (not
    following links) is before `started`, the label's first START: written before this launch, so not this run's
    outputs."""
    import os

    files = [Path(stage_out) / n for n in READ_OUTPUTS if os.path.lexists(Path(stage_out) / n)]   # EXT-154: no raise
    files += [Path(p) for p in extra]
    for d in cut_dirs:
        for top, _, names in os.walk(d):
            files += [Path(top) / n for n in names]
    out = []
    for p in sorted(files):
        try:
            if os.lstat(p).st_mtime < started:
                out.append(str(p.relative_to(stage_out)))
        except OSError:
            out.append(f"{p} (unreadable)")
    return out


def _guarded(fn, rule: str, field: str) -> list[Problem]:
    """Report an unreadable adapter input as a refusal (EXT-154)."""
    try:
        return fn()
    except OSError as exc:
        return [Problem(rule, field, f"a run file cannot be read ({exc.__class__.__name__}: "
                        f"{Path(exc.filename).name if exc.filename else exc})",
                        "make the run's outputs readable for the host user")]


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _trainer_config(path: Path, field: str, adapter: str) -> tuple[dict, list[Problem]]:
    """EXT-2 (design choice 14): `{backend: <adapter>, profile: smoke | full}`, nothing else."""
    fix = f"write the trainer config as `{{backend: {adapter}, profile: smoke | full}}`"
    try:
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return {}, [Problem("TRAINER-CONFIG", field, f"cannot read {path}: {exc.__class__.__name__}", fix)]
    if not isinstance(cfg, dict):
        return {}, [Problem("TRAINER-CONFIG", field, "the trainer config is not a mapping", fix)]
    out = [Problem("TRAINER-CONFIG", f"{field}.{k}", "unknown trainer config key", fix)
           for k in sorted(set(cfg) - set(TRAINER_CONFIG_KEYS))]
    if cfg.get("backend") != adapter:
        out.append(Problem("TRAINER-CONFIG", f"{field}.backend", f"backend {cfg.get('backend')!r} is not the adapter "
                           f"{adapter!r}", fix))
    if cfg.get("profile") not in PROFILES:
        out.append(Problem("TRAINER-CONFIG", f"{field}.profile", f"profile {cfg.get('profile')!r} is not smoke or full",
                           fix))
    return cfg, out


def read_item_set(items_dir: Path, payload: tuple[str, ...],
                  objects: tuple[str, ...] = ()) -> tuple[list[dict], list[str]]:
    """MM-6 (EXT-5): `<items>/items.jsonl`, one object per item with a unique non-empty string `item` and the
    evaluator's payload fields: strings, except the fields named in `objects`, which are JSON objects (EXT-98: an
    environment item's initial state). Returns (rows in file order, problems)."""
    path = Path(items_dir) / "items.jsonl"
    if not path.is_file():
        return [], [f"{path} is missing"]
    rows, bad, seen = [], [], {}
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [], [f"{path} cannot be read as UTF-8 text ({exc.__class__.__name__})"]
    for n, line in enumerate(text_lines(text), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            bad.append(f"line {n} does not parse")
            continue
        item = row.get("item") if isinstance(row, dict) else None
        if not (isinstance(item, str) and item):
            bad.append(f"line {n}: item {item!r} is not a non-empty string")
            continue
        if item in seen:
            bad.append(f"line {n}: item {item!r} repeats line {seen[item]}")
            continue
        missing = [f for f in payload if not isinstance(row.get(f), dict if f in objects else str)]
        if missing:
            bad.append(f"line {n}: item {item!r} has no string payload {missing}")
            continue
        seen[item] = n
        rows.append(row)
    if not rows and not bad:
        bad.append(f"{path} holds no items")
    return rows, bad[:8] + ([f"{len(bad) - 8} more"] if len(bad) > 8 else [])


def exact_overlap(item_rows: list[dict], data: Path, field: str) -> int:
    """MM-11: item-set rows whose payload text field equals a training row's field (exact text). Raises
    ValueError for a training line that does not parse."""
    from ..ledger import shard_table

    train: set[str] = set()
    for name, _ in shard_table(data):
        for n, line in enumerate(text_lines((Path(data) / name).read_text(encoding="utf-8")), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"{name}:{n} does not parse") from None
            if isinstance(row, dict) and isinstance(row.get(field), str):
                train.add(row[field])
    return sum(1 for r in item_rows if r.get(field) in train)


def write_lineage(wf, run_dir: Path, *, base_hash: str, base_model_hash: str, data: Path, cfg_path: Path,
                  cuts: list[dict], item_set_hash: str, precision: str, snapshot: str | None = None) -> None:
    """Lineage for the impact query (MM-17, MM-30; EXT-15): checkpoint rows carry both hashes; the training
    stage `used` the base (manifest and model hash) and the data snapshot at the observed precision; the gate
    `used` model hashes (05 section 3.3, R11-14)."""
    from ..ledger import ORPHAN_SCHEMA, shard_table

    run_id, stage_id, gate_id = run_dir.name, wf.stage["id"], wf.gate["id"]
    snapshot = snapshot if snapshot is not None else manifest_hash(data)
    ckpt_rows = [{"run_id": run_id, "stage_id": stage_id, "checkpoint_hash": c["cut_hash"], "model_hash": c["model_hash"],
                  "last_step_included": c["step"], "path": str(c["path"])} for c in cuts]
    shard_rows = [{"run_id": run_id, "stage_id": stage_id, "snapshot": snapshot, "shard": name,
                   "shard_sha256": file_sha256(data / name)} for name, _ in shard_table(data)]
    (run_dir / "lineage").mkdir()
    pq.write_table(pa.Table.from_pylist(ckpt_rows), run_dir / "lineage" / "checkpoints.parquet")
    pq.write_table(pa.Table.from_pylist(shard_rows), run_dir / "lineage" / "shards.parquet")
    pq.write_table(pa.Table.from_pylist([], schema=ORPHAN_SCHEMA), run_dir / "lineage" / "orphaned.parquet")
    last = cuts[-1]
    # MM-18 (EXT-120): a LoRA cut's model hash composes the base and the adapter; one `lora_compose` activity per cut
    # used both and generated the model hash, without merged weights. The id names the cut directory: two cuts may share
    # a global step (EXT-172: flow_grpo_sd35 epochs whose samples were all filtered train no step).
    compose = [{"id": f"{run_id}/{stage_id}/lora_compose/{Path(c['path']).name}", "kind": "lora_compose",
                "used": [{"role": "base", "hash": c["base_ref"]["manifest_hash"]},
                         {"role": "adapter", "hash": c["adapter_hash"]}],
                "generated": [{"role": "model", "model_hash": c["model_hash"], "checkpoint_hash": c["cut_hash"],
                               "last_step_included": c["step"]}]}
               for c in cuts if isinstance(c.get("base_ref"), dict) and c.get("adapter_hash")]
    write_canonical(run_dir / "lineage" / "edges.json", {
        "schema": "gatedtrain.lineage", "schema_version": "1.0", "tags": list(TAGS),
        "activities": [
            {"id": f"{run_id}/{stage_id}", "kind": "training_stage", "uses_plugin": wf.stage["uses"],
             "used": [{"role": "model", "hash": base_hash, "model_hash": base_model_hash},
                      {"role": "data", "hash": snapshot, "precision": precision,
                       "ledger": "ledger/consumption.parquet"},
                      {"role": "config", "hash": file_sha256(cfg_path)}],
             "generated": [{"role": "model", "hash": c["cut_hash"], "model_hash": c["model_hash"],
                            "last_step_included": c["step"]} for c in cuts]},
            {"id": f"{run_id}/{gate_id}", "kind": "gate", "uses_plugin": wf.gate["uses"],
             "used": [{"role": "model", "hash": last["model_hash"], "checkpoint_hash": last["cut_hash"]},
                      {"role": "baseline", "hash": base_model_hash, "checkpoint_hash": base_hash},
                      {"role": "items", "hash": item_set_hash},
                      {"role": "registration", "hash": wf.registration_sha256}],
             "generated": [{"role": "gate_report", "path": "report/gate_report.json"}]},
            *compose,
        ],
    })


def _paths(x) -> list[Path]:
    return [Path(v) for v in x] if isinstance(x, (list, tuple)) else [Path(x)]


def collect(spec_path: Path, *, adapter: str, stage_out: Path, launch_record: Path, smoke_record: Path,
            eval_baseline: Path | list[Path], eval_candidate: Path | list[Path],
            runs_root: Path) -> tuple[Path, dict[str, Any], str]:
    """Recorded external outputs to a run directory (EXT-2). Refuses before writing (EXT-34). With several clauses
    (R18-1), `eval_baseline` and `eval_candidate` list one arm per clause, in clause order."""
    from ..registration import item_set_field, item_set_hash
    from ..runner import _new_run_dir, check_pins
    from ..spec import load_spec

    wf = load_spec(Path(spec_path))                      # spec, registration, REG-CONSTRAINT (R12A-20)
    try:
        ad = registry.adapter(adapter)
    except KeyError:
        raise Refused([Problem("EXT-ADAPTER", "--adapter", f"no adapter {adapter!r} is registered",
                               f"use one of {', '.join(sorted(registry.ADAPTERS))}")])
    problems: list[Problem] = []
    uses_ok = tuple(getattr(ad, "trainer_ids", (ad.trainer_id,)))          # EXT-180: one adapter, two stages
    if wf.stage.get("uses") not in uses_ok:
        problems.append(Problem("EXT-SPEC", "stages[0].uses", f"adapter {adapter} reads {' or '.join(uses_ok)} outputs, "
                                f"not {wf.stage.get('uses')}", f"use `uses: {uses_ok[0]}`, or collect "
                                "rl.nemo_rl@0.1 with `gatedtrain tierb-collect` (MM-13)"))
    multi = wf.clause is None                             # R18-1: several clauses
    evaluators = wf.evaluators
    evaluator = evaluators[0]
    for j, ev in enumerate(evaluators):
        if ev not in ad.evaluator_ids:
            problems.append(Problem("EXT-SPEC", f"stages[1].evaluator[{j}]" if multi else "stages[1].evaluator",
                                    f"adapter {adapter} gates on {', '.join(ad.evaluator_ids)}, not {ev}",
                                    f"use `evaluator: {ad.evaluator_ids[0]}`"))
    eval_baseline, eval_candidate = _paths(eval_baseline), _paths(eval_candidate)
    for flag, given in (("--eval-baseline", eval_baseline), ("--eval-candidate", eval_candidate)):
        if len(given) != len(evaluators):
            problems.append(Problem("EXT-SPEC", flag, f"{len(given)} arm(s) given for {len(evaluators)} clause(s)",
                                    f"pass {flag} once per clause, in the order of the gate's evaluators "
                                    f"({', '.join(evaluators)})"))
    if problems:
        raise Refused(problems)
    rows = {ev: registry.EVALUATOR_ROWS[ev] for ev in evaluators}
    row = rows[evaluator]                                 # adapter-level checks read the first clause's row
    if len({r["payload_text_field"] for r in rows.values()}) > 1:
        raise Refused([Problem("EXT-SPEC", "stages[1].evaluator", "the clauses' evaluators name different item text "
                               f"fields {sorted({r['payload_text_field'] for r in rows.values()})}; the contamination "
                               "check reads one (R18-6)", "gate on evaluators that score the same item text")])
    reg = wf.registration
    names = wf.names
    base, data, items_dir, cfg_path = (wf.path_of(r) for r in ("model", "data", "items", "config"))
    base_field = f"artifacts.{names['model']}.uri"
    stage_out = Path(stage_out)

    # Trainer config (EXT-2) and pins (04:80).
    trainer_cfg, cfg_problems = _trainer_config(cfg_path, f"artifacts.{names['config']}", adapter)
    problems += cfg_problems
    # The base pin is the adapter's base hash, so it leaves out the adapter's `base_excludes` (EXT-162).
    problems += check_pins(wf, {names["model"]: tuple(getattr(ad, "base_excludes", ()))})

    # Item set (MM-6, EXT-5): every clause evaluator's payload (R18-1).
    payload = tuple(dict.fromkeys(f for r in rows.values() for f in r["item_payload"]))
    objects = tuple(dict.fromkeys(f for r in rows.values() for f in r.get("item_payload_objects", ())))
    item_rows, item_bad = read_item_set(items_dir, payload, objects)
    for msg in item_bad:
        problems.append(Problem("EXT-ITEMS", f"artifacts.{names['items']}.uri", msg,
                                "write items.jsonl with one unique non-empty string `item` per line and its payload"))
    items = [r["item"] for r in item_rows]
    clusters: dict[str, dict] = {}                        # clause name -> {item: cluster value} (R18-3)
    for i, c in enumerate(wf.clauses):
        key = c.get("cluster_key")
        if key is None:
            continue
        bad_items = [r["item"] for r in item_rows if not cluster_value_ok(r.get(key))]
        if bad_items:
            problems.append(Problem(
                "EXT-ITEMS", f"registration.clauses[{i}].cluster_key" if multi else "registration.clause.cluster_key",
                f"{len(bad_items)} item(s) of {items_dir}/items.jsonl have no string or integer `{key}` to cluster by "
                f"(R18-3): {', '.join(map(str, bad_items[:5]))}" + (" ..." if len(bad_items) > 5 else ""),
                f"give every item its `{key}`, or register the clause without `cluster_key`"))
        else:
            clusters[c.get("name", "")] = {r["item"]: r[key] for r in item_rows}
    actual_items_hash, items_entries = None, None
    if (items_dir / "items.jsonl").is_file():
        try:                                              # EXT-41: every manifest before the run directory
            items_entries = manifest_entries(items_dir)
            actual_items_hash = entries_hash(items_entries)
        except (OSError, ValueError) as exc:
            problems.append(Problem("EXT-ITEMS", f"artifacts.{names['items']}.uri", f"{items_dir} cannot be hashed: "
                                    f"{exc}", "keep the item set a plain readable directory with no symbolic links"))
    registered_items = item_set_hash(reg)
    if items_entries is not None and actual_items_hash != registered_items:
        problems.append(Problem("REG-FIELD", item_set_field(reg),
                                f"registered item set {registered_items} but {items_dir} is {actual_items_hash}",
                                "evaluate on the registered item set, or write a new registration"))
    data_ok = data.is_dir() and any(data.glob("*.jsonl"))
    data_entries = None
    if not data_ok:
        problems.append(Problem("SPEC-REF", f"artifacts.{names['data']}.uri", f"{data} holds no *.jsonl shards",
                                "point the training dataset at the directory of shard files the run read"))
    else:
        try:                                              # EXT-41
            data_entries = manifest_entries(data)
        except (OSError, ValueError) as exc:
            data_ok = False
            problems.append(Problem("SPEC-REF", f"artifacts.{names['data']}.uri", f"{data} cannot be hashed: {exc}",
                                    "keep the training dataset a plain readable directory with no symbolic links"))

    # Base model directory (MM-20, MM-21): plain, no symlink, no `.cache/`, a deployable model. An adapter may declare
    # `base_excludes`, top-level names its base hash leaves out (EXT-162: `flow_grpo_sd35` excludes `.cache`).
    base_entries, base_model = None, None
    excludes = tuple(getattr(ad, "base_excludes", ()))
    if ".cache" not in excludes and (base / ".cache").exists():
        problems.append(Problem("EXT-BASE", base_field, f"{base}/.cache exists (Hugging Face download metadata)",
                                f"delete {base}/.cache, then collect again"))
    if base.is_symlink() or not base.is_dir():
        problems.append(Problem("EXT-BASE", base_field, f"{base} is not a plain directory",
                                "download the model with `--local-dir` into a plain directory"))
    else:
        bad = ad.model_dir_problem(base)
        if bad:
            problems.append(Problem("EXT-BASE", base_field, f"{base} is not a model directory: {bad}",
                                    "point the base artifact at the downloaded model directory"))
        try:
            base_entries = manifest_entries(base, exclude=excludes)
            base_model = ad.identity(base, None, entries=base_entries)["model_hash"]
        except (OSError, ValueError) as exc:
            problems.append(Problem("EXT-BASE", base_field, f"{base} cannot be hashed: {exc}",
                                    "keep the base a plain readable directory with no symbolic links (manifest rule H2)"))

    # Stage outputs (MM-20, EXT-13).
    listed = True
    try:                                                  # EXT-154: the stage directory may be unreadable
        complete, incomplete = ad.find_cuts(stage_out)
        all_dirs = ad.checkpoint_dirs(stage_out)
    except OSError as exc:
        listed, complete, incomplete, all_dirs = False, [], [], []
        problems.append(Problem("EXT-STAGE", "--stage-out", f"{stage_out} cannot be listed ({exc.__class__.__name__})",
                                "make the run's output directory readable for the host user"))
    cuts: list[dict] = []
    if not listed:
        pass
    elif not complete:
        problems.append(Problem("EXT-STAGE", "--stage-out", f"no complete checkpoint-<N> cut under {stage_out}"
                                + (f" (incomplete: {[p.name for p in incomplete]})" if incomplete else ""),
                                "pass the run's output directory (the launch record's output_dir) after it finished"))
    elif all_dirs and all_dirs[-1][1] in incomplete:
        problems.append(Problem("EXT-STAGE", "--stage-out",
                                f"the last checkpoint {all_dirs[-1][1].name} is incomplete "
                                f"({ad.cut_problem(all_dirs[-1][1], all_dirs[-1][0])}); it cannot be the trained model",
                                "collect a finished run; an interrupted last save needs a resumed run that saves again"))
    else:
        for step, d in complete:
            try:
                ent = manifest_entries(d)
            except (OSError, ValueError) as exc:
                problems.append(Problem("EXT-STAGE", "--stage-out", f"{d} cannot be hashed: {exc}",
                                        "make the outputs readable for the host user; no symbolic links"))
                continue
            try:                                          # EXT-131: a LoRA identity hashes the base too
                ident = ad.identity(d, base, entries=ent)
            except (OSError, ValueError) as exc:
                problems.append(Problem("EXT-BASE", base_field, f"the model hash of {d.name} composes the spec's base, "
                                        f"which cannot be hashed: {exc}",
                                        "keep the base a plain readable directory with no symbolic links (manifest rule H2)"))
                continue
            cuts.append({"step": step, "path": d, "entries": ent, **ident})

    # Launch record (MM-34, MM-35), profile, GPU seconds (MM-36), framework pin (MM-33), base binding (MM-21).
    launch, gpu_s, framework, wall_s = None, None, None, None
    try:
        launch = _read_json(launch_record)
    except (OSError, ValueError) as exc:
        problems.append(Problem("EXT-LAUNCH", "--launch-record", f"cannot read {launch_record}: "
                                f"{exc.__class__.__name__}", "pass the launch.json `gatedtrain ext-launch` wrote"))
    if launch is not None:
        problems += record_problems(launch, adapter=adapter, stage_out=stage_out, framework=ad.framework_name)
        if isinstance(launch, dict) and hasattr(ad, "argv_problems"):
            problems += [Problem("EXT-LAUNCH", "--launch-record.argv", msg,
                                 "launch with the command `gatedtrain ext-launch` prints, then collect with its record")
                         for msg in ad.argv_problems(launch.get("argv"), profile=launch.get("profile"))]
        if isinstance(launch, dict):
            if trainer_cfg.get("profile") in PROFILES and launch.get("profile") != trainer_cfg.get("profile"):
                problems.append(Problem("EXT-PROFILE", f"artifacts.{names['config']}.profile",
                                        f"the trainer config says profile {trainer_cfg.get('profile')!r}, the launch "
                                        f"record {launch.get('profile')!r}",
                                        "collect the run of the spec's profile, or fix the trainer config"))
            hw, label = launch.get("host_work"), launch.get("label")
            if isinstance(hw, str) and isinstance(label, str):
                where = gpu_step_log_problem(Path(hw))         # EXT-46: the log gpu-step.sh writes
                if where:
                    problems.append(Problem("EXT-GPU", "--launch-record.host_work", where,
                                            "launch through the gpu-step.sh that logs to <host_work>/gpu-guard.log"))
                try:
                    wall_s = gpu_seconds(Path(hw) / "gpu-guard.log", label)      # START/END time of the label
                    started = first_start(Path(hw) / "gpu-guard.log", label)
                    # EXT-105 (MM-31): a CPU-only step (`gpus: false`) is logged with 0 GPU-hours; its wrapper time
                    # stays in `wall_seconds`. A GPU step's GPU time is its wrapper time, as before.
                    gpu_s = wall_s if launch.get("gpus") is not False else 0
                except (OSError, ValueError) as exc:
                    problems.append(Problem("EXT-GPU", "--launch-record.label", f"GPU time of {label!r}: {exc}",
                                            "collect after gpu-step.sh logged the label's END in <HOST_WORK>/gpu-guard.log"))
                else:
                    extra = ad.extra_read_outputs(stage_out) if hasattr(ad, "extra_read_outputs") else []
                    stale = stale_outputs(stage_out, [c["path"] for c in cuts], started, extra)  # EXT-64
                    if stale:
                        problems.append(Problem(
                            "EXT-STAGE", "--stage-out", f"{len(stale)} output file(s) predate the label's first "
                            f"START in gpu-guard.log, so an earlier run wrote them: {', '.join(stale[:5])}"
                            + (" ..." if len(stale) > 5 else ""),
                            "collect the outputs of this launch only: launch into a new, empty output directory"))
            try:
                framework = ad.framework_identity(stage_out, launch)
            except (OSError, ValueError) as exc:              # EXT-154: an unreadable freeze refuses too
                problems.append(Problem("EXT-FREEZE", "--stage-out", str(exc),
                                        "run the launcher's env-freeze pre-step in the same runtime, and launch "
                                        "with the framework version it shows"))
            problems += _guarded(lambda: ad.script_problems(stage_out, launch), "EXT-LAUNCH",
                                 "--launch-record.script_sha256")                       # EXT-83
            if cuts:
                env_kw = ({"eval_distribution": reg.get("eval_distribution")} if getattr(ad, "uses_env", False)
                          else {})                                                     # EXT-103
                if getattr(ad, "wants_uses", False):                                   # EXT-187
                    env_kw["uses"] = wf.stage.get("uses")
                if getattr(ad, "wants_evaluator", False):                              # EXT-253
                    env_kw["evaluator"] = evaluator
                problems += _guarded(lambda: ad.checks(last_cut=cuts[-1]["path"], base=base, launch=launch,
                                                       base_field=base_field, stage_out=stage_out, data=data,
                                                       data_field=f"artifacts.{names['data']}.uri", **env_kw),
                                     "EXT-STAGE", "--stage-out")                       # EXT-65, EXT-68

    # Observed consumption log (EXT-102): checked before anything is written.
    if cuts and hasattr(ad, "consumption_problems") and data_ok:
        fix = getattr(ad, "consumption_fix", "collect the run's own rollouts.jsonl as the run wrote it (MM-25, MM-27)")
        try:
            msgs = ad.consumption_problems(stage_out, data, [(c["step"], c["path"]) for c in cuts])
        except OSError as exc:                            # EXT-154
            msgs = [f"a consumption log or the training data cannot be read ({exc.__class__.__name__})"]
        for msg in msgs:
            problems.append(Problem("EXT-CONSUMPTION", "--stage-out", msg, fix))

    # EXT-125: an adapter whose cuts carry their deployment sampler checks it against the registered one.
    if cuts and hasattr(ad, "sampler_problems"):
        problems += _guarded(lambda: ad.sampler_problems([c["path"] for c in cuts], reg.get("sampler")),
                             "EXT-SAMPLER", "--stage-out")

    # MM-19 (EXT-109): an environment evaluator needs the registered evaluation environment.
    registered_env = (reg.get("eval_distribution") or {}).get("env_version") if isinstance(
        reg.get("eval_distribution"), dict) else None
    for ev, r_ in rows.items():
        if r_.get("env_version") and registered_env is None:
            problems.append(Problem("REG-FIELD", "registration.eval_distribution.env_version",
                                    f"evaluator {ev} scores in an environment, but the registration names no "
                                    "env_version (MM-19, R11-19)",
                                    "register the SHA-256 of the evaluation environment's env_version.json"))

    # Smoke record (MM-39, EXT-30).
    smoke = None
    try:
        smoke = _read_json(smoke_record)
    except (OSError, ValueError) as exc:
        problems.append(Problem("EXT-SMOKE", "--smoke-record", f"cannot read {smoke_record}: {exc.__class__.__name__}",
                                "pass the area's PASS smoke.json"))
    if smoke is not None:
        for msg in smoke_problems(smoke, adapter=adapter, score_range=row["score_range"], binary=row["binary"],
                                  direction=row.get("direction"),
                                  base_model_hash=base_model, item_set_hash=registered_items,
                                  evaluators=[(ev, rows[ev]) for ev in evaluators] if multi else None):   # R18-8
            problems.append(Problem("EXT-SMOKE", "--smoke-record", msg,
                                    "collect only after a PASS smoke run of this adapter, base and item set (MM-39)"))

    # Arms (MM-1 to MM-3, EXT-1, EXT-3): one pair per clause, checked against its evaluator's row (R18-1).
    sampler = reg["sampler"]
    evaluated = {"baseline": base if base_model is not None else None,
                 "candidate": cuts[-1]["path"] if cuts else None}
    arms_by_clause: list[dict[str, dict]] = []
    for ci, (clause, ev) in enumerate(zip(wf.clauses, evaluators)):
        c_row = rows[ev]
        seeds = list(clause["eval_seeds"])
        expected = {"schema_version": "1.0", "evaluator_id": ev, "adapter": adapter, "metric": c_row["metric"],
                    "direction": c_row["direction"], "score_range": list(c_row["score_range"]),
                    "scorer": dict(c_row["scorer"]), "normalizer_sha256": None, "env_version_sha256": None,
                    "item_set_hash": registered_items, "eval_seeds": seeds, "sampler": sampler,
                    "sampler_hash": sha256_bytes(canonical_json(sampler))}
        if c_row.get("env_version"):
            expected["env_version_sha256"] = registered_env                               # EXT-109
        arms: dict[str, dict] = {}
        for arm, path, flag in (("baseline", eval_baseline[ci], "--eval-baseline"),
                                ("candidate", eval_candidate[ci], "--eval-candidate")):
            meta, why = read_meta(path)
            want = dict(expected)
            if c_row.get("normalizer"):                 # EXT-109: the normalizer of the cut the arm must evaluate
                try:
                    want["normalizer_sha256"] = ad.normalizer_sha256(evaluated[arm])
                except (OSError, TypeError):
                    want.pop("normalizer_sha256")          # no such cut: refused above (EXT-BASE, EXT-STAGE)
            if meta is not None:
                why += meta_differences(meta, want)
                why += provenance_problems(path, meta, nel_benchmark=c_row.get("nel_benchmark"), row=c_row,
                                           item_rows=item_rows)                                 # EXT-47, EXT-52
                ck = meta.get("checkpoint")
                bmh = ck.get("base_model_hash") if isinstance(ck, dict) else None
                if bmh is not None and base_model is not None and bmh != base_model:
                    why.append(f"eval_meta.json checkpoint.base_model_hash is {bmh!r}, not the spec's base {base_model}")
            recs, line_bad = read_lines(path, seeds=seeds, score_range=c_row["score_range"],
                                        error_classes=tuple(c_row["error_classes"]))
            why += line_bad
            if meta is not None and recs and hasattr(ad, "arm_problems"):  # EXT-155: outputs name the registered items
                arm_kw = ({"items_dir": items_dir, "model_dir": evaluated[arm]}              # EXT-194, EXT-197
                          if getattr(ad, "arm_needs_items_dir", False) else {})
                try:
                    why += ad.arm_problems(path, recs, item_rows, **arm_kw)
                except OSError as exc:                                     # EXT-154
                    why.append(f"an output or the scoring record cannot be read ({exc.__class__.__name__})")
            for msg in why:
                problems.append(Problem("EXT-ARM", flag, f"{path}: {msg}",
                                        "pass an arm directory (items.jsonl and eval_meta.json) of this evaluator and "
                                        "registration; merge single-seed runs with `gatedtrain ext-merge-eval`"))
            arms[arm] = {"path": path, "meta": meta, "records": recs}
        arms_by_clause.append(arms)
    arms = arms_by_clause[0]
    if multi:
        seen_paths: dict[Path, str] = {}
        for ci, pair in enumerate(arms_by_clause):
            for arm, info in pair.items():
                key = resolve(info["path"])
                if key in seen_paths:
                    problems.append(Problem("EXT-ARM", f"--eval-{arm}", f"{info['path']} is passed twice "
                                            f"({seen_paths[key]} and {arm} of clause {ci})",
                                            "pass each clause's own arm directories"))
                seen_paths.setdefault(key, f"{arm} of clause {ci}")

    # Contamination (MM-11, EXT-21).
    overlap, allowed = None, bool((reg.get("contamination") or {}).get("allow_exact_overlap", False))
    field = row["payload_text_field"]
    if hasattr(ad, "training_payloads"):               # EXT-107: against the episodes the run logged, not the shard
        if cuts and item_rows and base_model is not None:
            texts, why = ad.training_payloads(stage_out, base, [(c["step"], c["path"]) for c in cuts],
                                              data=data if data_ok else None)
            if texts is None:
                problems.append(Problem("EXT-CONTAMINATION", "--stage-out", f"cannot read the training episodes: {why}",
                                        "keep the run's training logs (and a base run's) beside their cuts"))
            else:
                overlap = sum(1 for r in item_rows if r.get(field) in texts)
    elif data_ok and item_rows:
        try:
            overlap = exact_overlap(item_rows, data, field)
        except (OSError, ValueError) as exc:
            problems.append(Problem("EXT-CONTAMINATION", f"artifacts.{names['data']}.uri",
                                    f"cannot read the training rows: {exc}", "point the training dataset at JSONL shards"))
    # EXT-176 (flow_grpo_sd35): an adapter may declare `item_overlap_problems`, further item/training overlap rules
    # (the OCR area: normalized target text); each refuses, whatever `allow_exact_overlap` says.
    if hasattr(ad, "item_overlap_problems") and item_rows and data_ok:
        try:
            msgs = ad.item_overlap_problems(item_rows, data)
        except OSError as exc:                            # EXT-154
            msgs = [f"the training shard cannot be read ({exc.__class__.__name__})"]
        for msg in msgs:
            problems.append(Problem("EXT-CONTAMINATION", f"artifacts.{names['items']}.uri", msg,
                                    "build the item set and the training shard disjoint by target text"))
    if overlap and not allowed:
        problems.append(Problem("EXT-CONTAMINATION", "registration.contamination.allow_exact_overlap",
                                f"{overlap} item(s) of the item set have a `{field}` equal to "
                                + (getattr(ad, "payload_source", "a training episode's (exact overlap, MM-11, EXT-107)")
                                   if hasattr(ad, "training_payloads") else "a training row's (exact overlap, MM-11)"),
                                "remove the overlap from the training data or the item set, or register "
                                "`contamination: {allow_exact_overlap: true}` with a reason"))
    if problems:
        raise Refused(problems)

    # Everything checked: write the run directory (removed again if a write fails, EXT-41).
    try:
        run_dir = _new_run_dir(Path(runs_root).absolute())
    except OSError as exc:                                # EXT-75: the runs root cannot hold a new run
        raise Refused([write_problem(exc, "--runs-root", Path(runs_root))]) from exc
    try:
        return _write_run(run_dir, wf=wf, ad=ad, adapter=adapter, row=row, evaluator=evaluator, reg=reg,
                          items_dir=items_dir, items=items, arms=arms_by_clause if multi else arms,
                          clusters=clusters, rows=rows, launch_record=Path(launch_record),
                          smoke_record=Path(smoke_record), stage_out=stage_out, data=data, base=base,
                          base_entries=base_entries, base_model=base_model, cfg_path=cfg_path, cuts=cuts,
                          actual_items_hash=actual_items_hash, launch=launch, framework=framework, gpu_s=gpu_s,
                          wall_s=wall_s,
                          overlap=overlap, allowed=allowed, incomplete=incomplete,
                          known={data: data_entries, items_dir: items_entries})
    except OSError as exc:                  # EXT-59, EXT-75: a failed write is a refusal; other errors are bugs
        shutil.rmtree(run_dir, ignore_errors=True)
        raise Refused([write_problem(exc, "--runs-root", run_dir)]) from exc
    except BaseException:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise


def _write_run(run_dir: Path, *, wf, ad, adapter: str, row: dict, evaluator: str, reg: dict, items_dir: Path,
               items: list, arms: dict | list, launch_record: Path, smoke_record: Path, stage_out: Path, data: Path,
               base: Path, base_entries: list, base_model: str, cfg_path: Path, cuts: list[dict],
               actual_items_hash: str, launch: dict, framework: dict, gpu_s: int, overlap: int, allowed: bool,
               incomplete: list, known: dict, wall_s: int | None = None, clusters: dict | None = None,
               rows: dict | None = None) -> tuple[Path, dict[str, Any], str]:
    """The writes of `collect`, after every check passed. With several clauses (R18-1) `arms` lists one
    {baseline, candidate} pair per clause."""
    if isinstance(arms, list):
        return _write_run_multi(run_dir, wf=wf, ad=ad, adapter=adapter, reg=reg, items_dir=items_dir, items=items,
                                arms_by_clause=arms, launch_record=launch_record, smoke_record=smoke_record,
                                stage_out=stage_out, data=data, base=base, base_entries=base_entries,
                                base_model=base_model, cfg_path=cfg_path, cuts=cuts,
                                actual_items_hash=actual_items_hash, launch=launch, framework=framework, gpu_s=gpu_s,
                                overlap=overlap, allowed=allowed, incomplete=incomplete, known=known, wall_s=wall_s,
                                clusters=clusters or {}, rows=rows or {})
    from ..ledger import SCHEMA as LEDGER_SCHEMA
    from ..registration import deciding_evaluator
    from ..runner import _log, copy_registration, gate_and_report, write_manifests, write_plan

    copy_registration(wf, run_dir)
    run_id = run_dir.name
    _log(f"ext-collect {run_id}: {wf.spec['workflow']} via adapter {adapter} (prototype, unpromotable)")
    the_plan = write_plan(wf, run_dir, items_dir, n_available=len(items))
    clusters = (clusters or {}).get(wf.clause.get("name", ""))                       # R18-3 in the one-clause form
    for arm, info in arms.items():
        dst = run_dir / "eval" / arm
        dst.mkdir(parents=True)
        for name in ("items.jsonl", "eval_meta.json", "merge.json"):
            if (info["path"] / name).is_file():
                shutil.copyfile(info["path"] / name, dst / name)
        for rec in info["records"]:                      # referenced outputs stay resolvable inside the run
            ref = rec.get("output_ref")
            if ref:
                (dst / ref["path"]).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(info["path"] / ref["path"], dst / ref["path"])
    (run_dir / "records").mkdir()
    shutil.copyfile(launch_record, run_dir / "records" / "launch.json")
    shutil.copyfile(smoke_record, run_dir / "records" / "smoke.json")
    cons = ad.consumption(stage_out, data, [(c["step"], c["path"]) for c in cuts], run_id=run_id)
    (run_dir / "ledger").mkdir()
    pq.write_table(pa.Table.from_pylist(list(cons["rows"]), schema=LEDGER_SCHEMA),
                   run_dir / "ledger" / "consumption.parquet")
    base_hash = entries_hash(base_entries)
    write_lineage(wf, run_dir, base_hash=base_hash, base_model_hash=base_model, data=data, cfg_path=cfg_path,
                  cuts=cuts, item_set_hash=actual_items_hash, precision=cons["precision"],
                  snapshot=entries_hash(known[data]))
    last = cuts[-1]
    hashes = write_manifests(run_dir, base=base, final_ckpt=last["path"], data=data, prompts=items_dir,
                             known_entries={base: base_entries, last["path"]: last["entries"], **known})
    rt = launch["runtime"]
    lock = {
        "schema": "gatedtrain.lockfile", "schema_version": "1.0", "tags": list(TAGS), "run_id": run_id,
        "spec_sha256": file_sha256(wf.spec_path), "registration_sha256": wf.registration_sha256,
        "config_hash": file_sha256(cfg_path),
        "allow_list_checked": False,                       # no allow-list checks in the prototype
        "adapter": adapter, "direction": row["direction"], "gpu_seconds": gpu_s,
        "wall_seconds": wall_s,                                                             # EXT-105
        "smoke_record_sha256": file_sha256(smoke_record), "launch_record_sha256": file_sha256(launch_record),
        "independence_tier": row["tier"],
        "model_hashes": {"baseline": base_model, "candidate": last["model_hash"]},
        "backend": {
            "trainer": {"uses": wf.stage["uses"], "adapter": adapter, "runtime_kind": rt["kind"],
                        "image_id": rt["image"], "venv_lock_sha256": rt.get("venv_lock_sha256"),
                        "framework": launch["framework"], "freeze_sha256": framework["freeze_sha256"],
                        "profile": launch["profile"], "training_seed": launch["training_seed"],    # MM-29 claim
                        "seed_derivation": ad.seed_derivation,
                        "checkpoint_completeness": ad.checkpoint_completeness},                 # MM-20
            "evaluator": {"uses": evaluator, "scorer": dict(row["scorer"])},
        },
        "analysis_code_version": ANALYSIS_CODE_VERSION,
        "python_version": sys.version.split()[0],
        "artifacts": dict(sorted(hashes.items())),
    }
    lock_bytes = write_canonical(run_dir / "lockfile.json", lock)
    deciding = deciding_evaluator(reg, evaluator)
    assert deciding is not None                           # check_against_gate refuses an unlisted gate evaluator
    # 05 section 6.1 disclosures (R15-1 EXT-128, EXT-134; R16-1 EXT-148): the registration's statement, and the
    # manifest's (registry row) beside it.
    disclosures = {d["field"]: {d["fact"]: row[d["row_key"]][d["fact"]], "statement": reg.get(d["field"]),
                                "manifest_statement": row[d["row_key"]]["statement"]}
                   for d in registry.DISCLOSURES if row.get(d["row_key"])}
    notes = [f"Consumption precision {cons['precision']}: {cons['reason']}.",
             f"Checkpoint completeness ({adapter}): {ad.checkpoint_completeness['rule']} "
             f"(verified: {str(ad.checkpoint_completeness['verified']).lower()}).",
             f"Seeds: {ad.seed_derivation}"]
    if hasattr(ad, "report_notes"):                   # EXT-246: an adapter's own report lines (per-task success)
        notes += list(ad.report_notes({arm: info["records"] for arm, info in arms.items()}))
    report, text = gate_and_report(
        wf, run_dir, the_plan=the_plan, hashes=hashes, lock_bytes=lock_bytes, item_set_hash=actual_items_hash,
        evaluator_hash=registry.evaluator_identity_hash(evaluator), stubs=[], items=items,
        records={arm: [gate_record(r) for r in info["records"]] for arm, info in arms.items()},
        model_hashes={arm: info["meta"]["checkpoint"]["model_hash"] for arm, info in arms.items()},
        expected_models={"baseline": base_model, "candidate": last["model_hash"]}, clusters=clusters,
        extra={"metric_source": f"items.jsonl score ({row['scorer']['name']})", "adapter": adapter,
               "independence_tier": row["tier"], "gpu_seconds": gpu_s, "wall_seconds": wall_s,
               "contamination": {"exact_overlap": overlap, "allowed": allowed},
               "consumption_precision": cons["precision"],
               "incomplete_cuts": [p.name for p in incomplete],
               "baseline_model_hash": base_model, "candidate_model_hash": last["model_hash"], "notes": notes,
               **disclosures})
    return run_dir, report, text


def _write_run_multi(run_dir: Path, *, wf, ad, adapter: str, reg: dict, items_dir: Path, items: list,
                     arms_by_clause: list[dict], launch_record: Path, smoke_record: Path, stage_out: Path, data: Path,
                     base: Path, base_entries: list, base_model: str, cfg_path: Path, cuts: list[dict],
                     actual_items_hash: str, launch: dict, framework: dict, gpu_s: int, overlap: int, allowed: bool,
                     incomplete: list, known: dict, wall_s: int | None, clusters: dict,
                     rows: dict) -> tuple[Path, dict[str, Any], str]:
    """The writes of `collect` for several clauses (R18-1, R18-6): as `_write_run`, with each clause's arms under
    `eval/<arm>/<clause name>/`, one plan per clause, per-clause evaluator fields in the lockfile, and the combined
    gate (`runner.gate_and_report_multi`)."""
    from ..ledger import SCHEMA as LEDGER_SCHEMA
    from ..runner import _log, copy_registration, gate_and_report_multi, write_manifests, write_plans

    copy_registration(wf, run_dir)
    run_id = run_dir.name
    _log(f"ext-collect {run_id}: {wf.spec['workflow']} via adapter {adapter}, {len(wf.clauses)} clauses "
         "(prototype, unpromotable)")
    plans = write_plans(wf, run_dir, n_available=len(items))
    for clause, pair in zip(wf.clauses, arms_by_clause):
        for arm, info in pair.items():
            dst = run_dir / "eval" / arm / clause["name"]
            dst.mkdir(parents=True)
            for name in ("items.jsonl", "eval_meta.json", "merge.json"):
                if (info["path"] / name).is_file():
                    shutil.copyfile(info["path"] / name, dst / name)
            for rec in info["records"]:                  # referenced outputs stay resolvable inside the run
                ref = rec.get("output_ref")
                if ref:
                    (dst / ref["path"]).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(info["path"] / ref["path"], dst / ref["path"])
    (run_dir / "records").mkdir()
    shutil.copyfile(launch_record, run_dir / "records" / "launch.json")
    shutil.copyfile(smoke_record, run_dir / "records" / "smoke.json")
    cons = ad.consumption(stage_out, data, [(c["step"], c["path"]) for c in cuts], run_id=run_id)
    (run_dir / "ledger").mkdir()
    pq.write_table(pa.Table.from_pylist(list(cons["rows"]), schema=LEDGER_SCHEMA),
                   run_dir / "ledger" / "consumption.parquet")
    base_hash = entries_hash(base_entries)
    write_lineage(wf, run_dir, base_hash=base_hash, base_model_hash=base_model, data=data, cfg_path=cfg_path,
                  cuts=cuts, item_set_hash=actual_items_hash, precision=cons["precision"],
                  snapshot=entries_hash(known[data]))
    last = cuts[-1]
    hashes = write_manifests(run_dir, base=base, final_ckpt=last["path"], data=data, prompts=items_dir,
                             known_entries={base: base_entries, last["path"]: last["entries"], **known})
    evaluators = [c["evaluator"] for c in wf.clauses]
    rt = launch["runtime"]
    lock = {
        "schema": "gatedtrain.lockfile", "schema_version": "1.0", "tags": list(TAGS), "run_id": run_id,
        "spec_sha256": file_sha256(wf.spec_path), "registration_sha256": wf.registration_sha256,
        "config_hash": file_sha256(cfg_path),
        "allow_list_checked": False,                       # no allow-list checks in the prototype
        "adapter": adapter, "direction": [rows[ev]["direction"] for ev in evaluators], "gpu_seconds": gpu_s,
        "wall_seconds": wall_s,                                                             # EXT-105
        "smoke_record_sha256": file_sha256(smoke_record), "launch_record_sha256": file_sha256(launch_record),
        "independence_tier": [rows[ev]["tier"] for ev in evaluators],
        "model_hashes": {"baseline": base_model, "candidate": last["model_hash"]},
        "backend": {
            "trainer": {"uses": wf.stage["uses"], "adapter": adapter, "runtime_kind": rt["kind"],
                        "image_id": rt["image"], "venv_lock_sha256": rt.get("venv_lock_sha256"),
                        "framework": launch["framework"], "freeze_sha256": framework["freeze_sha256"],
                        "profile": launch["profile"], "training_seed": launch["training_seed"],    # MM-29 claim
                        "seed_derivation": ad.seed_derivation,
                        "checkpoint_completeness": ad.checkpoint_completeness},                 # MM-20
            "evaluator": {"uses": evaluators, "scorer": [dict(rows[ev]["scorer"]) for ev in evaluators]},
        },
        "analysis_code_version": ANALYSIS_CODE_VERSION,
        "python_version": sys.version.split()[0],
        "artifacts": dict(sorted(hashes.items())),
    }
    lock_bytes = write_canonical(run_dir / "lockfile.json", lock)
    # 05 section 6.1 disclosures: the registration's statement and the first declaring evaluator row's.
    disclosures = {}
    for d in registry.DISCLOSURES:
        for ev in evaluators:
            if rows[ev].get(d["row_key"]):
                r_ = rows[ev]
                disclosures[d["field"]] = {d["fact"]: r_[d["row_key"]][d["fact"]], "statement": reg.get(d["field"]),
                                           "manifest_statement": r_[d["row_key"]]["statement"], "evaluator": ev}
                break
    notes = [f"Consumption precision {cons['precision']}: {cons['reason']}.",
             f"Checkpoint completeness ({adapter}): {ad.checkpoint_completeness['rule']} "
             f"(verified: {str(ad.checkpoint_completeness['verified']).lower()}).",
             f"Seeds: {ad.seed_derivation}"]
    if hasattr(ad, "report_notes"):                   # EXT-246: an adapter's own report lines, per clause
        for clause, pair in zip(wf.clauses, arms_by_clause):
            notes += [f"{clause['name']}: {n}" for n in
                      ad.report_notes({arm: info["records"] for arm, info in pair.items()})]
    report, text = gate_and_report_multi(
        wf, run_dir, plans=plans, hashes=hashes, lock_bytes=lock_bytes, item_set_hash=actual_items_hash,
        evaluator_hashes={ev: registry.evaluator_identity_hash(ev) for ev in evaluators}, items=items,
        records=[{arm: [gate_record(r) for r in info["records"]] for arm, info in pair.items()}
                 for pair in arms_by_clause],
        model_hashes=[{arm: info["meta"]["checkpoint"]["model_hash"] for arm, info in pair.items()}
                      for pair in arms_by_clause],
        expected_models={"baseline": base_model, "candidate": last["model_hash"]}, clusters=clusters,
        clause_extra=[{"metric_source": f"items.jsonl score ({rows[ev]['scorer']['name']})",
                       "independence_tier": rows[ev]["tier"]} for ev in evaluators],
        extra={"adapter": adapter, "independence_tier": [rows[ev]["tier"] for ev in evaluators],
               "gpu_seconds": gpu_s, "wall_seconds": wall_s,
               "contamination": {"exact_overlap": overlap, "allowed": allowed},
               "consumption_precision": cons["precision"],
               "incomplete_cuts": [p.name for p in incomplete],
               "baseline_model_hash": base_model, "candidate_model_hash": last["model_hash"], "notes": notes,
               **disclosures})
    return run_dir, report, text
