"""`gatedtrain run`: the whole skeleton path of MVP-SPEC section 2.

spec -> registration -> plan -> baseline evaluation -> training stage ->
candidate evaluation -> manifests and lockfile -> ledger and lineage -> gate
-> report. Every refusal happens before the run directory exists
(tests/README.md section 2), and a run never reuses or overwrites a run
directory (MVP-SPEC section 2, last line).

Design revision 11 (tests/README.md section 11.16): new writes use item names (R11A-4): the gate's
evaluation set is `manifests/item_set.json` (type `item_set`), the lockfile's `artifacts.item_set`, the
lineage gate input role `items`, and the report's `n_items`, `samples_per_item`, `eval_distribution`. The
gate runs on per-item records (`gates.evaluate_items`, NeMo Evaluator-shaped records converted by
`items.from_nemo_evaluator`) with the registered item set as the expected items (R11A-6, R11A-17).
"""

from __future__ import annotations

import datetime as _dt
import json
import secrets
import shutil
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from . import ANALYSIS_CODE_VERSION, PACKAGE, STUB_MARK, TAGS, __version__
from .canonical import (canonical_json, entries_hash, file_sha256, manifest_document, manifest_entries, manifest_hash,
                        sha256_bytes, text_lines, write_canonical)
from .errors import Problem, Refused
from .gates import MISSING, PASS, apply_post_hoc, combine, evaluate_items, plan
from .items import from_nemo_evaluator, score_range
from .ledger import ORPHAN_SCHEMA, orphaned_rows, shard_table, write_ledger
from .registration import (MULTI_PLAN, deciding_evaluator, item_set_field, item_set_hash, max_unpaired, min_paired,
                           normalize, pilot_for)
from .report import build_multi_report, build_report, render_text
from .spec import EVALUATOR_BACKENDS, Workflow, load_spec

MANIFEST_TYPES = {
    "baseline_checkpoint": "model_checkpoint",
    "trained_checkpoint": "model_checkpoint",
    "training_shards": "training_dataset",
    "item_set": "item_set",
    "registration": "registration",
    "eval_baseline": "evaluation_results",
    "eval_candidate": "evaluation_results",
    "ledger": "consumption_ledger",
    "gate_report": "gate_report",
}

STUBS = [
    f"trainer sft.cpu_stub@0.1 ({STUB_MARK}): gatedtrain.stubs.trainer, supervised fine-tuning on CPU",
    f"evaluator addition_em.cpu_stub@0.1 ({STUB_MARK}): gatedtrain.stubs.evaluator, temperature sampling",
    f"model llm.char_transformer@0.1 ({STUB_MARK}): gatedtrain.stubs.model",
    f"task 2-digit addition, exact match ({STUB_MARK}): gatedtrain.stubs.task",
]


def _utc_now() -> str:
    """RFC 3339 UTC with milliseconds (the format conventions)."""
    now = _dt.datetime.now(_dt.timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def stub_evaluator_hash() -> str:
    """Identity of the stand-in evaluator: hashes of its code files. The report records it as `evaluator_hash`,
    and a Tier A registration carries it as the deciding evaluator's `manifest_hash` (R10A-2, R11A-24)."""
    from .stubs import evaluator, model, task

    files = {Path(m.__file__).name: file_sha256(Path(m.__file__)) for m in (evaluator, model, task)}
    return sha256_bytes(canonical_json({"evaluator": evaluator.EVALUATOR_ID, "backend": "cpu_stub",
                                        "package": PACKAGE, "version": __version__, "code": files}))


def preflight(wf: Workflow) -> dict[str, Any]:
    """Checks that need the artifacts on disk. Raises Refused before anything is written."""
    from .stubs import model as model_mod
    from .stubs import trainer as trainer_mod

    problems: list[Problem] = []
    names = wf.names
    if wf.trainer_backend == "ext":
        problems.append(Problem(
            "SPEC-REF", "stages[0].uses",
            f"{wf.stage['uses']} is an external-run trainer; `gatedtrain run` drives the Tier A stand-ins only",
            "train with the command `gatedtrain ext-launch` prints, then collect with `gatedtrain ext-collect` "
            "(MULTIMODAL-SPEC MM-9)"))
    elif wf.trainer_backend != "cpu_stub":
        problems.append(Problem(
            "SPEC-REF", "stages[0].uses",
            f"{wf.stage['uses']} is a Tier B backend; gatedtrain and Tier A runs never launch GPU jobs",
            "`gatedtrain run` never drives Tier B; on the GPU host follow docs/tier-b-runbook.md and turn the recorded outputs into a gate report with `gatedtrain tierb-collect`"))
    if EVALUATOR_BACKENDS.get(wf.gate.get("evaluator")) != "cpu_stub":
        problems.append(Problem("SPEC-REF", "stages[1].evaluator",
                                f"evaluator {wf.gate.get('evaluator')!r} has no Tier A backend",
                                f"use one of {', '.join(EVALUATOR_BACKENDS)}"))
    base = wf.path_of("model")
    if not (base / model_mod.MODEL_FILE).is_file() or not (base / model_mod.CONFIG_FILE).is_file():
        problems.append(Problem("SPEC-REF", f"artifacts.{names['model']}.uri",
                                f"{base} is not a cpu_stub checkpoint (no {model_mod.MODEL_FILE})",
                                "create it with `gatedtrain warmup <warmup.yaml> --out <dir>`"))
    data = wf.path_of("data")
    if not data.is_dir() or not list(data.glob("*.jsonl")):
        problems.append(Problem("SPEC-REF", f"artifacts.{names['data']}.uri", f"{data} holds no *.jsonl shards",
                                "point the training dataset at a directory of shard files"))
    prompts = wf.path_of("items")
    if not (prompts / "prompts.jsonl").is_file():
        problems.append(Problem("SPEC-REF", f"artifacts.{names['items']}.uri", f"{prompts}/prompts.jsonl is missing",
                                "point the item set at a directory holding prompts.jsonl"))
    cfg_path = wf.path_of("config")
    try:
        cfg = trainer_mod.load_config(cfg_path, kind="trainer")
    except trainer_mod.ConfigError as exc:
        key = exc.field.split(".", 1)[1] if "." in exc.field else exc.field
        problems.append(Problem("TRAINER-CONFIG", f"artifacts.{names['config']}.{key}", exc.message, exc.fix))
        cfg = None
    if problems:
        raise Refused(problems)
    # Under drop_last (decision D-1; nemo_rl/algorithms/grpo.py:883-890, drop_last=True at :888) a batch
    # larger than the dataset is never yielded, so the stage could not train. Rows are counted from the
    # shard files; the model is not loaded.
    rows = sum(n for _, n in shard_table(data))
    if cfg["batch_size"] > rows:
        raise Refused([Problem(
            "TRAINER-CONFIG", f"artifacts.{names['config']}.batch_size",
            f"batch_size {cfg['batch_size']} exceeds the {rows} training rows of {data}; with drop_last no "
            "batch is ever yielded",
            f"set batch_size to at most {rows}")])

    problems.extend(check_pins(wf))
    prompt_hash = manifest_hash(prompts)
    registered = item_set_hash(wf.registration)
    if prompt_hash != registered:
        problems.append(Problem("REG-FIELD", item_set_field(wf.registration),
                                f"registered item set {registered} but {prompts} is {prompt_hash} (05 section 3.7)",
                                "evaluate on the registered item set, or write a new registration"))
    if problems:
        raise Refused(problems)
    return {"item_set_hash": prompt_hash}


def check_pins(wf: Workflow, excludes: dict[str, tuple[str, ...]] | None = None) -> list[Problem]:
    """Artifacts are consumed by manifest hash (04:80): a non-registration `@sha256:` pin must match. `excludes` maps
    an artifact name to the top-level names its manifest leaves out (an ext adapter's `base_excludes`, EXT-162), so
    the pin is the same hash the adapter computes for that directory."""
    problems = []
    for name, pin in wf.artifact_pins.items():
        if name == wf.names.get("registration"):
            continue
        p = wf.artifact_paths[name]
        if p.is_dir():
            skip = (excludes or {}).get(name, ())
            try:
                actual = entries_hash(manifest_entries(p, exclude=skip)) if skip else manifest_hash(p)
            except (OSError, ValueError) as exc:
                problems.append(Problem("ARTIFACT-HASH", f"artifacts.{name}.uri", f"{p} cannot be hashed: {exc}",
                                        "keep the artifact a plain readable directory with no symbolic links"))
                continue
        elif p.is_file():
            actual = file_sha256(p)
        else:
            actual = None
        if actual != pin:
            problems.append(Problem("ARTIFACT-HASH", f"artifacts.{name}.uri", f"pinned {pin} but found {actual}",
                                    "restore the pinned content or update the pin"))
    return problems


def _new_run_dir(runs_root: Path) -> Path:
    runs_root.mkdir(parents=True, exist_ok=True)
    for _ in range(100):
        run_id = _dt.datetime.now(_dt.timezone.utc).strftime("run-%Y%m%dT%H%M%SZ-") + secrets.token_hex(4)
        d = runs_root / run_id
        try:
            d.mkdir()
            return d
        except FileExistsError:
            continue
    raise RuntimeError(f"could not create a new run directory under {runs_root}")


def _log(msg: str) -> None:
    print(f"[gatedtrain] {msg}", file=sys.stderr, flush=True)


def copy_registration(wf: Workflow, run_dir: Path) -> None:
    """Byte copy of the registration, SHA-256 checked (05:142). Decision D-4 and finding L-4(a): a copy
    that fails or hashes differently is a refusal, and the just-created run directory is removed."""
    (run_dir / "registration").mkdir()
    reg_copy = run_dir / "registration" / "registration.yaml"
    uri_field = f"artifacts.{wf.names['registration']}.uri"
    try:
        shutil.copyfile(wf.registration_path, reg_copy)
        copied = file_sha256(reg_copy)
    except OSError as exc:
        shutil.rmtree(run_dir)
        raise Refused([Problem(
            "REG-MISSING", uri_field,
            f"the registration could not be copied after validation ({exc.__class__.__name__}: {exc}); "
            "no run directory was kept",
            "restore the registration file and run again; do not move or edit it during a run")])
    if copied != wf.registration_sha256:
        shutil.rmtree(run_dir)
        raise Refused([Problem(
            "REG-HASH", uri_field,
            f"the registration changed while the run copied it: recorded {wf.registration_sha256}, "
            f"copied {copied}; no run directory was kept",
            "stop editing the registration during a run; an amendment is a new registration (05 section 3.7)")])


def item_ids(prompts: Path) -> list[int]:
    """The registered item set of a prompts.jsonl item-set directory: the 0-based line numbers, which are
    NeMo Evaluator's `problem_idx` and the cpu_stub evaluator's (benchmarks/gsm8k.py at ea35ba3)."""
    n = sum(1 for line in text_lines((prompts / "prompts.jsonl").read_text(encoding="utf-8")) if line.strip())
    return list(range(n))


def write_plan(wf: Workflow, run_dir: Path, prompts: Path, *, n_available: int | None = None) -> dict[str, Any]:
    """plan() under the registered sigma plan (05 section 3.6; MVP-SPEC section 2.3). `n_available` defaults to
    the lines of `<prompts>/prompts.jsonl`; ext-collect passes the size of its `items.jsonl` item set."""
    if n_available is None:
        n_available = sum(1 for line in text_lines((prompts / "prompts.jsonl").read_text(encoding="utf-8"))
                          if line.strip())
    the_plan = plan(wf.clause, wf.registration["sigma_pilot"], n_available=n_available)
    write_canonical(run_dir / "plan.json", {"schema": "gatedtrain.plan", "schema_version": "1.0", "tags": list(TAGS),
                                            **the_plan})
    if the_plan["underpowered"]:
        _log(f"plan: underpowered up front: n_available {n_available} < n_required {the_plan['n_required']}")
    return the_plan


def write_lineage(wf: Workflow, run_dir: Path, *, base: Path, data: Path, cfg_path: Path,
                  saved: list[tuple[int, Path]], item_set_hash: str,
                  known_hashes: dict[Path, str] | None = None, orphaned: list[dict] | None = None) -> None:
    """Lineage edges for the impact query (05 section 11.3; layout in gatedtrain.impact).

    `orphaned` (from ledger.orphaned_rows) are the ConsumptionRecords after the attempt's last
    committed cut (05 section 12.4, 05:424); they are kept in lineage/orphaned.parquet for the
    impact query's counts (R10A-9) and never enter the ledger or a rollup."""
    run_id, stage_id, gate_id = run_dir.name, wf.stage["id"], wf.gate["id"]
    snapshot = manifest_hash(data)
    known = known_hashes or {}
    base_hash = known.get(base) or manifest_hash(base)
    ckpt_rows = []
    for step, d in saved:
        try:
            rel = d.relative_to(run_dir).as_posix()
        except ValueError:
            rel = str(d)                                  # Tier B checkpoints stay where NeMo RL wrote them
        ckpt_rows.append({"run_id": run_id, "stage_id": stage_id, "checkpoint_hash": known.get(d) or manifest_hash(d),
                          "last_step_included": step, "path": rel})
    shard_rows = [{"run_id": run_id, "stage_id": stage_id, "snapshot": snapshot, "shard": name,
                   "shard_sha256": file_sha256(data / name)} for name, _ in shard_table(data)]
    (run_dir / "lineage").mkdir()
    pq.write_table(pa.Table.from_pylist(ckpt_rows), run_dir / "lineage" / "checkpoints.parquet")
    pq.write_table(pa.Table.from_pylist(shard_rows), run_dir / "lineage" / "shards.parquet")
    orphan_rows = [{"run_id": run_id, "stage_id": stage_id, **r} for r in orphaned or []]
    pq.write_table(pa.Table.from_pylist(orphan_rows, schema=ORPHAN_SCHEMA), run_dir / "lineage" / "orphaned.parquet")
    write_canonical(run_dir / "lineage" / "edges.json", {
        "schema": "gatedtrain.lineage", "schema_version": "1.0", "tags": list(TAGS),
        "activities": [
            {"id": f"{run_id}/{stage_id}", "kind": "training_stage", "uses_plugin": wf.stage["uses"],
             "used": [{"role": "model", "hash": base_hash},
                      {"role": "data", "hash": snapshot, "precision": "iteration",
                       "ledger": "ledger/consumption.parquet"},
                      {"role": "config", "hash": file_sha256(cfg_path)}],
             "generated": [{"role": "model", "hash": r["checkpoint_hash"],
                            "last_step_included": r["last_step_included"]} for r in ckpt_rows]},
            {"id": f"{run_id}/{gate_id}", "kind": "gate", "uses_plugin": wf.gate["uses"],
             "used": [{"role": "model", "hash": ckpt_rows[-1]["checkpoint_hash"]},
                      {"role": "baseline", "hash": base_hash},
                      {"role": "items", "hash": item_set_hash},
                      {"role": "registration", "hash": wf.registration_sha256}],
             "generated": [{"role": "gate_report", "path": "report/gate_report.json"}]},
        ],
    })


def write_manifests(run_dir: Path, *, base: Path, final_ckpt: Path, data: Path, prompts: Path,
                    known_entries: dict[Path, list] | None = None) -> dict[str, str]:
    """Content-addressed manifests for every artifact but the gate report."""
    roots = {
        "baseline_checkpoint": base, "trained_checkpoint": final_ckpt, "training_shards": data, "item_set": prompts,
        "registration": run_dir / "registration", "eval_baseline": run_dir / "eval" / "baseline",
        "eval_candidate": run_dir / "eval" / "candidate", "ledger": run_dir / "ledger",
    }
    hashes: dict[str, str] = {}
    for name, root in roots.items():
        doc = manifest_document(name, MANIFEST_TYPES[name], root, TAGS,
                                entries=(known_entries or {}).get(root))
        write_canonical(run_dir / "manifests" / f"{name}.json", doc)
        hashes[name] = doc["manifest_hash"]
    return hashes


# A record that stands for a line the gate cannot read as a NeMo Evaluator record (R11A-28): its unknown key
# makes `gates.evaluate_items` report `malformed_record` (R11A-7).
MALFORMED_LINE = "unreadable_results_line"


def read_arm_records(arm_dir: Path, *, scorer_key: str | None = None) -> list[dict[str, Any]]:
    """Every non-blank line of `<arm_dir>/results.jsonl` as a per-item record (R11A-28).

    Unlike `gates.load_paired_records` (NeMo Evaluator's reader: bad lines skipped, the last duplicate
    wins), nothing is dropped or overwritten: a line that does not parse, is not a JSON object or has no
    `problem_idx` becomes a malformed record, and a second record for one (problem_idx, repeat) reaches the
    gate as a duplicate; both make the clause MISSING `malformed_record`. A missing file gives no records
    (every item is then missing, `absent`)."""
    path = Path(arm_dir) / "results.jsonl"
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for n, record in results_lines(path):
        if record is None:
            out.append({MALFORMED_LINE: n})
            continue
        out.append(from_nemo_evaluator(record, scorer_key=scorer_key))
    return out


def results_lines(path: Path, text: str | None = None) -> list[tuple[int, dict[str, Any] | None]]:
    """The strict per-line rule of `read_arm_records` (R11A-28), shared with `tierb.merge_eval`: every non-blank
    line of a NeMo Evaluator `results.jsonl` with its 1-based line number, and the parsed record, or None when
    the line does not parse, is not a JSON object or has no `problem_idx`. Blank lines are skipped; nothing is
    coerced, dropped or overwritten (duplicates are the caller's to report). `text`: the content, already read."""
    out: list[tuple[int, dict[str, Any] | None]] = []
    if text is None:
        text = Path(path).read_text(encoding="utf-8")
    for n, line in enumerate(text_lines(text), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            record = None
        if not isinstance(record, dict) or record.get("problem_idx") is None:
            record = None
        out.append((n, record))
    return out


def gate_and_report(wf: Workflow, run_dir: Path, *, the_plan: dict, hashes: dict[str, str], lock_bytes: bytes,
                    item_set_hash: str, evaluator_hash: str, stubs: list[str], items: list,
                    extra: dict[str, Any] | None = None,
                    scorer_key: str | None = None, records: dict[str, list] | None = None,
                    model_hashes: dict[str, str] | None = None,
                    expected_models: dict[str, str] | None = None,
                    clusters: dict | None = None) -> tuple[dict[str, Any], str]:
    """Gate on per-item records paired by (item, repeat), then the report (05 section 3.7).

    The NeMo Evaluator-shaped records of each arm (comparison.py:360 at ea35ba3) are converted by
    `items.from_nemo_evaluator`: item = problem_idx, seed = metadata.eval_seed, score = `reward`, or with
    `scorer_key` (Tier B, interim rule G-21) `scorer_rewards[scorer_key]`; the records on disk are not
    changed. `items` is the registered item set (R11A-17); the evaluator's declared score range bounds the
    scores (R11-11).

    Design revision 10: `min_paired` and `pairing.max_unpaired` come from the registration (R10A-3,
    R10A-5); a registered evaluator hash that differs from `evaluator_hash` makes the gate MISSING
    (05 section 6.1; R10A-2); a `post_hoc: true` registration cannot PASS (R10A-4).

    The registered hash is the deciding evaluator's (the entry whose id
    is the gate's `evaluator`), and the arms are read by `read_arm_records`, which keeps every line.

    External runs (tests/README.md section 11.18; MULTIMODAL-SPEC MM-7, MM-17): `records` gives each arm's
    per-item records directly (already checked by `ext.arm`), `model_hashes` each arm's evaluated
    `model_hash` (eval_meta.checkpoint), and `expected_models` the model hashes the gate expects (EXT-11)."""
    clause, reg = wf.clause, wf.registration
    arms = {}
    for arm, ckpt in (("baseline", "baseline_checkpoint"), ("candidate", "trained_checkpoint")):
        recs = (records[arm] if records is not None
                else read_arm_records(run_dir / "eval" / arm, scorer_key=scorer_key))
        arms[arm] = {"records": recs,
                     "model_manifest_hash": model_hashes[arm] if model_hashes is not None else hashes[ckpt]}
    boot = reg["bootstrap"]
    result = evaluate_items(clause, arms["baseline"], arms["candidate"], items=items, plan=the_plan,
                            seed=int(boot["seed"]), n_boot=int(boot["replicates"]), min_paired=min_paired(reg),
                            max_unpaired=max_unpaired(reg), score_range=score_range(wf.gate.get("evaluator")),
                            expected_models=expected_models, **({"clusters": clusters} if clusters else {}))
    deciding = deciding_evaluator(reg, wf.gate.get("evaluator")) or {}
    registered_hash = deciding.get("manifest_hash")
    if registered_hash != evaluator_hash:
        reasons = [*result["status_reasons"], "evaluator_hash_mismatch"]
        result = {**result, "status": MISSING, "status_reasons": reasons}
    if reg.get("post_hoc") is True:
        result = apply_post_hoc(result)
    report = build_report(
        workflow=wf.spec["workflow"], gate_id=wf.gate["id"], clause=clause, result=result, plan=the_plan,
        registration=normalize(reg),
        hashes={
            "registration_hash": wf.registration_sha256,
            "item_set_hash": item_set_hash,
            "evaluator_hash": evaluator_hash,
            "evaluator_hash_registered": registered_hash,
            "sampler_hash": sha256_bytes(canonical_json(reg["sampler"])),
            "baseline_checkpoint_hash": hashes["baseline_checkpoint"],
            "candidate_checkpoint_hash": hashes["trained_checkpoint"],
            "lockfile_hash": sha256_bytes(lock_bytes),
        },
        disjoint_declared=any(isinstance(e, dict) and e.get("disjoint_from")
                              for e in wf.spec["artifacts"].values()),
        stubs=stubs, created_at=_utc_now(), run_id=run_dir.name, extra=extra,
        amendment=getattr(wf, "amendment", None),
    )
    text = render_text(report)
    (run_dir / "report").mkdir()
    write_canonical(run_dir / "report" / "gate_report.json", report)
    (run_dir / "report" / "gate_report.txt").write_text(text, encoding="utf-8")
    write_canonical(run_dir / "manifests" / "gate_report.json",
                    manifest_document("gate_report", MANIFEST_TYPES["gate_report"], run_dir / "report", TAGS))
    _log(f"gate {wf.gate['id']}: {report['status']}")
    return report, text


def write_plans(wf: Workflow, run_dir: Path, *, n_available: int) -> list[dict[str, Any]]:
    """plan() of every clause under its own sigma plan (05 section 3.6; R18-1), written as one plan.json with the
    clauses in order and, beside them, the joint power of independent clauses (the product of the registered powers,
    05 section 3.5)."""
    plans = [plan(c, pilot_for(wf.registration, c), n_available=n_available) for c in wf.clauses]
    joint = 1.0
    for c in wf.clauses:
        joint *= float(c["power"])
    write_canonical(run_dir / "plan.json", {
        "schema": "gatedtrain.plan", "schema_version": "1.0", "tags": list(TAGS), "multiplicity": MULTI_PLAN,
        "joint_power_independent": joint,
        "clauses": [{"name": c["name"], "evaluator": c["evaluator"], **pl} for c, pl in zip(wf.clauses, plans)]})
    for c, pl in zip(wf.clauses, plans):
        if pl["underpowered"]:
            _log(f"plan {c['name']}: underpowered up front: n_available {n_available} < n_required {pl['n_required']}")
    return plans


def gate_and_report_multi(wf: Workflow, run_dir: Path, *, plans: list[dict], hashes: dict[str, str],
                          lock_bytes: bytes, item_set_hash: str, evaluator_hashes: dict[str, str], items: list,
                          records: list[dict[str, list]], model_hashes: list[dict[str, str]],
                          expected_models: dict[str, str], clusters: dict[str, dict],
                          extra: dict[str, Any] | None = None,
                          clause_extra: list[dict[str, Any]] | None = None) -> tuple[dict[str, Any], str]:
    """Design revision 18 (05 sections 3.5, 3.8; R18-1, R18-2, R18-6): gate every clause on its own evaluator's
    records, as `gate_and_report` gates the one clause (registered evaluator hash, post hoc), then combine the
    clause statuses by `combine(..., "and")`: FAIL if any clause fails, else MISSING if any is missing, else
    INCONCLUSIVE if any is inconclusive, else PASS. `records[i]` and `model_hashes[i]` are clause i's arms, and
    `evaluator_hashes` maps each evaluator to the hash the run resolved;
    `clusters` maps a clause name to its {item: cluster} when it clusters (R18-3)."""
    reg = wf.registration
    boot = reg["bootstrap"]
    results = []
    for i, (clause, the_plan) in enumerate(zip(wf.clauses, plans)):
        ev = clause["evaluator"]
        arms = {arm: {"records": records[i][arm], "model_manifest_hash": model_hashes[i][arm]}
                for arm in ("baseline", "candidate")}
        kw = {"clusters": clusters[clause["name"]]} if clause["name"] in clusters else {}
        result = evaluate_items(clause, arms["baseline"], arms["candidate"], items=items, plan=the_plan,
                                seed=int(boot["seed"]), n_boot=int(boot["replicates"]), min_paired=min_paired(reg),
                                max_unpaired=max_unpaired(reg), score_range=score_range(ev),
                                expected_models=expected_models, **kw)
        registered_hash = (deciding_evaluator(reg, ev) or {}).get("manifest_hash")
        if registered_hash != evaluator_hashes[ev]:
            result = {**result, "status": MISSING, "status_reasons": [*result["status_reasons"],
                                                                      "evaluator_hash_mismatch"]}
        if reg.get("post_hoc") is True:
            result = apply_post_hoc(result)
        results.append((clause, result, the_plan, registered_hash))
    status = combine([r for _, r, _, _ in results], "and")
    reasons = [f"{c['name']}: {r['status']}" for c, r, _, _ in results if r["status"] != PASS]
    joint = 1.0
    for c in wf.clauses:
        joint *= float(c["power"])
    report = build_multi_report(
        workflow=wf.spec["workflow"], gate_id=wf.gate["id"], clauses=results, status=status, status_reasons=reasons,
        joint_power=joint, registration=normalize(reg),
        hashes={
            "registration_hash": wf.registration_sha256,
            "item_set_hash": item_set_hash,
            "evaluator_hashes": evaluator_hashes,
            "sampler_hash": sha256_bytes(canonical_json(reg["sampler"])),
            "baseline_checkpoint_hash": hashes["baseline_checkpoint"],
            "candidate_checkpoint_hash": hashes["trained_checkpoint"],
            "lockfile_hash": sha256_bytes(lock_bytes),
        },
        disjoint_declared=any(isinstance(e, dict) and e.get("disjoint_from")
                              for e in wf.spec["artifacts"].values()),
        stubs=[], created_at=_utc_now(), run_id=run_dir.name, extra=extra, clause_extra=clause_extra,
        amendment=getattr(wf, "amendment", None),
    )
    text = render_text(report)
    (run_dir / "report").mkdir()
    write_canonical(run_dir / "report" / "gate_report.json", report)
    (run_dir / "report" / "gate_report.txt").write_text(text, encoding="utf-8")
    write_canonical(run_dir / "manifests" / "gate_report.json",
                    manifest_document("gate_report", MANIFEST_TYPES["gate_report"], run_dir / "report", TAGS))
    summary = ", ".join(f"{c['name']}: {r['status']}" for c, r, _, _ in results)
    _log(f"gate {wf.gate['id']}: {report['status']} ({summary})")
    return report, text


def run(spec_path: Path, runs_root: Path) -> tuple[Path, dict[str, Any], str]:
    """Validate, then run the whole path (Tier A). Returns (run_dir, report, text report)."""
    import torch

    from .stubs import evaluator as evaluator_mod
    from .stubs import trainer as trainer_mod

    wf = load_spec(spec_path)
    pre = preflight(wf)
    clause, reg = wf.clause, wf.registration
    base, data, prompts, cfg_path = (wf.path_of(r) for r in ("model", "data", "items", "config"))

    run_dir = _new_run_dir(Path(runs_root).absolute())   # keep the user's path; no symlink resolution
    run_id = run_dir.name
    stage_id = wf.stage["id"]
    copy_registration(wf, run_dir)
    _log(f"run {run_id}: {wf.spec['workflow']} (prototype, unpromotable; stand-ins are {STUB_MARK})")
    the_plan = write_plan(wf, run_dir, prompts)

    sampler = reg["sampler"]
    eval_kw = {"eval_seeds": list(clause["eval_seeds"]), "temperature": float(sampler["temperature"]),
               "max_new_tokens": int(sampler["max_new_tokens"])}

    # 4. Baseline evaluation, K samples per prompt, registered seeds (05:97).
    _log("baseline evaluation")
    evaluator_mod.evaluate(base, prompts, run_dir / "eval" / "baseline", **eval_kw)

    # 5. Training stage.
    _log(f"training stage {stage_id} ({wf.stage['uses']})")
    stage_dir = run_dir / "stage" / stage_id
    saved = trainer_mod.run_stage(base, data, cfg_path, stage_dir)
    final_step, final_ckpt = saved[-1]

    # 6. Candidate evaluation on the same prompts, K and seeds.
    _log(f"candidate evaluation (step_{final_step})")
    evaluator_mod.evaluate(final_ckpt, prompts, run_dir / "eval" / "candidate", **eval_kw)

    # 9. Ledger from the trainer's own outputs (05 section 12.2; G-1), then lineage.
    write_ledger(stage_dir / "logs", stage_dir / "checkpoints", data, run_dir / "ledger" / "consumption.parquet",
                 run_id=run_id, attempt_id="0")
    write_lineage(wf, run_dir, base=base, data=data, cfg_path=cfg_path, saved=saved,
                  item_set_hash=pre["item_set_hash"],
                  orphaned=orphaned_rows(stage_dir / "logs", stage_dir / "checkpoints", data, attempt_id="0"))

    # 8. Manifests and the lockfile.
    hashes = write_manifests(run_dir, base=base, final_ckpt=final_ckpt, data=data, prompts=prompts)
    backend = {"name": "cpu_stub", "package": PACKAGE, "package_version": __version__,
               "torch_version": torch.__version__, "stub": STUB_MARK}
    lock = {
        "schema": "gatedtrain.lockfile", "schema_version": "1.0", "tags": list(TAGS), "run_id": run_id,
        "spec_sha256": file_sha256(wf.spec_path), "registration_sha256": wf.registration_sha256,
        "config_hash": file_sha256(cfg_path),
        "allow_list_checked": False,                       # no allow-list checks in the prototype
        "backend": {"trainer": {**backend, "uses": wf.stage["uses"]},
                    "evaluator": {**backend, "uses": wf.gate["evaluator"]}},
        "analysis_code_version": ANALYSIS_CODE_VERSION,
        "python_version": sys.version.split()[0],
        "artifacts": dict(sorted(hashes.items())),
    }
    lock_bytes = write_canonical(run_dir / "lockfile.json", lock)

    # 7. Gate and report.
    report, text = gate_and_report(wf, run_dir, the_plan=the_plan, hashes=hashes, lock_bytes=lock_bytes,
                                   item_set_hash=pre["item_set_hash"], evaluator_hash=stub_evaluator_hash(),
                                   stubs=STUBS, items=item_ids(prompts))
    return run_dir, report, text
