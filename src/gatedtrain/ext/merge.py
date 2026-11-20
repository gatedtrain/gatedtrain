"""`gatedtrain ext-merge-eval`: single- or multi-seed per-item arms into one arm (MULTIMODAL-SPEC.md MM-5;
tests/README.md section 11.18 EXT-10).

Repeats are relabelled by the registered seed order (G-16 semantics): a record evaluated with seed s becomes
repeat `eval_seeds.index(s)`. `eval_meta.eval_seeds` of the merged arm is the registered seed list, and
`merge.json` records each source's path, seeds and file hashes. Refused (`EXT-MERGE`, nothing written): a run
missing or empty; a registered seed not covered or covered twice; a seed outside the registration; a record
seed other than its run's `eval_seeds[repeat]` (or any other broken line, `ext.arm`); item sets that differ
between repeats; eval_meta values (checkpoint, scorer, sampler hash, normalizer, environment version, and every
other key but the seeds, compared as canonical JSON, EXT-44) that differ between runs; an `output_ref` naming a
file of the arm itself (EXT-45); an existing `--out`.

The registration is read with the evaluator the runs name (`eval_meta.evaluator_id`) as the deciding
evaluator (R11A-23); it is never routed through `tierb.load_registration` (R12A-27).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import yaml

from .. import TAGS
from ..canonical import write_canonical
from ..errors import Problem, Refused
from . import registry
from .armfs import ArmSnapshot
from .arm import copy_output_ref, items_text, merge_records, read_lines, read_meta, same_json

MERGE_SCHEMA = "gatedtrain.ext_merge_eval"


def load_registration(path: Path, evaluator_id: str | None, *, rule: str) -> dict:
    """Parse a registration and check its contents with `evaluator_id` deciding (R11A-23). Refuses a
    constraint clause (REG-CONSTRAINT, R12A-20) and an evaluator the registration does not list. Returns the
    registration in revision 11 names."""
    from ..registration import check_contents, clause_for_evaluator, deciding_evaluator, normalize

    try:
        reg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise Refused([Problem("REG-MISSING", "--registration", f"cannot read {path}: {exc.__class__.__name__}",
                               "pass the registration file")])
    if not isinstance(reg, dict):
        raise Refused([Problem("REG-FIELD", "registration", "the registration is not a mapping",
                               "write the registration as a YAML mapping")])
    problems = check_contents(reg, evaluator_id=evaluator_id)
    if evaluator_id is not None and deciding_evaluator(reg, evaluator_id) is None:
        problems.append(Problem(rule, "registration.evaluators", f"the registration does not list evaluator "
                                f"{evaluator_id} (R11A-23)", f"list `{{id: {evaluator_id}, manifest_hash: <hash>}}`"))
    elif evaluator_id is not None and not problems and clause_for_evaluator(reg, evaluator_id) is None:
        problems.append(Problem(rule, "registration.clauses", f"no clause of the registration is decided by evaluator "
                                f"{evaluator_id} (R18-1)", f"add a clause with `evaluator: {evaluator_id}`"))
    if isinstance(reg.get("clause"), dict) and reg["clause"].get("constraint_set") is not None:
        problems.append(Problem("REG-CONSTRAINT", "registration.clause.constraint_set",
                                "constraint clauses are not wired into the external-run commands in this slice "
                                "(tests/README.md section 11.18, deferred)",
                                "gate a constraint clause with gatedtrain.constraints.evaluate_constraint"))
    if problems:
        raise Refused(problems)
    return normalize(reg)


def merge_eval(registration: Path, out: Path, runs: list[Path]) -> Path:
    """Merge per-item arm runs into `out` (EXT-10). Raises Refused and writes nothing on any problem."""
    out = Path(out)
    fix = "pass finished arm directories that together cover each registered seed once"
    if out.exists():
        raise Refused([Problem("EXT-MERGE", "--out", f"{out} already exists", "pass a new --out directory")])
    problems: list[Problem] = []
    infos = []
    for r, run in enumerate(map(Path, runs)):
        fld = f"runs[{r}]"
        snap = ArmSnapshot.read(run)                     # read once: checked, hashed into merge.json (A5)
        meta, why = read_meta(run, snap)
        if meta is None or why:
            problems += [Problem("EXT-MERGE", fld, f"{run}: {m}", fix) for m in why]
            continue
        infos.append({"run": run, "meta": meta, "field": fld, "snap": snap})
    ids = sorted({i["meta"].get("evaluator_id") for i in infos}, key=str)
    if len(ids) > 1:
        problems.append(Problem("EXT-MERGE", "runs", f"the runs name different evaluators {ids}", fix))
    evaluator = ids[0] if len(ids) == 1 else None
    if evaluator is not None and evaluator not in registry.EVALUATOR_ROWS:
        problems.append(Problem("EXT-MERGE", "runs", f"evaluator {evaluator!r} is not in the adapter registry", fix))
        evaluator = None
    if problems:
        raise Refused(problems)
    reg = load_registration(registration, evaluator, rule="EXT-MERGE")
    from ..registration import clause_for_evaluator

    seeds = list(clause_for_evaluator(reg, evaluator)["eval_seeds"])    # R18-1: the clause it decides
    row = registry.EVALUATOR_ROWS[evaluator]

    covered: dict[int, str] = {}
    for info in infos:
        run, meta, fld = info["run"], info["meta"], info["field"]
        run_seeds = list(meta["eval_seeds"])
        for s in run_seeds:
            if s not in seeds:
                problems.append(Problem("EXT-MERGE", fld, f"seed {s} is not a registered seed {seeds}", fix))
            elif s in covered:
                problems.append(Problem("EXT-MERGE", fld, f"seed {s} is covered twice ({covered[s]} and {fld})", fix))
            else:
                covered[s] = fld
        recs, bad = read_lines(run, seeds=run_seeds, score_range=row["score_range"],
                               error_classes=tuple(row["error_classes"]), snap=info["snap"])
        problems += [Problem("EXT-MERGE", fld, f"{run}: {m}", fix) for m in bad]
        info["records"] = recs
    for s in seeds:
        if s not in covered:
            problems.append(Problem("EXT-MERGE", "runs", f"registered seed {s} is covered by no run", fix))
    # Every eval_meta key but the seeds (and the wall clock) must agree between runs (MM-5).
    first = infos[0]["meta"]
    for info in infos[1:]:
        for key in sorted((set(first) | set(info["meta"])) - {"eval_seeds", "created_at"}):
            if not same_json(first.get(key), info["meta"].get(key)):      # EXT-44: true is not 1
                problems.append(Problem("EXT-MERGE", info["field"], f"eval_meta {key} differs from runs[0]: "
                                        f"{info['meta'].get(key)!r} vs {first.get(key)!r}", fix))
    if not problems:
        # Item sets (MM-5): every (run, repeat) evaluated the same items.
        sets = {}
        for info in infos:
            for r in range(len(info["meta"]["eval_seeds"])):
                sets[(info["field"], r)] = {rec["item"] for rec in info["records"] if rec["repeat"] == r}
        ref_key = next(iter(sets))
        for key, items in sets.items():
            if items != sets[ref_key]:
                problems.append(Problem("EXT-MERGE", key[0], f"repeat {key[1]} evaluated {len(items)} items, not the "
                                        f"{len(sets[ref_key])} items of {ref_key[0]} repeat {ref_key[1]}", fix))
    if problems:
        raise Refused(problems)

    outputs = {}
    for info in infos:
        for rec in info["records"]:
            ref = rec.get("output_ref")
            if ref:
                if ref["path"] in outputs and outputs[ref["path"]][1]["sha256"] != ref["sha256"]:
                    problems.append(Problem("EXT-MERGE", info["field"], f"output {ref['path']} differs between runs",
                                            "give each run's outputs distinct paths"))
                outputs[ref["path"]] = (info, ref)      # canonical paths (read_lines): one spelling per file
    if problems:
        raise Refused(problems)
    merged = merge_records([(info["records"], info["meta"]["eval_seeds"]) for info in infos], seeds)

    try:                                                  # EXT-75: the parent cannot hold the output
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.tmp-", dir=out.parent))
    except OSError as exc:
        raise Refused([write_problem(exc, "--out", out)]) from exc
    try:
        (tmp / "items.jsonl").write_text(items_text(merged), encoding="utf-8")
        meta = {k: v for k, v in first.items() if k != "created_at"}
        meta["eval_seeds"] = seeds
        write_canonical(tmp / "eval_meta.json", meta)
        for rel, (info, ref) in outputs.items():                # MM-1: the checked bytes, never a swapped file
            (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            copy_output_ref(info["run"], ref, tmp / rel, flag=info["field"], key=info["snap"].key)
        write_canonical(tmp / "merge.json", {
            "schema": MERGE_SCHEMA, "schema_version": "1.0", "tags": list(TAGS),
            "sources": [{"path": os.path.abspath(i["run"]), "seeds": list(i["meta"]["eval_seeds"]),
                         "items_sha256": i["snap"].files["items.jsonl"].sha256,              # the bytes merged
                         "eval_meta_sha256": i["snap"].files["eval_meta.json"].sha256} for i in infos],
            "notes": ["Repeats relabelled by the registered seed order (G-16 semantics, MULTIMODAL-SPEC MM-5)."],
        })
        os.rename(tmp, out)
    except OSError as exc:                  # EXT-59, EXT-75: a failed write is a refusal; other errors are bugs
        shutil.rmtree(tmp, ignore_errors=True)
        raise Refused([write_problem(exc, "--out", out)]) from exc
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return out


def write_problem(exc: Exception, field: str, target: Path) -> Problem:
    """EXT-59: the refusal for a write that failed after every check passed (the partial output is removed)."""
    return Problem("EXT-WRITE", field, f"writing {target} failed after every check passed "
                   f"({exc.__class__.__name__}: {exc}); the partial output was removed",
                   "fix the cause (disk space, permissions) and run the command again")
