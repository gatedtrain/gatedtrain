"""AN-5: the stage machine.

Runs a project through its validated pipeline. State lives only in the project's artifacts and `attempts.jsonl`
(`status` is derived from them), so a new process after a crash sees the same status.

Attempt lines: `{stage, attempt, key, event, utc}` with `event` one of `start` (also `trains`: whether the stage trains, resolved at run time),
`done` (also lists the outputs),
`failed` (also names the error), `refused` (a stage refused in `run` before it wrote any record or spent alpha: also the rules and
the first reason; it is audit only and `_states` ignores it, so a refused attempt, such as a second gate or an
amendment after the alpha is spent, never changes the derived state) and `stale` (an audit note that a rerun of an
earlier stage invalidated this one). A refusal after the stage wrote records, spent alpha or changed a trust marker
(the alpha claim, the latest registration, the claim of the sealed evaluation), or raised by the output checks, is
`failed` (with its rules and reason).
A `done` line records the key parts (`stage_type`, `version`, `params_sha256`, `input_sha256s`, `decision_sha256`).
Staleness is derived from those parts, never from `stale` lines: a done stage is `stale` when its recorded type,
version, params or any input hash differs from the current spec, registry or the producer's current output (in
pipeline order, so it propagates), or when a producer is no longer done.

The idempotence key is the sha256 of (stage name, version, params, input artifact hashes, decision without its
`rerun` flag). A done stage with the same key returns its recorded outputs without calling `run`. A stage that is
done under a different key, or whose own recorded params, version or type differ from the current spec or
registry, refuses unless `decision["rerun"]` is true (a rerun appends `stale` audit lines for later stages); a
stage that is stale only because of upstream changes reruns without the flag.

The alpha ledger claims its single spend before it writes the ledger line; a crash in between leaves the budget
spent (`available()` stays 0.0) instead of risking a second spend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..canonical import canonical_json, sha256_bytes
from ..errors import Problem, Refused
from .pipeline import PipelineSpec, StageRef, _is_training, load_pipeline, pipeline_problems
from .project import ArtifactRef, Project, _utc_now
from .stage import Proposal, StageContext, stage_registry
from .trust import AlphaLedger, FixedAlphaLedger, InProcessTrust, TrustBackend


@dataclass(frozen=True)
class StageResult:
    stage: str
    key: str
    attempt: int
    outputs: dict[str, ArtifactRef]
    reused: bool


def _refuse(field: str, message: str, fix: str) -> Refused:
    return Refused([Problem("AN-5", field, message, fix)])


def _load_spec(project: Project) -> PipelineSpec:
    """`project.pipeline` is the path of the pipeline YAML (absolute, or relative to the project root). When the
    project records `pipeline_sha256` (`project create` copies the pipeline into the project), the file must still
    have that hash: changing the pipeline means a new project."""
    path = Path(project.pipeline)
    if not path.is_absolute():
        path = project.root / path
    if not path.is_file():
        raise _refuse("pipeline", f"the pipeline spec {project.pipeline!r} is not a file",
                      "recreate the project with the path of a pipeline YAML")
    if project.pipeline_sha256 is not None:
        got = sha256_bytes(path.read_bytes())
        if got != project.pipeline_sha256:
            raise _refuse("pipeline", f"the pipeline copy {project.pipeline!r} was changed after the project was "
                          f"created (recorded sha256 {project.pipeline_sha256}, file has {got})",
                          "restore the copy; to use another pipeline, create a new project")
    return load_pipeline(path)


def _validated(project: Project, spec: PipelineSpec | None, registry: dict | None) -> tuple[PipelineSpec, dict]:
    spec = spec if spec is not None else _load_spec(project)
    registry = registry if registry is not None else stage_registry()
    problems = pipeline_problems(spec, registry)
    if problems:
        raise Refused(problems)
    return spec, registry


def _states(project: Project, spec: PipelineSpec, registry: dict) -> dict[str, dict]:
    """Per stage: status, last attempt line, attempt count, and the recorded outputs and key of a done stage."""
    st: dict[str, dict] = {s.name: {"status": "pending", "last": None, "attempts": 0, "key": None, "outputs": None, "done_row": None,
                           "refusal": None} for s in spec.stages}
    before_start: dict[str, tuple] = {}
    for row in project.read_attempts():
        cur = st.get(row.get("stage"))
        if cur is None:
            continue
        ev = row.get("event")
        if ev == "refused":                   # audit only: undo the `start` it answers
            snap = before_start.pop(row["stage"], None)
            if snap is not None:
                cur["status"], cur["last"] = snap
            cur["refusal"] = {"attempt": row.get("attempt"), "rules": row.get("rules"), "reason": row.get("reason"),
                              "utc": row.get("utc")}
            continue
        cur["refusal"] = None                 # a later start, done or failed line supersedes the refusal
        before_start.pop(row.get("stage"), None)
        if ev == "start":
            before_start[row["stage"]] = (cur["status"], cur["last"])
        cur["last"] = row
        if ev == "start":
            cur["attempts"] = max(cur["attempts"], int(row.get("attempt", 0)))
            cur["status"] = "failed"          # a start without done or failed is an interrupted run
        elif ev == "done":
            cur["status"], cur["key"], cur["outputs"] = "done", row.get("key"), row.get("outputs")
            cur["done_row"] = row
        elif ev == "failed":
            cur["status"] = "failed"
        # `stale` lines are an audit trail only; staleness is derived below
    for sref in spec.stages:
        cur = st[sref.name]
        if cur["status"] == "done" and _is_stale(sref, cur, st, registry):
            cur["status"] = "stale"
    return st


def _params_sha(params: dict) -> str:
    return sha256_bytes(canonical_json(params))


def _own_changed(sref: StageRef, cur: dict, registry: dict) -> bool:
    """True when the stage's own recorded type, version or params differ from the current spec and registry."""
    cls = registry.get(sref.stage)
    row = cur["done_row"]
    if cls is None or row is None:
        return True
    return (row.get("stage_type") != sref.stage or row.get("version") != getattr(cls, "version", None)
            or row.get("params_sha256") != _params_sha(sref.params))


def _is_stale(sref: StageRef, cur: dict, st: dict[str, dict], registry: dict) -> bool:
    if _own_changed(sref, cur, registry):
        return True
    row = cur["done_row"]
    recorded = row.get("input_sha256s")
    if not isinstance(recorded, dict) or set(recorded) != set(sref.inputs):
        return True
    for key, src in sref.inputs.items():
        producer, _, name = src.partition(".")
        pst = st.get(producer)
        if pst is None or pst["status"] != "done":
            return True
        out = (pst["outputs"] or {}).get(name)
        if not isinstance(out, dict) or out.get("sha256") != recorded[key]:
            return True
    return False


def status(project: Project, *, spec: PipelineSpec | None = None, registry: dict | None = None) -> dict:
    spec = spec if spec is not None else _load_spec(project)
    registry = registry if registry is not None else stage_registry()
    return {name: {"status": s["status"], "last_attempt": s["last"], "last_refusal": s["refusal"]}
            for name, s in _states(project, spec, registry).items()}


def _ref_from_row(row: dict) -> ArtifactRef:
    return ArtifactRef(row["stage"], row["name"], row["schema"], row["sha256"], row["path"])


def _ref_row(ref: ArtifactRef) -> dict:
    return {"stage": ref.stage, "name": ref.name, "schema": ref.schema, "sha256": ref.sha256, "path": ref.path}


def _recorded_outputs(project: Project, state: dict) -> dict[str, ArtifactRef]:
    outs = state["outputs"] or {}
    refs = {name: _ref_from_row(row) for name, row in outs.items()}
    for ref in refs.values():
        project.read_artifact(ref)            # refuses a record changed after it was written
    return refs


def _locate(spec: PipelineSpec, ref: str) -> int:
    for i, s in enumerate(spec.stages):
        if s.name == ref:
            return i
    raise _refuse("stage", f"unknown stage {ref!r}", f"use one of: {', '.join(s.name for s in spec.stages)}")


def _check_ready(spec: PipelineSpec, states: dict[str, dict], i: int) -> None:
    for earlier in spec.stages[:i]:
        s = states[earlier.name]["status"]
        if s != "done":
            raise _refuse(f"stages[{i}]", f"stage {earlier.name!r} is {s}; stages run in pipeline order and each "
                          f"earlier stage must be done", f"run {earlier.name!r} first" if s != "stale"
                          else f"rerun {earlier.name!r}; an earlier rerun made it stale")


def _inputs(project: Project, spec: PipelineSpec, states: dict[str, dict], sref: StageRef) -> dict[str, ArtifactRef]:
    out: dict[str, ArtifactRef] = {}
    for key, src in sref.inputs.items():
        producer, _, name = src.partition(".")
        outs = _recorded_outputs(project, states[producer])
        if name not in outs:
            raise _refuse(f"inputs.{key}", f"stage {producer!r} recorded no output {name!r}", f"rerun {producer!r}")
        out[key] = outs[name]
    return out


def _key(sref: StageRef, version: str, inputs: dict[str, ArtifactRef], decision: dict) -> str:
    body = {k: v for k, v in decision.items() if k != "rerun"}
    doc = {"stage": sref.name, "type": sref.stage, "version": version, "params": sref.params,
           "inputs": {k: r.sha256 for k, r in sorted(inputs.items())}, "decision": body}
    try:
        return sha256_bytes(canonical_json(doc))
    except ValueError as exc:
        raise _refuse("decision", f"the decision or params are not canonical JSON: {exc}",
                      "use plain JSON values") from None


def _context(project: Project, sref: StageRef, inputs: dict[str, ArtifactRef], trust: TrustBackend | None,
             ledger: AlphaLedger | None) -> StageContext:
    return StageContext(project, inputs, dict(sref.params), trust if trust is not None else InProcessTrust(project),
                        ledger if ledger is not None else FixedAlphaLedger(project), sref.name)


def plan_stage(project: Project, ref: str, *, spec: PipelineSpec | None = None, registry: dict | None = None,
               trust: TrustBackend | None = None, ledger: AlphaLedger | None = None) -> Proposal:
    spec, registry = _validated(project, spec, registry)
    i = _locate(spec, ref)
    states = _states(project, spec, registry)
    _check_ready(spec, states, i)
    sref = spec.stages[i]
    ctx = _context(project, sref, _inputs(project, spec, states, sref), trust, ledger)
    return registry[sref.stage]().plan(ctx)


# Trust files whose change is a side effect of a stage: the alpha claim, the latest registration and the claim of
# the project's one sealed evaluation (the ledger is counted by lines, below).
_TRUST_MARKERS = ("trust/spent", "trust/registrations/latest", "trust/sealed_evaluation")


def _side_effects(project: Project, stage: str) -> tuple[dict[str, str], int, dict[str, str | None]]:
    """What a stage's `run` may have changed: the hash of every file under `stages/<stage>/`, the number of lines
    in the alpha ledger (a spend is a side effect) and the hash of each trust marker file (`_TRUST_MARKERS`)."""
    files: dict[str, str] = {}
    base = project.root / "stages" / stage
    if base.is_dir():
        for path in sorted(base.rglob("*")):
            if path.is_file() and not path.name.endswith(".tmp"):
                files[path.relative_to(base).as_posix()] = sha256_bytes(path.read_bytes())
    try:
        lines = (project.root / "trust" / "ledger.jsonl").read_bytes().count(b"\n")
    except OSError:
        lines = 0
    markers: dict[str, str | None] = {}
    for rel in _TRUST_MARKERS:
        try:
            markers[rel] = sha256_bytes((project.root / rel).read_bytes())
        except OSError:
            markers[rel] = None
    return files, lines, markers


def done_output(project: Project, stage: str, name: str, *, spec: PipelineSpec | None = None,
                registry: dict | None = None) -> dict | list | None:
    """The record of output `name` of `stage` when the stage is `done` (not stale, not failed), else None."""
    spec = spec if spec is not None else _load_spec(project)
    registry = registry if registry is not None else stage_registry()
    state = _states(project, spec, registry).get(stage)
    if state is None or state["status"] != "done":
        return None
    return project.read_artifact(_recorded_outputs(project, state)[name])


def recorded_output(project: Project, stage: str, name: str) -> dict | list | None:
    """The record of output `name` from the stage's last `done` line in the attempt log, whatever the stage's
    derived state is now (it may be stale); read with its hash check. None when the stage never finished."""
    row = None
    for r in project.read_attempts():
        if r.get("stage") == stage and r.get("event") == "done":
            row = r
    out = ((row or {}).get("outputs") or {}).get(name)
    return project.read_artifact(_ref_from_row(out)) if isinstance(out, dict) else None


def _event(project: Project, stage: str, attempt: int, key: str | None, event: str, **extra: Any) -> None:
    project.append_attempt({"stage": stage, "attempt": attempt, "key": key, "event": event, "utc": _utc_now(),
                            **extra})


def run_stage(project: Project, ref: str, decision: dict, *, spec: PipelineSpec | None = None,
              registry: dict | None = None, trust: TrustBackend | None = None,
              ledger: AlphaLedger | None = None) -> StageResult:
    spec, registry = _validated(project, spec, registry)
    i = _locate(spec, ref)
    states = _states(project, spec, registry)
    _check_ready(spec, states, i)
    sref = spec.stages[i]
    cls = registry[sref.stage]
    stage = cls()
    inputs = _inputs(project, spec, states, sref)
    key = _key(sref, cls.version, inputs, decision)
    cur = states[sref.name]

    own_changed = cur["status"] in ("done", "stale") and _own_changed(sref, cur, registry)
    if own_changed and not decision.get("rerun"):
        raise _refuse(f"stages[{i}]", f"stage {sref.name!r} was done with a different type, version or params",
                      'pass decision["rerun"] = true to rerun it; later stages become stale')
    if cur["status"] == "done":
        if cur["key"] == key:
            return StageResult(sref.name, key, cur["attempts"], _recorded_outputs(project, cur), True)
        if not decision.get("rerun"):
            raise _refuse(f"stages[{i}]", f"stage {sref.name!r} is done with a different key (inputs, params or "
                          "decision changed)", 'pass decision["rerun"] = true to rerun it; later stages become stale')
    if decision.get("rerun") and (cur["status"] == "done" or own_changed):
        for later in spec.stages[i + 1:]:        # audit trail only; status is derived from the key parts
            ls = states[later.name]
            if ls["status"] != "pending":
                _event(project, later.name, ls["attempts"], ls["key"], "stale", cause=sref.name)

    attempt = cur["attempts"] + 1
    # Resolved now, from the registry class and the spec, and recorded: later readers (preregister's post hoc test)
    # trust the log, not the pipeline file, which can be edited afterwards.
    _event(project, sref.name, attempt, key, "start", trains=_is_training(spec, sref, cls))
    ctx = _context(project, sref, inputs, trust, ledger)
    before = _side_effects(project, sref.name)
    try:
        outputs = stage.run(ctx, decision)
    except Refused as exc:
        rules = sorted({p.rule for p in exc.problems})
        reason = exc.problems[0].message[:300] if exc.problems else ""
        if _side_effects(project, sref.name) == before:      # refused before any write: audit only, state unchanged
            _event(project, sref.name, attempt, key, "refused", rules=rules, reason=reason)
        else:                                                # it wrote records or spent alpha first: a failed attempt
            _event(project, sref.name, attempt, key, "failed", error=type(exc).__name__, rules=rules, reason=reason)
        raise
    except Exception as exc:
        _event(project, sref.name, attempt, key, "failed", error=type(exc).__name__)
        raise
    try:
        _check_outputs(stage, ctx, outputs, sref.name)
    except Exception as exc:
        _event(project, sref.name, attempt, key, "failed", error=type(exc).__name__)
        raise
    _event(project, sref.name, attempt, key, "done", outputs={n: _ref_row(r) for n, r in sorted(outputs.items())},
           stage_type=sref.stage, version=cls.version, params_sha256=_params_sha(sref.params),
           input_sha256s={k: r.sha256 for k, r in sorted(inputs.items())},
           decision_sha256=sha256_bytes(canonical_json({k: v for k, v in decision.items() if k != "rerun"})))
    return StageResult(sref.name, key, attempt, outputs, False)


def _check_outputs(stage: Any, ctx: StageContext, outputs: dict[str, ArtifactRef], ref_name: str) -> None:
    declared = dict(stage.outputs)
    if not isinstance(outputs, dict) or set(outputs) != set(declared):
        raise _refuse(f"stage {stage.name}", f"the stage returned outputs {sorted(outputs) if isinstance(outputs, dict) else outputs!r} "
                      f"but declares {sorted(declared)}", "return exactly the declared outputs")
    for name, ref in outputs.items():
        if not isinstance(ref, ArtifactRef) or ref.stage != ref_name:
            got = ref.stage if isinstance(ref, ArtifactRef) else type(ref).__name__
            raise _refuse(f"stage {ref_name}", f"output {name!r} of stage {ref_name!r} is not written under its own "
                          f"stage name (it names {got!r})", "write every output with project.write_artifact(ctx.stage, ...)")
        if ref.schema != declared[name]:
            raise _refuse(f"stage {stage.name}", f"output {name!r} has schema {ref.schema!r}, declared "
                          f"{declared[name]!r}", "write the output with its declared schema")
        ctx.project.read_artifact(ref)
    problems = stage.check(ctx, outputs)
    if problems:
        raise Refused(list(problems))
