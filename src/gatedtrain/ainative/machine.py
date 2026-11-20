"""AN-5: the stage machine.

Runs a project through its validated pipeline. State lives only in the project's artifacts and `attempts.jsonl`
(`status` is derived from them), so a new process after a crash sees the same status.

Attempt lines: `{stage, attempt, key, event, utc}` with `event` one of `start`, `done` (also lists the outputs),
`failed` (also names the error) and `stale` (a rerun of an earlier stage invalidated this one).

The idempotence key is the sha256 of (stage name, version, params, input artifact hashes, decision without its
`rerun` flag). A done stage with the same key returns its recorded outputs without calling `run`; a different
key on a done stage refuses unless `decision["rerun"]` is true, which marks every later stage `stale`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..canonical import canonical_json, sha256_bytes
from ..errors import Problem, Refused
from .pipeline import PipelineSpec, StageRef, load_pipeline, pipeline_problems
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
    """`project.pipeline` is the path of the pipeline YAML (absolute, or relative to the project root)."""
    path = Path(project.pipeline)
    if not path.is_absolute():
        path = project.root / path
    if not path.is_file():
        raise _refuse("pipeline", f"the pipeline spec {project.pipeline!r} is not a file",
                      "recreate the project with the path of a pipeline YAML")
    return load_pipeline(path)


def _validated(project: Project, spec: PipelineSpec | None, registry: dict | None) -> tuple[PipelineSpec, dict]:
    spec = spec if spec is not None else _load_spec(project)
    registry = registry if registry is not None else stage_registry()
    problems = pipeline_problems(spec, registry)
    if problems:
        raise Refused(problems)
    return spec, registry


def _states(project: Project, spec: PipelineSpec) -> dict[str, dict]:
    """Per stage: status, last attempt line, attempt count, and the recorded outputs and key of a done stage."""
    st: dict[str, dict] = {s.name: {"status": "pending", "last": None, "attempts": 0, "key": None, "outputs": None}
                           for s in spec.stages}
    for row in project.read_attempts():
        cur = st.get(row.get("stage"))
        if cur is None:
            continue
        ev = row.get("event")
        cur["last"] = row
        if ev == "start":
            cur["attempts"] = max(cur["attempts"], int(row.get("attempt", 0)))
            cur["status"] = "failed"          # a start without done or failed is an interrupted run
        elif ev == "done":
            cur["status"], cur["key"], cur["outputs"] = "done", row.get("key"), row.get("outputs")
        elif ev == "failed":
            cur["status"] = "failed"
        elif ev == "stale":
            cur["status"] = "stale"
    return st


def status(project: Project, *, spec: PipelineSpec | None = None) -> dict:
    spec = spec if spec is not None else _load_spec(project)
    return {name: {"status": s["status"], "last_attempt": s["last"]} for name, s in _states(project, spec).items()}


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
    doc = {"stage": sref.name, "version": version, "params": sref.params,
           "inputs": {k: r.sha256 for k, r in sorted(inputs.items())}, "decision": body}
    try:
        return sha256_bytes(canonical_json(doc))
    except ValueError as exc:
        raise _refuse("decision", f"the decision or params are not canonical JSON: {exc}",
                      "use plain JSON values") from None


def _context(project: Project, sref: StageRef, inputs: dict[str, ArtifactRef], trust: TrustBackend | None,
             ledger: AlphaLedger | None) -> StageContext:
    return StageContext(project, inputs, dict(sref.params), trust if trust is not None else InProcessTrust(project),
                        ledger if ledger is not None else FixedAlphaLedger())


def plan_stage(project: Project, ref: str, *, spec: PipelineSpec | None = None, registry: dict | None = None,
               trust: TrustBackend | None = None, ledger: AlphaLedger | None = None) -> Proposal:
    spec, registry = _validated(project, spec, registry)
    i = _locate(spec, ref)
    states = _states(project, spec)
    _check_ready(spec, states, i)
    sref = spec.stages[i]
    ctx = _context(project, sref, _inputs(project, spec, states, sref), trust, ledger)
    return registry[sref.stage]().plan(ctx)


def _event(project: Project, stage: str, attempt: int, key: str | None, event: str, **extra: Any) -> None:
    project.append_attempt({"stage": stage, "attempt": attempt, "key": key, "event": event, "utc": _utc_now(),
                            **extra})


def run_stage(project: Project, ref: str, decision: dict, *, spec: PipelineSpec | None = None,
              registry: dict | None = None, trust: TrustBackend | None = None,
              ledger: AlphaLedger | None = None) -> StageResult:
    spec, registry = _validated(project, spec, registry)
    i = _locate(spec, ref)
    states = _states(project, spec)
    _check_ready(spec, states, i)
    sref = spec.stages[i]
    cls = registry[sref.stage]
    stage = cls()
    inputs = _inputs(project, spec, states, sref)
    key = _key(sref, cls.version, inputs, decision)
    cur = states[sref.name]

    if cur["status"] == "done":
        if cur["key"] == key:
            return StageResult(sref.name, key, cur["attempts"], _recorded_outputs(project, cur), True)
        if not decision.get("rerun"):
            raise _refuse(f"stages[{i}]", f"stage {sref.name!r} is done with a different key (inputs, params or "
                          "decision changed)", 'pass decision["rerun"] = true to rerun it; later stages become stale')
    if decision.get("rerun") and cur["status"] == "done":
        for later in spec.stages[i + 1:]:
            ls = states[later.name]
            if ls["status"] != "pending":
                _event(project, later.name, ls["attempts"], ls["key"], "stale", cause=sref.name)

    attempt = cur["attempts"] + 1
    _event(project, sref.name, attempt, key, "start")
    ctx = _context(project, sref, inputs, trust, ledger)
    try:
        outputs = stage.run(ctx, decision)
        _check_outputs(stage, ctx, outputs)
    except Exception as exc:
        _event(project, sref.name, attempt, key, "failed", error=type(exc).__name__)
        raise
    _event(project, sref.name, attempt, key, "done", outputs={n: _ref_row(r) for n, r in sorted(outputs.items())})
    return StageResult(sref.name, key, attempt, outputs, False)


def _check_outputs(stage: Any, ctx: StageContext, outputs: dict[str, ArtifactRef]) -> None:
    declared = dict(stage.outputs)
    if not isinstance(outputs, dict) or set(outputs) != set(declared):
        raise _refuse(f"stage {stage.name}", f"the stage returned outputs {sorted(outputs) if isinstance(outputs, dict) else outputs!r} "
                      f"but declares {sorted(declared)}", "return exactly the declared outputs")
    for name, ref in outputs.items():
        if ref.schema != declared[name]:
            raise _refuse(f"stage {stage.name}", f"output {name!r} has schema {ref.schema!r}, declared "
                          f"{declared[name]!r}", "write the output with its declared schema")
        ctx.project.read_artifact(ref)
    problems = stage.check(ctx, outputs)
    if problems:
        raise Refused(list(problems))
