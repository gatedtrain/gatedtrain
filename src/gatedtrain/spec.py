"""Workflow spec parsing and validation (MVP-SPEC section 2.1).

The skeleton spec is a strict syntactic subset of the 04 section 3 workflow
YAML (04-workflow-and-composition.md:36-72): one training stage followed by
one gate. It is parsed by this script,
not by the 04 section 4.2 validator; the checks below are the 04 section 4.2
required-field rules for a one-stage, one-gate workflow (04:90-105).

Every error names the rule, the field and the fix (
tests/README.md section 3).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import Problem, Refused
from .paths import resolve
from .registration import (GATE_TO_CLAUSE, check_against_gate, check_amendment, clause_list, is_multi,
                           normalize_clause, read_registration)

# 04 section 3 key vocabulary (04:36-72), plus the 04 section 4.2 table
# parameters (04:90-96) and the rule 4 gate fields (04:105).
TOP_KEYS = ("workflow", "version", "budget", "artifacts", "stages")
BUDGET_KEYS = ("gpu_hours", "labeling_usd")
ARTIFACT_KEYS = ("type", "uri", "disjoint_from")
STAGE_KEYS = ("id", "uses", "algorithm", "model_family", "rewards", "inputs", "outputs", "evaluator", "sampler",
              "test", "on_fail", "alpha_family", "labels", "on_timeout", "owner", "escalation", "quorum", "timeout",
              "escalate_after")
TEST_KEYS = ("metric", "bar", "alpha", "power", "mde", "cluster_by", "seeds", "type", "margin")

# Skeleton capability: which grammar keys each node kind can honor.
TRAIN_STAGE_KEYS = ("id", "uses", "algorithm", "model_family", "inputs", "outputs")
GATE_KEYS = ("id", "uses", "evaluator", "sampler", "inputs", "test", "on_fail", "on_timeout", "owner", "escalation",
             "quorum", "timeout", "escalate_after", "alpha_family")

# Design revision 10 (R10-7; 04:34, 04:100; tests/README.md section 11.15 R10A-6): the registration is
# authoritative for alpha, power, mde (planning effect) and margin, so the spec's `test` values are optional;
# a value the spec does state must match the registration (registration.check_against_gate).

TRAINER_BACKENDS = {"sft.cpu_stub@0.1": "cpu_stub", "rl.nemo_rl@0.1": "nemo_rl"}
# Tier A stand-in and the Tier B NeMo Evaluator GSM8K backend (TIER-B-SPEC H-7). `gatedtrain run`
# runs only cpu_stub; `gatedtrain tierb-collect` reads nemo_evaluator outputs.
EVALUATOR_BACKENDS = {"addition_em.cpu_stub@0.1": "cpu_stub", "gsm8k.nemo_evaluator@0.1": "nemo_evaluator"}


def _external_ids() -> None:
    """MULTIMODAL-SPEC MM-13 (tests/README.md section 11.18, EXT-7): the trainer and evaluator ids of the
    external-run adapter registry validate as backend `ext`; only `gatedtrain ext-collect` reads their outputs."""
    from .ext.registry import EVALUATOR_ROWS, TRAINER_ROWS

    for row in TRAINER_ROWS.values():
        TRAINER_BACKENDS.update({uses: "ext" for uses in row["uses"]})
    EVALUATOR_BACKENDS.update({eid: "ext" for eid in EVALUATOR_ROWS})


_external_ids()
TRAIN_INPUT_TYPES = {"model": "model_checkpoint", "data": "training_dataset", "config": "trainer_config"}
GATE_INPUT_TYPES = {"baseline": "model_checkpoint", "items": "item_set", "registration": "registration"}
# Design revision 11 (R11-19; 04 section 3, section 4.2): the gate's evaluation items come through role `items`
# of artifact type `item_set`. The revision 10 role `prompts` (type `prompt_dataset`) is its legacy alias
# (R11A-2); each role keeps its own type, and a gate naming both is refused on the legacy role. A training
# stage's `data` stays `training_dataset`. `Workflow.names["items"]` names the artifact either way.
LEGACY_GATE_ROLES = {"prompts": ("items", "prompt_dataset")}
OUTCOMES = ("stop", "pause", "reject")

USES_RE = re.compile(r"^(?P<name>[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+)@(?P<ver>\d+(?:\.\d+)*)$")
PINNED_RE = re.compile(r"^(?P<loc>.+)@sha256:(?P<hex>[0-9a-f]{64})$")


@dataclass
class Workflow:
    """A validated spec with resolved local paths."""

    spec_path: Path
    spec: dict
    stage: dict
    gate: dict
    gate_type: str
    trainer_backend: str
    artifact_paths: dict[str, Path]
    artifact_pins: dict[str, str]
    registration: dict                                    # as written (legacy names possible)
    registration_path: Path
    registration_sha256: str
    clause: dict | None                                   # revision 11 names (registration.normalize_clause);
                                                          # None for several clauses (R18-1): read `clauses`
    names: dict[str, str] = field(default_factory=dict)   # role -> artifact name
    amendment: str | None = None                          # registration.VERSION_ONLY or None (R17-1)
    clauses: list[dict] = field(default_factory=list)     # every clause, revision 11 names (R18-1)

    @property
    def evaluators(self) -> list[str]:
        """The gate's evaluators: one per clause, in clause order (R18-5)."""
        ev = self.gate.get("evaluator")
        return list(ev) if isinstance(ev, list) else [ev]

    @property
    def spec_dir(self) -> Path:
        return self.spec_path.parent

    def path_of(self, role: str) -> Path:
        return self.artifact_paths[self.names[role]]


def _uri_path(uri: str, spec_dir: Path) -> tuple[Path | None, str | None]:
    """`file:<relative path>[@sha256:<hex>]` -> (path, hex). The scheme is a naming choice (G-2)."""
    hex_ = None
    m = PINNED_RE.match(uri)
    if m:
        uri, hex_ = m.group("loc"), m.group("hex")
    if not uri.startswith("file:"):
        return None, hex_
    return resolve(spec_dir / uri[len("file:"):]), hex_        # lenient: a link loop never raises (S3)


def load_spec(spec_path: Path) -> Workflow:
    """Parse and validate the spec and its registration. Raises Refused with every problem found."""
    spec_path = resolve(spec_path)
    problems: list[Problem] = []

    def bad(rule: str, fld: str, msg: str, fix: str) -> None:
        problems.append(Problem(rule, fld, msg, fix))

    try:
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise Refused([Problem("SPEC-SHAPE", "spec", f"cannot read {spec_path}: {exc}", "pass an existing spec file")])
    except yaml.YAMLError as exc:
        raise Refused([Problem("SPEC-SHAPE", "spec", f"not valid YAML: {exc}", "fix the YAML syntax")])
    if not isinstance(spec, dict):
        raise Refused([Problem("SPEC-SHAPE", "spec", "the spec is not a mapping", "write the 04 section 3 workflow YAML")])

    # --- 04 section 3 grammar: no field the grammar lacks ---
    def subset(obj: Any, allowed: tuple[str, ...], path: str) -> None:
        if isinstance(obj, dict):
            for k in obj:
                if k not in allowed:
                    fld = f"{path}.{k}" if path else str(k)
                    bad("SPEC-SUBSET", fld, f"`{k}` is not a field of the 04 section 3 workflow grammar",
                        f"remove `{k}`; allowed here: {', '.join(allowed)}")

    subset(spec, TOP_KEYS, "")
    artifacts = spec.get("artifacts")
    stages = spec.get("stages")
    if isinstance(spec.get("budget"), dict):
        subset(spec["budget"], BUDGET_KEYS, "budget")
    if isinstance(artifacts, dict):
        for name, entry in artifacts.items():
            subset(entry, ARTIFACT_KEYS, f"artifacts.{name}")
    if isinstance(stages, list):
        for i, st in enumerate(stages):
            subset(st, STAGE_KEYS, f"stages[{i}]")
            if isinstance(st, dict):
                subset(st.get("test"), TEST_KEYS, f"stages[{i}].test")

    # --- top-level shape ---
    for k in ("workflow", "version", "artifacts", "stages"):
        if k not in spec:
            bad("SPEC-SHAPE", k, "missing top-level field (04 section 3)", f"add `{k}`")
    if "workflow" in spec and not isinstance(spec["workflow"], str):
        bad("SPEC-SHAPE", "workflow", "workflow must be a name", "set `workflow: <name>`")
    if "version" in spec and spec["version"] != 1:
        bad("SPEC-SHAPE", "version", f"version {spec['version']!r} is not supported", "set `version: 1`")
    budget = spec.get("budget")
    if budget is not None and not isinstance(budget, dict):
        bad("SPEC-SHAPE", "budget", "budget must be a mapping", "use `budget: {gpu_hours: 0}` on Tier A (G-10)")
    if isinstance(budget, dict):
        for k in ("gpu_hours", "labeling_usd"):
            if k in budget and not (isinstance(budget[k], (int, float)) and not isinstance(budget[k], bool)
                                    and budget[k] >= 0):
                bad("SPEC-SHAPE", f"budget.{k}", f"{k} must be a number >= 0 (04 section 4.1, 04:83)",
                    "set the cap, for example `gpu_hours: 8` on Tier B or 0 on Tier A (G-10)")
    if artifacts is not None and not isinstance(artifacts, dict):
        bad("SPEC-SHAPE", "artifacts", "artifacts must be a mapping of name -> {type, uri}", "fix `artifacts`")
        artifacts = None
    artifacts = artifacts or {}

    spec_dir = spec_path.parent
    artifact_paths: dict[str, Path] = {}
    artifact_pins: dict[str, str] = {}
    for name, entry in artifacts.items():
        fld = f"artifacts.{name}"
        if not isinstance(entry, dict) or not isinstance(entry.get("type"), str) or not isinstance(entry.get("uri"), str):
            bad("SPEC-SHAPE", fld, "an artifact needs a string `type` and `uri` (04 section 3)", "write {type, uri}")
            continue
        path, hex_ = _uri_path(entry["uri"], spec_dir)
        if path is None:
            bad("SPEC-REF", f"{fld}.uri", f"unsupported uri {entry['uri']!r}",
                "use `file:<path relative to the spec>` (Tier A)")
            continue
        artifact_paths[name] = path
        if hex_:
            artifact_pins[name] = "sha256:" + hex_
        for j, other in enumerate(entry.get("disjoint_from") or []):
            if other not in artifacts:
                bad("SPEC-REF", f"{fld}.disjoint_from[{j}]", f"unknown artifact {other!r}", "name an artifact of this spec")

    # --- one training stage, then one gate (MVP-SPEC section 2.1) ---
    if stages is not None and not isinstance(stages, list):
        bad("SPEC-SHAPE", "stages", "stages must be a list", "list one training stage and one gate")
        stages = None
    stages = stages or []
    kinds = [isinstance(s, dict) and str(s.get("uses", "")).startswith("gate.") for s in stages]
    if len(stages) != 2 or kinds != [False, True] or not all(isinstance(s, dict) for s in stages):
        bad("SPEC-SHAPE", "stages",
            f"the skeleton runs exactly one training stage followed by one gate; found {len(stages)} stage(s) "
            f"with gate positions {[i for i, g in enumerate(kinds) if g]}",
            "keep one training stage as stages[0] and one gate as stages[1]")
        raise Refused(problems)
    stage, gate = stages
    ids = [stage.get("id"), gate.get("id")]
    for i, sid in enumerate(ids):
        if not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", sid or ""):
            bad("SPEC-SHAPE", f"stages[{i}].id", "a stage id is a name of letters, digits, '_', '.' or '-'",
                "set `id: <name>`")
    if ids[0] == ids[1]:
        bad("SPEC-SHAPE", "stages[1].id", "stage ids must be unique", "rename one stage")

    names: dict[str, str] = {}

    # --- training stage ---
    for k in stage:
        if k in STAGE_KEYS and k not in TRAIN_STAGE_KEYS:
            bad("SPEC-SHAPE", f"stages[0].{k}", f"`{k}` is not supported on the skeleton's training stage",
                f"remove `{k}`; supported: {', '.join(TRAIN_STAGE_KEYS)}")
    uses = stage.get("uses")
    if uses not in TRAINER_BACKENDS:
        bad("SPEC-REF", "stages[0].uses", f"unknown training stage type {uses!r}",
            f"use one of {', '.join(TRAINER_BACKENDS)}")
    inputs = stage.get("inputs") if isinstance(stage.get("inputs"), dict) else {}
    if not isinstance(stage.get("inputs"), dict):
        bad("SPEC-SHAPE", "stages[0].inputs", "inputs must map role -> artifact name", "add `inputs: {model, data, config}`")
    for role, typ in TRAIN_INPUT_TYPES.items():
        ref = inputs.get(role)
        fld = f"stages[0].inputs.{role}"
        if ref is None:
            bad("SPEC-REF", fld, f"missing input role `{role}`", f"name a `{typ}` artifact")
        elif ref not in artifacts:
            bad("SPEC-REF", fld, f"unknown artifact {ref!r} (04 section 4.1: typed artifacts are consumed by name)",
                f"name one of: {', '.join(artifacts)}")
        elif isinstance(artifacts[ref], dict) and artifacts[ref].get("type") != typ:
            bad("SPEC-REF", fld, f"artifact {ref!r} has type {artifacts[ref].get('type')!r}, not {typ!r}",
                f"name a `{typ}` artifact")
        else:
            names[role] = ref
    for role in sorted(set(inputs) - set(TRAIN_INPUT_TYPES)):
        bad("SPEC-REF", f"stages[0].inputs.{role}", f"input role `{role}` is not used by {uses}",
            f"use the roles {', '.join(TRAIN_INPUT_TYPES)}")
    outputs = stage.get("outputs") if isinstance(stage.get("outputs"), dict) else {}
    out_model = outputs.get("model")
    if not isinstance(out_model, str):
        bad("SPEC-REF", "stages[0].outputs.model", "the training stage must name its output model",
            "add `outputs: {model: <name>}`")
    elif out_model in artifacts:
        bad("SPEC-REF", "stages[0].outputs.model", f"output name {out_model!r} collides with an artifact",
            "choose a new name")
    for role in sorted(set(outputs) - {"model"}):
        bad("SPEC-REF", f"stages[0].outputs.{role}", f"output role `{role}` is not produced by {uses}", "remove it")

    # --- gate (04 section 4.2) ---
    gpath = "stages[1]"
    gtype = str(gate.get("uses", "")).split("@")[0]
    for k in gate:
        if k in STAGE_KEYS and k not in GATE_KEYS:
            bad("SPEC-SHAPE", f"{gpath}.{k}", f"`{k}` is not supported on a skeleton gate",
                f"remove `{k}`; supported: {', '.join(GATE_KEYS)}")
    if gate.get("uses") not in {f"{g}@1.0" for g in GATE_TO_CLAUSE}:
        if gtype in ("gate.threshold", "gate.human"):
            bad("SPEC-SHAPE", f"{gpath}.uses", f"{gtype} is not supported by the skeleton",
                "use gate.superiority, gate.noninferiority or gate.equivalence (MVP-SPEC section 2.7)")
        else:
            bad("SPEC-REF", f"{gpath}.uses", f"unknown gate type {gate.get('uses')!r}",
                "use gate.superiority@1.0, gate.noninferiority@1.0 or gate.equivalence@1.0")
    test = gate.get("test") if isinstance(gate.get("test"), dict) else None
    if test is None:
        bad("GATE-REQUIRED", f"{gpath}.test", "a gate declares its test (04 section 4.2)",
            "add `test: {metric, alpha, power, mde | margin, cluster_by, seeds}`")
        test = {}
    for k in ("bar", "type"):
        if k in test:
            bad("SPEC-SHAPE", f"{gpath}.test.{k}", f"`{k}` belongs to gate.threshold or gate.human, not {gtype}",
                f"remove `{k}`")
    has_alpha, has_family = "alpha" in test, "alpha_family" in gate
    if has_alpha and has_family:
        bad("GATE-ALPHA", f"{gpath}.alpha_family", "a gate inside a family must not set its own alpha (04 section 3)",
            "remove `test.alpha` or `alpha_family`")
    elif has_family:
        bad("SPEC-SHAPE", f"{gpath}.alpha_family", "alpha families are not supported by the skeleton",
            "set `test.alpha` on the single gate (05 section 5 family schedules are out of scope)")
    for k in ("on_fail", "on_timeout"):
        if k in gate or k == "on_fail":
            v = gate.get(k)
            if v not in OUTCOMES:
                bad("GATE-OUTCOME", f"{gpath}.{k}", f"{k} must be one of stop, pause, reject; got {v!r}",
                    "use `stop` or `pause`")
            elif v == "reject":
                bad("GATE-OUTCOME", f"{gpath}.{k}", "`reject` is valid only for a gate inside a loop body (04 section 4.2 rule 3)",
                    "use `stop` or `pause`; the skeleton has no loops")
    if not isinstance(gate.get("owner"), str) or not gate.get("owner"):
        bad("GATE-FIELDS", f"{gpath}.owner", "a gate names its owner (04 section 4.2 rule 4)", "add `owner: <team>`")
    esc = gate.get("escalation")
    if not (isinstance(esc, list) and esc and all(isinstance(e, str) for e in esc)):
        bad("GATE-FIELDS", f"{gpath}.escalation", "a gate names an escalation list (04 section 4.2 rule 4)",
            "add `escalation: [<role>]`")
    gev = gate.get("evaluator")
    if isinstance(gev, list) and len(gev) >= 2 and all(isinstance(e, str) for e in gev):
        # Design revision 18 (R18-5): a gate over several clauses names one evaluator per clause, in clause order.
        for j, e in enumerate(gev):
            if not USES_RE.match(e):
                bad("SPEC-REF", f"{gpath}.evaluator[{j}]", f"evaluator {e!r} is not `<name>@<version>`",
                    "use for example addition_em.cpu_stub@0.1")
        if len(set(gev)) != len(gev):
            bad("SPEC-REF", f"{gpath}.evaluator", f"the gate lists an evaluator twice: {gev}",
                "list each clause's evaluator once (R18-5)")
    elif not isinstance(gev, str):
        bad("GATE-FIELDS", f"{gpath}.evaluator", "a gate names its evaluator (04 section 4.2 rule 2)",
            "add `evaluator: <id>@<version>`, or a list of two or more, one per registered clause (R18-5)")
    elif not USES_RE.match(gate["evaluator"]):
        bad("SPEC-REF", f"{gpath}.evaluator", f"evaluator {gate['evaluator']!r} is not `<name>@<version>`",
            "use for example addition_em.cpu_stub@0.1")
    if not isinstance(gate.get("sampler"), str):
        bad("GATE-FIELDS", f"{gpath}.sampler", "a gate names its deployment sampler (04 section 4.2 rule 5)",
            "add `sampler: deployment`")

    ginputs = gate.get("inputs") if isinstance(gate.get("inputs"), dict) else {}
    if ginputs.get("model") is None:
        bad("SPEC-REF", f"{gpath}.inputs.model", "the gate needs the model under test", f"set it to {out_model!r}")
    elif ginputs["model"] != out_model:
        bad("SPEC-REF", f"{gpath}.inputs.model", f"{ginputs['model']!r} is not the training stage's output",
            f"set it to {out_model!r}")
    if ginputs.get("baseline") is None:
        bad("SPEC-REF", f"{gpath}.inputs.baseline", "the paired gate needs a baseline (05 section 3.3)",
            f"set it to the stage's input model {inputs.get('model')!r}")
    elif ginputs["baseline"] != inputs.get("model"):
        bad("SPEC-REF", f"{gpath}.inputs.baseline",
            f"{ginputs['baseline']!r} is not the stage's input model; the skeleton compares the stage output with its input",
            f"set it to {inputs.get('model')!r}")
    if "items" in ginputs and "prompts" in ginputs:
        bad("SPEC-REF", f"{gpath}.inputs.prompts", "the gate names both `items` and its legacy alias `prompts` "
            "(R11-19)", "remove `prompts`; keep `items`")
    items_role = "prompts" if "prompts" in ginputs and "items" not in ginputs else "items"
    for role in (items_role, "registration"):
        ref = ginputs.get(role)
        fld = f"{gpath}.inputs.{role}"
        want = LEGACY_GATE_ROLES[role][1] if role in LEGACY_GATE_ROLES else GATE_INPUT_TYPES[role]
        if ref is None:
            if role == "registration":
                bad("REG-MISSING", fld, "the gate names no registration (05 section 3.7; G-2)",
                    "add a `type: registration` artifact pinned with @sha256: and name it here")
            else:
                bad("SPEC-REF", fld, "the gate needs its evaluation items (04 section 4.2)",
                    "name an `item_set` artifact as `items`")
        elif ref not in artifacts:
            bad("SPEC-REF", fld, f"unknown artifact {ref!r}", f"name one of: {', '.join(artifacts)}")
        elif artifacts[ref].get("type") != want:
            bad("SPEC-REF", fld, f"artifact {ref!r} has type {artifacts[ref].get('type')!r}, not {want!r}",
                f"name a `{want}` artifact")
        else:
            names["items" if role == "prompts" else role] = ref
    for role in sorted(set(ginputs) - {"model", "baseline", "items", "prompts", "registration"}):
        bad("SPEC-REF", f"{gpath}.inputs.{role}", f"input role `{role}` is not used by a skeleton gate", "remove it")

    # --- registration (05 section 3.7) ---
    registration, reg_sha, reg_path = None, None, None
    if "registration" in names:
        rname = names["registration"]
        uri_field = f"artifacts.{rname}.uri"
        m = PINNED_RE.match(artifacts[rname]["uri"])
        if not m:
            bad("REG-HASH", uri_field, "the registration uri carries no @sha256:<hash> pin (05 section 3.7; G-2)",
                "append @sha256:<SHA-256 of the registration file bytes> to the uri")
        elif rname in artifact_paths:
            reg_path = artifact_paths[rname]
            gate_evaluator = gate.get("evaluator") if isinstance(gate.get("evaluator"), str) else None
            registration, reg_sha, reg_problems = read_registration(reg_path, uri_field=uri_field,
                                                                    expected_hex=m.group("hex"),
                                                                    evaluator_id=gate_evaluator)
            problems.extend(reg_problems)
            if registration is not None:
                problems.extend(check_against_gate(registration, gate, gpath))
                if is_multi(registration) and TRAINER_BACKENDS.get(stage.get("uses")) in ("cpu_stub", "nemo_rl"):
                    # R18-1: only ext-collect gates several clauses.
                    problems.append(Problem(
                        "REG-FIELD", "registration.clauses",
                        f"a registration with several clauses is gated by `gatedtrain ext-collect`; "
                        f"{stage.get('uses')} runs through `gatedtrain run` or `tierb-collect`, which gate one clause",
                        "register one `clause` for this workflow"))
                rclause = registration.get("clause")
                if isinstance(rclause, dict) and rclause.get("constraint_set") is not None:
                    # Constraint clauses (tests/README.md section 11.17) are library-level only so far.
                    problems.append(Problem(
                        "REG-CONSTRAINT", "registration.clause.constraint_set",
                        "constraint clauses are not wired into `gatedtrain run` (tests/README.md section 11.17 scope)",
                        "gate a constraint clause with gatedtrain.constraints.evaluate_constraint on recorded verdicts"))
    if problems:
        raise Refused(problems)

    return Workflow(
        spec_path=spec_path, spec=spec, stage=stage, gate=gate, gate_type=gtype,
        trainer_backend=TRAINER_BACKENDS[stage["uses"]], artifact_paths=artifact_paths, artifact_pins=artifact_pins,
        registration=registration, registration_path=reg_path, registration_sha256=reg_sha,
        clause=None if is_multi(registration) else normalize_clause(registration["clause"]), names=names,
        amendment=check_amendment(registration, reg_path)[0], clauses=clause_list(registration),
    )
