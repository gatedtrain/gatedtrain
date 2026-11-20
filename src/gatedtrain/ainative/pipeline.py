"""AN-4: the pipeline spec and its validator.

A pipeline YAML lists stages in order. The validator is what makes "an agent cannot remove, reorder or wrap a
trust stage" true: it refuses a spec in which a trust stage is missing, duplicated, out of order, or followed by
a pluggable stage before `report`, and it checks the wiring (inputs from earlier stages only, matching schemas),
the stage types and the params.

    name: <pipeline name>
    training_stages: [<stage name or type>, ...]     # optional; a stage needing "gpu" is also a training stage
    stages:
      - {name: <ref>, stage: <registered type>, params: {...}, inputs: {<input>: "<ref>.<output>"}}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..errors import Problem, Refused
from .stage import TRUST_STAGES


@dataclass(frozen=True)
class StageRef:
    name: str
    stage: str
    params: dict = field(default_factory=dict)
    inputs: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineSpec:
    name: str
    stages: list[StageRef]
    training_stages: tuple[str, ...] = ()


def _bad(field_: str, message: str, fix: str) -> Refused:
    return Refused([Problem("AN-4", field_, message, fix)])


def load_pipeline(path: Path) -> PipelineSpec:
    try:
        raw = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise _bad("pipeline", f"cannot read the pipeline spec {path}: {exc}", "provide a readable YAML file")
    if not isinstance(raw, dict) or not isinstance(raw.get("stages"), list):
        raise _bad("stages", "the pipeline spec must be a mapping with a `stages` list",
                   "add `name:` and a `stages:` list of {name, stage, params, inputs}")
    training = raw.get("training_stages") or []
    if not isinstance(training, list) or not all(isinstance(t, str) for t in training):
        raise _bad("training_stages", "`training_stages` must be a list of stage names",
                   "list the names of the stages that train")
    stages: list[StageRef] = []
    for i, item in enumerate(raw["stages"]):
        if not isinstance(item, dict):
            raise _bad(f"stages[{i}]", "a stage entry must be a mapping", "use {name, stage, params, inputs}")
        for key in ("name", "stage"):
            if not isinstance(item.get(key), str) or not item[key]:
                raise _bad(f"stages[{i}].{key}", f"`{key}` must be a non-empty string", f"set `{key}` for this stage")
        params = item.get("params") or {}
        inputs = item.get("inputs") or {}
        if not isinstance(params, dict):
            raise _bad(f"stages[{i}].params", "`params` must be a mapping", "use `params: {key: value}`")
        if not isinstance(inputs, dict) or not all(isinstance(k, str) and isinstance(v, str)
                                                    for k, v in inputs.items()):
            raise _bad(f"stages[{i}].inputs", "`inputs` must map an input name to '<stage>.<output>'",
                       "use `inputs: {name: \"<stage ref>.<output>\"}`")
        stages.append(StageRef(item["name"], item["stage"], dict(params), dict(inputs)))
    return PipelineSpec(str(raw.get("name", "")), stages, tuple(training))


def _is_training(spec: PipelineSpec, ref: StageRef, cls: Any) -> bool:
    needs = getattr(cls, "needs", frozenset()) if cls is not None else frozenset()
    return "gpu" in needs or ref.name in spec.training_stages or ref.stage in spec.training_stages


def pipeline_problems(spec: PipelineSpec, registry: dict) -> list[Problem]:
    problems: list[Problem] = []

    def add(field_: str, message: str, fix: str) -> None:
        problems.append(Problem("AN-4", field_, message, fix))

    stages = spec.stages
    names = [s.name for s in stages]
    for i, ref in enumerate(stages):
        if names.index(ref.name) != i:
            add(f"stages[{i}].name", f"stage name {ref.name!r} is used twice", "give every stage a unique name")

    # Stage types, params, wiring.
    for i, ref in enumerate(stages):
        cls = registry.get(ref.stage)
        if cls is None:
            add(f"stages[{i}].stage", f"unknown stage type {ref.stage!r}",
                f"use one of: {', '.join(sorted(registry))}")
            continue
        if ref.name in TRUST_STAGES and ref.stage != ref.name:
            add(f"stages[{i}].name", f"stage name {ref.name!r} is reserved for the trust stage of that name",
                "rename the stage")
        accepted = frozenset(getattr(cls, "params", frozenset()))
        for key in ref.params:
            if key not in accepted:
                add(f"stages[{i}].params.{key}", f"stage {ref.stage!r} does not accept the param {key!r}",
                    f"accepted params: {', '.join(sorted(accepted)) or '(none)'}")
        declared = dict(getattr(cls, "inputs", {}))
        for key in declared:
            if key not in ref.inputs:
                add(f"stages[{i}].inputs.{key}", f"input {key!r} of stage {ref.stage!r} is not wired",
                    f"add `{key}: \"<earlier stage>.<output>\"`")
        for key, src in ref.inputs.items():
            field_ = f"stages[{i}].inputs.{key}"
            if key not in declared:
                add(field_, f"stage {ref.stage!r} declares no input {key!r}",
                    f"declared inputs: {', '.join(sorted(declared)) or '(none)'}")
                continue
            producer_name, dot, out = src.partition(".")
            if not dot or not producer_name or not out:
                add(field_, f"input source {src!r} is not '<stage>.<output>'", "write the source as '<stage>.<output>'")
                continue
            if producer_name not in names:
                add(field_, f"input comes from {producer_name!r}, which is not a stage of this pipeline",
                    "wire the input from an earlier stage of this pipeline")
                continue
            j = names.index(producer_name)
            if j >= i:
                add(field_, f"input comes from {producer_name!r}, which does not run before this stage",
                    "wire the input from an earlier stage")
                continue
            pcls = registry.get(stages[j].stage)
            if pcls is None:
                continue  # already refused as an unknown stage type
            pouts = dict(getattr(pcls, "outputs", {}))
            if out not in pouts:
                add(field_, f"stage {producer_name!r} has no output {out!r}",
                    f"outputs of {producer_name!r}: {', '.join(sorted(pouts)) or '(none)'}")
            elif pouts[out] != declared[key]:
                add(field_, f"input {key!r} expects schema {declared[key]!r} but {src} is {pouts[out]!r}",
                    "wire an output whose schema matches")

    # Trust order.
    idx: dict[str, list[int]] = {t: [] for t in TRUST_STAGES}
    for i, ref in enumerate(stages):
        if ref.stage in idx:
            idx[ref.stage].append(i)
    for t in TRUST_STAGES:
        if not idx[t]:
            add("stages", f"trust stage {t!r} is missing", f"add a stage of type {t!r}; the five trust stages are mandatory")
        for i in idx[t][1:]:
            add(f"stages[{i}].stage", f"trust stage {t!r} appears more than once", f"keep exactly one {t!r} stage")
    first = {t: v[0] for t, v in idx.items() if v}

    if "intake_seal" in first and first["intake_seal"] != 0:
        add(f"stages[{first['intake_seal']}].stage", "intake_seal must be the first stage",
            "move intake_seal to the start of the pipeline")
    training = [i for i, ref in enumerate(stages) if _is_training(spec, ref, registry.get(ref.stage))]
    if "preregister" in first:
        for i in training:
            if i < first["preregister"]:
                add(f"stages[{first['preregister']}].stage",
                    f"preregister must come before every training stage (stages[{i}] trains first)",
                    "move preregister before the first training stage")
                break
    if training:
        last_train = training[-1]
        for t in ("evaluate_sealed", "gate"):
            if t in first and first[t] < last_train:
                add(f"stages[{first[t]}].stage", f"{t} must come after the last training stage (stages[{last_train}])",
                    f"move {t} after every training stage")
    if "evaluate_sealed" in first and "gate" in first and first["gate"] < first["evaluate_sealed"]:
        add(f"stages[{first['gate']}].stage", "gate must come after evaluate_sealed",
            "move gate after evaluate_sealed")
    if "report" in first and first["report"] != len(stages) - 1:
        add(f"stages[{first['report']}].stage", "report must be the last stage", "move report to the end of the pipeline")
    if "evaluate_sealed" in first:
        for i, ref in enumerate(stages):
            if i > first["evaluate_sealed"] and ref.stage not in TRUST_STAGES:
                add(f"stages[{i}].stage", f"pluggable stage {ref.name!r} runs after evaluate_sealed",
                    "no pluggable stage may sit between evaluate_sealed and report; move it earlier")
    return problems
