"""A synthetic scenario pack for the AI-native stage machine (CPU only, no framework).

Toy task: map an input string to its label. An input is `<token>-<anything>` and its label depends on the token, so
a "model" is a lookup table `token -> label` learned from the training part by majority vote. `noise` (a recipe
parameter in [0, 1]) corrupts that share of the training labels before the vote: `noise=0` learns the task,
`noise=1` learns only wrong answers. The base model is the empty table, which answers nothing.

The pack is not registered as an entry point: a real pack is a separate distribution that registers its stages in
`gatedtrain.stages` and its `SCENARIO` (the generation function and the pipeline file) in `gatedtrain.scenarios`.
Tests inject this one by patching those two loaders (`tests/fixtures/ainative/run_cli.py`, `inject_pack` in
`tests/test_an_synthetic_pack.py`). The pipeline file `synthetic_pack.yaml` lists the five trust stages around the
six stages here. Every stage writes its outputs under its pipeline ref name (`ctx.stage`).

`smoke` trains on a few rows and scores a probe set with the exact-match scorer; `ext.smoke.smoke_verdict` (the
MM-38 rule for a binary metric) refuses a mean of 0 or 1, so a recipe that learns nothing never reaches `train`.
`train` declares `needs = {"gpu"}` and `trains = True` but runs on the CPU.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

from ..ainative import scorers
from ..ainative.project import ArtifactRef
from ..ainative.stage import Proposal, StageContext
from ..ainative.trust_stages import SCHEMA_DEPLOYMENT, SCHEMA_MODEL_REF, SCHEMA_TRAIN_PART
from ..errors import Problem, Refused
from ..ext.smoke import smoke_verdict

GENERATOR = "synthetic.lookup"
SCHEMA_VERSION = 1
SCHEMA_PROFILE = "gatedtrain.ainative.synthetic.profile"
SCHEMA_RECIPE = "gatedtrain.ainative.synthetic.recipe"
SCHEMA_SMOKE = "gatedtrain.ainative.synthetic.smoke"

SAMPLER = {"name": "lookup", "temperature": 1.0, "max_new_tokens": 1}
EVAL_SEEDS = [11, 12]
SMOKE_TRAIN_ROWS = 10
SMOKE_PROBE_ROWS = 20
NOISE_LABEL = "<noise>"


def _meta(schema: str) -> dict:
    return {"schema": schema, "schema_version": SCHEMA_VERSION}


def _refuse(rule: str, field_: str, message: str, fix: str) -> Refused:
    return Refused([Problem(rule, field_, message, fix)])


def feature(text: Any) -> Any:
    """The token a lookup model keys on: the text before the first '-'."""
    return text.split("-", 1)[0] if isinstance(text, str) else None


def generate(model_ref: dict, item: dict, *, sampler: dict | None = None, seed: int | None = None) -> Any:
    """The model's label for `item["input"]`, or None when its table has no entry for the token."""
    return (model_ref.get("table") or {}).get(feature(item["input"]))


def _share(noise_seed: int, text: str) -> float:
    return int(hashlib.sha256(f"{noise_seed}:{text}".encode()).hexdigest()[:8], 16) / 2 ** 32


def fit(rows: list[dict], noise: float, noise_seed: int = 0) -> dict[str, str]:
    """Majority label per token after `noise` of the rows (chosen by a hash of their input) lose their label."""
    votes: dict[str, Counter] = {}
    for row in rows:
        tok = feature(row["input"])
        label = NOISE_LABEL if _share(noise_seed, row["input"]) < noise else row["label"]
        votes.setdefault(tok, Counter())[label] += 1
    return {tok: sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] for tok, c in sorted(votes.items())}


def _read(ctx: StageContext, name: str) -> dict:
    rec = ctx.project.read_artifact(ctx.inputs[name])
    if not isinstance(rec, dict):
        raise _refuse("SYN-1", f"inputs.{name}", "the input record is not an object", "rerun the stage that writes it")
    return rec


def _rows(rec: dict, name: str) -> list[dict]:
    rows = rec.get("rows")
    if not (isinstance(rows, list) and all(isinstance(r, dict) and "input" in r and "label" in r for r in rows)):
        raise _refuse("SYN-1", f"inputs.{name}.rows", "the training rows are malformed", "rerun the stage that writes them")
    return rows


class _Base:
    version = "1"
    needs: frozenset[str] = frozenset()
    params: frozenset[str] = frozenset()
    trains = False

    def check(self, ctx: StageContext, outputs: dict[str, ArtifactRef]) -> list[Problem]:
        return []


class ProfileData(_Base):
    name = "profile_data"
    inputs = {"train_part": SCHEMA_TRAIN_PART}
    outputs = {"profile": SCHEMA_PROFILE}

    def _profile(self, ctx: StageContext) -> dict:
        rows = _rows(_read(ctx, "train_part"), "train_part")
        labels = Counter(str(r["label"]) for r in rows)
        return {"n": len(rows), "n_tokens": len({feature(r["input"]) for r in rows}), "n_labels": len(labels),
                "label_counts": dict(sorted(labels.items()))}

    def plan(self, ctx: StageContext) -> Proposal:
        p = self._profile(ctx)
        return Proposal(self.name, f"profile {p['n']} training rows: {p['n_tokens']} tokens, {p['n_labels']} labels",
                        {}, {"n": p["n"]})

    def run(self, ctx: StageContext, decision: dict) -> dict[str, ArtifactRef]:
        return {"profile": ctx.project.write_artifact(ctx.stage or self.name, "profile", {**_meta(SCHEMA_PROFILE), **self._profile(ctx)},
                                                      SCHEMA_PROFILE)}

    def check(self, ctx: StageContext, outputs: dict[str, ArtifactRef]) -> list[Problem]:
        p = ctx.project.read_artifact(outputs["profile"])
        if p.get("n") != sum(p.get("label_counts", {}).values()):
            return [Problem("SYN-1", "profile.n", "the row count differs from the label counts", "rerun profile_data")]
        return []


class ChooseTaskAndBase(_Base):
    name = "choose_task_and_base"
    inputs = {"profile": SCHEMA_PROFILE}
    outputs = {"deployment": SCHEMA_DEPLOYMENT, "base": SCHEMA_MODEL_REF}

    def plan(self, ctx: StageContext) -> Proposal:
        n = _read(ctx, "profile").get("n")
        return Proposal(self.name, f"task: map an input string to its label ({n} training rows); base model: the "
                        "empty lookup table", {"task": "lookup", "sampler": dict(SAMPLER), "eval_seeds": EVAL_SEEDS}, {})

    def run(self, ctx: StageContext, decision: dict) -> dict[str, ArtifactRef]:
        _read(ctx, "profile")
        deployment = {**_meta(SCHEMA_DEPLOYMENT), "sampler": dict(SAMPLER), "eval_seeds": list(EVAL_SEEDS)}
        base = {**_meta(SCHEMA_MODEL_REF), "generator": GENERATOR, "label": "base", "table": {}}
        return {"deployment": ctx.project.write_artifact(ctx.stage or self.name, "deployment", deployment, SCHEMA_DEPLOYMENT),
                "base": ctx.project.write_artifact(ctx.stage or self.name, "base", base, SCHEMA_MODEL_REF)}

    def check(self, ctx: StageContext, outputs: dict[str, ArtifactRef]) -> list[Problem]:
        base = ctx.project.read_artifact(outputs["base"])
        if base.get("table") != {}:
            return [Problem("SYN-1", "base.table", "the base model is not the empty table", "rerun choose_task_and_base")]
        return []


class BuildTrainSet(_Base):
    """Keeps the rows with a text input and a text label, one per input; emits a `train_part` shaped record."""

    name = "build_train_set"
    inputs = {"train_part": SCHEMA_TRAIN_PART}
    outputs = {"train_set": SCHEMA_TRAIN_PART}

    def _build(self, ctx: StageContext) -> dict:
        src = _read(ctx, "train_part")
        seen: set[str] = set()
        kept: list[dict] = []
        for r in _rows(src, "train_part"):
            if isinstance(r["input"], str) and isinstance(r["label"], str) and r["input"] not in seen:
                seen.add(r["input"])
                kept.append({"input": r["input"], "label": r["label"]})
        return {**_meta(SCHEMA_TRAIN_PART), "n": len(kept), "n_duplicates_removed": src.get("n_duplicates_removed", 0),
                "data_sha256": src.get("data_sha256"), "rows": kept}

    def plan(self, ctx: StageContext) -> Proposal:
        n = self._build(ctx)["n"]
        return Proposal(self.name, f"keep {n} text rows for training", {}, {"n": n})

    def run(self, ctx: StageContext, decision: dict) -> dict[str, ArtifactRef]:
        return {"train_set": ctx.project.write_artifact(ctx.stage or self.name, "train_set", self._build(ctx), SCHEMA_TRAIN_PART)}

    def check(self, ctx: StageContext, outputs: dict[str, ArtifactRef]) -> list[Problem]:
        t = ctx.project.read_artifact(outputs["train_set"])
        if t.get("n") != len(t.get("rows", [])):
            return [Problem("SYN-1", "train_set.n", "the row count does not match its rows", "rerun build_train_set")]
        return []


class WriteRecipe(_Base):
    name = "write_recipe"
    inputs = {"train_set": SCHEMA_TRAIN_PART}
    outputs = {"recipe": SCHEMA_RECIPE}
    params = frozenset({"noise"})

    def _noise(self, ctx: StageContext) -> float:
        noise = ctx.params.get("noise", 0.0)
        if isinstance(noise, bool) or not isinstance(noise, (int, float)) or not 0 <= noise <= 1:
            raise _refuse("SYN-1", "params.noise", f"noise {noise!r} is not a number from 0 to 1",
                          "set the share of training labels to corrupt, between 0 and 1")
        return float(noise)

    def plan(self, ctx: StageContext) -> Proposal:
        noise = self._noise(ctx)
        return Proposal(self.name, f"train a lookup table with {noise:g} of the labels corrupted", {"noise": noise}, {})

    def run(self, ctx: StageContext, decision: dict) -> dict[str, ArtifactRef]:
        recipe = {**_meta(SCHEMA_RECIPE), "noise": self._noise(ctx), "noise_seed": 0,
                  "train_set_sha256": ctx.inputs["train_set"].sha256, "smoke_train_rows": SMOKE_TRAIN_ROWS,
                  "smoke_probe_rows": SMOKE_PROBE_ROWS}
        return {"recipe": ctx.project.write_artifact(ctx.stage or self.name, "recipe", recipe, SCHEMA_RECIPE)}

    def check(self, ctx: StageContext, outputs: dict[str, ArtifactRef]) -> list[Problem]:
        r = ctx.project.read_artifact(outputs["recipe"])
        if r.get("train_set_sha256") != ctx.inputs["train_set"].sha256:
            return [Problem("SYN-1", "recipe.train_set_sha256", "the recipe is not for this training set",
                            "rerun write_recipe")]
        return []


class Smoke(_Base):
    """Fits a stand-in baseline (a table on rows[:10]), scores rows[10:30] with exact match and applies the MM-38
    verdict. The real base (the empty table) is not used: it always scores 0, which the binary-metric rule refuses."""

    name = "smoke"
    inputs = {"train_set": SCHEMA_TRAIN_PART, "recipe": SCHEMA_RECIPE}
    outputs = {"smoke": SCHEMA_SMOKE}

    def _scores(self, ctx: StageContext) -> list[float]:
        recipe = _read(ctx, "recipe")
        rows = _rows(_read(ctx, "train_set"), "train_set")
        n_fit, n_probe = recipe["smoke_train_rows"], recipe["smoke_probe_rows"]
        if len(rows) < n_fit + n_probe:
            raise _refuse("MM-38", "inputs.train_set", f"the smoke run needs {n_fit + n_probe} training rows, got "
                          f"{len(rows)}", "add labelled examples")
        table = fit(rows[:n_fit], recipe["noise"], recipe["noise_seed"])
        model = {"table": table}
        return [scorers.exact_match(generate(model, r), r["label"]) for r in rows[n_fit:n_fit + n_probe]]

    def plan(self, ctx: StageContext) -> Proposal:
        n = _read(ctx, "recipe")["smoke_probe_rows"]
        return Proposal(self.name, f"fit on a few rows and score {n} probe rows; refuse a degenerate baseline", {}, {})

    def run(self, ctx: StageContext, decision: dict) -> dict[str, ArtifactRef]:
        scores = self._scores(ctx)
        record = {"peak_gpu_mem_gb": 0.0, "projected_gpu_h": 0.0, "gpu_log_total_h": 0.0, "error_count": 0,
                  "baseline_scores": scores}
        verdict = smoke_verdict(record, score_range=(0, 1), binary=True, direction=scorers.DIRECTION)
        if verdict["verdict"] != "PASS":
            raise _refuse("MM-38", "smoke.baseline_scores", "the smoke baseline is degenerate "
                          f"({', '.join(verdict['reasons'])}; mean {sum(scores) / len(scores):g})",
                          "change the recipe or the data so the smoke model neither learns everything nor nothing")
        out = {**_meta(SCHEMA_SMOKE), "verdict": "PASS", "reasons": [], "baseline_scores": scores,
               "mean": sum(scores) / len(scores)}
        return {"smoke": ctx.project.write_artifact(ctx.stage or self.name, "smoke", out, SCHEMA_SMOKE)}

    def check(self, ctx: StageContext, outputs: dict[str, ArtifactRef]) -> list[Problem]:
        s = ctx.project.read_artifact(outputs["smoke"])
        if s.get("verdict") != "PASS" or not s.get("baseline_scores"):
            return [Problem("MM-38", "smoke.verdict", "the smoke record is not a PASS with scores", "rerun smoke")]
        return []


class Train(_Base):
    """Declares the `gpu` need (to exercise the order rule) and `trains`, but fits a table on the CPU."""

    name = "train"
    inputs = {"train_set": SCHEMA_TRAIN_PART, "recipe": SCHEMA_RECIPE, "smoke": SCHEMA_SMOKE}
    outputs = {"model": SCHEMA_MODEL_REF}
    needs = frozenset({"gpu"})
    trains = True

    def plan(self, ctx: StageContext) -> Proposal:
        n = _read(ctx, "train_set").get("n")
        return Proposal(self.name, f"fit the lookup table on {n} rows (CPU)", {"noise": _read(ctx, "recipe")["noise"]},
                        {"gpu_hours": 0.0})

    def run(self, ctx: StageContext, decision: dict) -> dict[str, ArtifactRef]:
        if _read(ctx, "smoke").get("verdict") != "PASS":
            raise _refuse("MM-38", "inputs.smoke", "the smoke run did not pass", "rerun smoke")
        recipe = _read(ctx, "recipe")
        rows = _rows(_read(ctx, "train_set"), "train_set")
        model = {**_meta(SCHEMA_MODEL_REF), "generator": GENERATOR, "label": "candidate",
                 "noise": recipe["noise"], "n_train": len(rows),
                 "table": fit(rows, recipe["noise"], recipe["noise_seed"])}
        return {"model": ctx.project.write_artifact(ctx.stage or self.name, "model", model, SCHEMA_MODEL_REF)}

    def check(self, ctx: StageContext, outputs: dict[str, ArtifactRef]) -> list[Problem]:
        m = ctx.project.read_artifact(outputs["model"])
        if m.get("generator") != GENERATOR or not isinstance(m.get("table"), dict):
            return [Problem("SYN-1", "model", "the model is not a lookup table of this pack", "rerun train")]
        return []


@dataclass(frozen=True)
class Scenario:
    """What the `gatedtrain.scenarios` entry point exposes: the generation functions of the pack, by the name a
    model reference gives in `generator`, and the path of the pack's pipeline file."""

    name: str
    generators: Mapping[str, Callable[..., Any]] = field(default_factory=dict)
    pipeline: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "generators", MappingProxyType(dict(self.generators)))


SCENARIO = Scenario("synthetic", {GENERATOR: generate}, Path(__file__).with_name("synthetic_pack.yaml"))
