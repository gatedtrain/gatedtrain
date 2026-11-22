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

The batch form (`synthetic_pack_batch.yaml`, a keeper project) swaps two stages: `choose_task_and_base_files` writes the
base model's table as a file, and `train_job` runs the fit's hand-off as a CPU job of the local backend
(`ctx.backend`), so the models are files the keeper evaluates with the batch generator in its inference container.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import stat
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

from ..ainative import scorers
from ..ainative.batch import BatchGenerator
from ..canonical import sha256_bytes
from ..ainative.project import ArtifactRef
from ..ainative.stage import Proposal, StageContext
from ..ainative.trust_stages import SCHEMA_DEPLOYMENT, SCHEMA_MODEL_REF, SCHEMA_TRAIN_PART
from ..errors import Problem, Refused
from ..ext.smoke import smoke_verdict
from .synthetic_batch_generate import feature

GENERATOR = "synthetic.lookup"
# The batch form of the same model (AN-25): a standard-library script that reads `<model>/table.json`, for the
# keeper's inference container. `synthetic_pack.generate` stays the per-item form.
BATCH_GENERATOR = "synthetic_table_batch"
IMAGE_PYTHON = "/usr/local/bin/python3"
BATCH = BatchGenerator(script=Path(__file__).resolve().with_name("synthetic_batch_generate.py"),
                       interpreter=IMAGE_PYTHON, time_limit_s=600)
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
    """Declares the `gpu` need (to exercise the order rule) and `trains`, but fits a table on the CPU.

    Test only: the decision key `noise_override` (a number from 0 to 1) fits with that noise instead of the recipe's,
    so a test can make a second candidate in one project without editing the pipeline, whose hash the project pins
    (preflight finding 1). A real pack has no such key."""

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
        noise = decision.get("noise_override", recipe["noise"])
        if isinstance(noise, bool) or not isinstance(noise, (int, float)) or not 0 <= noise <= 1:
            raise _refuse("SYN-1", "decision.noise_override", f"noise_override {noise!r} is not a number from 0 to 1",
                          "leave it out, or give the share of training labels to corrupt")
        model = {**_meta(SCHEMA_MODEL_REF), "generator": GENERATOR, "label": "candidate",
                 "noise": noise, "n_train": len(rows),
                 "table": fit(rows, noise, recipe["noise_seed"])}
        return {"model": ctx.project.write_artifact(ctx.stage or self.name, "model", model, SCHEMA_MODEL_REF)}

    def check(self, ctx: StageContext, outputs: dict[str, ArtifactRef]) -> list[Problem]:
        m = ctx.project.read_artifact(outputs["model"])
        if m.get("generator") != GENERATOR or not isinstance(m.get("table"), dict):
            return [Problem("SYN-1", "model", "the model is not a lookup table of this pack", "rerun train")]
        return []


# ---------------------------------------------------------------------------------------------------------------------
# The batch form of the pipeline (`synthetic_pack_batch.yaml`, AN-24, AN-25): the models are files, the candidate is
# made by a job of the local backend, and the keeper evaluates both in its inference container.
# ---------------------------------------------------------------------------------------------------------------------

TRAIN_JOB_SCRIPT = Path(__file__).resolve().with_name("synthetic_train_job.py")
TRAIN_JOB_POLL_S = 2.0
TABLE = "table.json"


def _table_bytes(table: dict) -> bytes:
    """The table as `synthetic_train_job.py` writes it (the batch writer recipe)."""
    return json.dumps(table, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _write_file(path: Path, data: bytes) -> None:
    """`path` replaced by `data` (mode 0644, never through a link): a temporary file in the same directory, renamed.
    A regular file that already holds `data` is left as it is (a re-run while a job reads it: same inode)."""
    with contextlib.suppress(OSError):
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as f:
            st = os.fstat(f.fileno())
            if stat.S_ISREG(st.st_mode) and stat.S_IMODE(st.st_mode) == 0o644 and f.read(len(data) + 1) == data:
                return
    tmp = path.with_name(f".{path.name}.tmp")
    with contextlib.suppress(FileNotFoundError):
        tmp.unlink()
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def _backend(ctx: StageContext, stage: str) -> Any:
    if ctx.backend is None:
        raise _refuse("SYN-1", "backend", f"stage {stage!r} needs the project's job backend (ctx.backend)",
                      "run the stage through the stage machine")
    return ctx.backend


class ChooseTaskAndBaseFiles(ChooseTaskAndBase):
    """`choose_task_and_base` for the batch generator: the base model is the empty table written as a file into a
    backend output directory, and its reference lists that file with its hash (`{path, sha256}`, as `hand_off`
    returns them) for the keeper to copy."""

    name = "choose_task_and_base_files"

    def run(self, ctx: StageContext, decision: dict) -> dict[str, ArtifactRef]:
        _read(ctx, "profile")
        out = _backend(ctx, self.name).new_output_dir("base")
        data = _table_bytes({})
        _write_file(out / TABLE, data)
        deployment = {**_meta(SCHEMA_DEPLOYMENT), "sampler": dict(SAMPLER), "eval_seeds": list(EVAL_SEEDS)}
        base = {**_meta(SCHEMA_MODEL_REF), "generator": BATCH_GENERATOR, "label": "base",
                "files": [{"path": str(out / TABLE), "sha256": sha256_bytes(data)}]}
        return {"deployment": ctx.project.write_artifact(ctx.stage or self.name, "deployment", deployment,
                                                         SCHEMA_DEPLOYMENT),
                "base": ctx.project.write_artifact(ctx.stage or self.name, "base", base, SCHEMA_MODEL_REF)}

    def check(self, ctx: StageContext, outputs: dict[str, ArtifactRef]) -> list[Problem]:
        base = ctx.project.read_artifact(outputs["base"])
        files = base.get("files")
        if base.get("generator") != BATCH_GENERATOR or not (isinstance(files, list) and len(files) == 1):
            return [Problem("SYN-1", "base.files", "the base model is not one table file of the batch generator",
                            "rerun choose_task_and_base")]
        try:
            data = Path(files[0]["path"]).read_bytes()
        except (OSError, KeyError, TypeError):
            return [Problem("SYN-1", "base.files[0]", "the base model's table file cannot be read",
                            "rerun choose_task_and_base")]
        if data != _table_bytes({}) or files[0].get("sha256") != sha256_bytes(data):
            return [Problem("SYN-1", "base.files[0]", "the base model's file is not the empty table it lists",
                            "rerun choose_task_and_base")]
        return []


class TrainJob(_Base):
    """`train` as a CPU job of the local backend: the table is fitted in process (`fit`), written with
    `synthetic_train_job.py` to the output directory `train-inputs`, and the job (stand-in adapter, the pipeline's
    `image`, no GPU) copies it to its output; the candidate reference lists the job's `table.json` from `hand_off`.
    The job is built before its output directory, which is asked for by the job's key (`new_output_dir("train",
    key=job.key())`): a re-run with the same inputs gets the same directory and `submit` reattaches to the same job,
    so a driver that died while the job ran is replaced by running the stage again."""

    name = "train_job"
    inputs = {"train_set": SCHEMA_TRAIN_PART, "recipe": SCHEMA_RECIPE, "smoke": SCHEMA_SMOKE}
    outputs = {"model": SCHEMA_MODEL_REF}
    params = frozenset({"image"})
    trains = True

    def _image(self, ctx: StageContext) -> str:
        image = ctx.params.get("image")
        if not isinstance(image, str) or not image:
            raise _refuse("SYN-1", "params.image", f"image {image!r} is not an image reference",
                          "set the pipeline's `image` param to an image pinned by digest that has "
                          f"{IMAGE_PYTHON}")
        return image

    def plan(self, ctx: StageContext) -> Proposal:
        n = _read(ctx, "train_set").get("n")
        return Proposal(self.name, f"fit the lookup table on {n} rows and hand it off through a CPU job in image "
                        f"{self._image(ctx)}", {"noise": _read(ctx, "recipe")["noise"]}, {"gpu_hours": 0.0})

    def run(self, ctx: StageContext, decision: dict) -> dict[str, ArtifactRef]:
        from ..ainative.backend import Job
        from ..ainative.runtime import Mount
        from . import standin

        if _read(ctx, "smoke").get("verdict") != "PASS":
            raise _refuse("MM-38", "inputs.smoke", "the smoke run did not pass", "rerun smoke")
        image = self._image(ctx)
        backend = _backend(ctx, self.name)
        recipe = _read(ctx, "recipe")
        rows = _rows(_read(ctx, "train_set"), "train_set")
        table = fit(rows, recipe["noise"], recipe["noise_seed"])
        inputs = backend.new_output_dir("train-inputs")
        _write_file(inputs / TABLE, _table_bytes(table))
        _write_file(inputs / TRAIN_JOB_SCRIPT.name, TRAIN_JOB_SCRIPT.read_bytes())
        job = Job(label="train", adapter=standin.NAME, runtime={"kind": "container", "image": image},
                  framework={"name": standin.FRAMEWORK_PACKAGE, "version": "0.1", "commit": None},
                  argv=(IMAGE_PYTHON, f"/inputs/{TRAIN_JOB_SCRIPT.name}", "--table", f"/inputs/{TABLE}",
                        "--out", "/out"),
                  env={"HF_HUB_OFFLINE": "1", "HOME": "/tmp/home"},
                  mounts=tuple(Mount(str(inputs / n), f"/inputs/{n}") for n in (TRAIN_JOB_SCRIPT.name, TABLE)),
                  output_dir="", output_container="/out", tmpfs=("/tmp",))
        job = dataclasses.replace(job, output_dir=str(backend.new_output_dir("train", key=job.key())))
        job_id = backend.submit(job)
        status = backend.wait(job_id, poll_s=TRAIN_JOB_POLL_S)
        if status.state != "succeeded":
            raise _refuse("SYN-1", "train.job", f"the train job {job_id} ended {status.state}"
                          + (f" ({status.reason})" if status.reason else ""),
                          "read `gatedtrain project job logs`, fix the cause and run the stage again")
        model = {**_meta(SCHEMA_MODEL_REF), "generator": BATCH_GENERATOR, "label": "candidate",
                 "noise": recipe["noise"], "n_train": len(rows), "job_id": job_id,
                 "files": backend.hand_off(job_id, [TABLE])}
        return {"model": ctx.project.write_artifact(ctx.stage or self.name, "model", model, SCHEMA_MODEL_REF)}

    def check(self, ctx: StageContext, outputs: dict[str, ArtifactRef]) -> list[Problem]:
        m = ctx.project.read_artifact(outputs["model"])
        files = m.get("files")
        if m.get("generator") != BATCH_GENERATOR or not (isinstance(files, list) and len(files) == 1
                                                         and Path(str(files[0].get("path"))).name == TABLE):
            return [Problem("SYN-1", "model", "the model is not one table file of this pack's batch generator",
                            "rerun train")]
        return []


@dataclass(frozen=True)
class Scenario:
    """What the `gatedtrain.scenarios` entry point exposes: the generation functions of the pack, by the name a
    model reference gives in `generator`, the path of the pack's pipeline file, and the pack's scorers registered by
    reference (`{name: {id, fn, code_sha256, planning_sd, direction, range, what, unit[, label_ok]}}`, AN-17; this
    pack declares none), and its batch generators by name (`BatchGenerator` declarations, AN-25; a name is either a
    per-item or a batch generator)."""

    name: str
    generators: Mapping[str, Callable[..., Any]] = field(default_factory=dict)
    pipeline: Path | None = None
    scorers: Mapping[str, dict] = field(default_factory=dict)
    batch_generators: Mapping[str, BatchGenerator] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "generators", MappingProxyType(dict(self.generators)))
        object.__setattr__(self, "batch_generators", MappingProxyType(dict(self.batch_generators)))
        object.__setattr__(self, "scorers", MappingProxyType({k: MappingProxyType(dict(v))
                                                              for k, v in self.scorers.items()}))


SCENARIO = Scenario("synthetic", {GENERATOR: generate}, Path(__file__).with_name("synthetic_pack.yaml"),
                    batch_generators={BATCH_GENERATOR: BATCH})
# The batch form of the pipeline (`choose_task_and_base_files`, `train_job`); a pack exposes one pipeline file, so this
# one is passed to `project create --pipeline` by path.
BATCH_PIPELINE = Path(__file__).with_name("synthetic_pack_batch.yaml")
