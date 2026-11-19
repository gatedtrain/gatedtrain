"""`gatedtrain ext-from-nel`: K single-seed NeMo Evaluator runs to one per-item arm (tests/README.md section 11.18
EXT-35 to EXT-38, EXT-G4; MULTIMODAL-SPEC.md MM-1, MM-2).

The runs are read as `merge-eval` reads them, with the same strict helpers (`tierb._read_run_strict`, R11A-28;
`tierb._run_seed_and_repeats`, `tierb._run_sampler`, `tierb._declared_scorers`): `<run>/<benchmark>/results.jsonl`
(orchestration/orchestrator.py:1046-1047, :1081 at ea35ba3) and the composed-config snapshot
`<run>/full_config.yaml` (config/snapshot.py:72-81), whose header line `# nemo-evaluator version: <v>`
(snapshot.py:59) names the framework version. Run r is repeat r with seed `eval_seeds[r]` of the registration,
in registered order.

Each record is converted by `items.from_nemo_evaluator(record, scorer_key="scorer:<scorer>")` under the deciding
evaluator's registry scorer (`scorer_rewards["scorer:numeric_match"]`, engine/eval_loop.py:785-794): a raising
scorer (`scoring_details[...] = {error, reward: 0.0}`, eval_loop.py:719-721) becomes an error record
`scorer_error`, a missing value `missing_score`; ext-collect counts both as missing items (EXT-4). `item` is the
registered item set's key of row `problem_idx` (never the index), `item_meta` keeps `problem_idx` and the SHA-256
of the model response (design choice: MM-1's `output_ref` is a file in the arm).

The registration is read with `--evaluator` deciding (R11A-23), never through `tierb.load_registration`
(R12A-27). Refusals: `EXT-NEL` (nothing written), `REG-FIELD` for an item set other than the registered one,
`EXT-ADAPTER` for an unknown adapter.

Evaluated model (EXT-47, a choice): `merge.json` records each run's NeMo Evaluator service `model`
(`services.<solver service>.model`, config/services.py:240-259 at ea35ba3) as `service_model`, and runs naming
different service models are refused. The service model is the name the inference server was started with
(for vLLM `--served-model-name`, or a path inside its container), so it cannot be compared with `--checkpoint`
on this host: that the server loaded `--checkpoint` is the operator's claim, and `checkpoint.model_hash` is the
hash of the directory the operator names.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import yaml

from .. import TAGS
from ..canonical import canonical_json, file_sha256, manifest_hash, sha256_bytes, text_lines, write_canonical
from ..errors import Problem, Refused
from . import registry
from .collect import read_item_set
from .arm import items_text
from .merge import load_registration, write_problem

FROM_NEL_SCHEMA = "gatedtrain.ext_from_nel"
VERSION_HEADER = "# nemo-evaluator version:"


def _framework_version(config_text: str) -> str | None:
    """The snapshot header's `nemo-evaluator version` (config/snapshot.py:59, :72-81 at ea35ba3)."""
    for line in text_lines(config_text):
        if line.startswith(VERSION_HEADER):
            return line[len(VERSION_HEADER):].strip() or None
        if not line.startswith("#"):
            break
    return None


def from_nel(*, registration: Path, adapter: str, evaluator: str, items: Path, checkpoint: Path, out: Path,
             runs: list[Path]) -> Path:
    """Convert K NeMo Evaluator runs into the arm `out` (EXT-35, EXT-36). Raises Refused; then writes nothing."""
    from ..registration import clause_for_evaluator, item_set_field, item_set_hash
    from ..tierb import _declared_scorers, _read_run_strict, _run_sampler, _run_seed_and_repeats

    try:
        ad = registry.adapter(adapter)
    except KeyError:
        raise Refused([Problem("EXT-ADAPTER", "--adapter", f"no adapter {adapter!r} is registered",
                               f"use one of {', '.join(sorted(registry.ADAPTERS))}")])
    out = Path(out)
    fix = "pass the K finished single-seed NeMo Evaluator runs of this registration, in registered seed order"
    if out.exists():
        raise Refused([Problem("EXT-NEL", "--out", f"{out} already exists", "pass a new --out directory")])
    if evaluator not in ad.evaluator_ids:
        raise Refused([Problem("EXT-NEL", "--evaluator", f"adapter {adapter} gates on {', '.join(ad.evaluator_ids)}, "
                               f"not {evaluator}", f"pass --evaluator {ad.evaluator_ids[0]}")])
    reg = load_registration(registration, evaluator, rule="EXT-NEL")
    row = registry.EVALUATOR_ROWS[evaluator]
    seeds = list(clause_for_evaluator(reg, evaluator)["eval_seeds"])    # R18-1: the clause it decides
    sampler = reg["sampler"]
    scorer = row["scorer"]["name"]
    key = f"scorer:{scorer}"
    problems: list[Problem] = []

    # Item set (EXT-36) and the evaluated checkpoint (EXT-14).
    item_rows, item_bad = read_item_set(Path(items), tuple(row["item_payload"]))
    problems += [Problem("EXT-ITEMS", "--items", m, "pass the registered item set directory") for m in item_bad]
    actual, hashed = None, True
    if (Path(items) / "items.jsonl").is_file():
        try:
            actual = manifest_hash(items)
        except (OSError, ValueError) as exc:                # EXT-41: a symbolic link or unreadable file
            hashed = False
            problems.append(Problem("EXT-ITEMS", "--items", f"{items} cannot be hashed: {exc}",
                                    "pass the registered item set as a plain directory"))
    if hashed and actual != item_set_hash(reg):
        problems.append(Problem("REG-FIELD", item_set_field(reg), f"registered item set {item_set_hash(reg)} but "
                                f"--items {items} is {actual}", "pass the registered item set"))
    bad = ad.model_dir_problem(Path(checkpoint))
    model = None
    if bad:
        problems.append(Problem("EXT-NEL", "--checkpoint", f"{checkpoint} is not a model directory: {bad}",
                                "pass the evaluated base directory or checkpoint-<N> cut"))
    else:
        model = ad.identity(Path(checkpoint), None)["model_hash"]
    if len(runs) != len(seeds):
        problems.append(Problem("EXT-NEL", "runs", f"{len(runs)} runs given for K = {len(seeds)} registered seeds",
                                f"pass exactly {len(seeds)} runs, in the order of eval_seeds {seeds}"))
    if problems:
        raise Refused(problems)

    per_run = []
    versions = set()
    for r, run in enumerate(map(Path, runs)):
        fld = f"runs[{r}]"
        results = run / row["nel_benchmark"] / "results.jsonl"
        recs, bad_lines = _read_run_strict(results)
        if bad_lines:
            problems += [Problem("EXT-NEL", fld, f"{results}: {m}", fix) for m in bad_lines]
            continue
        if not recs:
            problems.append(Problem("EXT-NEL", fld, f"{results} is missing or holds no records", fix))
            continue
        try:
            config_text = (run / "full_config.yaml").read_text(encoding="utf-8")
            seed, repeats = _run_seed_and_repeats(run)
            used = _run_sampler(run)
            declared = _declared_scorers(yaml.safe_load(config_text))
        except (OSError, ValueError, yaml.YAMLError, AttributeError) as exc:
            problems.append(Problem("EXT-NEL", fld, f"cannot read {run}/full_config.yaml: {exc.__class__.__name__}",
                                    "pass the run's own output directory, with its config snapshot"))
            continue
        if repeats != 1 or any(rep != 0 for _, rep in recs):
            problems.append(Problem("EXT-NEL", fld, f"the run has repeats = {repeats} or records with repeat other "
                                    "than 0; one run per seed (G-16)", "run with `repeats: 1`"))
        if seed != seeds[r]:
            problems.append(Problem("EXT-NEL", fld, f"the run's seed is {seed}, but repeat {r} is registered with "
                                    f"eval_seeds[{r}] = {seeds[r]}", fix))
        want = {"temperature": sampler.get("temperature"), "top_p": sampler.get("top_p"),
                "max_tokens": sampler.get("max_new_tokens"), "enable_thinking": bool(sampler.get("enable_thinking"))}
        for k, v in want.items():
            if k == "top_p" and v is None:
                continue
            if used[k] != v:
                problems.append(Problem("EXT-NEL", fld, f"the run used {k} = {used[k]!r}, but the registration fixes "
                                        f"{v!r} (05 section 3.7)", "regenerate the run configs and re-run"))
        if scorer not in (declared or []):
            problems.append(Problem("EXT-NEL", fld, f"the run's full_config.yaml declares scorers {declared!r}, not "
                                    f"the registry scorer `{scorer}` (G-21)", "evaluate with the scorer declared"))
        versions.add(_framework_version(config_text))
        per_run.append({"run": run, "recs": recs, "results": results, "config": run / "full_config.yaml",
                        "service_model": used.get("model")})
    if not problems:
        if len(versions) != 1 or None in versions:
            problems.append(Problem("EXT-NEL", "runs", f"the runs' nemo-evaluator versions are {sorted(map(str, versions))}"
                                    ", not one recorded version", "evaluate every run with one NeMo Evaluator install"))
        models = [info["service_model"] for info in per_run]
        if len({json.dumps(m) for m in models}) != 1:
            problems.append(Problem("EXT-NEL", "runs", f"the runs' service models are {models}, not one model "
                                    "(EXT-47)", "evaluate every seed against the same served checkpoint"))
        base_set = {pid for pid, _ in per_run[0]["recs"]}
        for r, info in enumerate(per_run[1:], start=1):
            if {pid for pid, _ in info["recs"]} != base_set:
                problems.append(Problem("EXT-NEL", f"runs[{r}]", "the problem set differs from runs[0]", fix))
        field = row["payload_text_field"]
        outside, drift = set(), set()
        for info in per_run:
            for (pid, _), rec in info["recs"].items():
                if not 0 <= pid < len(item_rows):
                    outside.add(pid)
                elif item_rows[pid][field] not in str(rec.get("prompt", "")):
                    drift.add(pid)
        if outside:
            problems.append(Problem("EXT-NEL", "runs", f"problem_idx {sorted(outside)[:5]} outside the "
                                    f"{len(item_rows)} items of the item set", "evaluate the registered item set only"))
        if drift:
            problems.append(Problem("EXT-NEL", "runs", f"the item's {field} is not in the record's prompt for "
                                    f"problem_idx {sorted(drift)[:5]} (dataset drift)",
                                    "evaluate the dataset revision the item set was built from"))
    if problems:
        raise Refused(problems)

    records, bad = convert_records([info["recs"] for info in per_run], item_rows, seeds, key)
    problems += [Problem("EXT-NEL", f"runs[{r}]", msg, "re-run the evaluation") for r, msg in bad]
    if problems:
        raise Refused(problems)
    meta = {"schema": "gatedtrain.item_eval", "schema_version": "1.0", "evaluator_id": evaluator, "adapter": adapter,
            "metric": row["metric"], "direction": row["direction"], "score_range": list(row["score_range"]),
            "checkpoint": {"model_hash": model, "base_model_hash": None}, "normalizer_sha256": None,
            "env_version_sha256": None, "item_set_hash": actual, "eval_seeds": seeds, "sampler": sampler,
            "sampler_hash": sha256_bytes(canonical_json(sampler)), "scorer": dict(row["scorer"]),
            "framework": {"name": row["framework"], "version": versions.pop()}}

    try:                                                  # EXT-75: the parent cannot hold the output
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.tmp-", dir=out.parent))
    except OSError as exc:
        raise Refused([write_problem(exc, "--out", out)]) from exc
    try:
        (tmp / "items.jsonl").write_text(items_text(records), encoding="utf-8")
        write_canonical(tmp / "eval_meta.json", meta)
        write_canonical(tmp / "merge.json", {
            "schema": FROM_NEL_SCHEMA, "schema_version": "1.0", "tags": list(TAGS),
            "sources": [{"repeat": r, "seed": seeds[r], "run_dir": str(info["run"].absolute()),
                         "results_sha256": file_sha256(info["results"]), "config_sha256": file_sha256(info["config"]),
                         "service_model": info["service_model"]}
                        for r, info in enumerate(per_run)],
            "notes": [f"Scores are NeMo Evaluator scorer_rewards[{key!r}] (engine/eval_loop.py:785-794 at ea35ba3); "
                      "a raising scorer or a missing value is an error record, never 0 (eval_loop.py:719-721).",
                      "service_model is the served model name in each run's config; that the server loaded "
                      "--checkpoint is a claim (EXT-47)."],
        })
        os.rename(tmp, out)
    except OSError as exc:                  # EXT-59, EXT-75: a failed write is a refusal; other errors are bugs
        shutil.rmtree(tmp, ignore_errors=True)
        raise Refused([write_problem(exc, "--out", out)]) from exc
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return out


def convert_records(runs: list[dict], item_rows: list[dict], seeds: list[int],
                    key: str) -> tuple[list[dict], list[tuple[int, str]]]:
    """Run r's `{(problem_idx, repeat): record}` (`tierb._read_run_strict`) to arm records of repeat r, sorted by
    (problem_idx, repeat); also [(r, problem)] for a score that is not a finite number or a problem_idx outside
    the item set. `ext-from-nel` writes these records, and `ext-collect` re-derives them (EXT-52)."""
    from ..items import from_nemo_evaluator

    records, bad = [], []
    for r, recs in enumerate(runs):
        for (pid, _), rec in sorted(recs.items()):
            if not (isinstance(pid, int) and 0 <= pid < len(item_rows)) or r >= len(seeds):
                bad.append((r, f"problem_idx {pid} is outside the {len(item_rows)} items of the item set"))
                continue
            conv = from_nemo_evaluator(rec, scorer_key=key)
            score = conv["score"]
            if score is not None and not (isinstance(score, (int, float)) and not isinstance(score, bool)
                                          and math.isfinite(score)):
                bad.append((r, f"problem_idx {pid}: scorer_rewards[{key!r}] is {score!r}, not a finite number"))
                continue
            resp = rec.get("model_response")
            records.append({
                "item": item_rows[pid]["item"], "repeat": r, "seed": seeds[r], "score": score, "error": conv["error"],
                "sub_metrics": None, "output_ref": None,
                "item_meta": {"problem_idx": pid,
                              "model_response_sha256": ("sha256:" + hashlib.sha256(resp.encode("utf-8")).hexdigest()
                                                        if isinstance(resp, str) else None)}})
    records.sort(key=lambda x: (x["item_meta"]["problem_idx"], x["repeat"]))
    return records, bad
