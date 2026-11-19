"""Tier B helpers for one GPU host (TIER-B-SPEC H-1 to H-4 and H-6; tests/README.md section 11).

This code never launches a Tier B job; these helpers prepare inputs and turn recorded
NeMo RL (4f39fd8) and NeMo Evaluator (ea35ba3) outputs into a run directory
with the same layout as `gatedtrain run`. Citations are file:line at those pins.

- H-1 `export_gsm8k`: GSM8K as local shard files, their concatenation, the
  held-out prompt set and NeMo Evaluator's prompt (G-14, G-15, G-17, G-19).
- H-2 `verify_export`: re-checks the export before a run (L-5).
- H-3 `merge_eval`: K single-seed NeMo Evaluator runs into one arm (G-16).
- H-4 `nel_configs`: K NeMo Evaluator config files from the registration.
- H-6 `collect`: manifests, lockfile, ledger, lineage, gate and report from
  the recorded outputs; refuses before writing anything. Unpaired items no
  longer refuse: above `pairing.max_unpaired` the gate is MISSING (05 section
  3.4; R10A-3, supersedes G-18's TIERB-PAIRING refusal).
- G-21: the deciding evaluator's registered `scorer` (R11A-23) flows through H-4 (scoring block), H-3 (config copies
  and check) and H-6 (gate on scorer_rewards, TIERB-SCORER) (tests/README.md section 11.14).
- Design revision 11 (tests/README.md section 11.16): registrations and specs in item names or the
  revision 10 names are read (R11A-2); `collect` gates on the registered item set (R11A-17: a held-out
  problem absent from both arms is a missing item, `absent`; a record without `metadata.eval_seed` is a
  missing item, `no_seed`; only problems outside the held-out set are refused), checks the item-set hash
  against `eval_distribution.item_set_hash`, writes item names (R11A-4), and declares NeMo RL's checkpoint
  completeness rule in the lockfile (R11A-20). Scorer errors and out-of-range rewards stay refusals
  (11.14 D-3, D-11; design choice 7). The export keeps `export.json` `prompt_set_hash` (an H-1 artifact).

Every refusal raises `Refused` with rule, field and fix, and nothing
is written.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from . import ANALYSIS_CODE_VERSION, TAGS
from .canonical import (canonical_json, entries_hash, file_sha256, manifest_document, manifest_entries,
                        manifest_hash, sha256_bytes, text_lines, write_canonical)
from .errors import Problem, Refused
from .gates import load_paired_records

DATASET, DATASET_CONFIG = "openai/gsm8k", "main"
# The dataset commit `export-gsm8k` downloads when no `--revision` is given: the one the reference Tier B export used
# (its export.json `revision`). A download names a full 40-hex commit, never a branch, so the bytes cannot move.
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
# SHA-256 of the parquet files at DATASET_REVISION (the Hub's LFS oids, checked against a download on 2026-11-18).
# A download at this revision is refused unless its bytes match; another revision has no expected value.
DATASET_PARQUET_SHA256 = {
    DATASET_REVISION: {
        "train": "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d",
        "test": "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59",
    },
}
REVISION_RE = re.compile(r"[0-9a-f]{40}")
NEMO_EVALUATOR_COMMIT = "ea35ba3"
EVALUATOR_ID = "gsm8k.nemo_evaluator@0.1"
TRAINER_ID = "rl.nemo_rl@0.1"
DEFAULT_ROWS_PER_SHARD = 500

# NeMo Evaluator's GSM8K prompt (src/nemo_evaluator/benchmarks/gsm8k.py:22-26 at ea35ba3) with
# `{question}` replaced by `{}`, because NeMo RL fills the prompt with `prompt.format(problem)`
# (nemo_rl/data/processors.py:574-576). Written with no trailing newline (G-15).
NEL_GSM8K_PROMPT = (
    "Solve the following math problem step by step. "
    "Put your final numerical answer after 'The answer is'.\n\n"
    "Problem: {}"
)

# The G-16 note carried by merge.json and the Tier B report.
SEED_NOTE = ("NeMo Evaluator sends one seed per run (orchestration/orchestrator.py:881-889 at ea35ba3), so the "
             "K registered evaluation seeds come from K single-seed runs merged by gatedtrain merge-eval (G-16); "
             "vLLM seeded sampling is reproducible only up to batch-composition numerics.")

# The G-21 note carried by the Tier B report when the registration names `numeric_match` (decision D-7).
NUMERIC_MATCH_NOTE = ("The gate reads NeMo Evaluator's numeric_match scorer (G-21): it compares the last number "
                      "in the response, commas removed, after stripping trailing zeros and then a trailing dot on "
                      "both sides (scoring/pattern.py:39-43 at ea35ba3), so integers without a trailing dot can "
                      "match falsely (10 and 1 match; 10. and 1 do not). Observed false-positive rate on the smoke "
                      "baseline: 25/5,276 = 0.47%.")


# R11A-20 (08 section 2.4 item 1, R11-15): the checkpoint completeness rule this adapter declares for NeMo RL
# 4f39fd8. Local storage writes a checkpoint into `tmp_step_N` and publishes it by renaming the directory to
# `step_N` (nemo_rl/utils/checkpoint_storage.py:47, :69-:98; nemo_rl/utils/checkpoint.py:379-392, :423-434),
# so only `step_N` directories are cuts (ledger.CKPT_DIR_RE); a `tmp_step_N` directory is an interrupted
# write and gets no lineage row and no manifest.
CHECKPOINT_COMPLETENESS = {
    "nemo_rl_commit": "4f39fd8",
    "rule": "a checkpoint is complete when its directory is named step_<N>; NeMo RL writes into tmp_step_<N> "
            "and renames it to step_<N> after the write (checkpoint_storage.py:47, :69-:98; checkpoint.py:379-392, "
            ":423-434), so a tmp_step_<N> directory is an interrupted write and never a cut",
    "complete_dir_pattern": "^step_(\\d+)$",
    "incomplete_dir_pattern": "^tmp_step_(\\d+)$",
}


def evaluator_identity_hash(scorer: str | None = None) -> str:
    """Decision D-4: the Tier B evaluator identity hash, SHA-256 of the canonical JSON {benchmark, evaluator,
    nemo_evaluator_commit[, scorer]}. The registration carries it as `evaluators[0].manifest_hash` (R10A-2)."""
    identity = {"evaluator": EVALUATOR_ID, "benchmark": "gsm8k", "nemo_evaluator_commit": NEMO_EVALUATOR_COMMIT}
    if scorer is not None:
        identity["scorer"] = scorer
    return sha256_bytes(canonical_json(identity))


def _row_bytes(question: str, answer: str) -> bytes:
    row = {"answer": answer, "question": question}
    return (json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def extract_answer(text: str) -> str:
    """The text after `####`, stripped, exactly as nemo_rl/data/datasets/response_datasets/gsm8k.py:22-25.
    Thousands separators are kept (G-19)."""
    if "####" not in text:
        raise ValueError(f"GSM8K answer has no '####': {text[-80:]!r}")
    return text.split("####")[1].strip()


# ---------------------------------------------------------------------------
# H-1 export-gsm8k
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in text_lines(Path(path).read_text(encoding="utf-8")) if line.strip()]


def _download(revision: str | None) -> tuple[list[dict], list[dict], str]:
    """Hugging Face `openai/gsm8k` config `main` at a pinned revision. Not tested (no network in tests).

    The dataset's parquet files are read with pyarrow at `revision`, a full 40-hex dataset commit
    (`DATASET_REVISION` when none is given; G-17). At a revision listed in `DATASET_PARQUET_SHA256` each file's
    SHA-256 must match (EXPORT-SHA256); another revision has no expected value, and the export's manifests
    (`train_manifest_hash`, `prompt_set_hash`) identify what was written.
    """
    import io
    import urllib.request

    import pyarrow.parquet as pq

    if revision is None:
        revision = DATASET_REVISION
    if not REVISION_RE.fullmatch(revision):
        raise Refused([Problem("EXPORT-REVISION", "--revision",
                               f"{revision!r} is not a full 40-hex dataset commit; a download must be pinned",
                               f"pass --revision <40-hex commit>, or omit it for {DATASET_REVISION}")])
    splits = {}
    for split in ("train", "test"):
        url = (f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{DATASET_CONFIG}/"
               f"{split}-00000-of-00001.parquet")
        with urllib.request.urlopen(url, timeout=300) as r:
            data = r.read()
        expected = DATASET_PARQUET_SHA256.get(revision, {}).get(split)
        if expected is not None and sha256_bytes(data) != "sha256:" + expected:
            raise Refused([Problem("EXPORT-SHA256", f"{split}-00000-of-00001.parquet",
                                   f"downloaded bytes hash to {sha256_bytes(data)}, expected sha256:{expected}",
                                   "retry the download; if it persists, the file on the Hub differs from the pin")])
        table = pq.read_table(io.BytesIO(data))
        splits[split] = table.select(["question", "answer"]).to_pylist()
    return splits["train"], splits["test"], revision


def export_gsm8k(out: Path, *, source: Path | None = None, revision: str | None = None,
                 rows_per_shard: int = DEFAULT_ROWS_PER_SHARD) -> Path:
    """Write the export of tests/README.md section 11.1. Refuses an existing `out`."""
    out = Path(out).absolute()
    if out.exists():
        raise Refused([Problem("EXPORT-OUT", "--out", f"{out} already exists; an export is never overwritten",
                               "pass a new --out directory")])
    if source is not None and not revision:
        raise Refused([Problem("EXPORT-REVISION", "--revision",
                               "a local copy carries no dataset revision; G-17 requires the export to record it",
                               "pass --revision <Hugging Face dataset commit hash of the copy>")])
    if rows_per_shard < 1:
        raise Refused([Problem("EXPORT-OUT", "--rows-per-shard", "rows per shard must be >= 1", "pass a positive count")])
    if source is not None:
        src = Path(source)
        try:
            train_src, test_src = _read_jsonl(src / "train.jsonl"), _read_jsonl(src / "test.jsonl")
        except (OSError, json.JSONDecodeError) as exc:
            raise Refused([Problem("EXPORT-SOURCE", "--source", f"cannot read {src}/train.jsonl and test.jsonl: {exc}",
                                   "point --source at a copy of openai/gsm8k main with train.jsonl and test.jsonl")])
    else:
        train_src, test_src, revision = _download(revision)
    try:
        train = [_row_bytes(r["question"], extract_answer(r["answer"])) for r in train_src]
        test = [_row_bytes(r["question"], extract_answer(r["answer"])) for r in test_src]
    except (KeyError, TypeError, ValueError) as exc:
        raise Refused([Problem("EXPORT-SOURCE", "--source", f"a row is not {{question, answer '#### N'}}: {exc}",
                               "use rows shaped like openai/gsm8k config main")])

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.tmp-", dir=out.parent))
    try:
        (tmp / "train").mkdir()
        shards = []
        for i in range(0, len(train), rows_per_shard):
            name = f"shard-{i // rows_per_shard:05d}.jsonl"
            (tmp / "train" / name).write_bytes(b"".join(train[i:i + rows_per_shard]))
            shards.append(name)
        # The concatenation NeMo RL reads lives outside train/ (L-5, G-14).
        (tmp / "train.jsonl").write_bytes(b"".join((tmp / "train" / n).read_bytes() for n in sorted(shards)))
        (tmp / "heldout").mkdir()
        (tmp / "heldout" / "prompts.jsonl").write_bytes(b"".join(test))
        (tmp / "nel_gsm8k_prompt.txt").write_bytes(NEL_GSM8K_PROMPT.encode("utf-8"))
        docs = {}
        for name, type_, sub in (("training_shards", "training_dataset", "train"),
                                 ("prompt_set", "prompt_dataset", "heldout")):
            doc = manifest_document(name, type_, tmp / sub, TAGS)
            doc["root"] = str(out / sub)          # recorded, not hashed (tests/README.md section 5.1)
            write_canonical(tmp / "manifests" / f"{name}.json", doc)
            docs[name] = doc
        write_canonical(tmp / "export.json", {
            "schema": "gatedtrain.gsm8k_export", "schema_version": "1.0", "tags": list(TAGS),
            "dataset": DATASET, "config": DATASET_CONFIG, "revision": revision,
            "rows": {"train": len(train), "test": len(test)}, "rows_per_shard": rows_per_shard,
            "train_manifest_hash": docs["training_shards"]["manifest_hash"],
            "prompt_set_hash": docs["prompt_set"]["manifest_hash"],
            "concat_sha256": file_sha256(tmp / "train.jsonl"),
            "prompt_file_sha256": file_sha256(tmp / "nel_gsm8k_prompt.txt"),
        })
        os.rename(tmp, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return out


# ---------------------------------------------------------------------------
# H-2 verify-export
# ---------------------------------------------------------------------------

def verify_export(export_dir: Path) -> dict[str, Any]:
    """Re-check an export (tests/README.md section 11.2). Returns export.json; raises Refused."""
    d = Path(export_dir)
    try:
        meta = json.loads((d / "export.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Refused([Problem("EXPORT-LAYOUT", "export.json", f"cannot read {d}/export.json: {exc}",
                               "point at a directory written by `gatedtrain export-gsm8k`")])
    problems: list[Problem] = []
    if (d / "train" / "train.jsonl").exists():
        problems.append(Problem("EXPORT-LAYOUT", "train/train.jsonl",
                                "a train.jsonl sits inside the shard directory, where it would be read as a shard",
                                "keep the concatenation at <export>/train.jsonl, outside train/ (G-14)"))
    shards = sorted((d / "train").glob("*.jsonl"), key=lambda p: p.name.encode("utf-8"))
    concat = sha256_bytes(b"".join(p.read_bytes() for p in shards))
    checks = [
        ("train/", "concatenation of the shards in file-name order", concat, meta.get("concat_sha256")),
        ("train.jsonl", "train.jsonl", file_sha256(d / "train.jsonl") if (d / "train.jsonl").is_file() else None,
         meta.get("concat_sha256")),
        ("train/", "train/ manifest hash", manifest_hash(d / "train"), meta.get("train_manifest_hash")),
        ("heldout/", "held-out prompt-set manifest hash",
         manifest_hash(d / "heldout") if (d / "heldout").is_dir() else None, meta.get("prompt_set_hash")),
        ("nel_gsm8k_prompt.txt", "prompt file",
         file_sha256(d / "nel_gsm8k_prompt.txt") if (d / "nel_gsm8k_prompt.txt").is_file() else None,
         meta.get("prompt_file_sha256")),
    ]
    for field, what, actual, recorded in checks:
        if actual != recorded:
            problems.append(Problem("EXPORT-HASH", field, f"{what} is {actual} but export.json records {recorded}",
                                    "re-export with `gatedtrain export-gsm8k` into a new directory; never edit an export"))
    if problems:
        raise Refused(problems)
    return meta


# ---------------------------------------------------------------------------
# Registration helper shared by H-3 and H-4
# ---------------------------------------------------------------------------

def load_registration(path: Path) -> dict:
    """Parse a registration and check its contents (05 section 3.7; G-15, G-18). Raises Refused.
    Returns the registration in revision 11 names (`registration.normalize`, R11A-2)."""
    from .registration import check_contents, normalize

    try:
        reg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise Refused([Problem("REG-MISSING", "--registration", f"cannot read {path}: {exc}",
                               "pass the registration file")])
    if not isinstance(reg, dict):
        raise Refused([Problem("REG-FIELD", "registration", "the registration is not a mapping",
                               "write the registration as a YAML mapping")])
    # The Tier B commands gate on the Tier B evaluator (R11A-23): it is the deciding entry wherever it is listed,
    # and it must be listed (R12A-27: no index-0 fallback).
    evaluators = reg.get("evaluators")
    problems = check_contents(reg, evaluator_id=EVALUATOR_ID)
    if isinstance(evaluators, list) and not any(isinstance(e, dict) and e.get("id") == EVALUATOR_ID
                                                for e in evaluators):
        problems.append(Problem("REG-FIELD", "registration.evaluators",
                                f"the Tier B commands gate on {EVALUATOR_ID}, which the registration does not list "
                                "(R11A-23, R12A-27)",
                                f"list `{{id: {EVALUATOR_ID}, manifest_hash: <resolved hash>}}` in `evaluators`"))
    if isinstance(reg.get("clause"), dict) and reg["clause"].get("constraint_set") is not None:
        # Constraint clauses (tests/README.md section 11.17) are library-level only so far.
        problems.append(Problem("REG-CONSTRAINT", "registration.clause.constraint_set",
                                "constraint clauses are not wired into the Tier B commands "
                                "(tests/README.md section 11.17 scope)",
                                "gate a constraint clause with gatedtrain.constraints.evaluate_constraint"))
    if problems:
        raise Refused(problems)
    return normalize(reg)


def registered_scorer(reg: dict, evaluator_id: str) -> str | None:
    """The deciding evaluator's `scorer` (interim rule G-21; R11A-23: the entry whose id is the gate's
    evaluator), already checked against the allowlist; None = default or no such entry."""
    from .registration import deciding_evaluator

    entry = deciding_evaluator(reg, evaluator_id)
    return entry.get("scorer") if entry else None


def _declared_scorers(cfg: Any) -> list[str] | None:
    """Scorer names a NeMo Evaluator config declares under its gsm8k benchmark: `scoring.metrics` entries
    `{type: scorer, name}` (BenchmarkConfig.scoring, config/benchmarks.py:63; ScorerMetric,
    config/scoring.py:24-30 at ea35ba3). None when the config has no single gsm8k benchmark."""
    if not isinstance(cfg, dict):
        return None
    benches = [b for b in cfg.get("benchmarks") or [] if isinstance(b, dict)]
    gsm = [b for b in benches if b.get("name") == "gsm8k"] or benches
    if len(gsm) != 1:
        return None
    metrics = (gsm[0].get("scoring") or {}).get("metrics") if isinstance(gsm[0].get("scoring"), dict) else None
    return [m["name"] for m in metrics or [] if isinstance(m, dict) and m.get("type") == "scorer"
            and isinstance(m.get("name"), str)]


# ---------------------------------------------------------------------------
# H-3 merge-eval
# ---------------------------------------------------------------------------

def _run_sampler(run: Path) -> dict[str, Any]:
    """The sampler a run actually used, from `<run>/full_config.yaml`: GenerationConfig temperature,
    top_p, max_tokens (config/services.py:38-41), the solver's fields overriding the service's
    (services.py:28-46), the service `model` (services.py:240-259) and
    `proxy.extra_body.chat_template_kwargs.enable_thinking` (ProxyConfig.extra_body, services.py:82-105)."""
    cfg = yaml.safe_load((run / "full_config.yaml").read_text(encoding="utf-8"))
    benches = [b for b in cfg.get("benchmarks") or [] if isinstance(b, dict)]
    bench = ([b for b in benches if b.get("name") == "gsm8k"] or benches or [{}])[0]
    solver = bench.get("solver") or {}
    service = ((cfg.get("services") or {}).get(solver.get("service")) or {})
    gen = dict(service.get("generation") or {})
    gen.update({k: v for k, v in (solver.get("generation") or {}).items() if v is not None})
    kwargs = (((service.get("proxy") or {}).get("extra_body") or {}).get("chat_template_kwargs") or {})
    return {"temperature": gen.get("temperature"), "top_p": gen.get("top_p"), "max_tokens": gen.get("max_tokens"),
            "model": service.get("model"), "enable_thinking": kwargs.get("enable_thinking")}


def _run_seed_and_repeats(run: Path) -> tuple[int | None, int | None]:
    """Seed and repeats from NeMo Evaluator's composed-config snapshot `<run>/full_config.yaml`
    (config/snapshot.py:39, :72-81; written by executors/local_executor.py:69). The seed is the solver's
    `generation.seed` if set, otherwise the solver service's (config/solvers.py:26-36; GenerationConfig
    merge, config/services.py:28-46; orchestrator.py:881-889). `repeats` defaults to 1
    (config/benchmarks.py:41)."""
    cfg = yaml.safe_load((run / "full_config.yaml").read_text(encoding="utf-8"))
    benches = [b for b in cfg.get("benchmarks") or [] if isinstance(b, dict)]
    gsm = [b for b in benches if b.get("name") == "gsm8k"] or benches
    if len(gsm) != 1:
        return None, None
    bench = gsm[0]
    repeats = bench.get("repeats", 1)
    solver = bench.get("solver") or {}
    seed = (solver.get("generation") or {}).get("seed")
    if seed is None:
        service = (cfg.get("services") or {}).get(solver.get("service"), {}) or {}
        seed = (service.get("generation") or {}).get("seed")
    return seed, repeats


def _read_run_strict(results: Path, limit: int = 5) -> tuple[dict[tuple[int, int], dict], list[str]]:
    """One run's records by (problem_idx, repeat), read with the strict per-line rule of `runner.read_arm_records`
    (`runner.results_lines`; R11A-28 extended to merge-eval). Unlike `gates.load_paired_records`, a line that does
    not parse, is not a JSON object or has no `problem_idx`, a `problem_idx` or `repeat` that is not an integer
    (design choice: nothing is coerced) and a second record for one (problem_idx, repeat) are reported, never
    skipped or overwritten. Returns (records, messages); at most `limit` lines are named, then a count."""
    from .runner import results_lines

    if not results.is_file():
        return {}, []
    recs: dict[tuple[int, int], dict] = {}
    first_line: dict[tuple[int, int], int] = {}
    bad: list[str] = []
    for n, rec in results_lines(results):
        if rec is None:
            bad.append(f"line {n} does not parse, is not a JSON object or has no `problem_idx` (R11A-28)")
            continue
        pid, rep = rec["problem_idx"], rec.get("repeat", 0)
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in (pid, rep)):
            bad.append(f"line {n} has problem_idx {pid!r} and repeat {rep!r}, not integers (R11A-28)")
            continue
        if (pid, rep) in recs:
            bad.append(f"line {n} is a duplicate of line {first_line[(pid, rep)]}: (problem_idx, repeat) = "
                       f"({pid}, {rep}) (R11A-28)")
            continue
        recs[(pid, rep)] = rec
        first_line[(pid, rep)] = n
    if len(bad) > limit:
        bad = bad[:limit] + [f"{len(bad) - limit} more unreadable or duplicate lines"]
    return recs, bad


def merge_eval(registration: Path, out: Path, runs: list[Path]) -> Path:
    """Merge K single-seed runs into `out/results.jsonl` (tests/README.md section 11.3; G-16).

    Each run's `full_config.yaml` is copied to `out/configs/rep<r>.yaml` and recorded in merge.json
    (decision D-1); with a registered scorer, every run must declare it (G-21, decision D-2). Each run is read
    with the strict per-line rule of the gate path (`_read_run_strict`, R11A-28): a bad or duplicate line refuses
    the merge, so the merged file never hides one."""
    reg = load_registration(registration)
    seeds = list(reg["clause"]["eval_seeds"])
    k = int(reg["clause"]["samples_per_item"])
    scorer = registered_scorer(reg, EVALUATOR_ID)
    out = Path(out)
    if out.exists():
        raise Refused([Problem("MERGE-EVAL", "--out", f"{out} already exists", "pass a new --out directory")])
    if len(runs) != k:
        raise Refused([Problem("MERGE-EVAL", "runs", f"{len(runs)} runs given for K = {k} registered seeds",
                               f"pass exactly {k} runs, in the order of eval_seeds {seeds}")])
    problems: list[Problem] = []
    per_run: list[dict] = []
    for r, run in enumerate(map(Path, runs)):
        fld = f"runs[{r}]"
        results = run / "gsm8k" / "results.jsonl"          # orchestrator.py:1046-1047, :1081
        recs, bad = _read_run_strict(results)
        if bad:
            problems.extend(Problem("MERGE-EVAL", fld, f"{results}: {msg}",
                                    "repair or re-run the evaluation so that every line is one NeMo Evaluator "
                                    "record with an integer `problem_idx` and one record per (problem_idx, repeat)")
                            for msg in bad)
            per_run.append({})
            continue
        if not results.is_file() or not recs:
            problems.append(Problem("MERGE-EVAL", fld, f"{results} is missing or holds no records",
                                    "pass the NeMo Evaluator output directory of a finished run"))
            per_run.append({})
            continue
        try:
            seed, repeats = _run_seed_and_repeats(run)
        except (OSError, yaml.YAMLError, AttributeError) as exc:
            problems.append(Problem("MERGE-EVAL", fld, f"cannot read {run}/full_config.yaml: {exc}",
                                    "pass the run's own output directory, with its config snapshot"))
            per_run.append({})
            continue
        if repeats != 1:
            problems.append(Problem("MERGE-EVAL", fld, f"the run has repeats = {repeats}, not 1 (G-16)",
                                    "run one evaluation per registered seed with `repeats: 1`"))
        if seed != seeds[r]:
            problems.append(Problem("MERGE-EVAL", fld, f"the run's seed is {seed}, but repeat {r} is registered "
                                    f"with eval_seeds[{r}] = {seeds[r]}",
                                    "pass the runs in registered seed order, each run with its registered seed"))
        try:
            used = _run_sampler(run)
        except (OSError, yaml.YAMLError, AttributeError) as exc:
            used = None
            problems.append(Problem("MERGE-EVAL", fld, f"cannot read the run's sampler: {exc}",
                                    "pass the run's own output directory, with its config snapshot"))
        if used is not None:
            sampler = reg["sampler"]
            want = {"temperature": sampler.get("temperature"), "top_p": sampler.get("top_p"),
                    "max_tokens": sampler.get("max_new_tokens"), "enable_thinking": False}
            for key, value in want.items():
                if key == "top_p" and value is None:
                    continue
                if used[key] != value:
                    problems.append(Problem("MERGE-EVAL", fld, f"the run used {key} = {used[key]!r}, but the "
                                            f"registration fixes {value!r} (05 section 3.7; G-15)",
                                            "regenerate the configs with `gatedtrain nel-config` and re-run"))
            if r > 0 and per_run and per_run[0].get("sampler") and used["model"] != per_run[0]["sampler"]["model"]:
                problems.append(Problem("MERGE-EVAL", fld, f"the run evaluated model {used['model']!r}, but runs[0] "
                                        f"evaluated {per_run[0]['sampler']['model']!r}",
                                        "merge the K runs of one model only"))
        if any(rep != 0 for _, rep in recs):
            problems.append(Problem("MERGE-EVAL", fld, "the run has records with repeat other than 0",
                                    "run with `repeats: 1`"))
        # Decision D-1: the run's config snapshot is copied into the arm, so collect can check it later.
        config_bytes = (run / "full_config.yaml").read_bytes()
        if scorer is not None:
            # Decision D-2: the run must have used the registered scorer (G-21); the snapshot is the raw
            # config as written (config/snapshot.py:72-81 at ea35ba3).
            declared = _declared_scorers(yaml.safe_load(config_bytes))
            if scorer not in (declared or []):
                problems.append(Problem("MERGE-EVAL", fld, f"the run's full_config.yaml declares scorers "
                                        f"{declared!r} under the gsm8k benchmark, not the registered "
                                        f"`{{type: scorer, name: {scorer}}}` (G-21)",
                                        "regenerate the configs with `gatedtrain nel-config` and re-run"))
        per_run.append({"run": run, "recs": recs, "sha": file_sha256(results), "seed": seed, "sampler": used,
                        "config": config_bytes})
    if not problems:
        base_set = {pid for pid, _ in per_run[0]["recs"]}
        for r in range(1, k):
            if {pid for pid, _ in per_run[r]["recs"]} != base_set:
                problems.append(Problem("MERGE-EVAL", f"runs[{r}]", "the problem set differs from runs[0]",
                                        "evaluate the same benchmark, dataset revision and problems in every run"))
    if problems:
        raise Refused(problems)

    merged = []
    for r, info in enumerate(per_run):
        for (pid, _), rec in info["recs"].items():
            new = dict(rec)
            new["repeat"] = r
            new["metadata"] = {**(rec.get("metadata") or {}), "eval_seed": seeds[r]}
            merged.append(new)
    merged.sort(key=lambda x: (int(x["problem_idx"]), x["repeat"]))
    out.mkdir(parents=True)
    with (out / "results.jsonl").open("w", encoding="utf-8") as f:
        for rec in merged:
            f.write(json.dumps(rec) + "\n")
    (out / "configs").mkdir()
    for r, info in enumerate(per_run):
        (out / "configs" / f"rep{r}.yaml").write_bytes(info["config"])
    write_canonical(out / "merge.json", {
        "schema": "gatedtrain.merge_eval", "schema_version": "1.0", "tags": list(TAGS),
        "sources": [{"repeat": r, "seed": seeds[r], "run_dir": str(Path(info["run"]).absolute()),
                     "results_sha256": info["sha"], "config": f"configs/rep{r}.yaml",
                     "config_sha256": sha256_bytes(info["config"])} for r, info in enumerate(per_run)],
        "notes": [SEED_NOTE],
    })
    return out


# ---------------------------------------------------------------------------
# H-4 nel-config
# ---------------------------------------------------------------------------

def nel_configs(registration: Path, *, model_url: str, model_id: str, eval_root: str, out: Path) -> list[Path]:
    """K NeMo Evaluator config files, one per registered seed (tests/README.md section 11.4).

    Keys at ea35ba3: EvalConfig services/benchmarks/output (config/eval_config.py:59-66);
    ExternalApiService type/url/protocol/model/generation/proxy (config/services.py:240-259), whose
    url is the full endpoint (:240-245); GenerationConfig temperature/top_p/max_tokens/seed
    (services.py:38-41); ProxyConfig.extra_body (services.py:82-105) carries
    chat_template_kwargs.enable_thinking (G-15); BenchmarkConfig name/solver/repeats
    (config/benchmarks.py:39-41); SimpleSolver type/service (config/solvers.py:32-33); OutputConfig
    dir/timestamped (config/output.py:41-49). No dataset revision key exists (the hf:// loader,
    environments/custom.py:126-148, takes none), so the export and the report record it (G-17).
    """
    reg = load_registration(registration)
    out = Path(out)
    if out.exists():
        raise Refused([Problem("NEL-CONFIG", "--out", f"{out} already exists", "pass a new --out directory")])
    sampler = reg["sampler"]
    scorer = registered_scorer(reg, EVALUATOR_ID)
    if sampler.get("enable_thinking") is not False:
        raise Refused([Problem("REG-FIELD", "registration.sampler.enable_thinking",
                               "Tier B needs `enable_thinking: false` in the registration sampler (G-15)",
                               "set `enable_thinking: false`")])
    url = model_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    paths = []
    out.mkdir(parents=True)
    for r, seed in enumerate(reg["clause"]["eval_seeds"]):
        generation = {"temperature": float(sampler["temperature"])}
        if "top_p" in sampler:
            generation["top_p"] = float(sampler["top_p"])
        generation["max_tokens"] = int(sampler["max_new_tokens"])
        generation["seed"] = int(seed)
        cfg = {
            "services": {"model": {
                "type": "api", "url": url, "protocol": "chat_completions", "model": model_id,
                "generation": generation,
                "proxy": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
            }},
            "benchmarks": [{"name": "gsm8k", "repeats": 1, "solver": {"type": "simple", "service": "model"}}],
            "output": {"dir": f"{eval_root.rstrip('/')}/rep{r}", "timestamped": False},
        }
        if scorer is not None:
            # G-21: the registered scorer as an extra scorer-type metric (BenchmarkConfig.scoring,
            # config/benchmarks.py:63; ScorerMetric, config/scoring.py:24-30); NeMo Evaluator then writes
            # scorer_rewards["scorer:<name>"] next to the default `reward` (engine/eval_loop.py:785-794).
            cfg["benchmarks"][0]["scoring"] = {"metrics": [{"type": "scorer", "name": scorer}]}
        header = (f"# NeMo Evaluator {NEMO_EVALUATOR_COMMIT} config written by gatedtrain nel-config "
                  f"(prototype, unpromotable).\n# Repeat {r} of {len(reg['clause']['eval_seeds'])}: seed "
                  f"eval_seeds[{r}] = {seed} (G-16). Run: nel eval run <this file>\n")
        p = out / f"nel-gsm8k-rep{r}.yaml"
        p.write_text(header + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# H-6 tierb-collect
# ---------------------------------------------------------------------------

def _load_trainer_config(path: Path, field_prefix: str) -> dict:
    """Tier B trainer config {backend: nemo_rl, recipe, profile} (tests/README.md section 11.5)."""
    from .backends.nemo_rl import PROFILES, RECIPE

    try:
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise Refused([Problem("TRAINER-CONFIG", field_prefix, f"cannot read {path}: {exc}", "fix the trainer config")])
    problems = []
    if not isinstance(cfg, dict):
        cfg = {}
        problems.append(Problem("TRAINER-CONFIG", field_prefix, "not a mapping", "write {backend, recipe, profile}"))
    for k in sorted(set(cfg) - {"backend", "recipe", "profile"}):
        problems.append(Problem("TRAINER-CONFIG", f"{field_prefix}.{k}", "unknown key", "use backend, recipe, profile"))
    if cfg.get("backend") != "nemo_rl":
        problems.append(Problem("TRAINER-CONFIG", f"{field_prefix}.backend", "Tier B needs `backend: nemo_rl`",
                                "set `backend: nemo_rl`"))
    if cfg.get("recipe") != RECIPE:
        problems.append(Problem("TRAINER-CONFIG", f"{field_prefix}.recipe", f"the skeleton supports {RECIPE} only",
                                f"set `recipe: {RECIPE}`"))
    if cfg.get("profile") not in PROFILES:
        problems.append(Problem("TRAINER-CONFIG", f"{field_prefix}.profile", f"unknown profile {cfg.get('profile')!r}",
                                f"use one of {sorted(PROFILES)}"))
    if problems:
        raise Refused(problems)
    return cfg


def _check_profile(final_ckpt: Path, profile: str) -> list[Problem]:
    """LOW-3: the run's own config (step_<last>/config.yaml, utils/checkpoint.py:414-419) must match the
    trainer.yaml profile (backends/nemo_rl.PROFILES)."""
    from .backends.nemo_rl import PROFILES

    try:
        cfg = yaml.safe_load((final_ckpt / "config.yaml").read_text(encoding="utf-8")) or {}
        grpo = cfg.get("grpo") or {}
        if not isinstance(grpo, dict):
            raise TypeError(f"grpo is a {type(grpo).__name__}, not a mapping")
    except (OSError, yaml.YAMLError, AttributeError, TypeError) as exc:
        # Final re-review LOW-1: a non-mapping config.yaml is a refusal, handled like _check_base_model.
        return [Problem("TIERB-PROFILE", "--stage-out",
                        f"{final_ckpt}/config.yaml is not the NeMo RL run config mapping "
                        f"(utils/checkpoint.py:414-419): {exc}",
                        "collect a complete, unedited run")]
    want = PROFILES[profile]
    diffs = [f"{k} = {grpo.get(k)!r} (profile {profile}: {want[k]})"
             for k in ("max_num_steps", "num_prompts_per_step", "num_generations_per_prompt")
             if grpo.get(k) != want[k]]
    if diffs:
        return [Problem("TIERB-PROFILE", "--stage-out", "the run's grpo config differs from the trainer.yaml "
                        "profile: " + "; ".join(diffs),
                        "set `profile` in the spec's trainer config to the profile the run used, or collect that run")]
    return []


def _check_data_config(final_ckpt: Path, export_name: str) -> list[Problem]:
    """R10A-7 (TIERB-DATA): the run's saved `data` section (step_<last>/config.yaml, utils/checkpoint.py:414-419)
    must keep the ledger's loader mapping valid: `shuffle: false`; one training dataset, a mapping or the
    one-element list setup_response_data saves (nemo_rl/data/utils.py:215-216); a `data_path` named
    train.jsonl in a directory named like the spec's export (container paths differ from host paths, F-17);
    `split_validation_size` 0, its default (response_dataset.py:50, :76 splits and reorders rows)."""
    from pathlib import PurePosixPath

    try:
        cfg = yaml.safe_load((final_ckpt / "config.yaml").read_text(encoding="utf-8")) or {}
        data = cfg.get("data")
    except (OSError, yaml.YAMLError, AttributeError) as exc:
        return [Problem("TIERB-DATA", "--stage-out", f"cannot read the data section of {final_ckpt}/config.yaml: "
                        f"{exc}", "collect a complete, unedited run")]
    bad: list[str] = []
    if not isinstance(data, dict):
        bad.append(f"data is {type(data).__name__}, not a mapping")
        data = {}
    if data.get("shuffle", None) is not False:
        bad.append(f"data.shuffle = {data.get('shuffle', '<absent>')!r}, not false (the ledger maps loader "
                   "positions to shards in file order)")
    train = data.get("train")
    if isinstance(train, dict):
        train = [train]
    if not (isinstance(train, list) and len(train) == 1 and isinstance(train[0], dict)):
        n = len(train) if isinstance(train, list) else type(train).__name__
        bad.append(f"data.train holds {n} dataset(s), not one")
    else:
        entry = train[0]
        default = data.get("default") if isinstance(data.get("default"), dict) else {}
        path = entry.get("data_path", default.get("data_path"))
        pp = PurePosixPath(path) if isinstance(path, str) else None
        if pp is None or pp.name != "train.jsonl" or pp.parent.name != export_name:
            bad.append(f"data.train data_path = {path!r} is not <dir>/{export_name}/train.jsonl, the spec's export "
                       "concatenation")
        split = entry.get("split_validation_size", default.get("split_validation_size", 0))
        if not (isinstance(split, (int, float)) and not isinstance(split, bool) and split == 0):
            bad.append(f"data.train split_validation_size = {split!r}, not 0 (a validation split drops and "
                       "reorders training rows)")
    if not bad:
        return []
    return [Problem("TIERB-DATA", "--stage-out",
                    f"the run's data config in {final_ckpt}/config.yaml breaks the ledger mapping: " + "; ".join(bad),
                    "train with build_command's data overrides (data.shuffle=false, one dataset at "
                    "<export>/train.jsonl, split_validation_size=0), or collect that run")]


def _read_problem(root: Path, field: str, exc: OSError) -> Problem:
    """TIERB-READ: a file or directory under a root that collect must hash cannot be read."""
    where = getattr(exc, "filename", None) or root
    return Problem("TIERB-READ", field,
                   f"cannot read {where} ({exc.__class__.__name__}: {exc.strerror or exc}); the whole directory "
                   f"{root} must be readable to be hashed",
                   "make the root-owned outputs readable for the host user (docs/tier-b-runbook.md step 6 "
                   "and findings F-25, for example an approved in-container `chmod -R a+rX`), then collect again")


def _check_base_model(final_ckpt: Path, base: Path, field: str) -> list[Problem]:
    """L-1 (TIERB-BASE): training must have started from the base-model directory the gate hashes.

    `policy.model_name` in `step_<last>/config.yaml` (written from the run config,
    utils/checkpoint.py:414-419) must be a local absolute path whose final component is the spec's
    `base` directory name. `build_command(..., base_model=...)` sets it (F-17); a Hub id such as
    `Qwen/Qwen3-0.6B` means NeMo RL loaded the model by name (setup.py:374-375, :941-942), not the
    hashed directory. The container path differs from the host path, so only the name is compared.
    """
    from pathlib import PurePosixPath

    try:
        cfg = yaml.safe_load((final_ckpt / "config.yaml").read_text(encoding="utf-8")) or {}
        model_name = (cfg.get("policy") or {}).get("model_name")
    except (OSError, yaml.YAMLError, AttributeError) as exc:
        return [Problem("TIERB-BASE", field, f"cannot read policy.model_name from {final_ckpt}/config.yaml: {exc}",
                        "collect a complete run")]
    ok = (isinstance(model_name, str) and PurePosixPath(model_name).is_absolute()
          and PurePosixPath(model_name).name == base.name)
    if ok:
        return []
    return [Problem("TIERB-BASE", field,
                    f"the run trained from policy.model_name = {model_name!r}, not from the spec's base-model "
                    f"directory {base.name!r} that the gate hashes (F-17)",
                    f"re-run training with build_command(..., base_model=<container path ending in /{base.name}>), "
                    "which sets policy.model_name; a Hub id or another directory cannot be gated against this baseline")]


def _check_prompts(prompts: Path, b_recs: dict, c_recs: dict) -> list[Problem]:
    """HIGH-2: the evaluated set is the registered held-out set.

    (a) every evaluated problem_idx is in range(len(heldout)) (R11A-17: a held-out problem absent from an
    arm, or from both, is a missing item of the gate, 05 section 3.4, not a refusal); (b) each expected_answer is the heldout
    answer with thousands commas removed, as NeMo Evaluator's `_prepare` does
    (benchmarks/gsm8k.py:29-31 at ea35ba3); (c) the heldout question appears in the record's prompt
    (the prompt template, benchmarks/gsm8k.py:22-26). A Hub revision change between the export and
    the evaluation shows up here (G-17)."""
    heldout = _read_jsonl(prompts / "prompts.jsonl")
    pids = {pid for pid, _ in b_recs} | {pid for pid, _ in c_recs}
    problems: list[Problem] = []
    want = set(range(len(heldout)))
    extra = sorted(pids - want)
    if extra:
        problems.append(Problem("TIERB-PROMPTS", "--eval-baseline",
                                f"problems outside the {len(heldout)} registered held-out problems were evaluated: "
                                f"problem_idx {extra[:5]}{'...' if len(extra) > 5 else ''}",
                                "evaluate the registered held-out set only (export.json, heldout/prompts.jsonl)"))
    bad_answer, bad_question = [], []
    for arm in (b_recs, c_recs):
        for (pid, _), rec in arm.items():
            if not 0 <= pid < len(heldout):
                continue
            row = heldout[pid]
            if str(rec.get("expected_answer")) != row["answer"].replace(",", ""):
                bad_answer.append(pid)
            if row["question"] not in str(rec.get("prompt", "")):
                bad_question.append(pid)
    if bad_answer:
        problems.append(Problem("TIERB-PROMPTS", "--eval-baseline",
                                f"expected_answer differs from the held-out answer (commas removed) for problem_idx "
                                f"{sorted(set(bad_answer))[:5]}: the evaluated dataset is not the exported one",
                                "evaluate the dataset revision recorded in export.json (G-17), or re-export"))
    if bad_question:
        problems.append(Problem("TIERB-PROMPTS", "--eval-baseline",
                                f"the held-out question is not in the record's prompt for problem_idx "
                                f"{sorted(set(bad_question))[:5]}",
                                "evaluate the dataset revision recorded in export.json (G-17), or re-export"))
    return problems


def _is_score(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _in_unit_range(x: Any) -> bool:
    """Decision D-11: a per-record score the gate reads is a finite number in [0, 1]. Python's json reads
    the NaN and Infinity literals, so a results.jsonl holding them loads without error."""
    return _is_score(x) and math.isfinite(x) and 0 <= x <= 1


def _check_arm_scorer(arm: Path, flag: str, scorer: str, k: int) -> list[Problem]:
    """TIERB-SCORER (G-21, decisions D-1 and D-2): the arm's copied run configs `configs/rep<r>.yaml`, one per
    registered seed, must all declare `{type: scorer, name: <scorer>}` under the gsm8k benchmark. They are
    read from the arm, not from merge.json's absolute run_dir paths. Defence in depth behind merge-eval."""
    bad = []
    for r in range(k):
        cfg = arm / "configs" / f"rep{r}.yaml"
        try:
            declared = _declared_scorers(yaml.safe_load(cfg.read_text(encoding="utf-8")))
        except (OSError, ValueError, yaml.YAMLError) as exc:   # ValueError: UnicodeDecodeError (D-12)
            bad.append(f"configs/rep{r}.yaml cannot be read ({exc.__class__.__name__})")
            continue
        if scorer not in (declared or []):
            bad.append(f"configs/rep{r}.yaml declares {declared!r}")
    if not bad:
        return []
    return [Problem("TIERB-SCORER", flag,
                    f"the registration gates on scorer {scorer!r} (G-21), but the arm's run configs do not show "
                    f"every run used it: {'; '.join(bad)}",
                    "evaluate with the configs from `gatedtrain nel-config` under this registration and merge the "
                    "runs with `gatedtrain merge-eval`, which copies their configs into the arm")]


def collect(spec_path: Path, *, stage_out: Path, eval_baseline: Path, eval_candidate: Path, image_id: str,
            nemo_rl_commit: str, runs_root: Path, nemo_rl_modified: bool | str = "unknown",
            image_build_notes: str | None = None) -> tuple[Path, dict[str, Any], str]:
    """Recorded Tier B outputs to a run directory (tests/README.md section 11.7). Refuses before writing.

    R10A-11: `nemo_rl_modified` (True, False, or "unknown" when not stated) and `image_build_notes` are
    recorded in the lockfile's `backend.trainer`; an absent flag is never recorded as unmodified."""
    from .backends.nemo_rl import samples_per_position_from_checkpoint
    from .ledger import CKPT_DIR_RE, ledger_rows, orphaned_rows, write_ledger
    from .registration import item_set_field, item_set_hash
    from .runner import (_log, _new_run_dir, check_pins, copy_registration, gate_and_report, item_ids,
                         write_lineage, write_manifests, write_plan)
    from .spec import load_spec

    wf = load_spec(spec_path)
    problems: list[Problem] = []
    if wf.stage.get("uses") != TRAINER_ID:
        problems.append(Problem("TIERB-SPEC", "stages[0].uses", f"tierb-collect reads {TRAINER_ID} outputs",
                                f"use `uses: {TRAINER_ID}`; Tier A runs with `gatedtrain run`"))
    if wf.gate.get("evaluator") != EVALUATOR_ID:
        problems.append(Problem("TIERB-SPEC", "stages[1].evaluator", f"tierb-collect reads {EVALUATOR_ID} results",
                                f"use `evaluator: {EVALUATOR_ID}`"))
    if wf.registration["sampler"].get("enable_thinking") is not False:
        problems.append(Problem("REG-FIELD", "registration.sampler.enable_thinking",
                                "Tier B needs `enable_thinking: false` (G-15)", "set `enable_thinking: false`"))
    if problems:
        raise Refused(problems)
    names = wf.names
    base, data, prompts, cfg_path = (wf.path_of(r) for r in ("model", "data", "items", "config"))
    trainer_cfg = _load_trainer_config(cfg_path, f"artifacts.{names['config']}")

    # The export (G-14): the spec's training dataset is <export>/train.
    export_dir = data.parent
    if data.name != "train" or not (export_dir / "export.json").is_file():
        raise Refused([Problem("EXPORT-LAYOUT", f"artifacts.{names['data']}.uri",
                               f"{data} is not the train/ directory of a `gatedtrain export-gsm8k` export",
                               "point the training dataset at <export>/train")])
    meta = verify_export(export_dir)
    prompt_hash = manifest_hash(prompts) if (prompts / "prompts.jsonl").is_file() else None
    registered = item_set_hash(wf.registration)
    if prompt_hash != registered:                                          # R11A-17
        problems.append(Problem("REG-FIELD", item_set_field(wf.registration),
                                f"registered {registered} but {prompts} is {prompt_hash}",
                                "register the exported held-out set (export.json prompt_set_hash) as "
                                "eval_distribution.item_set_hash"))
    problems.extend(check_pins(wf))                        # LOW-2, as in Tier A preflight (04:80)
    if (base / ".cache").exists():
        # LOW-4: `--local-dir` downloads leave Hugging Face metadata in base-model/.cache, which would
        # be hashed into the baseline manifest.
        problems.append(Problem("TIERB-BASELINE", f"artifacts.{names['model']}.uri",
                                f"{base}/.cache exists (Hugging Face download metadata)",
                                f"delete {base}/.cache (rm -r base-model/.cache), then collect again"))
    known_entries: dict[Path, list] = {}
    try:
        if base.is_symlink() or not base.is_dir():
            raise ValueError(f"{base} is not a plain directory")
        known_entries[base] = manifest_entries(base)
    except OSError as exc:
        problems.append(_read_problem(base, f"artifacts.{names['model']}.uri", exc))
    except ValueError as exc:
        problems.append(Problem("TIERB-BASE", f"artifacts.{names['model']}.uri",
                                f"the baseline must be a plain directory with no symbolic links (manifest rule H2): {exc}",
                                "download the model with `--local-dir` into a plain directory, not the HF cache"))

    # The stage outputs (examples/run_grpo.py:106; utils/logger.py:1993-2025).
    stage_out = Path(stage_out)
    exps = sorted(p for p in (stage_out / "logs").glob("exp_*") if p.is_dir()) if (stage_out / "logs").is_dir() else []
    ckpt_dir = stage_out / "checkpoints"
    saved: list[tuple[int, Path]] = []
    if ckpt_dir.is_dir():
        for p in ckpt_dir.iterdir():
            m = CKPT_DIR_RE.match(p.name)
            if m and p.is_dir():
                saved.append((int(m.group(1)), p))
    saved.sort()
    spp = None
    orphaned: list[dict] = []
    if len(exps) != 1:
        problems.append(Problem("TIERB-STAGE", "--stage-out",
                                f"{len(exps)} logs/exp_* directories; exactly one attempt is supported (draft risk R9)",
                                "collect one attempt; a manual restart writes a second exp_* directory"))
    if not saved:
        problems.append(Problem("TIERB-STAGE", "--stage-out", f"no checkpoints/step_<N> under {stage_out}",
                                "pass the directory build_command set as out_dir"))
    if not problems:
        try:
            spp = samples_per_position_from_checkpoint(saved[-1][1])
            ledger_rows(exps[0], ckpt_dir, data, run_id="check", samples_per_position=spp)
            orphaned = orphaned_rows(exps[0], ckpt_dir, data, attempt_id="0", samples_per_position=spp)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            problems.append(Problem("TIERB-STAGE", "--stage-out", f"the NeMo RL outputs are inconsistent: {exc}",
                                    "collect the outputs of one complete run; do not edit them"))
        problems.extend(_check_profile(saved[-1][1], trainer_cfg["profile"]))
        problems.extend(_check_data_config(saved[-1][1], export_dir.name))
    if saved:
        problems.extend(_check_base_model(saved[-1][1], base, f"artifacts.{names['model']}.uri"))
        # Final re-review LOW-3: every saved checkpoint, the trained one included, must be fully readable
        # before anything is created. The entries are reused for the manifests and the lineage.
        for _, d in saved:
            try:
                known_entries[d] = manifest_entries(d)
            except OSError as exc:
                problems.append(_read_problem(d, "--stage-out", exc))
            except ValueError as exc:
                problems.append(Problem("TIERB-STAGE", "--stage-out", f"{d} cannot be hashed: {exc}",
                                        "collect the unedited NeMo RL outputs (manifest rule H2: no symlinks)"))

    # The merged evaluation arms (H-3 output): results.jsonl directly inside each arm.
    scorer = registered_scorer(wf.registration, wf.gate["evaluator"])
    key = f"scorer:{scorer}"
    arms = {}
    for arm, path, flag in (("baseline", Path(eval_baseline), "--eval-baseline"),
                            ("candidate", Path(eval_candidate), "--eval-candidate")):
        recs = load_paired_records(path)
        if scorer is not None:
            problems.extend(_check_arm_scorer(path, flag, scorer, len(wf.clause["eval_seeds"])))
        if not (path / "results.jsonl").is_file() or not recs:
            problems.append(Problem("TIERB-RESULTS", flag,
                                    f"{path}/results.jsonl is missing or holds no records (a parent directory or an "
                                    "unmerged rep<r> run gives no results here)",
                                    "pass the arm directory written by `gatedtrain merge-eval`"))
        elif scorer is not None:
            # G-21: the gate reads scorer_rewards["scorer:<name>"] (engine/eval_loop.py:785-794 at ea35ba3); a
            # missing value is refused, never read as 0 (as LOW-6), and so is a value left by a raising
            # scorer (scoring_details[...] = {error, reward: 0.0}, eval_loop.py:719-721; decision D-3).
            # Decision D-12: scorer_rewards and scoring_details must be JSON objects before they are read.
            malformed = sorted({k[0] for k, rec in recs.items()
                                if any(f in rec and rec[f] is not None and not isinstance(rec[f], dict)
                                       for f in ("scorer_rewards", "scoring_details"))})
            if malformed:
                problems.append(Problem("TIERB-RESULTS", flag, f"{len(malformed)} problem(s) have records whose "
                                        "scorer_rewards or scoring_details is not a JSON object, for example "
                                        f"problem_idx {malformed[:5]}",
                                        "re-run the evaluation with the configs from `gatedtrain nel-config`; "
                                        "never edit results.jsonl"))
                arms[arm] = (path, recs)
                continue
            missing = sorted({k[0] for k, rec in recs.items()
                              if not _is_score((rec.get("scorer_rewards") or {}).get(key))})
            errored = sorted({k[0] for k, rec in recs.items()
                              if isinstance((rec.get("scoring_details") or {}).get(key), dict)
                              and "error" in rec["scoring_details"][key]})
            out_of_range = sorted({k[0] for k, rec in recs.items()
                                   if _is_score((rec.get("scorer_rewards") or {}).get(key))
                                   and not _in_unit_range(rec["scorer_rewards"][key])})
            if missing:
                problems.append(Problem("TIERB-RESULTS", flag, f"{len(missing)} problem(s) have records without a "
                                        f"numeric scorer_rewards[{key!r}], for example problem_idx {missing[:5]}",
                                        "re-run the evaluation with the configs from `gatedtrain nel-config`; "
                                        "never edit results.jsonl"))
            if errored:
                problems.append(Problem("TIERB-RESULTS", flag, f"{len(errored)} problem(s) have records whose "
                                        f"scoring_details[{key!r}] holds an error (the scorer raised), for example "
                                        f"problem_idx {errored[:5]}",
                                        "fix the cause and re-run the evaluation; never edit results.jsonl"))
            if out_of_range:
                problems.append(Problem("TIERB-RESULTS", flag, f"{len(out_of_range)} problem(s) have records whose "
                                        f"scorer_rewards[{key!r}] is not a finite number in [0, 1] (decision D-11), "
                                        f"for example problem_idx {out_of_range[:5]}",
                                        "re-run the evaluation; never edit results.jsonl"))
        elif any("reward" not in rec for rec in recs.values()):
            # LOW-6: a record without `reward` is refused, never read as 0.
            missing = sorted({k[0] for k, rec in recs.items() if "reward" not in rec})
            problems.append(Problem("TIERB-RESULTS", flag, f"{len(missing)} problem(s) have records without `reward`, "
                                    f"for example problem_idx {missing[:5]}",
                                    "re-run the evaluation; never edit results.jsonl"))
        elif any(not _in_unit_range(rec["reward"]) for rec in recs.values()):
            out_of_range = sorted({k[0] for k, rec in recs.items() if not _in_unit_range(rec["reward"])})
            problems.append(Problem("TIERB-RESULTS", flag, f"{len(out_of_range)} problem(s) have records whose "
                                    f"`reward` is not a finite number in [0, 1] (decision D-11), for example "
                                    f"problem_idx {out_of_range[:5]}",
                                    "re-run the evaluation; never edit results.jsonl"))
        arms[arm] = (path, recs)
    if problems:
        raise Refused(problems)
    b_recs, c_recs = arms["baseline"][1], arms["candidate"][1]
    # R10A-3 (05 section 3.4): unpaired or missing items are no longer refused here; the gate counts them per
    # arm and is MISSING above the registered `pairing.max_unpaired`.

    # HIGH-2: the evaluated problems are the registered held-out set (benchmarks/gsm8k.py:29-31 at ea35ba3).
    prompt_problems = _check_prompts(prompts, b_recs, c_recs)
    if prompt_problems:
        raise Refused(prompt_problems)

    # Everything checked: write the run directory.
    run_dir = _new_run_dir(Path(runs_root).absolute())
    copy_registration(wf, run_dir)
    run_id = run_dir.name
    _log(f"tierb-collect {run_id}: {wf.spec['workflow']} (prototype, unpromotable)")
    the_plan = write_plan(wf, run_dir, prompts)
    for arm, (path, _) in arms.items():
        (run_dir / "eval" / arm).mkdir(parents=True)
        shutil.copyfile(path / "results.jsonl", run_dir / "eval" / arm / "results.jsonl")
        if (path / "merge.json").is_file():
            shutil.copyfile(path / "merge.json", run_dir / "eval" / arm / "merge.json")
    write_ledger(exps[0], ckpt_dir, data, run_dir / "ledger" / "consumption.parquet", run_id=run_id,
                 attempt_id="0", samples_per_position=spp)
    known_hashes = {p: entries_hash(e) for p, e in known_entries.items()}
    write_lineage(wf, run_dir, base=base, data=data, cfg_path=cfg_path, saved=saved, item_set_hash=prompt_hash,
                  known_hashes=known_hashes, orphaned=orphaned)
    hashes = write_manifests(run_dir, base=base, final_ckpt=saved[-1][1], data=data, prompts=prompts,
                             known_entries=known_entries)
    lock = {
        "schema": "gatedtrain.lockfile", "schema_version": "1.0", "tags": list(TAGS), "run_id": run_id,
        "spec_sha256": file_sha256(wf.spec_path), "registration_sha256": wf.registration_sha256,
        "config_hash": file_sha256(cfg_path),
        "nemo_rl_config_sha256": file_sha256(saved[-1][1] / "config.yaml"),
        "allow_list_checked": False,                       # no allow-list checks in the prototype
        "backend": {
            "trainer": {"name": "nemo_rl", "image_id": image_id, "nemo_rl_commit": nemo_rl_commit,
                        "uses": wf.stage["uses"], "recipe": trainer_cfg["recipe"], "profile": trainer_cfg["profile"],
                        "samples_per_position": spp,
                        "nemo_rl_modified": nemo_rl_modified, "image_build_notes": image_build_notes,   # R10A-11
                        "checkpoint_completeness": CHECKPOINT_COMPLETENESS},                            # R11A-20
            "evaluator": {"name": "nemo_evaluator", "nemo_evaluator_commit": NEMO_EVALUATOR_COMMIT,
                          "uses": wf.gate["evaluator"],
                          **({"scorer": scorer} if scorer is not None else {})},          # G-21, decision D-4
        },
        "dataset_revision": meta["revision"],
        "analysis_code_version": ANALYSIS_CODE_VERSION,
        "python_version": sys.version.split()[0],
        "artifacts": dict(sorted(hashes.items())),
    }
    lock_bytes = write_canonical(run_dir / "lockfile.json", lock)
    # Decision D-4: the registered scorer is part of the evaluator identity; with none the hash is unchanged.
    evaluator_hash = evaluator_identity_hash(scorer)
    notes = [SEED_NOTE,
             "Training used NeMo Evaluator's GSM8K prompt with Qwen3 thinking off (G-15); part of any "
             "gain may be answer-format learning."]
    if scorer is not None:
        notes.append(NUMERIC_MATCH_NOTE)
    report, text = gate_and_report(
        wf, run_dir, the_plan=the_plan, hashes=hashes, lock_bytes=lock_bytes, item_set_hash=prompt_hash,
        evaluator_hash=evaluator_hash, stubs=[], items=item_ids(prompts),
        extra={"metric_source": f"scorer_rewards[{key}]" if scorer is not None else "reward",
               "dataset_revision": meta["revision"], "notes": notes},
        scorer_key=key if scorer is not None else None)
    return run_dir, report, text
