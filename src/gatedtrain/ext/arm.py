"""Per-item arm directories (MULTIMODAL-SPEC.md MM-1 to MM-3; tests/README.md section 11.18 EXT-1, EXT-3).

An arm is `items.jsonl` (one line per (item, repeat)) plus `eval_meta.json`. This reader is strict, in
the manner of `runner.read_arm_records` (R11A-28): nothing is skipped, coerced or overwritten, and every
broken line is reported. `ext-collect` and `ext-merge-eval` share it.

Line rules (EXT-1): keys `item` (non-empty string), `repeat` (in range(K)), `seed` (= the arm's
`eval_seeds[repeat]`), `score` and `error` (exactly one non-null), optional `sub_metrics` (finite numbers),
`output_ref` (`{path, sha256}` naming a file inside the arm with that SHA-256) and `item_meta` (an object,
never paired). A score is a finite number (not a boolean or a string) inside the evaluator's registered
range (MM-3); an error is `{class, message}` with a class the registry declares (MM-4). An item outside the
registered set is not refused here: the gate reports it (`unknown_item`, MM-6).

Metadata and provenance checks (EXT-44, EXT-45, EXT-47): eval_meta values compare as canonical
JSON bytes, so `true` never equals `1` (`same_json`); an `output_ref` never names a file the arm itself
holds (`items.jsonl`, `eval_meta.json`, `merge.json`, `judge_batches.jsonl`), so a merge or a collect copy
never overwrites one, and it resolves inside the arm (no symbolic link out); `provenance_problems` checks a
merged arm's `merge.json` against its sources.
"""

from __future__ import annotations

import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

from ..canonical import canonical_json, file_sha256, text_lines
from ..renamed import old_schema_note

META_SCHEMA = "gatedtrain.item_eval"
META_KEYS = ("schema", "schema_version", "evaluator_id", "adapter", "metric", "direction", "score_range", "scorer",
             "checkpoint", "normalizer_sha256", "env_version_sha256", "item_set_hash", "eval_seeds", "sampler",
             "sampler_hash", "framework")
META_OPTIONAL = ("created_at",)                       # wall clock, excluded from every hash (MM-2)
LINE_REQUIRED = ("item", "repeat", "seed", "score", "error")
LINE_OPTIONAL = ("sub_metrics", "output_ref", "item_meta")
MAX_MESSAGES = 8
RESERVED_FILES = ("items.jsonl", "eval_meta.json", "merge.json", "judge_batches.jsonl")   # EXT-45
MERGE_SCHEMAS = ("gatedtrain.ext_merge_eval", "gatedtrain.ext_from_nel")


def same_json(a: Any, b: Any) -> bool:
    """EXT-44: equal canonical JSON (RFC 8785) bytes; `true` is not `1`, `1.0` is `1`. A value that has no
    canonical form (NaN, a non-JSON type) equals nothing."""
    try:
        return canonical_json(a) == canonical_json(b)
    except (TypeError, ValueError):
        return False


def _int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def read_meta(arm: Path) -> tuple[dict | None, list[str]]:
    """eval_meta.json with exactly the MM-2 keys (`created_at` optional)."""
    path = Path(arm) / "eval_meta.json"
    if not path.is_file():
        return None, [f"{path} is missing"]
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, [f"{path} does not parse ({exc.__class__.__name__})"]
    if not isinstance(meta, dict):
        return None, [f"{path} is not a JSON object"]
    out = []
    missing = [k for k in META_KEYS if k not in meta]
    unknown = sorted(set(meta) - set(META_KEYS) - set(META_OPTIONAL))
    if missing:
        out.append(f"eval_meta.json lacks {missing} (MM-2)")
    if unknown:
        out.append(f"eval_meta.json has unknown keys {unknown} (MM-2)")
    if meta.get("schema") != META_SCHEMA:
        out.append(f"eval_meta.json schema is {meta.get('schema')!r}, not {META_SCHEMA}"
                   f"{old_schema_note(meta.get('schema'))}")
    ck = meta.get("checkpoint")
    if "checkpoint" in meta and not (isinstance(ck, dict) and set(ck) == {"model_hash", "base_model_hash"}
                                     and isinstance(ck.get("model_hash"), str)):
        out.append("eval_meta.json checkpoint is not {model_hash, base_model_hash} (MM-2)")
    fw = meta.get("framework")
    if "framework" in meta and not (isinstance(fw, dict) and set(fw) == {"name", "version"}
                                    and all(isinstance(fw[k], str) for k in fw)):
        out.append("eval_meta.json framework is not {name, version} (MM-2)")
    seeds = meta.get("eval_seeds")
    if "eval_seeds" in meta and not (isinstance(seeds, list) and seeds and all(_int(s) for s in seeds)
                                     and len(set(seeds)) == len(seeds)):
        out.append(f"eval_meta.json eval_seeds {seeds!r} is not a list of distinct integers")
    return meta, out


def meta_differences(meta: dict, expected: dict[str, Any]) -> list[str]:
    """`eval_meta` keys whose value differs from `expected` (MM-3 registry agreement, EXT-3)."""
    out = []
    for key, want in expected.items():
        got = meta.get(key)
        if isinstance(want, tuple):
            want = list(want)
        if not same_json(got, want):
            out.append(f"eval_meta.json {key} is {got!r}, not {want!r}")
    return out


def _output_ref_problem(arm: Path, ref: Any) -> str | None:
    if ref is None:
        return None
    if not (isinstance(ref, dict) and set(ref) == {"path", "sha256"} and isinstance(ref["path"], str)):
        return "output_ref is not {path, sha256}"
    rel = PurePosixPath(ref["path"])
    if rel.is_absolute() or ".." in rel.parts or not ref["path"]:
        return f"output_ref path {ref['path']!r} leaves the arm"
    if len(rel.parts) == 1 and rel.parts[0] in RESERVED_FILES:
        return f"output_ref path {ref['path']!r} names a file of the arm itself ({', '.join(RESERVED_FILES)})"
    target = (Path(arm) / ref["path"])
    try:
        inside = target.resolve().is_relative_to(Path(arm).resolve())
    except OSError:
        inside = False
    if not inside:
        return f"output_ref path {ref['path']!r} leaves the arm"
    if not target.is_file():
        return f"output_ref file {ref['path']!r} is absent"
    if ref["sha256"] != file_sha256(target):
        return f"output_ref file {ref['path']!r} differs from its sha256"
    return None


def read_lines(arm: Path, *, seeds: list[int], score_range, error_classes) -> tuple[list[dict], list[str]]:
    """Every line of `<arm>/items.jsonl` checked against the EXT-1 rules. Returns (records, problems); a
    record keeps its `item_meta` (the caller drops it before the gate)."""
    path = Path(arm) / "items.jsonl"
    if not path.is_file():
        return [], [f"{path} is missing (pass an arm directory, not its parent)"]
    lo, hi = (list(score_range) + [None, None])[:2]
    k = len(seeds)
    records: list[dict] = []
    bad: list[str] = []
    first: dict[tuple, int] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [], [f"{path} cannot be read ({exc.__class__.__name__})"]
    for n, line in enumerate(text_lines(text), start=1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            bad.append(f"line {n} does not parse")
            continue
        if not isinstance(rec, dict):
            bad.append(f"line {n} is not a JSON object")
            continue
        why = []
        missing = [f for f in LINE_REQUIRED if f not in rec]
        unknown = sorted(set(rec) - set(LINE_REQUIRED) - set(LINE_OPTIONAL))
        if missing:
            why.append(f"lacks {missing}")
        if unknown:
            why.append(f"has unknown keys {unknown}")
        item, rep, seed = rec.get("item"), rec.get("repeat"), rec.get("seed")
        score, error = rec.get("score"), rec.get("error")
        if not (isinstance(item, str) and item):
            why.append(f"item {item!r} is not a non-empty string")
        if not (_int(rep) and 0 <= rep < k):
            why.append(f"repeat {rep!r} is outside range({k})")
        elif not (_int(seed) and seed == seeds[rep]):
            why.append(f"seed {seed!r} is not eval_seeds[{rep}] = {seeds[rep]}")
        if (score is None) == (error is None):
            why.append("score and error are both set or both null")
        elif score is not None:
            if not _finite(score):
                why.append(f"score {score!r} is not a finite number")
            elif (lo is not None and score < lo) or (hi is not None and score > hi):
                why.append(f"score {score!r} is outside the registered range [{lo}, {hi}]")
        else:
            if not (isinstance(error, dict) and set(error) == {"class", "message"}
                    and isinstance(error.get("class"), str) and isinstance(error.get("message"), str)):
                why.append("error is not {class, message}")
            elif error["class"] not in error_classes:
                why.append(f"error class {error['class']!r} is not declared by the evaluator {list(error_classes)}")
        sub = rec.get("sub_metrics")
        if sub is not None and not (isinstance(sub, dict) and all(_finite(v) for v in sub.values())):
            why.append("sub_metrics is not an object of finite numbers")
        ref_bad = _output_ref_problem(Path(arm), rec.get("output_ref"))
        if ref_bad:
            why.append(ref_bad)
        meta = rec.get("item_meta")
        if meta is not None and not isinstance(meta, dict):
            why.append("item_meta is not an object")
        if not why and (item, rep) in first:
            why.append(f"duplicates line {first[(item, rep)]}: (item, repeat) = ({item!r}, {rep})")
        if why:
            bad.append(f"line {n} " + "; ".join(why))
            continue
        first[(item, rep)] = n
        records.append(rec)
    if not records and not bad:
        bad.append(f"{path} holds no records")
    if len(bad) > MAX_MESSAGES:
        bad = bad[:MAX_MESSAGES] + [f"{len(bad) - MAX_MESSAGES} more bad lines"]
    return records, bad


def _source_meta_differs(src_meta: Any, meta: dict) -> bool:
    drop = ("eval_seeds", "created_at")
    if not isinstance(src_meta, dict):
        return True
    return not same_json({k: v for k, v in src_meta.items() if k not in drop},
                         {k: v for k, v in meta.items() if k not in drop})


def provenance_problems(arm: Path, meta: dict, *, nel_benchmark: str | None, row: dict | None = None,
                        item_rows: list[dict] | None = None) -> list[str]:
    """EXT-47 (MM-5, MM-11 merge provenance): a merged arm's `merge.json` against its sources. None to check
    when the arm has no merge.json (an arm written in one piece).

    `gatedtrain.ext_merge_eval`: the sources' `seeds` together are the arm's `eval_seeds`, each once; every source
    `<path>/items.jsonl` and `<path>/eval_meta.json` exists with the recorded SHA-256; each source eval_meta
    equals the arm's but for `eval_seeds` and `created_at`.
    `gatedtrain.ext_from_nel`: source r is `{repeat: r, seed: eval_seeds[r]}`; `<run_dir>/<benchmark>/results.jsonl`
    and `<run_dir>/full_config.yaml` exist with the recorded SHA-256.
    A source that is gone cannot be checked and is refused (collect while the sources exist).

    EXT-52: when those checks pass, the arm's `items.jsonl` must be byte for byte the file the command writes from
    its sources: `merge_records` of the source arms (evaluator registry `row` for the line rules) for
    `ext_merge_eval`, `from_nel.convert_records` of the runs' `results.jsonl` (the registered `item_rows`) for
    `ext_from_nel`. An edited score, a dropped or reordered line is refused."""
    path = Path(arm) / "merge.json"
    if not path.exists():
        return []
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"merge.json does not parse ({exc.__class__.__name__})"]
    if not isinstance(doc, dict) or doc.get("schema") not in MERGE_SCHEMAS:
        got = doc.get("schema") if isinstance(doc, dict) else None
        return [f"merge.json schema is {got!r}, not one of {', '.join(MERGE_SCHEMAS)}{old_schema_note(got)}"]
    sources = doc.get("sources")
    if not (isinstance(sources, list) and sources and all(isinstance(s, dict) for s in sources)):
        return ["merge.json has no list of sources"]
    seeds = meta.get("eval_seeds")
    if not isinstance(seeds, list):
        return []                                          # read_meta reports the seeds
    out: list[str] = []

    def check_file(f: Any, want: Any, what: str) -> None:
        try:
            got = file_sha256(Path(f))
        except (OSError, TypeError, ValueError):
            out.append(f"merge.json source {what} {f} cannot be read; provenance cannot be checked")
            return
        if got != want:
            out.append(f"merge.json source {what} {f} is {got}, not the recorded {want}")

    if doc["schema"] == "gatedtrain.ext_merge_eval":
        listed = [x for s in sources for x in (s.get("seeds") if isinstance(s.get("seeds"), list) else [None])]
        if not (all(_int(x) for x in listed) and all(_int(x) for x in seeds) and len(set(listed)) == len(listed)
                and sorted(listed) == sorted(seeds)):
            out.append(f"merge.json sources cover seeds {listed}, not the arm's eval_seeds {seeds} once each")
        for i, s in enumerate(sources):
            src = s.get("path")
            if not isinstance(src, str):
                out.append(f"merge.json sources[{i}] has no path")
                continue
            check_file(Path(src) / "items.jsonl", s.get("items_sha256"), f"sources[{i}] items.jsonl")
            check_file(Path(src) / "eval_meta.json", s.get("eval_meta_sha256"), f"sources[{i}] eval_meta.json")
            try:
                src_meta = json.loads((Path(src) / "eval_meta.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue                                   # reported by check_file
            if _source_meta_differs(src_meta, meta):
                out.append(f"merge.json sources[{i}] eval_meta.json differs from the arm's beyond eval_seeds")
            elif not same_json(src_meta.get("eval_seeds"), s.get("seeds")):
                out.append(f"merge.json sources[{i}] seeds {s.get('seeds')} are not its eval_seeds "
                           f"{src_meta.get('eval_seeds')}")
    else:
        got = [(s.get("repeat"), s.get("seed")) for s in sources]
        if not same_json([list(x) for x in got], [[r, x] for r, x in enumerate(seeds)]):
            out.append(f"merge.json sources (repeat, seed) {got} are not the arm's eval_seeds {seeds} in order")
        for i, s in enumerate(sources):
            run = s.get("run_dir")
            if not isinstance(run, str) or nel_benchmark is None:
                out.append(f"merge.json sources[{i}] has no run_dir")
                continue
            check_file(Path(run) / nel_benchmark / "results.jsonl", s.get("results_sha256"),
                       f"sources[{i}] results.jsonl")
            check_file(Path(run) / "full_config.yaml", s.get("config_sha256"), f"sources[{i}] full_config.yaml")
    if not out and row is not None:
        out += _rederived_problems(Path(arm), doc, seeds, row=row, item_rows=item_rows)
    return out[:MAX_MESSAGES]


def _rederived_problems(arm: Path, doc: dict, seeds: list, *, row: dict, item_rows: list[dict] | None) -> list[str]:
    """EXT-52: the arm's items.jsonl against the one its sources give (sources already checked)."""
    if doc["schema"] == "gatedtrain.ext_merge_eval":
        parts = []
        for i, s in enumerate(doc["sources"]):
            recs, bad = read_lines(Path(s["path"]), seeds=list(s["seeds"]), score_range=row["score_range"],
                                   error_classes=tuple(row["error_classes"]))
            if bad:
                return [f"merge.json sources[{i}] has bad lines ({bad[0]}); the arm cannot be re-derived"]
            parts.append((recs, list(s["seeds"])))
        want = items_text(merge_records(parts, list(seeds)))
    else:
        from ..tierb import _read_run_strict
        from .from_nel import convert_records

        if not item_rows:
            return []                                      # the item set is refused elsewhere (EXT-5)
        runs = []
        for i, s in enumerate(doc["sources"]):
            recs, bad = _read_run_strict(Path(s["run_dir"]) / row["nel_benchmark"] / "results.jsonl")
            if bad:
                return [f"merge.json sources[{i}] results.jsonl has bad lines ({bad[0]}); the arm cannot be "
                        "re-derived"]
            runs.append(recs)
        records, bad = convert_records(runs, item_rows, list(seeds), f"scorer:{row['scorer']['name']}")
        if bad:
            return [f"merge.json sources[{bad[0][0]}]: {bad[0][1]}; the arm cannot be re-derived"]
        want = items_text(records)
    try:
        got = (arm / "items.jsonl").read_bytes()
    except OSError:
        return []                                          # read_lines reports a missing items.jsonl
    if got != want.encode("utf-8"):
        return [f"items.jsonl is not the file {doc['schema']} writes from the sources in merge.json; the arm was "
                "edited after it was written (EXT-52)"]
    return []


def merge_records(sources: list[tuple[list[dict], list[int]]], seeds: list[int]) -> list[dict]:
    """MM-5 (`ext-merge-eval`): the records of (source records, source eval_seeds) relabelled to the repeat
    of their seed in the registered order `seeds`, sorted by the first source's item order, then repeat."""
    merged = []
    for recs, run_seeds in sources:
        for rec in recs:
            seed = run_seeds[rec["repeat"]]
            merged.append({**rec, "repeat": seeds.index(seed), "seed": seed})
    order: dict[str, int] = {}
    for rec in sources[0][0] if sources else []:
        order.setdefault(rec["item"], len(order))
    merged.sort(key=lambda x: (order.get(x["item"], len(order)), x["repeat"]))
    return merged


def items_text(records: list[dict]) -> str:
    """`items.jsonl` as `ext-merge-eval` and `ext-from-nel` write it (EXT-52 compares these bytes)."""
    return "".join(json.dumps(r) + "\n" for r in records)


def gate_record(rec: dict) -> dict:
    """The record as `gates.evaluate_items` takes it: `item_meta` dropped (it never pairs), the other
    optional fields defaulted to null."""
    return {"item": rec["item"], "repeat": rec["repeat"], "seed": rec["seed"], "score": rec["score"],
            "error": rec["error"], "sub_metrics": rec.get("sub_metrics"), "output_ref": rec.get("output_ref")}
