"""Per-item arm directories (MULTIMODAL-SPEC.md MM-1 to MM-3; tests/README.md section 11.18 EXT-1, EXT-3).

An arm is `items.jsonl` (one line per (item, repeat)) plus `eval_meta.json`. This reader is strict, in
the manner of `runner.read_arm_records` (R11A-28): nothing is skipped, coerced or overwritten, and every
broken line is reported. `ext-collect` and `ext-merge-eval` share it.

Line rules (EXT-1): keys `item` (non-empty string), `repeat` (in range(K)), `seed` (= the arm's
`eval_seeds[repeat]`), `score` and `error` (exactly one non-null), optional `sub_metrics` (finite numbers),
`output_ref` (`{path, sha256}` naming a regular file inside the arm with that SHA-256) and `item_meta` (an object,
never paired). A score is a finite number (not a boolean or a string) inside the evaluator's registered
range (MM-3); an error is `{class, message}` with a class the registry declares (MM-4). An item outside the
registered set is not refused here: the gate reports it (`unknown_item`, MM-6).

Metadata and provenance checks (EXT-44, EXT-45, EXT-47): eval_meta values compare as canonical
JSON bytes, so `true` never equals `1` (`same_json`); an `output_ref` never names a file the arm itself
holds (`items.jsonl`, `eval_meta.json`, `merge.json`, `judge_batches.jsonl`), so a merge or a collect copy
never overwrites one; `provenance_problems` checks a merged arm's `merge.json` against its sources.

Files are read through `armfs` (docs/findings.md, "Review of the 3.12 path fix" and "Review of PR 2"): nothing is
read by path and no symbolic link is followed below the arm directory. `items.jsonl`, `eval_meta.json` and
`merge.json` are read once (`ArmSnapshot`) and every check parses those bytes. An `output_ref` path (MM-1) is a
relative path in canonical form (UTF-8, no NUL, no `.`, `..` or empty component) with no symbolic link (hard links
are allowed), and `copy_output_ref` (collect, merge) copies the bytes it hashes, so a file swapped after the check
is refused, never copied. Provenance sources (MM-5) are read the same way below the sources root. A path that
cannot be read is a problem, never a traceback, and a refused file's content is never echoed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from ..canonical import canonical_json, text_lines
from ..errors import Problem, Refused
from .armfs import (ArmSnapshot, Unreadable, errname, open_arm, open_dir, open_regular, path_problem, read_file,
                    sha256, under_root)
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


def _snapshot(arm: Path, snap: ArmSnapshot | None) -> ArmSnapshot:
    return snap if snap is not None else ArmSnapshot.read(arm)


def _json(data: bytes) -> tuple[Any, str | None]:
    try:
        return json.loads(data.decode("utf-8")), None
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        return None, exc.__class__.__name__


def read_meta(arm: Path, snap: ArmSnapshot | None = None) -> tuple[dict | None, list[str]]:
    """eval_meta.json with exactly the MM-2 keys (`created_at` optional), from the snapshot's bytes (`snap`, else
    read now)."""
    snap = _snapshot(arm, snap)
    path = Path(arm) / "eval_meta.json"
    why = snap.problem("eval_meta.json")
    if why:
        return None, [why]
    meta, bad = _json(snap.files["eval_meta.json"].data)
    if bad:
        return None, [f"{path} does not parse ({bad})"]
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


def _output_ref_problem(arm: Path, ref: Any, key: tuple[int, int] | None = None) -> str | None:
    if ref is None:
        return None
    if not (isinstance(ref, dict) and set(ref) == {"path", "sha256"} and isinstance(ref["path"], str)):
        return "output_ref is not {path, sha256}"
    why = path_problem(ref["path"])
    if why:
        return f"output_ref path {ref['path']!r} {why}"
    if "/" not in ref["path"] and ref["path"] in RESERVED_FILES:
        return f"output_ref path {ref['path']!r} names a file of the arm itself ({', '.join(RESERVED_FILES)})"
    fd, why = open_output_ref(arm, ref["path"], key)
    if fd is None:
        return why
    try:
        got = _sha256_fd(fd)
    except Unreadable as exc:
        return exc.text(f"output_ref file {ref['path']!r}")
    finally:
        os.close(fd)
    if ref["sha256"] != got:
        return f"output_ref file {ref['path']!r} differs from its sha256"
    return None


def _sha256_fd(fd: int, out=None) -> str:
    """SHA-256 of what is read from `fd`, written on to the binary file `out` if given. A read error is Unreadable
    (the arm's fault); a write error stays OSError (the output's)."""
    h = hashlib.sha256()
    while True:
        try:
            chunk = os.read(fd, 1 << 20)
        except OSError as exc:
            raise Unreadable("error", errname(exc)) from None
        if not chunk:
            return "sha256:" + h.hexdigest()
        h.update(chunk)
        if out is not None:
            out.write(chunk)


def open_output_ref(arm: Path, path: str, key: tuple[int, int] | None = None) -> tuple[int | None, str | None]:
    """MM-1, EXT-45: (a read-only descriptor of the regular file `path` names inside `arm`, None) or (None, why).
    The arm directory is opened as given (it may itself be a link) and, with `key`, must be the snapshot's; `path`
    (canonical, `path_problem`) is reached by `armfs.open_regular`: no link anywhere, only a regular file opened
    for reading, never limited by `PATH_MAX`."""
    try:
        dir_fd = open_arm(arm, key)
    except Unreadable as exc:
        what = "was replaced since it was read" if exc.kind == "changed" else exc.reason()
        return None, f"output_ref file {path!r} cannot be opened: the arm directory {what}"
    try:
        return open_regular(dir_fd, path), None
    except Unreadable as exc:
        if exc.kind == "link":
            return None, exc.text(f"output_ref path {path!r}")
        return None, exc.text(f"output_ref file {path!r}")
    finally:
        os.close(dir_fd)


def copy_output_ref(arm: Path, ref: dict, dst: Path, *, flag: str, key: tuple[int, int] | None = None) -> None:
    """Copy the `output_ref` file of `arm` to `dst` (collect, merge): the bytes come from the descriptor
    `open_output_ref` gives and are hashed while copied, so a file changed or swapped for a link after the line
    checks, or one that cannot be read, is refused (Refused, EXT-ARM on `flag`; the caller removes its partial
    output), never copied. A failed write of `dst` raises OSError."""
    fix = "do not change an arm while it is collected or merged"
    fd, why = open_output_ref(arm, ref["path"], key)
    if fd is None:
        raise Refused([Problem("EXT-ARM", flag, f"{arm}: {why} (changed after it was checked)", fix)])
    try:
        with open(dst, "wb") as out:
            got = _sha256_fd(fd, out)
    except Unreadable as exc:
        dst.unlink(missing_ok=True)
        what = "output_ref file " + repr(ref["path"])
        raise Refused([Problem("EXT-ARM", flag, f"{arm}: {exc.text(what)}", fix)]) from None
    finally:
        os.close(fd)
    if got != ref["sha256"]:
        dst.unlink(missing_ok=True)
        raise Refused([Problem("EXT-ARM", flag, f"{arm}: output_ref file {ref['path']!r} differs from its sha256 "
                               "(changed after it was checked)", fix)])


def read_lines(arm: Path, *, seeds: list[int], score_range, error_classes,
               snap: ArmSnapshot | None = None) -> tuple[list[dict], list[str]]:
    """Every line of `<arm>/items.jsonl` (the snapshot's bytes: `snap`, else read now) checked against the EXT-1
    rules. Returns (records, problems); a record keeps its `item_meta` (the caller drops it before the gate)."""
    snap = _snapshot(arm, snap)
    path = Path(arm) / "items.jsonl"
    why = snap.problem("items.jsonl")
    if why:
        return [], [why + (" (pass an arm directory, not its parent)" if why.endswith("is missing") else "")]
    lo, hi = (list(score_range) + [None, None])[:2]
    k = len(seeds)
    records: list[dict] = []
    bad: list[str] = []
    first: dict[tuple, int] = {}
    try:
        text = snap.files["items.jsonl"].data.decode("utf-8")
    except UnicodeDecodeError as exc:
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
        ref_bad = _output_ref_problem(Path(arm), rec.get("output_ref"), snap.key)
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
                        item_rows: list[dict] | None = None, snap: ArmSnapshot | None = None,
                        sources_root: Path | str | None = None) -> list[str]:
    """EXT-47 (MM-5, MM-11 merge provenance): a merged arm's `merge.json` against its sources. None to check
    when the arm has no merge.json (an arm written in one piece).

    `gatedtrain.ext_merge_eval`: the sources' `seeds` together are the arm's `eval_seeds`, each once; every source
    `<path>/items.jsonl` and `<path>/eval_meta.json` exists with the recorded SHA-256; each source eval_meta
    equals the arm's but for `eval_seeds` and `created_at`.
    `gatedtrain.ext_from_nel`: source r is `{repeat: r, seed: eval_seeds[r]}`; `<run_dir>/<benchmark>/results.jsonl`
    and `<run_dir>/full_config.yaml` exist with the recorded SHA-256.
    A source that is gone cannot be checked and is refused (collect while the sources exist).

    Sources (review of PR 2, A2): each `path` or `run_dir` is an absolute path in normal form at or below
    `sources_root` (`ext-collect`: the host's output root, or `--sources-root`); it is reached from the root's
    descriptor without following a link, and its files are read as an arm's (`armfs`): no FIFO, device or link is
    ever opened. A source outside the root, or none configured, cannot be checked and is refused. A mismatch says
    only that the file differs from the recorded hash, never the hash it has.

    EXT-52: when those checks pass, the arm's `items.jsonl` must be byte for byte the file the command writes from
    its sources: `merge_records` of the source arms (evaluator registry `row` for the line rules) for
    `ext_merge_eval`, `from_nel.convert_records` of the runs' `results.jsonl` (the registered `item_rows`) for
    `ext_from_nel`. An edited score, a dropped or reordered line is refused."""
    snap = _snapshot(arm, snap)
    f = snap.files["merge.json"]
    if f.problem == "missing":
        return []
    if f.problem:
        return [f"merge.json {f.problem}; provenance cannot be checked"]
    doc, bad = _json(f.data)
    if bad:
        return [f"merge.json does not parse ({bad})"]
    if not isinstance(doc, dict) or doc.get("schema") not in MERGE_SCHEMAS:
        got = doc.get("schema") if isinstance(doc, dict) else None
        return [f"merge.json schema is {got!r}, not one of {', '.join(MERGE_SCHEMAS)}{old_schema_note(got)}"]
    sources = doc.get("sources")
    if not (isinstance(sources, list) and sources and all(isinstance(s, dict) for s in sources)):
        return ["merge.json has no list of sources"]
    seeds = meta.get("eval_seeds")
    if not isinstance(seeds, list):
        return []                                          # read_meta reports the seeds
    if sources_root is None:
        return ["merge.json names sources but no sources root is configured; provenance cannot be checked"]
    try:
        root_fd = open_arm(sources_root)
    except Unreadable as exc:
        return [f"the sources root {sources_root} {exc.reason()}; provenance cannot be checked"]
    out: list[str] = []

    def source_dir(i: int, key: str, p: Any) -> int | None:
        rel = under_root(sources_root, p)
        if rel is None:
            out.append(f"merge.json sources[{i}] {key} {p!r} is not an absolute path in normal form under the "
                       f"sources root {sources_root}; provenance cannot be checked")
            return None
        try:
            return open_dir(root_fd, rel)
        except Unreadable as exc:
            out.append(f"merge.json sources[{i}] {key} {p!r} {exc.reason()}; provenance cannot be checked")
            return None

    def check(data: bytes | None, why: str | None, want: Any, what: str) -> None:
        if why is not None:
            out.append(f"merge.json source {what} {why}; provenance cannot be checked")
        elif sha256(data) != want:
            out.append(f"merge.json source {what} differs from the recorded hash")

    derived: list = []
    try:
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
                dfd = source_dir(i, "path", src)
                if dfd is None:
                    continue
                try:
                    src_snap = ArmSnapshot.read(src, dir_fd=dfd)
                finally:
                    os.close(dfd)
                derived.append(src_snap)
                for name, key in (("items.jsonl", "items_sha256"), ("eval_meta.json", "eval_meta_sha256")):
                    sf = src_snap.files[name]
                    check(sf.data, None if sf.problem is None else ("is missing" if sf.problem == "missing"
                                                                     else sf.problem), s.get(key),
                          f"sources[{i}] {name}")
                if src_snap.files["eval_meta.json"].problem is not None:
                    continue
                src_meta, bad = _json(src_snap.files["eval_meta.json"].data)
                if bad:
                    out.append(f"merge.json sources[{i}] eval_meta.json does not parse ({bad})")
                elif _source_meta_differs(src_meta, meta):
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
                if not isinstance(run, str):
                    out.append(f"merge.json sources[{i}] has no run_dir")
                    continue
                if nel_benchmark is None or path_problem(nel_benchmark):
                    out.append(f"the evaluator's nel_benchmark {nel_benchmark!r} is not a relative path in canonical "
                               "form; provenance cannot be checked")
                    continue
                dfd = source_dir(i, "run_dir", run)
                if dfd is None:
                    continue
                try:
                    files = {}
                    for name, key in ((f"{nel_benchmark}/results.jsonl", "results_sha256"),
                                      ("full_config.yaml", "config_sha256")):
                        try:
                            files[name] = read_file(dfd, name)
                            check(files[name], None, s.get(key), f"sources[{i}] {name}")
                        except Unreadable as exc:
                            check(None, exc.reason(), s.get(key), f"sources[{i}] {name}")
                    derived.append(files.get(f"{nel_benchmark}/results.jsonl"))
                finally:
                    os.close(dfd)
    finally:
        os.close(root_fd)
    if not out and row is not None:
        out += _rederived_problems(snap, doc, seeds, derived, row=row, item_rows=item_rows)
    return out[:MAX_MESSAGES]


def _rederived_problems(snap: ArmSnapshot, doc: dict, seeds: list, derived: list, *, row: dict,
                        item_rows: list[dict] | None) -> list[str]:
    """EXT-52: the arm's items.jsonl against the one its sources give (sources already checked and read:
    `derived` holds the source arms' snapshots, or the runs' results.jsonl bytes)."""
    if doc["schema"] == "gatedtrain.ext_merge_eval":
        parts = []
        for i, (s, src_snap) in enumerate(zip(doc["sources"], derived)):
            recs, bad = read_lines(Path(s["path"]), seeds=list(s["seeds"]), score_range=row["score_range"],
                                   error_classes=tuple(row["error_classes"]), snap=src_snap)
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
        for i, (s, data) in enumerate(zip(doc["sources"], derived)):
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                return [f"merge.json sources[{i}] results.jsonl is not UTF-8; the arm cannot be re-derived"]
            recs, bad = _read_run_strict(Path(s["run_dir"]) / row["nel_benchmark"] / "results.jsonl", text=text)
            if bad:
                return [f"merge.json sources[{i}] results.jsonl has bad lines ({bad[0]}); the arm cannot be "
                        "re-derived"]
            runs.append(recs)
        records, bad = convert_records(runs, item_rows, list(seeds), f"scorer:{row['scorer']['name']}")
        if bad:
            return [f"merge.json sources[{bad[0][0]}]: {bad[0][1]}; the arm cannot be re-derived"]
        want = items_text(records)
    got = snap.files["items.jsonl"].data
    if got is None:
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
