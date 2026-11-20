"""Impact query over the ledger and lineage edges with DuckDB (05 section 11.3, 05:352-354).

"Which checkpoints possibly consumed shard S": follow the stage's `used`
edge to the dataset snapshot, find the first step that consumed S in the
ledger (outcome `consumed`, the 05:354 default), and return every checkpoint
the stage generated whose `last_step_included` is at or after that step
(05 section 12.4, 05:406-408; a skeleton run is one attempt, one segment).
Each result reports the edge precision; an answer through a coarser-than-row
edge reads "possibly consumed" (05:350). The baseline is a `used` input of
the stage, never one of its generated checkpoints, so it is never returned
for a stage shard.

Lineage layout written by the runner (format free, tests/README.md section 5):
- `lineage/shards.parquet`: (run_id, stage_id, snapshot, shard, shard_sha256)
  for every shard file of the stage's training dataset;
- `lineage/checkpoints.parquet`: (run_id, stage_id, checkpoint_hash,
  last_step_included) for every checkpoint the stage generated;
- `lineage/edges.json`: the same graph as `used` / `wasGeneratedBy` edges;
- `lineage/orphaned.parquet`: (run_id, stage_id, attempt_id, step, snapshot,
  shard, precision) for every record after the attempt's last committed cut.

Counts (05 section 12.4, 05:424; the impact join reports them, 05:463; R10A-9):
`orphaned` counts the queried shard's ConsumptionRecords (precision
`iteration`, one per (step, shard)) after the last committed cut; they enter
no rollup. `superseded` is 0: a skeleton run is one attempt and one segment.
A resumed external run (EXT-RES-5) also keeps its orphaned rows in the ledger with `status` `orphaned`; the
first-use query reads only rows whose status is `kept` (or absent), and the run's lineage/orphaned.parquet
carries the same rows for the counts and ranges.

Design revision 11 (tests/README.md section 11.16):
- R11-18 (R11A-18): each result carries `derivation`, `reconstructed` if any contributing ledger record is,
  `observed` otherwise, and null for a ledger written before revision 11 (no `derivation` column);
- R11-16 (R11A-19; 05 section 12.4, section 13.2): the query "which checkpoints possibly consumed shard S"
  is a membership question, so the answer adds step ranges per run, attempt and shard:
  `orphaned` from lineage/orphaned.parquet (first and last orphaned step of the queried shard), and
  `superseded`, empty for one attempt and one segment.
"""

from __future__ import annotations

import re
from pathlib import Path

import pyarrow.parquet as pq
from typing import Any

import duckdb

from .canonical import file_sha256

SHA_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

QUERY = """
WITH target AS (
    SELECT DISTINCT run_id, stage_id, snapshot, shard
    FROM read_parquet($shards)
    WHERE shard_sha256 = $target
),
first_use AS (
    SELECT t.run_id, t.stage_id, MIN(l.step) AS first_step,
           -- coarsest precision on the path (05:350); order: row < step_range < iteration < stage
           arg_max(l.precision, CASE l.precision WHEN 'row' THEN 0 WHEN 'step_range' THEN 1
                                WHEN 'iteration' THEN 2 ELSE 3 END) AS precision,
           {derivation} AS derivation
    FROM read_parquet($ledger, union_by_name = true) AS l
    JOIN target AS t
      ON l.run_id = t.run_id
     AND l.data_ref.snapshot = t.snapshot
     AND l.data_ref.shard = t.shard
    WHERE l.outcome = 'consumed'{kept}
    GROUP BY t.run_id, t.stage_id
)
SELECT c.checkpoint_hash, c.run_id, c.stage_id, c.last_step_included, f.first_step, f.precision, f.derivation,
       CASE WHEN f.precision = 'row' THEN 'consumed' ELSE 'possibly consumed' END AS answer
FROM read_parquet($checkpoints) AS c
JOIN first_use AS f ON c.run_id = f.run_id AND c.stage_id = f.stage_id
WHERE c.last_step_included >= f.first_step
ORDER BY c.run_id, c.stage_id, c.last_step_included
"""


class UnknownTarget(ValueError):
    pass


def resolve_target(shard: str) -> str:
    """`sha256:<hex>` as given, or the SHA-256 of a shard file's bytes."""
    if shard.startswith("sha256:"):
        if not SHA_RE.match(shard):
            raise UnknownTarget(f"{shard!r} is not sha256:<64 lowercase hex>")
        return shard
    path = Path(shard)
    if not path.is_file():
        raise UnknownTarget(f"{shard!r} is neither sha256:<hex> nor an existing shard file")
    return file_sha256(path)


def run_dirs(*, run_dir: Path | None = None, runs_root: Path | None = None) -> list[Path]:
    if (run_dir is None) == (runs_root is None):
        raise ValueError("give exactly one of run_dir and runs_root")
    if run_dir is not None:
        dirs = [Path(run_dir)]
    else:
        dirs = sorted(p for p in Path(runs_root).iterdir() if p.is_dir())
    out = [d for d in dirs if (d / "lineage" / "checkpoints.parquet").is_file()
           and (d / "lineage" / "shards.parquet").is_file() and (d / "ledger" / "consumption.parquet").is_file()]
    if run_dir is not None and not out:
        raise UnknownTarget(f"{run_dir} is not a completed gatedtrain run directory (no ledger or lineage)")
    return out


# R11A-18: `reconstructed` wins over `observed`; NULL rows (a pre-revision-11 file read with union_by_name)
# count as unknown.
DERIVATION_EXPR = ("CASE WHEN bool_or(l.derivation = 'reconstructed') THEN 'reconstructed' "
                   "WHEN bool_and(l.derivation = 'observed') THEN 'observed' ELSE NULL END")


def _has_derivation(ledgers: list[str]) -> bool:
    """Whether any ledger file has the revision 11 `derivation` column."""
    return any("derivation" in pq.read_schema(p).names for p in ledgers)


def _has_status(ledgers: list[str]) -> bool:
    """Whether any ledger file has the `status` column (EXT-RES-5); a file without it holds only kept rows."""
    return any("status" in pq.read_schema(p).names for p in ledgers)


# EXT-RES-5: an orphaned row enters no rollup; a NULL or absent status is `kept`.
KEPT_FILTER = "\n      AND COALESCE(l.status, 'kept') = 'kept'"


RANGE_QUERY = """
SELECT o.run_id, o.attempt_id, o.shard, MIN(o.step) AS first_step, MAX(o.step) AS last_step
FROM read_parquet($orphaned, union_by_name = true) AS o
JOIN (SELECT DISTINCT run_id, stage_id, snapshot, shard FROM read_parquet($shards) WHERE shard_sha256 = $target) AS t
  ON o.run_id = t.run_id AND o.stage_id = t.stage_id AND o.snapshot = t.snapshot AND o.shard = t.shard
GROUP BY o.run_id, o.attempt_id, o.shard
ORDER BY o.run_id, o.attempt_id, o.shard
"""


def impact_ranges(shard: str, *, run_dir: Path | None = None, runs_root: Path | None = None) -> dict[str, list]:
    """{orphaned, superseded} step ranges of the queried shard, one per run, attempt and shard
    (05 section 12.4, R11-16). A run directory without lineage/orphaned.parquet contributes none."""
    target = resolve_target(shard)
    dirs = [d for d in run_dirs(run_dir=run_dir, runs_root=runs_root) if (d / "lineage" / "orphaned.parquet").is_file()]
    orphaned: list[dict[str, Any]] = []
    if dirs:
        con = duckdb.connect(":memory:")
        try:
            cur = con.execute(RANGE_QUERY, {
                "target": target,
                "orphaned": [str(d / "lineage" / "orphaned.parquet") for d in dirs],
                "shards": [str(d / "lineage" / "shards.parquet") for d in dirs],
            })
            orphaned = [{"run_id": r[0], "attempt_id": r[1], "shard": r[2], "first_step": int(r[3]),
                         "last_step": int(r[4])} for r in cur.fetchall()]
        finally:
            con.close()
    return {"orphaned": orphaned, "superseded": []}


ORPHAN_QUERY = """
SELECT COUNT(*) FROM read_parquet($orphaned, union_by_name = true) AS o
JOIN (SELECT DISTINCT run_id, stage_id, snapshot, shard FROM read_parquet($shards) WHERE shard_sha256 = $target) AS t
  ON o.run_id = t.run_id AND o.stage_id = t.stage_id AND o.snapshot = t.snapshot AND o.shard = t.shard
"""


def impact_counts(shard: str, *, run_dir: Path | None = None, runs_root: Path | None = None) -> dict[str, int]:
    """{orphaned, superseded} for the queried shard over the selected runs (05:424, 05:463; R10A-9)."""
    target = resolve_target(shard)
    dirs = [d for d in run_dirs(run_dir=run_dir, runs_root=runs_root) if (d / "lineage" / "orphaned.parquet").is_file()]
    orphaned = 0
    if dirs:
        con = duckdb.connect(":memory:")
        try:
            orphaned = int(con.execute(ORPHAN_QUERY, {
                "target": target,
                "orphaned": [str(d / "lineage" / "orphaned.parquet") for d in dirs],
                "shards": [str(d / "lineage" / "shards.parquet") for d in dirs],
            }).fetchone()[0])
        finally:
            con.close()
    return {"orphaned": orphaned, "superseded": 0}


def impact_query(shard: str, *, run_dir: Path | None = None, runs_root: Path | None = None) -> list[dict[str, Any]]:
    """Every checkpoint that possibly consumed the shard (tests/README.md section 6.4)."""
    target = resolve_target(shard)
    dirs = run_dirs(run_dir=run_dir, runs_root=runs_root)
    if not dirs:
        return []
    params = {
        "target": target,
        "shards": [str(d / "lineage" / "shards.parquet") for d in dirs],
        "ledger": [str(d / "ledger" / "consumption.parquet") for d in dirs],
        "checkpoints": [str(d / "lineage" / "checkpoints.parquet") for d in dirs],
    }
    derivation = DERIVATION_EXPR if _has_derivation(params["ledger"]) else "CAST(NULL AS VARCHAR)"
    con = duckdb.connect(":memory:")
    try:
        query = QUERY.replace("{derivation}", derivation).replace(
            "{kept}", KEPT_FILTER if _has_status(params["ledger"]) else "")
        cur = con.execute(query, params)
        cols = [c[0] for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        con.close()
    for r in rows:
        r["last_step_included"] = int(r["last_step_included"])
        r["first_step"] = int(r["first_step"])
    lineage = {d: _lineage(d) for d in dirs}
    model_of = {(c["run_id"], c["stage_id"], c["checkpoint_hash"]): c.get("model_hash")
                for lin in lineage.values() for c in lin["checkpoints"]}
    for r in rows:
        r["model_hash"] = model_of.get((r["run_id"], r["stage_id"], r["checkpoint_hash"]))
        r["via_model_hash"] = None
    rows += _stage_edge_rows(target, lineage)
    rows = _transitive(rows, lineage)
    for r in rows:
        r["shard_sha256"] = target
    rows.sort(key=lambda r: (r["run_id"], r["stage_id"], r["last_step_included"]))
    return rows


# ---------------------------------------------------------------------------
# `stage` edges and transitive impact (tests/README.md section 11.18, EXT-19; MULTIMODAL-SPEC MM-23, MM-30;
# design revision 13, R13-10, 05 section 11.3)
# ---------------------------------------------------------------------------

PRECISION_RANK = {"row": 0, "step_range": 1, "iteration": 2, "stage": 3}


def _coarsest(a: str, b: str) -> str:
    return a if PRECISION_RANK.get(a, 3) >= PRECISION_RANK.get(b, 3) else b


def _answer(precision: str) -> str:
    return "consumed" if precision == "row" else "possibly consumed"


def _lineage(d: Path) -> dict[str, Any]:
    """A run directory's checkpoints, shards and training-stage edges (model `used` with its `model_hash`, the
    data edge's precision). Directories written before the external-run slice have no `model_hash`."""
    import json

    checkpoints = pq.read_table(d / "lineage" / "checkpoints.parquet").to_pylist()
    shards = pq.read_table(d / "lineage" / "shards.parquet").to_pylist()
    stages: dict[tuple[str, str], dict[str, Any]] = {}
    edges = d / "lineage" / "edges.json"
    if edges.is_file():
        doc = json.loads(edges.read_text(encoding="utf-8"))
        for act in doc.get("activities") or []:
            if act.get("kind") != "training_stage" or "/" not in str(act.get("id")):
                continue
            run_id, stage_id = str(act["id"]).split("/", 1)
            used = {u.get("role"): u for u in act.get("used") or [] if isinstance(u, dict)}
            stages[(run_id, stage_id)] = {"data_precision": (used.get("data") or {}).get("precision"),
                                          "model_hash": (used.get("model") or {}).get("model_hash")}
    return {"checkpoints": checkpoints, "shards": shards, "stages": stages}


def _row(c: dict, *, precision: str, derivation: str | None, via: str | None) -> dict[str, Any]:
    return {"checkpoint_hash": c["checkpoint_hash"], "run_id": c["run_id"], "stage_id": c["stage_id"],
            "last_step_included": int(c["last_step_included"]), "first_step": None, "precision": precision,
            "derivation": derivation, "answer": _answer(precision), "model_hash": c.get("model_hash"),
            "via_model_hash": via}


def _stage_edge_rows(target: str, lineage: dict[Path, dict]) -> list[dict[str, Any]]:
    """05 section 11.3 (R11-12, R13-10): a stage whose data edge has precision `stage` and whose snapshot holds
    the shard possibly consumed it with every checkpoint it generated; no first step, no derivation."""
    out = []
    for lin in lineage.values():
        hit = {(s["run_id"], s["stage_id"]) for s in lin["shards"] if s["shard_sha256"] == target}
        for key in sorted(hit):
            if (lin["stages"].get(key) or {}).get("data_precision") != "stage":
                continue
            out += [_row(c, precision="stage", derivation=None, via=None) for c in lin["checkpoints"]
                    if (c["run_id"], c["stage_id"]) == key]
    return out


def _path(r: dict[str, Any]) -> dict[str, Any]:
    """The direct path of a row: its stage's own data edge, the chain's first and only edge (EXT-77)."""
    edge = {"edge": "data", "run_id": r["run_id"], "stage_id": r["stage_id"], "precision": r["precision"],
            "derivation": r["derivation"], "first_step": r["first_step"]}
    return {"via_model_hash": r.get("via_model_hash"), "precision": r["precision"], "derivation": r["derivation"],
            "first_step": r["first_step"], "chain": [edge]}


def _finest(paths: list[dict[str, Any]]) -> dict[str, Any]:
    """The path with the finest precision; on a tie the direct path (no `via_model_hash`), then the first."""
    return min(paths, key=lambda p: (PRECISION_RANK.get(p["precision"], 3), p["via_model_hash"] is not None))


def _transitive(rows: list[dict[str, Any]], lineage: dict[Path, dict]) -> list[dict[str, Any]]:
    """MM-30 (R13-10): a stage whose `used` model has the `model_hash` of a returned checkpoint contributes every
    checkpoint it generated, at that checkpoint's answer precision (the coarsest edge on the path, R13-10);
    repeated until nothing changes.

    Design revision 14 (R14-1, 05 section 11.2; EXT-42, EXT-58): a cut reached by several paths (its own data
    edge, a `used` model edge from each returned parent model) lists every path in `paths`
    (`via_model_hash` null for the direct path) and takes the finest of their precisions; `first_step`,
    `derivation`, `via_model_hash` and `answer` are those of that finest path. A row-level answer is never
    weakened by a coarser path. Paths are keyed by `via_model_hash` and their precision only gets finer, so
    the loop ends.

    EXT-77 (05 section 11.2, "report the precision of every edge"): each path carries `chain`, its edges from the
    shard to the cut: the consuming stage's data edge (`edge: data`, with its precision, derivation and first
    step), then one `edge: model` per `used` model edge (`model_hash`, the parent `checkpoint_hash`, the using
    `run_id` and `stage_id`; precision null: a model edge is an identity by hash, not a consumption edge). The
    path's `precision` is the coarsest precision on its chain."""
    for r in rows:
        r["paths"] = [_path(r)]
    users = [(key, st["model_hash"], lin) for lin in lineage.values() for key, st in lin["stages"].items()
             if st.get("model_hash")]
    by_cut = {(r["run_id"], r["stage_id"], r["checkpoint_hash"]): r for r in rows}
    changed = True
    while changed:
        changed = False
        reach: dict[str, dict[str, Any]] = {}
        for r in rows:
            mh = r.get("model_hash")
            if mh:
                prev = reach.get(mh)
                if prev is None or PRECISION_RANK.get(r["precision"], 3) < PRECISION_RANK.get(prev["precision"], 3):
                    reach[mh] = r
        for key, mh, lin in users:
            if mh not in reach:
                continue
            parent = reach[mh]
            hop = {"edge": "model", "model_hash": mh, "checkpoint_hash": parent["checkpoint_hash"],
                   "run_id": key[0], "stage_id": key[1], "precision": None}
            via = {"via_model_hash": mh, "precision": parent["precision"], "derivation": parent["derivation"],
                   "first_step": None, "chain": [dict(e) for e in _finest(parent["paths"])["chain"]] + [hop]}
            for c in lin["checkpoints"]:
                if (c["run_id"], c["stage_id"]) != key:
                    continue
                cut = (c["run_id"], c["stage_id"], c["checkpoint_hash"])
                have = by_cut.get(cut)
                if have is None:
                    have = _row(c, precision=via["precision"], derivation=via["derivation"], via=mh)
                    have["paths"] = []
                    rows.append(have)
                    by_cut[cut] = have
                old = next((p for p in have["paths"] if p["via_model_hash"] == mh), None)
                if old is None:
                    have["paths"].append(dict(via))
                elif PRECISION_RANK.get(via["precision"], 3) < PRECISION_RANK.get(old["precision"], 3):
                    old.update(via)
                else:
                    continue
                best = _finest(have["paths"])
                have.update(precision=best["precision"], first_step=best["first_step"],
                            derivation=best["derivation"], via_model_hash=best["via_model_hash"],
                            answer=_answer(best["precision"]))
                changed = True
    for r in rows:
        r["paths"].sort(key=lambda p: (p["via_model_hash"] is not None, p["via_model_hash"] or ""))
    return rows
