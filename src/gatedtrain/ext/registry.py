"""The external-run adapter registry (MULTIMODAL-SPEC.md section 6 and MM-37; tests/README.md section 11.18).

The two section 6 tables are data: `TRAINER_ROWS` (training side, keyed by adapter name) and `EVALUATOR_ROWS`
(evaluation side, keyed by evaluator id), with `ADAPTERS` (adapter name -> "module:class"). They are filled at import
from the plugins of the `gatedtrain.adapters` entry point group: each names a module whose `PLUGIN` is
{"adapters": {name: "module:class"}, "trainer_rows": {name: row}, "evaluator_rows": {id: row}}. A recipe package
registers its adapter that way; two plugins naming the same adapter or evaluator is an error, as is an adapter
without a training row. Plugins load in entry point name order.
`items.EVALUATORS` reads the evaluator rows, so the registration checks (`registration.check_contents`, R11A-23) and
the gate agree with this table.

`evaluator_identity_hash(evaluator_id)` is the resolved hash a registration carries for an external
evaluator (EXT-6, design choice 4): the SHA-256 of the canonical JSON `{evaluator, metric, direction,
scorer}`. The score range is fixed by the id and left out.

This module imports no framework (MM-12).
"""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points
from typing import Any

from ..canonical import canonical_json, sha256_bytes
from .rowkit import CONTAINER_PATHS, HF_LLM_COMPONENTS   # noqa: F401  (re-exported: rows of older plugins)

# Registration disclosures (05 section 6.1; tests/README.md EXT-127, EXT-134, EXT-147). An evaluator row that declares
# `row_key` {<fact>, statement} needs, when it decides a gate, the registration field `field` (a non-empty text); a
# registration that gives the field for a deciding evaluator without that declaration is refused; ext-collect puts
# {<fact>, statement (the registration's), manifest_statement (the row's)} under `field` in the report, and the text
# report prints `<field>: ...` (and `<field>_manifest: ...` when the two differ). One mechanism, two named fields:
# `construct_alignment` (R15-1) predates it and keeps its name (area 3's registration is pinned by hash); the field
# names are design choices.
# Message parts: `declares` (the row's fact filled in),
# `missing` (what the registration then fails to state), `undeclared` (a statement given for an evaluator whose row
# declares none) and `what` (the fix for an empty statement).
DISCLOSURES = (
    {"field": "construct_alignment", "row_key": "construct_aligned", "fact": "reward", "rule": "05 section 6.1, R15-1",
     "declares": "construct-aligned with the training reward {}", "missing": "what a pass shows",
     "undeclared": "is not construct-aligned with a training reward (its registry row declares none)",
     "what": "state what a pass of the construct-aligned metric shows"},
    {"field": "shared_annotation", "row_key": "shared_annotation", "fact": "convention", "rule": "05 section 6.1, R16-1",
     "declares": "scored against held-out references of the training labels' annotation process ({})",
     "missing": "the shared annotation convention and what a pass shows",
     "undeclared": ("is not scored against references of the training labels' annotation process (its registry row "
                    "declares no `shared_annotation`)"),
     "what": "state the shared annotation convention and what a pass shows"},
)


def disclosure_problems(rows: dict[str, dict[str, Any]]) -> list[str]:
    """Check that each declared disclosure has a fact and a statement, both nonempty strings."""
    out = []
    for eid, row in rows.items():
        for d in DISCLOSURES:
            if d["row_key"] not in row:
                continue
            decl = row[d["row_key"]]
            if not (isinstance(decl, dict)
                    and all(isinstance(decl.get(k), str) and decl[k].strip() for k in (d["fact"], "statement"))):
                out.append(f"evaluator row {eid}: `{d['row_key']}` is not {{{d['fact']}: text, statement: text}}")
    return out


# The `gatedtrain.adapters` entry point group (MM-12: one pure-Python adapter class per framework, rows as data).
PLUGIN_GROUP = "gatedtrain.adapters"
TRAINER_ROWS: dict[str, dict[str, Any]] = {}
EVALUATOR_ROWS: dict[str, dict[str, Any]] = {}
ADAPTERS: dict[str, str] = {}


def register(plugin: dict[str, Any], source: str) -> None:
    """Merge one plugin's adapters and rows; ValueError on a name or id another plugin already registered."""
    for table, key in ((ADAPTERS, "adapters"), (TRAINER_ROWS, "trainer_rows"), (EVALUATOR_ROWS, "evaluator_rows")):
        for name, value in (plugin.get(key) or {}).items():
            if name in table:
                raise ValueError(f"plugin {source}: {key} {name!r} is already registered by another plugin")
            table[name] = value
    missing = sorted(set(plugin.get("adapters") or {}) - set(TRAINER_ROWS))
    if missing:
        raise ValueError(f"plugin {source}: adapters {missing} have no training row")


def _load_plugins() -> None:
    for ep in sorted(entry_points(group=PLUGIN_GROUP), key=lambda e: e.name):
        register(ep.load(), f"{ep.name} ({ep.value})")


_load_plugins()
_bad_disclosures = disclosure_problems(EVALUATOR_ROWS)
if _bad_disclosures:                                     # checked at import, before any command writes anything
    raise ValueError("; ".join(_bad_disclosures))


def adapter(name: str):
    """The adapter instance registered under `name`; KeyError when unknown."""
    if name not in ADAPTERS or name not in TRAINER_ROWS:
        raise KeyError(name)
    module, cls = ADAPTERS[name].split(":")
    return getattr(importlib.import_module(module), cls)()


def input_roots(name: str) -> tuple[str, ...]:
    """EXT-79: the adapter's allowed read-only input roots; KeyError when unknown."""
    return tuple(TRAINER_ROWS[name]["input_roots"])


def env_names(name: str) -> tuple[str, ...]:
    """EXT-92: the adapter's environment variable allow-list; KeyError when unknown."""
    return tuple(TRAINER_ROWS[name]["env_names"])


def scan_skip(name: str) -> tuple[str, ...]:
    """EXT-93: host directories the credential file-name scan skips (exact paths); KeyError when unknown."""
    return tuple(TRAINER_ROWS[name]["scan_skip"])


def interpreter_trees(name: str) -> tuple[str, ...]:
    """EXT-93: interpreter trees whose standard library `secrets` module is exempt; KeyError when unknown."""
    return tuple(TRAINER_ROWS[name]["interpreter_trees"])


def interpreter(name: str) -> str:
    """EXT-113: the adapter's interpreter, argv[0] exactly as the request writes it; KeyError when unknown."""
    return str(TRAINER_ROWS[name]["interpreter"])


def container_paths(name: str) -> dict[str, Any]:
    """EXT-112: the adapter's allowed container paths (`CONTAINER_PATHS` shape); KeyError when unknown."""
    return dict(TRAINER_ROWS[name]["container_paths"])


def env_paths(name: str) -> tuple[str, ...]:
    """EXT-114: the adapter's path-valued environment names (each value under /tmp or /out); KeyError when unknown."""
    return tuple(TRAINER_ROWS[name]["env_paths"])


def image_interpreters(name: str) -> dict[str, str]:
    """EXT-192: script file name -> the interpreter inside the pinned image that runs it (empty for the adapters whose
    interpreter is always a host file in a read-only mount); KeyError when the adapter is unknown."""
    return dict(TRAINER_ROWS[name].get("image_interpreters") or {})


def sidecar(name: str) -> dict[str, Any] | None:
    """EXT-190: the adapter's paired sidecar container rules, or None (no sidecar allowed); KeyError when unknown."""
    row = TRAINER_ROWS[name].get("sidecar")
    return dict(row) if row else None


def interpreter_flags(name: str) -> dict[str, str]:
    """EXT-199: script file name -> the one interpreter option allowed between the interpreter and that script."""
    return dict((TRAINER_ROWS.get(name) or {}).get("interpreter_flags") or {})


def b18(name: str) -> dict[str, Any] | None:
    """EXT-199: the executor-image command layout (scripts, tmpfs and limit options, environment names), or None."""
    row = (TRAINER_ROWS.get(name) or {}).get("b18")
    return dict(row) if row else None


def run_options(name: str, script: str) -> tuple[tuple[str, ...], str | None]:
    """EXT-191: (fixed docker options, tmpfs options or None) the printed command adds for `script` (argv[1]'s file
    name); ((), None) for the adapters without such rows or an unknown adapter."""
    row = TRAINER_ROWS.get(name) or {}
    return (tuple((row.get("run_options") or {}).get(script, ())),
            (row.get("tmpfs_options") or {}).get(script))


def evaluator_identity_hash(evaluator_id: str) -> str:
    """EXT-6: SHA-256 of the canonical JSON {evaluator, metric, direction, scorer}; KeyError when unknown."""
    row = EVALUATOR_ROWS[evaluator_id]
    return sha256_bytes(canonical_json({"evaluator": evaluator_id, "metric": row["metric"],
                                        "direction": row["direction"], "scorer": dict(row["scorer"])}))


def trainer_ids() -> dict[str, str]:
    """Spec `uses` of every registered external trainer -> adapter name (MM-13)."""
    return {uses: name for name, row in TRAINER_ROWS.items() for uses in row["uses"]}
