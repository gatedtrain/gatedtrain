"""AN-7 and AN-17: the scorer library of the AI-native layer and pack scorers registered by reference.

The library has five scorers, one file per family (`exact.py`, `numeric.py`, `normalised.py`, `sets.py`); each file
imports only the standard library and `gatedtrain.canonical`, so its file hash is its code hash. Every library scorer
is higher-is-better on [0, 1] with the planning SD `PLANNING_SD_BOUND` = sqrt(0.5) of the paired difference (a [0, 1]
score's SD is at most 0.5, so two independent arms give at most sqrt(0.5^2 * 2)).

A scenario pack may also declare scorers (`Scenario.scorers`: `{name: {id, fn, code_sha256, planning_sd, direction,
range: [lo, hi], what, unit[, label_ok]}}`). They are registered by reference, not reimplemented: `resolve` refuses
(AN-17) an unknown name, a pack scorer named like a library scorer (or with a library id), and one whose
`code_sha256` is not the hash of the file that defines `fn`. A pack scorer without `label_ok` takes any JSON value.

A registration carries `evaluator_entry(spec, params)`: the evaluator id, the manifest hash (scorer, direction,
parameters and code hashes, so a changed tolerance or changed code changes the registered identity) and the
parameters when the scorer has any. `registration.resolved_evaluator_hash` resolves every library id and pack scorer
through `spec_for_evaluator`. Step 1 names are kept: `EVALUATOR_ID`, `DIRECTION`, `exact_match`, and
`manifest_hash()` with no arguments (exact match, no parameters).
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ...canonical import canonical_json, file_sha256, sha256_bytes
from ...errors import Problem, Refused
from . import exact, normalised, numeric, sets
from .exact import exact_match

__all__ = ["DIRECTION", "EVALUATOR_ID", "PLANNING_SD_BOUND", "ScorerSpec", "bind", "evaluator_entry", "exact_match",
           "library", "manifest_document", "manifest_hash", "resolve", "spec_for_evaluator"]

EVALUATOR_ID = "exact_match.ainative@0.1"
DIRECTION = "higher_is_better"
DIRECTIONS = ("higher_is_better", "lower_is_better")
PLANNING_SD_BOUND = math.sqrt(0.5)
LIBRARY_SOURCE = "library"
PACK_KEYS = frozenset({"id", "fn", "code_sha256", "planning_sd", "direction", "range", "what", "unit"})
PACK_OPTIONAL_KEYS = frozenset({"label_ok"})


@dataclass(frozen=True)
class ScorerSpec:
    """One scorer: `fn(prediction, label, **params) -> float`; `params` holds the defaults; `source` is "library" or
    the scenario pack name."""

    name: str
    evaluator_id: str
    fn: Callable[..., float]
    direction: str
    planning_sd: float
    score_range: tuple[float, float]
    what: str
    unit: str
    params: dict[str, float]
    label_ok: Callable[[Any], bool]
    code_files: tuple[Path, ...]
    source: str


def _lib(name: str, fn: Callable[..., float], module: Any, label_ok: Callable[[Any], bool], what: str, unit: str,
         params: dict[str, float] | None = None) -> ScorerSpec:
    return ScorerSpec(name, f"{name}.ainative@0.1", fn, DIRECTION, PLANNING_SD_BOUND, (0.0, 1.0), what, unit,
                      dict(params or {}), label_ok, (Path(module.__file__).resolve(),), LIBRARY_SOURCE)


_LIBRARY: dict[str, ScorerSpec] = {s.name: s for s in (
    _lib("exact_match", exact.exact_match, exact, exact.label_ok,
         "the share of items whose answer matches yours exactly", "share of items answered exactly (0 to 1)"),
    _lib("numeric_match", numeric.numeric_match, numeric, numeric.label_ok,
         "the share of items whose number is within the tolerance of yours",
         "share of items whose number is within the tolerance (0 to 1)", {"rel_tol": 0.0, "abs_tol": 0.0}),
    _lib("normalised_match", normalised.normalised_match, normalised, normalised.label_ok,
         "the share of items whose text matches yours after Unicode, case and whitespace normalisation",
         "share of items matching after normalisation (0 to 1)"),
    _lib("set_f1", sets.set_f1, sets, sets.set_label_ok,
         "the mean F1 between the predicted set of values and yours", "mean set F1 (0 to 1)"),
    _lib("field_f1", sets.field_f1, sets, sets.field_label_ok,
         "the mean F1 over the top-level fields of the predicted object and yours", "mean field F1 (0 to 1)"),
)}


def library() -> dict[str, ScorerSpec]:
    """The library scorers by name (a fresh dict; the specs are frozen)."""
    return dict(_LIBRARY)


def _refuse(field: str, message: str, fix: str) -> Refused:
    return Refused([Problem("AN-17", field, message, fix)])


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _params(spec: ScorerSpec, params: Any) -> dict[str, float]:
    """The scorer's defaults overlaid with `params`; refuses (AN-17) a non-object, an unknown key and a value that is
    not a finite number >= 0."""
    if params is None:
        params = {}
    if not isinstance(params, Mapping):
        raise _refuse("scorer_params", "the scorer parameters are not an object",
                      "give scorer_params as an object of numbers, or leave it out")
    out = dict(spec.params)
    allowed = ", ".join(sorted(spec.params)) or "no parameters"
    for key, value in params.items():
        if key not in spec.params:
            raise _refuse(f"scorer_params.{key}", f"scorer {spec.name} has no parameter {key!r}",
                          f"use only: {allowed}")
        if not (_num(value) and value >= 0):
            raise _refuse(f"scorer_params.{key}", f"parameter {key} is not a finite number >= 0",
                          "give a finite number >= 0")
        out[key] = float(value)
    return out


def bind(spec: ScorerSpec, params: dict) -> Callable[[Any, Any], float]:
    """`score(prediction, label)` with the parameters bound; a prediction that is not canonical JSON scores 0.0."""
    bound = _params(spec, params)
    fn = spec.fn

    def score(prediction: Any, label: Any) -> float:
        try:
            canonical_json(prediction)
        except ValueError:
            return 0.0
        return fn(prediction, label, **bound)

    return score


def manifest_document(spec: ScorerSpec, params: dict) -> dict:
    return {"evaluator": spec.evaluator_id, "scorer": spec.name, "direction": spec.direction, "version": 1,
            "params": _params(spec, params), "code": {f.name: file_sha256(f) for f in spec.code_files}}


def manifest_hash(spec: ScorerSpec | None = None, params: dict | None = None) -> str:
    """The evaluator identity a registration carries as `manifest_hash`. No arguments: exact match without
    parameters (the step 1 call)."""
    return sha256_bytes(canonical_json(manifest_document(spec or _LIBRARY["exact_match"], params or {})))


def evaluator_entry(spec: ScorerSpec, params: dict) -> dict:
    """The registration's `evaluators[0]`: `{id, manifest_hash}`, plus `params` when the scorer has any."""
    bound = _params(spec, params)
    entry: dict = {"id": spec.evaluator_id, "manifest_hash": manifest_hash(spec, bound)}
    if spec.params:
        entry["params"] = bound
    return entry


def _pack_scorers() -> Mapping[str, tuple[str, Mapping]]:
    from .. import trust_stages        # lazy: trust_stages imports this package

    return trust_stages.scenario_scorers()


def _pack_spec(name: str, pack: str, decl: Mapping) -> ScorerSpec:
    """The spec of a pack scorer declaration, refused (AN-17) when malformed, clashing with the library or with an
    ext evaluator row id, or when its code hash is not the hash of the file defining `fn`."""
    field = f"scenario pack {pack}: scorers.{name}"
    missing = sorted(PACK_KEYS - set(decl))
    extra = sorted(set(decl) - PACK_KEYS - PACK_OPTIONAL_KEYS)
    if missing or extra:
        said = [f"lacks {', '.join(missing)}"] if missing else []
        said += [f"has unknown keys {', '.join(extra)}"] if extra else []
        raise _refuse(field, f"the scorer declaration {'; '.join(said)}",
                      "declare id, fn, code_sha256, planning_sd, direction, range, what, unit and optionally label_ok")
    lib_ids = {s.evaluator_id for s in _LIBRARY.values()}
    if name in _LIBRARY or decl["id"] in lib_ids:
        raise _refuse(field, "the pack scorer has the name or id of a library scorer",
                      "give the pack scorer its own name and id")
    sid, fn, sd, rng = decl["id"], decl["fn"], decl["planning_sd"], decl["range"]
    if not (isinstance(sid, str) and sid):
        raise _refuse(f"{field}.id", "the evaluator id is not a non-empty string", "give an id such as name.pack@0.1")
    from ...ext import registry as ext_registry   # lazy: trust_stages imports this package lazily too
    if sid in ext_registry.EVALUATOR_ROWS:
        raise _refuse(field, f"the pack scorer id {sid!r} is the id of an ext evaluator row, which would resolve before it at "
                      "registration", "give the pack scorer its own id")
    if not callable(fn):
        raise _refuse(f"{field}.fn", "fn is not callable", "give the scoring function fn(prediction, label)")
    if not (_num(sd) and sd > 0):
        raise _refuse(f"{field}.planning_sd", "the planning SD is not a finite number > 0",
                      "give the planning SD of the paired per-item difference")
    if decl["direction"] not in DIRECTIONS:
        raise _refuse(f"{field}.direction", "the direction is not higher_is_better or lower_is_better",
                      "give higher_is_better or lower_is_better")
    if not (isinstance(rng, (list, tuple)) and len(rng) == 2 and all(_num(x) for x in rng) and rng[0] < rng[1]):
        raise _refuse(f"{field}.range", "the range is not [lo, hi] with finite lo < hi",
                      "give the score range [lo, hi]")
    for key in ("what", "unit"):
        if not (isinstance(decl[key], str) and decl[key].strip()):
            raise _refuse(f"{field}.{key}", f"{key} is not a non-empty text", f"describe the scorer in {key}")
    label_ok = decl.get("label_ok", exact.label_ok)
    if not callable(label_ok):
        raise _refuse(f"{field}.label_ok", "label_ok is not callable", "give label_ok(label) -> bool, or leave it out")
    try:
        src = inspect.getsourcefile(fn)
        actual = file_sha256(Path(src)) if src else None
    except (TypeError, OSError):
        src, actual = None, None
    if actual is None:
        raise _refuse(f"{field}.fn", "the source file of the scorer cannot be read, so its code cannot be hashed",
                      "define fn in a Python source file of the scenario pack")
    if actual != decl["code_sha256"]:
        raise _refuse(f"{field}.code_sha256", "the scorer's code hash differs from the declared code_sha256",
                      "update code_sha256 in the scenario pack to the hash of the file that defines fn, or restore "
                      "that file")
    return ScorerSpec(name, sid, fn, decl["direction"], float(sd), (float(rng[0]), float(rng[1])), decl["what"],
                      decl["unit"], {}, label_ok, (Path(src).resolve(),), pack)


def resolve(name: str) -> ScorerSpec:
    """The scorer named `name`: the library first, then the scorers of the installed scenario packs."""
    packs = _pack_scorers()
    if isinstance(name, str) and name in _LIBRARY:
        if name in packs:
            raise _refuse(f"scenario pack {packs[name][0]}: scorers.{name}",
                          "the pack scorer has the name of a library scorer", "give the pack scorer its own name")
        return _LIBRARY[name]
    if not (isinstance(name, str) and name in packs):
        known = sorted({*_LIBRARY, *packs})
        raise _refuse("scorer", f"{name!r} is neither a library scorer nor a scorer of an installed scenario pack",
                      f"use one of: {', '.join(known)}")
    pack, decl = packs[name]
    return _pack_spec(name, pack, decl)


def spec_for_evaluator(entry: dict) -> tuple[ScorerSpec, dict] | None:
    """(spec, its bound parameters) of a registration's evaluator entry, by id; None when no library or pack
    scorer has the id, when the parameters are refused, or when the pack scorer is refused."""
    if not isinstance(entry, dict):
        return None
    eid = entry.get("id")
    for spec in _LIBRARY.values():
        if spec.evaluator_id == eid:
            try:
                return spec, _params(spec, entry.get("params"))
            except Refused:
                return None
    if not isinstance(eid, str):
        return None
    for name, (pack, decl) in _pack_scorers().items():
        if decl.get("id") == eid:
            try:
                spec = _pack_spec(name, pack, decl)
                return spec, _params(spec, entry.get("params"))
            except Refused:
                return None
    return None
