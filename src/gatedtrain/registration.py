"""Gate pre-registration (05 section 3.7, 05-evaluation-and-lineage.md:142).

The registration is a YAML file named by a spec artifact of `type:
registration` whose URI carries `@sha256:<hex>` (interim rule G-2). The run records its SHA-256 and refuses
when the file is missing (REG-MISSING), when its hash differs from the one
the spec recorded ("a mismatching registration hash fails validation",
05:142; REG-HASH), when a blocking clause has no margin (05:82; REG-MARGIN),
when a field is missing or unusable (REG-FIELD), or when an evaluator's `scorer`
is not on the allowlist (REG-SCORER, interim rule G-21).

Design revision 10 (tests/README.md section 11.15): `evaluators[0].manifest_hash` must be resolved
(05:151, R10A-2; the fix names the hash the prototype resolves), and the optional fields
`pairing.max_unpaired` (count or fraction, 05:116, R10A-3), `amends` and `post_hoc` (05:151, R10A-4)
and `min_paired` (05:145, R10A-5) are checked.

Design revision 11 (tests/README.md section 11.16; R11-19, R11-10, R11-21): the clause is written in item
names (`unit: item`, `samples_per_item`, `direction`, optional `estimator`) and the top-level
`eval_distribution: {item_set_hash, env_version, env_params_version}` replaces `prompt_set_hash`. The
revision 10 names stay readable field by field (R11A-2): `prompt_set_hash`, `samples_per_prompt` and
`unit: prompt` are aliases, a clause in the legacy K form may omit `direction` (read as
`higher_is_better`), and a document giving both forms of one field is refused on the legacy key.
`normalize` returns a copy in revision 11 names; the registration hash stays the SHA-256 of the file
bytes. The direction, score range and estimators each prototype evaluator declares come from
`items.EVALUATORS` (R11A-1). Design revision 13 (R13-7): an optional `created_at` (RFC 3339, UTC) records
when the registration was written, for the `post_hoc` comparison (`gates.is_post_hoc`).

Design revisions 12 and 13 (tests/README.md section 11.17; R12-1 to R12-5, R13-1 to R13-4): a clause with
`constraint_set` is a constraint clause. It takes `bar` (threshold form, `type: superiority`) or `margin`
(non-inferiority), `interval: clopper_pearson` or `clopper_pearson_discordant`, and no sigma plan (neither
`sigma_plan` nor `sigma_pilot` is required). The registration then holds `constraint_set` and, when the set
has judges, `judge_qualification`; `constraints.check_registration` checks them (REG-CONSTRAINT, REG-JUDGE,
REG-EGRESS). A spec `gate.threshold` maps to `superiority` and `test.constraint_set` to the clause field (G-4).

The deciding evaluator is the entry whose
`id` is the spec gate's `evaluator` (`deciding_evaluator`), not `evaluators[0]`; every evaluator's
`manifest_hash` must be resolved; an optional `baseline_pilot_mean` (R13-9) must lie strictly inside the
deciding evaluator's declared range; `created_at` is required with `amends`.

Interim rule G-4: the registration is authoritative. A value the spec's gate
also states (type, metric, alpha, power, mde -> planning_effect, margin,
cluster_by -> unit, seeds -> len(eval_seeds); 04:98) that differs from the
registration is refused with REG-FIELD. The same holds for the gate's
sampler and evaluator, which the registration also records (05:142).

Design revision 18 (05 sections 3.1, 3.4, 3.5, 3.7; R18-1 to R18-5): a registration may hold several blocking clauses
in `clauses` instead of the one `clause`. Each entry has the 05 section 3.1 fields plus `name` (unique) and
`evaluator` (the id of a listed evaluator, one evaluator per clause, which decides that clause); `multiplicity` is then
`{plan: all_clauses_must_pass}` (intersection-union, each clause at its own alpha); `sigma_pilot` and an optional
`baseline_pilot_mean` are keyed by clause name. Any clause may set `cluster_key` (an item field: clusters of items are
resampled) and `bootstrap_variant` (`bca`, the default, or `percentile`). The singular `clause` form is unchanged:
a registration without these fields validates exactly as before.
"""

from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import yaml

from . import ANALYSIS_CODE_VERSION
from .canonical import canonical_json, sha256_bytes
from .errors import Problem
from .gates import planning_distance, planning_sigma
from .renamed import (PRE_RENAME_VERSIONS, REGISTRATION_AMENDMENT, VERSION_ONLY_FIELDS, apply_name_map,
                      is_old_analysis_code_version)
from .items import DIRECTIONS, ESTIMATORS, EVALUATORS
from .ext.registry import DISCLOSURES as _DISCLOSURES

SHA_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

# 05 section 3.1 clause fields (05:71-86), revision 11 names. `direction` is required for a clause in the
# revision 11 K form (`samples_per_item`); `estimator` is optional (absent = mean; design choice).
CLAUSE_FIELDS = ("type", "metric", "direction", "margin", "alpha", "power", "planning_effect", "sigma_plan", "unit",
                 "samples_per_item", "eval_seeds", "interval", "blocking")
OPTIONAL_CLAUSE_FIELDS = ("estimator", "constraint_set", "bar",    # the last two: constraint clauses (R12-1)
                          "cluster_key", "bootstrap_variant")      # R18-3, R18-4
# Design revision 18 (R18-1): an entry of `clauses` also names itself and its deciding evaluator; it takes no constraint
# fields (constraint clauses keep the singular form) and no legacy names.
MULTI_CLAUSE_FIELDS = ("name", "evaluator")
MULTI_OPTIONAL_CLAUSE_FIELDS = ("estimator", "cluster_key", "bootstrap_variant")
MULTI_PLAN = "all_clauses_must_pass"                                 # R18-2
CLAUSE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
BOOTSTRAP_VARIANTS = ("bca", "percentile")                           # R18-4
# Legacy (revision 10) clause key -> revision 11 key (R11A-2).
LEGACY_CLAUSE_KEYS = {"samples_per_prompt": "samples_per_item"}
# 05 section 3.7 registration contents (05:166), keyed as in tests/README.md section 4.2; `eval_distribution`
# may be given as the legacy `prompt_set_hash` (R11A-2).
REGISTRATION_KEYS = ("registration_version", "registration_id", "signed_by", "workflow", "gate", "clause",
                     "margin_justification", "sigma_pilot", "eval_distribution", "sampler", "evaluators",
                     "multiplicity", "analysis_code_version", "bootstrap")
LEGACY_KEYS = {"prompt_set_hash": "eval_distribution"}
EVAL_DISTRIBUTION_KEYS = ("item_set_hash", "env_version", "env_params_version")
UNITS = {"item": "item", "prompt": "item"}          # `prompt` is the legacy alias of `item` (R11A-2)
# 05 section 3.7 (05:151) optional registration fields: `pairing: {max_unpaired}` (05 section 3.4, default 0),
# `min_paired` (05 section 3.6, default 10), `amends` (the prior registration's hash), `post_hoc`, and
# `created_at` (R13-7, required with `amends`, R11A-29), and `baseline_pilot_mean` (R13-9; name and
# optionality: design choice, R11A-29).
OPTIONAL_KEYS = ("pairing", "min_paired", "amends", "post_hoc", "created_at", "baseline_pilot_mean", "constraint_set",
                 "judge_qualification", "contamination", "clauses", "base_model",
                 "rename_proof") + tuple(d["field"] for d in _DISCLOSURES)   # rename_proof: R17-3
# Design revision 15 (05 section 6.1, R15-1; tests/README.md section 11.20, EXT-127): a construct-aligned deciding
# evaluator (its registry row declares `construct_aligned`) needs the registration to state what a pass shows, in
# `construct_alignment` (a non-empty text; name: design choice). Design revision 16 (05 section
# 6.1, R16-1; tests/README.md section 11.21, EXT-147): a deciding evaluator whose references share the training labels'
# annotation process (its row declares `shared_annotation`) needs `shared_annotation`, the same way. Both are entries of
# `ext.registry.DISCLOSURES`, checked by one loop.
# MULTIMODAL-SPEC MM-11 (tests/README.md section 11.18, EXT-21; design choice 13): `contamination:
# {allow_exact_overlap: true}` lets ext-collect accept an exact text overlap between the item set and the training
# data; absent means false. The overlap count is reported either way.
CONTAMINATION_KEYS = ("allow_exact_overlap",)
DEFAULT_MIN_PAIRED = 10
DEFAULT_MAX_UNPAIRED = 0
# Interim rule G-21: scorers an evaluator's `scorer` may name; `numeric_match` is in NeMo Evaluator's
# scorer registry (scoring/__init__.py:38-44 at ea35ba3).
SCORER_ALLOWLIST = ("numeric_match",)
# Decision D-10: only the Tier B evaluator (tierb.EVALUATOR_ID) reads NeMo Evaluator scorer results.
SCORER_EVALUATOR = "gsm8k.nemo_evaluator@0.1"

# 04 section 4.2 (04:98): workflow gate type -> 05 clause type.
GATE_TO_CLAUSE = {"gate.superiority": "superiority", "gate.noninferiority": "non_inferiority",
                  "gate.equivalence": "equivalence"}
# 04 section 4.2 (R12-4): `gate.threshold` is a superiority clause against `bar`. Checked against the
# registration only; `gatedtrain run` still accepts the three gate types above.
SPEC_GATE_TO_CLAUSE = {**GATE_TO_CLAUSE, "gate.threshold": "superiority"}


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def resolved_evaluator_hash(evaluator: dict) -> str | None:
    """The evaluator hash the prototype resolves at run time (R10A-2): Tier B, the evaluator identity
    hash (decision D-4, scorer included); Tier A, the stand-in identity hash. None for another id."""
    if evaluator.get("id") == SCORER_EVALUATOR:
        from .tierb import evaluator_identity_hash

        scorer = evaluator.get("scorer")
        return evaluator_identity_hash(scorer if isinstance(scorer, str) else None)
    from .ainative import scorers as ainative_scorers

    if evaluator.get("id") == ainative_scorers.EVALUATOR_ID:
        return ainative_scorers.manifest_hash()
    from .ext.registry import EVALUATOR_ROWS, evaluator_identity_hash as ext_identity_hash

    if evaluator.get("id") in EVALUATOR_ROWS:
        # External-run evaluators (tests/README.md section 11.18, EXT-6): {evaluator, metric, direction, scorer}.
        return ext_identity_hash(evaluator["id"])
    # Last, so external-run evaluators resolve without the Tier A extra (the stub imports torch).
    from .stubs import evaluator as stub_evaluator

    if evaluator.get("id") == stub_evaluator.EVALUATOR_ID:
        from .runner import stub_evaluator_hash

        return stub_evaluator_hash()
    return None


def deciding_evaluator(reg: dict, evaluator_id: str | None) -> dict | None:
    """The registration entry of `evaluators` whose `id` equals the spec gate's `evaluator` (R11A-23): it
    decides direction, estimator, score range, scorer and the run-time hash check. None when absent."""
    evaluators = reg.get("evaluators") if isinstance(reg, dict) else None
    if not isinstance(evaluators, list) or evaluator_id is None:
        return None
    for e in evaluators:
        if isinstance(e, dict) and e.get("id") == evaluator_id:
            return e
    return None


def _deciding_index(reg: dict, evaluator_id: str | None) -> int | None:
    evaluators = reg.get("evaluators")
    if not isinstance(evaluators, list):
        return None
    for i, e in enumerate(evaluators):
        if isinstance(e, dict) and e.get("id") == evaluator_id:
            return i
    return None


def normalize(reg: dict) -> dict:
    """A copy of `reg` in revision 11 names (R11A-2); the input is not changed.

    `prompt_set_hash` becomes `eval_distribution: {item_set_hash, env_version: null, env_params_version:
    null}`, `samples_per_prompt` becomes `samples_per_item`, `unit: prompt` becomes `item`, and a clause
    without `direction` gets `higher_is_better`. Absent env fields of `eval_distribution` read as null.
    Run on a validated registration (a document with both forms of a field is refused before)."""
    out: dict = {}
    for key, value in reg.items():
        if key == "prompt_set_hash":
            if "eval_distribution" not in reg:
                out["eval_distribution"] = {"item_set_hash": value, "env_version": None, "env_params_version": None}
        elif key == "eval_distribution" and isinstance(value, dict):
            out[key] = {k: copy.deepcopy(value.get(k)) for k in EVAL_DISTRIBUTION_KEYS}
            out[key].update({k: copy.deepcopy(v) for k, v in value.items() if k not in EVAL_DISTRIBUTION_KEYS})
        elif key == "clause" and isinstance(value, dict):
            out[key] = normalize_clause(value)
        elif key == "clauses" and isinstance(value, list):
            out[key] = [normalize_clause(c) if isinstance(c, dict) else copy.deepcopy(c) for c in value]
        else:
            out[key] = copy.deepcopy(value)
    return out


def normalize_clause(clause: dict) -> dict:
    """A copy of one clause in revision 11 names (see `normalize`)."""
    c: dict = {}
    for key, value in clause.items():
        if key in LEGACY_CLAUSE_KEYS:
            if LEGACY_CLAUSE_KEYS[key] not in clause:
                c[LEGACY_CLAUSE_KEYS[key]] = copy.deepcopy(value)
        elif key == "unit":
            c[key] = UNITS.get(value, value) if isinstance(value, str) else value
        else:
            c[key] = copy.deepcopy(value)
        if key == "metric" and "direction" not in clause:
            c["direction"] = "higher_is_better"
    if "direction" not in c:
        c["direction"] = "higher_is_better"
    return c


def is_multi(reg: dict) -> bool:
    """A registration in the revision 18 form: several clauses in `clauses` (R18-1)."""
    return isinstance(reg, dict) and "clauses" in reg and "clause" not in reg


def clause_list(reg: dict) -> list[dict]:
    """The registration's clauses in revision 11 names: `clauses` (R18-1), or the one `clause` as a list of one."""
    if is_multi(reg):
        return [normalize_clause(c) for c in reg["clauses"]]
    return [normalize_clause(reg["clause"])]


def clause_for_evaluator(reg: dict, evaluator_id: str | None) -> dict | None:
    """The clause `evaluator_id` decides, in revision 11 names: in the revision 18 form the entry of `clauses` whose
    `evaluator` it is (None when no clause names it); otherwise the one `clause`."""
    if not is_multi(reg):
        return normalize_clause(reg["clause"]) if isinstance(reg.get("clause"), dict) else None
    for c in reg.get("clauses") or []:
        if isinstance(c, dict) and c.get("evaluator") == evaluator_id:
            return normalize_clause(c)
    return None


def pilot_for(reg: dict, clause: dict) -> Any:
    """The sigma pilot of `clause`: `sigma_pilot`, or in the revision 18 form `sigma_pilot[<clause name>]` (R18-1)."""
    pilot = reg.get("sigma_pilot")
    if is_multi(reg):
        return pilot.get(clause.get("name")) if isinstance(pilot, dict) else None
    return pilot


def item_set_hash(reg: dict) -> str | None:
    """The registered item-set hash, from `eval_distribution` or the legacy `prompt_set_hash`."""
    dist = reg.get("eval_distribution")
    if isinstance(dist, dict):
        return dist.get("item_set_hash")
    return reg.get("prompt_set_hash")


def item_set_field(reg: dict) -> str:
    """The error field that names the registered item-set hash in this document's own form."""
    return "registration.prompt_set_hash" if "prompt_set_hash" in reg else "registration.eval_distribution.item_set_hash"


def min_paired(reg: dict) -> int:
    """`min_paired` (05 section 3.6, 05:145); absent means 10."""
    return int(reg.get("min_paired", DEFAULT_MIN_PAIRED))


def max_unpaired(reg: dict) -> int | float:
    """`pairing.max_unpaired` (05 section 3.4, 05:116): a count or a fraction; absent means 0."""
    return (reg.get("pairing") or {}).get("max_unpaired", DEFAULT_MAX_UNPAIRED)


def read_registration(path: Path, *, uri_field: str, expected_hex: str,
                      evaluator_id: str | None = None) -> tuple[dict | None, str | None, list[Problem]]:
    """Read the file, check the recorded hash, parse it. Returns (registration, sha256 id, problems).
    `evaluator_id` is the spec gate's evaluator (the deciding evaluator, R11A-23; see `check_contents`)."""
    path = Path(path)
    if not path.is_file():
        return None, None, [Problem("REG-MISSING", uri_field, f"registration file {path} does not exist",
                                    "write the registration before the run (05 section 3.7) or fix the uri")]
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_hex:
        return None, "sha256:" + actual, [Problem(
            "REG-HASH", uri_field,
            f"registration hash mismatch: the spec recorded sha256:{expected_hex} but the file is "
            f"sha256:{actual}; the registration changed after the spec recorded its hash",
            "an amendment is a new registration (05 section 3.7): review the change, then put the new "
            "sha256 in the spec's registration uri")]
    try:
        reg = yaml.safe_load(data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        return None, "sha256:" + actual, [Problem("REG-FIELD", "registration", f"cannot parse the registration: {exc}",
                                                  "write the registration as a YAML mapping")]
    if not isinstance(reg, dict):
        return None, "sha256:" + actual, [Problem("REG-FIELD", "registration", "the registration is not a mapping",
                                                  "write the registration as a YAML mapping")]
    _, amendment = check_amendment(reg, path)
    return reg, "sha256:" + actual, check_contents(reg, evaluator_id=evaluator_id) + amendment


VERSION_ONLY = "version_only"


def check_amendment(reg: dict, path: Path) -> tuple[str | None, list[Problem]]:
    """05 section 3.7 (R17-1): the kind of amendment `reg` (read from `path`) is, and its problems.

    The prior registration is the file next to `path` whose SHA-256 is `amends`. When the prior records a
    pre-rename analysis code version (`renamed.PRE_RENAME_VERSIONS`) and `reg` this code's version, the amendment
    crosses the rename: if every other top-level field is equal (only `VERSION_ONLY_FIELDS` differ) it is a
    version-only amendment, returned as `VERSION_ONLY` and not post hoc; if anything else changed it follows the
    general rule, and `post_hoc: false` is refused (REG-FIELD). Either kind keeps a prior's `post_hoc: true`
    (R19-1): declaring it false is refused (REG-FIELD). Any other amendment, or one whose prior is not found,
    keeps its declared `post_hoc` and returns None.

    R17-3: `evaluators` may also differ, in `manifest_hash` only, when `rename_proof` proves each changed hash is
    a scorer rename (`scorer_rename_problems`)."""
    amends = reg.get("amends")
    if not (isinstance(amends, str) and SHA_RE.match(amends)):
        return None, []
    prior = None
    for cand in sorted(path.parent.iterdir()):
        if cand != path and cand.is_file() and cand.suffix in (".yaml", ".yml") \
                and "sha256:" + hashlib.sha256(cand.read_bytes()).hexdigest() == amends:
            try:
                prior = yaml.safe_load(cand.read_bytes().decode("utf-8"))
            except (UnicodeDecodeError, yaml.YAMLError):
                prior = None
            break
    if not (isinstance(prior, dict) and prior.get("analysis_code_version") in PRE_RENAME_VERSIONS
            and reg.get("analysis_code_version") == ANALYSIS_CODE_VERSION):
        return None, []
    changed = sorted(k for k in set(prior) | set(reg)
                     if k not in VERSION_ONLY_FIELDS and prior.get(k, _ABSENT) != reg.get(k, _ABSENT))
    unproved: list[str] = []
    if "evaluators" in changed:
        unproved = scorer_rename_problems(prior, reg, path.parent)
        if not unproved:
            changed.remove("evaluators")
    if prior.get("post_hoc") is True and reg.get("post_hoc") is not True:
        return None, [Problem(
            "REG-FIELD", "registration.post_hoc",
            f"the amendment of {amends} declares post_hoc false but the prior registration is post_hoc: true; an "
            "amendment across the rename keeps the prior's post_hoc (05 section 3.7, R19-1)",
            "set `post_hoc: true`, as in the prior registration (a PASS is then reported INCONCLUSIVE)")]
    if not changed:
        return VERSION_ONLY, []
    if reg.get("post_hoc") is True:
        return None, []
    return None, [Problem(
        "REG-FIELD", "registration.post_hoc",
        f"the amendment of {amends} changes {', '.join(changed)} besides analysis_code_version, so it is not a "
        "version-only amendment and post_hoc: false is not established (05 section 3.7, R17-1"
        + (f"; no scorer rename proof (R17-3): {'; '.join(unproved)})" if unproved else ")"),
        "set `post_hoc: true` (a PASS is then reported INCONCLUSIVE), or amend only `analysis_code_version` "
        "(with `amends`, `created_at`, `post_hoc`) and make every other field equal to the prior registration")]


_ABSENT = object()


def _scorer_code_sha256(blobs: dict[str, bytes], composite: bool) -> str:
    """A scorer's `code_sha256` from its file bytes: the file's SHA-256, or for a composite the SHA-256 of the
    compact sorted JSON {file name: its SHA-256} (the form the recipes' scorers write)."""
    if not composite:
        (data,) = blobs.values()
        return sha256_bytes(data)
    doc = {name: sha256_bytes(data) for name, data in blobs.items()}
    return sha256_bytes(json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def _without_hash(entry: dict) -> dict:
    return {k: v for k, v in entry.items() if k != "manifest_hash"}


def scorer_rename_problems(prior: dict, reg: dict, base: Path) -> list[str]:
    """05 section 3.7, R17-3: why the `evaluators` change from `prior` to `reg` is not a scorer rename ([] when
    it is). Every entry must be equal except `manifest_hash`; for each changed hash, `reg["rename_proof"]` has an
    entry `{evaluator: <id>, code_sha256_before: sha256:<hex>, files: [<path>, ...], composite: <bool>}` (paths
    relative to the registration's directory: the pre-rename scorer file(s)) such that
    - the evaluator row's identity with the scorer's `code_sha256` set to `code_sha256_before` is the prior hash,
      and its identity now is the amended hash (so only the scorer's `code_sha256` changed);
    - the files' code hash is `code_sha256_before`, and the same files with `renamed.NAME_MAP` applied give the
      row's `code_sha256` now."""
    from .ext.registry import EVALUATOR_ROWS

    before, after = prior.get("evaluators"), reg.get("evaluators")
    if not (isinstance(before, list) and isinstance(after, list) and len(before) == len(after)
            and all(isinstance(e, dict) for e in before + after)):
        return ["the evaluator lists differ in length or shape"]
    proofs = reg.get("rename_proof")
    if not isinstance(proofs, list):
        return ["`rename_proof` is missing or not a list"]
    out: list[str] = []
    for b, a in zip(before, after):
        if _without_hash(b) != _without_hash(a):
            out.append(f"evaluator {a.get('id')!r} changes more than manifest_hash")
            continue
        if b.get("manifest_hash") == a.get("manifest_hash"):
            continue
        eid = a.get("id")
        entry = [p for p in proofs if isinstance(p, dict) and p.get("evaluator") == eid]
        row = EVALUATOR_ROWS.get(eid)
        if len(entry) != 1:
            out.append(f"no single rename_proof entry for evaluator {eid!r}")
            continue
        if row is None or not isinstance(row.get("scorer"), dict):
            out.append(f"evaluator {eid!r} has no registered scorer row")
            continue
        entry = entry[0]
        old_code, files, composite = entry.get("code_sha256_before"), entry.get("files"), entry.get("composite")
        if not (isinstance(old_code, str) and SHA_RE.match(old_code) and isinstance(files, list) and files
                and all(isinstance(f, str) for f in files) and isinstance(composite, bool)
                and (composite or len(files) == 1)):
            out.append(f"rename_proof for {eid!r} is not {{evaluator, code_sha256_before, files, composite}}")
            continue

        def identity(code: str) -> str:
            return sha256_bytes(canonical_json({"evaluator": eid, "metric": row["metric"],
                                                "direction": row["direction"],
                                                "scorer": {**row["scorer"], "code_sha256": code}}))
        if identity(old_code) != b.get("manifest_hash"):
            out.append(f"evaluator {eid!r}: the row with code_sha256 {old_code} is not the prior manifest_hash "
                       "(something besides the scorer code changed)")
            continue
        if identity(row["scorer"]["code_sha256"]) != a.get("manifest_hash"):
            out.append(f"evaluator {eid!r}: the amended manifest_hash is not the row's identity now")
            continue
        blobs = {}
        for f in files:
            fp = base / f
            if not fp.is_file():
                out.append(f"evaluator {eid!r}: pre-rename scorer file {f} not found next to the registration")
                break
            blobs[Path(f).name] = fp.read_bytes()
        else:
            if len(blobs) != len(files):
                out.append(f"evaluator {eid!r}: rename_proof names two files with one name")
            elif _scorer_code_sha256(blobs, composite) != old_code:
                out.append(f"evaluator {eid!r}: the pre-rename files do not hash to {old_code}")
            elif _scorer_code_sha256({n: apply_name_map(d) for n, d in blobs.items()},
                                     composite) != row["scorer"]["code_sha256"]:
                out.append(f"evaluator {eid!r}: the pre-rename files with the name map applied do not give the "
                           f"scorer the row pins ({row['scorer']['code_sha256']}): the scorer changed beyond the "
                           "rename")
    return out


def check_contents(reg: dict, *, evaluator_id: str | None = None) -> list[Problem]:
    """05 section 3.7 contents and the 05 section 3.1 clause.

    `evaluator_id` is the spec gate's `evaluator`: the deciding evaluator is the registration entry with that
    id (R11A-23), and its declaration in `items.EVALUATORS` checks direction, estimator and
    `baseline_pilot_mean`. A gate evaluator the registration does not list is refused by
    `check_against_gate`; one the prototype table does not declare is refused here (design choice:
    its direction cannot be checked). Without `evaluator_id` (library callers that hold no spec gate)
    the first entry stands in, and an undeclared one is not refused."""
    from .constraints import check_registration, clause_form, is_constraint_clause

    p: list[Problem] = []

    def bad(field: str, message: str, fix: str, rule: str = "REG-FIELD") -> None:
        p.append(Problem(rule, f"registration.{field}", message, fix))

    constraint = is_constraint_clause(reg.get("clause"))
    multi = is_multi(reg)                      # R18-1: `clauses` instead of `clause`
    if "clauses" in reg and "clause" in reg:
        bad("clauses", "both `clause` and `clauses` are given (05 section 3.7, R18-1)",
            "keep one form: `clause` for one clause, `clauses` for several")
    for key in REGISTRATION_KEYS:
        if key == "sigma_pilot" and constraint:
            continue                           # binomial power and n (05 section 3.9; P2)
        if key == "clause" and multi:
            continue
        legacy = [old for old, new in LEGACY_KEYS.items() if new == key]
        if key not in reg and not any(old in reg for old in legacy):
            bad(key, "missing registration field (05 section 3.7)", f"add `{key}` to the registration")
    for old, new in LEGACY_KEYS.items():
        if old in reg and new in reg:
            bad(old, f"both `{new}` and its legacy alias `{old}` are given (R11-19)",
                f"remove `{old}`; keep `{new}`")
    for key in sorted(set(reg) - set(REGISTRATION_KEYS) - set(OPTIONAL_KEYS) - set(LEGACY_KEYS)):
        bad(key, "unknown registration field", f"remove `{key}`; allowed: {', '.join(REGISTRATION_KEYS)}")

    clause = reg.get("clause")
    if "clause" in reg and not isinstance(clause, dict):
        bad("clause", "the clause is not a mapping", "write the 05 section 3.1 clause fields")
        clause = None
    evaluators = reg.get("evaluators")
    if multi:
        p.extend(_check_multi(reg))
        evaluator_id = None                    # each clause names its own deciding evaluator (R18-1)
    gate_given = evaluator_id is not None
    if not gate_given:
        evaluator_id = (evaluators[0].get("id") if isinstance(evaluators, list) and evaluators
                        and isinstance(evaluators[0], dict) else None)
    deciding_index = None if multi else _deciding_index(reg, evaluator_id)
    if (gate_given and deciding_index is not None and not constraint and isinstance(evaluator_id, str)
            and evaluator_id not in EVALUATORS):
        bad(f"evaluators[{deciding_index}].id",
            f"the deciding evaluator {evaluator_id!r} declares no direction, score range or estimators the "
            "prototype knows (items.EVALUATORS), so the clause cannot be checked against it (05 section 3.1)",
            f"gate on one of {', '.join(sorted(EVALUATORS))}")
    if deciding_index is None:
        evaluator_id = None                    # not registered: check_against_gate refuses it
    if isinstance(clause, dict):
        blocking = clause.get("blocking")
        legacy_k = "samples_per_prompt" in clause
        for f in CLAUSE_FIELDS:
            if f == "samples_per_item" and legacy_k:
                continue
            if f == "direction" and legacy_k and "direction" not in clause:
                continue                       # a legacy clause reads as higher_is_better (R11A-2)
            if constraint and (f == "sigma_plan" or (f == "margin" and clause_form(clause) == "threshold")):
                continue                       # constraint clause: no sigma plan (P2); threshold uses `bar` (P1)
            if f not in clause or clause[f] is None:
                if f == "margin" and blocking is True:
                    bad("clause.margin", "a blocking gate declares no margin (05 section 3.1)",
                        "declare `margin` (delta, or delta_sup = 0 for superiority) and justify it",
                        rule="REG-MARGIN")
                elif f == "direction":
                    bad("clause.direction", "the clause states no direction (05 section 3.1, R11-10)",
                        "set `direction: higher_is_better` or `lower_is_better`, as the evaluator declares")
                else:
                    bad(f"clause.{f}", "missing clause field (05 section 3.1)", f"add `{f}` to the clause")
        for old, new in LEGACY_CLAUSE_KEYS.items():
            if old in clause and new in clause:
                bad(f"clause.{old}", f"both `{new}` and its legacy alias `{old}` are given (R11-19)",
                    f"remove `{old}`; keep `{new}`")
        for f in sorted(set(clause) - set(CLAUSE_FIELDS) - set(OPTIONAL_CLAUSE_FIELDS) - set(LEGACY_CLAUSE_KEYS)):
            bad(f"clause.{f}", "unknown clause field (05 section 3.1)", f"remove `{f}`")
        if "bar" in clause and not constraint:
            bad("clause.bar", "`bar` is read only in a constraint clause in the skeleton (05 section 3.9)",
                "remove `bar`, or name the constraint set in `constraint_set`")
        _check_clause(clause, bad, evaluator_id, constraint=constraint)
    p.extend(check_registration(reg))

    pilot = reg.get("sigma_pilot")
    if isinstance(clause, dict) and "sigma_pilot" in reg:
        _check_pilot(pilot, clause.get("sigma_plan"), "sigma_pilot", bad)

    if "prompt_set_hash" in reg and not (isinstance(reg["prompt_set_hash"], str) and SHA_RE.match(reg["prompt_set_hash"])):
        bad("prompt_set_hash", "not a sha256:<64 hex> manifest hash", "record the item-set manifest hash")
    if "eval_distribution" in reg:
        _check_eval_distribution(reg["eval_distribution"], bad)
    if "base_model" in reg:                  # AI-native AN-7: the registered base model reference (05 section 18)
        bm = reg["base_model"]
        h = bm.get("model_ref_sha256") if isinstance(bm, dict) else None
        extra = set(bm) - {"model_ref_sha256"} if isinstance(bm, dict) else set()
        served = extra == {"scenario_pack", "generator"} and all(isinstance(bm[k], str) and bm[k] for k in extra)
        if not (isinstance(bm, dict) and "model_ref_sha256" in bm and (not extra or served)
                and isinstance(h, str) and SHA_RE.match(h)):
            bad("base_model", "base_model is not {model_ref_sha256: sha256:<64 hex>} (with optional scenario_pack and "
                "generator names, both or neither)",
                "give `base_model: {model_ref_sha256: sha256:<64 hex>}` or remove it")
    if "created_at" in reg and _utc_time(reg["created_at"]) is None:
        bad("created_at", f"created_at {reg['created_at']!r} is not an RFC 3339 UTC time (05 section 3.7, R13-7)",
            "set `created_at` to the registration's creation time in RFC 3339 UTC format")
    # 05 section 6.1 disclosures (ext.registry.DISCLOSURES): R15-1 `construct_alignment` (EXT-127, EXT-134) and R16-1
    # `shared_annotation` (EXT-147), each required for a deciding evaluator whose registry row declares it and refused
    # for one that does not.
    from .ext.registry import DISCLOSURES as _disclosures, EVALUATOR_ROWS as _rows
    for d in (() if multi else _disclosures):  # the revision 18 form: `_check_multi`
        field = d["field"]
        decl = (_rows.get(evaluator_id) or {}).get(d["row_key"]) if isinstance(evaluator_id, str) else None
        statement = reg.get(field)
        if field in reg and not (isinstance(statement, str) and statement.strip()):
            bad(field, f"{field} is not a non-empty text", f"{d['what']} ({d['rule']})")
        elif field in reg and isinstance(evaluator_id, str) and not decl:
            bad(field, f"the deciding evaluator {evaluator_id} {d['undeclared']}, so the statement would claim what "
                f"the gate does not show ({d['rule']})", f"remove `{field}`")
        elif decl and field not in reg:
            bad(field, f"the deciding evaluator {evaluator_id} is {d['declares'].format(decl[d['fact']])} "
                f"({d['rule']}), but the registration does not state {d['missing']}",
                f"add `{field}:` stating, for example: {decl['statement']}")
    sampler = reg.get("sampler")
    # EXT-106 (tests/README.md section 11.19): an external evaluator whose registry row declares `sampler_keys`
    # (a deterministic control policy has no temperature or token budget) registers exactly those keys.
    from .ext.registry import EVALUATOR_ROWS as _ext_rows
    policy_keys = (_ext_rows.get(evaluator_id) or {}).get("sampler_keys") if isinstance(evaluator_id, str) else None
    if multi:
        policy_keys = _multi_policy_keys(reg)
    if "sampler" in reg and policy_keys is not None:
        if not isinstance(sampler, dict):
            bad("sampler", "sampler is not a mapping", f"give {{{', '.join(policy_keys)}}}")
        else:
            for key in sorted(set(policy_keys) - set(sampler)):
                bad(f"sampler.{key}", f"missing sampler field of evaluator {evaluator_id} (EXT-106)",
                    f"register the evaluator's sampler: {', '.join(policy_keys)}")
            for key in sorted(set(sampler) - set(policy_keys), key=str):
                bad(f"sampler.{key}", f"unknown sampler field for evaluator {evaluator_id} (EXT-106)",
                    f"use {', '.join(policy_keys)}")
    elif "sampler" in reg:
        if not isinstance(sampler, dict):
            bad("sampler", "sampler is not a mapping", "give {name, temperature, max_new_tokens}")
        else:
            if not isinstance(sampler.get("name"), str):
                bad("sampler.name", "missing sampler name", "name the deployment sampler")
            if not (_num(sampler.get("temperature")) and sampler["temperature"] > 0):
                bad("sampler.temperature", "temperature must be > 0", "set a sampling temperature")
            if not (_int(sampler.get("max_new_tokens")) and sampler["max_new_tokens"] >= 1):
                bad("sampler.max_new_tokens", "max_new_tokens must be a positive integer", "set `max_new_tokens`")
            # Tier B sampler fields (TIER-B-SPEC H-7): top_p is NeMo Evaluator GenerationConfig.top_p
            # (config/services.py:39 at ea35ba3); enable_thinking goes to the chat template (G-15).
            if "top_p" in sampler and not (_num(sampler["top_p"]) and 0 < sampler["top_p"] <= 1):
                bad("sampler.top_p", "top_p must be in (0, 1]", "set `top_p: 1.0`")
            if "enable_thinking" in sampler and sampler["enable_thinking"] is not False:
                bad("sampler.enable_thinking",
                    "thinking must be off on both sides: the prototype trains and evaluates on NeMo Evaluator's "
                    "GSM8K prompt with enable_thinking false (interim rule G-15)", "set `enable_thinking: false`")
            for key in sorted(set(sampler) - {"name", "temperature", "max_new_tokens", "top_p", "enable_thinking"}):
                bad(f"sampler.{key}", "unknown sampler field",
                    "use name, temperature, max_new_tokens, top_p, enable_thinking")
    if "evaluators" in reg:
        if not (isinstance(evaluators, list) and evaluators and all(isinstance(e, dict) and isinstance(e.get("id"), str)
                                                                     for e in evaluators)):
            bad("evaluators", "evaluators must be a non-empty list of {id, manifest_hash}", "list the evaluators")
        else:
            seen_ids: dict[str, int] = {}
            for i, e in enumerate(evaluators):
                if e["id"] in seen_ids:
                    # R12A-26: the deciding evaluator is found by id (R11A-23), so an id is listed once.
                    bad(f"evaluators[{i}].id",
                        f"evaluator {e['id']!r} is listed again (first at evaluators[{seen_ids[e['id']]}]); the "
                        "deciding evaluator would be ambiguous (R11A-23)",
                        "list each evaluator once, with one manifest_hash and at most one scorer")
                seen_ids.setdefault(e["id"], i)
                mh = e.get("manifest_hash")
                if mh is None:
                    # 05 section 3.7 (R10A-2, R11-4, R13-7; R11A-25): every evaluator's hash is resolved;
                    # null or absent is refused whether or not the registration is signed.
                    resolved = resolved_evaluator_hash(e)
                    fix = (f"set `manifest_hash: \"{resolved}\"`, the hash the prototype resolves for "
                           f"{e['id']}" if resolved else "record the evaluator manifest hash (sha256:<64 hex>)")
                    bad(f"evaluators[{i}].manifest_hash",
                        "the evaluator manifest hash is null or absent; it must be resolved when the registration "
                        "is signed (05 section 3.7)", fix)
                elif not (isinstance(mh, str) and SHA_RE.match(mh)):
                    # R13-7 (R11A-25): anything other than sha256:<64 lower-case hex> is unresolved.
                    bad(f"evaluators[{i}].manifest_hash", f"{mh!r} is not a resolved sha256:<64 hex> hash",
                        "record the evaluator manifest hash (05 section 3.7)")
                # Interim rule G-21: an optional NeMo Evaluator scorer; absent or null keeps the benchmark's
                # default scorer and the top-level `reward`. Checked on every entry, so the deciding one
                # (R11A-23) is covered wherever it stands.
                scorer = e.get("scorer")
                if scorer is not None and not (isinstance(scorer, str) and scorer in SCORER_ALLOWLIST):
                    bad(f"evaluators[{i}].scorer",
                        f"scorer {scorer!r} is not on the prototype allowlist (interim rule G-21)",
                        f"use one of {', '.join(SCORER_ALLOWLIST)} (NeMo Evaluator ea35ba3 "
                        "scoring/__init__.py:38-44), or remove `scorer` to gate on the default `reward`",
                        rule="REG-SCORER")
                elif scorer is not None and e["id"] != SCORER_EVALUATOR:
                    bad(f"evaluators[{i}].scorer",
                        f"evaluator {e['id']!r} takes no scorer: only the Tier B evaluator "
                        f"{SCORER_EVALUATOR} reads NeMo Evaluator scorer results (interim rule G-21, decision D-10)",
                        f"remove `scorer`, or use evaluator {SCORER_EVALUATOR}", rule="REG-SCORER")
    mult = reg.get("multiplicity")
    if "multiplicity" in reg and multi:
        if not isinstance(mult, dict) or set(mult) - {"plan", "family"} or mult.get("family") is not None \
                or mult.get("plan") != MULTI_PLAN:
            bad("multiplicity", f"a registration with several clauses combines them by intersection-union: "
                f"`{{plan: {MULTI_PLAN}}}` (05 section 3.5, R18-2)", f"use `{{plan: {MULTI_PLAN}}}`")
    elif "multiplicity" in reg:
        if not isinstance(mult, dict) or mult.get("family") is not None or mult.get("plan") != "single_gate":
            bad("multiplicity", "the skeleton supports one gate with no family (plan: single_gate, family: null)",
                "use `{plan: single_gate, family: null}`; families need 05 section 5")
    if "analysis_code_version" in reg and reg["analysis_code_version"] != ANALYSIS_CODE_VERSION:
        if is_old_analysis_code_version(reg["analysis_code_version"]):
            bad("analysis_code_version",
                f"registered analysis code {reg['analysis_code_version']!r} names the framework before its rename to "
                f"gatedtrain; this code is {ANALYSIS_CODE_VERSION!r} and reads no old name",
                f"{REGISTRATION_AMENDMENT}. A gate decided before the rename is reproduced with the commit that "
                "decided it")
        else:
            bad("analysis_code_version",
                f"registered analysis code {reg['analysis_code_version']!r} is not this code ({ANALYSIS_CODE_VERSION!r})",
                f"run the registered analysis code version, or {REGISTRATION_AMENDMENT}")
    boot = reg.get("bootstrap")
    if "bootstrap" in reg:
        if not isinstance(boot, dict):
            bad("bootstrap", "bootstrap is not a mapping", "give {replicates, seed}")
        else:
            if not (_int(boot.get("replicates")) and boot["replicates"] >= 100):
                bad("bootstrap.replicates", "replicates must be an integer >= 100 (05 section 3.4 uses 10,000)",
                    "set `replicates: 10000`")
            if not (_int(boot.get("seed")) and boot["seed"] >= 0):
                bad("bootstrap.seed", "the bootstrap seed must be a non-negative integer", "set `seed`")
    if "pairing" in reg:
        pairing = reg["pairing"]
        if not (isinstance(pairing, dict) and set(pairing) <= {"max_unpaired"}):
            bad("pairing", "pairing must be {max_unpaired: <count or fraction>} (05 section 3.4)",
                "set `pairing: {max_unpaired: 0}` or remove it (absent means 0)")
        elif "max_unpaired" in pairing:
            b = pairing["max_unpaired"]
            if not ((_int(b) and b >= 0) or (isinstance(b, float) and 0 <= b < 1)):
                bad("pairing.max_unpaired",
                    f"max_unpaired {b!r} is neither an integer count >= 0 nor a fraction in [0, 1) (05 section 3.4)",
                    "set a count such as `max_unpaired: 0` or a fraction such as `max_unpaired: 0.01`")
    if "contamination" in reg:
        cont = reg["contamination"]
        if not (isinstance(cont, dict) and set(cont) <= set(CONTAMINATION_KEYS)
                and all(isinstance(v, bool) for v in cont.values())):
            bad("contamination", "contamination must be {allow_exact_overlap: true | false} (MULTIMODAL-SPEC MM-11)",
                "set `contamination: {allow_exact_overlap: false}` or remove it (absent means false)")
    if "min_paired" in reg and not (_int(reg["min_paired"]) and reg["min_paired"] >= 1):
        bad("min_paired", f"min_paired {reg['min_paired']!r} is not an integer >= 1 (05 section 3.6)",
            "set `min_paired: 10` (the default) or remove it")
    if "amends" in reg and "created_at" not in reg:
        bad("created_at", "an amended registration states when it was written (05 section 3.7, R13-7)",
            "add `created_at` with the amendment's creation time in RFC 3339 UTC format")
    if "baseline_pilot_mean" in reg and not multi:     # the revision 18 form: `_check_multi`
        _check_pilot_mean(reg["baseline_pilot_mean"], EVALUATORS.get(evaluator_id or ""), evaluator_id, bad)
    if "amends" in reg and not (isinstance(reg["amends"], str) and SHA_RE.match(reg["amends"])):
        bad("amends", "amends must be the prior registration's hash, sha256:<64 hex> (05 section 3.7)",
            "set `amends: sha256:<SHA-256 of the prior registration file>` or remove it")
    if "post_hoc" in reg and not isinstance(reg["post_hoc"], bool):
        bad("post_hoc", "post_hoc must be true or false (05 section 3.7)",
            "set `post_hoc: true` when the amendment follows candidate-arm results, else false")
    if "signed_by" in reg and not isinstance(reg["signed_by"], str):
        bad("signed_by", "signed_by must name the signer", "use `prototype-unsigned` for the prototype")

    if not p and isinstance(clause, dict) and isinstance(pilot, dict) and not constraint:
        _check_planning(clause, pilot, "clause", "sigma_pilot", bad)
    if not p and multi:
        for i, c in enumerate(reg["clauses"]):
            _check_planning(c, pilot[c["name"]], f"clauses[{i}]", f"sigma_pilot.{c['name']}", bad)
    return p


def _check_pilot(pilot, sigma_plan, field: str, bad) -> None:
    """One sigma pilot against its clause's sigma plan (05 section 3.6)."""
    if not isinstance(pilot, dict):
        bad(field, "sigma_pilot is not a mapping", "give {s, df, source} or {sigma, source}")
    elif sigma_plan == "ucb80":
        if not (_num(pilot.get("s")) and pilot["s"] > 0):
            bad(f"{field}.s", "ucb80 needs a positive pilot SD s (05 section 3.6)", "set `s`")
        if not (_int(pilot.get("df")) and pilot["df"] >= 1):
            bad(f"{field}.df", "ucb80 needs integer degrees of freedom df >= 1", "set `df`")
    elif sigma_plan == "fixed":
        if not (_num(pilot.get("sigma")) and pilot["sigma"] > 0):
            bad(f"{field}.sigma", "a fixed sigma plan needs a positive sigma (05 section 3.6)", "set `sigma`")
    if isinstance(pilot, dict) and not pilot.get("source"):
        bad(f"{field}.source", "the sigma source is not named (05 section 3.6 `fixed`, `ucb80`)",
            "name where the pilot SD comes from")


def _check_planning(clause: dict, pilot: dict, cfield: str, pfield: str, bad) -> None:
    """The planning sigma and planning distance of one clause (05 sections 3.1 and 3.6)."""
    try:
        planning_sigma(clause, pilot)
        d = planning_distance(clause)
        if d is not None and d <= 0:
            bad(f"{cfield}.planning_effect",
                f"the planning distance is {d}: planning_effect must exceed delta_sup for superiority and "
                "-delta for non-inferiority (05 section 3.1)", "raise `planning_effect`")
        elif d is None and abs(float(clause["planning_effect"])) >= float(clause["margin"]):
            bad(f"{cfield}.planning_effect",
                f"|planning_effect| {abs(float(clause['planning_effect']))} is not below the equivalence margin "
                f"{clause['margin']}: the TOST sample size is undefined (05 section 3.6)",
                "plan with |planning_effect| < margin")
    except (KeyError, ValueError, TypeError) as exc:
        bad(pfield, f"cannot compute the planning sigma: {exc}", "fix sigma_plan and sigma_pilot")


def _multi_policy_keys(reg: dict) -> tuple | None:
    """EXT-106 in the revision 18 form: the sampler keys the clauses' evaluators declare (`_check_multi` refuses
    clauses whose evaluators declare different keys, R18-5)."""
    from .ext.registry import EVALUATOR_ROWS as _ext_rows

    keys = {(_ext_rows.get(c.get("evaluator")) or {}).get("sampler_keys")
            for c in reg.get("clauses") or [] if isinstance(c, dict) and isinstance(c.get("evaluator"), str)}
    keys = {tuple(k) if isinstance(k, (list, tuple)) else k for k in keys}
    return next(iter(keys)) if len(keys) == 1 else None


def _check_multi(reg: dict) -> list[Problem]:
    """Design revision 18 (05 sections 3.1, 3.5, 3.7; R18-1, R18-2, R18-5): the `clauses` form. Each clause is checked
    as the singular clause is, against its own evaluator; `sigma_pilot` and `baseline_pilot_mean` are keyed by clause
    name; a 05 section 6.1 disclosure is required when any clause's evaluator row declares it."""
    from .ext.registry import DISCLOSURES as _disclosures, EVALUATOR_ROWS as _rows

    p: list[Problem] = []

    def bad(field: str, message: str, fix: str, rule: str = "REG-FIELD") -> None:
        p.append(Problem(rule, f"registration.{field}", message, fix))

    clauses = reg.get("clauses")
    if not (isinstance(clauses, list) and len(clauses) >= 2 and all(isinstance(c, dict) for c in clauses)):
        bad("clauses", "clauses must be a list of at least two clause mappings (05 section 3.7, R18-1)",
            "list each blocking clause, or write one clause as `clause`")
        return p
    evaluators = reg.get("evaluators")
    listed = {e.get("id") for e in evaluators if isinstance(e, dict)} if isinstance(evaluators, list) else set()
    names: dict[str, int] = {}
    deciding: dict[str, int] = {}
    for i, c in enumerate(clauses):
        f = f"clauses[{i}]"
        for key in CLAUSE_FIELDS + MULTI_CLAUSE_FIELDS:
            if key not in c or c[key] is None:
                if key == "margin" and c.get("blocking") is True:
                    bad(f"{f}.margin", "a blocking gate declares no margin (05 section 3.1)",
                        "declare `margin` (delta, or delta_sup = 0 for superiority) and justify it", rule="REG-MARGIN")
                else:
                    bad(f"{f}.{key}", "missing clause field (05 section 3.1, R18-1)", f"add `{key}` to the clause")
        for key in sorted(set(c) - set(CLAUSE_FIELDS) - set(MULTI_CLAUSE_FIELDS) - set(MULTI_OPTIONAL_CLAUSE_FIELDS),
                          key=str):
            why = ("a constraint clause keeps the singular `clause` form" if key in ("constraint_set", "bar")
                   else "unknown clause field")
            bad(f"{f}.{key}", f"{why} (05 section 3.1, R18-1)", f"remove `{key}`")
        name = c.get("name")
        if "name" in c and not (isinstance(name, str) and CLAUSE_NAME_RE.match(name)):
            bad(f"{f}.name", f"clause name {name!r} is not a name of letters, digits, '_', '.' or '-'",
                "name the clause, for example `name: quality`")
        elif isinstance(name, str):
            if name in names:
                bad(f"{f}.name", f"clause name {name!r} is used again (first at clauses[{names[name]}])",
                    "give each clause its own name")
            names.setdefault(name, i)
        ev = c.get("evaluator")
        if "evaluator" in c and not isinstance(ev, str):
            bad(f"{f}.evaluator", "evaluator must be the id of a listed evaluator", "name the clause's evaluator")
            ev = None
        if isinstance(ev, str):
            if ev in deciding:
                bad(f"{f}.evaluator", f"evaluator {ev!r} already decides clauses[{deciding[ev]}]; one evaluator per "
                    "clause (R18-1)", "give each clause its own evaluator (its own metric)")
            deciding.setdefault(ev, i)
            if ev not in listed:
                bad(f"{f}.evaluator", f"evaluator {ev!r} is not listed in `evaluators` (05 section 3.7)",
                    f"list `{{id: {ev}, manifest_hash: <hash>}}`")
            elif ev not in EVALUATORS:
                bad(f"{f}.evaluator", f"the deciding evaluator {ev!r} declares no direction, score range or "
                    "estimators the prototype knows (items.EVALUATORS), so the clause cannot be checked against it "
                    "(05 section 3.1)", f"gate on one of {', '.join(sorted(EVALUATORS))}")
        if c.get("blocking") is False:
            bad(f"{f}.blocking", f"every clause of `{MULTI_PLAN}` blocks (05 section 3.5, R18-2)",
                "set `blocking: true`, or report the metric outside the gate")
        _check_clause(c, bad, ev if isinstance(ev, str) and ev in listed else None, prefix=f)

    # sigma_pilot keyed by clause name (R18-1).
    pilot = reg.get("sigma_pilot")
    if "sigma_pilot" in reg:
        if not isinstance(pilot, dict):
            bad("sigma_pilot", "sigma_pilot is not a mapping of clause name -> pilot (R18-1)",
                "give `sigma_pilot: {<clause name>: {sigma, source} | {s, df, source}, ...}`")
        else:
            for c in clauses:
                name = c.get("name")
                if isinstance(name, str) and name not in pilot:
                    bad(f"sigma_pilot.{name}", f"no sigma pilot for clause {name!r} (05 section 3.6, R18-1)",
                        f"add `{name}: {{sigma, source}}` or `{{s, df, source}}` under sigma_pilot")
                elif isinstance(name, str):
                    _check_pilot(pilot[name], c.get("sigma_plan"), f"sigma_pilot.{name}", bad)
            for key in sorted(set(pilot) - set(names), key=str):
                bad(f"sigma_pilot.{key}", f"{key!r} names no clause (R18-1)", f"remove `{key}`")
    # baseline_pilot_mean keyed by clause name (R13-9, per clause).
    if "baseline_pilot_mean" in reg:
        bpm = reg["baseline_pilot_mean"]
        if not isinstance(bpm, dict):
            bad("baseline_pilot_mean", "with several clauses baseline_pilot_mean maps clause name -> mean (R18-1)",
                "give `baseline_pilot_mean: {<clause name>: <mean>, ...}` or remove it")
        else:
            for key in sorted(bpm, key=str):
                if key not in names:
                    bad(f"baseline_pilot_mean.{key}", f"{key!r} names no clause (R18-1)", f"remove `{key}`")
                    continue
                ev = clauses[names[key]].get("evaluator")
                ev = ev if isinstance(ev, str) else None

                def bad_k(field, message, fix, rule="REG-FIELD", _k=key):
                    bad(f"baseline_pilot_mean.{_k}", message, fix, rule)
                _check_pilot_mean(bpm[key], EVALUATORS.get(ev or ""), ev, bad_k)
    # 05 section 6.1 disclosures: required when any clause's evaluator row declares one, refused when none does.
    evs = [c.get("evaluator") for c in clauses if isinstance(c.get("evaluator"), str)]
    for d in _disclosures:
        field = d["field"]
        decl = [(ev, (_rows.get(ev) or {}).get(d["row_key"])) for ev in evs]
        decl = [(ev, x) for ev, x in decl if x]
        statement = reg.get(field)
        if field in reg and not (isinstance(statement, str) and statement.strip()):
            bad(field, f"{field} is not a non-empty text", f"{d['what']} ({d['rule']})")
        elif field in reg and not decl:
            bad(field, f"none of the clauses' evaluators ({', '.join(evs)}) declares it: each {d['undeclared']}, so "
                f"the statement would claim what the gate does not show ({d['rule']})", f"remove `{field}`")
        elif decl and field not in reg:
            ev, x = decl[0]
            bad(field, f"the deciding evaluator {ev} is {d['declares'].format(x[d['fact']])} "
                f"({d['rule']}), but the registration does not state {d['missing']}",
                f"add `{field}:` stating, for example: {x['statement']}")
    # EXT-106: every clause's evaluator declares the same sampler keys, or none does (R18-5).
    keys = {(_rows.get(ev) or {}).get("sampler_keys") for ev in evs}
    keys = {tuple(k) if isinstance(k, (list, tuple)) else k for k in keys}
    if len(keys) > 1:
        bad("sampler", f"the clauses' evaluators declare different sampler fields ({sorted(map(str, keys))}); one "
            "registered sampler serves every clause (R18-5)", "gate on evaluators that share one deployment sampler")
    return p


def _utc_time(value) -> _dt.datetime | None:
    """An RFC 3339 time with a zero UTC offset, or None."""
    if isinstance(value, _dt.datetime):
        t = value
    elif isinstance(value, str) and "T" in value:
        try:
            t = _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if t.tzinfo is None or t.utcoffset() != _dt.timedelta(0):
        return None
    return t


def _check_pilot_mean(value, declared: dict | None, evaluator_id: str | None, bad) -> None:
    """05 section 6.1 (R13-9; R11A-29): the pilot baseline mean, stated beside `sigma_pilot`, is a finite
    number inside the deciding evaluator's declared range and not at a closed bound (a floor or ceiling
    leaves no room to detect change). An open (None) bound checks nothing on its side."""
    field = "baseline_pilot_mean"
    if not (_num(value) and math.isfinite(float(value))):
        bad(field, f"baseline_pilot_mean {value!r} is not a finite number (05 section 6.1)",
            "state the pilot baseline mean of the deciding metric, or remove the field")
        return
    lo, hi = declared["score_range"] if declared else (None, None)
    if (lo is not None and value <= lo) or (hi is not None and value >= hi):
        bad(field, f"baseline_pilot_mean {value} is at or outside the score range [{lo}, {hi}] evaluator "
            f"{evaluator_id} declares: a baseline at the floor or ceiling cannot show the change (05 section 6.1, "
            "R13-9)", "choose a deciding metric with room on both sides of the baseline")


def _check_eval_distribution(dist, bad) -> None:
    """`eval_distribution` (05 section 3.7, R11-19; shape: design choice 4): `item_set_hash` sha256:<64 hex>,
    optional `env_version` and `env_params_version` (null or a non-empty string), no other key."""
    if not isinstance(dist, dict):
        bad("eval_distribution", "eval_distribution is not a mapping",
            "give {item_set_hash, env_version, env_params_version}")
        return
    h = dist.get("item_set_hash")
    if not (isinstance(h, str) and SHA_RE.match(h)):
        bad("eval_distribution.item_set_hash", f"item_set_hash {h!r} is not a sha256:<64 hex> manifest hash",
            "record the item-set manifest hash")
    for key in ("env_version", "env_params_version"):
        v = dist.get(key)
        if v is not None and not (isinstance(v, str) and v):
            bad(f"eval_distribution.{key}", f"{key} must be null or a non-empty string (08 section 6.1)",
                f"set `{key}: null` or name the version")
    for key in sorted(set(dist) - set(EVAL_DISTRIBUTION_KEYS), key=str):
        bad(f"eval_distribution.{key}", "unknown eval_distribution field",
            f"remove `{key}`; allowed: {', '.join(EVAL_DISTRIBUTION_KEYS)}")


def _check_clause(c: dict, bad, evaluator_id: str | None = None, *, constraint: bool = False,
                  prefix: str = "clause") -> None:
    if "type" in c and c["type"] not in GATE_TO_CLAUSE.values():
        bad(f"{prefix}.type", f"unknown clause type {c['type']!r}", "use superiority, non_inferiority or equivalence")
    if "metric" in c and not isinstance(c["metric"], str):
        bad(f"{prefix}.metric", "metric must be a string", "name the deciding metric")
    if c.get("margin") is not None and not (_num(c["margin"]) and c["margin"] >= 0):
        bad(f"{prefix}.margin", "margin must be a number >= 0 in metric units", "set delta (05 section 3.1)")
    if "alpha" in c and not (_num(c["alpha"]) and 0 < c["alpha"] < 0.5):
        bad(f"{prefix}.alpha", "alpha must be in (0, 0.5)", "use the one-sided level, 0.025 by default (05 section 3.2)")
    if "power" in c and not (_num(c["power"]) and 0 < c["power"] < 1):
        bad(f"{prefix}.power", "power must be in (0, 1)", "use for example 0.8")
    if "planning_effect" in c and not _num(c["planning_effect"]):
        bad(f"{prefix}.planning_effect", "planning_effect must be a number", "set the assumed Delta")
    if "sigma_plan" in c and c["sigma_plan"] not in ("ucb80", "fixed"):
        bad(f"{prefix}.sigma_plan", f"sigma_plan {c['sigma_plan']!r} is not supported by the skeleton",
            "use ucb80 or fixed (internal_pilot needs reestimate, not in the skeleton)")
    if "unit" in c and c["unit"] not in UNITS:
        bad(f"{prefix}.unit", f"unit {c['unit']!r} is not supported: the skeleton gate analyses the item "
            "(05 section 3.3; `training_seed` is not implemented)", "set `unit: item`")
    declared = EVALUATORS.get(evaluator_id or "")
    if "direction" in c and c["direction"] is not None:
        if c["direction"] not in DIRECTIONS:
            bad(f"{prefix}.direction", f"direction {c['direction']!r} is not one of {', '.join(DIRECTIONS)}",
                "set `direction: higher_is_better` or `lower_is_better` (05 section 3.1)")
        elif declared and c["direction"] != declared["direction"]:
            bad(f"{prefix}.direction", f"the clause says {c['direction']} but evaluator {evaluator_id} declares "
                f"{declared['direction']} (05 section 3.1, R11-10)", f"set `direction: {declared['direction']}`")
    if "estimator" in c:
        if c["estimator"] not in ESTIMATORS:
            bad(f"{prefix}.estimator", f"estimator {c['estimator']!r} is not one of {', '.join(ESTIMATORS)}",
                "set `estimator: mean` or remove it (05 section 3.1)")
        elif declared and c["estimator"] not in declared["estimators"]:
            bad(f"{prefix}.estimator", f"evaluator {evaluator_id} declares the estimators "
                f"{', '.join(declared['estimators'])}, not {c['estimator']} (its records carry no numerator "
                "and denominator; 05 section 3.1, R11-21)", "set `estimator: mean` or remove it")
    if "interval" in c and c["interval"] != "cluster_bootstrap" and not constraint:   # constraints: R12A-1
        bad(f"{prefix}.interval", f"interval {c['interval']!r} is not implemented in the skeleton",
            "use `cluster_bootstrap` (05 section 3.4)")
    if "blocking" in c and not isinstance(c["blocking"], bool):
        bad(f"{prefix}.blocking", "blocking must be true or false", "set `blocking: true`")
    # R18-3: `cluster_key` names the item field whose value groups items into resampled clusters.
    if "cluster_key" in c:
        ck = c["cluster_key"]
        if constraint:
            bad(f"{prefix}.cluster_key", "a constraint clause's exact interval resamples nothing (05 section 3.9)",
                "remove `cluster_key`")
        elif not (isinstance(ck, str) and ck and ck != "item"):
            bad(f"{prefix}.cluster_key", f"cluster_key {ck!r} is not the name of an item field other than `item` "
                "(05 section 3.4, R18-3)", "name the item-set field whose value clusters items, for example `speaker`")
    # R18-4: the bootstrap interval variant.
    if "bootstrap_variant" in c:
        if constraint:
            bad(f"{prefix}.bootstrap_variant", "a constraint clause's interval is exact (05 section 3.9)",
                "remove `bootstrap_variant`")
        elif c["bootstrap_variant"] not in BOOTSTRAP_VARIANTS:
            bad(f"{prefix}.bootstrap_variant", f"bootstrap_variant {c['bootstrap_variant']!r} is not one of "
                f"{', '.join(BOOTSTRAP_VARIANTS)} (05 section 3.4, R18-4)",
                "set `bootstrap_variant: bca` (the default) or `percentile`")
    kf = "samples_per_item" if "samples_per_item" in c else "samples_per_prompt"
    k, seeds = c.get(kf), c.get("eval_seeds")
    if kf in c and k is not None and not (_int(k) and k >= 1):
        bad(f"{prefix}.{kf}", "K must be a positive integer", "set `samples_per_item`")
    if "eval_seeds" in c:
        if not (isinstance(seeds, list) and seeds and all(_int(s) for s in seeds) and len(set(seeds)) == len(seeds)):
            bad(f"{prefix}.eval_seeds", "eval_seeds must be a non-empty list of distinct integers", "list one seed per repeat")
        elif _int(k) and len(seeds) != k:
            bad(f"{prefix}.eval_seeds", f"{len(seeds)} seeds for K = {k}: repeat r uses eval_seeds[r] (05 section 3.3)",
                "give exactly samples_per_item seeds")


def check_against_gate(reg: dict, gate: dict, gate_path: str) -> list[Problem]:
    """Interim rule G-4: the registration is authoritative; a differing spec value is refused.

    Design revision 18 (04 section 4.2, R18-5): with `clauses`, the gate's `evaluator` is the list of the clauses'
    evaluators in clause order, its `uses` names the first clause's type, and a `test` value applies to every clause
    (each must hold it)."""
    p: list[Problem] = []
    multi = is_multi(reg)
    if multi:
        clauses = [c for c in reg.get("clauses") or [] if isinstance(c, dict)] if isinstance(reg.get("clauses"), list) \
            else []
        targets = [(f"clauses[{i}]", c) for i, c in enumerate(clauses)]
    else:
        targets = [("clause", reg.get("clause") if isinstance(reg.get("clause"), dict) else {})]
    test = gate.get("test") or {}
    gtype = str(gate.get("uses", "")).split("@")[0]

    def differ(reg_field: str, spec_field: str, reg_value: Any, spec_value: Any) -> None:
        p.append(Problem("REG-FIELD", f"registration.{reg_field}",
                         f"the spec sets {gate_path}.{spec_field} = {spec_value!r} but the registration has "
                         f"{reg_value!r}; the registration is authoritative (interim rule G-4)",
                         f"change {gate_path}.{spec_field} to {reg_value!r}, or write a new registration"))

    for n, (cf, clause) in enumerate(targets):
        if (n == 0 and gtype in SPEC_GATE_TO_CLAUSE and clause.get("type") is not None
                and clause["type"] != SPEC_GATE_TO_CLAUSE[gtype]):
            differ(f"{cf}.type", "uses", clause["type"], gtype)
        pairs = (("metric", "metric"), ("alpha", "alpha"), ("power", "power"), ("mde", "planning_effect"),
                 ("margin", "margin"), ("cluster_by", "unit"), ("bar", "bar"), ("constraint_set", "constraint_set"))
        for spec_key, clause_key in pairs:
            if spec_key in test and clause_key in clause:
                spec_value, reg_value = test[spec_key], clause[clause_key]
                if clause_key == "unit":                           # `prompt` is an alias of `item` (R11A-13)
                    spec_value = UNITS.get(spec_value, spec_value) if isinstance(spec_value, str) else spec_value
                    reg_value = UNITS.get(reg_value, reg_value) if isinstance(reg_value, str) else reg_value
                if spec_value != reg_value:
                    differ(f"{cf}.{clause_key}", f"test.{spec_key}", clause[clause_key], test[spec_key])
        if "constraint_set" in test and test["constraint_set"] is not None and "constraint_set" not in clause:
            differ(f"{cf}.constraint_set", "test.constraint_set", None, test["constraint_set"])
        seeds = clause.get("eval_seeds")
        if "seeds" in test and isinstance(seeds, list) and test["seeds"] != len(seeds):
            differ(f"{cf}.eval_seeds", "test.seeds", f"{len(seeds)} seeds", test["seeds"])
    sampler = reg.get("sampler") if isinstance(reg.get("sampler"), dict) else {}
    if "sampler" in gate and "name" in sampler and gate["sampler"] != sampler["name"]:
        differ("sampler.name", "sampler", sampler["name"], gate["sampler"])
    evaluators = reg.get("evaluators") if isinstance(reg.get("evaluators"), list) else []
    ids = [e.get("id") for e in evaluators if isinstance(e, dict)]
    if multi:
        want = [c.get("evaluator") for _, c in targets]
        if "evaluator" in gate and gate["evaluator"] != want:
            p.append(Problem("REG-FIELD", "registration.clauses",
                             f"the spec sets {gate_path}.evaluator = {gate['evaluator']!r} but the registration's "
                             f"clauses are decided by {want!r}, in that order (R18-5); the registration is "
                             "authoritative (interim rule G-4)",
                             f"set {gate_path}.evaluator to {want!r}, or write a new registration"))
    elif "evaluator" in gate and isinstance(gate["evaluator"], list):
        p.append(Problem("REG-FIELD", "registration.clause",
                         f"the spec's {gate_path}.evaluator lists {len(gate['evaluator'])} evaluators but the "
                         "registration has one clause (R18-5)",
                         f"name one evaluator in {gate_path}.evaluator, or register one clause per evaluator in "
                         "`clauses`"))
    elif "evaluator" in gate and ids and gate["evaluator"] not in ids:
        differ("evaluators", "evaluator", ids, gate["evaluator"])
    return p
