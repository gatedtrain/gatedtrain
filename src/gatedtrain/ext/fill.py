"""`gatedtrain ext-arm-fill`: fill and verify the gatedtrain hashes of an arm an external evaluator wrote in one piece
(tests/README.md section 11.19, EXT-110; MULTIMODAL-SPEC.md MM-1, MM-2, MM-17, MM-19).

An evaluator outside gatedtrain (the drone-hover recipe's `eval_hover.py`) writes
`items.jsonl` and `eval_meta.json` but cannot compute three values that only gatedtrain defines:
`checkpoint.model_hash` (the adapter's `identity(cut)`, MM-17), `item_set_hash` (the content-addressed manifest hash of the item
set, MM-6) and `sampler_hash` (the SHA-256 of the RFC 8785 canonical JSON of `sampler`, MM-2). This command
computes them from the evaluated cut and the item set, and

- writes each one that is null in `eval_meta.json`;
- refuses (exit 3, `EXT-FILL`, nothing written) when one is set to another value, and when the arm's
  `normalizer_sha256` or `env_version_sha256` is not the cut's (evaluator rows with `normalizer` or `env_version`,
  MM-19), so a mismatch is never overwritten;
- rewrites `eval_meta.json` as canonical JSON (MM-2), keeping every other key, `created_at` included; with nothing
  to fill it changes nothing (the file is left as it is).

It never reads the scores, never opens the cut's pickles (it hashes files, MM-12) and never touches `items.jsonl`.

For LoRA checkpoints (EXT-126), a cut's model hash composes
the base (MM-18), so `--base` (the plain base directory) is required for an adapter whose `needs_base(checkpoint)` is
true; with `--base`, `checkpoint.base_model_hash` is the base's model hash and is filled or verified too. The area's
evaluator writes `sampler_hash` as the SHA-256 of compact sorted JSON (Python `json.dumps(..., sort_keys=True,
separators=(",", ":"))`), not RFC 8785; a `sampler_hash` equal to that encoding of the arm's own `sampler` is
converted to the RFC 8785 hash (reported as `converted`); any other value is still refused.

An adapter that declares `arm_env_version_sha256(arm)` (EXT-241)
verifies the arm's `env_version_sha256` against that (the evaluator's own `env_version.json` in the arm) instead of
`env_version_sha256(checkpoint)` (a cut that carries one, area "control").
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..canonical import canonical_json, manifest_hash, sha256_bytes, write_canonical
from ..errors import Problem, Refused
from ..renamed import old_schema_note
from . import registry
from .arm import META_KEYS, META_OPTIONAL, META_SCHEMA


def compact_sha256(obj: Any) -> str:
    """The evaluators' own sampler hash encoding (area2-drones eval_hover.py, area3-sd15 eval_images.py): SHA-256 of
    compact sorted JSON. Equal to the RFC 8785 hash for integers and strings, not for floats such as 1.0 (EXT-126)."""
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


FIX = "evaluate the cut and item set named here, or pass the cut and item set the arm evaluated"


def arm_fill(*, adapter: str, arm: Path, checkpoint: Path, items: Path, base: Path | None = None) -> dict[str, Any]:
    """Fill and verify `<arm>/eval_meta.json` (EXT-110, EXT-126). Returns {filled: [keys], converted: [keys], values:
    {key: value}}. Raises Refused; then writes nothing."""
    try:
        ad = registry.adapter(adapter)
    except KeyError:
        raise Refused([Problem("EXT-ADAPTER", "--adapter", f"no adapter {adapter!r} is registered",
                               f"use one of {', '.join(sorted(registry.ADAPTERS))}")])
    arm, checkpoint, items = Path(arm), Path(checkpoint), Path(items)
    path = arm / "eval_meta.json"
    problems: list[Problem] = []
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise Refused([Problem("EXT-FILL", "ARM", f"cannot read {path} ({exc.__class__.__name__})",
                               "pass the arm directory the evaluator wrote")])
    if not isinstance(meta, dict):
        raise Refused([Problem("EXT-FILL", "ARM", f"{path} is not a JSON object", "pass an MM-2 arm")])
    missing = [k for k in META_KEYS if k not in meta]
    unknown = sorted(set(meta) - set(META_KEYS) - set(META_OPTIONAL))
    if missing or unknown or meta.get("schema") != META_SCHEMA:
        raise Refused([Problem("EXT-FILL", "ARM", f"{path} is not an MM-2 eval_meta.json (missing {missing}, unknown "
                               f"{unknown}, schema {meta.get('schema')!r}){old_schema_note(meta.get('schema'))}",
                               "pass an MM-2 arm")])
    ck = meta.get("checkpoint")
    if not (isinstance(ck, dict) and set(ck) == {"model_hash", "base_model_hash"}):
        raise Refused([Problem("EXT-FILL", "ARM.checkpoint", "checkpoint is not {model_hash, base_model_hash} (MM-2)",
                               "pass an MM-2 arm")])
    if meta.get("adapter") != adapter or meta.get("evaluator_id") not in ad.evaluator_ids:
        problems.append(Problem("EXT-FILL", "ARM.evaluator_id", f"the arm is adapter {meta.get('adapter')!r}, evaluator "
                                f"{meta.get('evaluator_id')!r}, not one of adapter {adapter}'s "
                                f"{', '.join(ad.evaluator_ids)}", "pass --adapter of the evaluator that wrote the arm"))
    bad = ad.model_dir_problem(checkpoint)
    if bad:
        raise Refused(problems + [Problem("EXT-FILL", "--checkpoint", f"{checkpoint} is not a complete cut: {bad}",
                                          "pass the evaluated cut directory")])
    if base is None and getattr(ad, "needs_base", lambda d: False)(checkpoint):          # EXT-126 (MM-18)
        raise Refused(problems + [Problem("EXT-FILL", "--base", f"{checkpoint} is a LoRA cut: its model hash composes "
                                          "the base directory's (MM-18)", "pass --base <the plain base directory>")])
    if base is not None:
        base = Path(base)
        why = ad.model_dir_problem(base)
        if why is None and getattr(ad, "needs_base", lambda d: False)(base):
            why = "it is a cut, not a base directory"
        if why:
            raise Refused(problems + [Problem("EXT-FILL", "--base", f"{base} is not a base model directory: {why}",
                                              "pass the plain base directory the cut was trained on")])
    if not (items / "items.jsonl").is_file():
        raise Refused(problems + [Problem("EXT-FILL", "--items", f"{items}/items.jsonl is missing",
                                          "pass the item set directory the arm evaluated")])
    try:
        computed = {"checkpoint.model_hash": ad.identity(checkpoint, base)["model_hash"],
                    "item_set_hash": manifest_hash(items),
                    "sampler_hash": sha256_bytes(canonical_json(meta.get("sampler")))}
        if base is not None:                                                            # EXT-126 (MM-2, MM-18)
            computed["checkpoint.base_model_hash"] = ad.identity(base)["model_hash"]
        evaluator_hash = compact_sha256(meta.get("sampler"))
    except (OSError, ValueError, TypeError) as exc:
        raise Refused(problems + [Problem("EXT-FILL", "--checkpoint", f"cannot hash the cut, the item set or the "
                                          f"sampler: {exc}", "keep them plain readable files; no symbolic links")])
    row = registry.EVALUATOR_ROWS.get(meta.get("evaluator_id")) or {}
    verify = {}
    try:                                                # EXT-154: unreadable files refuse
        if row.get("normalizer"):                       # MM-19: the normalizer the evaluation loaded
            verify["normalizer_sha256"] = ad.normalizer_sha256(checkpoint)
        if row.get("env_version"):
            # EXT-241: an adapter whose cuts carry no environment version (area "VLA": the evaluator writes
            # env_version.json into the arm) declares `arm_env_version_sha256`, the hash of the arm's own file.
            verify["env_version_sha256"] = (ad.arm_env_version_sha256(arm) if hasattr(ad, "arm_env_version_sha256")
                                            else ad.env_version_sha256(checkpoint))
    except OSError as exc:
        raise Refused(problems + [Problem("EXT-FILL", "--checkpoint", f"cannot hash the cut's normalizer or "
                                          f"environment version ({exc.__class__.__name__})",
                                          "keep the cut's files readable")])
    filled: list[str] = []
    converted: list[str] = []
    for key, want in computed.items():
        got = ck[key.split(".", 1)[1]] if key.startswith("checkpoint.") else meta.get(key)
        if got is None:
            filled.append(key)
        elif key == "sampler_hash" and got != want and got == evaluator_hash:
            converted.append(key)                          # the evaluator's compact-JSON encoding (EXT-126)
        elif got != want:
            problems.append(Problem("EXT-FILL", f"ARM.{key}", f"eval_meta.json {key} is {got!r}, but the "
                                    f"{'cut' if key.startswith('checkpoint') else 'item set' if key == 'item_set_hash' else 'sampler'}"
                                    f" gives {want} (EXT-110)", FIX))
    for key, want in verify.items():
        if meta.get(key) != want:
            problems.append(Problem("EXT-FILL", f"ARM.{key}", f"eval_meta.json {key} is {meta.get(key)!r}, not the "
                                    f"cut's {want} (MM-19, EXT-110)", FIX))
    if problems:
        raise Refused(problems)
    if filled or converted:
        checkpoint_values = {k.split(".", 1)[1]: v for k, v in computed.items() if k.startswith("checkpoint.")}
        meta = dict(meta, checkpoint=dict(ck, **checkpoint_values),
                    item_set_hash=computed["item_set_hash"], sampler_hash=computed["sampler_hash"])
        write_canonical(path, meta)
    return {"filled": filled, "converted": converted, "values": computed}
