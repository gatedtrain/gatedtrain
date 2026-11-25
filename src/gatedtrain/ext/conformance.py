"""Adapter conformance: what gatedtrain's commands require of an adapter plugin, checked without any run output.

A recipe repository runs this in its CI, so a plugin that registers but breaks ext-launch, ext-arm-fill or ext-collect
fails before a GPU run:

    from gatedtrain.ext.conformance import AdapterConformance

    class TestConformance(AdapterConformance, unittest.TestCase):
        ADAPTER = "my_adapter"

`adapter_problems(name)` lists every problem (empty when the adapter conforms). It checks the registration (the
plugin's adapter, training row and evaluator rows), the row schemas (MULTIMODAL-SPEC section 6), the adapter's
attributes and methods the commands call (MM-12), their agreement with the rows, and the hooks an evaluator row makes
necessary (`normalizer_sha256` for a normalizer, `env_version_sha256` or `arm_env_version_sha256` for an environment
version, MM-19). Behaviour on real outputs is the recipe's own tests' job.
"""

from __future__ import annotations

import re
import unittest
from pathlib import PurePosixPath
from typing import Any

from ..items import DIRECTIONS

# Training row keys every adapter's row carries (MULTIMODAL-SPEC section 6, EXT-79 to EXT-114); the optional ones
# (`image_interpreters`, `sidecar`, `run_options`, `interpreter_flags`, `b18`, `envs`, `nemo_rl_commit`) are the
# adapters' that need them.
TRAINER_ROW_KEYS = ("components", "consumption", "container_paths", "env_names", "env_paths", "evaluators",
                    "framework", "gpu_h_estimate", "input_roots", "interpreter", "interpreter_trees", "outputs", "pins",
                    "scan_skip", "spec_binding", "uses")
EVALUATOR_ROW_KEYS = ("binary", "direction", "env_version", "error_classes", "framework", "item_payload", "metric",
                      "normalizer", "payload_text_field", "samples", "score_outcomes", "score_range", "scorer", "tier")
SCORER_KEYS = ("name", "version", "weights_sha256", "code_sha256")
CONSUMPTION_TARGETS = ("row", "iteration", "stage")              # MM-23
TIERS = ("T0", "T1", "T2", "T3")                                  # 05 section 6.1
CONTAINER_PATH_KEYS = ("ro_identity", "ro_under", "rw", "tmpfs")
COMPLETENESS_KEYS = ("library", "rule", "complete_dir_pattern", "sources", "verified")     # MM-20
# Attributes and methods ext-launch, ext-arm-fill and ext-collect use without a `hasattr` guard (MM-12).
REQUIRED_ATTRIBUTES = ("trainer_id", "evaluator_ids", "framework_name", "checkpoint_completeness", "seed_derivation")
REQUIRED_METHODS = ("identity", "model_dir_problem", "checkpoint_dirs", "find_cuts", "cut_problem", "checks",
                    "consumption", "framework_identity", "script_problems", "mount_binding_problems", "metric")
SHA = re.compile(r"sha256:[0-9a-f]{64}")


def _abs(p: Any) -> bool:
    """An absolute path, or the host's output root or one under it (`{output_root}`, `{output_root}/...`, filled in by
    ext-launch)."""
    if p == "{output_root}":
        return True
    if isinstance(p, str) and p.startswith("{output_root}/"):
        p = "/" + p[len("{output_root}/"):]
    return isinstance(p, str) and PurePosixPath(p).is_absolute()


def _strs(x: Any) -> bool:
    return isinstance(x, (list, tuple)) and all(isinstance(i, str) and i for i in x)


def trainer_row_problems(name: str, row: Any) -> list[str]:
    """The training row's schema."""
    where = f"training row {name}"
    if not isinstance(row, dict):
        return [f"{where} is not a mapping"]
    out = [f"{where} has no `{k}`" for k in TRAINER_ROW_KEYS if k not in row]
    if out:
        return out
    if not (_strs(row["uses"]) and row["uses"]):
        out.append(f"{where}: `uses` is not a non-empty list of trainer ids")
    if not (_strs(row["evaluators"]) and row["evaluators"]):
        out.append(f"{where}: `evaluators` is not a non-empty list of evaluator ids (the deciding one first)")
    cons = row["consumption"]
    if not (isinstance(cons, dict) and cons.get("target") in CONSUMPTION_TARGETS
            and cons.get("fallback") in CONSUMPTION_TARGETS + (None,)):
        out.append(f"{where}: `consumption` is not {{target, fallback}} of {CONSUMPTION_TARGETS} (fallback may be null)")
    for k in ("input_roots", "scan_skip", "interpreter_trees"):
        if not (isinstance(row[k], (list, tuple)) and all(_abs(p) for p in row[k])):
            out.append(f"{where}: `{k}` is not a list of absolute paths")
    if not _abs(row["interpreter"]):
        out.append(f"{where}: `interpreter` is not an absolute path")
    for k in ("env_names", "env_paths", "pins", "components", "outputs"):
        if not isinstance(row[k], (list, tuple)) or not all(isinstance(i, str) for i in row[k]):
            out.append(f"{where}: `{k}` is not a list of strings")
    if isinstance(row["env_paths"], (list, tuple)) and set(row["env_paths"]) - set(row["env_names"]):
        out.append(f"{where}: `env_paths` names {sorted(set(row['env_paths']) - set(row['env_names']))} outside "
                   "`env_names`")
    cp = row["container_paths"]
    if not (isinstance(cp, dict) and set(cp) == set(CONTAINER_PATH_KEYS)):
        out.append(f"{where}: `container_paths` keys are not {list(CONTAINER_PATH_KEYS)}")
    if not isinstance(row["spec_binding"], bool):
        out.append(f"{where}: `spec_binding` is not a boolean")
    gpu = row["gpu_h_estimate"]
    if not (isinstance(gpu, (int, float)) and not isinstance(gpu, bool) and gpu >= 0):
        out.append(f"{where}: `gpu_h_estimate` is not a non-negative number")
    return out


def evaluator_row_problems(eid: str, row: Any) -> list[str]:
    """The evaluator row's schema."""
    where = f"evaluator row {eid}"
    if not isinstance(row, dict):
        return [f"{where} is not a mapping"]
    out = [f"{where} has no `{k}`" for k in EVALUATOR_ROW_KEYS if k not in row]
    if out:
        return out
    if row["direction"] not in DIRECTIONS:
        out.append(f"{where}: `direction` {row['direction']!r} is not one of {list(DIRECTIONS)}")
    if row["tier"] not in TIERS:
        out.append(f"{where}: `tier` {row['tier']!r} is not one of {list(TIERS)}")
    for k in ("binary", "normalizer", "env_version"):
        if not isinstance(row[k], bool):
            out.append(f"{where}: `{k}` is not a boolean")
    rng = row["score_range"]
    if not (isinstance(rng, (list, tuple)) and len(rng) == 2
            and all(x is None or (isinstance(x, (int, float)) and not isinstance(x, bool)) for x in rng)):
        out.append(f"{where}: `score_range` is not (low, high) with numbers or null")
    sc = row["scorer"]
    if not (isinstance(sc, dict) and set(SCORER_KEYS) <= set(sc)):
        out.append(f"{where}: `scorer` has not all of {list(SCORER_KEYS)}")
    else:
        for k in ("weights_sha256", "code_sha256"):
            if not (sc[k] is None or (isinstance(sc[k], str) and SHA.fullmatch(sc[k]))):
                out.append(f"{where}: scorer `{k}` is neither null nor sha256:<64 hex>")
    for k in ("error_classes", "score_outcomes", "item_payload"):
        if not _strs(row[k]):
            out.append(f"{where}: `{k}` is not a list of strings")
    # MM-11: the item record field compared with the training payloads (a payload field or the item id).
    if not (isinstance(row["payload_text_field"], str) and row["payload_text_field"]):
        out.append(f"{where}: `payload_text_field` is not a field name")
    return out


def adapter_problems(name: str) -> list[str]:
    """Every way the registered adapter `name` does not conform; [] when it does."""
    from . import registry

    if name not in registry.ADAPTERS:
        return [f"adapter {name!r} is not registered (no `gatedtrain.adapters` entry point names it)"]
    row = registry.TRAINER_ROWS.get(name)
    out = trainer_row_problems(name, row)
    if out:
        return out
    try:
        ad = registry.adapter(name)
    except Exception as exc:                              # the plugin's class path or constructor is broken
        return [f"adapter {name!r} ({registry.ADAPTERS[name]}) cannot be instantiated: {exc.__class__.__name__}: {exc}"]
    out += [f"adapter {name}: no attribute `{a}`" for a in REQUIRED_ATTRIBUTES if not hasattr(ad, a)]
    out += [f"adapter {name}: no method `{m}`" for m in REQUIRED_METHODS if not callable(getattr(ad, m, None))]
    if out:
        return out
    if getattr(ad, "name", name) != name:
        out.append(f"adapter {name}: its `name` is {ad.name!r}")
    if ad.trainer_id not in row["uses"]:
        out.append(f"adapter {name}: trainer_id {ad.trainer_id!r} is not one of its row's `uses` {list(row['uses'])}")
    if tuple(ad.evaluator_ids) != tuple(row["evaluators"]):
        out.append(f"adapter {name}: evaluator_ids {list(ad.evaluator_ids)} are not its row's `evaluators` "
                   f"{list(row['evaluators'])}")
    if hasattr(ad, "binds_spec") and bool(ad.binds_spec) != row["spec_binding"]:
        out.append(f"adapter {name}: binds_spec {ad.binds_spec!r} is not its row's `spec_binding`")
    cc = ad.checkpoint_completeness
    if not (isinstance(cc, dict) and set(COMPLETENESS_KEYS) <= set(cc)):
        out.append(f"adapter {name}: checkpoint_completeness has not all of {list(COMPLETENESS_KEYS)} (MM-20)")
    if not (isinstance(ad.seed_derivation, str) and ad.seed_derivation.strip()):
        out.append(f"adapter {name}: seed_derivation is not a statement (MM-29)")
    if not (isinstance(ad.framework_name, str) and ad.framework_name):
        out.append(f"adapter {name}: framework_name is not a name")
    for eid in row["evaluators"]:
        erow = registry.EVALUATOR_ROWS.get(eid)
        if erow is None:
            out.append(f"adapter {name}: evaluator {eid!r} of its row is not registered")
            continue
        out += evaluator_row_problems(eid, erow)
        if erow.get("normalizer") is True and not callable(getattr(ad, "normalizer_sha256", None)):
            out.append(f"adapter {name}: evaluator {eid} has a normalizer, the adapter no `normalizer_sha256` (MM-19)")
        if erow.get("env_version") is True and not (callable(getattr(ad, "env_version_sha256", None))
                                                     or callable(getattr(ad, "arm_env_version_sha256", None))):
            out.append(f"adapter {name}: evaluator {eid} has an environment version, the adapter neither "
                       "`env_version_sha256` nor `arm_env_version_sha256` (MM-19)")
        try:
            m = ad.metric(eid)
        except Exception as exc:
            out.append(f"adapter {name}: metric({eid!r}) raises {exc.__class__.__name__}")
            continue
        for k in ("metric", "direction", "binary", "tier"):
            if m.get(k) != erow.get(k):
                out.append(f"adapter {name}: metric({eid!r}) {k} {m.get(k)!r} is not the row's {erow.get(k)!r}")
    out += [f"adapter {name}: {p}" for p in registry.disclosure_problems(
        {eid: registry.EVALUATOR_ROWS[eid] for eid in row["evaluators"] if eid in registry.EVALUATOR_ROWS})]
    return out


class AdapterConformance:
    """Mix into a `unittest.TestCase` and set ADAPTER to the plugin's adapter name."""

    ADAPTER: str | None = None

    def test_adapter_conforms(self) -> None:
        assert isinstance(self, unittest.TestCase)
        if self.ADAPTER is None:
            self.skipTest("set ADAPTER to the adapter name the plugin registers")
        self.assertEqual(adapter_problems(self.ADAPTER), [])
