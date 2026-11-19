"""Spec and registration errors (MVP-SPEC sections 2.1, 2.2 and 5 "Errors";
"error messages that name the rule, the field, and the fix").

Each case edits a private copy of demo/ and runs `gatedtrain validate <spec>`.
Expected: exit code 3 and at least one stderr line

    error[<RULE>] <field>: <message> | fix: <fix>

with the rule ID and field path fixed in tests/README.md section 3.
`validate` checks the spec and the registration only; it does not need the
baseline checkpoint, so these tests need no warm-up.
"""

from __future__ import annotations

import hashlib
import re
import unittest
from pathlib import Path

import yaml

from _support import (
    EXIT_OK,
    EXIT_REFUSED,
    EXIT_USAGE,
    WORKSPACE,
    describe,
    parse_errors,
    run_cli,
)


def _load(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _dump(path: Path, data) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _rehash_registration(spec_path: Path, reg_name: str = "registration.yaml") -> None:
    spec = _load(spec_path)
    digest = hashlib.sha256((spec_path.parent / reg_name).read_bytes()).hexdigest()
    spec["artifacts"]["registration"]["uri"] = f"file:{reg_name}@sha256:{digest}"
    _dump(spec_path, spec)


class SpecErrorCase(unittest.TestCase):
    counter = 0

    def make_copy(self) -> Path:
        SpecErrorCase.counter += 1
        return WORKSPACE.fresh_demo_copy(f"spec-errors-{SpecErrorCase.counter}")

    def assert_refused(self, demo: Path, spec_name: str, rule: str, field: str):
        proc = run_cli(["validate", spec_name], cwd=demo, timeout=60)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        errors = parse_errors(proc.stderr)
        self.assertTrue(errors, "no 'error[RULE] field: message | fix: ...' line\n" + describe(proc))
        match = [e for e in errors if e["rule"] == rule and e["field"] == field]
        self.assertTrue(match, f"expected error[{rule}] {field}; got {errors}")
        self.assertTrue(match[0]["message"].strip())
        self.assertGreater(len(match[0]["fix"].strip()), 3)
        return proc


class TestValidSpecs(SpecErrorCase):
    def test_demo_specs_validate(self):
        demo = self.make_copy()
        for name in ("spec.yaml", "spec-no-train.yaml", "spec-few-prompts.yaml", "spec-bad-train.yaml",
                     "spec-bad-train-ni.yaml", "spec-no-train-few-prompts.yaml"):
            with self.subTest(spec=name):
                proc = run_cli(["validate", name], cwd=demo, timeout=60)
                self.assertEqual(proc.returncode, EXIT_OK, describe(proc))

    def test_usage_error(self):
        proc = run_cli(["validate"], cwd=self.make_copy(), timeout=60)
        self.assertEqual(proc.returncode, EXIT_USAGE, describe(proc))


class TestSpecSubset(SpecErrorCase):
    """04 section 3 grammar (04:36-72); "no field the 04 section 3 grammar lacks"."""

    def test_unknown_top_level_field(self):
        demo = self.make_copy()
        spec = _load(demo / "spec.yaml")
        spec["notes"] = "not in the grammar"
        _dump(demo / "spec.yaml", spec)
        self.assert_refused(demo, "spec.yaml", "SPEC-SUBSET", "notes")

    def test_unknown_stage_field(self):
        demo = self.make_copy()
        spec = _load(demo / "spec.yaml")
        spec["stages"][0]["learning_rate"] = 0.1
        _dump(demo / "spec.yaml", spec)
        self.assert_refused(demo, "spec.yaml", "SPEC-SUBSET", "stages[0].learning_rate")

    def test_unknown_test_field(self):
        demo = self.make_copy()
        spec = _load(demo / "spec.yaml")
        spec["stages"][1]["test"]["n_boot"] = 100
        _dump(demo / "spec.yaml", spec)
        self.assert_refused(demo, "spec.yaml", "SPEC-SUBSET", "stages[1].test.n_boot")

    def test_unknown_artifact_field(self):
        demo = self.make_copy()
        spec = _load(demo / "spec.yaml")
        spec["artifacts"]["train"]["shards"] = 5
        _dump(demo / "spec.yaml", spec)
        self.assert_refused(demo, "spec.yaml", "SPEC-SUBSET", "artifacts.train.shards")


class TestSpecShapeAndRefs(SpecErrorCase):
    def test_two_training_stages(self):
        """One stage and one gate (MVP-SPEC section 2.1)."""
        demo = self.make_copy()
        spec = _load(demo / "spec.yaml")
        extra = dict(spec["stages"][0])
        extra["id"] = "sft2"
        spec["stages"].insert(1, extra)
        _dump(demo / "spec.yaml", spec)
        self.assert_refused(demo, "spec.yaml", "SPEC-SHAPE", "stages")

    def test_unknown_input_reference(self):
        """Typed artifacts consumed by name (04 section 4.1, 04:80)."""
        demo = self.make_copy()
        spec = _load(demo / "spec.yaml")
        spec["stages"][0]["inputs"]["data"] = "nope"
        _dump(demo / "spec.yaml", spec)
        self.assert_refused(demo, "spec.yaml", "SPEC-REF", "stages[0].inputs.data")


class TestGateRules(SpecErrorCase):
    """04 section 4.2 (04:90-105) and the alpha rules of 04 section 3 (04:34)."""

    def _edit_gate(self, spec_name, fn):
        demo = self.make_copy()
        spec = _load(demo / spec_name)
        fn(spec["stages"][1])
        _dump(demo / spec_name, spec)
        return demo

    # Design revision 10 (R10-7, tests/README.md section 11.15, decision R10A-6): the registration is
    # authoritative and the spec's statistical fields are optional, so a spec without `power`, `mde`,
    # `margin` or `alpha` is no longer refused (GATE-REQUIRED / GATE-ALPHA on stages[1].test.*). The
    # replacements are test_r10_registration.TestRegistrationAuthoritative.

    def test_alpha_inside_family(self):
        demo = self._edit_gate("spec.yaml", lambda g: g.__setitem__("alpha_family", {"name": "final", "alpha_fw": 0.025}))
        self.assert_refused(demo, "spec.yaml", "GATE-ALPHA", "stages[1].alpha_family")

    def test_reject_outside_loop(self):
        demo = self._edit_gate("spec.yaml", lambda g: g.__setitem__("on_fail", "reject"))
        self.assert_refused(demo, "spec.yaml", "GATE-OUTCOME", "stages[1].on_fail")

    def test_missing_owner(self):
        demo = self._edit_gate("spec.yaml", lambda g: g.pop("owner"))
        self.assert_refused(demo, "spec.yaml", "GATE-FIELDS", "stages[1].owner")


class TestRegistrationRules(SpecErrorCase):
    """05 section 3.7 (05:142) and 05 section 3.1 (05:82); MVP-SPEC section 2.2."""

    def test_registration_input_absent(self):
        demo = self.make_copy()
        spec = _load(demo / "spec.yaml")
        spec["stages"][1]["inputs"].pop("registration")
        _dump(demo / "spec.yaml", spec)
        self.assert_refused(demo, "spec.yaml", "REG-MISSING", "stages[1].inputs.registration")

    def test_registration_file_missing(self):
        demo = self.make_copy()
        (demo / "registration.yaml").unlink()
        self.assert_refused(demo, "spec.yaml", "REG-MISSING", "artifacts.registration.uri")

    def test_registration_changed_after_hash_recorded(self):
        demo = self.make_copy()
        reg = demo / "registration.yaml"
        # The pinned digest is the unedited demo file's hash (computed, not a literal: R10A-2 changed the
        # demo registration and its pin; the spec-pin agreement itself is checked in test_e2e_slow).
        pinned = hashlib.sha256(reg.read_bytes()).hexdigest()
        reg.write_text(reg.read_text(encoding="utf-8").replace("alpha: 0.025", "alpha: 0.05"), encoding="utf-8")
        proc = self.assert_refused(demo, "spec.yaml", "REG-HASH", "artifacts.registration.uri")
        # The message names both digests so the reader can see what changed.
        digest = hashlib.sha256(reg.read_bytes()).hexdigest()
        self.assertNotEqual(digest, pinned)
        self.assertIn(digest, proc.stderr)
        self.assertIn(pinned, proc.stderr)

    def test_blocking_gate_without_margin(self):
        demo = self.make_copy()
        reg = demo / "registration.yaml"
        text = reg.read_text(encoding="utf-8")
        self.assertIn("  margin: 0.0\n", text)
        reg.write_text(text.replace("  margin: 0.0\n", ""), encoding="utf-8")
        _rehash_registration(demo / "spec.yaml")
        self.assert_refused(demo, "spec.yaml", "REG-MARGIN", "registration.clause.margin")

    def test_registration_missing_prompt_set_hash(self):
        """Design revision 11 (R11-19, R11A-1): `prompt_set_hash` is the legacy alias of `eval_distribution`;
        with neither, the refusal names the revision 11 field."""
        demo = self.make_copy()
        reg = demo / "registration.yaml"
        text = reg.read_text(encoding="utf-8")
        new = re.sub(r'^prompt_set_hash: .*\n', "", text, flags=re.M)
        self.assertNotEqual(new, text)
        reg.write_text(new, encoding="utf-8")
        _rehash_registration(demo / "spec.yaml")
        self.assert_refused(demo, "spec.yaml", "REG-FIELD", "registration.eval_distribution")


class TestRegistrationIsAuthoritative(SpecErrorCase):
    """Interim rule G-4: the registration is
    authoritative; a spec value that differs from it makes `run` refuse with
    REG-FIELD (exit 3) before any run directory exists. Mapping 04:98.
    These cases need no baseline: the refusal comes from validation."""

    def refuse_run(self, demo: Path, spec_name: str, field: str):
        runs_root = demo / "runs-g4"
        proc = run_cli(["run", spec_name, "--runs-root", str(runs_root)], cwd=demo, timeout=120)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        errors = parse_errors(proc.stderr)
        self.assertTrue(any(e["rule"] == "REG-FIELD" and e["field"] == field for e in errors),
                        f"expected error[REG-FIELD] {field}; got {errors}\n" + describe(proc))
        self.assertFalse(runs_root.exists() and any(p.is_dir() for p in runs_root.iterdir()),
                         "no run directory may be created")

    def _edit_gate(self, spec_name, fn):
        demo = self.make_copy()
        spec = _load(demo / spec_name)
        fn(spec["stages"][1])
        _dump(demo / spec_name, spec)
        return demo

    def test_alpha(self):
        demo = self._edit_gate("spec.yaml", lambda g: g["test"].__setitem__("alpha", 0.05))
        self.refuse_run(demo, "spec.yaml", "registration.clause.alpha")

    def test_power(self):
        demo = self._edit_gate("spec.yaml", lambda g: g["test"].__setitem__("power", 0.9))
        self.refuse_run(demo, "spec.yaml", "registration.clause.power")

    def test_mde(self):
        demo = self._edit_gate("spec.yaml", lambda g: g["test"].__setitem__("mde", 0.2))
        self.refuse_run(demo, "spec.yaml", "registration.clause.planning_effect")

    def test_margin(self):
        demo = self._edit_gate("spec-bad-train-ni.yaml", lambda g: g["test"].__setitem__("margin", 0.05))
        self.refuse_run(demo, "spec-bad-train-ni.yaml", "registration.clause.margin")

    def test_type(self):
        def to_ni(g):
            g["uses"] = "gate.noninferiority@1.0"
            g["test"]["margin"] = 0.0      # equal to the registration margin; only the type differs
        demo = self._edit_gate("spec.yaml", to_ni)
        self.refuse_run(demo, "spec.yaml", "registration.clause.type")

    def test_seeds(self):
        demo = self._edit_gate("spec.yaml", lambda g: g["test"].__setitem__("seeds", 3))
        self.refuse_run(demo, "spec.yaml", "registration.clause.eval_seeds")

    def test_sampler(self):
        demo = self._edit_gate("spec.yaml", lambda g: g.__setitem__("sampler", "greedy"))
        self.refuse_run(demo, "spec.yaml", "registration.sampler.name")

    def test_evaluator(self):
        demo = self._edit_gate("spec.yaml", lambda g: g.__setitem__("evaluator", "other_em.cpu_stub@0.1"))
        self.refuse_run(demo, "spec.yaml", "registration.evaluators")


if __name__ == "__main__":
    unittest.main()
