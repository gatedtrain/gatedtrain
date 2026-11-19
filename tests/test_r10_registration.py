"""Design revision 10 alignment of registration validation (tests/README.md section 11.15).

Design (current files, revision 10):
- R10-5: 05 section 3.7 (05:151) "Evaluator manifest hashes must be resolved ... when the registration is
  signed; a null hash is refused." Supersedes interim rule G-6.
- R10-7: 04 section 3 alpha fields (04:34), 04 section 4.2 (04:100), 05 section 3.2 (05:86): the
  registration is authoritative; spec statistical fields are optional; alpha must be in the
  registration or come from a family.
- R10-8: 05 section 3.4 (05:116) `pairing: {max_unpaired: <count or fraction>}`, default 0.
- R10-9: 05 section 3.7 (05:151) amendments reference the prior registration by hash; an amendment
  after candidate results is `post_hoc: true`.
- R10-10: 05 section 3.6 (05:145) `min_paired`, default 10.

Every case validates a private copy: the Tier B fixture registration (tests/fixtures/tierb) rendered
by _tierb.render_inputs, or the Tier A demo. Edits go through YAML and the spec is re-pinned.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from _support import DEMO_DIR, EXIT_OK, EXIT_REFUSED, SHA256_ID, describe, file_sha256, parse_errors, run_cli
from _tierb import evaluator_identity_hash, render_inputs
from test_tierb_spec import reference_export

DUMMY_HASH = "sha256:" + "5" * 64
PRIOR_HASH = "sha256:" + "9a" * 32


def repin(spec: Path, reg: Path) -> None:
    pat = re.escape(reg.name) + r"@sha256:[0-9a-f]{64}"
    text, n = re.subn(pat, f"{reg.name}@{file_sha256(reg)}", spec.read_text(encoding="utf-8"))
    assert n == 1, f"{spec} pins {reg.name} {n} times"
    spec.write_text(text, encoding="utf-8")


def rewrite_registration(reg: Path, spec: Path, mutate) -> None:
    data = yaml.safe_load(reg.read_text(encoding="utf-8"))
    mutate(data)
    reg.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    repin(spec, reg)


class Case(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-r10-reg-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    # Tier B fixture ---------------------------------------------------------
    def tierb(self, mutate=None, *, scorer=None) -> Path:
        root = self.tmp / "b"
        root.mkdir()
        paths = render_inputs(root, reference_export(root), scorer=scorer, set_scorer=scorer is not None)
        if mutate is not None:
            rewrite_registration(paths["registration"], paths["spec"], mutate)
        return root

    # Tier A demo ------------------------------------------------------------
    def tiera(self, mutate_reg=None, mutate_spec=None, *, spec_name="spec.yaml", reg_name="registration.yaml") -> Path:
        demo = self.tmp / "a"
        shutil.copytree(DEMO_DIR, demo, ignore=shutil.ignore_patterns("baseline", "runs", "tierb", "__pycache__",
                                                                       ".DS_Store"))

        def resolved(d):
            # Keep R10-5 out of the way: validate only refuses a null hash (decision R10A-2).
            d["evaluators"][0]["manifest_hash"] = d["evaluators"][0].get("manifest_hash") or DUMMY_HASH
            if mutate_reg is not None:
                mutate_reg(d)

        rewrite_registration(demo / reg_name, demo / spec_name, resolved)
        if mutate_spec is not None:
            spec = yaml.safe_load((demo / spec_name).read_text(encoding="utf-8"))
            mutate_spec(spec["stages"][1])
            (demo / spec_name).write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
        return demo

    def validate(self, root: Path, spec_name: str = "spec.yaml"):
        return run_cli(["validate", spec_name], cwd=root, timeout=60)

    def assert_ok(self, root: Path, spec_name: str = "spec.yaml"):
        proc = self.validate(root, spec_name)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))

    def assert_refused(self, root: Path, rule: str, field: str, spec_name: str = "spec.yaml", *, prefix=False):
        proc = self.validate(root, spec_name)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        errs = [e for e in parse_errors(proc.stderr)
                if e["rule"] == rule and (e["field"].startswith(field) if prefix else e["field"] == field)]
        self.assertTrue(errs, f"expected error[{rule}] {field}\n" + describe(proc))
        self.assertNotIn("Traceback", proc.stderr)
        return errs[0]


# ---------------------------------------------------------------------------
# R10A-2 (R10-5): a null evaluator manifest hash is refused
# ---------------------------------------------------------------------------

class TestEvaluatorHashResolved(Case):
    FIELD = "registration.evaluators[0].manifest_hash"

    def test_tierb_null_refused_and_fix_names_resolved_hash(self):
        root = self.tierb(lambda d: d["evaluators"][0].__setitem__("manifest_hash", None))
        err = self.assert_refused(root, "REG-FIELD", self.FIELD)
        self.assertIn(evaluator_identity_hash(), err["fix"], "the fix gives the hash the prototype resolves")

    def test_tierb_null_with_scorer_fix_names_scorer_identity(self):
        """Decision D-4: the scorer is part of the evaluator identity, so the resolved hash includes it
        (sha256:2c196d9d..., the value of the real long-run report)."""
        root = self.tierb(lambda d: d["evaluators"][0].__setitem__("manifest_hash", None), scorer="numeric_match")
        err = self.assert_refused(root, "REG-FIELD", self.FIELD)
        self.assertIn(evaluator_identity_hash("numeric_match"), err["fix"])

    def test_tierb_absent_refused(self):
        root = self.tierb(lambda d: d["evaluators"][0].pop("manifest_hash"))
        self.assert_refused(root, "REG-FIELD", self.FIELD)

    def test_tierb_resolved_hash_accepted(self):
        """Control: the fixture template carries the resolved hash."""
        self.assert_ok(self.tierb())

    def test_tiera_null_refused(self):
        root = self.tiera(lambda d: d["evaluators"][0].__setitem__("manifest_hash", None))
        err = self.assert_refused(root, "REG-FIELD", self.FIELD)
        self.assertRegex(err["fix"], r"sha256:[0-9a-f]{64}")

    def test_demo_registrations_carry_resolved_hash(self):
        """The demo registrations pin the resolved evaluator hashes, including the Tier B numeric_match scorer."""
        for name in ("registration.yaml", "registration-few-prompts.yaml", "registration-ni.yaml"):
            with self.subTest(registration=name):
                reg = yaml.safe_load((DEMO_DIR / name).read_text(encoding="utf-8"))
                mh = reg["evaluators"][0].get("manifest_hash")
                self.assertIsInstance(mh, str, f"demo/{name}: manifest_hash is {mh!r}")
                self.assertRegex(mh, SHA256_ID)
                root = self.tmp / f"null-{name}"
                shutil.copytree(DEMO_DIR, root, ignore=shutil.ignore_patterns("baseline", "runs", "tierb",
                                                                               "__pycache__", ".DS_Store"))
                spec = {"registration.yaml": "spec.yaml", "registration-few-prompts.yaml": "spec-few-prompts.yaml",
                        "registration-ni.yaml": "spec-bad-train-ni.yaml"}[name]
                rewrite_registration(root / name, root / spec,
                                     lambda d: d["evaluators"][0].__setitem__("manifest_hash", None))
                proc = self.validate(root, spec)
                errs = [e for e in parse_errors(proc.stderr) if e["field"] == self.FIELD]
                self.assertTrue(errs, describe(proc))
                self.assertIn(mh, errs[0]["fix"], "the demo carries the resolved hash")
        tb = yaml.safe_load((DEMO_DIR / "tierb" / "registration.yaml").read_text(encoding="utf-8"))
        self.assertEqual(tb["evaluators"][0].get("manifest_hash"), evaluator_identity_hash("numeric_match"))


# ---------------------------------------------------------------------------
# R10A-3 (R10-8): pairing.max_unpaired is a count or a fraction
# ---------------------------------------------------------------------------

class TestMaxUnpairedField(Case):
    def set_pairing(self, value):
        return self.tierb(lambda d: d.__setitem__("pairing", {"max_unpaired": value}))

    def test_count_accepted(self):
        self.assert_ok(self.set_pairing(3))

    def test_fraction_accepted(self):
        self.assert_ok(self.set_pairing(0.05))

    def test_bad_values_refused(self):
        for value in (-1, 1.5, -0.1, True, "2", None):
            with self.subTest(value=value):
                shutil.rmtree(self.tmp / "b", ignore_errors=True)
                self.assert_refused(self.set_pairing(value), "REG-FIELD", "registration.pairing", prefix=True)

    def test_absent_means_zero_accepted(self):
        self.assert_ok(self.tierb(lambda d: d.pop("pairing")))


# ---------------------------------------------------------------------------
# R10A-4 (R10-9): amends and post_hoc
# ---------------------------------------------------------------------------

class TestAmendmentFields(Case):
    def test_amends_accepted(self):
        # R11A-29: `created_at` is required with `amends`.
        self.assert_ok(self.tierb(lambda d: d.update(amends=PRIOR_HASH, created_at="2026-11-11T09:00:00Z")))

    def test_amends_must_be_a_digest(self):
        for value in ("sha256:abc", "9a" * 32, None, 3):
            with self.subTest(value=value):
                shutil.rmtree(self.tmp / "b", ignore_errors=True)
                self.assert_refused(self.tierb(lambda d: d.__setitem__("amends", value)), "REG-FIELD",
                                    "registration.amends")

    def test_post_hoc_accepted(self):
        # R11A-29: `created_at` is required with `amends`.
        self.assert_ok(self.tierb(lambda d: d.update(amends=PRIOR_HASH, created_at="2026-11-11T09:00:00Z", post_hoc=True)))
        shutil.rmtree(self.tmp / "b")
        self.assert_ok(self.tierb(lambda d: d.update(post_hoc=False)))

    def test_post_hoc_must_be_boolean(self):
        for value in ("yes", 1, None):
            with self.subTest(value=value):
                shutil.rmtree(self.tmp / "b", ignore_errors=True)
                self.assert_refused(self.tierb(lambda d: d.__setitem__("post_hoc", value)), "REG-FIELD",
                                    "registration.post_hoc")


# ---------------------------------------------------------------------------
# R10A-5 (R10-10): min_paired
# ---------------------------------------------------------------------------

class TestMinPairedField(Case):
    def test_accepted(self):
        self.assert_ok(self.tierb(lambda d: d.__setitem__("min_paired", 10)))

    def test_bad_values_refused(self):
        for value in (0, -1, 2.5, True, "10", None):
            with self.subTest(value=value):
                shutil.rmtree(self.tmp / "b", ignore_errors=True)
                self.assert_refused(self.tierb(lambda d: d.__setitem__("min_paired", value)), "REG-FIELD",
                                    "registration.min_paired")


# ---------------------------------------------------------------------------
# R10A-6 (R10-7): the registration is authoritative; spec statistical fields are optional
# ---------------------------------------------------------------------------

class TestRegistrationAuthoritative(Case):
    """Spec `test.alpha`, `power`, `mde` and `margin` may be left out: the registration carries them.
    Supersedes the GATE-ALPHA spec-side refusal (`stages[1].test.alpha`) and the GATE-REQUIRED refusals
    for power, mde and margin (test_spec_errors.TestGateRules, revised)."""

    def test_spec_without_alpha_validates(self):
        self.assert_ok(self.tiera(mutate_spec=lambda g: g["test"].pop("alpha")))

    def test_spec_without_power_and_mde_validates(self):
        self.assert_ok(self.tiera(mutate_spec=lambda g: (g["test"].pop("power"), g["test"].pop("mde"))))

    def test_ni_spec_without_margin_validates(self):
        root = self.tiera(mutate_spec=lambda g: g["test"].pop("margin"), spec_name="spec-bad-train-ni.yaml",
                          reg_name="registration-ni.yaml")
        self.assert_ok(root, "spec-bad-train-ni.yaml")

    def test_registration_without_alpha_refused(self):
        """Alpha missing from the registration, no family: refused although the spec still sets it."""
        root = self.tiera(lambda d: d["clause"].pop("alpha"))
        self.assert_refused(root, "REG-FIELD", "registration.clause.alpha")

    def test_alpha_missing_everywhere_refused(self):
        root = self.tiera(lambda d: d["clause"].pop("alpha"), lambda g: g["test"].pop("alpha"))
        self.assert_refused(root, "REG-FIELD", "registration.clause.alpha")

    def test_family_and_alpha_still_refused(self):
        root = self.tiera(mutate_spec=lambda g: g.__setitem__("alpha_family", {"name": "final", "alpha_fw": 0.025}))
        self.assert_refused(root, "GATE-ALPHA", "stages[1].alpha_family")

    def test_differing_spec_value_still_refused(self):
        """G-4 / R10-7: a spec value that differs from the registration is refused."""
        root = self.tiera(mutate_spec=lambda g: g["test"].__setitem__("power", 0.9))
        proc = run_cli(["run", "spec.yaml", "--runs-root", str(root / "runs")], cwd=root, timeout=120)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertTrue(any(e["rule"] == "REG-FIELD" and e["field"] == "registration.clause.power"
                            for e in parse_errors(proc.stderr)), describe(proc))


if __name__ == "__main__":
    unittest.main()
