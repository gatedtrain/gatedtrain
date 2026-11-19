"""Design revision 11: item names in the registration and the spec, legacy names still read
(tests/README.md section 11.16).

Design (revision 12 files):
- R11-19 (C5): "prompt" as the analysis unit becomes `item`: clause `unit: item`, `samples_per_item`; 04 gate
  input role `items` of artifact type `item_set` (training prompt datasets keep `prompt_dataset`);
  `cluster_by: item`; the registration's prompt-set hash becomes `eval_distribution` (item-set hash, plus
  `env_version` and `env_params_version` for environments). 05 section 3.1 (clause YAML), section 3.7;
  04 section 3 (examples, 04:43, 04:58-67), section 4.2 (04:100).
- R11-10: each clause states `direction`, matching the evaluator manifest (05 section 3.1).
- R11-21: ratio-of-sums metrics (05 section 3.1); the clause names the estimator (design choice).
- R11-6: the registration is authoritative for every clause field it holds (05 section 3.7; 04 section 4.2).
- R11-4: a null or unresolved evaluator hash is refused whether or not the registration is signed.

Legacy (revision 10) names stay readable: `unit: prompt`, `samples_per_prompt`, `prompt_set_hash`, gate role
`prompts` with type `prompt_dataset`, `cluster_by: prompt`. Each legacy key is an alias of its new key; a
document giving both forms of one field is refused. Every case validates a private copy of the Tier B fixture
(rendered by _tierb.render_inputs) or the Tier A demo, through the CLI (`gatedtrain validate`).
"""

from __future__ import annotations

import copy
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from _r11 import longrun_registration, rewrite, to_item_names, to_item_names_registration
from _support import DEMO_DIR, EXIT_OK, EXIT_REFUSED, describe, parse_errors, run_cli
from _tierb import render_inputs
from test_tierb_spec import reference_export

DUMMY_HASH = "sha256:" + "5" * 64


class Case(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-r11-reg-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.n = 0

    def tierb(self, *, new_names=True, reg=None, spec=None) -> Path:
        """The Tier B fixture, in revision 11 names unless `new_names` is false, plus mutations."""
        self.n += 1
        root = self.tmp / f"b{self.n}"
        root.mkdir()
        paths = render_inputs(root, reference_export(root))
        if new_names:
            to_item_names(paths["spec"], paths["registration"], reg_extra=reg, spec_extra=spec)
        elif reg is not None or spec is not None:
            rewrite(paths["spec"], paths["registration"], reg_mutate=reg, spec_mutate=spec)
        return root

    def tiera(self, *, reg=None, spec=None) -> Path:
        """The Tier A demo in revision 11 names, evaluator hash set (R10A-2 out of the way)."""
        self.n += 1
        root = self.tmp / f"a{self.n}"
        shutil.copytree(DEMO_DIR, root, ignore=shutil.ignore_patterns("baseline", "runs", "tierb", "__pycache__",
                                                                       ".DS_Store"))

        def r(d):
            d["evaluators"][0]["manifest_hash"] = d["evaluators"][0].get("manifest_hash") or DUMMY_HASH
            if reg is not None:
                reg(d)

        to_item_names(root / "spec.yaml", root / "registration.yaml", reg_extra=r, spec_extra=spec)
        return root

    def validate(self, root: Path):
        return run_cli(["validate", "spec.yaml"], cwd=root, timeout=60)

    def assert_ok(self, root: Path):
        proc = self.validate(root)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))

    def assert_refused(self, root: Path, rule: str, field: str):
        """Refused with error[rule] field, and with no other error: the mutation alone is the cause (so a
        document that fails for an unrelated reason, such as an unknown revision 11 name, does not pass)."""
        proc = self.validate(root)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        all_errs = parse_errors(proc.stderr)
        errs = [e for e in all_errs if e["rule"] == rule and e["field"] == field]
        self.assertTrue(errs, f"expected error[{rule}] {field}\n" + describe(proc))
        others = sorted({(e["rule"], e["field"]) for e in all_errs} - {(rule, field)})
        self.assertEqual(others, [], "only the mutated field may be refused\n" + describe(proc))
        self.assertNotIn("Traceback", proc.stderr)
        return errs[0]


# ---------------------------------------------------------------------------
# R11A-1: item names in the registration
# ---------------------------------------------------------------------------

class TestRegistrationItemNames(Case):

    def test_tierb_item_names_validate(self):
        self.assert_ok(self.tierb())

    def test_tiera_item_names_validate(self):
        self.assert_ok(self.tiera())

    def test_direction_required(self):
        root = self.tierb(reg=lambda d: d["clause"].pop("direction"))
        self.assert_refused(root, "REG-FIELD", "registration.clause.direction")

    def test_direction_values(self):
        for value in ("higher", "maximize", None, 1):
            with self.subTest(value=value):
                root = self.tierb(reg=lambda d: d["clause"].__setitem__("direction", value))
                self.assert_refused(root, "REG-FIELD", "registration.clause.direction")

    def test_direction_must_match_the_evaluator(self):
        """05 section 3.1: the clause's direction matches the evaluator manifest; both prototype evaluators
        (gsm8k.nemo_evaluator@0.1, addition_em.cpu_stub@0.1) declare higher_is_better."""
        root = self.tierb(reg=lambda d: d["clause"].__setitem__("direction", "lower_is_better"))
        self.assert_refused(root, "REG-FIELD", "registration.clause.direction")
        root = self.tiera(reg=lambda d: d["clause"].__setitem__("direction", "lower_is_better"))
        self.assert_refused(root, "REG-FIELD", "registration.clause.direction")

    def test_unit_training_seed_not_supported(self):
        root = self.tierb(reg=lambda d: d["clause"].__setitem__("unit", "training_seed"))
        self.assert_refused(root, "REG-FIELD", "registration.clause.unit")

    def test_samples_per_item_must_match_seeds(self):
        root = self.tierb(reg=lambda d: d["clause"].__setitem__("samples_per_item", 3))
        self.assert_refused(root, "REG-FIELD", "registration.clause.eval_seeds")

    def test_samples_per_item_required(self):
        root = self.tierb(reg=lambda d: d["clause"].pop("samples_per_item"))
        self.assert_refused(root, "REG-FIELD", "registration.clause.samples_per_item")

    def test_eval_distribution_required(self):
        root = self.tierb(reg=lambda d: d.pop("eval_distribution"))
        self.assert_refused(root, "REG-FIELD", "registration.eval_distribution")

    def test_eval_distribution_item_set_hash(self):
        for value in (None, "sha256:abc", 7):
            with self.subTest(value=value):
                root = self.tierb(reg=lambda d: d["eval_distribution"].__setitem__("item_set_hash", value))
                self.assert_refused(root, "REG-FIELD", "registration.eval_distribution.item_set_hash")

    def test_eval_distribution_env_fields(self):
        """`env_version` and `env_params_version` (08 section 6.1): null, or a non-empty string."""
        self.assert_ok(self.tierb(reg=lambda d: d["eval_distribution"].update(env_version="gym-pybullet-drones@2.0",
                                                                             env_params_version="v3")))
        root = self.tierb(reg=lambda d: d["eval_distribution"].__setitem__("env_version", ""))
        self.assert_refused(root, "REG-FIELD", "registration.eval_distribution.env_version")
        root = self.tierb(reg=lambda d: d["eval_distribution"].__setitem__("env_params_version", 3))
        self.assert_refused(root, "REG-FIELD", "registration.eval_distribution.env_params_version")

    def test_eval_distribution_unknown_key(self):
        root = self.tierb(reg=lambda d: d["eval_distribution"].__setitem__("prompt_template", "x"))
        self.assert_refused(root, "REG-FIELD", "registration.eval_distribution.prompt_template")

    def test_estimator(self):
        """Verifier choice: clause `estimator: mean | ratio_of_sums` (absent means mean). The prototype
        evaluators write no numerator or denominator, so `ratio_of_sums` is refused for them."""
        self.assert_ok(self.tierb(reg=lambda d: d["clause"].__setitem__("estimator", "mean")))
        root = self.tierb(reg=lambda d: d["clause"].__setitem__("estimator", "median"))
        self.assert_refused(root, "REG-FIELD", "registration.clause.estimator")
        root = self.tierb(reg=lambda d: d["clause"].__setitem__("estimator", "ratio_of_sums"))
        self.assert_refused(root, "REG-FIELD", "registration.clause.estimator")


# ---------------------------------------------------------------------------
# R11A-2: legacy names are read; both forms of one field are refused
# ---------------------------------------------------------------------------

class TestLegacyNames(Case):

    def test_legacy_registration_and_spec_validate(self):
        """The revision 10 fixture as it is (prompt names, no direction) still validates."""
        self.assert_ok(self.tierb(new_names=False))

    def test_legacy_top_level_with_new_clause(self):
        """Per-field aliases: `prompt_set_hash` with a revision 11 clause is read (the shape test_r12_constraints
        builds from the demo registration)."""
        def mutate(d):
            d["prompt_set_hash"] = d.pop("eval_distribution")["item_set_hash"]
        self.assert_ok(self.tierb(reg=mutate))

    def test_both_set_hash_forms_refused(self):
        def mutate(d):
            d["prompt_set_hash"] = d["eval_distribution"]["item_set_hash"]
        self.assert_refused(self.tierb(reg=mutate), "REG-FIELD", "registration.prompt_set_hash")

    def test_both_k_forms_refused(self):
        def mutate(d):
            d["clause"]["samples_per_prompt"] = d["clause"]["samples_per_item"]
        self.assert_refused(self.tierb(reg=mutate), "REG-FIELD", "registration.clause.samples_per_prompt")

    def test_unit_prompt_is_an_alias(self):
        self.assert_ok(self.tierb(reg=lambda d: d["clause"].__setitem__("unit", "prompt")))

    def test_legacy_spec_role_with_new_registration(self):
        """Spec in revision 10 names (role `prompts`, type `prompt_dataset`, `cluster_by: prompt`) with a
        revision 11 registration."""
        root = self.tierb(new_names=False, reg=to_item_names_registration)
        self.assert_ok(root)


class TestNormalize(unittest.TestCase):
    """`registration.normalize(reg)` returns a copy in revision 11 names; the input is not changed."""

    def normalize(self, reg):
        from gatedtrain.registration import normalize

        return normalize(reg)

    def test_real_long_run_registration(self):
        """The registration of the real long run (fixtures/r11/longrun, revision 9/10 names)."""
        reg = longrun_registration()
        before = copy.deepcopy(reg)
        out = self.normalize(reg)
        self.assertEqual(reg, before, "normalize must not change its input")
        c = out["clause"]
        self.assertEqual(c["unit"], "item")
        self.assertEqual(c["samples_per_item"], 4)
        self.assertNotIn("samples_per_prompt", c)
        self.assertEqual(c["direction"], "higher_is_better", "legacy clauses default to higher_is_better")
        self.assertEqual(c["eval_seeds"], [11, 12, 13, 14])
        self.assertNotIn("prompt_set_hash", out)
        self.assertEqual(out["eval_distribution"], {
            "item_set_hash": "sha256:47c1e13489ce8cb9827858e85e5675dd0dc941917d2ee4cee4ef5ff72f31caee",
            "env_version": None, "env_params_version": None})

    def test_new_names_unchanged(self):
        reg = longrun_registration()
        to_item_names_registration(reg)
        self.assertEqual(self.normalize(reg), reg)

    def test_demo_registrations(self):
        for name in ("registration.yaml", "registration-few-prompts.yaml", "registration-ni.yaml"):
            with self.subTest(registration=name):
                out = self.normalize(yaml.safe_load((DEMO_DIR / name).read_text(encoding="utf-8")))
                self.assertEqual(out["clause"]["unit"], "item")
                self.assertEqual(out["clause"]["samples_per_item"], 4)
                self.assertRegex(out["eval_distribution"]["item_set_hash"], r"^sha256:[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# R11A-3: the spec's gate role `items` of type `item_set`; `cluster_by: item`
# ---------------------------------------------------------------------------

class TestSpecItemsRole(Case):
    GATE = "stages[1]"

    def test_items_role_needs_item_set(self):
        def spec(s):
            s["artifacts"][s["stages"][1]["inputs"]["items"]]["type"] = "prompt_dataset"
        self.assert_refused(self.tierb(spec=spec), "SPEC-REF", "stages[1].inputs.items")

    def test_prompts_role_needs_prompt_dataset(self):
        """Legacy role, new type: refused (each role keeps its own type)."""
        def spec(s):
            s["artifacts"][s["stages"][1]["inputs"]["prompts"]]["type"] = "item_set"
        self.assert_refused(self.tierb(new_names=False, spec=spec), "SPEC-REF", "stages[1].inputs.prompts")

    def test_both_roles_refused(self):
        def spec(s):
            s["stages"][1]["inputs"]["prompts"] = s["stages"][1]["inputs"]["items"]
        self.assert_refused(self.tierb(spec=spec), "SPEC-REF", "stages[1].inputs.prompts")

    def test_items_role_missing(self):
        self.assert_refused(self.tierb(spec=lambda s: s["stages"][1]["inputs"].pop("items")), "SPEC-REF",
                            "stages[1].inputs.items")

    def test_training_prompt_dataset_keeps_its_type(self):
        """A training stage's data stays `training_dataset`; `item_set` there is refused (R11-19: training
        prompt datasets keep their type)."""
        def spec(s):
            s["artifacts"][s["stages"][0]["inputs"]["data"]]["type"] = "item_set"
        self.assert_refused(self.tierb(spec=spec), "SPEC-REF", "stages[0].inputs.data")

    def test_cluster_by_legacy_prompt_alias(self):
        self.assert_ok(self.tierb(spec=lambda s: s["stages"][1]["test"].__setitem__("cluster_by", "prompt")))


# ---------------------------------------------------------------------------
# R11A-13: the registration is authoritative (alias-aware)
# ---------------------------------------------------------------------------

class TestAuthoritative(Case):

    def test_cluster_by_differs_from_unit(self):
        root = self.tierb(spec=lambda s: s["stages"][1]["test"].__setitem__("cluster_by", "training_seed"))
        self.assert_refused(root, "REG-FIELD", "registration.clause.unit")

    def test_seeds_differ_from_k(self):
        root = self.tierb(spec=lambda s: s["stages"][1]["test"].__setitem__("seeds", 3))
        self.assert_refused(root, "REG-FIELD", "registration.clause.eval_seeds")

    def test_metric_differs(self):
        root = self.tierb(spec=lambda s: s["stages"][1]["test"].__setitem__("metric", "accuracy"))
        self.assert_refused(root, "REG-FIELD", "registration.clause.metric")


# ---------------------------------------------------------------------------
# R11A-14: a null evaluator hash is refused whether or not the registration is signed
# ---------------------------------------------------------------------------

class TestEvaluatorHashSigned(Case):

    def test_signed_registration_with_null_hash(self):
        def mutate(d):
            d["signed_by"] = "eval-team"
            d["evaluators"][0]["manifest_hash"] = None
        self.assert_refused(self.tierb(reg=mutate), "REG-FIELD", "registration.evaluators[0].manifest_hash")


if __name__ == "__main__":
    unittest.main()
