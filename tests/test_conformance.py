"""The adapter conformance kit (gatedtrain.ext.conformance) and the plugin registry's merge rules."""

from __future__ import annotations

import copy
import sys
import types
import unittest
from unittest import mock

import _support  # noqa: F401  (the test host layout)
from gatedtrain.ext import registry
from gatedtrain.ext.conformance import AdapterConformance, adapter_problems


class TestRegisteredAdaptersConform(unittest.TestCase):

    def test_every_registered_adapter_conforms(self):
        """The synthetic test adapter (installed by `_support`) and every installed recipe plugin."""
        self.assertIn("synthetic", registry.ADAPTERS)
        for name in registry.ADAPTERS:
            with self.subTest(adapter=name):
                self.assertEqual(adapter_problems(name), [])


class TestMixin(AdapterConformance, unittest.TestCase):
    ADAPTER = "synthetic"


class _Fake:
    """A minimal conforming adapter for the negative cases (`fake_adapter`)."""
    name = "fake_adapter"
    trainer_id = "rl.fake@0.1"
    evaluator_ids = ("fake.items@0.1",)
    framework_name = "fake"
    checkpoint_completeness = {"library": "x", "rule": "x", "complete_dir_pattern": "x", "sources": [], "verified": False}
    seed_derivation = "seed from run_args.json"
    binds_spec = False

    def metric(self, eid):
        row = registry.EVALUATOR_ROWS[eid]
        return {k: row[k] for k in ("metric", "direction", "binary", "tier")}


for _m in ("identity", "model_dir_problem", "checkpoint_dirs", "find_cuts", "cut_problem", "checks", "consumption",
           "framework_identity", "script_problems", "mount_binding_problems"):
    setattr(_Fake, _m, lambda self, *a, **k: None)

FAKE_TRAINER = {"uses": ("rl.fake@0.1",), "pins": (), "outputs": (), "components": ("w",),
                "consumption": {"target": "row", "fallback": None}, "gpu_h_estimate": 0.0,
                "evaluators": ("fake.items@0.1",), "framework": "fake", "input_roots": ("/r",), "spec_binding": False,
                "env_names": ("HOME",), "scan_skip": (), "interpreter_trees": ("/py",), "interpreter": "/r/.venv/bin/python",
                "container_paths": {"ro_identity": True, "ro_under": ("/inputs",), "rw": ("/out",), "tmpfs": ("/tmp",)},
                "env_paths": ("HOME",)}
FAKE_EVALUATOR = {"metric": "m", "direction": "higher_is_better", "score_range": (0, 1),
                  "scorer": {"name": "s", "version": "1", "weights_sha256": None, "code_sha256": None},
                  "tier": "T3", "binary": True, "error_classes": ("e",), "score_outcomes": ("o",),
                  "payload_text_field": "prompt", "item_payload": ("prompt",), "normalizer": False,
                  "env_version": False, "framework": "fake", "samples": "10, 1"}


class TestNegativeCases(unittest.TestCase):

    def setUp(self):
        mod = types.ModuleType("gatedtrain_fake_plugin")
        mod.FakeAdapter = type("FakeAdapter", (_Fake,), {})
        sys.modules["gatedtrain_fake_plugin"] = mod
        self.addCleanup(sys.modules.pop, "gatedtrain_fake_plugin", None)
        self.mod = mod

    def problems(self, trainer=None, evaluator=None, cls=None):
        if cls is not None:
            self.mod.FakeAdapter = cls
        with mock.patch.dict(registry.ADAPTERS, {"fake_adapter": "gatedtrain_fake_plugin:FakeAdapter"}), \
                mock.patch.dict(registry.TRAINER_ROWS, {"fake_adapter": trainer or copy.deepcopy(FAKE_TRAINER)}), \
                mock.patch.dict(registry.EVALUATOR_ROWS, {"fake.items@0.1": evaluator or copy.deepcopy(FAKE_EVALUATOR)}):
            return adapter_problems("fake_adapter")

    def assertProblem(self, problems, text):
        self.assertTrue(any(text in p for p in problems), problems)

    def test_fake_conforms(self):
        self.assertEqual(self.problems(), [])

    def test_unregistered(self):
        self.assertProblem(adapter_problems("no_such_adapter"), "is not registered")

    def test_trainer_row_schema(self):
        row = copy.deepcopy(FAKE_TRAINER)
        del row["env_paths"]
        self.assertProblem(self.problems(trainer=row), "has no `env_paths`")
        row = {**copy.deepcopy(FAKE_TRAINER), "input_roots": ("relative",)}
        self.assertProblem(self.problems(trainer=row), "`input_roots` is not a list of absolute paths")
        row = {**copy.deepcopy(FAKE_TRAINER), "consumption": {"target": "epoch", "fallback": None}}
        self.assertProblem(self.problems(trainer=row), "`consumption`")
        row = {**copy.deepcopy(FAKE_TRAINER), "env_paths": ("XDG_CACHE_HOME",)}
        self.assertProblem(self.problems(trainer=row), "outside `env_names`")

    def test_evaluator_row_schema(self):
        self.assertProblem(self.problems(evaluator={**FAKE_EVALUATOR, "tier": "T9"}), "`tier`")
        self.assertProblem(self.problems(evaluator={**FAKE_EVALUATOR, "direction": "up"}), "`direction`")
        bad = {**FAKE_EVALUATOR, "scorer": {**FAKE_EVALUATOR["scorer"], "code_sha256": "abc"}}
        self.assertProblem(self.problems(evaluator=bad), "code_sha256")

    def test_missing_method_and_mismatches(self):
        cls = type("NoChecks", (_Fake,), {"checks": None})
        self.assertProblem(self.problems(cls=cls), "no method `checks`")
        cls = type("OtherEvaluators", (_Fake,), {"evaluator_ids": ("other@0.1",)})
        self.assertProblem(self.problems(cls=cls), "are not its row's `evaluators`")
        cls = type("OtherTrainer", (_Fake,), {"trainer_id": "rl.other@0.1"})
        self.assertProblem(self.problems(cls=cls), "is not one of its row's `uses`")

    def test_hooks_an_evaluator_row_requires(self):
        self.assertProblem(self.problems(evaluator={**FAKE_EVALUATOR, "normalizer": True}), "normalizer_sha256")
        self.assertProblem(self.problems(evaluator={**FAKE_EVALUATOR, "env_version": True}), "env_version_sha256")

    def test_broken_class_path(self):
        def boom(self):
            raise RuntimeError("no")
        self.assertProblem(self.problems(cls=type("Boom", (_Fake,), {"__init__": boom})), "cannot be instantiated")


class TestRegistryMerge(unittest.TestCase):

    def test_duplicate_and_rowless_plugins_refused(self):
        with mock.patch.dict(registry.ADAPTERS), mock.patch.dict(registry.TRAINER_ROWS), \
                mock.patch.dict(registry.EVALUATOR_ROWS):
            with self.assertRaisesRegex(ValueError, "already registered"):
                registry.register({"adapters": {"synthetic": "x:Y"}}, "dup")
            with self.assertRaisesRegex(ValueError, "have no training row"):
                registry.register({"adapters": {"rowless": "x:Y"}}, "rowless")


if __name__ == "__main__":
    unittest.main()
