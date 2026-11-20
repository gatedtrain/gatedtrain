"""AN-5: the stage machine, the trust interfaces and their step 1 stubs (tests/README.md section 12).

Stubs stand in for the real stages: the machine must work with any registry entry for the five trust names.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from gatedtrain.ainative.machine import plan_stage, run_stage, status
from gatedtrain.ainative.project import Project, create_project
from gatedtrain.ainative.stage import TRUST_STAGES, Proposal
from gatedtrain.ainative.trust import FixedAlphaLedger, InProcessTrust
from gatedtrain.errors import Refused

S = "gatedtrain.ainative.stub"
CALLS: list[str] = []
FAIL_ONCE: set[str] = set()


def make_stub(stage_name: str, first: bool):
    class Stub:
        name = stage_name
        version = "1"
        inputs = {} if first else {"in": S}
        outputs = {"out": S}
        needs = frozenset()
        params = frozenset({"p"})
        trains = False

        def plan(self, ctx):
            return Proposal(self.name, "stub", {}, {})

        def run(self, ctx, decision):
            CALLS.append(self.name)
            if self.name in FAIL_ONCE:
                FAIL_ONCE.discard(self.name)
                raise RuntimeError("boom")
            upstream = {k: ctx.project.read_artifact(r) for k, r in ctx.inputs.items()}
            obj = {"stage": self.name, "decision": {k: v for k, v in decision.items() if k != "rerun"},
                   "up": upstream}
            return {"out": ctx.project.write_artifact(self.name, "out", obj, S)}

        def check(self, ctx, outputs):
            return []

    return Stub


REGISTRY = {name: make_stub(name, i == 0) for i, name in enumerate(TRUST_STAGES)}


class MachineTest(unittest.TestCase):
    def setUp(self) -> None:
        CALLS.clear()
        FAIL_ONCE.clear()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp = Path(self._tmp.name)
        data = tmp / "d.jsonl"
        data.write_text('{"q": 1}\n')
        spec = {"name": "t", "stages": [
            {"name": n, "stage": n, "params": {}, "inputs": ({} if i == 0 else {"in": f"{TRUST_STAGES[i - 1]}.out"})}
            for i, n in enumerate(TRUST_STAGES)]}
        (tmp / "pipeline.yaml").write_text(yaml.safe_dump(spec))
        self.root = tmp / "proj"
        self.project = create_project(self.root, goal="g", data=data, pipeline=str(tmp / "pipeline.yaml"),
                                      budget={"gpu_hours": 1, "llm_usd": 1})

    def run_(self, ref: str, decision: dict | None = None, project: Project | None = None):
        return run_stage(project or self.project, ref, decision or {}, registry=REGISTRY)

    def states(self) -> dict[str, str]:
        return {k: v["status"] for k, v in status(self.project).items()}

    def lines(self) -> list[dict]:
        return [json.loads(x) for x in (self.root / "attempts.jsonl").read_text().splitlines()]

    def test_run_order_enforced(self) -> None:
        with self.assertRaises(Refused) as cm:
            self.run_("gate")
        self.assertEqual(cm.exception.problems[0].rule, "AN-5")
        with self.assertRaises(Refused):
            self.run_("preregister")           # intake_seal is not done
        self.assertEqual(CALLS, [])
        self.run_("intake_seal")
        self.run_("preregister")
        self.assertEqual(CALLS, ["intake_seal", "preregister"])

    def test_invalid_pipeline_refused_before_any_run(self) -> None:
        reg = dict(REGISTRY)
        del reg["gate"]
        with self.assertRaises(Refused) as cm:
            run_stage(self.project, "intake_seal", {}, registry=reg)
        self.assertEqual(cm.exception.problems[0].rule, "AN-4")
        self.assertEqual(CALLS, [])

    def test_same_key_reuses_outputs_without_run(self) -> None:
        r1 = self.run_("intake_seal", {"a": 1})
        r2 = self.run_("intake_seal", {"a": 1})
        self.assertEqual(CALLS, ["intake_seal"])
        self.assertTrue(r2.reused and not r1.reused)
        self.assertEqual(r1.outputs, r2.outputs)
        self.assertEqual(r1.key, r2.key)

    def test_different_key_on_done_stage_refuses_without_rerun(self) -> None:
        self.run_("intake_seal", {"a": 1})
        with self.assertRaises(Refused) as cm:
            self.run_("intake_seal", {"a": 2})
        self.assertEqual(cm.exception.problems[0].rule, "AN-5")
        self.assertEqual(CALLS, ["intake_seal"])

    def test_rerun_marks_later_stages_stale_and_refuses_stale_input(self) -> None:
        for n in TRUST_STAGES[:3]:
            self.run_(n)
        k_pre = self.run_("preregister").key
        self.assertEqual(self.states(), {"intake_seal": "done", "preregister": "done", "evaluate_sealed": "done",
                                         "gate": "pending", "report": "pending"})
        self.run_("intake_seal", {"a": 2, "rerun": True})
        self.assertEqual(self.states(), {"intake_seal": "done", "preregister": "stale", "evaluate_sealed": "stale",
                                         "gate": "pending", "report": "pending"})
        with self.assertRaises(Refused):
            self.run_("evaluate_sealed")       # its input producer is stale
        n = len(CALLS)
        self.assertNotEqual(self.run_("preregister").key, k_pre)   # a stale stage reruns with the new input
        self.run_("evaluate_sealed")
        self.assertEqual(len(CALLS), n + 2)
        self.assertEqual(set(self.states().values()), {"done", "pending"})

    def test_rerun_flag_is_not_part_of_the_key(self) -> None:
        a = self.run_("intake_seal", {"a": 1})
        b = self.run_("intake_seal", {"a": 1, "rerun": True})
        self.assertTrue(b.reused)
        self.assertEqual(a.key, b.key)

    def test_changed_params_change_the_key(self) -> None:
        r1 = self.run_("intake_seal")
        spec_path = Path(self.project.pipeline)
        raw = yaml.safe_load(spec_path.read_text())
        raw["stages"][0]["params"] = {"p": 1}
        spec_path.write_text(yaml.safe_dump(raw))
        with self.assertRaises(Refused):
            self.run_("intake_seal")
        r2 = self.run_("intake_seal", {"rerun": True})
        self.assertNotEqual(r1.key, r2.key)

    def test_exception_leaves_failed_with_attempt_line_and_retry_succeeds(self) -> None:
        FAIL_ONCE.add("intake_seal")
        with self.assertRaises(RuntimeError):
            self.run_("intake_seal")
        self.assertEqual(self.states()["intake_seal"], "failed")
        events = [(x["stage"], x["attempt"], x["event"]) for x in self.lines()]
        self.assertEqual(events, [("intake_seal", 1, "start"), ("intake_seal", 1, "failed")])
        self.assertNotIn("done", {x["event"] for x in self.lines()})
        with self.assertRaises(Refused):
            self.run_("preregister")           # a failed stage is not a done input
        r = self.run_("intake_seal")
        self.assertEqual(r.attempt, 2)
        self.assertEqual(self.states()["intake_seal"], "done")

    def test_interrupted_start_reads_as_failed_in_a_new_process(self) -> None:
        self.project.append_attempt({"stage": "intake_seal", "attempt": 1, "key": "k", "event": "start",
                                     "utc": "2026-01-01T00:00:00Z"})
        fresh = Project.load(self.root)
        self.assertEqual(status(fresh)["intake_seal"]["status"], "failed")
        self.assertEqual(self.run_("intake_seal", project=fresh).attempt, 2)

    def test_attempt_lines(self) -> None:
        self.run_("intake_seal")
        rows = self.lines()
        self.assertEqual([r["event"] for r in rows], ["start", "done"])
        for r in rows:
            self.assertEqual(r["stage"], "intake_seal")
            self.assertEqual(r["attempt"], 1)
            self.assertRegex(r["key"], r"^sha256:[0-9a-f]{64}$")
            self.assertRegex(r["utc"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        self.assertEqual(rows[0]["key"], rows[1]["key"])

    def test_status_is_derived_from_records_in_a_new_process(self) -> None:
        self.run_("intake_seal")
        self.run_("preregister")
        fresh = Project.load(self.root)
        st = status(fresh)
        self.assertEqual([st[n]["status"] for n in TRUST_STAGES], ["done", "done", "pending", "pending", "pending"])
        self.assertEqual(st["preregister"]["last_attempt"]["event"], "done")
        self.assertIsNone(st["gate"]["last_attempt"])
        # resume: the same key reuses the recorded outputs in the new process
        n = len(CALLS)
        self.assertTrue(self.run_("intake_seal", project=fresh).reused)
        self.assertEqual(len(CALLS), n)

    def test_tampered_output_refused_on_reuse(self) -> None:
        self.run_("intake_seal")
        (self.root / "stages" / "intake_seal" / "out.json").write_text('{"x": 1}')
        with self.assertRaises(Refused):
            self.run_("intake_seal")

    def test_wrong_outputs_fail_the_stage(self) -> None:
        class Bad(REGISTRY["intake_seal"]):
            def run(self, ctx, decision):
                return {}
        with self.assertRaises(Refused):
            run_stage(self.project, "intake_seal", {}, registry={**REGISTRY, "intake_seal": Bad})
        self.assertEqual(self.states()["intake_seal"], "failed")

    def test_plan_has_no_side_effects(self) -> None:
        p = plan_stage(self.project, "intake_seal", registry=REGISTRY)
        self.assertEqual(p.stage, "intake_seal")
        self.assertFalse((self.root / "attempts.jsonl").exists())
        with self.assertRaises(Refused):
            plan_stage(self.project, "gate", registry=REGISTRY)


class TrustStubTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp = Path(self._tmp.name)
        data = tmp / "d.jsonl"
        data.write_text("x\n")
        self.project = create_project(tmp / "p", goal="g", data=data, pipeline="p",
                                      budget={"gpu_hours": 1, "llm_usd": 1})

    def test_fixed_ledger_refuses_second_spend(self) -> None:
        led = FixedAlphaLedger(self.project)
        self.assertEqual(led.available(), 0.05)
        self.assertEqual(led.spend("a1"), 0.05)
        self.assertEqual(led.available(), 0.0)
        with self.assertRaises(Refused) as cm:
            led.spend("a2")
        p = cm.exception.problems[0]
        self.assertEqual((p.rule, p.field), ("AN-9", "gate"))
        led.record("a1", True)
        self.assertEqual(led.entries(), [{"attempt_id": "a1", "alpha": 0.05, "passed": True}])
        with self.assertRaises(Refused):
            led.record("nope", False)

    def test_ledger_persists_across_instances(self) -> None:
        a = FixedAlphaLedger(self.project)
        a.spend("a1")
        b = FixedAlphaLedger(Project.load(self.project.root))
        self.assertEqual(b.available(), 0.0)
        with self.assertRaises(Refused) as cm:
            b.spend("a2")
        self.assertEqual((cm.exception.problems[0].rule, cm.exception.problems[0].field), ("AN-9", "gate"))
        b.record("a1", False)
        c = FixedAlphaLedger(Project.load(self.project.root))
        self.assertEqual(c.entries(), [{"attempt_id": "a1", "alpha": 0.05, "passed": False}])
        lines = (self.project.root / "trust" / "ledger.jsonl").read_text().splitlines()
        self.assertEqual([json.loads(x)["event"] for x in lines], ["spend", "record"])

    def test_registrations_and_sealed_summary_survive_reload(self) -> None:
        t = InProcessTrust(self.project)
        ref = t.seal([{"q": "a", "label": "x"}, {"q": "b", "label": "x"}])
        h = t.register({"metric": "exact_match"})
        t2 = InProcessTrust(Project.load(self.project.root))
        self.assertEqual(t2.registration(h), {"metric": "exact_match"})
        self.assertEqual(t2.sealed_summary(ref), t.sealed_summary(ref))
        self.assertEqual(t2.run_sealed(ref, lambda it: {"q": it["q"]}), [{"q": "a"}, {"q": "b"}])

    def test_no_colon_in_any_file_name(self) -> None:
        t = InProcessTrust(self.project)
        t.seal([{"q": "a"}])
        t.register({"m": 1})
        FixedAlphaLedger(self.project).spend("a1")
        names = [p.name for p in self.project.root.rglob("*")]
        self.assertTrue(any(n.endswith(".json") for n in names))
        self.assertEqual([n for n in names if ":" in n], [])

    def test_symlinked_trust_dir_refused(self) -> None:
        outside = self.project.root.parent / "outside"
        outside.mkdir()
        (self.project.root / "trust").symlink_to(outside, target_is_directory=True)
        for fn in (lambda: InProcessTrust(self.project).seal([{"q": 1}]),
                   lambda: InProcessTrust(self.project).register({"m": 1}),
                   lambda: FixedAlphaLedger(self.project).spend("a1"),
                   lambda: FixedAlphaLedger(self.project).available()):
            with self.assertRaises(Refused) as cm:
                fn()
            self.assertEqual(cm.exception.problems[0].rule, "AN-2")
        self.assertEqual(list(outside.iterdir()), [])

    def test_in_process_trust(self) -> None:
        t = InProcessTrust(self.project)
        self.assertEqual(t.sealed_guarantee, "unsealed (step 1): the operator process can read the sealed items")
        items = [{"q": "a", "label": "x"}, {"q": "b", "label": "y"}, {"q": "c", "label": "x"}]
        ref = t.seal(items)
        s = t.sealed_summary(ref)
        self.assertEqual((s["n"], s["label_counts"]), (3, {"x": 2, "y": 1}))
        self.assertRegex(s["manifest_sha256"], r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(t.seal(items), ref)
        recs = t.run_sealed(ref, lambda it: {"ok": it["q"] == "a"})
        self.assertEqual(recs, [{"ok": True}, {"ok": False}, {"ok": False}])
        h = t.register({"metric": "exact_match"})
        self.assertEqual(t.registration(h), {"metric": "exact_match"})
        with self.assertRaises(Refused):
            t.registration("sha256:" + "0" * 64)
        with self.assertRaises(Refused):
            t.registration("../x")


if __name__ == "__main__":
    unittest.main()
