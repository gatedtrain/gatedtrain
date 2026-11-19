"""L2-E-skel-compat fixture (verification-plan.md:257).

The skeleton's workflow spec files must be a strict syntactic subset of the
04 section 3 workflow YAML and satisfy the 04 section 4.2 required-field
rules for a one-stage, one-gate workflow. This fixture does not import
gatedtrain: it checks the demo files against the design text itself, so it
passes before any implementation exists, and it must be re-run unchanged
against package E's validator once it exists.

The allowed key sets are read from the design file (read only):
- 04 section 3 YAML example (04-workflow-and-composition.md:36-72);
- 04 section 4.2 table of required parameters (04:90-96);
- 04 section 4.2 rule 4 gate fields (04:105).
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import yaml

from _support import DEMO_DIR, DESIGN_DIR, file_sha256, manifest_hash

DESIGN_04 = DESIGN_DIR / "04-workflow-and-composition.md"
DEMO_SPECS = sorted(DEMO_DIR.glob("spec*.yaml"))

# 05 section 3.1 clause fields (05-evaluation-and-lineage.md:66-80).
CLAUSE_FIELDS = {
    "type", "metric", "margin", "alpha", "power", "planning_effect", "sigma_plan",
    "unit", "samples_per_prompt", "eval_seeds", "interval", "blocking",
}
# 04 section 4.2 mapping of workflow gate types to 05 clause types (04:98).
GATE_TO_CLAUSE = {
    "gate.superiority": "superiority",
    "gate.noninferiority": "non_inferiority",
    "gate.equivalence": "equivalence",
    "gate.threshold": "superiority",
}
USES_RE = re.compile(r"^(?P<name>[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+)@(?P<ver>\d+(?:\.\d+)*)$")
HASHED_URI_RE = re.compile(r"^(?P<loc>.+)@sha256:(?P<hex>[0-9a-f]{64})$")


def _design_text() -> str:
    if not DESIGN_04.is_file():
        raise unittest.SkipTest(f"design file not available: {DESIGN_04}")
    return DESIGN_04.read_text(encoding="utf-8")


def _section(text: str, start_heading: str, end_heading: str) -> str:
    start = text.index(start_heading)
    end = text.index(end_heading, start)
    return text[start:end]


def grammar_from_design() -> dict[str, set[str]]:
    """Collect the 04 section 3 key vocabulary from the design file."""
    text = _design_text()
    sec3 = _section(text, "## 3. Workflow spec", "## 4. Semantics owned by the workflow layer")
    block = re.search(r"```yaml\n(.*?)```", sec3, re.S)
    assert block, "04 section 3 YAML example not found"
    example = yaml.safe_load(block.group(1))

    top = set(example)
    budget = set(example.get("budget", {}))
    artifact_keys: set[str] = set()
    for entry in example["artifacts"].values():
        artifact_keys |= set(entry)
    stage_keys: set[str] = set()
    test_keys: set[str] = set()
    label_keys: set[str] = set()
    alpha_family_keys: set[str] = set()
    reward_keys: set[str] = set()
    for stage in example["stages"]:
        stage_keys |= set(stage)
        test_keys |= set(stage.get("test", {}) or {})
        label_keys |= set(stage.get("labels", {}) or {})
        alpha_family_keys |= set(stage.get("alpha_family", {}) or {})
        for r in stage.get("rewards", []) or []:
            reward_keys |= set(r)

    sec42 = _section(text, "### 4.2 Gates", "### 4.3 Gate atomicity")
    gate_types: dict[str, set[str]] = {}
    for row in re.finditer(r"^\| `(gate\.[a-z]+)` \| (?:[^|\\]|\\\|)+ \| (?P<params>[^|]+) \|$", sec42, re.M):
        params = set(re.findall(r"`([a-z_]+)`", row.group("params")))
        gate_types[row.group(1)] = params
    assert gate_types, "04 section 4.2 gate table not found"
    for params in gate_types.values():
        test_keys |= params - {"labels"}
    fields_rule = re.search(r"4\. \*\*Fields\.\*\*(.*)", sec42)
    assert fields_rule, "04 section 4.2 rule 4 not found"
    rule4_fields = set(re.findall(r"`([a-z_]+)`", fields_rule.group(1).split(". The owner")[0]))
    stage_keys |= rule4_fields - {"decide_gate"}

    return {
        "top": top,
        "budget": budget,
        "artifact": artifact_keys,
        "stage": stage_keys,
        "test": test_keys,
        "labels": label_keys,
        "alpha_family": alpha_family_keys,
        "reward": reward_keys,
        "gate_types": set(gate_types),
        "rule4": rule4_fields,
        **{f"required:{k}": v for k, v in gate_types.items()},
    }


class TestGrammarExtraction(unittest.TestCase):
    """Guards the extraction itself, so a silent parse change cannot weaken the fixture."""

    def test_extracted_vocabulary_matches_design_text(self):
        g = grammar_from_design()
        self.assertEqual(g["top"], {"workflow", "version", "budget", "artifacts", "stages"})
        self.assertEqual(g["artifact"], {"type", "uri", "disjoint_from"})
        self.assertTrue({"id", "uses", "inputs", "outputs", "evaluator", "sampler", "test", "on_fail"} <= g["stage"])
        self.assertTrue({"owner", "escalation", "quorum", "timeout", "escalate_after"} <= g["stage"])
        self.assertTrue({"metric", "bar", "alpha", "power", "mde", "cluster_by", "seeds", "margin"} <= g["test"])
        self.assertEqual(
            g["gate_types"],
            {"gate.threshold", "gate.superiority", "gate.noninferiority", "gate.equivalence", "gate.human"},
        )
        self.assertEqual(g["required:gate.superiority"], {"power", "mde", "alpha"})
        self.assertEqual(g["required:gate.noninferiority"], {"margin", "power", "alpha"})
        self.assertEqual(g["required:gate.equivalence"], {"margin", "power", "alpha"})


class TestDemoSpecsAreStrictSubset(unittest.TestCase):
    """04 section 3: no field the 04 section 3 grammar lacks."""

    @classmethod
    def setUpClass(cls):
        cls.g = grammar_from_design()
        cls.specs = {p.name: yaml.safe_load(p.read_text(encoding="utf-8")) for p in DEMO_SPECS}

    def test_expected_demo_files_exist(self):
        names = set(self.specs)
        self.assertTrue(
            {"spec.yaml", "spec-no-train.yaml", "spec-few-prompts.yaml", "spec-bad-train.yaml", "spec-bad-train-ni.yaml"}
            <= names,
            names,
        )

    def test_keys_are_in_04_grammar(self):
        g = self.g
        for name, spec in self.specs.items():
            with self.subTest(spec=name):
                self.assertIsInstance(spec, dict)
                self.assertLessEqual(set(spec), g["top"], "top-level keys")
                self.assertLessEqual(set(spec.get("budget", {})), g["budget"], "budget keys")
                for aname, entry in spec["artifacts"].items():
                    self.assertLessEqual(set(entry), g["artifact"], f"artifacts.{aname}")
                for i, stage in enumerate(spec["stages"]):
                    self.assertLessEqual(set(stage), g["stage"], f"stages[{i}]")
                    self.assertLessEqual(set(stage.get("test", {}) or {}), g["test"], f"stages[{i}].test")

    def test_value_shapes(self):
        for name, spec in self.specs.items():
            with self.subTest(spec=name):
                self.assertIsInstance(spec["workflow"], str)
                self.assertEqual(spec["version"], 1)
                self.assertIsInstance(spec["artifacts"], dict)
                for aname, entry in spec["artifacts"].items():
                    self.assertIsInstance(entry["type"], str, aname)
                    self.assertIsInstance(entry["uri"], str, aname)
                    for other in entry.get("disjoint_from", []):
                        self.assertIn(other, spec["artifacts"], f"{aname}.disjoint_from")
                ids = [s["id"] for s in spec["stages"]]
                self.assertEqual(len(ids), len(set(ids)), "stage ids unique")
                for stage in spec["stages"]:
                    self.assertRegex(stage["uses"], USES_RE)
                    for role, ref in (stage.get("inputs") or {}).items():
                        self.assertIsInstance(role, str)
                        self.assertIsInstance(ref, str)
                    for role, ref in (stage.get("outputs") or {}).items():
                        self.assertIsInstance(ref, str)


class TestDemoSpecsSatisfy42Rules(unittest.TestCase):
    """04 section 4.2 required-field rules for a one-stage, one-gate workflow."""

    @classmethod
    def setUpClass(cls):
        cls.g = grammar_from_design()
        cls.specs = {p.name: (p, yaml.safe_load(p.read_text(encoding="utf-8"))) for p in DEMO_SPECS}

    def _split(self, spec):
        gates = [s for s in spec["stages"] if s["uses"].startswith("gate.")]
        trains = [s for s in spec["stages"] if not s["uses"].startswith("gate.")]
        return trains, gates

    def test_one_stage_one_gate(self):
        for name, (_, spec) in self.specs.items():
            with self.subTest(spec=name):
                trains, gates = self._split(spec)
                self.assertEqual(len(trains), 1)
                self.assertEqual(len(gates), 1)
                self.assertIs(spec["stages"][0], trains[0], "the training stage precedes the gate")

    def test_gate_required_parameters(self):
        for name, (_, spec) in self.specs.items():
            with self.subTest(spec=name):
                _, (gate,) = self._split(spec)
                gtype = gate["uses"].split("@")[0]
                self.assertIn(gtype, self.g["gate_types"])
                self.assertIn(gtype, GATE_TO_CLAUSE, "skeleton gates compile to a 05 clause")
                test = gate["test"]
                required = self.g[f"required:{gtype}"]
                for param in required - {"alpha"}:
                    self.assertIn(param, test, f"04 section 4.2 table: {gtype} requires {param}")
                # "`alpha` or a family" (04 section 4.2 table) and never both (04 section 3, line 34).
                has_alpha = "alpha" in test
                has_family = "alpha_family" in gate
                self.assertTrue(has_alpha or has_family, "alpha or a family")
                self.assertFalse(has_alpha and has_family, "a gate inside a family must not set alpha")

    def test_gate_outcome_and_fields(self):
        for name, (_, spec) in self.specs.items():
            with self.subTest(spec=name):
                _, (gate,) = self._split(spec)
                # Rule 3 (04:104): stop | pause | reject; reject only inside a loop body.
                self.assertIn(gate["on_fail"], {"stop", "pause"})
                # Rule 4 (04:105): owner and escalation; rule 2 (04:103): an evaluator.
                self.assertIsInstance(gate["owner"], str)
                self.assertIsInstance(gate["escalation"], list)
                self.assertTrue(gate["escalation"])
                self.assertIsInstance(gate["evaluator"], str)
                self.assertRegex(gate["evaluator"], USES_RE)

    def test_references_resolve_and_design_is_paired(self):
        for name, (_, spec) in self.specs.items():
            with self.subTest(spec=name):
                (train,), (gate,) = self._split(spec)
                known = set(spec["artifacts"])
                for ref in train["inputs"].values():
                    self.assertIn(ref, known)
                produced = set(train["outputs"].values())
                for ref in gate["inputs"].values():
                    self.assertIn(ref, known | produced)
                # The gate evaluates the stage output against the stage's input model
                # on the same prompts (paired design, 05 section 3.3, 05:97).
                self.assertEqual(gate["inputs"]["model"], train["outputs"]["model"])
                self.assertEqual(gate["inputs"]["baseline"], train["inputs"]["model"])
                self.assertEqual(spec["artifacts"][gate["inputs"]["prompts"]]["type"], "prompt_dataset")
                self.assertIn(
                    train["inputs"]["data"], spec["artifacts"][gate["inputs"]["prompts"]].get("disjoint_from", []),
                    "held-out prompts declared disjoint from the training data",
                )

    def test_registration_hash_and_contents(self):
        for name, (path, spec) in self.specs.items():
            with self.subTest(spec=name):
                _, (gate,) = self._split(spec)
                reg_art = spec["artifacts"][gate["inputs"]["registration"]]
                self.assertEqual(reg_art["type"], "registration")
                m = HASHED_URI_RE.match(reg_art["uri"])
                self.assertIsNotNone(m, "registration uri carries @sha256:<hash> (04 section 3 notation)")
                loc = m.group("loc")
                self.assertTrue(loc.startswith("file:"))
                reg_path = path.parent / loc[len("file:"):]
                self.assertEqual(file_sha256(reg_path), "sha256:" + m.group("hex"), "registration SHA-256 (05 section 3.7)")

                reg = yaml.safe_load(reg_path.read_text(encoding="utf-8"))
                for key in (
                    "clause", "margin_justification", "sigma_pilot", "prompt_set_hash", "sampler",
                    "evaluators", "multiplicity", "analysis_code_version", "signed_by",
                ):
                    self.assertIn(key, reg, f"05 section 3.7 registration content: {key}")
                self.assertEqual(reg["signed_by"], "prototype-unsigned")
                clause = reg["clause"]
                self.assertEqual(set(clause), CLAUSE_FIELDS)
                self.assertTrue(clause["blocking"])
                self.assertIsNotNone(clause.get("margin"), "05 section 3.1: blocking gate declares a margin")

                prompts_uri = spec["artifacts"][gate["inputs"]["prompts"]]["uri"]
                prompts_dir = path.parent / prompts_uri[len("file:"):]
                self.assertEqual(reg["prompt_set_hash"], manifest_hash(prompts_dir))

                # 04 section 4.2 mapping (04:98): type, mde -> planning_effect,
                # seeds -> number of eval_seeds, cluster_by -> unit.
                gtype = gate["uses"].split("@")[0]
                test = gate["test"]
                self.assertEqual(clause["type"], GATE_TO_CLAUSE[gtype])
                self.assertEqual(clause["metric"], test["metric"])
                self.assertEqual(clause["alpha"], test["alpha"])
                self.assertEqual(clause["power"], test["power"])
                self.assertEqual(len(clause["eval_seeds"]), test["seeds"])
                self.assertEqual(clause["unit"], test["cluster_by"])
                if "mde" in test:
                    self.assertEqual(clause["planning_effect"], test["mde"])
                if "margin" in test:
                    self.assertEqual(clause["margin"], test["margin"])


if __name__ == "__main__":
    unittest.main()
