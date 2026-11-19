"""Collection checks for unreadable inputs, incomplete item coverage and evaluator disclosures.

See EXT-127, EXT-147 and EXT-154 to EXT-159 in tests/README.md.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from _support import EXIT_OK, EXIT_REFUSED, describe, file_sha256, parse_errors, parse_run_dir, run_cli

NOT_PASS = 1


def registry():
    from gatedtrain.ext import registry as reg
    return reg


def _root_can_read() -> bool:
    return os.geteuid() == 0


class Case(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-review9-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.n = 0

    def path(self, name: str) -> Path:
        type(self).n += 1
        return self.tmp / f"{name}-{type(self).n}"

    def unreadable(self, p: Path) -> None:
        if _root_can_read():
            self.skipTest("running as root: chmod 0 does not make a file unreadable")
        p.chmod(0)
        self.addCleanup(p.chmod, 0o644)

    def refused(self, proc, rule: str, field_prefix: str | None = None, text: str | None = None):
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertNotIn("Traceback", proc.stderr)
        errs = parse_errors(proc.stderr)
        hits = [e for e in errs if e["rule"] == rule and (field_prefix is None or e["field"].startswith(field_prefix))
                and (text is None or text in e["message"])]
        self.assertTrue(hits, f"expected error[{rule}] {field_prefix} {text!r}; got {errs}\n{describe(proc)}")
        return hits


# ---------------------------------------------------------------------------
# EXT-154 (review 9 M1): unreadable run and data files refuse with exit 3, every adapter
# ---------------------------------------------------------------------------


class TestUnreadableOthers(Case):
    """The sweep: the synthetic adapter through the generic paths (freeze, cut files, stage directory); the sb3_ppo case lives in
    its recipe repository."""

    def test_trl_freeze(self):
        from _ext import ExtScenario
        sc = ExtScenario(self.path("t"))
        self.unreadable(sc.stage_out / "env-freeze.txt")
        self.refused(sc.collect(self.path("runs")), "EXT-FREEZE", "--stage-out")


# ---------------------------------------------------------------------------
# EXT-155 (review 9 L1, collect side): the arm's per-item references are the registered ones
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# EXT-156 (review 9 L1, launch side): the evaluation scripts' inputs and the script itself
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# EXT-157 (review 9 L1, coverage): arms covering fewer items than registered are MISSING (05 section 3.4, G-18)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# EXT-158 (review 9 L2): a step log longer than the run refuses
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# EXT-159 (review 9 L3): the disclosure table
# ---------------------------------------------------------------------------

class TestDisclosureTable(unittest.TestCase):

    def test_optional_keys_follow_the_table(self):
        from gatedtrain import registration
        for d in registry().DISCLOSURES:
            self.assertIn(d["field"], registration.OPTIONAL_KEYS)

    def test_rows_shaped(self):
        """Every declared disclosure is {<fact>: text, statement: text}; `disclosure_problems` finds a malformed one
        (it runs when the registry is imported, before any run directory exists)."""
        reg = registry()
        self.assertEqual(reg.disclosure_problems(reg.EVALUATOR_ROWS), [])
        bad = {"x@0.1": {"shared_annotation": {"statement": "s"}}, "y@0.1": {"construct_aligned": "text"},
               "z@0.1": {"shared_annotation": {"convention": "c", "statement": " "}}}
        self.assertEqual(len(reg.disclosure_problems(bad)), 3)


class TestDisclosuresAgainstTheRow(unittest.TestCase):
    """05 section 6.1 (R15-1, R16-1; EXT-127, EXT-147): a registration states `construct_alignment` or
    `shared_annotation` exactly for a deciding evaluator whose registry row declares it. The synthetic evaluator
    declares neither; a copy of its row that declares both stands for a recipe's evaluator."""

    DECLARING = "synthetic.declaring_items@0.1"

    def registration(self) -> dict:
        import yaml
        from _ext import EXT
        text = (EXT / "registration.template.yaml").read_text(encoding="utf-8")
        return yaml.safe_load(text.replace("{item_set_hash}", "sha256:" + "a" * 64)
                              .replace("{evaluator_hash}", "sha256:" + "b" * 64))

    def fields(self, reg: dict, evaluator: str) -> list[str]:
        from gatedtrain.registration import check_contents
        return [p.field for p in check_contents(reg, evaluator_id=evaluator)
                if p.field.endswith(("construct_alignment", "shared_annotation"))]

    def test_refused_where_the_row_declares_none(self):
        from _ext import EVALUATOR_ID
        reg = self.registration()
        self.assertEqual(self.fields(reg, EVALUATOR_ID), [])
        reg["shared_annotation"] = "The references share the training labels' annotation."
        self.assertEqual(self.fields(reg, EVALUATOR_ID), ["registration.shared_annotation"])
        del reg["shared_annotation"]
        reg["construct_alignment"] = "A pass shows a better reward."
        self.assertEqual(self.fields(reg, EVALUATOR_ID), ["registration.construct_alignment"])

    def test_required_where_the_row_declares_it(self):
        from unittest import mock
        from _ext import EVALUATOR_ID
        from gatedtrain import items
        reg_mod = registry()
        row = dict(reg_mod.EVALUATOR_ROWS[EVALUATOR_ID],
                   construct_aligned={"reward": "the training reward", "statement": "A pass shows the reward."},
                   shared_annotation={"convention": "one annotation process", "statement": "Shared references."})
        with mock.patch.dict(reg_mod.EVALUATOR_ROWS, {self.DECLARING: row}), \
                mock.patch.dict(items.EVALUATORS, {self.DECLARING: dict(items.EVALUATORS[EVALUATOR_ID])}):
            reg = self.registration()
            reg["evaluators"][0]["id"] = self.DECLARING
            self.assertEqual(sorted(self.fields(reg, self.DECLARING)),
                             ["registration.construct_alignment", "registration.shared_annotation"])
            reg["construct_alignment"] = "A pass shows that the model earns more of the training reward."
            reg["shared_annotation"] = "The references come from the training labels' annotation process."
            self.assertEqual(self.fields(reg, self.DECLARING), [])


if __name__ == "__main__":
    unittest.main()
