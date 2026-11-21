"""AN-8 to AN-10, AN-14 to AN-16, AN-18: the keeper end to end through the command line (tests/README.md section 13).

Every call is a subprocess of `tests/fixtures/ainative/run_cli.py --json ...` (the synthetic pack injected, as in
`test_an_e2e.py`); every decision comes from a file; stderr is empty on every call (AN-1). The keeper is a subprocess of
the test user (`tests/_keeper.py`). A second candidate in one project comes from the test-only `train` decision key
`noise_override` (preflight finding 1): on `toy.jsonl` (exact match, margin 0.05) `noise_override` 1.0 gates
INCONCLUSIVE at alpha 0.02375 and the rerun at the pipeline's noise 0 gates PASS at 0.011586107554417413.

Tampering: every signed copy in a project (the registration, each receipt line, the gate result, report.json,
report.md, the key copy `trust/keeper.pub`) has one value edited in turn; `verify --keeper` exits 3 naming AN-14.
An untouched project gets exit 1 from `verify --keeper` (the test keeper runs as the owner's uid, which can read its
key, so it cannot anchor trust) and exit 0 from `verify --fingerprint`.
`project run report` on an edited registration refuses: AN-2 (the project's hash pin), and AN-10 when the pin is
rewritten to match (the record differs from the keeper's signed registration). No sealed input appears in any stdout, stderr, project file or the keeper's log.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from _keeper import KeeperProcess, skip_unless_linux
from _support import FIXTURES, cli_env

from gatedtrain.ainative import alpha_investing as ai, signing
from gatedtrain.ainative.keeper import texts
from gatedtrain.ainative.keeper_trust import KEEPER_PUB, RECEIPTS
from gatedtrain.ainative.verify import SAME_USER_KEEPER
from gatedtrain.canonical import sha256_bytes
from gatedtrain.testing import synthetic_pack as pack

WRAPPER = FIXTURES / "ainative" / "run_cli.py"
TOY = FIXTURES / "ainative" / "toy.jsonl"
EXIT_OK, EXIT_NOT_PASS, EXIT_REFUSED = 0, 1, 3
MARGIN = 0.05
ALPHA1 = 0.02375                                         # W0 / 2
BID2 = 0.011586107554417413                              # the bid after one attempt that did not pass
BEFORE_EVAL = ("intake_seal", "profile_data", "choose_task_and_base")
TO_TRAIN = ("preregister", "build_train_set", "write_recipe", "smoke", "train")
DECISIONS = {"intake_seal": {"accept_inconclusive": True},
             "preregister": {"metric_confirmed": True, "margin": MARGIN}}
REGISTRATION = "stages/preregister/registration.json"
GATE_RESULT = "stages/gate/gate_result.json"
REPORT_JSON = "stages/report/report.json"
REPORT_MD = "stages/report/report.md"


@skip_unless_linux
class KeeperEndToEndTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.kp = KeeperProcess()
        self.kp.__enter__()
        self.addCleanup(self.kp.__exit__, None, None, None)
        self.outputs: list[str] = []                     # every stdout and stderr, for the sealed-input check
        self.data = self.tmp / "data.jsonl"              # one data file: every project shares its sealed set
        self.data.write_bytes(TOY.read_bytes())
        self._n = 0

    # the command line -----------------------------------------------------------------------------------------------
    def cli(self, *args: str, expect: int | None = EXIT_OK) -> dict:
        proc = subprocess.run([sys.executable, str(WRAPPER), "--json", *args], cwd=self.tmp, env=cli_env(),
                              capture_output=True, text=True, timeout=300)
        self.outputs += [proc.stdout, proc.stderr]
        where = f"{args!r}\nstdout: {proc.stdout[-3000:]}\nstderr: {proc.stderr[-3000:]}"
        self.assertEqual(proc.stderr, "", where)                                  # AN-1
        self.assertEqual(proc.stdout.count("\n"), 1, where)
        doc = json.loads(proc.stdout)
        self.assertEqual((doc["schema"], doc["exit"]), ("gatedtrain.cli_result", proc.returncode), where)
        if expect is not None:
            self.assertEqual(proc.returncode, expect, where)
        return doc

    def decision(self, doc: dict) -> str:
        self._n += 1
        path = self.tmp / f"decision-{self._n}.json"
        path.write_text(json.dumps(doc))
        return str(path)

    def create(self, name: str) -> Path:
        root = self.tmp / name
        doc = self.cli("project", "create", str(root), "--goal", "lookup", "--data", str(self.data),
                       "--pipeline", str(pack.SCENARIO.pipeline), "--keeper", str(self.kp.socket))
        self.assertEqual(doc["data"]["keeper"]["fingerprint"], self.fingerprint())
        return root

    def stage(self, root: Path, ref: str, decision: dict | None = None, *, expect: int | None = EXIT_OK) -> dict:
        decision = DECISIONS.get(ref) if decision is None else decision
        extra = ["--decision", self.decision(decision)] if decision else []
        return self.cli("project", "run", str(root), ref, *extra, expect=expect)

    def to_train(self, root: Path, train: dict | None = None) -> None:
        for ref in BEFORE_EVAL:
            self.stage(root, ref)
        doc = self.cli("keeper", "confirm", str(root), "--decision",
                       self.decision({"metric_confirmed": True, "scorer": "exact_match", "margin": MARGIN}), "--yes")
        self.assertEqual(doc["command"], "keeper confirm")
        for ref in TO_TRAIN:
            self.stage(root, ref, train if ref == "train" else None)

    def gate(self, root: Path, expect: int) -> dict:
        self.stage(root, "evaluate_sealed")
        self.stage(root, "gate", expect=expect)
        return self.read(root, GATE_RESULT)

    def verify(self, root: Path, expect: int = EXIT_NOT_PASS) -> dict:
        """`verify --keeper`: exit 1 when every record checks, since the test keeper runs as the owner's uid and so
        cannot anchor trust (final review finding 2); 3 on a refusal."""
        doc = self.cli("verify", str(root), "--keeper", str(self.kp.socket), expect=expect)
        if expect == EXIT_NOT_PASS:
            self.assertEqual((doc["problems"], doc["warnings"]), ([], [SAME_USER_KEEPER]), doc)
        return doc

    # helpers --------------------------------------------------------------------------------------------------------
    def fingerprint(self) -> str:
        return signing.fingerprint(signing.load_public_pem(
            self.kp.client.call("hello", None, {})["public_pem"].encode("ascii")))

    @staticmethod
    def read(root: Path, rel: str) -> dict:
        return json.loads((root / rel).read_text())

    def assert_refused(self, doc: dict, rule: str) -> None:
        self.assertFalse(doc["ok"], doc)
        self.assertEqual(doc["problems"][0]["rule"], rule, doc)

    def assert_no_sealed_input(self, roots: list[Path]) -> None:
        """No input outside the train part (of any project here) in any output, project file or the keeper's log."""
        every = {json.loads(line)["input"] for line in TOY.read_text().splitlines() if line}
        train: set[str] = set()
        for root in roots:
            train |= {r["input"] for r in self.read(root, "stages/intake_seal/train_part.json")["rows"]}
        sealed = every - train
        self.assertGreater(len(sealed), 100)
        pattern = re.compile(r"(?<![\w-])(?:" + "|".join(re.escape(s) for s in sorted(sealed, key=len, reverse=True))
                             + r")(?![\w-])")
        seen = {f"output {i}": t for i, t in enumerate(self.outputs)}
        seen["keeper log"] = self.kp.output()
        for root in roots:
            for path in root.rglob("*"):
                if path.is_file():
                    seen[str(path.relative_to(self.tmp))] = path.read_text(encoding="utf-8", errors="replace")
        for where, text in seen.items():
            hit = pattern.search(text)
            self.assertIsNone(hit, f"{where} holds the sealed input {hit and hit.group(0)!r}")

    # tests ----------------------------------------------------------------------------------------------------------
    def test_a_keeper_project_passes_and_verifies(self):              # AN-8 to AN-10, AN-14, AN-18
        root = self.create("proj")
        self.to_train(root)
        g = self.gate(root, EXIT_OK)
        self.assertEqual((g["status"], g["alpha"], g["alpha_spent"], g["alpha_earned"]),
                         ("PASS", ALPHA1, 0, ai.PAYOUT))          # a PASS costs nothing
        doc = self.stage(root, "report")
        self.assertEqual(doc["data"]["stage"], "report")
        doc = self.cli("project", "report", str(root))
        self.assertEqual(doc["data"]["report"]["status"], "PASS")
        status = self.cli("project", "status", str(root))
        self.assertEqual({v["status"] for v in status["data"]["stages"].values()}, {"done"})
        self.assertEqual(status["data"]["gate"], "PASS")
        doc = self.verify(root)
        self.assertEqual((doc["data"]["anchor"], doc["data"]["fingerprint"]), ("keeper", self.fingerprint()))
        self.assertEqual(doc["warnings"], [SAME_USER_KEEPER])
        doc = self.cli("verify", str(root), "--fingerprint", self.fingerprint())
        self.assertEqual(doc["data"]["anchor"], "fingerprint")
        self.assert_no_sealed_input([root])

    def test_two_attempts_in_one_project(self):                       # AN-9, AN-10, AN-15, AN-16
        root = self.create("proj")
        self.to_train(root, {"noise_override": 1.0})
        g1 = self.gate(root, EXIT_NOT_PASS)
        self.assertEqual((g1["status"], g1["alpha"], g1["alpha_earned"]), ("INCONCLUSIVE", ALPHA1, 0.0))
        self.assertAlmostEqual(g1["alpha_spent"], ALPHA1 / (1 - ALPHA1))
        self.stage(root, "train", {"rerun": True})                    # the pipeline's noise 0
        g2 = self.gate(root, EXIT_OK)
        self.assertEqual((g2["status"], g2["alpha"], g2["alpha_earned"]), ("PASS", BID2, ai.PAYOUT))
        self.assertEqual(BID2, ai.bid(ai.INITIAL_WEALTH - ALPHA1 / (1 - ALPHA1)))
        self.assertNotEqual(g1["attempt_id"], g2["attempt_id"])

        self.stage(root, "report")
        report = self.read(root, REPORT_JSON)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual([(a["attempt_id"], a["alpha"], a["status"]) for a in report["attempts"]],
                         [(g1["attempt_id"], ALPHA1, "INCONCLUSIVE"), (g2["attempt_id"], BID2, "PASS")])
        ledger = report["ledger"]
        self.assertEqual([(e["j"], e["alpha"], e["passed"], e["earned"]) for e in ledger],
                         [(1, ALPHA1, False, 0.0), (2, BID2, True, ai.PAYOUT)])
        self.assertAlmostEqual(ledger[0]["cost"], ALPHA1 / (1 - ALPHA1))
        self.assertEqual(ledger[1]["cost"], 0)
        self.assertEqual(report["alpha_rule"]["guarantee"], ai.GUARANTEE)
        self.assertIn(ai.GUARANTEE, (root / REPORT_MD).read_text())
        self.assertIn("mFDR", ai.GUARANTEE)
        self.verify(root)

        self.stage(root, "train", {"rerun": True, "noise_override": 0.5})   # a third candidate
        doc = self.stage(root, "evaluate_sealed", expect=EXIT_REFUSED)      # the report closed the set
        self.assert_refused(doc, "AN-16")
        self.assert_no_sealed_input([root])

    def test_projects_on_the_same_data_share_the_wealth(self):        # AN-15, AN-16
        a = self.create("a")
        self.to_train(a, {"noise_override": 1.0})
        ga = self.gate(a, EXIT_NOT_PASS)
        self.assertEqual((ga["status"], ga["alpha"]), ("INCONCLUSIVE", ALPHA1))

        b = self.create("b")                                         # taken in before a's report
        self.to_train(b)
        gb = self.gate(b, EXIT_OK)
        self.assertEqual((gb["status"], gb["alpha"]), ("PASS", BID2))   # the shared wealth's bid, not 0.02375
        status = self.cli("project", "status", str(b))
        self.assertEqual([(e["j"], e["alpha"]) for e in status["data"]["alpha_ledger"]], [(2, BID2)])

        self.stage(a, "report")                                       # a's report closes the shared set
        self.assertEqual(self.read(a, REPORT_JSON)["status"], "INCONCLUSIVE")
        self.assertEqual([e["j"] for e in self.read(a, REPORT_JSON)["ledger"]], [1, 2])
        c = self.create("c")
        doc = self.stage(c, "intake_seal", expect=EXIT_REFUSED)
        self.assert_refused(doc, "AN-16")
        self.assertIn("closed", doc["problems"][0]["message"])
        self.assertFalse((c / "stages" / "intake_seal").exists())
        for root in (a, b):
            self.verify(root)
        self.assert_no_sealed_input([a, b])

    def test_tampering_with_any_signed_copy_is_refused(self):         # AN-14
        root = self.create("proj")
        self.to_train(root, {"noise_override": 1.0})
        self.gate(root, EXIT_NOT_PASS)
        self.stage(root, "train", {"rerun": True})
        self.gate(root, EXIT_OK)
        self.stage(root, "report")
        self.verify(root)

        def tampered(rel: str, edit, what: str) -> None:
            path = root / rel
            saved = path.read_bytes()
            changed = edit(saved)
            self.assertNotEqual(changed, saved, what)
            path.write_bytes(changed)
            try:
                doc = self.verify(root, EXIT_REFUSED)
                self.assert_refused(doc, "AN-14")
            finally:
                path.write_bytes(saved)

        def json_edit(key: str, value):
            def edit(raw: bytes) -> bytes:
                doc = json.loads(raw)
                self.assertIn(key, doc)
                self.assertNotEqual(doc[key], value)
                return json.dumps({**doc, key: value}).encode()
            return edit

        tampered(REGISTRATION, json_edit("alpha", 0.05), "registration alpha")
        tampered(GATE_RESULT, json_edit("status", "FAIL"), "gate status")
        tampered(GATE_RESULT, json_edit("alpha", 0.05), "gate alpha")
        tampered(REPORT_JSON, json_edit("status", "FAIL"), "report status")
        tampered(REPORT_MD, lambda b: b.replace(b"PASS", b"FAIL", 1), "report.md status")
        tampered(KEEPER_PUB, lambda _b: signing.public_pem(Ed25519PrivateKey.generate()), "replaced key copy")

        lines = (root / RECEIPTS).read_bytes().splitlines(keepends=True)
        kinds = [json.loads(x)["kind"] for x in lines]
        for kind in ("registration", "metric_confirmation", "claim", "ledger_entry", "gate_summary", "report"):
            self.assertIn(kind, kinds)
        for i, line in enumerate(lines):
            env = json.loads(line)
            key = next(k for k in sorted(env["record"]) if k not in ("schema", "schema_version"))
            old = env["record"][key]
            new = "x" if not isinstance(old, str) else old + "x"
            changed = json.dumps({**env, "record": {**env["record"], key: new}}).encode() + b"\n"
            tampered(RECEIPTS, lambda b, i=i, changed=changed: b"".join([*lines[:i], changed, *lines[i + 1:]]),
                     f"receipt line {i} ({env['kind']}.{key})")
        self.verify(root)                                               # every copy restored

    def test_a_tampered_registration_refuses_the_report(self):        # AN-14
        root = self.create("proj")
        self.to_train(root)
        self.gate(root, EXIT_OK)
        path = root / REGISTRATION
        saved = path.read_bytes()
        doc = json.loads(saved)
        self.assertNotEqual(doc["alpha"], 0.05)
        path.write_text(json.dumps({**doc, "alpha": 0.05}))
        doc = self.stage(root, "report", expect=EXIT_REFUSED)
        self.assert_refused(doc, "AN-2")                               # the project's own hash pin
        self.assertFalse((root / "stages" / "report").exists())

        path.write_bytes(saved)                                # the hash pin rewritten too: the keeper's copy
        forged = json.dumps({**json.loads(saved), "alpha": 0.05}, separators=(",", ":")).encode()
        old, new = (sha256_bytes(b).removeprefix("sha256:").encode() for b in (saved, forged))
        for f in root.rglob("*"):
            if f.is_file() and old in f.read_bytes():
                f.write_bytes(f.read_bytes().replace(old, new))
        path.write_bytes(forged)
        doc = self.stage(root, "report", expect=EXIT_REFUSED)
        self.assert_refused(doc, "AN-10")                               # differs from the keeper's signed copy
        self.assertFalse((root / "stages" / "report").exists())
        self.assert_refused(self.verify(root, EXIT_REFUSED), "AN-14")


class ReadmeTest(unittest.TestCase):
    def test_the_readme_quotes_the_keeper_texts_exactly(self):
        readme = " ".join((Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8").split())
        for text in (texts.KEEPER_GUARANTEE, texts.KEEPER_SAME_USER_GUARANTEE, ai.GUARANTEE):
            self.assertIn(f'"{text}"', readme)
        for value in ("W0 = 0.05 * (1 - 0.05) = 0.0475", "alpha_j = W_{j-1} / 2", "alpha_j / (1 - alpha_j)",
                      "payout 0.05", "below 0.005", "W0 / 2 = 0.02375"):
            self.assertIn(value, readme)


if __name__ == "__main__":
    unittest.main()
