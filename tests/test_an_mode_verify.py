"""AN-14, AN-27, AN-28, AN-29: the keeper mode in the client, `verify`, the report and every plain-words surface
(tests/README.md section 13).

The keeper is a subprocess of the test user, so it is a same-user keeper (the mode the kernel confirms here). A keeper
in separate-user mode cannot be started without a second OS user, so a test that needs one lets the keeper claim it
(`strong_keeper`: its `hello` says `separate_user` and the kernel's answer is injected, as it would be for a listener
of another uid). Reports are re-signed with the keeper's own key to build the other texts and modes. Uids come from the
running process.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import unittest
from unittest import mock

from _keeper import skip_unless_linux

from test_an_keeper_cli import CliProject, run, run_text
from gatedtrain import cli
from gatedtrain.ainative import signing
from gatedtrain.ainative.keeper import modes, protocol, texts
from gatedtrain.ainative.keeper.protocol import KeeperClient
from gatedtrain.ainative.keeper_trust import RECEIPTS, KeeperTrust
from gatedtrain.ainative.trust_stages import render_markdown
from gatedtrain.ainative.verify import (MODE_UNKNOWN, OFFLINE_LIMIT, OWNER_RUN_REPORT, ROOT_DOCKER_REPORT, SAME_USER_KEEPER,
                                        SAME_USER_REPORT, UNKNOWN_GUARANTEE_REPORT)
from gatedtrain.ainative.verify_mode import ACCEPTED_WEAKER_MODE
from gatedtrain.errors import Refused

OTHER_UID = os.geteuid() + 1


@skip_unless_linux
class ModeCase(CliProject, unittest.TestCase):
    def setUp(self):
        self.start()

    def public(self):
        return signing.load_private_key(self.kp.state / "keeper.key").public_key()

    @contextlib.contextmanager
    def strong_keeper(self, attested: int | None = OTHER_UID, proj=None, **hello):
        """The keeper claims separate-user mode (and no rootful access unless `hello` says so), and the kernel's
        answer about its listener is `attested`. With `proj`, its `records` answer is the project's receipts (the
        keeper signed what a test re-signed with its key)."""
        real = KeeperClient.call

        def call(client, op, project, args):
            if op == "records" and proj is not None:
                return [json.loads(x) for x in (proj.root / RECEIPTS).read_bytes().splitlines() if x]
            out = real(client, op, project, args)
            if op == "hello":
                out = {**out, "same_user": False, "keeper_mode": modes.SEPARATE_USER, "owner_has_root_docker": False,
                       **hello}
            return out
        with mock.patch.object(KeeperClient, "call", call), \
                mock.patch.object(protocol, "attested_listener_uid", lambda sock: attested):
            yield

    def resign(self, proj, **changes) -> dict:
        """Re-sign the project's report with the keeper's key after `changes` (a value `...` deletes the field), and
        refresh the receipts and report.md."""
        key = signing.load_private_key(self.kp.state / "keeper.key")
        rel = proj.root / "stages" / "report"
        doc = json.loads((rel / "report.json").read_text())
        if "sealed_guarantee" in changes:
            doc["known_limits"][0]["text"] = changes["sealed_guarantee"]
        for k, v in changes.items():
            if v is ...:
                doc.pop(k, None)
            else:
                doc[k] = v
        env = signing.sign(key, "report", doc)
        lines = [json.loads(x) for x in (proj.root / RECEIPTS).read_bytes().splitlines() if x]
        lines = [env if e["kind"] == "report" else e for e in lines]
        (proj.root / RECEIPTS).write_bytes(b"".join(json.dumps(e).encode() + b"\n" for e in lines))
        (rel / "report.json").write_text(json.dumps(doc))
        try:
            md = render_markdown(doc)
        except (KeyError, TypeError):                       # a report that names no valid mode cannot be rendered
            md = "not renderable\n"
        (rel / "report.md").write_text(md, encoding="utf-8")
        return doc

    def uniform(self, proj, mode: str) -> dict:
        """Every signed statement of the project (receipts, report) says `mode`, with the text that goes with it."""
        self.restamp(proj, mode)
        text = texts.KEEPER_GUARANTEE if mode == "separate_user" else texts.KEEPER_OWNER_RUN_GUARANTEE
        doc = json.loads((proj.root / "stages" / "report" / "report.json").read_text())
        return self.resign(proj, sealed_guarantee=text, keeper_mode=mode,
                           modes_seen=[{**m, "mode": mode} for m in doc["modes_seen"]],
                           attempts=[{**a, "keeper_mode": mode} for a in doc["attempts"]])

    def as_v2(self, proj, text: str) -> dict:
        """The report the way a keeper before AN-27 signed it: version 2 without the mode fields."""
        doc = json.loads((proj.root / "stages" / "report" / "report.json").read_text())
        attempts = [{k: v for k, v in a.items() if k != "keeper_mode"} for a in doc["attempts"]]
        return self.resign(proj, schema_version=2, keeper_mode=..., modes_seen=..., require_keeper_mode=...,
                           attempts=attempts, sealed_guarantee=text)

    def project_at_report(self):
        proj = self.create()
        self.to_report(proj)
        return proj, str(proj.root), str(self.kp.socket), signing.fingerprint(self.public())


class VerifyMatrixTest(ModeCase):
    def test_a_same_user_keeper_exits_1_and_states_its_mode(self):
        proj, root, sock, fp = self.project_at_report()
        code, doc = run(["verify", root, "--keeper", sock])
        d = doc["data"]
        self.assertEqual((code, doc["ok"], doc["problems"]), (1, False, []), doc)
        self.assertEqual((d["keeper_mode"], d["effective_mode"], d["same_user"]), ("same_user",) * 2 + (True,))
        self.assertEqual({k: v for k, v in d["mode_sources"].items() if k != "receipts"},
                         {"keeper_claim": "same_user", "kernel_attestation": "same_user", "report": ["same_user"]})
        self.assertEqual(set(d["mode_sources"]["receipts"]), {"same_user"})
        self.assertEqual(doc["warnings"], [SAME_USER_KEEPER, OWNER_RUN_REPORT])
        self.assertNotIn("accepted_weaker_mode", d)

    def test_accept_mode_turns_exactly_a_same_user_result_into_exit_0(self):
        proj, root, sock, fp = self.project_at_report()
        code, doc = run(["verify", root, "--keeper", sock, "--accept-mode", "same-user"])
        self.assertEqual((code, doc["ok"], doc["data"]["accepted_weaker_mode"]), (0, True, True), doc)
        self.assertIn(OWNER_RUN_REPORT, doc["warnings"])                   # the mode warning stays
        code, text, err = run_text(["verify", root, "--keeper", sock, "--accept-mode", "same-user"])
        self.assertEqual(code, 0)
        self.assertEqual(text.splitlines()[-1], ACCEPTED_WEAKER_MODE)
        self.assertEqual(ACCEPTED_WEAKER_MODE, "ACCEPTED WEAKER MODE: same-user (procedure followed; sealed data not "
                                               "protected from the user's account)")
        self.assertIn(OWNER_RUN_REPORT, err)
        code, doc = run(["verify", root, "--fingerprint", fp, "--accept-mode", "same-user"])   # a signed same-user report
        self.assertEqual((code, doc["data"]["accepted_weaker_mode"]), (0, True), doc)
        code, doc = run(["verify", root, "--accept-mode", "separate-user"])            # no such value
        self.assertEqual(code, 2, doc)

    def test_accept_mode_does_not_accept_another_weaker_statement(self):
        proj, root, sock, fp = self.project_at_report()
        for text, warning in ((texts.KEEPER_ROOT_DOCKER_GUARANTEE, ROOT_DOCKER_REPORT),
                              ("a guarantee no version states", UNKNOWN_GUARANTEE_REPORT)):
            with self.subTest(text[:20]):
                self.uniform(proj, "separate_user")
                self.resign(proj, sealed_guarantee=text, keeper_mode=modes.mode_of_text(text) or "same_user")
                code, doc = run(["verify", root, "--fingerprint", fp, "--accept-mode", "same-user"])
                self.assertEqual(code, 1, doc)
                self.assertNotIn("accepted_weaker_mode", doc["data"])
                self.assertIn(warning, doc["warnings"])
        code, doc = run(["verify", root, "--accept-mode", "same-user"])                   # no anchor
        self.assertEqual(code, 1, doc)

    def test_the_strong_result_is_exit_0_when_claim_kernel_and_report_agree(self):
        proj, root, sock, fp = self.project_at_report()
        self.uniform(proj, "separate_user")
        with self.strong_keeper(proj=proj):
            code, doc = run(["verify", root, "--keeper", sock])
        self.assertEqual((code, doc["ok"], doc["warnings"]), (0, True, []), doc)
        self.assertEqual({k: v for k, v in doc["data"]["mode_sources"].items() if k != "receipts"},
                         {"keeper_claim": "separate_user", "kernel_attestation": "separate_user",
                          "report": ["separate_user"]})
        self.assertEqual((doc["data"]["keeper_mode"], doc["data"]["effective_mode"]), ("separate_user",) * 2)
        code, doc = run(["verify", root, "--fingerprint", fp])                          # the signed report is the source
        self.assertEqual((code, doc["data"]["effective_mode"]), (0, "separate_user"), doc)

    def test_a_strong_keeper_with_a_same_user_report_is_exit_1(self):
        proj, root, sock, fp = self.project_at_report()                               # the report states same-user
        with self.strong_keeper():
            code, doc = run(["verify", root, "--keeper", sock])
            self.assertEqual((code, doc["warnings"], doc["data"]["effective_mode"]),
                             (1, [OWNER_RUN_REPORT], "same_user"), doc)
            code, doc = run(["verify", root, "--keeper", sock, "--accept-mode", "same-user"])
            self.assertEqual((code, doc["data"]["accepted_weaker_mode"]), (0, True), doc)

    def test_a_rootful_strong_keeper_is_not_accepted_as_same_user(self):
        proj, root, sock, fp = self.project_at_report()
        self.uniform(proj, "separate_user")
        with self.strong_keeper(proj=proj, owner_has_root_docker=True):
            code, doc = run(["verify", root, "--keeper", sock, "--accept-mode", "same-user"])
        self.assertEqual((code, doc["data"]["root_docker"]), (1, True), doc)
        self.assertNotIn("accepted_weaker_mode", doc["data"])

    def test_the_kernel_contradicting_the_claim_is_refused(self):                        # AN-29
        proj, root, sock, fp = self.project_at_report()
        with self.strong_keeper(attested=os.geteuid()):
            code, doc = run(["verify", root, "--keeper", sock, "--accept-mode", "same-user"])
        self.assertEqual((code, doc["problems"][0]["rule"]), (3, "AN-29"), doc)
        self.assertIn("process of your own uid", doc["problems"][0]["fix"])

    def test_missing_attestation_never_upgrades(self):
        proj, root, sock, fp = self.project_at_report()
        self.uniform(proj, "separate_user")
        with self.strong_keeper(attested=None, proj=proj):
            code, doc = run(["verify", root, "--keeper", sock])
            self.assertEqual((code, doc["data"]["effective_mode"], doc["data"]["mode_sources"]["kernel_attestation"]),
                             (1, "same_user", None), doc)
            self.assertIn("kernel attestation unavailable", doc["data"]["mode_note"])
            code, doc = run(["verify", root, "--keeper", sock, "--accept-mode", "same-user"])
            self.assertEqual((code, doc["data"]["accepted_weaker_mode"]), (0, True), doc)

    def test_a_report_whose_mode_contradicts_its_text_or_names_no_mode_is_refused(self):   # AN-14
        proj, root, sock, fp = self.project_at_report()
        for changes in ({"keeper_mode": "separate_user"},                    # the report keeps the same-user text
                        {"keeper_mode": "banana"}, {"keeper_mode": None}, {"keeper_mode": ...}):
            with self.subTest(changes):
                self.resign(proj, **{"sealed_guarantee": texts.KEEPER_OWNER_RUN_GUARANTEE, **changes})
                code, doc = run(["verify", root, "--fingerprint", fp])
                self.assertEqual((code, doc["problems"][0]["rule"]), (3, "AN-14"), doc)
                self.assertIn("keeper_mode", doc["problems"][0]["field"])

    def test_a_version_2_report_still_verifies_with_its_mode_from_its_text(self):
        proj, root, sock, fp = self.project_at_report()
        self.restamp(proj, "separate_user")                     # a keeper whose records state the strong mode
        self.as_v2(proj, texts.KEEPER_GUARANTEE)
        code, doc = run(["verify", root, "--fingerprint", fp])
        self.assertEqual((code, doc["warnings"]), (0, [OFFLINE_LIMIT, texts.MODE_STATED_NOTE]), doc)
        self.assertEqual(doc["data"]["mode_sources"]["report"], ["separate_user"])
        self.as_v2(proj, texts.KEEPER_SAME_USER_GUARANTEE)                           # the legacy same-user text
        code, doc = run(["verify", root, "--fingerprint", fp])
        self.assertEqual((code, doc["warnings"], doc["data"]["mode_sources"]["report"]),
                         (1, [OFFLINE_LIMIT, SAME_USER_REPORT, texts.MODE_STATED_NOTE], ["same_user"]), doc)
        code, doc = run(["verify", root, "--fingerprint", fp, "--accept-mode", "same-user"])
        self.assertEqual((code, doc["data"]["accepted_weaker_mode"]), (0, True), doc)
        self.as_v2(proj, "a guarantee no version states")
        code, doc = run(["verify", root, "--fingerprint", fp])
        self.assertEqual((code, doc["data"]["mode_sources"]["report"]), (1, [None]), doc)
        self.assertIn(UNKNOWN_GUARANTEE_REPORT, doc["warnings"])

    def test_a_version_2_report_renders_as_before(self):
        proj, *_ = self.project_at_report()
        doc = self.as_v2(proj, texts.KEEPER_GUARANTEE)
        md = render_markdown(doc)
        self.assertTrue(md.startswith("# PASS"), md[:40])
        self.assertFalse([ln for ln in md.splitlines() if ln.startswith("keeper mode:")])
        self.assertNotIn("same-user keeper", md.splitlines()[0])

    def test_the_registered_requirement_refuses_a_same_user_result_whatever_the_option(self):    # AN-28
        proj, root, sock, fp = self.project_at_report()
        self.resign(proj, require_keeper_mode="separate_user")                       # the report states the policy
        for extra in ([], ["--accept-mode", "same-user"]):
            for anchor in (["--keeper", sock], ["--fingerprint", fp], []):
                with self.subTest(msg=" ".join([*extra, *anchor])), self.strong_keeper(proj=proj):
                    code, doc = run(["verify", root, *anchor, *extra])
                    self.assertEqual((code, doc["problems"][0]["rule"]), (3, "AN-28"), doc)
                    self.assertTrue(doc["problems"][0]["fix"])
        self.uniform(proj, "separate_user")
        self.resign(proj, require_keeper_mode="separate_user")
        code, doc = run(["verify", root, "--fingerprint", fp])
        self.assertEqual((code, doc["data"]["require_keeper_mode"]), (0, "separate_user"), doc)


class ReceiptModesTest(ModeCase):
    """Review round 1: the effective mode folds in every signed statement, never only the keeper's claim."""

    def test_keeper_mode_in_the_data_is_the_effective_mode_not_the_claim(self):
        proj, root, sock, fp = self.project_at_report()
        with self.strong_keeper(attested=None):                         # the keeper claims separate_user
            code, doc = run(["verify", root, "--keeper", sock])
        d = doc["data"]
        self.assertEqual((code, d["keeper_mode"], d["effective_mode"]), (1, "same_user", "same_user"), doc)
        self.assertEqual(d["mode_sources"]["keeper_claim"], "separate_user")

    def test_a_report_stronger_than_the_receipts_it_covers_is_refused(self):                 # AN-14
        proj, root, sock, fp = self.project_at_report()                 # the receipts state same_user
        self.resign(proj, sealed_guarantee=texts.KEEPER_GUARANTEE, keeper_mode="separate_user")
        for anchor in (["--fingerprint", fp], []):
            code, doc = run(["verify", root, *anchor])
            self.assertEqual((code, doc["problems"][0]["rule"]), (3, "AN-14"), doc)
            self.assertIn("keeper_mode", doc["problems"][0]["field"])
        self.as_v2(proj, texts.KEEPER_GUARANTEE)                       # a version 2 report states its text's mode
        code, doc = run(["verify", root, "--fingerprint", fp])
        self.assertEqual((code, doc["problems"][0]["rule"]), (3, "AN-14"), doc)

    def test_a_uniform_strong_project_verifies_and_lists_the_receipt_modes(self):
        proj, root, sock, fp = self.project_at_report()
        self.uniform(proj, "separate_user")
        code, doc = run(["verify", root, "--fingerprint", fp])
        self.assertEqual((code, doc["data"]["effective_mode"], doc["warnings"]), (0, "separate_user", [OFFLINE_LIMIT, texts.MODE_STATED_NOTE]),
                         doc)
        self.assertEqual(set(doc["data"]["mode_sources"]["receipts"]), {"separate_user"})

    def test_one_same_user_receipt_makes_the_project_same_user(self):
        proj, root, sock, fp = self.project_at_report()
        self.uniform(proj, "separate_user")
        key = signing.load_private_key(self.kp.state / "keeper.key")
        lines = [json.loads(x) for x in (proj.root / RECEIPTS).read_bytes().splitlines() if x]
        claim = next(e for e in lines if e["kind"] == "claim")
        rec = {k: v for k, v in claim["record"].items() if k not in ("schema", "schema_version", "envelope_sha256")}
        lines[lines.index(claim)] = signing.sign(key, "claim", {**rec, "keeper_mode": "same_user"})
        (proj.root / RECEIPTS).write_bytes(b"".join(json.dumps(e).encode() + b"\n" for e in lines))
        code, doc = run(["verify", root, "--fingerprint", fp])      # the report's attempt and mode say separate_user
        self.assertEqual((code, doc["problems"][0]["rule"]), (3, "AN-14"), doc)

    def test_attempts_or_modes_seen_that_contradict_the_receipts_are_refused(self):          # AN-14
        proj, root, sock, fp = self.project_at_report()
        base = self.uniform(proj, "same_user")
        for changes in ({"attempts": [{**a, "keeper_mode": "separate_user"} for a in base["attempts"]]},
                        {"modes_seen": [{**m, "mode": "separate_user"} for m in base["modes_seen"]]}):
            with self.subTest(list(changes)):
                self.resign(proj, **changes)
                code, doc = run(["verify", root, "--fingerprint", fp])
                self.assertEqual((code, doc["problems"][0]["rule"]), (3, "AN-14"), doc)
                self.resign(proj, **{k: base[k] for k in changes})
        code, doc = run(["verify", root, "--fingerprint", fp])
        self.assertEqual(code, 1, doc)

    def test_a_v3_report_without_an_attempt_mode_is_refused_not_a_crash(self):               # AN-14
        proj, root, sock, fp = self.project_at_report()
        base = self.uniform(proj, "same_user")
        stripped = [{k: v for k, v in a.items() if k != "keeper_mode"} for a in base["attempts"]]
        self.resign(proj, attempts=stripped)
        code, doc = run(["verify", root, "--fingerprint", fp])
        self.assertEqual((code, doc["problems"][0]["rule"]), (3, "AN-14"), doc)
        self.assertIn("attempts", doc["problems"][0]["field"])
        self.resign(proj, attempts=[{**a, "keeper_mode": "banana"} for a in base["attempts"]])
        code, doc = run(["verify", root, "--fingerprint", fp])
        self.assertEqual((code, doc["problems"][0]["rule"]), (3, "AN-14"), doc)

    def test_a_project_without_a_report_states_its_receipts_mode_and_exits_1(self):
        proj = self.create()
        self.to_report(proj, until="gate")
        root, fp = str(proj.root), signing.fingerprint(self.public())
        code, doc = run(["verify", root, "--fingerprint", fp])
        self.assertEqual((code, doc["data"]["effective_mode"], doc["data"]["keeper_mode"]),
                         (1, "same_user", "same_user"), doc)
        self.assertTrue(doc["warnings"])
        code, doc = run(["verify", root, "--fingerprint", fp, "--accept-mode", "same-user"])
        self.assertEqual((code, doc["data"]["accepted_weaker_mode"]), (0, True), doc)

    def test_a_receipt_mode_that_is_not_a_mode_is_refused(self):
        proj = self.create()
        self.to_report(proj, until="gate")
        self.restamp(proj, "banana")
        code, doc = run(["verify", str(proj.root), "--fingerprint", signing.fingerprint(self.public())])
        self.assertEqual((code, doc["problems"][0]["rule"]), (3, "AN-14"), doc)

    def test_receipts_without_any_mode_say_mode_unknown_and_exit_1(self):
        proj = self.create()
        self.to_report(proj, until="gate")
        self.restamp(proj, None)                                       # signed before AN-27
        root, fp = str(proj.root), signing.fingerprint(self.public())
        for extra in ([], ["--accept-mode", "same-user"]):
            code, doc = run(["verify", root, "--fingerprint", fp, *extra])
            self.assertEqual((code, doc["data"]["effective_mode"], doc["ok"]), (1, None, False), doc)
            self.assertIn(MODE_UNKNOWN, doc["warnings"])
            self.assertNotIn("accepted_weaker_mode", doc["data"])

    def test_a_separate_user_requirement_with_no_determinable_mode_fails_an_28(self):
        proj = self.create()
        self.to_report(proj, until="gate")
        self.restamp(proj, None, require_keeper_mode="separate_user")
        for extra in ([], ["--accept-mode", "same-user"]):
            code, doc = run(["verify", str(proj.root), "--fingerprint", signing.fingerprint(self.public()), *extra])
            self.assertEqual((code, doc["problems"][0]["rule"]), (3, "AN-28"), doc)

    def test_accepting_same_user_offline_keeps_the_consistency_warnings(self):
        proj, root, sock, fp = self.project_at_report()
        code, strict = run(["verify", root, "--fingerprint", fp])
        code, doc = run(["verify", root, "--fingerprint", fp, "--accept-mode", "same-user"])
        self.assertEqual(code, 0, doc)
        self.assertIn(OFFLINE_LIMIT, doc["warnings"])
        code, doc = run(["verify", root, "--accept-mode", "same-user"])                       # no anchor at all
        self.assertEqual(code, 1, doc)
        code, doc = run(["verify", root, "--keeper", sock, "--accept-mode", "same-user"])
        self.assertEqual(code, 0, doc)
        self.assertNotIn(OFFLINE_LIMIT, doc["warnings"])
        with self.strong_keeper(proj=proj, owner_has_root_docker=False):
            self.uniform(proj, "same_user")
            code, doc = run(["verify", root, "--keeper", sock, "--accept-mode", "same-user"])
            self.assertEqual(code, 0, doc)


class DriverAttestationTest(ModeCase):
    def test_the_driver_refuses_a_keeper_that_contradicts_the_kernel(self):                # AN-29
        proj = self.create()
        trust = KeeperTrust(proj)
        with self.strong_keeper(attested=os.geteuid()):
            with self.assertRaises(Refused) as cm:
                trust.check()
        self.assertEqual(cm.exception.problems[0].rule, "AN-29")

    def test_the_driver_exposes_the_effective_mode(self):
        proj = self.create()
        trust = KeeperTrust(proj)
        hello = trust.check()
        self.assertEqual((trust.keeper_mode, hello["effective_mode"]), ("same_user", "same_user"))
        with self.strong_keeper():
            hello = trust.check()
            self.assertEqual((trust.keeper_mode, hello["effective_mode"]), ("separate_user", "separate_user"))
        with self.strong_keeper(attested=None):
            self.assertEqual(trust.check()["effective_mode"], "same_user")
            self.assertEqual(trust.keeper_mode, "same_user")

    def test_stages_refuse_a_contradicting_keeper(self):
        proj = self.create()
        with self.strong_keeper(attested=os.geteuid()):
            with self.assertRaises(Refused) as cm:
                self.go(proj, "intake_seal")
        self.assertEqual(cm.exception.problems[0].rule, "AN-29")


class SurfacesTest(ModeCase):
    """A6: every surface states the mode, and none says "sealed" in a PASS or headline without the qualifier."""

    SAME = texts.MODE_LINE["same_user"]

    def test_1_keeper_fingerprint(self):
        code, text, err = run_text(["keeper", "fingerprint", "--socket", str(self.kp.socket)])
        self.assertEqual(code, 0, err)
        self.assertIn(self.SAME, text.splitlines())
        self.assertIn("harden: gatedtrain setup keeper --mode separate-user", text)
        self.assertIn(texts.MODE_STATED_NOTE, text.splitlines())                 # the limit of an offline statement
        code, doc = run(["keeper", "fingerprint", "--socket", str(self.kp.socket)])
        self.assertEqual(doc["data"]["keeper_mode"], "same_user")
        self.assertIn("reasons", doc["data"]["mode_facts"])
        with self.strong_keeper():
            code, text, _ = run_text(["keeper", "fingerprint", "--socket", str(self.kp.socket)])
            self.assertIn(texts.MODE_LINE["separate_user"], text.splitlines())
            self.assertIn(texts.MODE_STATED_NOTE, text.splitlines())
            self.assertNotIn("harden:", text)

    def test_2_project_create_prints_the_mode_before_any_data_is_taken_in(self):
        data = self.tmp / "d.jsonl"
        from test_an_keeper_trust import TOY
        from gatedtrain.testing import synthetic_pack as pack

        data.write_bytes(TOY)
        argv = ["project", "create", str(self.tmp / "p"), "--goal", "lookup", "--data", str(data), "--pipeline",
                str(pack.SCENARIO.pipeline), "--keeper", str(self.kp.socket)]
        code, text, err = run_text(argv)
        self.assertEqual(code, 0, err)
        self.assertIn(self.SAME, text.splitlines())
        self.assertFalse((self.tmp / "p" / "stages" / "intake_seal").exists(), "data were taken in before the line")
        code, doc = run([*argv[:2], str(self.tmp / "p2"), *argv[3:]])
        self.assertEqual(doc["data"]["keeper"]["keeper_mode"], "same_user")
        self.assertIn("reasons", doc["data"]["keeper"]["mode_facts"])

    def test_3_intake_prints_the_mode_and_records_it(self):
        proj = self.create()
        code, text, err = run_text(["project", "run", str(proj.root), "intake_seal", "--decision",
                                    self.decision_with_accept()])
        self.assertEqual(code, 0, err)
        self.assertIn(self.SAME, text.splitlines())
        summary = json.loads((next((proj.root / "stages").glob("*/sealed_summary.json"))).read_text())
        self.assertEqual((summary["keeper_mode"], summary["guarantee"]),
                         ("same_user", texts.KEEPER_OWNER_RUN_GUARANTEE))

    def decision_with_accept(self) -> str:
        path = self.tmp / "intake-decision.json"
        path.write_text(json.dumps({"accept_inconclusive": True}))
        return str(path)

    def test_4_confirm_prints_the_mode_before_the_sentence(self):
        proj = self.create()
        self.upto_base(proj)
        code, text, err = run_text(["keeper", "confirm", str(proj.root), "--decision", self.decision()],
                                   stdin="yes\n")
        self.assertEqual(code, 0, err)
        lines = err.splitlines()
        self.assertEqual(lines[0], self.SAME)                                  # the prompt block: mode line first
        self.assertTrue(lines[1].startswith("I will count"), lines)
        self.assertIn(self.SAME, text)

    def test_5_and_7_report_and_project_report(self):
        proj = self.create()
        self.to_report(proj)
        md = (proj.root / "stages" / "report" / "report.md").read_text()
        lines = md.splitlines()
        self.assertEqual(lines[0], self.SAME)
        self.assertTrue(lines[2].startswith("# PASS") and lines[2].endswith(texts.HEADLINE_SUFFIX_SAME_USER), lines[:3])
        code, text, err = run_text(["project", "report", str(proj.root)])
        self.assertEqual(code, 0, err)
        self.assertEqual(text.splitlines()[0], self.SAME)
        code, doc = run(["project", "report", str(proj.root)])
        self.assertEqual((doc["data"]["keeper_mode"], doc["data"]["report"]["keeper_mode"]), ("same_user",) * 2)
        self.assertEqual(doc["data"]["report"]["schema_version"], 3)
        self.assertIn(render_markdown(doc["data"]["report"]), md)

    def test_no_surface_says_sealed_next_to_a_headline_without_the_qualifier(self):
        proj = self.create()
        self.to_report(proj)
        md = (proj.root / "stages" / "report" / "report.md").read_text()
        for line in md.splitlines():
            if line.startswith("# "):
                self.assertIn("same-user", line)
        for text in (texts.MODE_LINE["same_user"], texts.HEADLINE_SUFFIX_SAME_USER, texts.KEEPER_OWNER_RUN_GUARANTEE):
            self.assertIn("same-user", text)

    def test_mixed_modes_are_stated_in_the_report(self):
        proj = self.create()
        self.to_report(proj)
        doc = json.loads((proj.root / "stages" / "report" / "report.json").read_text())
        seen = [*doc["modes_seen"], {"mode": "separate_user", "first_utc": doc["modes_seen"][0]["first_utc"]}]
        doc = self.resign(proj, modes_seen=seen)
        md = render_markdown(doc)
        self.assertIn("served in more than one keeper mode", md)
        self.assertIn("the weakest, same_user", md)


class ClientModeSurfaceTest(ModeCase):
    """Review round 1: the plain-words surfaces print the weakest of the keeper's statement and this client's check."""

    def test_effective_mode_is_the_weakest_of_the_statement_and_the_check(self):
        from gatedtrain.cli import _effective_mode

        proj = self.create()
        self.assertEqual(_effective_mode(proj, "separate_user"), ("same_user", None))      # the real keeper is same-user
        self.assertEqual(_effective_mode(proj, "same_user"), ("same_user", None))
        self.assertEqual(_effective_mode(proj, None), ("same_user", None))      # a keeper project without a statement
        from gatedtrain.ainative.project import create_project
        self.assertEqual(_effective_mode(create_project(self.tmp / "nokeeper", goal="g", data=self.tmp / "proj.jsonl", pipeline="x", budget={"gpu_hours": 0, "llm_usd": 0}), None), (None, None))
        with self.strong_keeper():
            self.assertEqual(_effective_mode(proj, "separate_user"), ("separate_user", None))
            self.assertEqual(_effective_mode(proj, "same_user"), ("same_user", None))
        with self.strong_keeper(attested=os.geteuid()), self.assertRaises(Refused):          # AN-29
            _effective_mode(proj, "separate_user")

    def test_project_report_prints_the_weaker_effective_mode(self):
        proj, root, sock, fp = self.project_at_report()
        from gatedtrain.ainative import machine

        real = machine.done_output

        def stronger(project, stage, name):                       # the report states separate_user; the keeper is not
            out = real(project, stage, name)
            return {**out, "keeper_mode": "separate_user"} if stage == "report" else out
        with mock.patch("gatedtrain.cli._sync_receipts", return_value=[]), \
                mock.patch.object(machine, "done_output", stronger):
            code, text, err = run_text(["project", "report", root])
            code2, doc = run(["project", "report", root])
        self.assertEqual(code, 0, err)
        self.assertEqual(text.splitlines()[:1], [texts.MODE_LINE["same_user"]])
        self.assertEqual((doc["data"]["keeper_mode"], doc["data"]["report_keeper_mode"]),
                         ("same_user", "separate_user"))

    def test_project_report_with_a_contradicting_keeper_is_refused(self):
        proj, root, sock, fp = self.project_at_report()
        with mock.patch("gatedtrain.cli._sync_receipts", return_value=[]), self.strong_keeper(attested=os.geteuid()):
            code, doc = run(["project", "report", root])
        self.assertEqual((code, doc["problems"][0]["rule"]), (3, "AN-29"), doc)


class RequireKeeperModeCliTest(ModeCase):
    def argv(self, name: str, *extra: str) -> list[str]:
        from test_an_keeper_trust import TOY
        from gatedtrain.testing import synthetic_pack as pack

        data = self.tmp / f"{name}.jsonl"
        data.write_bytes(TOY)
        return ["project", "create", str(self.tmp / name), "--goal", "lookup", "--data", str(data), "--pipeline",
                str(pack.SCENARIO.pipeline), *extra]

    def test_a_same_user_keeper_is_refused_early_and_nothing_is_created(self):
        code, doc = run(self.argv("p", "--keeper", str(self.kp.socket), "--require-keeper-mode", "separate-user"))
        self.assertEqual((code, doc["problems"][0]["rule"]), (3, "AN-28"), doc)
        self.assertIn("separate-user", doc["problems"][0]["fix"])
        self.assertFalse((self.tmp / "p").exists())

    def test_the_default_is_any_and_the_values_are_checked(self):
        code, doc = run(self.argv("p", "--keeper", str(self.kp.socket), "--require-keeper-mode", "any"))
        self.assertEqual((code, doc["data"]["keeper"]["require_keeper_mode"]), (0, "any"), doc)
        code, doc = run(self.argv("q", "--keeper", str(self.kp.socket), "--require-keeper-mode", "same-user"))
        self.assertEqual(code, 2, doc)

    def test_a_requirement_without_a_keeper_is_refused(self):
        code, doc = run(self.argv("p", "--require-keeper-mode", "separate-user"))
        self.assertEqual((code, doc["problems"][0]["rule"]), (3, "AN-28"), doc)

    def test_the_requirement_reaches_the_keeper(self):
        with self.strong_keeper():
            code, doc = run(self.argv("p", "--keeper", str(self.kp.socket), "--require-keeper-mode", "separate-user"))
        # the CLI's early check passes on the claim; the real keeper still refuses a same-user open (AN-28)
        self.assertEqual((code, doc["problems"][0]["rule"]), (3, "AN-28"), doc)


class OldKeeperCompatTest(ModeCase):
    """Review round 2: a 0.6.0 keeper (no `keeper_mode` in `hello`, `open_project` refuses any argument, AN-12) is
    bound without a half-made project; a requirement it cannot enforce is refused early with a clear message."""

    @contextlib.contextmanager
    def old_keeper(self, claim_separate: bool = False, open_fails: bool = False):
        real = KeeperClient.call
        self.open_args: list[dict] = []

        def call(client, op, project, args):
            if op == "open_project":
                self.open_args.append(dict(args))
                if args:                                           # the 0.6.0 keeper: no such argument
                    raise protocol.refused(f"args.{sorted(args)[0]}", "the op takes no such argument",
                                           "send only: no arguments")
                if open_fails:
                    raise protocol.refused("keeper", "the keeper refuses this project", "try again")
            out = real(client, op, project, args)
            if op == "hello":
                out = {k: v for k, v in out.items() if k not in ("keeper_mode", "mode_facts", "modes_seen")}
                out["same_user"] = not claim_separate
            return out
        with mock.patch.object(KeeperClient, "call", call):
            yield

    def argv(self, name: str, *extra: str) -> list[str]:
        from test_an_keeper_trust import TOY
        from gatedtrain.testing import synthetic_pack as pack

        data = self.tmp / f"{name}.jsonl"
        data.write_bytes(TOY)
        return ["project", "create", str(self.tmp / name), "--goal", "lookup", "--data", str(data), "--pipeline",
                str(pack.SCENARIO.pipeline), "--keeper", str(self.kp.socket), *extra]

    def test_the_default_requirement_sends_no_argument(self):
        with self.old_keeper():
            code, doc = run(self.argv("p"))
        self.assertEqual(code, 0, doc)
        self.assertEqual(self.open_args, [{}])
        self.assertEqual(doc["data"]["keeper"]["require_keeper_mode"], "any")
        self.assertTrue((self.tmp / "p" / "trust" / "keeper.json").is_file())

    def test_a_separate_user_requirement_is_sent_when_it_is_not_any(self):
        with self.strong_keeper():
            real = KeeperClient.call
            seen = []

            def spy(client, op, project, args):
                if op == "open_project":
                    seen.append(dict(args))
                return real(client, op, project, args)
            with mock.patch.object(KeeperClient, "call", spy):
                run(self.argv("p", "--require-keeper-mode", "separate-user"))
        self.assertEqual(seen, [{"require_keeper_mode": "separate_user"}])

    def test_a_requirement_against_a_keeper_that_predates_modes_is_refused_early(self):
        with self.old_keeper(claim_separate=True), \
                mock.patch.object(protocol, "attested_listener_uid", lambda sock: os.geteuid() + 1):
            code, doc = run(self.argv("p", "--require-keeper-mode", "separate-user"))
        self.assertEqual((code, doc["problems"][0]["rule"]), (3, "AN-28"), doc)
        self.assertIn("keeper predates AN-28; upgrade it", doc["problems"][0]["message"])
        self.assertEqual(self.open_args, [])                       # the keeper was not asked to open anything
        self.assertFalse((self.tmp / "p").exists())

    def test_a_refused_binding_leaves_no_project_behind(self):
        with self.old_keeper(open_fails=True):
            code, doc = run(self.argv("p"))
        self.assertEqual(code, 3, doc)
        self.assertFalse((self.tmp / "p").exists(), "a half-made project was left")

    def test_a_refused_binding_leaves_a_pre_existing_empty_directory_empty(self):
        (self.tmp / "p").mkdir()
        with self.old_keeper(open_fails=True):
            code, doc = run(self.argv("p"))
        self.assertEqual(code, 3, doc)
        self.assertEqual(list((self.tmp / "p").iterdir()), [])


class EverySurfaceStatesTheModeTest(ModeCase):
    """Review round 2: every command that prints a result for a keeper-bound project prints the mode line, carries
    `data.keeper_mode` and, in same-user mode, one warning with the same text; the gate result carries the headline
    qualifier."""

    SAME = texts.MODE_LINE["same_user"]

    def stages(self):
        return ["intake_seal", "profile_data", "choose_task_and_base", "preregister", "build_train_set", "write_recipe",
                "smoke", "train", "evaluate_sealed", "gate", "report"]

    def test_every_stage_run_states_the_mode_in_text_and_json(self):
        proj = self.create()
        self.to_report(proj)
        root = str(proj.root)
        accept = self.tmp / "accept.json"
        accept.write_text(json.dumps({"accept_inconclusive": True}))
        from test_an_keeper_trust import MARGIN

        prereg = self.tmp / "prereg.json"                      # the decision `to_report` ran the stage with
        prereg.write_text(json.dumps({"metric_confirmed": True, "margin": MARGIN}))
        for ref in self.stages():
            dec = {"intake_seal": str(accept), "preregister": str(prereg)}.get(ref)
            extra = ["--decision", dec] if dec else []
            code, text, err = run_text(["project", "run", root, ref, *extra])
            self.assertEqual(code, 0, (ref, err))
            self.assertIn(self.SAME, text.splitlines(), ref)
            self.assertNotIn(self.SAME, err, ref)                  # once: the warning is for the JSON envelope
            code, doc = run(["project", "run", root, ref, *extra])
            self.assertEqual(doc["data"]["keeper_mode"], "same_user", ref)
            self.assertEqual(doc["warnings"].count(self.SAME), 1, (ref, doc["warnings"]))

    def test_the_gate_result_line_carries_the_headline_qualifier(self):
        proj = self.create()
        self.to_report(proj)
        code, text, _ = run_text(["project", "run", str(proj.root), "gate"])
        self.assertIn("gate: PASS" + texts.HEADLINE_SUFFIX_SAME_USER, text.splitlines())
        code, text, _ = run_text(["project", "status", str(proj.root)])
        self.assertIn("gate: PASS" + texts.HEADLINE_SUFFIX_SAME_USER, text.splitlines())

    def test_status_states_the_mode_in_text_and_json(self):
        proj = self.create()
        self.to_report(proj)
        code, text, _ = run_text(["project", "status", str(proj.root)])
        self.assertIn(self.SAME, text.splitlines())
        code, doc = run(["project", "status", str(proj.root)])
        self.assertEqual(doc["data"]["keeper_mode"], "same_user")
        self.assertEqual(doc["warnings"].count(self.SAME), 1, doc["warnings"])
        self.assertTrue(doc["data"]["passed"])

    def test_status_before_the_report_states_the_mode_too(self):
        proj = self.create()
        code, text, _ = run_text(["project", "status", str(proj.root)])
        self.assertIn(self.SAME, text.splitlines())

    def test_project_report_adds_the_warning(self):
        proj = self.create()
        self.to_report(proj)
        code, doc = run(["project", "report", str(proj.root)])
        self.assertEqual(doc["warnings"].count(self.SAME), 1, doc["warnings"])

    def test_plan_and_create_state_the_mode(self):
        proj = self.create()
        code, text, _ = run_text(["project", "plan", str(proj.root), "intake_seal"])
        self.assertIn(self.SAME, text.splitlines())
        code, doc = run(["project", "plan", str(proj.root), "intake_seal"])
        self.assertEqual(doc["data"]["keeper_mode"], "same_user")
        self.assertEqual(doc["warnings"].count(self.SAME), 1, doc["warnings"])

    def test_the_intake_plan_does_not_call_the_data_sealed_without_the_qualifier(self):
        proj = self.create()
        code, doc = run(["project", "plan", str(proj.root), "intake_seal"])
        summary = doc["data"]["summary"]
        self.assertIn("will be sealed", summary)
        self.assertIn(texts.PLAN_SEALED_SAME_USER, summary)
        with self.strong_keeper():
            code, doc = run(["project", "plan", str(proj.root), "intake_seal"])
        self.assertNotIn(texts.PLAN_SEALED_SAME_USER, doc["data"]["summary"])
        self.assertEqual(doc["warnings"], [])
        self.assertEqual(doc["data"]["keeper_mode"], "separate_user")

    def test_verify_help_lists_the_one_value_of_accept_mode(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            cli.main(["verify", "--help"])
        self.assertIn("{same-user}", " ".join(out.getvalue().split()))

    def test_a_project_without_a_keeper_states_no_mode(self):
        from gatedtrain.ainative.project import create_project
        from gatedtrain.testing import synthetic_pack as pack

        data = self.tmp / "d.jsonl"
        from test_an_keeper_trust import TOY
        data.write_bytes(TOY)
        proj = create_project(self.tmp / "local", goal="lookup", data=data, pipeline=str(pack.SCENARIO.pipeline),
                              budget={"gpu_hours": 1, "llm_usd": 1})
        code, doc = run(["project", "status", str(proj.root)])
        self.assertNotIn("keeper_mode", doc["data"])
        self.assertEqual(doc["warnings"], [])


if __name__ == "__main__":
    unittest.main()
