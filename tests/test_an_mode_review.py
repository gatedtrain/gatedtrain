"""AN-14, AN-27, AN-29 final review: the account that owns the project files, the basis of a stated mode, the verify
wording, one definition of each version constant (tests/README.md section 13).

Same fixtures as `test_an_mode_verify`: the keeper is a subprocess of the test user (same-user), `strong_keeper` lets it
claim separate-user mode with the kernel's answer injected, and the reader's own uid is patched where a third account is
needed. Uids come from the running process.
"""

from __future__ import annotations

import os
import stat
import unittest
from pathlib import Path
from unittest import mock

from _keeper import skip_unless_linux

from test_an_keeper_cli import run, run_text
from test_an_mode_verify import ModeCase
from gatedtrain import cli
from gatedtrain.ainative import keeper_trust, signing, trust_stages, verify, verify_mode
from gatedtrain.ainative.keeper import modes, protocol, state, texts
from gatedtrain.errors import Refused

OWNER_UID = os.geteuid()
THIRD_UID = OWNER_UID + 2                 # a reader that is neither the project's account nor the listener's


class EffectiveClientModeTest(unittest.TestCase):
    def test_a_listener_with_the_uid_of_the_project_files_is_the_same_account(self):
        with self.assertRaises(Refused) as ctx:
            modes.effective_client_mode(modes.SEPARATE_USER, 1000, THIRD_UID, same_account=frozenset({1000}))
        self.assertEqual(ctx.exception.problems[0].rule, modes.RULE_ATTESTATION)
        self.assertIn("owns the project", ctx.exception.problems[0].message)

    def test_another_uid_still_confirms_and_the_own_euid_still_refuses(self):
        self.assertEqual(modes.effective_client_mode(modes.SEPARATE_USER, 7, 5, same_account=frozenset({1000}))[0],
                         modes.SEPARATE_USER)
        with self.assertRaises(Refused) as ctx:
            modes.effective_client_mode(modes.SEPARATE_USER, 5, 5, same_account=frozenset({1000}))
        self.assertIn("your own uid", ctx.exception.problems[0].message)

    def test_a_same_user_claim_needs_no_account_check(self):
        self.assertEqual(modes.effective_client_mode(modes.SAME_USER, 1000, 5, same_account=frozenset({1000}))[0],
                         modes.SAME_USER)

    def test_the_attested_mode_names_the_same_account(self):
        self.assertEqual(modes.attested_mode(1000, 5, frozenset({1000})), modes.SAME_USER)
        self.assertEqual(modes.attested_mode(5, 5, frozenset()), modes.SAME_USER)
        self.assertEqual(modes.attested_mode(7, 5, frozenset({1000})), modes.SEPARATE_USER)
        self.assertIsNone(modes.attested_mode(None, 5, frozenset()))


@skip_unless_linux
class ProjectAccountTest(ModeCase):
    """A1: a third-account reader (admin, CI) must not take a same-user keeper's separate-user claim on the kernel's
    word: the listener has the uid of the account that wrote the project."""

    def test_account_uids_are_the_owners_of_the_root_and_the_binding(self):
        proj = self.create()
        self.assertEqual(keeper_trust.project_account_uids(proj.root), frozenset({os.lstat(proj.root).st_uid}))

    def test_account_uids_do_not_follow_symlinks(self):
        proj = self.create()
        seen = []
        real = os.lstat

        def spy(path, *a, **kw):
            seen.append(Path(path))
            return real(path, *a, **kw)
        with mock.patch("os.lstat", spy), mock.patch("os.stat", side_effect=AssertionError("stat follows links")):
            keeper_trust.project_account_uids(proj.root)
        self.assertIn(proj.root / keeper_trust.KEEPER_BINDING, seen)
        self.assertIn(proj.root, seen)

    def test_verify_refuses_a_third_account_reader_when_the_listener_owns_the_project(self):     # AN-29
        proj, root, sock, fp = self.project_at_report()
        owner = os.lstat(proj.root).st_uid
        with self.strong_keeper(attested=owner), mock.patch("os.geteuid", return_value=THIRD_UID):
            for extra in ([], ["--accept-mode", "same-user"]):
                code, doc = run(["verify", root, "--keeper", sock, *extra])
                self.assertEqual((code, doc["problems"][0]["rule"]), (3, "AN-29"), doc)
                self.assertIn("owns the project", doc["problems"][0]["message"])

    def test_verify_still_confirms_a_listener_under_a_fourth_account(self):
        proj, root, sock, fp = self.project_at_report()
        self.uniform(proj, "separate_user")
        with self.strong_keeper(attested=THIRD_UID + 1, proj=proj), mock.patch("os.geteuid", return_value=THIRD_UID):
            code, doc = run(["verify", root, "--keeper", sock])
        self.assertEqual((code, doc["data"]["effective_mode"], doc["data"]["mode_basis"]),
                         (0, "separate_user", "kernel_attestation"), doc)

    def test_the_driver_refuses_the_same_keeper_for_a_third_account(self):                       # AN-29
        proj = self.create()
        owner = os.lstat(proj.root).st_uid
        with self.strong_keeper(attested=owner), mock.patch("os.geteuid", return_value=THIRD_UID):
            with self.assertRaises(Refused) as ctx:
                keeper_trust.KeeperTrust(proj).check()
        self.assertEqual(ctx.exception.problems[0].rule, modes.RULE_ATTESTATION)

    def test_the_human_text_and_the_docs_state_who_the_kernel_check_covers(self):
        self.assertIn("account that owns the project files", texts.MODE_STATED_NOTE)
        self.assertIn("third account", texts.MODE_STATED_NOTE)
        self.assertIn("fingerprint taken from the real keeper", texts.MODE_STATED_NOTE)


@skip_unless_linux
class ModeBasisTest(ModeCase):
    """A2: a signature proves only that the key holder wrote the records, so every offline result says so."""

    def test_a_fingerprint_result_says_the_mode_is_as_stated_and_names_its_basis(self):
        proj, root, sock, fp = self.project_at_report()
        self.uniform(proj, "separate_user")
        code, doc = run(["verify", root, "--fingerprint", fp])
        self.assertEqual(code, 0, doc)
        self.assertEqual(doc["data"]["mode_basis"], "signed_records")
        self.assertEqual((doc["data"]["keeper_mode"], doc["data"]["effective_mode"]), ("separate_user",) * 2)
        self.assertIn(texts.MODE_STATED_NOTE, doc["warnings"])
        code, text, err = run_text(["verify", root, "--fingerprint", fp])
        self.assertEqual(code, 0, err)
        self.assertIn("mode as stated in the signed records: separate_user", text.splitlines())
        self.assertNotIn(texts.MODE_LINE["separate_user"], text.splitlines())        # a stated mode is not observed
        self.assertIn(texts.MODE_STATED_NOTE, err)

    def test_a_forged_same_user_project_is_exit_0_by_fingerprint_but_never_unqualified(self):
        """forge.py: the key holder re-signs a same-user project as separate-user; nothing offline can tell."""
        proj, root, sock, fp = self.project_at_report()
        self.uniform(proj, "separate_user")
        code, doc = run(["verify", root, "--fingerprint", fp])
        self.assertEqual(code, 0, doc)
        self.assertTrue(any("as stated by the holder of the key" in w for w in doc["warnings"]), doc["warnings"])
        self.assertIn("a same-user keeper's key is readable by the user", texts.MODE_STATED_NOTE)

    def test_no_anchor_carries_the_note_too(self):
        proj, root, sock, fp = self.project_at_report()
        code, doc = run(["verify", root])
        self.assertEqual(code, 1, doc)
        self.assertIn(texts.MODE_STATED_NOTE, doc["warnings"])
        self.assertEqual(doc["data"]["mode_basis"], "signed_records")

    def test_the_keeper_anchor_with_a_kernel_answer_is_the_attestation_basis(self):
        proj, root, sock, fp = self.project_at_report()
        self.uniform(proj, "separate_user")
        with self.strong_keeper(proj=proj):
            code, doc = run(["verify", root, "--keeper", sock])
            self.assertEqual((code, doc["warnings"], doc["data"]["mode_basis"]), (0, [], "kernel_attestation"), doc)
            code, text, err = run_text(["verify", root, "--keeper", sock])
        self.assertIn(texts.MODE_LINE["separate_user"], text.splitlines())             # A4: the exit-0 text names it
        self.assertNotIn("mode as stated", text)

    def test_without_a_kernel_answer_the_basis_is_the_records(self):
        proj, root, sock, fp = self.project_at_report()
        self.uniform(proj, "separate_user")
        with self.strong_keeper(attested=None, proj=proj):
            code, doc = run(["verify", root, "--keeper", sock])
        self.assertEqual((code, doc["data"]["mode_basis"], doc["data"]["effective_mode"]),
                         (1, "signed_records", "same_user"), doc)

    def test_a_version_2_report_with_no_receipt_modes_is_the_report_text_basis(self):
        proj, root, sock, fp = self.project_at_report()
        self.restamp(proj, None)
        self.as_v2(proj, texts.KEEPER_GUARANTEE)
        code, doc = run(["verify", root, "--fingerprint", fp])
        self.assertEqual((code, doc["data"]["mode_basis"], doc["data"]["effective_mode"]),
                         (0, "report_text", "separate_user"), doc)
        self.assertIn(texts.MODE_STATED_NOTE, doc["warnings"])

    def test_a_project_that_states_no_mode_has_no_basis(self):
        proj = self.create()
        self.to_report(proj, until="gate")
        self.restamp(proj, None)
        code, doc = run(["verify", str(proj.root), "--fingerprint", signing.fingerprint(self.public())])
        self.assertEqual((code, doc["data"]["mode_basis"], doc["data"]["effective_mode"]), (1, None, None), doc)
        self.assertNotIn(texts.MODE_STATED_NOTE, doc["warnings"])         # nothing is stated, so nothing to qualify

    def test_the_accepted_same_user_result_names_its_basis_and_leads_with_the_acceptance(self):    # A4
        proj, root, sock, fp = self.project_at_report()
        code, text, err = run_text(["verify", root, "--fingerprint", fp, "--accept-mode", "same-user"])
        self.assertEqual(code, 0, err)
        self.assertTrue(text.startswith("accepted (weaker mode): "), text)
        self.assertNotIn("verified against", text.splitlines()[0])
        code, doc = run(["verify", root, "--fingerprint", fp, "--accept-mode", "same-user"])
        self.assertEqual(doc["data"]["mode_basis"], "signed_records")


@skip_unless_linux
class ModeUnknownHeadlineTest(ModeCase):
    """A3: a project written before 0.7.0 (receipts, no report) is mode-unknown, not a failed report guarantee."""

    def test_mode_unknown_has_its_own_headline(self):
        proj = self.create()
        self.to_report(proj, until="gate")
        self.restamp(proj, None)
        code, text, err = run_text(["verify", str(proj.root), "--fingerprint", signing.fingerprint(self.public())])
        self.assertEqual(code, 1, err)
        head = text.splitlines()[0]
        self.assertIn("the records carry no keeper mode (written before 0.7.0) and no report states one", head)
        self.assertNotIn("report's guarantee does not hold", text)
        self.assertIn(verify.MODE_UNKNOWN, err)

    def test_a_bad_report_guarantee_keeps_its_headline(self):
        proj, root, sock, fp = self.project_at_report()
        self.uniform(proj, "separate_user")
        self.resign(proj, sealed_guarantee=texts.KEEPER_ROOT_DOCKER_GUARANTEE)
        code, text, err = run_text(["verify", root, "--fingerprint", fp])
        self.assertEqual(code, 1, err)
        self.assertIn("the report's guarantee does not hold", text.splitlines()[0])


@skip_unless_linux
class SurfaceWordingTest(ModeCase):
    def test_keeper_confirm_json_carries_the_same_user_warning(self):                           # A4
        proj = self.create()
        self.upto_base(proj)
        code, doc = run(["keeper", "confirm", str(proj.root), "--decision", self.decision(), "--yes"])
        self.assertEqual(code, 0, doc)
        self.assertIn(texts.MODE_LINE["same_user"], doc["warnings"])

    def test_project_create_help_qualifies_the_seal(self):
        import contextlib
        import io

        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            cli.main(["project", "create", "--help"])
        flat = " ".join(out.getvalue().split())
        self.assertIn("seals, scores and gates (sealed from your account only in separate-user mode)", flat)

    def test_project_report_from_files_with_an_unreachable_keeper_says_it_is_not_verified(self):  # A4
        proj = self.create()
        self.to_report(proj)
        self.kp.stop()
        code, doc = run(["project", "report", str(proj.root)])
        text = " ".join(doc["warnings"])
        self.assertIn("not verified against the keeper", text, doc)


class OneDefinitionTest(unittest.TestCase):
    """A5: each version number and each kernel constant is named once; the others import it."""

    def test_report_versions_come_from_one_definition(self):
        from gatedtrain.ainative import sealed_eval

        self.assertEqual(sealed_eval.REPORT_SCHEMA_VERSION, 3)
        self.assertEqual(sealed_eval.LEGACY_REPORT_SCHEMA_VERSION, 2)
        self.assertEqual(sealed_eval.REPORT_SCHEMA_VERSIONS, (2, 3))
        self.assertIs(trust_stages.KEEPER_REPORT_VERSIONS, sealed_eval.REPORT_SCHEMA_VERSIONS)
        self.assertEqual(state.REPORT_SCHEMA_VERSION, sealed_eval.REPORT_SCHEMA_VERSION)
        self.assertEqual(keeper_trust.KEEPER_RECORD_VERSIONS,
                         (sealed_eval.KEEPER_SCHEMA_VERSION, sealed_eval.REPORT_SCHEMA_VERSION))
        self.assertEqual(trust_stages.KEEPER_SCHEMA_VERSION, sealed_eval.KEEPER_SCHEMA_VERSION)

    def test_no_module_spells_the_report_version_as_a_literal(self):
        root = Path(verify.__file__).parent
        for name in ("verify_mode.py", "trust_stages.py", "keeper_trust.py", "verify.py"):
            for n, line in enumerate((root / name).read_text().splitlines(), 1):
                code = line.split("#")[0]
                self.assertNotIn('schema_version") == 3', code, f"{name}:{n}")
                self.assertNotIn('schema_version") != 3', code, f"{name}:{n}")
                self.assertNotIn("(2, 3)", code, f"{name}:{n}")

    def test_the_peer_credential_format_and_no_uid_are_named_once(self):
        self.assertEqual(protocol.PEERCRED_FORMAT, "3i")
        self.assertEqual(protocol.NO_UID, -1)
        src = Path(protocol.__file__).read_text()
        self.assertEqual(src.count('"3i"'), 1)


@skip_unless_linux
class RequireMaintainerImagesTest(ModeCase):
    """B-F5: the flag must not pass on a project that states nothing about its images."""

    def test_a_project_without_a_report_fails_the_flag(self):
        proj = self.create()
        self.to_report(proj, until="gate")
        root, fp = str(proj.root), signing.fingerprint(self.public())
        code, doc = run(["verify", root, "--fingerprint", fp, "--accept-mode", "same-user",
                         "--require-maintainer-images"])
        self.assertEqual((code, doc["ok"]), (1, False), doc)
        self.assertIn(verify.NO_REPORT_IMAGES, doc["warnings"])
        code, doc = run(["verify", root, "--fingerprint", fp, "--accept-mode", "same-user"])
        self.assertEqual(code, 0, doc)                                  # without the flag nothing changes
        self.assertNotIn(verify.NO_REPORT_IMAGES, doc["warnings"])

    def test_the_flag_text_names_every_failure(self):
        import contextlib
        import io

        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            cli.main(["verify", "--help"])
        flat = " ".join(out.getvalue().split())
        self.assertIn("no report", flat)
        self.assertIn("undeclared", flat)


class ImageWordingTest(unittest.TestCase):
    """B-F6: the keeper recognises an ID; it never saw which image ran."""

    def row(self, locked: bool) -> dict:
        from gatedtrain import imagelock

        recipe = {"name": "r", "source": None, "commit": None, "containerfile_sha256": None, "build": None}
        iid = "sha256:" + "a" * 64
        if locked:
            return {"id": iid, "logical": "n", "origin": imagelock.MAINTAINER, "recipe": recipe,
                    "in_maintainer_lock": True, "source": "lock", "user_added": False, "declared_by": "keeper",
                    "checked_by_keeper": {"in_maintainer_lock": True}}
        return {"id": iid, "logical": None, "origin": None, "recipe": None, "in_maintainer_lock": False,
                "source": "unlisted", "user_added": True, "declared_by": "driver",
                "checked_by_keeper": {"in_maintainer_lock": False}}

    def test_the_lines_do_not_say_the_keeper_declared_what_ran(self):
        from gatedtrain.ainative import image_provenance as ip

        for locked in (True, False):
            line = ip.render_lines([self.row(locked)])[0]
            self.assertNotIn("declared by the keeper", line)
            self.assertNotIn("declared by the driver", line)
            self.assertIn("the driver declares that jobs ran in it", line)
        self.assertIn("the keeper recognises this ID", ip.render_lines([self.row(True)])[0])

    def test_verify_prints_the_limit_line_when_images_are_stated(self):
        import contextlib
        import io

        from gatedtrain.ainative import image_provenance as ip

        def verify_text(**extra):
            out = {"anchor": "fingerprint", "checked": 3, "fingerprint": "fp", "same_user": None, "root_docker": None,
                   "report_guarantees": [texts.KEEPER_GUARANTEE], "user_added_images": [], **extra}
            o, e = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e), \
                    mock.patch("gatedtrain.ainative.verify.verify_project", lambda *a, **k: out):
                cli.main(["verify", "x", "--fingerprint", "fp"])
            return o.getvalue()
        self.assertIn(texts.LIMIT_IMAGES_NOT_OBSERVED, verify_text(images_undeclared=ip.IMAGES_STATED))
        self.assertNotIn(texts.LIMIT_IMAGES_NOT_OBSERVED, verify_text(images_undeclared=ip.IMAGES_NOT_STATED))

    def test_the_limit_line_is_one_named_text(self):
        self.assertEqual(texts.LIMIT_IMAGES_NOT_OBSERVED.split(":")[0], "not observed")
        self.assertIn("job records are the driver's", texts.LIMIT_IMAGES_NOT_OBSERVED)


if __name__ == "__main__":
    unittest.main()
