"""AN-14, AN-18, AN-19: the keeper's command line (tests/README.md section 13).

`setup keeper` prints the privileged commands and a systemd unit and runs or writes nothing (AN-19); `keeper
fingerprint` prints the key fingerprint the keeper serves; `keeper confirm` shows the metric sentence and records the
confirmation only after `yes` (AN-18); `project create --keeper` binds the new project; `verify` checks a project's
signed records against an anchor (the key the keeper serves, or a fingerprint held out of band) and exits 0, 1 when
it could only check consistency, 3 on a bad signature or a mismatch (AN-14); `project status` reads the alpha ledger
through `machine.backends`.

The CLI runs in this process (`cli.main`), so the synthetic pack injected by `inject_pack` serves the stages; the keeper
is a subprocess of the test user (`tests/_keeper.py`).
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from _keeper import skip_unless_linux
from test_an_keeper_trust import MARGIN, TOY, KeeperProject

from gatedtrain import __version__, cli
from gatedtrain.ainative import scorers, signing
from gatedtrain.ainative.keeper import texts
from gatedtrain.ainative.keeper_trust import KEEPER_BINDING, KEEPER_PUB, RECEIPTS, KeeperAlphaLedger, KeeperTrust
from gatedtrain.ainative.project import Project
from gatedtrain.ainative.verify import CONSISTENCY_ONLY, OFFLINE_LIMIT
from gatedtrain.testing import synthetic_pack as pack


def run(argv: list[str], stdin: str | None = None) -> tuple[int, dict]:
    """`gatedtrain --json <argv>` in this process: (exit code, the AN-1 envelope)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), \
            mock.patch("sys.stdin", io.StringIO(stdin if stdin is not None else "")):
        code = cli.main(["--json", *argv])
    doc = json.loads(out.getvalue())
    assert doc["exit"] == code, doc
    assert err.getvalue() == "", f"stderr under --json: {err.getvalue()!r}"       # AN-1
    return code, doc


def run_text(argv: list[str], stdin: str = "") -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), mock.patch("sys.stdin", io.StringIO(stdin)):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


def _forbidden(*_a, **_k):
    raise AssertionError("setup keeper started a process")


@contextlib.contextmanager
def no_processes():
    names = [n for n in ("system", "execv", "execve", "execvp", "execvpe", "execl", "execle", "execlp", "execlpe",
                         "spawnv", "spawnve", "spawnvp", "spawnvpe", "spawnl", "spawnle", "posix_spawn",
                         "posix_spawnp", "fork", "popen") if hasattr(os, n)]
    with contextlib.ExitStack() as stack:
        for n in names:
            stack.enter_context(mock.patch.object(os, n, _forbidden))
        for n in ("run", "Popen", "call", "check_call", "check_output"):
            stack.enter_context(mock.patch.object(subprocess, n, _forbidden))
        yield


class SetupKeeperTest(unittest.TestCase):                          # AN-19
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def argv(self) -> list[str]:
        return ["setup", "keeper", "--state", str(self.tmp / "state"), "--socket",
                str(self.tmp / "run" / "keeper.sock"), "--venv", str(self.tmp / "venv"), "--owner-uid", "4321",
                "--packages", "gatedtrain-pack-x==1.2.3"]

    def test_prints_the_commands_and_runs_nothing(self):
        with no_processes():
            code, doc = run(self.argv())
            tcode, text, err = run_text(self.argv())
        self.assertEqual((code, doc["ok"], doc["command"]), (0, True, "setup keeper"), doc)
        self.assertEqual(set(doc["data"]), {"commands", "unit"})
        self.assertEqual(list(self.tmp.iterdir()), [], "setup created a file")
        commands, unit = doc["data"]["commands"], doc["data"]["unit"]
        joined = "\n".join(commands)
        state, sock, venv = self.tmp / "state", self.tmp / "run" / "keeper.sock", self.tmp / "venv"

        def at(text: str) -> int:
            for i, c in enumerate(commands):
                if text in c:
                    return i
            self.fail(f"{text!r} is not in the commands:\n{joined}")

        order = [at("useradd"), at(f"-m 0700 {state}"), at(str(sock.parent)), at(f"-m venv {venv}"),
                 at(f"gatedtrain=={__version__}"), at("gatedtrain-keeper.service"), at("systemctl daemon-reload"),
                 at("systemctl enable --now"), at(f"gatedtrain keeper fingerprint --socket {sock}"),
                 at("fingerprint")]
        self.assertEqual(order, sorted(order), joined)
        self.assertIn("install -d", joined)
        self.assertIn("gatedtrain-pack-x==1.2.3", commands[at(f"gatedtrain=={__version__}")])
        self.assertIn("User=gatedtrain-keeper\n", unit)
        self.assertIn(f"ExecStart={venv}/bin/gatedtrain keeper serve --state {state} --socket {sock} "
                      "--owner-uid 4321\n", unit)
        self.assertIn(unit, joined)                                    # the unit is written by a printed command
        self.assertTrue(commands[-1].startswith("#") and "keep" in commands[-1], commands[-1])
        self.assertEqual(tcode, 0)
        self.assertEqual(err, "")
        for c in commands:
            self.assertIn(c, text)

    def test_defaults_use_a_runtime_directory(self):                  # preflight finding 27
        with no_processes():
            code, doc = run(["setup", "keeper"])
        self.assertEqual(code, 0, doc)
        unit, joined = doc["data"]["unit"], "\n".join(doc["data"]["commands"])
        self.assertIn("RuntimeDirectory=gatedtrain-keeper\n", unit)
        self.assertIn(f"ExecStart=/opt/gatedtrain-keeper/venv/bin/gatedtrain keeper serve --state "
                      f"/var/lib/gatedtrain-keeper --socket /run/gatedtrain-keeper/keeper.sock --owner-uid "
                      f"{os.getuid()}\n", unit)
        self.assertIn("install -d -o gatedtrain-keeper -g gatedtrain-keeper -m 0700 /var/lib/gatedtrain-keeper",
                      joined)
        self.assertNotIn("install -d -o gatedtrain-keeper -g gatedtrain-keeper -m 0755 /run/gatedtrain-keeper",
                         joined)
        self.assertNotIn("sudo -u gatedtrain-keeper", joined)        # the venv is root-owned (design rev-23 AN-13)

    def test_an_unpinned_package_or_a_bad_user_is_refused(self):
        with no_processes():
            code, doc = run(["setup", "keeper", "--packages", "gatedtrain-pack-x"])
            self.assertEqual((code, doc["problems"][0]["rule"]), (3, "AN-19"), doc)
            code, doc = run(["setup", "keeper", "--user", "x; rm -rf /"])
            self.assertEqual((code, doc["problems"][0]["rule"]), (3, "AN-19"), doc)
            code, doc = run(["setup", "keeper", "--state", "relative/state"])
            self.assertEqual((code, doc["problems"][0]["rule"]), (3, "AN-19"), doc)


@skip_unless_linux
class KeeperCliTest(KeeperProject, unittest.TestCase):
    def setUp(self):
        self.start()

    def create(self, name: str = "proj") -> Project:
        data = self.tmp / f"{name}.jsonl"
        data.write_bytes(TOY)
        code, doc = run(["project", "create", str(self.tmp / name), "--goal", "lookup", "--data", str(data),
                         "--pipeline", str(pack.SCENARIO.pipeline), "--keeper", str(self.kp.socket)])
        self.assertEqual(code, 0, doc)
        return Project.load(self.tmp / name)

    def decision(self, margin: float = MARGIN) -> str:
        path = self.tmp / "decision.json"
        path.write_text(json.dumps({"metric_confirmed": True, "scorer": "exact_match", "margin": margin}))
        return str(path)

    def upto_base(self, proj) -> None:
        for ref in ("intake_seal", "profile_data", "choose_task_and_base"):
            self.go(proj, ref)

    def to_report(self, proj, until: str = "report") -> None:
        self.upto_base(proj)
        code, doc = run(["keeper", "confirm", str(proj.root), "--decision", self.decision(), "--yes"])
        self.assertEqual(code, 0, doc)
        for ref in ("preregister", "build_train_set", "write_recipe", "smoke", "train", "evaluate_sealed", "gate",
                    "report"):
            self.go(proj, ref)
            if ref == until:
                return

    def test_fingerprint(self):
        code, doc = run(["keeper", "fingerprint", "--socket", str(self.kp.socket)])
        self.assertEqual((code, doc["command"]), (0, "keeper fingerprint"), doc)
        self.assertEqual(doc["data"], {"fingerprint": signing.fingerprint(self.public()),
                                       "sealed_guarantee": texts.KEEPER_SAME_USER_GUARANTEE})
        code, doc = run(["keeper", "fingerprint", "--socket", str(self.tmp / "none.sock")])
        self.assertEqual((code, doc["problems"][0]["rule"]), (3, "AN-12"), doc)

    def test_project_create_binds_the_project(self):
        data = self.tmp / "d.jsonl"
        data.write_bytes(TOY)
        code, doc = run(["project", "create", str(self.tmp / "p"), "--goal", "lookup", "--data", str(data),
                         "--pipeline", str(pack.SCENARIO.pipeline), "--keeper", str(self.kp.socket)])
        self.assertEqual(code, 0, doc)
        proj = Project.load(self.tmp / "p")
        binding = json.loads((proj.root / KEEPER_BINDING).read_text())
        self.assertEqual(doc["data"]["keeper"], {"fingerprint": signing.fingerprint(self.public()),
                                                 "project_id": binding["project_id"]})
        self.assertTrue((proj.root / KEEPER_PUB).is_file())
        code, doc = run(["project", "create", str(self.tmp / "q"), "--goal", "lookup", "--data", str(data),
                         "--pipeline", str(pack.SCENARIO.pipeline), "--keeper", str(self.tmp / "none.sock")])
        self.assertEqual((code, doc["problems"][0]["rule"]), (3, "AN-12"), doc)
        self.assertFalse((self.tmp / "q").exists(), "a project was created for an unreachable keeper")

    def test_confirm_needs_yes(self):                                  # AN-18
        proj = self.create()
        self.upto_base(proj)
        spec = scorers.resolve("exact_match")
        argv = ["keeper", "confirm", str(proj.root), "--decision", self.decision()]
        for answer in ("no\n", "", "y\n"):
            code, text, err = run_text(argv, stdin=answer)
            self.assertEqual((code, text), (3, ""), (answer, err))
            self.assertIn("error[AN-18]", err)
        code, doc = run(argv, stdin="yes\n")                          # --json never prompts (AN-1): refused
        self.assertEqual((code, doc["problems"][0]["rule"]), (3, "AN-18"), doc)
        self.assertIn("--yes", doc["problems"][0]["fix"])
        self.assertIn(spec.what, doc["problems"][0]["message"])        # the sentence, no sealed content
        self.assertFalse(any(e["kind"] == "metric_confirmation" for e in KeeperTrust(proj).receipts()))
        code, doc = run([*argv, "--yes"])
        self.assertEqual((code, doc["command"]), (0, "keeper confirm"), doc)
        self.assertEqual(set(doc["data"]), {"sentence", "envelope_sha256"})
        self.assertIn(spec.what, doc["data"]["sentence"])
        self.assertIn(f"more than {MARGIN * 100:g} points", doc["data"]["sentence"])
        env = [e for e in KeeperTrust(proj).receipts() if e["kind"] == "metric_confirmation"]
        self.assertEqual(signing.envelope_sha256(env[-1]), doc["data"]["envelope_sha256"])
        self.assertEqual((env[-1]["record"]["scorer"], env[-1]["record"]["margin"], env[-1]["record"]["sentence"]),
                         ("exact_match", MARGIN, doc["data"]["sentence"]))
        code, text, err = run_text(["keeper", "confirm", str(proj.root), "--decision", self.decision()],
                                   stdin="yes\n")
        self.assertEqual(code, 0, err)
        self.assertIn(doc["data"]["sentence"], err)                    # the prompt goes to stderr, stdout is the result
        self.assertIn("confirmed", text)

    def test_verify(self):                                             # AN-14, preflight finding 10
        proj = self.create()
        self.to_report(proj)
        fp = signing.fingerprint(self.public())
        root = str(proj.root)

        code, doc = run(["verify", root, "--keeper", str(self.kp.socket)])
        self.assertEqual((code, doc["command"], doc["data"]["anchor"], doc["data"]["fingerprint"]),
                         (0, "verify", "keeper", fp), doc)
        n = len((proj.root / RECEIPTS).read_bytes().splitlines())
        self.assertEqual(doc["data"]["checked"], n + 3)                # every receipt, registration, gate, report
        code, doc = run(["verify", root, "--fingerprint", fp])
        self.assertEqual((code, doc["data"]["anchor"]), (0, "fingerprint"), doc)
        code, doc = run(["verify", root, "--keeper", str(self.kp.socket), "--fingerprint", fp])
        self.assertEqual((code, doc["data"]["anchor"]), (0, "keeper"), doc)
        code, doc = run(["verify", root])
        self.assertEqual((code, doc["ok"], doc["data"]["anchor"]), (1, False, "none"), doc)
        self.assertEqual(doc["warnings"], [CONSISTENCY_ONLY, OFFLINE_LIMIT])
        code, doc = run(["verify", root, "--fingerprint", fp])
        self.assertEqual(doc["warnings"], [OFFLINE_LIMIT])
        code, text, err = run_text(["verify", root, "--fingerprint", fp])
        self.assertIn(OFFLINE_LIMIT, err)

        other = signing.fingerprint(Ed25519PrivateKey.generate())
        code, doc = run(["verify", root, "--fingerprint", other])
        self.assertEqual((code, doc["problems"][0]["rule"]), (3, "AN-14"), doc)
        code, doc = run(["verify", root, "--keeper", str(self.kp.socket), "--fingerprint", other])
        self.assertEqual((code, doc["problems"][0]["rule"]), (3, "AN-14"), doc)

        def tampered(rel: str, edit) -> None:
            path = proj.root / rel
            saved = path.read_bytes()
            changed = edit(saved)
            self.assertNotEqual(changed, saved, rel)
            path.write_bytes(changed)
            try:
                for argv in (["--keeper", str(self.kp.socket)], ["--fingerprint", fp], []):
                    code, doc = run(["verify", root, *argv])
                    self.assertEqual(code, 3, (rel, argv, doc))
                    self.assertEqual(doc["problems"][0]["rule"], "AN-14", (rel, argv, doc))
            finally:
                path.write_bytes(saved)

        new_pub = signing.public_pem(Ed25519PrivateKey.generate())
        tampered(KEEPER_PUB, lambda _b: new_pub)                      # a replaced key copy
        binding = json.loads((proj.root / KEEPER_BINDING).read_text())
        tampered(KEEPER_BINDING, lambda _b: json.dumps({**binding, "fingerprint": other}).encode())
        tampered(RECEIPTS, lambda b: b.replace(b'"margin":0.05', b'"margin":0.01', 1))
        tampered("stages/gate/gate_result.json", lambda b: b.replace(b'"status":"PASS"', b'"status":"FAIL"'))
        tampered("stages/preregister/registration.json", lambda b: b.replace(b'"margin":0.05', b'"margin":0.01'))
        tampered("stages/report/report.json", lambda b: b.replace(b'"status":"PASS"', b'"status":"FAIL"'))
        tampered("stages/report/report.md", lambda b: b + b"\nextra\n")
        for rel in ("stages/preregister/registration.json", "stages/gate/gate_result.json",
                    "stages/report/report.json"):                       # a wrong schema is refused, never skipped
            tampered(rel, lambda b: b.replace(b'"schema":"gatedtrain.ainative.', b'"schema":"gatedtrain.other.', 1))

        report_json = proj.root / "stages" / "report" / "report.json"     # report.md without its report.json
        saved = report_json.read_bytes()
        report_json.unlink()
        try:
            for argv in (["--keeper", str(self.kp.socket)], ["--fingerprint", fp], []):
                code, doc = run(["verify", root, *argv])
                self.assertEqual((code, doc["problems"][0]["rule"]), (3, "AN-14"), (argv, doc))
        finally:
            report_json.write_bytes(saved)

        code, doc = run(["verify", root, "--keeper", str(self.kp.socket)])
        self.assertEqual(code, 0, doc)                                 # restored

    def test_verify_refuses_a_record_the_keeper_does_not_hold(self):  # AN-14: has_record
        proj = self.create()
        self.to_report(proj)
        env =[e for e in KeeperTrust(proj).receipts() if e["kind"] == "registration"][-1]
        forged = signing.sign(signing.load_private_key(self.kp.state / "keeper.key"), "registration",
                              {**env["record"], "utc": "2000-01-01T00:00:00Z"})
        with open(proj.root / RECEIPTS, "ab") as f:
            f.write(json.dumps(forged).encode() + b"\n")
        code, doc = run(["verify", str(proj.root), "--fingerprint", signing.fingerprint(self.public())])
        self.assertEqual(code, 0, doc)                                # signed by the key: offline it checks
        code, doc = run(["verify", str(proj.root), "--keeper", str(self.kp.socket)])
        self.assertEqual((code, doc["problems"][0]["rule"]), (3, "AN-14"), doc)

    def test_verify_refuses_another_projects_report(self):           # review finding 1
        a = self.create("a")
        self.to_report(a)
        b = self.create("b")                                            # same keeper, no attempts
        fp = signing.fingerprint(self.public())
        env = [e for e in KeeperTrust(a).receipts() if e["kind"] == "report"][-1]
        (b.root / RECEIPTS).parent.mkdir(parents=True, exist_ok=True)
        with open(b.root / RECEIPTS, "ab") as f:
            f.write(json.dumps(env).encode() + b"\n")
        (b.root / "stages" / "report").mkdir(parents=True)
        for name in ("report.json", "report.md"):
            (b.root / "stages" / "report" / name).write_bytes((a.root / "stages" / "report" / name).read_bytes())
        for argv in (["--keeper", str(self.kp.socket)], ["--fingerprint", fp], []):
            code, doc = run(["verify", str(b.root), *argv])
            self.assertEqual((code, doc["problems"][0]["rule"]), (3, "AN-14"), (argv, doc))

    def test_verify_with_the_keeper_finds_a_deleted_record(self):    # review finding 3
        proj = self.create()
        self.to_report(proj, until="gate")
        fp = signing.fingerprint(self.public())
        code, doc = run(["verify", str(proj.root), "--keeper", str(self.kp.socket)])
        self.assertEqual(code, 0, doc)
        lines = (proj.root / RECEIPTS).read_bytes().splitlines(keepends=True)
        kept = [x for x in lines if json.loads(x)["kind"] != "gate_summary"]
        self.assertEqual(len(kept), len(lines) - 1)
        (proj.root / RECEIPTS).write_bytes(b"".join(kept))
        (proj.root / "stages" / "gate" / "gate_result.json").unlink()
        code, doc = run(["verify", str(proj.root), "--keeper", str(self.kp.socket)])
        self.assertEqual((code, doc["problems"][0]["rule"]), (3, "AN-14"), doc)
        code, doc = run(["verify", str(proj.root), "--fingerprint", fp])
        self.assertEqual((code, doc["warnings"]), (0, [OFFLINE_LIMIT]), doc)     # offline: stated, not detected

    def test_status_reads_the_keeper_ledger(self):                   # Task 10 carry: project_id and j
        proj, other = self.create(), self.create("other")             # same data: one sealed set, one ledger
        self.to_report(other, until="gate")
        self.to_report(proj)
        entries = KeeperAlphaLedger(KeeperTrust(proj)).entries()
        pids = [json.loads((p.root / KEEPER_BINDING).read_text())["project_id"] for p in (other, proj)]
        self.assertEqual([(e["project_id"], e["j"]) for e in entries], [(pids[0], 1), (pids[1], 2)])
        code, doc = run(["project", "status", str(proj.root)])
        self.assertEqual(code, 0, doc)
        self.assertEqual(doc["data"]["alpha_ledger"], entries[1:])     # this project's attempts only
        self.assertEqual(doc["data"]["gate"], "PASS")
        code, doc = run(["project", "report", str(proj.root)])
        self.assertEqual((code, doc["data"]["report"]["schema_version"]), (0, 2), doc)

        self.kp.stop()
        code, doc = run(["project", "status", str(proj.root)])
        self.assertEqual(code, 0, doc)
        self.assertIsNone(doc["data"]["alpha_ledger"])
        self.assertEqual(doc["data"]["stages"]["report"]["status"], "done")
        self.assertEqual(len(doc["warnings"]), 1)
        self.assertIn("AN-12", doc["warnings"][0])


if __name__ == "__main__":
    unittest.main()
