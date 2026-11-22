"""AN-20, AN-19: `gatedtrain setup host` and the keeper's inference options in `setup keeper` (tests/README.md
section 14).

`setup host` prints one root bash script (never runs it) that sets up rootless Docker for the owner and the keeper
user, removes the owner from the rootful `docker` group before any user manager or rootless daemon of the owner is
started, and with `--inference-image` wires the keeper's inference options into a drop-in of
its unit. Every check (root, the owner's uid, an NVIDIA CDI spec, the eval interpreter tree, the parents of the
created directories) comes before the first change. `setup keeper --inference-image ID --eval-tree DIR --docker-host
URL` prints them into `ExecStart=`. Nothing here starts a process or creates a file; the only shell use is the
test's own `bash -n` on a temporary copy of the script.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path, PurePosixPath as PurePosix

from test_an_keeper_cli import no_processes, run, run_text

from gatedtrain.ainative import hostsetup
from gatedtrain.ainative.keeper import setup

HEX = "0123456789abcdef" * 4
REF = f"ghcr.io/example/cuda-base:12.8@sha256:{HEX}"
PYROOT = hostsetup.DEFAULT_EVAL_PYTHON_ROOT
EVAL_PYTHON = f"{PYROOT}/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12"
REQS = (b"# uv export --format requirements-txt --no-emit-project\n"
        b"vllm==0.11.0 \\\n    --hash=sha256:" + b"1" * 64 + b" \\\n    --hash=sha256:" + b"2" * 64 + b"\n"
        b"numpy==2.2.6 --hash=sha256:" + b"3" * 64 + b"\n")


def commands_only(script: str) -> str:
    """The script without its comment lines (the shebang stays)."""
    return "\n".join(line for line in script.splitlines()
                     if not line.lstrip().startswith("#") or line.startswith("#!"))


class SetupHostTest(unittest.TestCase):                             # AN-20
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.work = self.tmp / "cwd"
        self.work.mkdir()
        cwd = os.getcwd()
        os.chdir(self.work)
        self.addCleanup(os.chdir, cwd)

    def host(self, *extra: str) -> tuple[int, dict]:
        with no_processes():
            code, doc = run(["setup", "host", "--owner", "alice", "--owner-uid", "1234", *extra])
        self.assertEqual(list(self.work.iterdir()), [], "setup host created a file")
        return code, doc

    def script(self, *extra: str) -> str:
        code, doc = self.host(*extra)
        self.assertEqual((code, doc["ok"], doc["command"]), (0, True, "setup host"), doc)
        self.assertEqual(set(doc["data"]), {"script", "command"})
        self.assertEqual(doc["data"]["command"], "sudo bash gatedtrain-host-setup.sh")
        return doc["data"]["script"]

    def assert_bash_parses(self, script: str) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is not installed")
        path = self.tmp / "copy.sh"
        path.write_text(script, encoding="utf-8")
        proc = subprocess.run([bash, "-n", str(path)], capture_output=True, text=True, timeout=30)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def assert_order(self, text: str, needles: list[str]) -> list[int]:
        at = []
        for needle in needles:
            i = text.find(needle)
            self.assertGreaterEqual(i, 0, f"{needle!r} is not in the script:\n{text}")
            at.append(i)
        self.assertEqual(at, sorted(at), list(zip(needles, at)))
        return at

    def test_default_script(self):
        script = self.script()
        cmds = commands_only(script)
        self.assertTrue(script.startswith("#!/bin/bash\n"), script[:80])
        self.assertIn("set -euo pipefail\ncd /\numask 022", cmds)      # no program runs in the caller's cwd
        self.assertIn("gatedtrain-host-setup.sh", script.split("set -euo pipefail")[0])     # header: save, run, delete
        # the checks, then the changes in order (the group removal before any user manager of the owner starts)
        self.assert_order(cmds, ['"$(id -u)"', 'id -u "$OWNER"', "nvidia.com/gpu", "apt-get install -y uidmap",
                                 "docker-ce-rootless-extras=", "usr.bin.rootlesskit", "--add-subuids",
                                 "gpasswd -d", "loginctl enable-linger", '"cdi": true',
                                 "dockerd-rootless-setuptool.sh install", "systemctl --user restart docker",
                                 "context update gatedtrain"])
        self.assertNotIn("sudo", cmds)
        self.assertNotIn("read -r -p", cmds)                         # no pause without --confirm-group-removal
        for path in (hostsetup.DEFAULT_HANDOFF, hostsetup.DEFAULT_OWNER_DATA_ROOT, hostsetup.DEFAULT_KEEPER_HOME,
                     hostsetup.DEFAULT_KEEPER_DATA_ROOT, *hostsetup.CDI_SPEC_DIRS):
            self.assertIn(path, cmds)
        self.assertIn(f"-m 0700 {hostsetup.DEFAULT_KEEPER_HOME}", cmds)
        self.assertIn(f"-m 0755 {hostsetup.DEFAULT_HANDOFF}", cmds)
        self.assertIn(str(hostsetup.SUBID_COUNT), cmds)
        self.assertIn("runuser -u", cmds)
        self.assertIn("XDG_RUNTIME_DIR=/run/user/", cmds)
        self.assertIn("DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/", cmds)
        self.assertIn("unix:///run/user/1234/docker.sock", cmds)
        # no inference, no eval venv; an earlier inference drop-in is removed
        for absent in ("--inference-image", "--eval-tree", "-m venv", "--require-hashes", "cat > /etc/systemd"):
            self.assertNotIn(absent, cmds)
        self.assert_order(cmds, [f"rm -f {hostsetup.DROPIN_PATH}", "systemctl daemon-reload",
                                 "systemctl try-restart gatedtrain-keeper.service"])
        self.assertNotIn("systemctl restart gatedtrain-keeper.service", cmds)
        # closing comments, as the owner
        tail = script[script.rindex("systemctl --user restart docker"):]
        for text in ("systemctl restart user@1234.service", "rootless = true",
                     f'output_root = "{hostsetup.DEFAULT_HANDOFF}"', "host.toml", "docker --context gatedtrain",
                     "gatedtrain keeper fingerprint"):
            self.assertIn(text, tail)
        self.assertIn("without --eval-requirements passes no --eval-tree", " ".join(tail.split()))
        self.assert_bash_parses(script)

    def test_user_manager_already_running_and_subids(self):
        script = self.script()
        cmds = commands_only(script)
        # apt's lists are refreshed before uidmap is installed
        self.assert_order(cmds, ["cd /", "apt-get update", "apt-get install -y uidmap"])
        # an owner's user manager that already runs keeps the docker group: a warning when it is seen, before the
        # start, and again at the end; the group removal still comes first
        self.assert_order(cmds, ["gpasswd -d", 'systemctl is-active --quiet "user@$OWNER_UID.service"',
                                 "loginctl enable-linger", 'systemctl start "user@$OWNER_UID.service"',
                                 "dockerd-rootless-setuptool.sh install", '[ -z "$OWNER_MANAGER_WAS_ACTIVE" ]',
                                 "done; the next steps"])
        warn = [l for l in cmds.splitlines() if "OWNER_MANAGER_WAS_ACTIVE" in l and "echo" in l]
        self.assertEqual(len(warn), 1, cmds)
        self.assertIn("systemctl restart user@$OWNER_UID.service", cmds)
        tail = script[script.rindex("# Next steps:"):]
        self.assertIn("already running", " ".join(tail.split()))
        self.assertIn("keeper fingerprint", " ".join(tail.split()))
        # overlapping subordinate id ranges refuse before any change, and again once the keeper user exists
        self.assert_order(cmds, ["check_subids() {", "check_subids before-changes", "apt-get update",
                                 'KUID=$(id -u "$KEEPER")', "check_subids after-keeper-created", "--add-subuids",
                                 "--add-subgids", "check_subids after-allocation", "gpasswd -d",
                                 "dockerd-rootless-setuptool.sh install"])
        for f in ("/etc/subuid", "/etc/subgid"):
            self.assertIn(f, cmds[:cmds.index("apt-get update")])
        self.assert_bash_parses(script)

    def test_subid_overlap_check_runs(self):
        """The script's `covers` function, run by bash on temporary files (the only part of the script run here)."""
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is not installed")
        cmds = commands_only(self.script())
        start = cmds.index("covers() {")
        func = cmds[start:cmds.index("\n}\n", start) + 3]
        subuid = self.tmp / "subuid"
        subuid.write_text("alice:100000:65536\nkeeper:165536:65536\n")
        for user, uid, ident, covered in (("alice", 1234, 100000, True), ("alice", 1234, 165536, False),
                                          ("1234", 1234, 100001, False), ("keeper", 999, 231071, True),
                                          ("bob", 1, 100000, False)):
            with self.subTest(user=user, ident=ident):
                proc = subprocess.run([bash, "-c", func + f'\ncovers {subuid} {user} {uid} {ident}'],
                                      capture_output=True, text=True, timeout=30)
                self.assertEqual(proc.returncode == 0, covered, proc.stderr)
        proc = subprocess.run([bash, "-c", func + f"\ncovers {self.tmp / 'absent'} alice 1234 100000"],
                              capture_output=True, text=True, timeout=30)
        self.assertNotEqual(proc.returncode, 0)

    def test_network_driver_and_extras_hold(self):
        cmds = commands_only(self.script())
        # a rootless network driver is checked before any change
        self.assert_order(cmds, ["cd /", "command -v slirp4netns", "command -v pasta", "apt-get install -y uidmap"])
        # the extras: skipped at the right version, unheld and held again around the install
        self.assert_order(cmds, ["apt-get install -y uidmap",
                                 "dpkg-query -W -f='${Version}' docker-ce-rootless-extras",
                                 '"$EXTRAS_VERSION" != "$DOCKER_CE_VERSION"',
                                 "apt-mark showhold docker-ce-rootless-extras",
                                 "apt-mark unhold docker-ce-rootless-extras",
                                 '--allow-downgrades --no-install-recommends '
                                 '"docker-ce-rootless-extras=$DOCKER_CE_VERSION"',
                                 "apt-mark hold docker-ce-rootless-extras", "usr.bin.rootlesskit"])
        # a replaced daemon.json is reported with its backup
        self.assertIn("daemon.json.before-gatedtrain", cmds[cmds.index("echo ", cmds.index("as_owner cp")):])
        self.assertRegex(cmds, r'echo "[^"\n]*replaced[^"\n]*daemon\.json\.before-gatedtrain')

    def test_plain_mode_prints_the_script_only(self):
        with no_processes():
            _, doc = run(["setup", "host", "--owner", "alice", "--owner-uid", "1234"])
            code, out, err = run_text(["setup", "host", "--owner", "alice", "--owner-uid", "1234"])
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(out, doc["data"]["script"])
        self.assertEqual(list(self.work.iterdir()), [])

    def test_confirm_pause_comes_before_the_group_removal(self):
        cmds = commands_only(self.script("--confirm-group-removal"))
        self.assert_order(cmds, ["usr.bin.rootlesskit", "read -r -p", "gpasswd -d", "loginctl enable-linger",
                                 "runuser -u"])

    def test_inference_and_eval_venv(self):
        reqs = self.tmp / "requirements.txt"
        reqs.write_bytes(REQS)
        script = self.script("--inference-image", REF, "--eval-requirements", str(reqs), "--eval-python",
                             EVAL_PYTHON)
        cmds = commands_only(script)
        venv = hostsetup.DEFAULT_EVAL_VENV
        # the interpreter tree is checked before any change; the venv is built from it, root-owned
        self.assert_order(cmds, [f"PYROOT={PYROOT}", "-perm /022", "apt-get install -y uidmap",
                                 f"-m venv --clear {venv}", "--require-hashes",
                                 f"chmod -R go-w {venv}", "docker pull", "image inspect -f '{{.Id}}'",
                                 "^sha256:[0-9a-f]{64}$", "inference.conf", "systemctl daemon-reload",
                                 "systemctl try-restart gatedtrain-keeper.service"])
        # root runs the interpreter isolated from the cwd and the environment
        self.assertIn(f'"$EVAL_PYTHON" -I -m venv --clear {venv}', cmds)
        pip = next(line for line in cmds.splitlines() if "--require-hashes" in line)
        for text in ("PIP_CONFIG_FILE=/dev/null", "--isolated", "--only-binary :all:", "--require-hashes",
                     "--no-deps"):
            self.assertIn(text, pip)
        self.assertLess(cmds.index("cd /"), cmds.index("apt-get"))
        self.assertIn(f"EVAL_PYTHON={EVAL_PYTHON}", cmds)
        self.assertIn(REQS.decode(), script)                           # embedded verbatim
        self.assertIn("uv python install", script)                       # how root installs the interpreter
        self.assertNotIn("uv python install", cmds)                      # ... in a comment only
        dropin = script[script.index("inference.conf"):]
        for text in ("After=user@${KUID}.service", "Wants=user@${KUID}.service", "[Service]", "ExecStart=\n"):
            self.assertIn(text, dropin)
        serve = setup.serve_argv(PurePosix(setup.DEFAULT_VENV), PurePosix(setup.DEFAULT_STATE),
                                 PurePosix(setup.DEFAULT_SOCKET), 1234, inference_image="${IMG}",
                                 eval_trees=[PYROOT, venv], docker_host="unix:///run/user/${KUID}/docker.sock")
        self.assertIn("ExecStart=" + " ".join(serve) + "\n", dropin)       # one serve builder
        self.assertIn(f"--eval-tree {PYROOT} --eval-tree {venv}", dropin)
        self.assertNotIn("sudo", cmds)
        self.assert_bash_parses(script)

    def test_inference_without_eval_venv(self):
        cmds = commands_only(self.script("--inference-image", REF))
        self.assertIn("inference.conf", cmds)
        self.assertIn("--inference-image ${IMG} --docker-host unix:///run/user/${KUID}/docker.sock\n", cmds)
        self.assertNotIn("--eval-tree", cmds)
        self.assertNotIn("PYROOT", cmds)

    def test_refusals(self):
        unhashed = self.tmp / "unhashed.txt"
        unhashed.write_bytes(b"numpy==2.2.6 --hash=sha256:" + b"3" * 64 + b"\nvllm==0.11.0\n")
        hashed = self.tmp / "hashed.txt"
        hashed.write_bytes(REQS)
        cases = [
            (["--inference-image", "python:3.12-slim"], "--inference-image"),
            (["--inference-image", f"sha256:{HEX}"], "--inference-image"),         # a bare ID cannot be pulled
            (["--eval-requirements", str(unhashed), "--eval-python", EVAL_PYTHON], "--eval-requirements"),
            (["--eval-requirements", str(self.tmp / "missing.txt"), "--eval-python", EVAL_PYTHON],
             "--eval-requirements"),
            (["--eval-requirements", str(hashed)], "--eval-python"),
            (["--eval-requirements", str(hashed), "--eval-python", "/usr/bin/python3"], "--eval-python"),
            (["--eval-requirements", str(hashed), "--eval-python", f"{PYROOT}/../x/bin/python"], "--eval-python"),
            (["--eval-python", EVAL_PYTHON], "--eval-python"),
            (["--keeper-user", "alice"], "--keeper-user"),
            (["--keeper-user", "x; rm -rf /"], "--keeper-user"),
        ]
        for flag in ("--keeper-home", "--keeper-data-root", "--owner-data-root", "--handoff", "--eval-venv",
                     "--state", "--socket", "--venv"):
            cases += [([flag, "/srv/with space"], flag), ([flag, "relative/dir"], flag),
                      ([flag, "/srv/a`id`"], flag), ([flag, "/srv/a$HOME"], flag)]
        for extra, field in cases:
            with self.subTest(extra=extra):
                code, doc = self.host(*extra)
                self.assertEqual((code, doc["ok"]), (3, False), doc)
                self.assertEqual((doc["problems"][0]["rule"], doc["problems"][0]["field"]), ("AN-20", field), doc)
        with no_processes():
            code, doc = run(["setup", "host", "--owner", "bad name", "--owner-uid", "1234"])
        self.assertEqual((code, doc["problems"][0]["rule"], doc["problems"][0]["field"]), (3, "AN-20", "--owner"))

    def test_heredoc_terminator_in_requirements_is_refused(self):
        bad = REQS + hostsetup.REQUIREMENTS_EOF.encode() + b"\n"
        with self.assertRaises(hostsetup.Refused) as cm:
            hostsetup.host_script(owner="alice", owner_uid=1234, eval_requirements=bad, eval_python=EVAL_PYTHON)
        self.assertEqual(cm.exception.problems[0].rule, "AN-20")

    def test_function_defaults(self):
        out = hostsetup.host_script(owner="alice", owner_uid=1234)
        self.assertEqual(set(out), {"script", "command"})
        self.assertEqual(out["command"], f"sudo bash {hostsetup.SCRIPT_NAME}")
        self.assertEqual(hostsetup.RULE, "AN-20")
        self.assertEqual(hostsetup.SUBID_COUNT, 65536)
        self.assertEqual((hostsetup.DEFAULT_HANDOFF, hostsetup.DEFAULT_OWNER_DATA_ROOT, hostsetup.DEFAULT_KEEPER_HOME,
                          hostsetup.DEFAULT_KEEPER_DATA_ROOT, hostsetup.DEFAULT_EVAL_VENV, hostsetup.CDI_SPEC_DIRS,
                          hostsetup.DEFAULT_EVAL_PYTHON_ROOT),
                         ("/srv/gatedtrain/out", "/srv/gatedtrain/docker", "/var/lib/gatedtrain-keeper-home",
                          "/var/lib/gatedtrain-keeper-docker", "/opt/gatedtrain-keeper/eval-venv",
                          ("/etc/cdi", "/var/run/cdi"), "/opt/gatedtrain-keeper/python"))


class SetupKeeperInferenceTest(unittest.TestCase):                 # AN-19 with the AN-25 serve options
    def test_inference_options_go_into_exec_start(self):
        with no_processes():
            code, doc = run(["setup", "keeper", "--owner-uid", "4321", "--packages", "p==1",
                             "--inference-image", f"sha256:{HEX}", "--eval-tree", "/x", "--eval-tree", "/y/z",
                             "--docker-host", "unix:///run/user/999/docker.sock"])
        self.assertEqual(code, 0, doc)
        line = next(x for x in doc["data"]["unit"].splitlines() if x.startswith("ExecStart="))
        self.assertTrue(line.endswith(f"--owner-uid 4321 --inference-image sha256:{HEX} --eval-tree /x "
                                      "--eval-tree /y/z --docker-host unix:///run/user/999/docker.sock"), line)

    def test_bad_inference_options_are_refused(self):
        cases = [(["--inference-image", "python:3.12-slim"], "--inference-image"),
                 (["--eval-tree", "/x"], "--eval-tree"),
                 (["--docker-host", "unix:///run/user/1/docker.sock"], "--docker-host"),
                 (["--inference-image", f"sha256:{HEX}", "--eval-tree", "/with space"], "--eval-tree"),
                 (["--inference-image", f"sha256:{HEX}", "--eval-tree", "/a:b"], "--eval-tree"),
                 (["--inference-image", f"sha256:{HEX}", "--docker-host=-H"], "--docker-host"),
                 (["--inference-image", f"sha256:{HEX}", "--docker-host", "unix:///a b"], "--docker-host")]
        for extra, field in cases:
            with self.subTest(extra=extra), no_processes():
                code, doc = run(["setup", "keeper", "--packages", "p==1", *extra])
                self.assertEqual((code, doc["problems"][0]["rule"], doc["problems"][0]["field"]),
                                 (3, "AN-19", field), doc)

    def test_unit_and_serve_argv_share_one_builder(self):
        out = setup.setup_commands(owner_uid=7, packages=[], inference_image=f"sha256:{HEX}", eval_trees=["/t"])
        argv = setup.serve_argv(PurePosix(setup.DEFAULT_VENV), PurePosix(setup.DEFAULT_STATE),
                                PurePosix(setup.DEFAULT_SOCKET), 7, inference_image=f"sha256:{HEX}",
                                eval_trees=["/t"])
        self.assertIn("ExecStart=" + " ".join(argv) + "\n", out["unit"])
        self.assertTrue(re.search(r"ExecStart=\S+/gatedtrain keeper serve ", out["unit"]))


if __name__ == "__main__":
    unittest.main()
