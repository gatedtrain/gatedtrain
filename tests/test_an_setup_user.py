"""AN-30, AN-31: `gatedtrain setup keeper --mode same-user` (the default) prints user-level commands only.

Same-user mode has no privileged step: the state directory under `$XDG_DATA_HOME`, the socket under
`$XDG_RUNTIME_DIR`, a `systemctl --user` unit running the keeper from the current environment. Nothing here starts a
process or creates a file. `--mode separate-user` keeps today's output (tests/test_an_keeper_cli.py).
"""

from __future__ import annotations

import configparser
import json
import os
import re
import stat
import sys
import tempfile
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from test_an_keeper_cli import no_processes, run, run_text

from gatedtrain.ainative.keeper import envcheck, setup, setup_user
from gatedtrain.errors import Refused

HEX = "0123456789abcdef" * 4
HEADER = ("# same-user mode: no sudo; run these as YOUR user, not with sudo. The keeper runs under your own OS user, so it does NOT seal data from your "
          "account (an agent with your shell can read it). To harden: gatedtrain setup keeper --mode separate-user "
          "(one sudo).")


# The session facts the commands read (HOME, the XDG directories) are set here, to short fake values: the commands are
# only printed, so nothing needs to exist, and a socket path under a long runtime directory of the host running the
# tests (limit 107 bytes) cannot make a test depend on it.
SESSION = {"HOME": "/home/alice", "XDG_RUNTIME_DIR": "/rt/1234", "XDG_DATA_HOME": "/home/alice/.local/share",
           "XDG_CONFIG_HOME": "/home/alice/.config"}


def session_env(case: unittest.TestCase) -> None:
    patcher = mock.patch.dict(os.environ, SESSION)
    patcher.start()
    case.addCleanup(patcher.stop)


def user_setup(**kw) -> dict:
    kw.setdefault("owner_uid", 1234)
    kw.setdefault("env", {"HOME": "/home/alice"})
    kw.setdefault("executable", "/home/alice/venv/bin/gatedtrain")
    return setup_user.setup_commands_same_user(**kw)


def unit_options(unit: str) -> configparser.ConfigParser:
    cp = configparser.ConfigParser(strict=False, interpolation=None)
    cp.optionxform = str
    cp.read_string(unit)
    return cp


class SameUserCommandsTest(unittest.TestCase):                      # AN-31
    def test_no_sudo_and_no_system_paths(self):
        out = user_setup(env={"HOME": "/home/alice", "XDG_RUNTIME_DIR": "/run/user/1234"})
        text = "\n".join(out["commands"]) + out["unit"]
        self.assertNotIn("sudo", text)
        for root in ("/etc", "/opt", "/srv", "/var"):
            self.assertNotRegex(text, re.escape(root) + r"(/|\b)", root)
        self.assertEqual(out["mode"], "same_user")
        self.assertEqual(set(out), {"commands", "unit", "mode"})

    def test_default_paths_follow_xdg_then_home(self):
        out = user_setup(env={"HOME": "/home/alice", "XDG_DATA_HOME": "/data/alice", "XDG_RUNTIME_DIR": "/rt/1234",
                              "XDG_CONFIG_HOME": "/cfg/alice"})
        joined = "\n".join(out["commands"])
        self.assertIn("mkdir -m 0700 /data/alice/gatedtrain-keeper", joined)
        self.assertIn("--state /data/alice/gatedtrain-keeper --socket /rt/1234/gatedtrain-keeper/keeper.sock "
                      "--owner-uid 1234", out["unit"])
        self.assertIn("/cfg/alice/systemd/user/gatedtrain-keeper.service", joined)
        out = user_setup(env={"HOME": "/home/alice"}, owner_uid=1234)
        joined = "\n".join(out["commands"])
        self.assertIn("mkdir -m 0700 /home/alice/.local/share/gatedtrain-keeper", joined)
        self.assertIn("--socket /run/user/1234/gatedtrain-keeper/keeper.sock", out["unit"])
        self.assertIn("/home/alice/.config/systemd/user/gatedtrain-keeper.service", joined)

    def test_relative_xdg_values_are_ignored(self):
        out = user_setup(env={"HOME": "/home/alice", "XDG_DATA_HOME": "rel/data", "XDG_RUNTIME_DIR": "rel/run"})
        self.assertIn("--state /home/alice/.local/share/gatedtrain-keeper", out["unit"])
        self.assertIn("--socket /run/user/1234/gatedtrain-keeper/keeper.sock", out["unit"])

    def test_unit_is_a_user_unit_that_runs_keeper_serve(self):
        out = user_setup(env={"HOME": "/home/alice", "XDG_RUNTIME_DIR": "/run/user/1234"})
        cp = unit_options(out["unit"])
        self.assertEqual(set(cp.sections()), {"Unit", "Service", "Install"})
        svc = cp["Service"]
        self.assertEqual(svc["ExecStart"], "/home/alice/venv/bin/gatedtrain keeper serve --state "
                         "/home/alice/.local/share/gatedtrain-keeper --socket "
                         "/run/user/1234/gatedtrain-keeper/keeper.sock --owner-uid 1234")
        self.assertEqual(svc["RuntimeDirectory"], "gatedtrain-keeper")        # user managers support it
        self.assertEqual(svc["UMask"], "0077")
        self.assertNotIn("User", svc)
        self.assertNotIn("Group", svc)
        self.assertEqual(cp["Install"]["WantedBy"], "default.target")

    def test_a_socket_outside_the_runtime_directory_gets_its_own_directory(self):
        out = user_setup(socket="/home/alice/sock/keeper.sock", env={"HOME": "/home/alice",
                                                                      "XDG_RUNTIME_DIR": "/run/user/1234"})
        joined = "\n".join(out["commands"])
        self.assertIn("mkdir -m 0700 /home/alice/sock", joined)
        self.assertNotIn("RuntimeDirectory", out["unit"])

    def test_the_runtime_directory_is_not_also_created_by_a_command(self):
        out = user_setup(env={"HOME": "/home/alice", "XDG_RUNTIME_DIR": "/run/user/1234"})
        self.assertNotIn("/run/user/1234/gatedtrain-keeper", "\n".join(c for c in out["commands"] if "mkdir" in c))

    def test_no_command_can_change_the_mode_of_an_existing_directory(self):
        """`install -d -m` resets an existing directory's mode: it is never printed. A new leaf gets 0700 by `mkdir -m`
        (leaf only); an existing one is checked (owner and mode) and refused with a message, never changed."""
        out = user_setup(socket="/home/alice/sock/keeper.sock", env={"HOME": "/home/alice"})
        text = "\n".join(out["commands"])
        self.assertNotIn("install -d", text)
        self.assertNotIn("chmod", text)
        for leaf in ("/home/alice/.local/share/gatedtrain-keeper", "/home/alice/sock"):
            line = next(c for c in out["commands"] if f"mkdir -m 0700 {leaf}" in c)
            self.assertIn('$(id -u):700', line)                              # the guard: owned by the user, mode 0700
            self.assertIn("not changed", line)
            self.assertIn(f"mkdir -p {os.path.dirname(leaf)}", line)         # parents are created plain, leaf only 0700
        self.assertIn("mkdir -p /home/alice/.config/systemd/user", text)      # shared parents: created, never chmod-ed

    def test_the_guard_refuses_an_existing_directory_that_is_not_ours_and_0700(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            leaf = Path(tmp) / "a" / "keeper"
            line = next(c for c in user_setup(state=str(leaf))["commands"] if f"mkdir -m 0700 {leaf}" in c)
            run_sh = lambda: subprocess.run(["sh", "-c", line], capture_output=True, text=True)  # noqa: E731
            self.assertEqual(run_sh().returncode, 0)                           # created
            self.assertEqual(stat.S_IMODE(leaf.stat().st_mode), 0o700)
            self.assertEqual(run_sh().returncode, 0)                           # rerun: idempotent
            leaf.chmod(0o755)
            res = run_sh()
            self.assertNotEqual(res.returncode, 0)                              # shared: refused, with a message
            self.assertIn("not changed", res.stderr)
            self.assertEqual(stat.S_IMODE(leaf.stat().st_mode), 0o755)         # and the mode was not touched

    def test_state_or_socket_in_home_the_runtime_base_or_a_shared_directory_is_refused(self):
        env = {"HOME": "/home/alice", "XDG_RUNTIME_DIR": "/run/user/1234"}
        for kw, field in (({"state": "/home/alice"}, "--state"), ({"state": "/run/user/1234"}, "--state"),
                          ({"state": "/"}, "--state"), ({"socket": "/home/alice/k.sock"}, "--socket"),
                          ({"socket": "/run/user/1234/k.sock"}, "--socket")):
            with self.subTest(kw=kw), self.assertRaises(Refused) as ctx:
                user_setup(env=env, **kw)
            self.assertEqual((ctx.exception.problems[0].rule, ctx.exception.problems[0].field), ("AN-31", field))

    def test_an_existing_shared_directory_is_refused_sticky_or_foreign(self):
        env = {"HOME": "/home/alice"}
        for st_mode, st_uid in ((stat.S_IFDIR | 0o1777, 1234), (stat.S_IFDIR | 0o755, 0)):
            fake = lambda p, m=st_mode, u=st_uid: os.stat_result((m, 0, 0, 1, u, 0, 0, 0, 0, 0))  # noqa: E731
            for kw, field in (({"state": "/srv/shared"}, "--state"), ({"socket": "/srv/shared/k.sock"}, "--socket")):
                with self.subTest(mode=oct(st_mode), kw=kw), self.assertRaises(Refused) as ctx:
                    user_setup(env=env, lstat=lambda p, f=fake: f(p) if str(p) == "/srv/shared" else
                               (_ for _ in ()).throw(FileNotFoundError()), **kw)
                self.assertEqual(ctx.exception.problems[0].field, field)
                self.assertIn("shared", ctx.exception.problems[0].message)

    def test_the_fingerprint_line_uses_the_same_executable_as_the_unit(self):
        out = user_setup(env={"HOME": "/home/alice", "XDG_RUNTIME_DIR": "/run/user/1234"})
        line = next(c for c in out["commands"] if "keeper fingerprint --socket" in c and not c.startswith("#"))
        self.assertEqual(line, "/home/alice/venv/bin/gatedtrain keeper fingerprint --socket "
                         "/run/user/1234/gatedtrain-keeper/keeper.sock")

    def test_the_changes_line_is_true_for_custom_paths(self):
        first = user_setup(state="/srv/k/state", socket="/srv/k/run/k.sock")["commands"][0]
        self.assertNotIn("nothing outside your home", first)
        self.assertIn("never changed", first)

    def test_command_order_and_the_unit_is_written_by_a_printed_command(self):
        out = user_setup()
        commands = out["commands"]

        def at(text: str) -> int:
            for i, c in enumerate(commands):
                if text in c:
                    return i
            self.fail(f"{text!r} is not in the commands")

        order = [at("mkdir -m 0700"), at("tee "), at("systemctl --user daemon-reload"),
                 at("systemctl --user enable --now gatedtrain-keeper.service"),
                 at("no systemd user session?"), at("loginctl enable-linger"),
                 at("gatedtrain keeper fingerprint --socket"), at("keep the fingerprint")]
        self.assertEqual(order, sorted(order), "\n".join(commands))
        self.assertIn(out["unit"], commands[at("tee ")])
        self.assertTrue(commands[-1] == setup.KEEP_FINGERPRINT)
        fallback = commands[at("no systemd user session?")]
        self.assertTrue(fallback.startswith("# ") and "keeper serve --state" in fallback, fallback)
        linger = commands[at("loginctl enable-linger")]
        self.assertTrue(linger.startswith("#") and "administrator" in linger, linger)
        self.assertNotIn("sudo", linger)

    def test_inference_options_reach_exec_start(self):
        out = user_setup(inference_image=f"sha256:{HEX}", eval_trees=["/x"],
                         docker_host="unix:///run/user/1234/docker.sock")
        line = unit_options(out["unit"])["Service"]["ExecStart"]
        self.assertTrue(line.endswith(f"--owner-uid 1234 --inference-image sha256:{HEX} --eval-tree /x "
                                      "--docker-host unix:///run/user/1234/docker.sock"), line)

    def test_bad_values_are_refused(self):
        for kw, field in (({"state": "relative"}, "--state"), ({"socket": "/a/../b.sock"}, "--socket"),
                          ({"state": "/with space/k"}, "--state"), ({"inference_image": "python:3"}, "--inference-image"),
                          ({"socket": "/s/" + "x" * 200}, "--socket")):
            with self.subTest(kw=kw):
                with self.assertRaises(Refused) as ctx:
                    user_setup(**kw)
                self.assertEqual((ctx.exception.problems[0].rule, ctx.exception.problems[0].field),
                                 ("AN-31", field))

    def test_no_home_is_refused(self):
        with self.assertRaises(Refused) as ctx:
            user_setup(env={}, home="")
        self.assertEqual(ctx.exception.problems[0].rule, "AN-31")

    def test_default_executable_is_the_one_next_to_this_interpreter(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(setup_user.keeper_executable(python=str(Path(tmp) / "python")), None)
            exe = Path(tmp) / "gatedtrain"
            exe.write_text("#!/bin/sh\n")
            exe.chmod(0o755)
            self.assertEqual(setup_user.keeper_executable(python=str(Path(tmp) / "python")), str(exe))
        self.assertIsInstance(setup_user.keeper_executable(), (str, type(None)))

    def test_envcheck_accepts_the_owners_own_directories(self):     # the keeper serve start-up check as the owner
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "data" / "gatedtrain-keeper"
            state.mkdir(parents=True, mode=0o700)
            os.chmod(state, 0o700)
            sock_dir = Path(tmp) / "run" / "gatedtrain-keeper"
            sock_dir.mkdir(parents=True, mode=0o700)
            uid = os.geteuid()
            problems = envcheck.path_problems([state.parent], uid) + envcheck.path_problems([sock_dir], uid)
            problems += envcheck.state_dir_problems(state, uid)
            own = [p for p in problems if str(tmp) in p.message or "gatedtrain-keeper" in p.message]
            self.assertEqual(own, [], problems)
            self.assertEqual(stat.S_IMODE(state.stat().st_mode), 0o700)


class SetupKeeperModeCliTest(unittest.TestCase):                    # AN-30, AN-31
    def setUp(self):
        session_env(self)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def test_default_mode_is_same_user_and_prints_the_header_first(self):
        with no_processes():
            code, text, err = run_text(["setup", "keeper"])
        self.assertEqual(code, 0, err)
        self.assertEqual(text.splitlines()[0], HEADER)
        self.assertNotRegex("\n".join(text.splitlines()[1:]), r"\bsudo\b")   # the venv path may hold the letters
        self.assertIn("systemctl --user enable --now gatedtrain-keeper.service", text)

    def test_json_data_carries_the_mode(self):
        with no_processes():
            code, doc = run(["setup", "keeper"])
            self.assertEqual((code, doc["data"]["mode"], set(doc["data"])), (0, "same_user",
                                                                           {"commands", "unit", "mode"}), doc)
            code, doc = run(["setup", "keeper", "--mode", "same-user"])
            self.assertEqual(doc["data"]["mode"], "same_user")
            code, doc = run(["setup", "keeper", "--mode", "separate-user", "--packages", "p==1"])
            self.assertEqual((code, doc["data"]["mode"]), (0, "separate_user"), doc)
            self.assertIn("sudo useradd", "\n".join(doc["data"]["commands"]))

    def test_unknown_mode_is_a_usage_error(self):
        with self.assertRaises(SystemExit) as ctx, no_processes():
            run_text(["setup", "keeper", "--mode", "root"])
        self.assertEqual(ctx.exception.code, 2)

    def test_user_and_venv_are_refused_in_same_user_mode(self):
        with no_processes():
            for flag, value in (("--user", "gatedtrain-keeper"), ("--venv", "/opt/x/venv")):
                with self.subTest(flag=flag):
                    code, doc = run(["setup", "keeper", flag, value])
                    p = doc["problems"][0]
                    self.assertEqual((code, p["rule"], p["field"]), (3, "AN-30", flag), doc)
                    self.assertIn("same-user", p["message"])
                    self.assertIn("separate-user", p["fix"])

    def test_state_and_socket_are_allowed_in_same_user_mode(self):
        with no_processes():
            code, doc = run(["setup", "keeper", "--state", "/srv/gk/s", "--socket", "/srv/gk/r/k.sock",
                             "--owner-uid", "4321"])
        self.assertEqual(code, 0, doc)
        self.assertIn("--state /srv/gk/s --socket /srv/gk/r/k.sock --owner-uid 4321", doc["data"]["unit"])
        self.assertEqual(list(self.tmp.iterdir()), [], "setup created a file")

    def test_packages_are_ignored_with_a_warning_in_same_user_mode(self):
        with no_processes():
            code, doc = run(["setup", "keeper", "--packages", "gatedtrain-pack-x==1.0"])
        self.assertEqual(code, 0, doc)
        self.assertTrue(any("--packages" in w and "ignored" in w for w in doc["warnings"]), doc["warnings"])
        self.assertNotIn("gatedtrain-pack-x", "\n".join(doc["data"]["commands"]) + doc["data"]["unit"])

    def test_no_pack_warning_is_not_raised_in_same_user_mode(self):
        with no_processes():
            code, doc = run(["setup", "keeper"])
        self.assertFalse(any("scenario pack" in w for w in doc["warnings"]), doc["warnings"])

    def test_same_user_mode_runs_nothing_and_writes_nothing(self):
        cwd = os.getcwd()
        os.chdir(self.tmp)
        self.addCleanup(os.chdir, cwd)
        with no_processes():
            code, doc = run(["setup", "keeper"])
        self.assertEqual((code, list(self.tmp.iterdir())), (0, []))

    def test_separate_user_refuses_nothing_new(self):
        with no_processes():
            code, doc = run(["setup", "keeper", "--mode", "separate-user", "--user", "kp", "--venv", "/opt/v",
                             "--packages", "p==1"])
        self.assertEqual(code, 0, doc)
        self.assertIn("User=kp\n", doc["data"]["unit"])
        self.assertIn(f"ExecStart=/opt/v/bin/gatedtrain keeper serve --state {setup.DEFAULT_STATE}",
                      doc["data"]["unit"])


class ReviewRound2Test(unittest.TestCase):                          # L9
    def setUp(self):
        session_env(self)

    def test_the_header_says_to_run_the_commands_as_your_own_user(self):
        self.assertIn("run these as YOUR user, not with sudo", setup_user.HEADER)
        with no_processes():
            _, text, _ = run_text(["setup", "keeper"])
        self.assertIn("run these as YOUR user, not with sudo", text.splitlines()[0])

    def test_a_restart_step_is_printed_as_a_comment_for_changed_options(self):
        cmds = user_setup()["commands"]
        line = next(c for c in cmds if "restart" in c)
        self.assertTrue(line.startswith("#"), line)
        self.assertIn("to apply changed options", line)
        self.assertIn("systemctl --user restart gatedtrain-keeper.service", line)
        enable = next(c for c in cmds if "enable --now" in c)
        self.assertEqual(cmds.index(line), cmds.index(enable) + 1)         # right after the enable it complements

    def test_a_failing_parent_mkdir_does_not_claim_the_leaf_exists(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "afile").write_text("x")
            leaf = Path(tmp) / "afile" / "sub" / "keeper"                  # mkdir -p cannot create under a file
            line = next(c for c in user_setup(state=str(leaf))["commands"] if f"mkdir -m 0700 {leaf}" in c)
            res = subprocess.run(["sh", "-c", line], capture_output=True, text=True)
            self.assertNotEqual(res.returncode, 0)
            self.assertNotIn("exists and is not a directory", res.stderr)
            self.assertTrue(res.stderr.strip())                            # mkdir's own message is there

    def test_empty_option_values_are_refused_in_both_modes(self):
        with no_processes():
            for mode in ("same-user", "separate-user"):
                for flag in ("--state", "--socket", "--user", "--venv"):
                    if mode == "same-user" and flag in ("--user", "--venv"):
                        continue                                           # refused for another reason there
                    code, doc = run(["setup", "keeper", "--mode", mode, flag, ""])
                    self.assertEqual(code, 3, (mode, flag))
                    self.assertIn("empty", json.dumps(doc), (mode, flag))


class PrintedScriptStopsTest(unittest.TestCase):                    # final review F1
    """The printed block is ONE script that stops at the first failing step: a bad existing directory must not be
    followed by the unit write and `systemctl --user enable --now`."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.home = self.tmp / "h"
        self.home.mkdir()
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        self.calls = self.tmp / "systemctl.calls"
        self.stub("systemctl", f'echo "$@" >> {self.calls}\nexit 0\n')
        self.stub("gatedtrain", "exit 0\n")
        self.state = self.home / ".local" / "share" / "gatedtrain-keeper"
        self.unit = self.home / ".config" / "systemd" / "user" / "gatedtrain-keeper.service"

    def stub(self, name: str, body: str) -> None:
        path = self.bin / name
        path.write_text("#!/bin/sh\n" + body)
        path.chmod(0o755)

    def printed(self) -> tuple[Path, dict]:
        env = {"HOME": str(self.home), "XDG_RUNTIME_DIR": "/rt/1234", "XDG_DATA_HOME": str(self.home / ".local/share"),
               "XDG_CONFIG_HOME": str(self.home / ".config")}
        out = setup_user.setup_commands_same_user(owner_uid=os.getuid(), env=env, home=str(self.home),
                                                  executable=str(self.bin / "gatedtrain"))
        script = self.tmp / "printed.sh"
        script.write_text("\n".join(out["commands"]) + "\n")
        return script, out

    def sh(self, script: Path) -> subprocess.CompletedProcess:
        env = {"HOME": str(self.home), "PATH": f"{self.bin}:{os.environ['PATH']}"}
        return subprocess.run(["sh", str(script)], capture_output=True, text=True, env=env)

    def test_the_block_starts_with_the_script_header_and_set_e(self):
        _, out = self.printed()
        self.assertIn("as one script with sh", out["commands"][1])
        self.assertIn("or paste it whole", out["commands"][1])
        self.assertEqual(out["commands"][2], "set -e")
        self.assertTrue(all(c.startswith("#") for c in out["commands"][:2]))     # nothing runs before `set -e`
        text = "\n".join(out["commands"])
        self.assertIn("exit 1", text)
        self.assertNotIn("false; }", text)                         # the guard exits, it does not just evaluate false

    def test_a_shared_state_directory_stops_the_script_before_the_unit_and_systemctl(self):
        script, _ = self.printed()
        self.state.parent.mkdir(parents=True)
        self.state.mkdir(mode=0o755)
        os.chmod(self.state, 0o755)
        res = self.sh(script)
        self.assertNotEqual(res.returncode, 0, res.stdout + res.stderr)
        self.assertIn("stopped:", res.stderr)
        self.assertFalse(self.unit.exists(), "the unit was written after the guard stopped the script")
        self.assertFalse(self.calls.exists(), "systemctl was reached after the guard stopped the script")
        self.assertEqual(stat.S_IMODE(self.state.stat().st_mode), 0o755)      # and nothing was changed

    def test_a_clean_run_succeeds_and_a_second_run_is_idempotent(self):
        script, out = self.printed()
        first = self.sh(script)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assertEqual(stat.S_IMODE(self.state.stat().st_mode), 0o700)
        self.assertEqual(self.unit.read_text(), out["unit"])
        calls = self.calls.read_text().splitlines()
        self.assertEqual(calls, ["--user daemon-reload", "--user enable --now gatedtrain-keeper.service"])
        second = self.sh(script)
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertEqual(self.unit.read_text(), out["unit"])
        self.assertEqual(stat.S_IMODE(self.state.stat().st_mode), 0o700)

    def test_a_failing_systemctl_stops_before_the_fingerprint_step(self):
        script, _ = self.printed()
        self.stub("systemctl", "exit 1\n")
        self.stub("gatedtrain", f'echo ran >> {self.tmp / "fingerprint.ran"}\n')
        res = self.sh(script)
        self.assertNotEqual(res.returncode, 0)
        self.assertFalse((self.tmp / "fingerprint.ran").exists())

    def test_the_doc_and_module_text_no_longer_overclaim(self):
        self.assertIn("the script stops there", setup_user.CHANGES)


class ControlCharactersTest(unittest.TestCase):                     # final review F2
    BAD = ("/tmp/a\u001b[2Jb\u0000c", "/tmp/a\u2028b", "/tmp/a\u2029b", "/tmp/a\u0085b", "/tmp/a\u007fb",
           "/tmp/a\u202eb")

    def test_validators_refuse_every_non_printable_value(self):
        from gatedtrain.ainative import hostsetup
        for bad in self.BAD:
            with self.subTest(value=bad):
                for call in (lambda: setup._absolute("--state", bad), lambda: hostsetup._path("--state", bad),
                             lambda: setup.check_eval_tree(bad),
                             lambda: setup._inference_options(f"sha256:{HEX}", [], "unix://" + bad)):
                    with self.assertRaises(Refused) as ctx:
                        call()
                    text = ctx.exception.problems[0].line()
                    self.assertTrue(text.isprintable(), repr(text))
        for ref in ("name@sha256:" + HEX + "\u001b", "na\u0000me@sha256:" + HEX):
            with self.assertRaises(Refused):
                hostsetup.check_inference_ref(ref)

    def test_same_user_refuses_them_through_flags_and_the_layout(self):
        session_env(self)
        for bad in self.BAD:
            for flag in ("--state", "--socket"):
                with self.subTest(flag=flag, value=bad), no_processes():
                    code, text, err = run_text(["setup", "keeper", flag, bad])
                    self.assertEqual(code, 3, (text, err))
                    self.assertEqual(text, "")
                    self.assertTrue((text + err).replace("\n", "").isprintable(), repr(err))
            with tempfile.TemporaryDirectory() as tmp, no_processes():
                lay = Path(tmp) / "l.toml"
                lay.write_text(f'schema_version = 1\n[same_user]\nstate = {json.dumps(bad)}\n')
                code, text, err = run_text(["setup", "keeper", "--layout", str(lay)])
                self.assertEqual(code, 3, (text, err))
                self.assertEqual(text, "")
                self.assertTrue(err.replace("\n", "").isprintable(), repr(err))

    def test_separate_user_and_host_refuse_them_too(self):
        session_env(self)
        for bad in self.BAD:
            for argv in (["setup", "keeper", "--mode", "separate-user", "--state", bad],
                         ["setup", "keeper", "--mode", "separate-user", "--eval-tree", bad, "--inference-image",
                          f"sha256:{HEX}"],
                         ["setup", "host", "--keeper-home", bad]):
                with self.subTest(argv=argv), no_processes():
                    code, text, err = run_text(argv)
                    self.assertEqual(code, 3, (text, err))
                    self.assertTrue(err.replace("\n", "").isprintable(), repr(err))
            with tempfile.TemporaryDirectory() as tmp, no_processes():
                lay = Path(tmp) / "l.toml"
                lay.write_text(f'schema_version = 1\n[keeper]\nstate = {json.dumps(bad)}\n')
                code, text, err = run_text(["setup", "keeper", "--mode", "separate-user", "--layout", str(lay)])
                self.assertEqual(code, 3, (text, err))

    def test_the_printing_path_escapes_a_control_character_that_reaches_a_command_line(self):
        """Defence in depth: whatever a future validator misses, the terminal never receives a raw escape."""
        session_env(self)
        forged = {"commands": ["echo ok", "echo \u001b[2Jhidden\u2028forged"], "unit": "[Unit]\n", "mode": "same_user"}
        with no_processes(), mock.patch.object(setup_user, "setup_commands_same_user", return_value=forged):
            code, text, err = run_text(["setup", "keeper"])
        self.assertEqual(code, 0, err)
        self.assertNotIn("\u001b", text)
        self.assertNotIn("\u2028", text)
        self.assertIn("\\x1b[2Jhidden\\u2028forged", text)
        self.assertTrue(text.replace("\n", "").isprintable())


class ExecutableTest(unittest.TestCase):                            # final review F10
    def test_the_executable_next_to_this_interpreter_is_used_and_the_path_is_never_searched(self):
        with tempfile.TemporaryDirectory() as tmp:
            py = Path(tmp) / "venv" / "bin" / "python"
            py.parent.mkdir(parents=True)
            other = Path(tmp) / "other"
            other.mkdir()
            exe = other / "gatedtrain"
            exe.write_text("#!/bin/sh\n")
            exe.chmod(0o755)                                          # on PATH, but in another environment
            with mock.patch.dict(os.environ, {"PATH": str(other)}), mock.patch.object(sys, "executable", str(py)):
                self.assertIsNone(setup_user.keeper_executable())
                with self.assertRaises(Refused) as ctx:
                    setup_user.setup_commands_same_user(owner_uid=1, env=SESSION, home="/home/alice")
            p = ctx.exception.problems[0]
            self.assertEqual((p.rule, p.field), ("AN-31", "--executable"))
            self.assertIn(str(py.parent), p.message)
            self.assertIn("--executable", p.fix)
            own = py.parent / "gatedtrain"
            own.write_text("#!/bin/sh\n")
            own.chmod(0o755)
            with mock.patch.object(sys, "executable", str(py)):
                self.assertEqual(setup_user.keeper_executable(), str(own))

    def test_a_relative_interpreter_path_gives_an_absolute_executable(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp) / "bin"
            bin_dir.mkdir()
            exe = bin_dir / "gatedtrain"
            exe.write_text("#!/bin/sh\n")
            exe.chmod(0o755)
            cwd = os.getcwd()
            os.chdir(tmp)
            self.addCleanup(os.chdir, cwd)
            self.assertEqual(setup_user.keeper_executable(python="bin/python"), str(exe.resolve()) if False else
                             os.path.join(os.getcwd(), "bin", "gatedtrain"))

    def test_the_unit_directory_is_not_held_to_the_unit_safe_rule(self):
        env = {"HOME": "/home/a b", "XDG_CONFIG_HOME": "/cfg/100%$ x", "XDG_RUNTIME_DIR": "/rt/1"}
        out = setup_user.setup_commands_same_user(owner_uid=1, env=env, home="/home/a b", state="/srv/k/state",
                                                  socket="/rt/1/gatedtrain-keeper/k.sock",
                                                  executable="/home/alice/venv/bin/gatedtrain")
        text = "\n".join(out["commands"])
        self.assertIn("mkdir -p '/cfg/100%$ x/systemd/user'", text)
        self.assertIn("tee '/cfg/100%$ x/systemd/user/gatedtrain-keeper.service'", text)
        self.assertNotIn("/cfg/100", out["unit"])                    # the directory is not part of ExecStart

    def test_a_state_default_under_an_unsafe_home_is_still_refused_naming_the_flag(self):
        with self.assertRaises(Refused) as ctx:
            user_setup(env={"HOME": "/home/a b", "XDG_RUNTIME_DIR": "/rt/1"}, home="/home/a b")
        p = ctx.exception.problems[0]
        self.assertEqual(p.field, "--state")
        self.assertIn("--state", p.fix)

    def test_an_executable_path_systemd_would_rewrite_names_the_options(self):
        for bad in ("/home/a b/venv/bin/gatedtrain", "/home/100%/v/bin/gatedtrain", "/home/$x/v/gatedtrain"):
            with self.subTest(path=bad), self.assertRaises(Refused) as ctx:
                user_setup(executable=bad)
            p = ctx.exception.problems[0]
            self.assertEqual((p.rule, p.field), ("AN-31", "--executable"))
            self.assertIn("--executable", p.fix)
            self.assertIn("[same_user] executable", p.fix)

    def test_the_executable_option_and_layout_key_are_used_validated_and_mode_checked(self):
        session_env(self)
        with no_processes():
            code, doc = run(["setup", "keeper", "--executable", "/opt/plain/gatedtrain"])
            self.assertEqual(code, 0, doc)
            self.assertIn("ExecStart=/opt/plain/gatedtrain keeper serve", doc["data"]["unit"])
            self.assertIn("/opt/plain/gatedtrain keeper fingerprint --socket", "\n".join(doc["data"]["commands"]))
            for bad in ("relative/gatedtrain", "/a b/gatedtrain", "/a/../gatedtrain"):
                code, doc = run(["setup", "keeper", "--executable", bad])
                self.assertEqual((code, doc["problems"][0]["field"]), (3, "--executable"), bad)
            code, doc = run(["setup", "keeper", "--mode", "separate-user", "--executable", "/opt/p/gatedtrain"])
            self.assertEqual((code, doc["problems"][0]["rule"], doc["problems"][0]["field"]),
                             (3, "AN-30", "--executable"), doc)
            with tempfile.TemporaryDirectory() as tmp:
                lay = Path(tmp) / "l.toml"
                lay.write_text('schema_version = 1\n[same_user]\nexecutable = "/opt/layout/gatedtrain"\n')
                code, doc = run(["setup", "keeper", "--layout", str(lay)])
                self.assertEqual(code, 0, doc)
                self.assertIn("ExecStart=/opt/layout/gatedtrain ", doc["data"]["unit"])
                self.assertEqual(doc["data"]["layout"]["keys"]["same_user.executable"], "layout")
                code, doc = run(["setup", "keeper", "--layout", str(lay), "--executable", "/opt/flag/gatedtrain"])
                self.assertIn("ExecStart=/opt/flag/gatedtrain ", doc["data"]["unit"])


class DefaultModeWarningTest(unittest.TestCase):                    # final review F8
    def setUp(self):
        session_env(self)

    def test_an_omitted_mode_warns_once_in_json_and_on_stderr(self):
        with no_processes():
            code, doc = run(["setup", "keeper"])
            self.assertEqual(code, 0, doc)
            hits = [w for w in doc["warnings"] if w == setup_user.DEFAULT_MODE_WARNING]
            self.assertEqual(len(hits), 1, doc["warnings"])
            code, text, err = run_text(["setup", "keeper"])
        self.assertEqual(err.count(setup_user.DEFAULT_MODE_WARNING), 1, err)
        self.assertNotIn(setup_user.DEFAULT_MODE_WARNING, text)
        self.assertIn("default changed in 0.7.0 to same-user (no sudo)", setup_user.DEFAULT_MODE_WARNING)
        self.assertIn("--mode separate-user", setup_user.DEFAULT_MODE_WARNING)

    def test_an_explicit_mode_has_no_such_warning(self):
        with no_processes():
            for mode in ("same-user", "separate-user"):
                code, doc = run(["setup", "keeper", "--mode", mode, "--packages", "p==1"])
                self.assertEqual(code, 0, doc)
                self.assertNotIn(setup_user.DEFAULT_MODE_WARNING, doc["warnings"], mode)
                code, text, err = run_text(["setup", "keeper", "--mode", mode, "--packages", "p==1"])
                self.assertNotIn("default changed", err, mode)

    def test_the_default_output_is_still_the_same_user_output(self):
        with no_processes():
            _, default = run(["setup", "keeper"])
            _, explicit = run(["setup", "keeper", "--mode", "same-user"])
        self.assertEqual(default["data"], explicit["data"])


if __name__ == "__main__":
    unittest.main()
