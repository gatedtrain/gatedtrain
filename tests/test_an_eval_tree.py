"""AN-36: `gatedtrain setup eval-tree` prints a user-level script that builds the keeper's eval tree for a same-user
keeper, and `setup eval-tree --check` verifies a built tree with the keeper's own rule.

The command prints and never runs (AN-19); `--check` only reads. The script shares the requirements validator with
`setup host` (AN-20), builds from hash-pinned binary wheels only, and ends by running the check, so the keeper's
`envcheck.eval_tree_problems` stays the one authority on what a tree may be. The tests run the printed script against
fake `uv`, `realpath`-able interpreters and a fake `id` in a temporary directory: no network, no real uv, no root.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import re
import shlex
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

from test_an_keeper_cli import no_processes, run, run_text

from gatedtrain.ainative import evaltree, hostsetup, layout
from gatedtrain.ainative.keeper import envcheck, setup, setup_user
from gatedtrain.errors import Refused

REQS = (b"# uv export --format requirements-txt --no-emit-project\n"
        b"vllm==0.11.0 \\\n    --hash=sha256:" + b"1" * 64 + b" \\\n    --hash=sha256:" + b"2" * 64 + b"\n"
        b"numpy==2.2.6 --hash=sha256:" + b"3" * 64 + b"\n")
SESSION = {"HOME": "/home/alice", "XDG_DATA_HOME": "/home/alice/.local/share", "XDG_RUNTIME_DIR": "/rt/1234",
           "XDG_CONFIG_HOME": "/home/alice/.config"}
EXE = "/venvs/gt/bin/gatedtrain"


def help_text(*argv: str) -> str:
    from gatedtrain import cli

    with contextlib.redirect_stdout(io.StringIO()) as out, self_exit_ok():
        cli.main(list(argv))
    return out.getvalue()


@contextlib.contextmanager
def self_exit_ok():
    try:
        yield
    except SystemExit as exc:                                        # argparse prints help and exits 0
        assert exc.code == 0, exc.code


def build(**kw) -> dict:
    args = dict(requirements=REQS, python_version="3.12", root=None, env=SESSION, home="/home/alice", owner_uid=1234,
                executable=EXE)
    args.update(kw)
    return evaltree.eval_tree_script(**args)


def commands_only(script: str) -> str:
    return "\n".join(line for line in script.splitlines() if not line.lstrip().startswith("#"))


class ScriptTest(unittest.TestCase):                                # AN-36: what is printed
    def test_the_default_root_comes_from_the_xdg_data_directory_not_a_constant(self):
        out = build()
        self.assertEqual(out["root"], "/home/alice/.local/share/gatedtrain-keeper-eval")
        out = build(env={"HOME": "/home/bob"}, home="/home/bob")
        self.assertEqual(out["root"], "/home/bob/.local/share/gatedtrain-keeper-eval")
        out = build(env={"HOME": "/h", "XDG_DATA_HOME": "relative"}, home="/h")     # a relative XDG value is invalid
        self.assertEqual(out["root"], "/h/.local/share/gatedtrain-keeper-eval")
        with self.assertRaises(Refused) as cm:
            build(env={}, home=None)
        self.assertEqual(cm.exception.problems[0].rule, evaltree.RULE)

    def test_an_explicit_root_wins_and_is_validated(self):
        self.assertEqual(build(root="/data/ev")["root"], "/data/ev")
        for bad in ("rel/ev", "/", "/a b", "/a:b", "/a,b", "/a$b", "/a;b", "/a\nb", ""):
            with self.subTest(root=bad), self.assertRaises(Refused) as cm:
                build(root=bad)
            self.assertEqual(cm.exception.problems[0].rule, evaltree.RULE)

    def test_the_trees_and_interpreter_to_hand_to_the_keeper(self):
        out = build(root="/data/ev")
        self.assertEqual(out["trees"], ["/data/ev/python", "/data/ev/eval-venv"])
        self.assertEqual(out["eval_python"], "/data/ev/eval-venv/bin/python")
        # the keeper's own validators accept exactly these values, and `setup keeper --mode same-user` prints them
        for tree in out["trees"]:
            setup.check_eval_tree(tree)
        setup.check_eval_python(out["eval_python"])
        unit = setup_user.setup_commands_same_user(
            owner_uid=1234, env=SESSION, home="/home/alice", executable=EXE, inference_image="sha256:" + "a" * 64,
            eval_trees=out["trees"], eval_python=out["eval_python"], docker_host="unix:///rt/1234/docker.sock")
        for tree in out["trees"]:
            self.assertIn(f"--eval-tree {tree}", unit["unit"])
        self.assertIn("--eval-python /data/ev/eval-venv/bin/python", unit["unit"])
        self.assertTrue(all(a in out["keeper_args"] for a in ("--eval-tree", "--eval-python", out["eval_python"])))

    def test_the_script_has_no_privilege_and_no_machine_path(self):
        script = build()["script"]
        body = commands_only(script)
        for word in ("sudo", "chown", "su ", "setfacl", "/opt", "/srv", "/etc", "/var", "apt-get", "systemctl",
                     "docker"):
            self.assertNotIn(word, body, word)
        self.assertNotIn("alice", script.replace("/home/alice", ""))      # only the printed root names the home
        self.assertTrue(script.startswith("#!/bin/sh\n"))
        self.assertIn("\nset -eu\n", script)
        self.assertIn("umask 022", script)

    def test_the_header_says_who_this_tree_is_for(self):
        script = build()["script"]
        self.assertIn("same-user keeper only", script)
        self.assertIn("a separate-user keeper refuses this tree", script)
        self.assertIn("setup host", script)                                # where the separate-user tree comes from
        self.assertIn("rm -rf", script)                                    # the rollback is printed, as a comment
        self.assertTrue(all(l.lstrip().startswith(("#", "[")) and (l.lstrip().startswith("#") or "|| fail" in l)
                            for l in script.splitlines() if "rm -rf" in l))       # a comment, or text of a refusal

    def test_requirements_are_validated_by_the_code_setup_host_uses(self):
        bad = {
            "unhashed": b"vllm==0.11.0\n",
            "option line": b"--index-url https://example.invalid/simple\nvllm==0.11.0 --hash=sha256:" + b"1" * 64 + b"\n",
            "control": b"vllm==0.11.0 --hash=sha256:" + b"1" * 64 + b"\x07\n",
            "marker": b"vllm==0.11.0 --hash=sha256:" + b"1" * 64 + b"\nGATEDTRAIN_EVAL_REQUIREMENTS\n",
            "empty": b"# nothing\n",
            "not utf-8": b"\xff\xfe",
        }
        for name, data in bad.items():
            with self.subTest(name), self.assertRaises(Refused) as cm:
                build(requirements=data)
            self.assertEqual(cm.exception.problems[0].rule, evaltree.RULE)
            self.assertEqual(cm.exception.problems[0].field, "--requirements")
        # the same function, so a requirement `setup host` refuses is refused here and the reverse
        with self.assertRaises(Refused) as cm:
            hostsetup._requirements(bad["unhashed"])
        self.assertEqual(cm.exception.problems[0].rule, hostsetup.RULE)

    def test_the_requirements_text_and_its_hash_are_in_the_script_once(self):
        out = build()
        digest = hashlib.sha256(REQS).hexdigest()
        self.assertEqual(out["requirements_sha256"], digest)
        self.assertEqual(out["script"].count(digest), 1)
        self.assertIn(REQS.decode(), out["script"])
        self.assertIn("--require-hashes", out["script"])
        self.assertIn("--only-binary :all:", out["script"])
        self.assertIn("--no-deps", out["script"])

    def test_the_python_version_is_required_and_plain(self):
        for ok in ("3.12", "3.12.16", "3.13"):
            self.assertIn(shlex.quote(ok), build(python_version=ok)["script"])
        for bad in ("", "3", "3.12.x", "-p", "3.12 ", "3.12;rm", "cpython-3.12", "3.12\n", None, 3.12):
            with self.subTest(v=bad), self.assertRaises(Refused) as cm:
                build(python_version=bad)
            self.assertEqual(cm.exception.problems[0].field, "--python-version")

    def test_a_root_uid_and_a_missing_executable_are_refused(self):
        with self.assertRaises(Refused):
            build(owner_uid=0)
        with self.assertRaises(Refused) as cm:
            build(executable="relative/gatedtrain")
        self.assertEqual(cm.exception.problems[0].field, "--executable")

    def test_the_output_is_deterministic(self):
        self.assertEqual(build(), build())


class CliTest(unittest.TestCase):                                   # AN-36 through the command line
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.reqs = self.tmp / "eval-requirements.txt"
        self.reqs.write_bytes(REQS)
        cwd = os.getcwd()
        work = self.tmp / "cwd"
        work.mkdir()
        os.chdir(work)
        self.addCleanup(os.chdir, cwd)
        self.work = work
        self.base = ["setup", "eval-tree", "--requirements", str(self.reqs), "--python-version", "3.12",
                     "--root", str(self.tmp / "ev"), "--executable", EXE]

    def test_prints_and_creates_nothing(self):
        with no_processes():
            code, doc = run(self.base)
        self.assertEqual(code, 0, doc)
        self.assertEqual(list(self.work.iterdir()), [])
        self.assertFalse((self.tmp / "ev").exists())
        data = doc["data"]
        self.assertEqual(set(data), {"script", "command", "root", "trees", "eval_python", "keeper_args",
                                     "requirements_sha256", "mode"})
        self.assertEqual(data["mode"], "same_user")
        self.assertEqual(data["command"], f"sh {evaltree.SCRIPT_NAME}")

    def test_text_output_is_the_script(self):
        with no_processes():
            code, out, err = run_text(self.base)
        self.assertEqual((code, err), (0, ""))
        self.assertTrue(out.startswith("#!/bin/sh\n"))

    def test_requirements_and_python_version_are_required_unless_checking(self):
        for drop in ("--requirements", "--python-version"):
            argv = list(self.base)
            i = argv.index(drop)
            del argv[i:i + 2]
            code, doc = run(argv)
            self.assertEqual(code, 3, drop)
            self.assertEqual(doc["problems"][0]["rule"], evaltree.RULE)
            self.assertEqual(doc["problems"][0]["field"], drop)

    def test_an_unreadable_requirements_file_is_refused(self):
        argv = list(self.base)
        argv[argv.index("--requirements") + 1] = str(self.tmp / "missing.txt")
        code, doc = run(argv)
        self.assertEqual(code, 3)
        self.assertEqual(doc["problems"][0]["field"], "--requirements")

    def test_the_layout_key_sets_the_root_and_a_flag_overrides_it(self):
        lay = self.tmp / "layout.toml"
        lay.write_text('schema_version = 1\n[eval_tree]\nroot = "/data/from-layout"\n')
        argv = [a for a in self.base if a not in ("--root", str(self.tmp / "ev"))]
        code, doc = run([*argv, "--layout", str(lay)])
        self.assertEqual((code, doc["data"]["root"]), (0, "/data/from-layout"), doc)
        self.assertIn("# layout:", doc["data"]["script"])
        code, doc = run([*argv, "--layout", str(lay), "--root", "/data/flag"])
        self.assertEqual((code, doc["data"]["root"]), (0, "/data/flag"))

    def test_mode_words_are_not_the_sealed_claim(self):
        with no_processes():
            _, doc = run(self.base)
        text = doc["data"]["script"]
        self.assertNotIn("sealed", text.lower().replace("does not seal", ""))      # comments and echoes included
        self.assertIn("does not seal", text)

    def test_the_root_may_not_start_with_two_slashes(self):
        with self.assertRaises(Refused):
            build(root="//data/ev")


class ScriptRunTest(unittest.TestCase):
    """The printed script is run by `sh` against stand-ins for `uv`, `realpath`-able interpreters and pip."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        self.log = self.tmp / "calls.log"
        self.root = self.tmp / "data" / "ev"
        self.write_fake_tools()

    def tool(self, name: str, body: str) -> None:
        p = self.bin / name
        p.write_text("#!/bin/sh\n" + body)
        p.chmod(0o755)

    def write_fake_tools(self):
        # uv: `python install` makes <UV_PYTHON_INSTALL_DIR>/cpython-3.12.1-x/bin/python3.12 (a fake `python -m venv`
        # and pip), and a minor-version link, as real uv does; `python find` prints the link's interpreter.
        self.tool("uv", f'''
echo "uv $*" >> {self.log}
case "$*" in
  *"python install"*)
    case "$*" in *--no-bin*--no-registry*|*--no-registry*--no-bin*) ;; *) echo "uv without --no-bin --no-registry" >&2; exit 9 ;; esac
    d="$UV_PYTHON_INSTALL_DIR/cpython-3.12.1-linux-x86_64-gnu"
    mkdir -p "$d/bin"
    cat > "$d/bin/python3.12" <<'PYEOF'
#!/bin/sh
echo "python $*" >> {self.log}
if [ "$1" = "-I" ] && [ "$2" = "-m" ] && [ "$3" = "venv" ]; then
  v="$5"; mkdir -p "$v/bin"
  cat > "$v/bin/pip" <<'PIPEOF'
#!/bin/sh
echo "pip $*" >> {self.log}
echo "PIP_CONFIG_FILE=$PIP_CONFIG_FILE" >> {self.log}
exit 0
PIPEOF
  chmod 755 "$v/bin/pip"; ln -s python3.12 "$v/bin/python"
fi
PYEOF
    chmod 755 "$d/bin/python3.12"
    touch "$UV_PYTHON_INSTALL_DIR/.lock"; chmod 666 "$UV_PYTHON_INSTALL_DIR/.lock"
    ln -s "$d" "$UV_PYTHON_INSTALL_DIR/cpython-3.12-linux-x86_64-gnu"
    chmod 755 "$UV_PYTHON_INSTALL_DIR"
    ;;
  *"python find"*) echo "$UV_PYTHON_INSTALL_DIR/cpython-3.12-linux-x86_64-gnu/bin/python3.12" ;;
esac
''')
        self.tool("id", f'case "$1" in -u) echo "${{FAKE_UID:-{os.getuid()}}}";; *) /usr/bin/id "$@";; esac\n')

    def printed(self, **kw) -> Path:
        env = {"HOME": str(self.tmp / "home")}
        out = build(env=env, home=env["HOME"], root=str(self.root), owner_uid=os.getuid(),
                    executable=str(self.bin / "gatedtrain"), **kw)
        path = self.tmp / "script.sh"
        path.write_text(out["script"])
        self.assertEqual(subprocess.run(["sh", "-n", str(path)]).returncode, 0)
        return path

    def sh(self, path: Path, **env) -> subprocess.CompletedProcess:
        e = {"PATH": f"{self.bin}:/usr/bin:/bin", "HOME": str(self.tmp / "home"), **env}
        return subprocess.run(["sh", str(path)], env=e, capture_output=True, text=True, cwd=self.tmp)

    def write_check_tool(self):
        self.tool("gatedtrain", f'echo "gatedtrain $*" >> {self.log}\nexit ${{FAKE_CHECK_RC:-0}}\n')

    def test_a_first_run_builds_hardens_marks_and_checks(self):
        self.write_check_tool()
        res = self.sh(self.printed())
        self.assertEqual(res.returncode, 0, res.stderr)
        calls = self.log.read_text()
        self.assertIn("uv python install", calls)
        self.assertIn("--no-config", calls)
        self.assertIn("python -I -m venv --clear", calls)
        self.assertIn("pip --isolated install --no-cache-dir --require-hashes --no-deps --only-binary :all: -r", calls)
        self.assertIn("PIP_CONFIG_FILE=/dev/null", calls)
        self.assertIn(f"gatedtrain setup eval-tree --check --root {self.root}", calls)
        for dirpath, dirs, files in os.walk(self.root):                 # nothing writable by group or other (uv's `.lock`)
            for name in dirs + files:
                p = Path(dirpath, name)
                if not p.is_symlink():
                    self.assertEqual(p.stat().st_mode & 0o022, 0, p)
        self.assertEqual((self.root / evaltree.REQUIREMENTS_MARK).read_text().strip(), f"{hashlib.sha256(REQS).hexdigest()} python=3.12")
        self.assertIn(str(self.root / "eval-venv/bin/python"), res.stdout)        # the arguments for `setup keeper`
        self.assertEqual(os.stat(self.root).st_mode & 0o022, 0)

    def test_a_second_run_changes_nothing(self):
        self.write_check_tool()
        script = self.printed()
        self.assertEqual(self.sh(script).returncode, 0)
        self.log.write_text("")
        res = self.sh(script)
        self.assertEqual(res.returncode, 0, res.stderr)
        calls = self.log.read_text()
        self.assertNotIn("uv python install", calls)
        self.assertNotIn("venv", calls)
        self.assertNotIn("pip ", calls)
        self.assertIn("setup eval-tree --check", calls)                  # still verifies
        self.assertIn("already", res.stdout)

    def test_other_requirements_in_an_existing_tree_stop_the_script(self):
        self.write_check_tool()
        self.assertEqual(self.sh(self.printed()).returncode, 0)
        other = build(requirements=REQS + b"six==1.16.0 --hash=sha256:" + b"4" * 64 + b"\n", env={"HOME": "/h"},
                      home="/h", root=str(self.root), owner_uid=os.getuid(), executable=str(self.bin / "gatedtrain"))
        path = self.tmp / "other.sh"
        path.write_text(other["script"])
        before = sorted(str(p) for p in self.root.rglob("*"))
        res = self.sh(path)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("other requirements", res.stderr)
        self.assertEqual(sorted(str(p) for p in self.root.rglob("*")), before)          # nothing was changed or removed

    def test_another_python_version_in_an_existing_tree_stops_the_script(self):
        self.write_check_tool()
        self.assertEqual(self.sh(self.printed()).returncode, 0)
        before = sorted(str(p) for p in self.root.rglob("*"))
        res = self.sh(self.printed(python_version="3.13"))
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("another CPython version", res.stderr)
        self.assertEqual(sorted(str(p) for p in self.root.rglob("*")), before)

    def test_a_root_that_holds_other_files_is_not_adopted_or_changed(self):
        self.write_check_tool()
        self.root.mkdir(parents=True)
        (self.root / "mine.txt").write_text("x")
        self.root.chmod(0o775)
        res = self.sh(self.printed())
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("did not make", res.stderr)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["mine.txt"])
        self.assertEqual(stat.S_IMODE(self.root.stat().st_mode), 0o775)           # not chmod'ed
        self.assertNotIn("uv python install", self.log.read_text() if self.log.exists() else "")

    def test_an_existing_empty_root_is_used_but_its_mode_is_left_to_the_check(self):
        self.write_check_tool()
        self.root.mkdir(parents=True)
        self.root.chmod(0o775)
        res = self.sh(self.printed())
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(stat.S_IMODE(self.root.stat().st_mode), 0o775)           # only a root the script made is chmod'ed

    def test_a_symbolic_link_as_the_root_is_refused(self):
        self.write_check_tool()
        real = self.tmp / "real"
        real.mkdir()
        self.root.parent.mkdir(parents=True)
        os.symlink(real, self.root)
        res = self.sh(self.printed())
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("symbolic link", res.stderr)
        self.assertEqual(list(real.iterdir()), [])

    def test_the_requirements_file_is_removed_by_a_trap_and_the_script_names_the_cleanup(self):
        script = (self.printed()).read_text()
        self.assertIn("trap 'rm -f", script)
        self.assertIn("rm -rf", script.split("exists without")[1].split("\n")[0])

    def test_a_half_built_tree_stops_the_script_instead_of_clearing_it(self):
        (self.root / "eval-venv").mkdir(parents=True)
        res = self.sh(self.printed())
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("eval-venv", res.stderr)
        self.assertEqual(list((self.root / "eval-venv").iterdir()), [])

    def test_a_failed_install_leaves_no_mark(self):
        self.write_check_tool()
        self.tool("uv", 'echo "boom" >&2; exit 5\n')
        res = self.sh(self.printed())
        self.assertNotEqual(res.returncode, 0)
        self.assertFalse((self.root / evaltree.REQUIREMENTS_MARK).exists())

    def test_root_and_the_wrong_uid_and_a_missing_uv_are_refused(self):
        self.write_check_tool()
        script = self.printed()
        res = self.sh(script, FAKE_UID="0")
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("not as root", res.stderr)
        self.assertFalse(self.root.exists())
        res = self.sh(script, FAKE_UID="4321")
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("printed for uid", res.stderr)
        (self.bin / "uv").unlink()
        res = self.sh(script)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("uv", res.stderr)
        self.assertFalse(self.root.exists())

    def test_a_failing_final_check_fails_the_script(self):
        self.write_check_tool()
        res = self.sh(self.printed(), FAKE_CHECK_RC="3")
        self.assertEqual(res.returncode, 3)

    def test_a_root_not_owned_by_the_user_is_refused(self):
        if os.getuid() == 0:
            self.skipTest("root owns everything")
        self.root.parent.mkdir(parents=True)
        self.tool("stat", 'case "$*" in *"%u"*) echo 4321;; *) /usr/bin/stat "$@";; esac\n')
        res = self.sh(self.printed())
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("not owned by you", res.stderr)


class CheckTest(unittest.TestCase):                                  # AN-36: `setup eval-tree --check`
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "ev"
        (self.root / "python" / "cpython-3.12.1" / "bin").mkdir(parents=True)
        (self.root / "python" / "cpython-3.12.1" / "bin" / "python3.12").write_text("#!/bin/sh\n")
        (self.root / "eval-venv" / "bin").mkdir(parents=True)
        os.symlink("../../python/cpython-3.12.1/bin/python3.12", self.root / "eval-venv" / "bin" / "python")
        for p in self.root.rglob("*"):
            if not p.is_symlink():
                p.chmod(0o755 if p.is_dir() or p.name.startswith("python") else 0o644)
        self.root.chmod(0o755)

    def check(self, *extra):
        return run(["setup", "eval-tree", "--check", "--root", str(self.root), *extra])

    def test_a_clean_user_owned_tree_passes_with_the_keepers_rule(self):
        code, doc = self.check()
        self.assertEqual(code, 0, doc)
        self.assertEqual(doc["data"]["trees"], [str(self.root / "python"), str(self.root / "eval-venv")])
        self.assertEqual(doc["data"]["problems"], 0)
        self.assertEqual(doc["data"]["checked_as_uid"], os.geteuid())
        self.assertIn("separate-user keeper refuses", doc["data"]["applies_to"])
        with no_processes():
            _, out, _ = run_text(["setup", "eval-tree", "--check", "--root", str(self.root)])
        self.assertIn(f"as uid {os.geteuid()}", out)
        self.assertIn("applies to a same-user keeper", out)
        self.assertEqual(envcheck.eval_tree_problems([self.root / "python", self.root / "eval-venv"], os.geteuid()), [])

    def test_a_group_or_other_writable_node_is_refused_and_named(self):
        bad = self.root / "python" / ".lock"
        bad.write_text("")
        bad.chmod(0o666)                                             # what uv leaves behind
        code, doc = self.check()
        self.assertEqual(code, 3)
        self.assertTrue(any(".lock" in p["message"] for p in doc["problems"]), doc)
        self.assertEqual(doc["problems"][0]["rule"], evaltree.RULE)
        self.assertIn("chmod", doc["problems"][0]["fix"])
        # a problem outside the trees (an ancestor) gets no chmod hint for them
        os.chmod(self.root, 0o777)
        _, doc2 = self.check()
        outside = [p for p in doc2["problems"] if str(self.root) + " " in p["message"] or p["message"].startswith(str(self.root) + " is")]
        self.assertTrue(outside, doc2)
        self.assertTrue(all("chmod -R go-w" not in p["fix"] for p in outside))

    def test_a_link_leaving_the_trees_is_refused(self):
        os.symlink("/usr/bin/env", self.root / "eval-venv" / "bin" / "escape")
        code, doc = self.check()
        self.assertEqual(code, 3)
        self.assertTrue(any("resolves outside" in p["message"] for p in doc["problems"]), doc)

    def test_a_missing_tree_is_refused_with_the_command_that_builds_it(self):
        code, doc = run(["setup", "eval-tree", "--check", "--root", str(self.root / "none")])
        self.assertEqual(code, 3)
        self.assertIn("setup eval-tree", doc["problems"][0]["fix"])

    def test_the_problem_list_is_capped_but_counted(self):
        for i in range(evaltree.MAX_CHECK_PROBLEMS + 25):
            f = self.root / "python" / f"f{i}"
            f.write_text("")
            f.chmod(0o666)
        code, doc = self.check()
        self.assertEqual(code, 3)
        self.assertEqual(len([p for p in doc["problems"] if p["rule"] == evaltree.RULE and "writable" in p["message"]]),
                         evaltree.MAX_CHECK_PROBLEMS)
        self.assertTrue(any(str(evaltree.MAX_CHECK_PROBLEMS + 25) in p["message"] or "more" in p["message"]
                            for p in doc["problems"]), doc)

    def test_check_needs_neither_requirements_nor_version_and_runs_nothing(self):
        with no_processes():
            code, _ = self.check()
        self.assertEqual(code, 0)


class RuleOfTheOtherKeeperTest(unittest.TestCase):
    def test_a_user_owned_tree_is_refused_for_another_keeper_uid(self):
        """The fail-closed half: a separate-user keeper (another uid) refuses the tree this command builds."""
        with tempfile.TemporaryDirectory() as t:
            tree = Path(t) / "ev"
            tree.mkdir()
            tree.chmod(0o755)
            other = os.geteuid() + 1 if os.geteuid() else 1
            problems = envcheck.eval_tree_problems([tree], other)
            self.assertTrue(problems)
            self.assertIn("not the keeper", problems[0].message)


class SurfaceTest(unittest.TestCase):
    def test_layout_has_a_scope_and_a_key_for_the_root(self):
        self.assertIn(layout.EVAL, layout.SCOPES)
        keys = {k.dotted: k for k in layout.keys_for(layout.EVAL)}
        self.assertEqual(set(keys), {"eval_tree.root", "inference.eval_requirements"})
        self.assertEqual(keys["eval_tree.root"].flags[layout.EVAL], "--root")
        text = layout.render(layout.EVAL, env=SESSION, uid=1234, home="/home/alice")
        self.assertIn('root = "/home/alice/.local/share/gatedtrain-keeper-eval"', text)

    def test_the_command_is_listed_and_the_help_names_the_mode(self):
        flat = re.sub(r"\s+", " ", help_text("setup", "eval-tree", "--help")).replace("- ", "-")
        for word in ("--requirements", "--python-version", "--root", "--check", "--layout", "same-user"):
            self.assertIn(word, flat)

    def test_no_help_text_still_calls_a_keeper_eval_tree_root_owned(self):
        flat = re.sub(r"\s+", " ", help_text("setup", "keeper", "--help"))
        self.assertNotIn("a root-owned eval tree", flat)
        self.assertIn("owned by the keeper's uid or root", flat)


if __name__ == "__main__":
    unittest.main()
