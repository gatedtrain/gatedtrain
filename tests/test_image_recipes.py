"""C3: the build recipes shipped in containers/: the two derivatives change the same ten files, the Containerfiles name no
image ID, and `nemo-rl-tierb/build.sh` is read-only until it builds, has a dry run and never writes the owner config."""

from __future__ import annotations

import contextlib
import io
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
CONTAINERS = ROOT / "containers"
if not CONTAINERS.is_dir():            # an sdist or wheel install has no containers/ beside tests/
    raise unittest.SkipTest("not a source checkout (no containers/ directory beside tests/): an sdist or wheel install")
UID1000 = CONTAINERS / "nemo-rl-uid1000" / "Containerfile"
ROOTLESS = CONTAINERS / "nemo-rl-rootless" / "Containerfile"
BUILD = CONTAINERS / "nemo-rl-tierb" / "build.sh"
FILES_RE = re.compile(r"for f in ((?:[\w/.]+(?: \\\n\s*)?)+); do")


def directives(path: Path) -> dict:
    text = path.read_text()
    code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    m = FILES_RE.search(code)
    files = re.findall(r"[\w/]+\.py", m.group(1))
    return {"args": re.findall(r"^ARG (\w+)", code, re.M), "from": re.findall(r"^FROM (\S+)", code, re.M),
            "files": files, "venv": re.search(r"V=(\S+?);", code).group(1),
            "chown": re.search(r'chown (\S+) "\$V/\$f"', code).group(1)}


class TestDerivatives(unittest.TestCase):
    def test_same_ten_files_same_venv_same_base_argument(self):
        a, b = directives(UID1000), directives(ROOTLESS)
        self.assertEqual(len(a["files"]), 10)
        self.assertEqual(a["files"], b["files"])
        self.assertEqual(a["venv"], b["venv"])
        self.assertEqual(a["args"], b["args"])
        self.assertEqual(a["from"], b["from"])
        self.assertEqual(a["args"], ["BASE"])
        self.assertEqual(a["from"], ["${BASE}"])

    def test_only_the_owner_differs(self):
        self.assertEqual(directives(UID1000)["chown"], "1000:1000")
        self.assertEqual(directives(ROOTLESS)["chown"], "0:0")

    def test_no_image_id_or_digest_in_the_recipes(self):
        for p in CONTAINERS.rglob("*"):
            if p.is_file():
                self.assertIsNone(re.search(r"sha256:[0-9a-f]{64}", p.read_text()), p)

    def test_readmes_exist(self):
        for d in ("nemo-rl-tierb", "nemo-rl-rootless", "nemo-rl-uid1000"):
            self.assertTrue((CONTAINERS / d / "README.md").exists(), d)

    def test_lock_makes_no_hash_claim_about_the_shipped_uid1000_containerfile(self):
        """The lock's build predates the ARG-BASE file: it must not carry the shipped file's hash as if it built it."""
        import hashlib
        from gatedtrain import imagelock
        acc = imagelock.resolve("nemo-rl/tierb-uid1000", config=None)[0]
        self.assertIsNone(acc.recipe["containerfile_sha256"])
        shipped = "sha256:" + hashlib.sha256(UID1000.read_bytes()).hexdigest()
        self.assertNotIn(shipped, (Path(imagelock.__file__).parent / "data" / "image-lock.json").read_text())


class TestBuildScript(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        self.log = self.tmp / "docker.log"
        stub = self.bin / "docker"
        stub.write_text(f'#!/bin/bash\necho "$@" >> {self.log}\n'
                        'case "$*" in\n'
                        '  *"context inspect"*) exit 0;;\n'
                        f'  *"image inspect"*) {{ [ -n "$STUB_HAS_IMAGE" ] || grep -q buildx {self.log}; }} && echo "sha256:$(printf %064d 7)" && exit 0; exit 1;;\n'
                        '  *buildx*) echo built;;\n'
                        'esac\nexit 0\n')
        stub.chmod(0o755)
        self.src = self.tmp / "checkout"
        (self.src / "docker").mkdir(parents=True)
        (self.src / "docker" / "Dockerfile").write_text("FROM scratch\n")
        for cmd in (["git", "init", "-q"], ["git", "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A"],
                    ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x"]):
            subprocess.run(cmd, cwd=self.src, check=True, capture_output=True)
        self.commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.src, capture_output=True, text=True,
                                     check=True).stdout.strip()
        self.cfg = self.tmp / "xdg"

    def run_build(self, *args, has_image=False):
        env = {"PATH": f"{self.bin}:{os.environ['PATH']}", "XDG_CONFIG_HOME": str(self.cfg), "HOME": str(self.tmp)}
        if has_image:
            env["STUB_HAS_IMAGE"] = "1"
        return subprocess.run(["bash", str(BUILD), *args], capture_output=True, text=True, env=env)

    def good(self, *extra):
        return ["--context", "ctx", "--source-dir", str(self.src), "--commit", self.commit, "--tag", "t:1",
                "--max-jobs", "2", *extra]

    def test_syntax(self):
        self.assertEqual(subprocess.run(["bash", "-n", str(BUILD)]).returncode, 0)
        self.assertIn("set -euo pipefail", BUILD.read_text())

    def test_help_and_missing_arguments(self):
        self.assertEqual(self.run_build("--help").returncode, 0)
        r = self.run_build("--context", "ctx")
        self.assertEqual(r.returncode, 2)
        self.assertIn("--commit", r.stderr)
        self.assertFalse(self.log.exists())

    def test_dry_run_prints_the_plan_and_builds_nothing(self):
        r = self.run_build(*self.good("--dry-run"))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("dry run", r.stdout)
        self.assertIn("buildx build", r.stdout)
        self.assertIn(f"NEMO_RL_COMMIT={self.commit}", r.stdout)
        self.assertNotIn("buildx", self.log.read_text() if self.log.exists() else "")

    def test_commit_mismatch_refused_before_any_build(self):
        r = self.run_build("--context", "ctx", "--source-dir", str(self.src), "--commit", "0" * 40, "--tag", "t:1",
                           "--max-jobs", "2")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("HEAD", r.stderr)
        self.assertNotIn("buildx", self.log.read_text() if self.log.exists() else "")

    def test_build_then_prints_the_add_command_and_writes_no_config(self):
        r = self.run_build(*self.good())
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("buildx", self.log.read_text())
        self.assertIn("gatedtrain image add nemo-rl/tierb --id sha256:", r.stdout)
        self.assertIn(f"--recipe-commit {self.commit}", r.stdout)
        self.assertIn("--containerfile-sha256 sha256:", r.stdout)
        self.assertFalse(self.cfg.exists())
        self.assertFalse((self.tmp / ".config").exists())

    def test_idempotent_existing_tag_is_not_rebuilt(self):
        r = self.run_build(*self.good(), has_image=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("image tag exists; not rebuilt", r.stdout)
        self.assertNotIn("built image ID", r.stdout)                  # nothing was built: do not say so
        self.assertNotIn("buildx", self.log.read_text())
        self.assertIn("gatedtrain image add nemo-rl/tierb --id sha256:", r.stdout)

    def test_existing_tag_under_dry_run_says_both(self):
        r = self.run_build(*self.good("--dry-run"), has_image=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("dry run", r.stdout)
        self.assertIn("image tag exists; not rebuilt", r.stdout)
        self.assertIn("gatedtrain image add nemo-rl/tierb --id sha256:", r.stdout)
        self.assertNotIn("buildx", self.log.read_text() if self.log.exists() else "")

    def test_a_fresh_build_says_built(self):
        r = self.run_build(*self.good())
        self.assertIn("built image ID: sha256:", r.stdout)
        self.assertNotIn("not rebuilt", r.stdout)

    def test_the_printed_add_command_quotes_every_value(self):
        import shlex
        name = "my recipe; touch /tmp/x $(id) 'q'"
        r = self.run_build(*self.good("--recipe-name", name))
        self.assertEqual(r.returncode, 0, r.stderr)
        line = next(l for l in r.stdout.splitlines() if l.strip().startswith("gatedtrain image add"))
        argv = shlex.split(line)
        self.assertEqual(argv[argv.index("--recipe") + 1], name)

    def test_the_modified_note_is_quoted_and_parses(self):
        import shlex
        (self.src / "new.txt").write_text("x")
        r = self.run_build(*self.good("--allow-modified"))
        line = next(l for l in r.stdout.splitlines() if l.strip().startswith("gatedtrain image add"))
        argv = shlex.split(line)
        self.assertEqual(argv[argv.index("--note") + 1], "built from a locally modified checkout")

    def test_an_option_without_a_value_is_a_clear_error(self):
        for opt in ("--context", "--source-dir", "--commit", "--tag", "--max-jobs", "--recipe-name"):
            r = self.run_build("--context", "ctx", opt) if opt != "--context" else self.run_build(opt)
            self.assertEqual(r.returncode, 2, (opt, r.stderr))
            self.assertIn(opt, r.stderr)
            self.assertIn("needs a value", r.stderr)

    def test_ignored_files_in_the_build_context_are_refused(self):
        (self.src / ".gitignore").write_text("*.cache\n")
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A"], cwd=self.src, check=True)
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "ign"], cwd=self.src,
                       check=True, capture_output=True)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.src, capture_output=True, text=True,
                                check=True).stdout.strip()
        (self.src / "secret.cache").write_text("x")
        args = ["--context", "ctx", "--source-dir", str(self.src), "--commit", commit, "--tag", "t:1", "--max-jobs", "2"]
        r = self.run_build(*args)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("local changes", r.stderr)
        self.assertNotIn("buildx", self.log.read_text() if self.log.exists() else "")
        self.assertEqual(self.run_build(*args, "--allow-modified", "--dry-run").returncode, 0)

    def test_untracked_files_are_still_refused(self):
        (self.src / "stray.txt").write_text("x")
        self.assertNotEqual(self.run_build(*self.good()).returncode, 0)

    def test_the_readme_states_that_the_build_is_not_reproducible(self):
        text = (CONTAINERS / "nemo-rl-tierb" / "README.md").read_text()
        self.assertIn("by tag", text)
        self.assertIn("not bit-reproducible", text)
        self.assertIn("host and time specific", text)
        self.assertIn("docker/Dockerfile", text)

    def test_a_tag_or_context_that_looks_like_an_option_is_refused(self):
        for opt, value in (("--tag", "-v"), ("--tag", "--rm"), ("--context", "-H=tcp://x"), ("--tag", "bad tag"),
                           ("--tag", "")):
            args = self.good()
            args[args.index(opt) + 1] = value
            r = self.run_build(*args)
            self.assertIn(r.returncode, (1, 2), (opt, value))
            self.assertNotEqual(r.returncode, 0)
            self.assertNotIn("buildx", self.log.read_text() if self.log.exists() else "")

    def test_max_jobs_must_be_at_least_one(self):
        for bad in ("0", "00", "08", "-1", "x", "1.5"):
            args = self.good()
            args[args.index("--max-jobs") + 1] = bad
            self.assertEqual(self.run_build(*args).returncode, 2, bad)

    def test_docker_gets_the_tag_after_a_separator(self):
        self.run_build(*self.good())
        inspect = [l for l in self.log.read_text().splitlines() if "image inspect" in l]
        self.assertTrue(inspect)
        self.assertTrue(all(" -- t:1" in l for l in inspect), inspect)

    def test_the_command_for_an_existing_tag_says_it_was_not_built_by_this_run(self):
        import shlex
        r = self.run_build(*self.good(), has_image=True)
        argv = shlex.split(next(l for l in r.stdout.splitlines() if l.strip().startswith("gatedtrain image add")))
        self.assertEqual(argv[argv.index("--note") + 1], "tag existed; not built by this run")
        # Nothing in this run vouches for the existing image, so the command claims no provenance: `image add` then
        # refuses until the owner supplies the true recipe commit or Containerfile hash.
        self.assertNotIn("--recipe-commit", argv)
        self.assertNotIn("--containerfile-sha256", argv)
        self.assertIn("supply", r.stdout)
        (self.src / "new.txt").write_text("x")
        r = self.run_build(*self.good("--allow-modified"), has_image=True)
        argv = shlex.split(next(l for l in r.stdout.splitlines() if l.strip().startswith("gatedtrain image add")))
        note = argv[argv.index("--note") + 1]
        self.assertIn("tag existed; not built by this run", note)
        self.assertIn("built from a locally modified checkout", note)
        (self.src / "new.txt").unlink()
        r = self.run_build(*self.good())                            # a fresh build carries neither
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("--note", r.stdout)
        self.assertIn("--recipe-commit", r.stdout)

    def test_the_command_for_an_existing_tag_is_refused_by_image_add_until_provenance_is_supplied(self):
        import shlex
        from gatedtrain import cli
        r = self.run_build(*self.good(), has_image=True)
        argv = shlex.split(next(l for l in r.stdout.splitlines() if l.strip().startswith("gatedtrain image add")))[1:]
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": tmp}):
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                code = cli.main(["--json", *argv, "--yes"])
            self.assertEqual(code, 3, out.getvalue())
            self.assertIn("recipe commit or a Containerfile hash", out.getvalue())
            self.assertEqual(list(Path(tmp).rglob("images.json")), [])   # nothing was recorded

    def test_the_readmes_say_the_recipes_are_not_in_the_wheel(self):
        for d in ("nemo-rl-tierb", "nemo-rl-rootless", "nemo-rl-uid1000"):
            text = " ".join((CONTAINERS / d / "README.md").read_text().split())
            self.assertIn("not shipped in the wheel or sdist", text, d)

    def test_the_rootless_readme_says_which_image_the_driver_declared(self):
        text = " ".join((CONTAINERS / "nemo-rl-rootless" / "README.md").read_text().split())
        self.assertIn("a result says which image the driver declared and that it is not in the maintainer lock", text)
        self.assertNotIn("says which image ran", text)

    def test_script_has_no_machine_facts(self):
        text = BUILD.read_text()
        self.assertIsNone(re.search(r"https?://|sha256:[0-9a-f]{64}|/home/|/Users/|/opt/|/srv/", text))
        self.assertNotIn("images.json", text)


if __name__ == "__main__":
    unittest.main()
