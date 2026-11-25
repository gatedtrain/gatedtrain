"""IMG-5, IMG-6: `gatedtrain image list|check|add`."""

from __future__ import annotations

import contextlib
import io
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gatedtrain import cli, cli_image, imagelock

NEW = "sha256:" + "e" * 64
COMMIT = "2" * 40


def run(argv, *, tty=False, answer="yes"):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), \
            mock.patch.object(cli_image, "_stdin_is_tty", lambda: tty), \
            mock.patch.object(cli_image, "_ask", lambda prompt: answer):
        try:
            code = cli.main(argv)
        except SystemExit as exc:                      # argparse usage error in text mode
            code = exc.code
    return code, out.getvalue(), err.getvalue()


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cfg = Path(self._tmp.name) / "gt" / "images.json"
        p = mock.patch.object(imagelock, "user_config_path", lambda: self.cfg)
        p.start()
        self.addCleanup(p.stop)
        self.calls = []

        def runner(argv, timeout):
            self.calls.append((argv, timeout))
            return self.rc, self.stdout
        self.rc, self.stdout = 0, NEW + "\n"
        q = mock.patch.object(cli_image, "RUNNER", runner)
        q.start()
        self.addCleanup(q.stop)

    def add_args(self, *extra):
        return ["image", "add", "nemo-rl/tierb", "--id", NEW, "--recipe", "my-recipe", "--recipe-commit", COMMIT, *extra]


class TestList(Base):
    def test_list_human_and_json(self):
        code, out, _ = run(["image", "list"])
        self.assertEqual(code, 0)
        for logical in ("nemo-rl/tierb", "nemo-rl/tierb-uid1000"):
            self.assertIn(logical, out)
        self.assertIn("lock", out)
        code, out, _ = run(["--json", "image", "list"])
        doc = json.loads(out)
        self.assertEqual(doc["command"], "image list")
        row = doc["data"]["images"][0]
        self.assertTrue({"logical", "id", "origin", "recipe", "source"} <= set(row))

    def test_list_prints_the_sha256_of_the_lock_it_read_as_information(self):
        import hashlib
        want = "sha256:" + hashlib.sha256(imagelock.LOCK_PATH.read_bytes()).hexdigest()
        code, out, _ = run(["image", "list"])
        line = next(l for l in out.splitlines() if l.startswith("lock: "))
        self.assertIn(want, line)
        self.assertIn("informational", line)                            # no release hashes are published yet
        _, out, _ = run(["--json", "image", "list"])
        self.assertEqual(json.loads(out)["data"]["lock_sha256"], want)

    def test_list_shows_config_additions(self):
        run(self.add_args("--yes"))
        _, out, _ = run(["--json", "image", "list"])
        rows = json.loads(out)["data"]["images"]
        mine = [r for r in rows if r["id"] == NEW]
        self.assertEqual([(r["source"], r["origin"]) for r in mine], [("owner_config", "user-built")])

    def test_list_refuses_a_bad_config(self):
        self.cfg.parent.mkdir(parents=True)
        self.cfg.write_text("{}")
        self.cfg.chmod(0o666)
        code, _, err = run(["image", "list"])
        self.assertEqual(code, 3)
        self.assertIn("IMG-3", err)


class TestCheck(Base):
    def test_check_by_id_accepted(self):
        known = imagelock.maintainer_ids("nemo-rl/tierb")[0]
        code, out, _ = run(["image", "check", "nemo-rl/tierb", "--id", known])
        self.assertEqual(code, 0)
        self.assertEqual(self.calls, [])                      # no docker call for --id

    def test_check_by_ref_runs_docker_inspect_with_context(self):
        self.stdout = imagelock.maintainer_ids("nemo-rl/tierb")[0] + "\n"
        code, _, _ = run(["image", "check", "nemo-rl/tierb", "--ref", "some:tag", "--docker-context", "ctx"])
        self.assertEqual(code, 0)
        argv, timeout = self.calls[0]
        self.assertEqual(argv[:3], ["docker", "--context", "ctx"])
        self.assertIn("some:tag", argv)
        self.assertIn("{{.Id}}", argv)
        self.assertGreater(timeout, 0)

    def test_not_accepted_prints_the_add_command(self):
        code, out, _ = run(["image", "check", "nemo-rl/tierb", "--id", NEW])
        self.assertEqual(code, 1)
        self.assertIn(f"gatedtrain image add nemo-rl/tierb --id {NEW}", out)
        self.assertIn("containers/", out)

    def test_accepted_after_add(self):
        run(self.add_args("--yes"))
        code, _, _ = run(["image", "check", "nemo-rl/tierb", "--id", NEW])
        self.assertEqual(code, 0)

    def test_usage_needs_exactly_one_of_ref_id(self):
        self.assertEqual(run(["image", "check", "nemo-rl/tierb"])[0], 2)
        self.assertEqual(run(["image", "check", "nemo-rl/tierb", "--id", NEW, "--ref", "x"])[0], 2)

    def test_bad_id_and_unknown_logical_refused(self):
        self.assertEqual(run(["image", "check", "nemo-rl/tierb", "--id", "nope"])[0], 3)
        code, _, err = run(["image", "check", "nope/none", "--id", NEW])
        self.assertEqual(code, 3)
        self.assertIn("IMG-4", err)

    def test_docker_failure_is_refused(self):
        self.rc, self.stdout = 1, ""
        code, _, err = run(["image", "check", "nemo-rl/tierb", "--ref", "x"])
        self.assertEqual(code, 3)
        self.assertIn("IMG-5", err)

    def test_docker_output_that_is_not_an_id_is_refused(self):
        self.stdout = "garbage\n"
        self.assertEqual(run(["image", "check", "nemo-rl/tierb", "--ref", "x"])[0], 3)

    def test_json(self):
        code, out, _ = run(["--json", "image", "check", "nemo-rl/tierb", "--id", NEW])
        doc = json.loads(out)
        self.assertEqual((code, doc["ok"], doc["data"]["accepted"]), (1, False, False))


class TestAdd(Base):
    def test_requires_yes_without_a_tty(self):
        code, _, err = run(self.add_args())
        self.assertEqual(code, 3)
        self.assertIn("--yes", err)
        self.assertFalse(self.cfg.exists())

    def test_tty_prompt_yes_writes(self):
        code, out, _ = run(self.add_args(), tty=True, answer="yes")
        self.assertEqual(code, 0)
        self.assertTrue(self.cfg.exists())
        self.assertIn("user-added", out)
        self.assertIn("not in the maintainer lock", out)

    def test_the_output_says_plainly_that_it_changes_nothing_for_gating(self):
        _, out, _ = run(self.add_args("--yes"))
        self.assertIn("recorded for provenance; packs adopt accepted_images(logical) in a later release", out)
        _, out, _ = run(["--json", "image", "add", "nemo-rl/tierb", "--id", "sha256:" + "d" * 64,
                         "--recipe", "r", "--recipe-commit", COMMIT, "--yes"])
        self.assertIn("provenance", json.loads(out)["data"]["gating_note"])

    def test_text_that_could_forge_a_report_line_is_refused(self):
        base = ["image", "add", "nemo-rl/tierb", "--id", NEW, "--recipe-commit", COMMIT, "--yes"]
        for bad in ("x\n## Verdict", "r\n- keeper mode: separate-user", "a\x1b[2Jb"):
            for extra in (["--recipe", "r", "--note", bad], ["--recipe", bad]):
                code, _, err = run([*base, *extra])
                self.assertEqual(code, 3, extra)
                self.assertIn("IMG-6", err)
        self.assertFalse(self.cfg.exists())

    def test_tty_prompt_no_writes_nothing(self):
        code, _, _ = run(self.add_args(), tty=True, answer="no")
        self.assertEqual(code, 1)
        self.assertFalse(self.cfg.exists())

    def test_file_mode_dir_mode_and_content(self):
        code, _, _ = run(self.add_args("--yes", "--containerfile-sha256", "sha256:" + "f" * 64, "--note", "mine",
                                       "--built-utc", "2026-01-02"))
        self.assertEqual(code, 0)
        self.assertEqual(stat.S_IMODE(self.cfg.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.cfg.parent.stat().st_mode), 0o700)
        doc = json.loads(self.cfg.read_text())
        e = doc["images"]["nemo-rl/tierb"]["accepted"][0]
        self.assertEqual((e["id"], e["origin"], e["note"], e["built_utc"]), (NEW, "user-built", "mine", "2026-01-02"))
        self.assertEqual(e["recipe"]["commit"], COMMIT)
        self.assertEqual(sorted(os.listdir(self.cfg.parent)), ["images.json", "images.json" + imagelock.LOCK_SUFFIX])  # no temp
        imagelock.load_user_config(self.cfg)                                     # and it validates

    def test_second_add_appends_and_repeat_is_idempotent(self):
        run(self.add_args("--yes"))
        other = "sha256:" + "d" * 64
        code, _, _ = run(["image", "add", "nemo-rl/tierb", "--id", other, "--recipe", "r2", "--recipe-commit", COMMIT, "--yes"])
        self.assertEqual(code, 0)
        self.assertEqual(run(self.add_args("--yes"))[0], 0)
        ids = [e["id"] for e in json.loads(self.cfg.read_text())["images"]["nemo-rl/tierb"]["accepted"]]
        self.assertEqual(ids, [NEW, other])

    def test_adding_a_lock_id_is_a_noop(self):
        known = imagelock.maintainer_ids("nemo-rl/tierb")[0]
        code, out, _ = run(["image", "add", "nemo-rl/tierb", "--id", known, "--recipe", "r", "--recipe-commit", COMMIT, "--yes"])
        self.assertEqual(code, 0)
        self.assertFalse(self.cfg.exists())
        self.assertIn("already in the maintainer lock", out)

    def test_unknown_logical_bad_id_or_missing_recipe_info_refused(self):
        self.assertEqual(run(["image", "add", "nope/x", "--id", NEW, "--recipe", "r", "--recipe-commit", COMMIT, "--yes"])[0], 3)
        self.assertEqual(run(["image", "add", "nemo-rl/tierb", "--id", "bad", "--recipe", "r", "--recipe-commit", COMMIT, "--yes"])[0], 3)
        self.assertEqual(run(["image", "add", "nemo-rl/tierb", "--id", NEW, "--recipe", "r", "--yes"])[0], 3)
        self.assertFalse(self.cfg.exists())

    def test_refuses_to_extend_an_untrustworthy_config(self):
        self.cfg.parent.mkdir(parents=True)
        real = self.cfg.parent / "real.json"
        real.write_text(json.dumps(imagelock.empty_config()))
        real.chmod(0o600)
        self.cfg.symlink_to(real)
        code, _, err = run(self.add_args("--yes"))
        self.assertEqual(code, 3)
        self.assertIn("IMG-3", err)

    def test_no_remove_command(self):
        self.assertEqual(run(["image", "remove", "nemo-rl/tierb"])[0], 2)

    def test_the_environment_supplies_no_accepted_id(self):
        with mock.patch.dict(os.environ, {"GATEDTRAIN_IMAGES": "/tmp/x", "XDG_CONFIG_HOME": "/nonexistent"}):
            code, _, _ = run(self.add_args("--yes"))      # user_config_path is patched by the test; env is ignored
        self.assertEqual(code, 0)
        self.assertTrue(self.cfg.exists())

    def test_json_envelope(self):
        code, out, _ = run(["--json", "image", "add", *self.add_args("--yes")[2:]])
        doc = json.loads(out)
        self.assertEqual((code, doc["command"], doc["data"]["user_added"]), (0, "image add", True))


class TestReviewRound2(Base):
    """L8 (messages), L11: option-like values, dates, hostile files, concurrent adds."""

    def test_a_ref_or_context_that_looks_like_an_option_is_refused_and_docker_gets_a_separator(self):
        for argv in (["--ref=-v"], ["--ref", "x", "--docker-context=--host=tcp://evil"]):
            code, _, err = run(["image", "check", "nemo-rl/tierb", *argv])
            self.assertEqual(code, 3, argv)
            self.assertIn("IMG-5", err)
        self.assertEqual(self.calls, [])
        self.stdout = imagelock.maintainer_ids("nemo-rl/tierb")[0] + "\n"
        run(["image", "check", "nemo-rl/tierb", "--ref", "some:tag"])
        argv = self.calls[0][0]
        self.assertEqual(argv[argv.index("--") + 1:], ["some:tag"])

    def test_built_utc_must_be_a_real_date(self):
        code, _, err = run(self.add_args("--yes", "--built-utc", "2026-02-30"))
        self.assertEqual(code, 3)
        self.assertIn("IMG-6", err)
        self.assertFalse(self.cfg.exists())

    def test_a_deeply_nested_config_is_refused_not_a_traceback(self):
        self.cfg.parent.mkdir(parents=True)
        self.cfg.write_text("[" * 100_000 + "]" * 100_000)
        self.cfg.chmod(0o600)
        for argv in (["image", "list"], self.add_args("--yes")):
            code, _, err = run(argv)
            self.assertEqual(code, 3)
            self.assertIn("IMG-2" if argv[1] == "list" else "IMG-6", err)

    def test_a_hostile_logical_name_cannot_inject_terminal_text(self):
        code, _, err = run(["image", "check", "x\x1b[31m\nforged", "--id", NEW])
        self.assertEqual(code, 3)
        self.assertNotIn("\x1b", err)
        self.assertEqual(len(err.strip().splitlines()), 1)

    def test_the_not_accepted_message_says_the_recipes_are_not_in_the_wheel(self):
        _, out, _ = run(["image", "check", "nemo-rl/tierb", "--id", NEW])
        self.assertIn("not shipped in the wheel or sdist", out)
        self.assertIn("source checkout", out)

    def test_two_adds_in_a_row_keep_both_entries(self):
        other = "sha256:" + "d" * 64
        run(self.add_args("--yes"))
        run(["image", "add", "nemo-rl/tierb", "--id", other, "--recipe", "r", "--recipe-commit", COMMIT, "--yes"])
        ids = [e["id"] for e in json.loads(self.cfg.read_text())["images"]["nemo-rl/tierb"]["accepted"]]
        self.assertEqual(ids, [NEW, other])


if __name__ == "__main__":
    unittest.main()
