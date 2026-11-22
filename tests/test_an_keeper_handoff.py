"""AN-13: model hand-off: the keeper copies the files a model reference lists into its own state, hashes the copy and
runs only the copy (design 05 §18.9).

`KeeperState` runs in process with a recording generator (the synthetic pack's, wrapped): it notes the paths and bytes
of the model files it receives, and can run a hook at its first candidate call, which is how a file is swapped after
`evaluate` started scoring (the in-process form of the `block` switch of `tests/fixtures/ainative/run_cli.py`).
"""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from _keeper import skip_unless_linux

from test_an_keeper_attempts import BASE, TOY, arm, model, registration
from test_an_keeper_intake import confirm_args, project_id
from gatedtrain.ainative import trust_stages
from gatedtrain.ainative.keeper import envcheck, state as state_mod, texts
from gatedtrain.ainative.keeper.protocol import RefusedAfterSpend
from gatedtrain.ainative.keeper.state import KeeperState
from gatedtrain.canonical import sha256_bytes
from gatedtrain.errors import Refused
from gatedtrain.testing import synthetic_pack as pack

WEIGHTS = b"\x08\x00\x00\x00\x00\x00\x00\x00{}      "
TEMPLATE = b"{% for m in messages %}{{ m.content }}{% endfor %}"
OTHER_UID = 65534 if os.getuid() != 65534 else 65533
_REAL_OPEN = os.open
TRAVERSE_FIX = ("put model files under a directory the keeper user can traverse, such as the output root "
                "(mode 0755)")


def _tmpdir(case: unittest.TestCase) -> Path:
    d = Path(tempfile.mkdtemp(prefix="gk-handoff-"))
    case.addCleanup(shutil.rmtree, d, True)
    return d


class _Recorder:
    """The synthetic pack's generator, noting what model files a candidate call received."""

    def __init__(self) -> None:
        self.paths: set[str] = set()
        self.contents: set[bytes] = set()
        self.on_first_candidate = None
        self.raise_on_candidate = False

    def __call__(self, model_ref: dict, item: dict, **kw):
        if model_ref.get("label") != "base":
            for f in model_ref.get("files", []):
                self.paths.add(f["path"])
                self.contents.add(Path(f["path"]).read_bytes())
            hook, self.on_first_candidate = self.on_first_candidate, None
            if hook is not None:
                hook(model_ref)
            if self.raise_on_candidate:
                raise ValueError("scoring failed")
        return pack.generate(model_ref, item, **kw)


class _ScenarioEP:
    name = "synthetic"
    value = "gatedtrain.testing.synthetic_pack:SCENARIO"

    def __init__(self, recorder: _Recorder) -> None:
        self.recorder = recorder

    def load(self):
        return pack.Scenario(pack.SCENARIO.name, {pack.GENERATOR: self.recorder}, pack.SCENARIO.pipeline,
                             pack.SCENARIO.scorers)


_ORIGINAL_EPS = trust_stages._scenario_entry_points


@skip_unless_linux
class CopyModelFilesTest(unittest.TestCase):
    """`envcheck.copy_model_files` and `copied_record` on their own."""

    def setUp(self):
        self.tmp = _tmpdir(self)
        self.src = self.tmp / "out"
        self.src.mkdir(mode=0o755)

    def _file(self, name: str, data: bytes, where: Path | None = None) -> dict:
        p = (where or self.src) / name
        p.write_bytes(data)
        return {"path": str(p), "sha256": sha256_bytes(data)}

    def test_suffixes(self):
        self.assertEqual(envcheck.MODEL_FILE_SUFFIXES, (".safetensors", ".json", ".jsonl", ".txt", ".jinja"))

    def test_copy_and_record(self):
        entries = [self._file("model.safetensors", WEIGHTS), self._file("chat_template.jinja", TEMPLATE)]
        record = {"label": "candidate", "files": entries}
        dest = self.tmp / "copy"
        out = envcheck.copy_model_files(record, dest)
        self.assertEqual(out, [{"name": "model.safetensors", "sha256": sha256_bytes(WEIGHTS), "size": len(WEIGHTS)},
                               {"name": "chat_template.jinja", "sha256": sha256_bytes(TEMPLATE),
                                "size": len(TEMPLATE)}])
        self.assertEqual(stat.S_IMODE(os.lstat(dest).st_mode), 0o700)
        for name, data in (("model.safetensors", WEIGHTS), ("chat_template.jinja", TEMPLATE)):
            self.assertEqual((dest / name).read_bytes(), data)
            self.assertEqual(stat.S_IMODE(os.lstat(dest / name).st_mode), 0o600)
        copied = envcheck.copied_record(record, dest)
        self.assertEqual(copied["label"], "candidate")
        self.assertEqual(copied["files"], [{"path": str(dest / "model.safetensors"), "sha256": entries[0]["sha256"]},
                                           {"path": str(dest / "chat_template.jinja"),
                                            "sha256": entries[1]["sha256"]}])
        self.assertEqual(record["files"], entries)                # the original record is unchanged
        self.assertEqual(envcheck.copied_record({"label": "base"}, dest), {"label": "base"})

    def test_dest_must_not_exist(self):
        dest = self.tmp / "copy"
        dest.mkdir()
        with self.assertRaises(FileExistsError):
            envcheck.copy_model_files({"files": [self._file("model.safetensors", WEIGHTS)]}, dest)

    def test_refusals_name_every_entry_and_leave_nothing(self):
        other = self.tmp / "other"
        other.mkdir()
        good = self._file("model.safetensors", WEIGHTS)
        bad_hash = {**self._file("config.json", b"{}"), "sha256": sha256_bytes(b"other")}
        dup = self._file("model.safetensors", WEIGHTS, other)
        dest = self.tmp / "copy"
        with self.assertRaises(Refused) as cm:
            envcheck.copy_model_files({"files": [good, bad_hash, dup]}, dest)
        problems = cm.exception.problems
        self.assertEqual([p.field for p in problems], ["files[1]", "files[2]"])
        self.assertTrue(all(p.rule == "AN-13" for p in problems))
        self.assertIn("sha256", problems[0].message)
        self.assertIn("base name", problems[1].message)
        self.assertFalse(os.path.lexists(dest))

    def _open_hook(self, target: str, before=None, after=None):
        """Patch `os.open` so opening `target` runs `before()` first and `after()` once it is open."""
        def fake(path, flags, *args, **kwargs):
            hit = os.fspath(path) == target
            if hit and before is not None:
                before()
            fd = _REAL_OPEN(path, flags, *args, **kwargs)
            if hit and after is not None:
                after()
            return fd
        return mock.patch.object(os, "open", fake)

    def _refused_changed(self, entry: dict) -> None:
        dest = self.tmp / "copy"
        with self.assertRaises(Refused) as cm:
            envcheck.copy_model_files({"files": [entry]}, dest)
        p = cm.exception.problems[0]
        self.assertEqual((p.rule, p.field), ("AN-13", "files[0]"))
        self.assertIn("changed between the check and the copy", p.message)
        self.assertFalse(os.path.lexists(dest))

    def test_other_file_at_open_refused(self):                  # the dev/ino check
        entry = self._file("model.safetensors", WEIGHTS)
        other = self.src / "other.safetensors"
        other.write_bytes(WEIGHTS)
        with self._open_hook(entry["path"], before=lambda: os.replace(other, entry["path"])):
            self._refused_changed(entry)

    def test_growth_after_check_refused(self):                  # the copy is bounded by the checked size
        entry = self._file("model.safetensors", WEIGHTS)

        def grow():
            with open(entry["path"], "ab") as f:
                f.write(b"more")
        with self._open_hook(entry["path"], after=grow):
            self._refused_changed(entry)

    def test_opened_ok_sees_the_opened_file(self):
        entry = self._file("model.safetensors", WEIGHTS)
        seen = []

        def opened_ok(fd: int, st: os.stat_result, name: str):
            seen.append((os.readlink(f"/proc/self/fd/{fd}"), st.st_size, name))
            return "refused by the caller", "the caller's fix"
        with self.assertRaises(Refused) as cm:
            envcheck.copy_model_files({"files": [entry]}, self.tmp / "copy", opened_ok=opened_ok)
        self.assertEqual(seen, [(entry["path"], len(WEIGHTS), "model.safetensors")])
        p = cm.exception.problems[0]
        self.assertEqual((p.rule, p.field, p.message, p.fix),
                         ("AN-13", "files[0]", "refused by the caller", "the caller's fix"))

    @unittest.skipIf(os.geteuid() == 0, "root traverses any directory")
    def test_unreachable_file_names_the_traverse_fix(self):
        entry = self._file("model.safetensors", WEIGHTS)
        os.chmod(self.src, 0o000)
        self.addCleanup(os.chmod, self.src, 0o755)
        with self.assertRaises(Refused) as cm:
            envcheck.copy_model_files({"files": [entry]}, self.tmp / "copy")
        self.assertEqual(cm.exception.problems[0].fix, TRAVERSE_FIX)


@skip_unless_linux
class KeeperHandoffTest(unittest.TestCase):
    """The keeper's `evaluate` runs the copies (in process)."""

    def setUp(self):
        self.tmp = _tmpdir(self)
        self.dir = self.tmp / "state"
        self.dir.mkdir(mode=0o700)
        self.out = self.tmp / "out"
        self.out.mkdir(mode=0o755)
        self.recorder = _Recorder()
        trust_stages._scenario_entry_points = lambda: [_ScenarioEP(self.recorder)]
        trust_stages._reset_scenario_snapshot()
        self.addCleanup(trust_stages._reset_scenario_snapshot)
        self.addCleanup(setattr, trust_stages, "_scenario_entry_points", _ORIGINAL_EPS)
        self.state = self.open()
        self.pid = project_id("p1")
        self.state.open_project(self.pid)
        r = self.state.intake(self.pid, TOY, sha256_bytes(TOY))
        self.train = r["train_rows"]
        args = confirm_args()
        sentence = args.pop("sentence")
        self.state.confirm_metric(self.pid, args, sentence, os.getuid())
        self.reg = self.state.register(self.pid, registration(r["sealed_id"]))["hash"]

    def open(self) -> KeeperState:
        st = KeeperState(self.dir, owner_uid=os.getuid())
        self.addCleanup(st.close)
        return st

    def _file(self, name: str, data: bytes, where: Path | None = None) -> dict:
        p = (where or self.out) / name
        p.write_bytes(data)
        return {"path": str(p), "sha256": sha256_bytes(data)}

    def candidate(self, files: list[dict], variant: str = "c") -> dict:
        full = pack.fit(self.train, 0.0)
        tok = sorted(full)[0]
        return {**model({tok: full[tok]}, variant), "files": files}

    def evaluate(self, candidate: dict) -> dict:
        return self.state.evaluate(self.pid, self.reg, arm(BASE), arm(candidate))

    def models_left(self) -> list[Path]:
        models = self.dir / "models"
        return sorted(models.rglob("*")) if models.exists() else []

    def test_copy_is_what_runs(self):
        files = [self._file("model.safetensors", WEIGHTS), self._file("config.json", b"{}")]
        seen_modes: list[int] = []
        self.recorder.on_first_candidate = lambda ref: seen_modes.extend(
            stat.S_IMODE(os.lstat(f["path"]).st_mode) for f in ref["files"])
        self.assertEqual(self.evaluate(self.candidate(files))["j"], 1)
        originals = {f["path"] for f in files}
        self.assertEqual({os.path.basename(p) for p in self.recorder.paths}, {"model.safetensors", "config.json"})
        models = os.path.realpath(self.dir / "models") + os.sep
        for p in self.recorder.paths:
            self.assertNotIn(p, originals)
            self.assertTrue(os.path.realpath(p).startswith(models), p)
            self.assertEqual(Path(p).parent.name, "candidate")
        self.assertEqual(seen_modes, [0o600, 0o600])
        self.assertEqual(self.recorder.contents, {WEIGHTS, b"{}"})

    def test_swap_after_check_has_no_effect(self):
        entry = self._file("model.safetensors", WEIGHTS)
        original = Path(entry["path"])

        def swap(ref: dict) -> None:                              # scoring has started: the check is long done
            original.write_bytes(b"swapped weights")
            tmp = original.with_name("new.safetensors")
            tmp.write_bytes(b"replaced by rename")
            os.replace(tmp, original)

        self.recorder.on_first_candidate = swap
        att = self.evaluate(self.candidate([entry]))
        self.assertEqual(original.read_bytes(), b"replaced by rename")
        self.assertEqual(self.recorder.contents, {WEIGHTS})          # every call read the copy
        self.assertEqual(att["claim"]["record"]["candidate_model_sha256"], arm(self.candidate([entry]))["sha256"])

    def test_copy_refuses_before_spend(self):                    # Review Focus 2
        before, records = self.state.alpha(self.pid), self.state.records(self.pid)
        real = self._file("real.safetensors", WEIGHTS)
        link = self.out / "link.safetensors"
        link.symlink_to(real["path"])
        fifo = self.out / "fifo.safetensors"
        os.mkfifo(fifo)
        sub = self.out / "sub"
        sub.mkdir()
        cases = {
            "link": [{"path": str(link), "sha256": real["sha256"]}],
            "fifo": [{"path": str(fifo), "sha256": sha256_bytes(b"")}],
            "hash": [{**self._file("model.safetensors", WEIGHTS), "sha256": sha256_bytes(b"other")}],
            "duplicate": [self._file("config.json", b"{}"), self._file("config.json", b"{}", sub)],
            "pickle": [self._file("model.pt", b"weights")],
        }
        for name, files in cases.items():
            with self.subTest(name):
                self.assert_refused_before_spend(name, files, before, records)
        self.assertEqual(self.recorder.paths, set())

    def assert_refused_before_spend(self, name: str, files: list[dict], before: dict, records: list) -> None:
        with self.assertRaises(Refused) as cm:
            self.evaluate(self.candidate(files, name))
        self.assertNotIsInstance(cm.exception, RefusedAfterSpend)
        problems = cm.exception.problems
        self.assertTrue(problems and all(p.rule == "AN-13" for p in problems), name)
        self.assertTrue(all(p.field.startswith("candidate.record.files[") for p in problems), name)
        for p in problems:                                    # no file content in a refusal
            self.assertNotIn("weights", p.message + p.fix)
        self.assertEqual(self.state.alpha(self.pid), before, name)
        self.assertEqual(self.state.records(self.pid), records, name)
        self.assertEqual(self.models_left(), [], name)

    def test_jinja_accepted(self):
        files = [self._file("model.safetensors", WEIGHTS), self._file("chat_template.jinja", TEMPLATE)]
        self.assertEqual(self.evaluate(self.candidate(files))["j"], 1)
        self.assertIn(TEMPLATE, self.recorder.contents)

    def test_copies_removed(self):
        self.evaluate(self.candidate([self._file("model.safetensors", WEIGHTS)]))
        self.assertEqual(self.models_left(), [])
        # a failure while scoring removes them too
        self.recorder.raise_on_candidate = True
        with self.assertRaises(RefusedAfterSpend):
            self.evaluate(self.candidate([self._file("model.safetensors", WEIGHTS)], "c2"))
        self.assertEqual(self.models_left(), [])
        # a leftover from a crash is removed when the keeper starts
        left = self.dir / "models" / ("ab" * 16) / "candidate"
        left.mkdir(parents=True, mode=0o700)
        (left / "model.safetensors").write_bytes(WEIGHTS)
        self.state.close()
        self.state = self.open()
        self.assertEqual(self.models_left(), [])


    def swap_parent_after_path_checks(self, target_dir: Path) -> mock._patch:
        """Replace `out/sub` by a link to `target_dir` once the keeper's path-level checks passed, just before the
        copy opens the files (a parent swap that `O_NOFOLLOW` on the last component does not see)."""
        real = envcheck.copy_model_files
        sub = self.out / "sub"

        def swapped(record, dest, **kwargs):
            if record.get("files") and not sub.is_symlink():      # the candidate's call (the baseline has none)
                shutil.rmtree(sub)
                sub.symlink_to(target_dir)
            return real(record, dest, **kwargs)
        return mock.patch.object(envcheck, "copy_model_files", swapped)

    def assert_fd_check_refuses(self, plant: Path, words: str) -> None:
        sub = self.out / "sub"
        sub.mkdir()
        decoy = self._file("model.safetensors", WEIGHTS, sub)
        (plant / "model.safetensors").write_bytes(WEIGHTS)       # same bytes: only the fd check can refuse
        os.chmod(plant / "model.safetensors", 0o600)              # the keeper's user only
        before, records = self.state.alpha(self.pid), self.state.records(self.pid)
        with self.swap_parent_after_path_checks(plant), self.assertRaises(Refused) as cm:
            self.evaluate(self.candidate([decoy]))
        self.assertNotIsInstance(cm.exception, RefusedAfterSpend)
        p = cm.exception.problems[0]
        self.assertEqual((p.rule, p.field), ("AN-13", "candidate.record.files[0]"))
        self.assertIn(words, p.message)
        self.assertEqual(self.state.alpha(self.pid), before)
        self.assertEqual(self.state.records(self.pid), records)
        self.assertEqual(self.models_left(), [])
        self.assertEqual(self.recorder.paths, set())

    def test_opened_file_inside_state_refused(self):             # fix round 1, Important
        plant = self.dir / "plant"
        plant.mkdir(mode=0o700)
        self.assert_fd_check_refuses(plant, "inside the keeper's state directory")

    def as_other_owner(self) -> None:
        """The owner is another uid from here on; the test's directory is opened to it (mode 0755), as /tmp is."""
        os.chmod(self.tmp, 0o755)
        self.state.owner_uid = OTHER_UID
        self.addCleanup(setattr, self.state, "owner_uid", os.getuid())

    def test_opened_file_owner_cannot_read_refused(self):
        plant = self.tmp / "private"
        plant.mkdir(mode=0o700)
        self.as_other_owner()
        self.assert_fd_check_refuses(plant, "not readable by the owner uid")

    def test_opened_file_owner_cannot_search_refused(self):
        """The opened file is readable by every user, but a directory above it is not searchable by the owner."""
        locked = self.tmp / "locked"
        locked.mkdir(mode=0o700)
        plant = locked / "open"
        plant.mkdir(mode=0o755)
        self.as_other_owner()
        real = envcheck.copy_model_files

        def readable(record, dest, **kwargs):                     # the planted copy is 0644, unlike the helper's
            for f in plant.iterdir():
                os.chmod(f, 0o644)
            return real(record, dest, **kwargs)
        with mock.patch.object(envcheck, "copy_model_files", readable):
            self.assert_fd_check_refuses(plant, "cannot search a directory above the file")

    def test_owner_cannot_search_an_ancestor_refused_before_spend(self):
        """A 0644 file under a 0700 directory of another uid than the owner's is refused by the path check."""
        secret = self.out / "secret"
        secret.mkdir(mode=0o700)
        self.as_other_owner()
        before, records = self.state.alpha(self.pid), self.state.records(self.pid)
        entry = self._file("model.safetensors", WEIGHTS, secret)
        os.chmod(entry["path"], 0o644)
        self.assert_refused_before_spend("ancestor", [entry], before, records)
        with self.assertRaises(Refused) as cm:
            self.evaluate(self.candidate([entry], "again"))
        self.assertIn("cannot search a directory above the file", cm.exception.problems[0].message)
        self.assertEqual(cm.exception.problems[0].fix, envcheck.TRAVERSE_FIX)
        os.chmod(secret, 0o755)                                   # searchable: the same file passes
        self.assertEqual(self.evaluate(self.candidate([entry], "open"))["j"], 1)

    def test_cleanup_failure_does_not_change_the_result(self):  # fix round 1, best-effort removal
        with mock.patch.object(state_mod.shutil, "rmtree", side_effect=OSError("busy")):
            att = self.evaluate(self.candidate([self._file("model.safetensors", WEIGHTS)]))
        self.assertEqual(att["j"], 1)
        self.assertEqual(att["claim"]["record"]["j"], 1)
        self.assertNotEqual(self.models_left(), [])               # left behind ...
        self.state.close()
        self.state = self.open()                                  # ... and swept at the next start
        self.assertEqual(self.models_left(), [])


class ModelReferenceLimitTest(unittest.TestCase):
    def test_limit_text(self):
        self.assertEqual(texts.LIMIT_KEEPER_MODEL_REFERENCE,
                         "a model reference must content-address its weights: the keeper copies the files a "
                         "reference lists (`files`) into its own state, hashes the copy and runs only the copy, so "
                         "the hash in the claim is the hash of what ran; weights the reference reaches in any other "
                         "way are not covered")


if __name__ == "__main__":
    unittest.main()
