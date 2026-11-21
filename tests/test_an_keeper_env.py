"""AN-13: the keeper refuses to start when its code, environment or state directory is open to another uid, and
refuses model files whose format runs code on load (tests/README.md section 13)."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from _keeper import KeeperProcess, KeeperStartError, skip_unless_linux

from gatedtrain.ainative import signing
from gatedtrain.ainative.keeper import envcheck, server
from gatedtrain.errors import Refused

OTHER_UID = os.getuid() + 1


def _tmpdir(case: unittest.TestCase) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="gk-env-"))
    case.addCleanup(shutil.rmtree, tmp, True)
    return tmp


def _hex(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


@skip_unless_linux
class TestPathProblems(unittest.TestCase):
    def setUp(self):
        self.tmp = _tmpdir(self)
        os.chmod(self.tmp, 0o700)

    def test_own_tree_is_clean_for_the_owner(self):
        f = self.tmp / "mod.py"
        f.write_text("x = 1\n", encoding="utf-8")
        os.chmod(f, 0o644)
        self.assertEqual(envcheck.path_problems([f], os.getuid()), [])

    def test_group_writable_directory(self):
        d = self.tmp / "pkg"
        d.mkdir()
        os.chmod(d, 0o770)
        problems = envcheck.path_problems([d], os.getuid())
        self.assertEqual(len(problems), 1, problems)
        self.assertEqual(problems[0].rule, "AN-13")
        self.assertIn(str(d), problems[0].message)
        self.assertIn("0770", problems[0].message)

    def test_other_writable_file(self):
        f = self.tmp / "mod.py"
        f.write_text("x = 1\n", encoding="utf-8")
        os.chmod(f, 0o646)
        problems = envcheck.path_problems([f], os.getuid())
        self.assertEqual([p.rule for p in problems], ["AN-13"])
        self.assertIn(str(f), problems[0].message)
        self.assertIn("0646", problems[0].message)

    def test_a_writable_parent_is_found_from_a_clean_file(self):
        d = self.tmp / "pkg"
        d.mkdir()
        f = d / "mod.py"
        f.write_text("x = 1\n", encoding="utf-8")
        os.chmod(f, 0o644)
        os.chmod(d, 0o775)
        problems = envcheck.path_problems([f], os.getuid())
        self.assertEqual(len(problems), 1, problems)
        self.assertIn(str(d), problems[0].message)

    def test_parent_owned_by_the_test_user(self):
        f = self.tmp / "mod.py"
        f.write_text("x = 1\n", encoding="utf-8")
        os.chmod(f, 0o644)
        self.assertEqual(envcheck.path_problems([f], os.getuid()), [])
        problems = envcheck.path_problems([f], OTHER_UID)       # another keeper uid: the test user's tree is open
        messages = " ".join(p.message for p in problems)
        self.assertIn(str(self.tmp), messages)
        self.assertIn(f"uid {os.getuid()}", messages)

    def test_root_owned_sticky_directory_is_allowed(self):
        tmp_root = Path("/tmp")
        st = os.lstat(tmp_root)
        if st.st_uid != 0 or not st.st_mode & stat.S_ISVTX:
            self.skipTest("/tmp is not a root-owned sticky directory on this host")
        self.assertEqual(envcheck.path_problems([tmp_root], os.getuid()), [])

    def test_missing_path_is_skipped(self):
        self.assertEqual(envcheck.path_problems([self.tmp / "missing"], os.getuid()), [])

    def test_symlink_target_is_checked(self):
        d = self.tmp / "real"
        d.mkdir()
        os.chmod(d, 0o777)
        link = self.tmp / "link"
        link.symlink_to(d)
        problems = envcheck.path_problems([link], os.getuid())
        self.assertTrue(any(str(d) in p.message for p in problems), problems)

    def test_each_link_hop_is_checked(self):
        hop_dir = self.tmp / "hops"
        hop_dir.mkdir()
        real = self.tmp / "real"
        real.mkdir()
        os.chmod(hop_dir, 0o700)
        os.chmod(real, 0o700)
        (hop_dir / "second").symlink_to(real)
        first = self.tmp / "first"
        first.symlink_to(hop_dir / "second")
        self.assertEqual(envcheck.path_problems([first], os.getuid()), [])
        os.chmod(hop_dir, 0o777)                               # the middle link could be replaced
        problems = envcheck.path_problems([first], os.getuid())
        self.assertEqual([str(hop_dir) in p.message for p in problems], [True], problems)


@skip_unless_linux
class TestStateDir(unittest.TestCase):
    def setUp(self):
        self.tmp = _tmpdir(self)
        self.state = self.tmp / "state"
        self.state.mkdir()
        os.chmod(self.state, 0o700)

    def test_0700_owned_is_clean(self):
        self.assertEqual(envcheck.state_dir_problems(self.state, os.getuid()), [])

    def test_mode_0750_refused(self):
        os.chmod(self.state, 0o750)
        problems = envcheck.state_dir_problems(self.state, os.getuid())
        self.assertEqual([(p.rule, p.field) for p in problems], [("AN-13", "state")])
        self.assertIn("0750", problems[0].message)

    def test_symlinked_state_dir_refused(self):
        link = self.tmp / "state-link"
        link.symlink_to(self.state)
        problems = envcheck.state_dir_problems(link, os.getuid())
        self.assertEqual([p.rule for p in problems], ["AN-13"])
        self.assertIn("symbolic link", problems[0].message)

    def test_missing_and_not_a_directory_refused(self):
        self.assertEqual(len(envcheck.state_dir_problems(self.tmp / "missing", os.getuid())), 1)
        f = self.tmp / "file"
        f.write_text("", encoding="utf-8")
        os.chmod(f, 0o700)
        self.assertEqual(len(envcheck.state_dir_problems(f, os.getuid())), 1)

    def test_other_owner_refused(self):
        problems = envcheck.state_dir_problems(self.state, OTHER_UID)
        self.assertEqual(len(problems), 1)
        self.assertIn(f"uid {os.getuid()}", problems[0].message)


class TestModelFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = _tmpdir(self)

    def _file(self, name: str, data: bytes) -> dict:
        p = self.tmp / name
        p.write_bytes(data)
        return {"path": str(p), "sha256": _hex(data)}

    def _refused(self, entry: dict) -> list:
        with self.assertRaises(Refused) as cm:
            envcheck.check_model_files({"files": [entry]})
        self.assertTrue(all(p.rule == "AN-13" for p in cm.exception.problems))
        return cm.exception.problems

    def test_suffix_lists(self):
        self.assertEqual(envcheck.MODEL_FILE_SUFFIXES, (".safetensors", ".json", ".jsonl", ".txt"))
        self.assertEqual(envcheck.PICKLE_SUFFIXES, (".pt", ".pth", ".bin", ".pkl", ".pickle", ".ckpt", ".joblib"))

    def test_pickle_suffix_refused(self):
        problems = self._refused(self._file("model.pt", b"weights"))
        self.assertIn("pickle", problems[0].message)

    def test_unknown_suffix_refused(self):
        self._refused(self._file("model.onnx", b"weights"))

    def test_pickle_bytes_in_a_json_refused(self):
        problems = self._refused(self._file("config.json", b"\x80\x04\x95 pickled"))
        self.assertIn("pickle", problems[0].message)

    def test_zip_in_a_safetensors_refused(self):
        problems = self._refused(self._file("model.safetensors", b"PK\x03\x04 torch zip"))
        self.assertIn("zip", problems[0].message)

    def test_hash_mismatch_refused(self):
        entry = self._file("model.safetensors", b"weights")
        entry["sha256"] = _hex(b"other")
        problems = self._refused(entry)
        self.assertIn("sha256", problems[0].message)

    def test_symlink_refused(self):
        entry = self._file("real.safetensors", b"weights")
        link = self.tmp / "model.safetensors"
        link.symlink_to(entry["path"])
        self._refused({"path": str(link), "sha256": entry["sha256"]})

    def test_missing_file_refused(self):
        self._refused({"path": str(self.tmp / "missing.safetensors"), "sha256": _hex(b"")})

    def test_matching_safetensors_accepted(self):
        entries = [self._file("model.safetensors", b"\x08\x00\x00\x00\x00\x00\x00\x00{}      "),
                   self._file("config.json", b"{}"), self._file("tokenizer.txt", b"a b c")]
        self.assertEqual(envcheck.check_model_files({"files": entries}), entries)

    def test_no_files_key(self):
        self.assertEqual(envcheck.check_model_files({}), [])

    def test_every_bad_entry_is_named(self):
        good = self._file("model.safetensors", b"weights")
        with self.assertRaises(Refused) as cm:
            envcheck.check_model_files({"files": [good, self._file("a.pkl", b"x"), self._file("b.bin", b"y")]})
        self.assertEqual([p.field for p in cm.exception.problems], ["files[1]", "files[2]"])


class TestEnvironmentRecord(unittest.TestCase):
    def test_record_shape_and_scope(self):
        record = envcheck.environment_record()
        self.assertEqual(set(record), {"python", "distributions"})
        self.assertEqual(record["python"], sys.version)
        dists = record["distributions"]
        names = [d["name"].lower() for d in dists]
        self.assertEqual(names, sorted(names))
        for required in ("gatedtrain", "numpy", "scipy", "pyarrow", "duckdb", "pyyaml", "cryptography", "cffi"):
            self.assertIn(required, names)
        self.assertNotIn("torch", names)                       # an extra (tier-a), not a runtime dependency
        for d in dists:
            self.assertEqual(set(d), {"name", "version", "files_sha256"})
            self.assertRegex(d["files_sha256"], r"^sha256:[0-9a-f]{64}$")

    def test_sha256_is_stable(self):
        a = envcheck.environment_record()
        self.assertEqual(envcheck.environment_sha256(a), envcheck.environment_sha256(envcheck.environment_record()))
        self.assertRegex(envcheck.environment_sha256(a), r"^sha256:[0-9a-f]{64}$")
        b = {**a, "python": a["python"] + " "}
        self.assertNotEqual(envcheck.environment_sha256(a), envcheck.environment_sha256(b))


@skip_unless_linux
class TestKeeperCodePaths(unittest.TestCase):
    def test_paths_cover_the_interpreter_and_the_imported_files(self):
        import numpy
        import gatedtrain.ainative.keeper.envcheck as mod
        paths = {os.path.realpath(p) for p in envcheck.keeper_code_paths()}
        self.assertIn(os.path.realpath(sys.prefix), paths)
        self.assertIn(os.path.realpath(numpy.__file__), paths)
        self.assertIn(os.path.realpath(mod.__file__), paths)

    def test_this_environment_passes_for_the_test_user(self):
        # A same-user keeper on the test venv. If a host or CI toolcache leaves a group- or other-writable path,
        # this fails by design (AN-13): fix the permissions, never the rule.
        problems = envcheck.path_problems(envcheck.keeper_code_paths(), os.getuid())
        self.assertEqual(problems, [], "\n".join(p.line() for p in problems[:20]))


@skip_unless_linux
class TestServe(unittest.TestCase):
    def setUp(self):
        self.tmp = _tmpdir(self)
        self.state = self.tmp / "state"
        self.state.mkdir()
        os.chmod(self.state, 0o700)
        self.sock = self.tmp / "keeper.sock"

    def test_state_dir_0755_refused_before_anything_is_written(self):
        os.chmod(self.state, 0o755)
        with mock.patch.object(envcheck, "keeper_code_paths", lambda: []), self.assertRaises(Refused) as cm:
            server.serve(self.state, self.sock, os.getuid(), stop=threading.Event())
        self.assertEqual([(p.rule, p.field) for p in cm.exception.problems], [("AN-13", "state")])
        self.assertEqual(list(self.state.iterdir()), [])
        self.assertFalse(self.sock.exists())

    def test_writable_code_path_refused(self):
        code = self.tmp / "site"
        code.mkdir()
        os.chmod(code, 0o777)
        with mock.patch.object(envcheck, "keeper_code_paths", lambda: [code]), self.assertRaises(Refused) as cm:
            server.serve(self.state, self.sock, os.getuid(), stop=threading.Event())
        self.assertEqual({p.rule for p in cm.exception.problems}, {"AN-13"})
        self.assertTrue(any(str(code) in p.message for p in cm.exception.problems))
        self.assertEqual(list(self.state.iterdir()), [])

    def test_problem_list_is_capped(self):
        many = [self.tmp / f"d{i}" for i in range(50)]
        for d in many:
            d.mkdir()
            os.chmod(d, 0o777)
        with mock.patch.object(envcheck, "keeper_code_paths", lambda: many), self.assertRaises(Refused) as cm:
            server.serve(self.state, self.sock, os.getuid(), stop=threading.Event())
        problems = cm.exception.problems
        self.assertEqual(len(problems), server.MAX_START_PROBLEMS + 1)
        self.assertIn("more", problems[-1].message)


@skip_unless_linux
class TestKeeperProcessEnvironment(unittest.TestCase):
    def test_state_dir_0755_fails_to_start_with_an13(self):
        with self.assertRaises(KeeperStartError) as cm:
            with KeeperProcess(state_mode=0o755):
                pass
        self.assertEqual(cm.exception.returncode, 3, cm.exception.output)
        self.assertIn("error[AN-13] state", cm.exception.output)

    def test_hello_reports_key_and_environment_and_the_key_persists(self):
        with KeeperProcess() as k:
            hello = k.client.call("hello", None, {})
            key_path = k.state / "keeper.key"
            self.assertEqual(stat.S_IMODE(os.lstat(key_path).st_mode), 0o600)
            key = signing.load_private_key(key_path)
            self.assertEqual(hello["fingerprint"], signing.fingerprint(key))
            self.assertEqual(hello["public_pem"], signing.public_pem(key).decode("ascii"))
            self.assertRegex(hello["environment_sha256"], r"^sha256:[0-9a-f]{64}$")
            self.assertIs(hello["same_user"], True)
            k.restart()
            again = k.client.call("hello", None, {})
            self.assertEqual(again["fingerprint"], hello["fingerprint"])
            self.assertEqual(again["environment_sha256"], hello["environment_sha256"])


if __name__ == "__main__":
    unittest.main()
