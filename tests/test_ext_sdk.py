"""The adapter SDK (`gatedtrain.ext.sdk`), the row building blocks (`gatedtrain.ext.rowkit`), the B18 executor policy
(`gatedtrain.ext.executor`) and the NeMo RL family facts (`gatedtrain.families.nemo_rl`).

Recipe adapters import these helpers from the installed gatedtrain (MM-12) and their CI installs it from PyPI, so the
framework suite pins them: every reader returns (value, None) or (None, why) and never raises on a bad file;
`parse_options` and `runtime_argv_problems` resolve options as argparse does (EXT-133, EXT-150, EXT-156: a unique
prefix counts, so `--lim` is `--limit`); `runtime_script_problems` binds the executed script (EXT-83).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from _support import file_sha256

from gatedtrain.ext import launch, rowkit, sdk
from gatedtrain.ext.executor import EXECUTOR_ENV, EXECUTOR_LIMIT_OPTIONS, EXECUTOR_LIMITS, EXECUTOR_TMPFS_OPTIONS
from gatedtrain.families import nemo_rl


class TmpCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-sdk-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def write(self, name: str, data: str | bytes) -> Path:
        p = self.tmp / name
        p.parent.mkdir(parents=True, exist_ok=True)
        (p.write_bytes if isinstance(data, bytes) else p.write_text)(data)
        return p


class TestReaders(TmpCase):

    def test_read_jsonl(self):
        p = self.write("a.jsonl", '{"a": 1}\n\n  \n{"b": 2}\r\n')
        self.assertEqual(sdk.read_jsonl(p), ([{"a": 1}, {"b": 2}], None))
        self.assertEqual(sdk.read_jsonl(self.write("e.jsonl", "")), ([], None))
        for name, data, why in (("bad.jsonl", '{"a": 1}\n{"a": \n', "bad.jsonl line 2 does not parse"),
                                ("list.jsonl", '{"a": 1}\n[1]\n', "list.jsonl line 2 is not a JSON object"),
                                ("deep.jsonl", "[" * 100000 + "\n", "deep.jsonl line 1 does not parse"),
                                ("bin.jsonl", b"\xff\xfe\n", "bin.jsonl cannot be read (UnicodeDecodeError)")):
            with self.subTest(name=name):
                self.assertEqual(sdk.read_jsonl(self.write(name, data)), (None, why))
        self.assertEqual(sdk.read_jsonl(self.tmp / "gone.jsonl"), (None, "gone.jsonl is not a plain file"))
        link = self.tmp / "link.jsonl"
        link.symlink_to(p)
        self.assertEqual(sdk.read_jsonl(link), (None, "link.jsonl is not a plain file"))       # never followed

    def test_read_json_and_run_args(self):
        self.assertEqual(sdk.read_json(self.write("a.json", '{"x": [1]}')), ({"x": [1]}, None))
        self.assertEqual(sdk.read_json(self.write("b.json", "{")), (None, "b.json does not parse (JSONDecodeError)"))
        self.assertEqual(sdk.read_json(self.tmp / "c.json"), (None, "c.json is missing"))
        self.assertEqual(sdk.read_run_args(self.tmp), (None, "run_args.json is missing"))
        self.write("run_args.json", "[1]")
        self.assertEqual(sdk.read_run_args(self.tmp), (None, "run_args.json is not a JSON object"))
        self.write("run_args.json", "{")
        self.assertEqual(sdk.read_run_args(self.tmp)[1], "run_args.json does not parse (JSONDecodeError)")
        self.write("run_args.json", '{"seed": 3}')
        self.assertEqual(sdk.read_run_args(self.tmp), ({"seed": 3}, None))

    def test_read_cut_marker(self):
        self.assertEqual(sdk.read_cut_marker(self.tmp),
                         (None, "cut.json is missing (cut.json is written last)"))
        self.write("cut.json", '{"files": []}')
        self.assertEqual(sdk.read_cut_marker(self.tmp), (None, "cut.json has no `files` object"))
        self.write("cut.json", '{"files": {}, "epoch": 1}')
        self.assertEqual(sdk.read_cut_marker(self.tmp), ({"files": {}, "epoch": 1}, None))

    def test_sha_or_none(self):
        p = self.write("f", "x")
        self.assertEqual(sdk.sha_or_none(p), file_sha256(p))
        self.assertIsNone(sdk.sha_or_none(self.tmp / "gone"))


def safetensors(header: dict | bytes, data: bytes = b"", *, n: int | None = None) -> bytes:
    raw = header if isinstance(header, bytes) else json.dumps(header).encode()
    return struct.pack("<Q", len(raw) if n is None else n) + raw + data


class TestSafetensors(TmpCase):
    """EXT-43: a whole file is exactly 8 + header + the largest end offset bytes."""

    def test_whole_and_broken(self):
        good = {"__metadata__": {"format": "pt"}, "w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
        self.assertIsNone(sdk.safetensors_problem(self.write("ok.safetensors", safetensors(good, b"\0" * 8))))
        cases = (
            ("short", b"\1\0", "shorter than its 8-byte header length"),
            ("cap", struct.pack("<Q", sdk.MAX_HEADER_SIZE + 1), "exceeds the"),
            ("past", safetensors(b"{}", n=50), "runs past the end of the file"),
            ("json", safetensors(b"{nope"), "header does not parse"),
            ("list", safetensors(b"[1]"), "header is not a JSON object"),
            ("nooffs", safetensors({"w": {"dtype": "F32"}}), "has no data_offsets"),
            ("bool", safetensors({"w": {"data_offsets": [False, True]}}, b"\0"), "has no data_offsets"),
            ("order", safetensors({"w": {"data_offsets": [8, 0]}}, b"\0" * 8), "begin after end"),
            ("trunc", safetensors(good, b"\0" * 7), "truncated or padded"),
            ("pad", safetensors(good, b"\0" * 9), "truncated or padded"))
        for name, data, why in cases:
            with self.subTest(name=name):
                msg = sdk.safetensors_problem(self.write(f"{name}.safetensors", data))
                self.assertIsNotNone(msg)
                self.assertIn(why, msg)
        self.assertEqual(sdk.safetensors_problem(self.tmp / "gone.safetensors"), "gone.safetensors is missing")


class TestValues(unittest.TestCase):

    def test_numbers(self):
        self.assertTrue(sdk.is_int(3))
        self.assertFalse(sdk.is_int(True))
        self.assertFalse(sdk.is_int(3.0))
        self.assertTrue(sdk.is_num(0.5))
        for x in (True, float("nan"), float("inf"), "1"):
            self.assertFalse(sdk.is_num(x), x)
        self.assertEqual(sdk.float32(0.1), struct.unpack("<f", struct.pack("<f", 0.1))[0])
        self.assertNotEqual(sdk.float32(0.1), 0.1)

    def test_hashes_and_seeds(self):
        obj = {"b": 1.0, "a": "é"}
        self.assertEqual(sdk.compact_sha256(obj), "sha256:" + hashlib.sha256('{"a":"é","b":1.0}'.encode()).hexdigest())
        want = int.from_bytes(hashlib.sha256(b"sample:7:1:3").digest()[:8], "big") >> 1
        self.assertEqual(sdk.derive_seed("sample", 7, 1, 3), want)
        self.assertLess(sdk.derive_seed("x"), sdk.MAX_SEED)
        self.assertEqual(sdk.norm_path("/a//b/./c/../d"), "/a/b/d")
        self.assertEqual(sdk.norm_path(None), None)


class TestTrainingData(TmpCase):

    def test_single_shard_and_concat(self):
        d = self.tmp / "data"
        d.mkdir()
        self.assertIsNone(sdk.single_shard(d))
        self.write("data/b.jsonl", '{"r": 2}\n')
        self.assertEqual(sdk.single_shard(d), "b.jsonl")
        self.write("data/a.jsonl", '{"r": 1}\n')
        self.assertIsNone(sdk.single_shard(d))
        want = "sha256:" + hashlib.sha256(b'{"r": 1}\n{"r": 2}\n').hexdigest()       # file-name order
        self.assertEqual(sdk.shards_concat_sha256(d), want)

    def test_bound_mounts(self):
        a = self.tmp / "a"
        a.mkdir()
        (self.tmp / "link").symlink_to(a)
        mounts = [{"host": str(a), "container": "/x", "mode": "ro"},
                  {"host": str(self.tmp / "link"), "container": "/y", "mode": "ro"},
                  {"host": str(a), "container": "/out", "mode": "rw"},
                  {"host": "relative/a", "container": "/z", "mode": "ro"}, "junk"]
        self.assertEqual([m["container"] for m in sdk.bound_mounts(mounts, a)], ["/x", "/y"])
        self.assertEqual(sdk.bound_mounts(None, a), [])


class TestRuntimeScript(TmpCase):
    """EXT-83: env-runtime.json `script_sha256` is the launch record's."""

    def fields(self, launch_record: dict) -> list:
        return [(p.rule, p.field) for p in sdk.runtime_script_problems(self.tmp, launch_record)]

    def test_rules(self):
        rec = {"script_sha256": "sha256:" + "a" * 64}
        want = [("EXT-LAUNCH", "--launch-record.script_sha256")]
        self.assertEqual(self.fields(rec), want)                                  # missing
        self.write("env-runtime.json", "{")
        self.assertEqual(self.fields(rec), want)                                  # does not parse
        self.write("env-runtime.json", json.dumps({"script_sha256": "sha256:" + "b" * 64}))
        self.assertEqual(self.fields(rec), want)                                  # another script
        self.write("env-runtime.json", "[]")
        self.assertEqual(self.fields(rec), want)                                  # not an object
        self.write("env-runtime.json", json.dumps({"script_sha256": rec["script_sha256"]}))
        self.assertEqual(self.fields(rec), [])


TRAIN_OPTIONS = {"--profile": True, "--ema": False, "--no-verify-base": False, "--out": True}
EVAL_OPTIONS = {"--base": True, "--items": True, "--limit": True, "--out": True}
RT = "/data/runtime"
PY = f"{RT}/.venv/bin/python"


class TestOptions(unittest.TestCase):
    """argparse's reading of options: exact, `=value`, a unique prefix (allow_abbrev); an ambiguous prefix and a
    single dash are not options."""

    def test_resolve_option(self):
        opts = {"--limit": True, "--lr": True, "--items": True}
        self.assertEqual(sdk.resolve_option("--items", opts), ("--items", None))
        self.assertEqual(sdk.resolve_option("--items=/x", opts), ("--items", "/x"))
        self.assertEqual(sdk.resolve_option("--it=", opts), ("--items", ""))
        self.assertEqual(sdk.resolve_option("--li", opts), ("--limit", None))
        self.assertEqual(sdk.resolve_option("--l", opts), (None, None))            # ambiguous: --limit, --lr
        for tok in ("-l", "--", "items", 5, None, "--other"):
            self.assertEqual(sdk.resolve_option(tok, opts), (None, None), tok)

    def test_parse_options(self):
        got = sdk.parse_options(["--prof", "smoke", "--ema", "pos", "--out=/out", "--unknown", "--profile"],
                                TRAIN_OPTIONS)
        self.assertEqual(got, [("--profile", "smoke", "--prof"), ("--ema", None, "--ema"),
                               ("--out", "/out", "--out=/out"), ("--profile", "", "--profile")])
        # a value option takes the next token even when it looks like an option (argparse does the same for
        # a value that is not a known option; here the value is consumed and not parsed again)
        self.assertEqual(sdk.parse_options(["--out", "--ema"], TRAIN_OPTIONS), [("--out", "--ema", "--out")])

    def problems(self, argv, profile="full"):
        return sdk.runtime_argv_problems(argv, profile=profile,
                                         options_of={"train.py": TRAIN_OPTIONS, "eval.py": EVAL_OPTIONS},
                                         eval_script="eval.py", default_base_rel="models/base", rule="EXT-X")

    def test_runtime_argv_scripts(self):
        self.assertEqual(self.problems([PY, f"{RT}/train.py", "--profile", "full"]),
                         ([], "train.py", [("--profile", "full", "--profile")]))
        for argv in ([PY, "/elsewhere/train.py"], [PY, f"{RT}/other.py"], [PY, f"{RT}/sub/train.py"]):
            with self.subTest(argv=argv):
                out, script, parsed = self.problems(argv)
                self.assertEqual((script, parsed), (None, []))
                self.assertEqual(len(out), 1)
                self.assertIn("is not one of the runtime's scripts", out[0])
                self.assertIn("(EXT-X)", out[0])
        for argv in (None, [PY], [PY, 3], "x"):
            self.assertEqual(self.problems(argv), ([], None, []))

    def test_runtime_argv_eval_defaults(self):
        ok = [PY, f"{RT}/eval.py", "--base", f"{RT}/models/base", f"--items={RT}/items", "--out", "/out"]
        self.assertEqual(self.problems(ok)[0], [])
        out, _, _ = self.problems([PY, f"{RT}/eval.py", "--ba", "/data/other", "--items", "/data/items2"])
        self.assertEqual(len(out), 2)
        self.assertTrue(all("is not its default" in m for m in out), out)

    def test_runtime_argv_limit(self):
        argv = [PY, f"{RT}/eval.py", "--lim", "20"]
        self.assertEqual(self.problems(argv, profile="smoke")[0], [])
        out = self.problems(argv, profile="full")[0]
        self.assertEqual(len(out), 1)
        self.assertIn("--limit '20' under --profile full", out[0])
        # the training script's options are not the evaluation script's
        self.assertEqual(self.problems([PY, f"{RT}/train.py", "--limit", "20"], profile="full")[0], [])


class TestMountHelpers(TmpCase):
    """The SDK's re-exports of ext-launch's read-only mount resolution (EXT-91)."""

    def test_container_file(self):
        code = self.tmp / "code"
        self.write("code/train.py", "x")
        mounts = [{"host": str(code), "container": "/code", "mode": "ro"}]
        self.assertEqual(sdk.ro_table(mounts), [("/code", os.path.realpath(code))])
        self.assertEqual(sdk.container_file("/code/train.py", mounts),
                         ("/code/train.py", os.path.join(os.path.realpath(code), "train.py")))
        self.assertIsNone(sdk.container_file("/code", mounts))                     # a directory
        self.assertIsNone(sdk.container_file("/out/train.py", mounts))              # no read-only mount
        self.assertTrue(sdk.within("/code/a", "/code"))
        self.assertFalse(sdk.within("/codex", "/code"))


class TestRowkit(TmpCase):

    def test_runtime_dir(self):
        anchor = self.tmp / "repo" / "src" / "pkg" / "rows.py"
        with mock.patch.dict(os.environ, {"X_RUNTIME": "/srv/rt"}):
            self.assertEqual(rowkit.runtime_dir("X_RUNTIME", str(anchor)), "/srv/rt")
        with mock.patch.dict(os.environ, {"X_RUNTIME": ""}):
            self.assertEqual(rowkit.runtime_dir("X_RUNTIME", str(anchor)),
                             str((self.tmp / "repo").resolve() / "runtime"))

    def test_venv_rows(self):
        rt = str(self.tmp / "rt")
        rows = rowkit.venv_runtime_rows(rt, "area-x")                              # no venv yet
        self.assertIsNone(rowkit.python_tree(rt))
        self.assertEqual(rows, {"input_roots": (rt, "{output_root}/area-x"), "scan_skip": (f"{rt}/.venv",),
                                "interpreter_trees": (), "interpreter": f"{rt}/.venv/bin/python"})
        tree = "/opt/uv/python/cpython-3.12.16-linux-x86_64-gnu"
        self.write("rt/.venv/pyvenv.cfg", f"version_info = 3.12.16\nhome = {tree}/bin\nnot a pair\n")
        self.assertEqual(rowkit.python_tree(rt), tree)
        rows = rowkit.venv_runtime_rows(rt, "area-x")
        self.assertEqual(rows["input_roots"], ("/opt/uv/python", rt, "{output_root}/area-x"))
        self.assertEqual(rows["scan_skip"], (f"{rt}/.venv", f"{tree}/lib/python3.12/site-packages"))
        self.assertEqual(rows["interpreter_trees"], (tree,))
        self.write("rt/.venv/pyvenv.cfg", "home =\n")
        self.assertIsNone(rowkit.python_tree(rt))

    def test_shared_tables(self):
        self.assertEqual(rowkit.CONTAINER_PATHS["rw"], ("/out",))
        self.assertIn("model-*-of-*.safetensors", rowkit.HF_LLM_COMPONENTS)


class TestExecutorPolicy(unittest.TestCase):
    """B18: the executor-image steps (`b18` rows, printed by `build_command`) and the paired sidecar
    (`sidecar_commands`) print the same limits; both come from `gatedtrain.ext.executor`."""

    def test_limit_options_agree(self):
        row = {"name_prefix": "x-", "tmpfs_options": EXECUTOR_TMPFS_OPTIONS}
        side = {"runtime": {"image": "sha256:" + "0" * 64}, "argv": ["/p"], "user": "1000:1000", "tmpfs": ["/tmp"],
                "limits": dict(EXECUTOR_LIMITS), "mounts": []}
        start, _ = launch.sidecar_commands(side, {}, row, "x")
        self.assertIn(" " + " ".join(EXECUTOR_LIMIT_OPTIONS) + " ", start)
        self.assertIn(" --read-only --tmpfs /tmp:rw,nosuid,nodev,noexec,size=4g ", start)
        self.assertEqual(set(EXECUTOR_LIMITS), set(launch.SIDECAR_LIMIT_KEYS))
        self.assertIn("noexec", EXECUTOR_TMPFS_OPTIONS.split(","))
        self.assertEqual(EXECUTOR_ENV, ("HOME", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED"))


class TestNemoRlFamily(unittest.TestCase):
    """The NeMo RL pins and output layout the NeMo RL recipes share."""

    def test_pins(self):
        for image in nemo_rl.NEMO_RL_IMAGES:
            self.assertTrue(launch.IMAGE_RE.fullmatch(image), image)              # pinned by image ID (MM-32)
        self.assertEqual(nemo_rl.NEMO_RL_IMAGES[0], nemo_rl.NEMO_RL_IMAGE)
        self.assertRegex(nemo_rl.NEMO_RL_COMMIT, r"^[0-9a-f]{40}$")
        m = nemo_rl.FREEZE_COMMIT_RE.fullmatch(f"# nemo-rl commit {nemo_rl.NEMO_RL_COMMIT}")
        self.assertEqual(m.group(1), nemo_rl.NEMO_RL_COMMIT)
        for py in (nemo_rl.NEMO_RL_PYTHON, nemo_rl.VLLM_WORKER_PYTHON):
            self.assertRegex(os.path.basename(py), launch.INTERPRETER_RE)
        # every option the printed command adds is an option or the value of the option before it (EXT-39)
        opts = nemo_rl.NEMO_RL_RUN_OPTIONS
        self.assertTrue(opts[0].startswith("--shm-size="))
        self.assertTrue(all(o.startswith("-") or opts[i - 1] in ("--ulimit", "-w") for i, o in enumerate(opts)))

    def test_layout(self):
        self.assertEqual(nemo_rl.CHECKPOINT_DIR_RE.match("checkpoint-12").group(1), "12")
        for name in (".checkpoint-12.partial", "checkpoint-", "checkpoint-1x", "step_12"):
            self.assertIsNone(nemo_rl.CHECKPOINT_DIR_RE.match(name), name)
        self.assertEqual(nemo_rl.TRAIN_DATA_RE.fullmatch("train_data_step7.jsonl").group(1), "7")
        self.assertTrue(nemo_rl.EXP_DIR_RE.fullmatch("exp_003"))
        self.assertIsNone(nemo_rl.EXP_DIR_RE.fullmatch("exp_"))
        self.assertEqual((nemo_rl.CUTS_DIR, nemo_rl.CUT_MARKER, nemo_rl.NEMO_LOGS), ("cuts", "cut.json",
                                                                                     ("nemo", "logs")))


if __name__ == "__main__":
    unittest.main()
