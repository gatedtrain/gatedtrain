"""Owner-side settings: the host file's GPU keys and layout keys, the keeper's GPU and eval-interpreter options, and the
pack settings table. Each is validated, defaults to the old behaviour, and reaches a recorded hash only when set."""

from __future__ import annotations

import io
import json
import math
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

from gatedtrain import cli
from gatedtrain.ainative import batch, runtime
from gatedtrain.ainative.keeper import inference, setup
from gatedtrain.errors import Refused
from gatedtrain.ext import launch, rowkit, smoke

IMAGE = "sha256:" + "a" * 64
# sha256 of the canonical default environment record, taken from gatedtrain 0.6.0 (no GPU, no eval-python option set)
ENV_RECORD_SHA256_0_6_0 = "6cc014e68e5feca17d636c255830a27875f3c32af36211ade65c4d6bed0cc4f7"
SERVE = ["keeper", "serve", "--state", "/s", "--socket", "/k", "--owner-uid", "5"]


def write_host(tmp: Path, text: str) -> dict[str, str]:
    path = tmp / "host.toml"
    path.write_text(text, encoding="utf-8")
    return {launch.ENV_HOST_CONFIG: str(path)}


BASE = 'host_work = "/w"\noutput_root = "/o"\n'


class HostFileNumbers(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def cfg(self, extra: str = "") -> launch.HostConfig:
        return launch.host_config(write_host(self.tmp, BASE + extra))

    def test_defaults(self) -> None:
        cfg = self.cfg()
        self.assertIsNone(cfg.gpu_mem_limit_gb)
        self.assertIsNone(cfg.gpu_cap_h)
        self.assertEqual(launch.gpu_mem_limit_gb(cfg), 22.5)
        self.assertEqual(launch.gpu_cap_h(cfg), 1000)

    def test_set(self) -> None:
        cfg = self.cfg("gpu_mem_limit_gb = 31.5\ngpu_cap_h = 40\n")
        self.assertEqual((launch.gpu_mem_limit_gb(cfg), launch.gpu_cap_h(cfg)), (31.5, 40))

    def test_validation(self) -> None:
        for bad in ("0", "-1", "true", '"22"', "nan", "inf", "[1]"):
            for key in ("gpu_mem_limit_gb", "gpu_cap_h"):
                with self.subTest(key=key, bad=bad), self.assertRaises(Refused) as cm:
                    self.cfg(f"{key} = {bad}\n")
                self.assertIn(key, str(cm.exception))

    def test_a_huge_integer_is_refused_not_a_crash(self) -> None:
        for key in ("gpu_mem_limit_gb", "gpu_cap_h"):
            with self.assertRaises(Refused):
                self.cfg(f"{key} = 1{'0' * 400}\n")

    def test_the_bounds_for_collect_need_no_layout_and_refuse_a_bad_file(self) -> None:
        env = write_host(self.tmp, "gpu_cap_h = 40\n")
        self.assertEqual(launch.owner_gpu_bounds(env), {"gpu_mem_limit_gb": 22.5, "gpu_cap_h": 40})
        with self.assertRaises(Refused):
            launch.owner_gpu_bounds(write_host(self.tmp, "gpu_cap_h = -1\n"))

    def test_the_owner_bound_needs_no_layout(self) -> None:
        env = write_host(self.tmp, "gpu_mem_limit_gb = 31.5\n")           # no host_work, no output_root
        self.assertEqual(launch.owner_gpu_mem_limit_gb(env), 31.5)
        self.assertEqual(launch.owner_gpu_mem_limit_gb({launch.ENV_HOST_CONFIG: str(self.tmp / "none")}), 22.5)
        with self.assertRaises(Refused):
            launch.owner_gpu_mem_limit_gb(write_host(self.tmp, "gpu_mem_limit_gb = 0\n"))

    def test_no_environment_variable_sets_them(self) -> None:
        env = {**write_host(self.tmp, BASE), "GATEDTRAIN_EXT_GPU_MEM_LIMIT_GB": "99", "GATEDTRAIN_GPU_CAP_H": "1"}
        cfg = launch.host_config(env)
        self.assertEqual((launch.gpu_mem_limit_gb(cfg), launch.gpu_cap_h(cfg)), (22.5, 1000))

    def test_hash_unchanged_unless_set(self) -> None:
        plain = launch.host_config_sha256(self.cfg())
        self.assertEqual(plain, launch.host_config_sha256(launch.HostConfig(host_work="/w", output_root="/o")))
        mem = launch.host_config_sha256(self.cfg("gpu_mem_limit_gb = 31.5\n"))
        cap = launch.host_config_sha256(self.cfg("gpu_cap_h = 40\n"))
        self.assertEqual(len({plain, mem, cap}), 3)
        self.assertEqual(mem, launch.host_config_sha256(self.cfg("gpu_mem_limit_gb = 31.5\n")))
        self.assertNotEqual(mem, launch.host_config_sha256(self.cfg("gpu_mem_limit_gb = 31.0\n")))

    def test_unknown_keys_are_still_refused(self) -> None:
        with self.assertRaises(Refused):
            self.cfg("gpu_memory = 1\n")


class SmokeUsesTheHostsBounds(unittest.TestCase):
    REC = {"peak_gpu_mem_gb": 23.0, "projected_gpu_h": 1.0, "gpu_log_total_h": 5.0, "error_count": 0,
           "baseline_scores": [0.2, 0.4, 0.6, 0.8]}

    def verdict(self, **kw):
        return smoke.smoke_verdict(self.REC, score_range=(0, 1), binary=False, **kw)

    def test_default_is_unchanged(self) -> None:
        self.assertEqual(self.verdict()["reasons"], ["gpu_memory"])

    def test_the_hosts_memory_bound(self) -> None:
        self.assertEqual(self.verdict(gpu_mem_limit_gb=31.5)["verdict"], "PASS")
        self.assertEqual(self.verdict(gpu_mem_limit_gb=10)["reasons"], ["gpu_memory"])

    def test_the_hosts_budget(self) -> None:
        self.assertIn("budget", self.verdict(gpu_mem_limit_gb=31.5, gpu_cap_h=5)["reasons"])
        self.assertEqual(self.verdict(gpu_mem_limit_gb=31.5, gpu_cap_h=6)["verdict"], "PASS")

    def test_problems_forward_the_bounds(self) -> None:
        rec = {**self.REC, "schema": smoke.SMOKE_SCHEMA, "adapter": "a", "verdict": "PASS",
               **{k: None for k in smoke.SMOKE_KEYS if k not in self.REC and k not in ("schema", "adapter", "verdict")}}
        rec["base_model_hash"] = rec["item_set_hash"] = "x"
        kw = dict(adapter="a", score_range=(0, 1), binary=False)
        self.assertTrue([p for p in smoke.smoke_problems(rec, **kw) if "gpu_memory" in p])
        self.assertFalse([p for p in smoke.smoke_problems(rec, gpu_mem_limit_gb=31.5, **kw) if "gpu_memory" in p])

    def test_one_definition_of_the_default(self) -> None:
        self.assertIs(smoke.GPU_MEM_LIMIT_GB, launch.GPU_MEM_LIMIT_GB)


class CapIsPrinted(unittest.TestCase):
    def record(self, root: Path) -> dict:
        (root / "inputs").mkdir(exist_ok=True)
        return {"adapter": "synthetic", "argv": ["/inputs/bin/python", "/inputs/train.py"], "gpu_used_h": 0,
                "host_work": "/w", "label": "run-0123456789ab", "gpus": True, "network": "none", "user": "0:0",
                "mounts": [{"host": str(root / "inputs"), "container": "/inputs", "mode": "ro"}], "tmpfs": ["/tmp"],
                "runtime": {"kind": "container", "image": IMAGE}}

    def test_the_command_carries_the_hosts_cap(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            rec = self.record(Path(d))
            self.assertIn("GATEDTRAIN_GPU_CAP_H=1000 ", launch.build_command(rec, {}, est_h=1.0))
            self.assertIn("GATEDTRAIN_GPU_CAP_H=40 ", launch.build_command(rec, {}, est_h=1.0, gpu_cap_h=40))

    def test_ext_launch_passes_the_hosts_bounds(self) -> None:
        import inspect
        src = inspect.getsource(launch.ext_launch)
        self.assertIn("gpu_cap_h(cfg)", src)
        self.assertIn("gpu_mem_limit_gb(cfg)", src)


class LayoutKeysReplaceTheEnvironment(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def test_file_keys(self) -> None:
        env = write_host(self.tmp, BASE + 'input_roots = ["/i1", "/i2"]\ninterpreter_trees = ["/t"]\n'
                         'interpreter = "/t/bin/python"\nmirror_root = "/m"\n')
        cfg = launch.host_config(env)
        self.assertEqual((cfg.input_roots, cfg.interpreter_trees, cfg.interpreter, cfg.mirror_root),
                         (("/i1", "/i2"), ("/t",), "/t/bin/python", "/m"))

    def test_file_keys_are_validated(self) -> None:
        for body in ('input_roots = "/i"\n', 'input_roots = ["rel"]\n', 'interpreter = "bin/python"\n',
                     "mirror_root = 3\n", 'interpreter_trees = [1]\n'):
            with self.subTest(body=body), self.assertRaises(Refused):
                launch.host_config(write_host(self.tmp, BASE + body))

    def test_the_variable_still_wins_and_warns_with_a_removal_version(self) -> None:
        env = {**write_host(self.tmp, BASE + 'mirror_root = "/m"\n'), launch.ENV_MIRROR_ROOT: "/env"}
        self.assertEqual(launch.host_config(env).mirror_root, "/env")
        [line] = launch.override_warnings(env)
        self.assertIn("deprecated", line)
        self.assertIn(launch.ENV_REMOVED_IN, line)
        self.assertIn("mirror_root", line)

    def test_every_variable_has_a_file_key(self) -> None:
        named = {launch.ENV_HOST_WORK: "host_work", launch.ENV_OUTPUT_ROOT: "output_root",
                 launch.ENV_INPUT_ROOTS: "input_roots", launch.ENV_INTERPRETER_TREES: "interpreter_trees",
                 launch.ENV_INTERPRETER: "interpreter", launch.ENV_MIRROR_ROOT: "mirror_root"}
        self.assertEqual(named, launch.ENV_TO_FILE_KEY)
        known = (launch.HOST_FILE_KEYS + launch.HOST_FILE_PATH_KEYS + launch.HOST_FILE_LIST_KEYS
                 + launch.HOST_FILE_PATHLIST_KEYS)
        for key in named.values():
            self.assertIn(key, known)

    def test_the_hash_of_a_layout_is_the_same_from_either_channel(self) -> None:
        a = launch.host_config({**write_host(self.tmp, BASE), launch.ENV_MIRROR_ROOT: "/m"})
        b = launch.host_config(write_host(self.tmp, BASE + 'mirror_root = "/m"\n'))
        self.assertEqual(launch.host_config_sha256(a), launch.host_config_sha256(b))


class PackSettings(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def test_a_malformed_host_file_does_not_stop_the_import_of_a_pack(self) -> None:
        env = write_host(self.tmp, BASE + "gpu_cap_h = -1\n")
        anchor = str(self.tmp / "repo" / "src" / "pkg" / "adapter.py")
        err = io.StringIO()
        with mock.patch.dict(os.environ, env), redirect_stderr(err):
            os.environ.pop("X_RUNTIME", None)
            got = rowkit.runtime_dir("X_RUNTIME", anchor, pack="p")
        self.assertTrue(got.endswith("runtime"))
        self.assertIn("malformed", err.getvalue())
        self.assertIn("gpu_cap_h", err.getvalue())

    def test_the_deprecation_line_is_printed_once_per_variable(self) -> None:
        anchor = str(self.tmp / "repo" / "src" / "pkg" / "adapter.py")
        rowkit._WARNED.discard("X_ONCE")
        err = io.StringIO()
        with mock.patch.dict(os.environ, {"X_ONCE": "/e"}), redirect_stderr(err):
            rowkit.runtime_dir("X_ONCE", anchor, pack="p")
            rowkit.runtime_dir("X_ONCE", anchor, pack="p")
        self.assertEqual(err.getvalue().count("deprecated"), 1)

    def test_the_table(self) -> None:
        env = write_host(self.tmp, BASE + '[packs.text-gsm8k]\nruntime_dir = "/rt"\n')
        self.assertEqual(launch.pack_setting("text-gsm8k", "runtime_dir", env), "/rt")
        self.assertIsNone(launch.pack_setting("text-gsm8k", "other", env))
        self.assertIsNone(launch.pack_setting("vlm-clevr", "runtime_dir", env))

    def test_validation(self) -> None:
        for body in ('[packs]\nx = 1\n', '[packs.p]\nruntime_dir = 3\n', '[packs.p]\nruntime_dir = "rel"\n',
                     '[packs.p]\n"bad key" = "/a"\n'):
            with self.subTest(body=body), self.assertRaises(Refused):
                launch.host_config(write_host(self.tmp, BASE + body))

    def test_no_file_no_setting(self) -> None:
        self.assertIsNone(launch.pack_setting("p", "runtime_dir", {launch.ENV_HOST_CONFIG: str(self.tmp / "none")}))

    def test_the_hash_binds_a_pack_table_only_when_present(self) -> None:
        plain = launch.host_config_sha256(launch.host_config(write_host(self.tmp, BASE)))
        withp = launch.host_config_sha256(launch.host_config(
            write_host(self.tmp, BASE + '[packs.p]\nruntime_dir = "/rt"\n')))
        self.assertNotEqual(plain, withp)

    def test_runtime_dir_precedence(self) -> None:
        anchor = str(self.tmp / "repo" / "src" / "pkg" / "adapter.py")
        env_file = write_host(self.tmp, BASE + '[packs.p]\nruntime_dir = "/from-file"\n')
        with mock.patch.dict(os.environ, env_file, clear=False):
            os.environ.pop("X_RUNTIME", None)
            self.assertEqual(rowkit.runtime_dir("X_RUNTIME", anchor, pack="p"), "/from-file")
            self.assertTrue(rowkit.runtime_dir("X_RUNTIME", anchor, pack="q").endswith("runtime"))
            rowkit._WARNED.discard("X_RUNTIME")
            with mock.patch.dict(os.environ, {"X_RUNTIME": "/from-env"}), redirect_stderr(io.StringIO()) as err:
                self.assertEqual(rowkit.runtime_dir("X_RUNTIME", anchor, pack="p"), "/from-env")
            self.assertIn("deprecated", err.getvalue())
            self.assertIn("X_RUNTIME", err.getvalue())


class KeeperGpuAndInterpreter(unittest.TestCase):
    def config(self, **kw) -> inference.InferenceConfig:
        return inference.InferenceConfig(image_id=IMAGE, docker_host="unix:///x/docker.sock", **kw)

    def test_defaults(self) -> None:
        c = self.config()
        self.assertEqual((c.gpu_wait_s, c.gpu_busy_mib, c.eval_python), (runtime.GPU_WAIT_S, runtime.GPU_BUSY_MIB, None))

    def test_validation(self) -> None:
        for kw in ({"gpu_busy_mib": runtime.MAX_GPU_BUSY_MIB + 1}, {"gpu_wait_s": 10 ** 400}, {"gpu_wait_s": 0}, {"gpu_wait_s": -1}, {"gpu_wait_s": True}, {"gpu_wait_s": math.inf},
                   {"gpu_wait_s": math.nan}, {"gpu_wait_s": "9"}, {"gpu_busy_mib": -1}, {"gpu_busy_mib": 1.5},
                   {"gpu_busy_mib": True}, {"eval_python": Path("rel/python")}, {"eval_python": Path("/a:b/python")}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self.config(**kw)

    def keeper(self, **kw) -> inference.KeeperInference:
        from gatedtrain.testing.fake_runtime import FakeRuntime
        with tempfile.TemporaryDirectory() as d:
            return inference.KeeperInference(self.config(**kw), FakeRuntime({}), Path(d))

    def test_the_default_environment_record_is_byte_identical_to_0_6_0(self) -> None:
        import hashlib
        from gatedtrain.canonical import canonical_json
        got = hashlib.sha256(canonical_json(self.keeper().environment())).hexdigest()
        self.assertEqual(got, ENV_RECORD_SHA256_0_6_0)

    def test_the_environment_record_is_unchanged_by_default_and_names_a_change(self) -> None:
        plain = self.keeper().environment()
        self.assertNotIn("gpu", plain)
        self.assertNotIn("eval_python", plain)
        changed = self.keeper(gpu_wait_s=300, gpu_busy_mib=2048, eval_python=Path("/e/bin/python")).environment()
        self.assertEqual(changed["gpu"], {"wait_s": 300, "busy_mib": 2048})
        self.assertEqual(changed["eval_python"], "/e/bin/python")
        only_wait = self.keeper(gpu_wait_s=300).environment()
        self.assertEqual(only_wait["gpu"], {"wait_s": 300, "busy_mib": runtime.GPU_BUSY_MIB})

    def test_the_interpreter_override_applies_only_to_eval_tree_generators(self) -> None:
        k = self.keeper(eval_python=Path("/e/bin/python"))
        with_trees = batch.BatchGenerator(script=Path("/s.py"), interpreter="/old/python", gpu=True, eval_trees=True)
        image = batch.BatchGenerator(script=Path("/s.py"), interpreter="/usr/bin/python3", gpu=True)
        self.assertEqual(k._resolve(with_trees).interpreter, "/e/bin/python")
        self.assertEqual(k._resolve(image).interpreter, "/usr/bin/python3")
        self.assertEqual(self.keeper()._resolve(with_trees).interpreter, "/old/python")

    def test_the_fit_check_uses_the_configured_wait(self) -> None:
        from gatedtrain.ainative.keeper import protocol
        k = self.keeper(gpu_wait_s=runtime.GPU_WAIT_S + 1000)
        gen = batch.BatchGenerator(script=Path("/s.py"), interpreter="/p", gpu=True)
        with self.assertRaises(Refused) as cm:
            k.ready(gen)
        self.assertIn("does not fit", str(cm.exception))
        self.assertGreater(batch.evaluation_wait_bound(k.config.gpu_wait_s, gen.time_limit_s),
                           protocol.LONG_OP_TIMEOUT_S)

    def test_serve_argv_prints_only_what_was_set(self) -> None:
        from pathlib import PurePosixPath as P
        base = setup.serve_argv(P("/v"), P("/s"), P("/k"), 5, inference_image=IMAGE)
        self.assertNotIn("--gpu-wait-s", base)
        argv = setup.serve_argv(P("/v"), P("/s"), P("/k"), 5, inference_image=IMAGE, gpu_wait_s=300, gpu_busy_mib=2048,
                                eval_python="/e/bin/python")
        self.assertEqual(argv[len(base):], ["--gpu-wait-s", "300", "--gpu-busy-mib", "2048", "--eval-python",
                                           "/e/bin/python"])

    def test_the_cli_builds_the_config(self) -> None:
        args = cli.build_parser().parse_args(
            SERVE + ["--inference-image", IMAGE, "--docker-host", "unix:///x/d.sock", "--gpu-wait-s", "300",
             "--gpu-busy-mib", "2048", "--eval-python", "/e/bin/python"])
        c = cli._inference_config(args)
        self.assertEqual((c.gpu_wait_s, c.gpu_busy_mib, c.eval_python), (300, 2048, Path("/e/bin/python")))
        plain = cli._inference_config(cli.build_parser().parse_args(
            SERVE + ["--inference-image", IMAGE, "--docker-host", "unix:///x/d.sock"]))
        self.assertEqual((plain.gpu_wait_s, plain.gpu_busy_mib, plain.eval_python),
                         (runtime.GPU_WAIT_S, runtime.GPU_BUSY_MIB, None))

    def test_the_options_need_the_image(self) -> None:
        args = cli.build_parser().parse_args(SERVE + ["--gpu-wait-s", "300"])
        with self.assertRaises(Refused):
            cli._inference_config(args)

    def test_setup_keeper_prints_them_into_the_unit(self) -> None:
        out = setup.setup_commands(owner_uid=1000, inference_image=IMAGE, docker_host="unix:///x/d.sock",
                                   gpu_wait_s=300, gpu_busy_mib=2048, eval_python="/e/bin/python")
        self.assertIn("--gpu-wait-s 300 --gpu-busy-mib 2048 --eval-python /e/bin/python", out["unit"])
        for bad in ({"gpu_wait_s": 0}, {"gpu_busy_mib": -5}, {"eval_python": "rel"}, {"eval_python": "/a b"}):
            with self.subTest(bad=bad), self.assertRaises(Refused):
                setup.setup_commands(owner_uid=1000, inference_image=IMAGE, docker_host="unix:///x/d.sock", **bad)
        with self.assertRaises(Refused):
            setup.setup_commands(owner_uid=1000, inference_image=IMAGE, docker_host="unix:///x/d.sock",
                                 gpu_wait_s=10 ** 400)
        with self.assertRaises(Refused):                     # the options belong to the inference setup
            setup.setup_commands(owner_uid=1000, gpu_wait_s=300)


if __name__ == "__main__":
    unittest.main()
