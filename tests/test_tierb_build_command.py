"""H-5 `build_command` extension (TIER-B-SPEC section 1; draft sections 1.4-1.6 and section 2
items 3-9; interim rules G-7 and G-15; tests/README.md section 11.6).

Golden contract: tests/fixtures/tierb/golden/build_command.json. Overrides compare as sets;
the program, the working directory and the docker flag groups compare exactly.
The keys themselves are checked against the pinned recipe in test_tierb_citations.
"""

from __future__ import annotations

import json
import shlex
import shutil
import tempfile
import unittest
from pathlib import Path

from _tierb import TIERB, build_stage_out

GOLDEN = json.loads((TIERB / "golden" / "build_command.json").read_text(encoding="utf-8"))
ARGS = GOLDEN["args"]
STAGE = {"id": "rl", "uses": "rl.nemo_rl@0.1", "algorithm": "grpo", "model_family": "llm.qwen3@0.1",
         "inputs": {"model": "base", "data": "train", "config": "trainer_cfg"}, "outputs": {"model": "rl_ckpt"}}
FLAGS_WITH_VALUE = {"--gpus", "--ulimit", "-e", "-v", "-w", "--runtime", "--shm-size", "--ipc", "-u"}


def expected_overrides(profile: str) -> set[str]:
    sub = {"{out}": ARGS["out_dir"], "{export}": ARGS["export_dir"], "{base_model}": ARGS["base_model"]}
    out = set()
    for o in GOLDEN["common_overrides"] + GOLDEN["profiles"][profile]["overrides"]:
        for k, v in sub.items():
            o = o.replace(k, v)
        out.add(o)
    return out


def spec(launcher: str, profile: str) -> dict:
    from gatedtrain.backends.nemo_rl import launch_spec

    return launch_spec(STAGE, nemo_rl_dir=ARGS["nemo_rl_dir"], launcher=launcher, out_dir=ARGS["out_dir"],
                       export_dir=ARGS["export_dir"], profile=profile, image=ARGS["image"],
                       work_dir=ARGS["work_dir"], hf_cache=ARGS["hf_cache"], base_model=ARGS["base_model"])


def split_program(argv: list[str], program_head: list[str]) -> tuple[list[str], list[str], list[str]]:
    """(prefix before the program, program head, overrides after it)."""
    n = len(program_head)
    for i in range(len(argv) - n + 1):
        if argv[i:i + n] == program_head:
            return argv[:i], argv[i:i + n], argv[i + n:]
    raise AssertionError(f"program {program_head} not found in {argv}")


class TestLocalLauncher(unittest.TestCase):
    def test_profiles(self):
        for profile in ("smoke", "long"):
            with self.subTest(profile=profile):
                s = spec("local", profile)
                prefix, _, overrides = split_program(s["argv"], GOLDEN["program"]["local"])
                self.assertEqual(prefix, [])
                self.assertEqual(set(overrides), expected_overrides(profile))
                self.assertEqual(len(overrides), len(set(overrides)), "no duplicate overrides")
                self.assertEqual(s["cwd"], GOLDEN["cwd"]["local"], "draft section 2 item 8: working directory")
                self.assertEqual(s["samples_per_position"], GOLDEN["profiles"][profile]["samples_per_position"])

    def test_build_command_is_launch_spec_argv(self):
        from gatedtrain.backends.nemo_rl import build_command

        argv = build_command(STAGE, nemo_rl_dir=ARGS["nemo_rl_dir"], launcher="local", out_dir=ARGS["out_dir"],
                             export_dir=ARGS["export_dir"], profile="long", image=ARGS["image"],
                             work_dir=ARGS["work_dir"], hf_cache=ARGS["hf_cache"], base_model=ARGS["base_model"])
        self.assertEqual(argv, spec("local", "long")["argv"])


class TestDockerLauncher(unittest.TestCase):
    def test_profiles(self):
        for profile in ("smoke", "long"):
            with self.subTest(profile=profile):
                s = spec("docker", profile)
                prefix, _, overrides = split_program(s["argv"], GOLDEN["program"]["docker"])
                self.assertEqual(prefix[:2], ["docker", "run"])
                self.assertEqual(prefix[-1], ARGS["image"], "the image's own source; no checkout mounted")
                groups, i, flags = [], 2, prefix[2:-1]
                i = 0
                while i < len(flags):
                    f = flags[i]
                    if f in FLAGS_WITH_VALUE:
                        groups.append([f, flags[i + 1]])
                        i += 2
                    else:
                        groups.append([f])
                        i += 1
                shm = [g for g in groups if g[0].startswith("--shm-size")]
                self.assertTrue(shm, "--shm-size is set (docs/about/installation.md:25)")
                rest = sorted(g for g in groups if not g[0].startswith("--shm-size"))
                self.assertEqual(rest, sorted(GOLDEN["docker_flag_groups"]))
                self.assertFalse(any(g[0] == "-v" and g[1].split(":")[1] == "/opt/nemo-rl" for g in groups),
                                 "mounting a checkout over /opt/nemo-rl trips the fingerprint check (nemo_rl/__init__.py:105)")
                self.assertNotIn("-it", prefix)
                self.assertEqual(set(overrides), expected_overrides(profile))
                self.assertEqual(s["cwd"], GOLDEN["cwd"]["docker"])
                self.assertEqual(s["samples_per_position"], GOLDEN["profiles"][profile]["samples_per_position"])


class TestSbatchLauncher(unittest.TestCase):
    def test_profiles(self):
        for profile in ("smoke", "long"):
            with self.subTest(profile=profile):
                argv = spec("sbatch", profile)["argv"]
                self.assertEqual(argv[:2], ["sbatch", "--no-requeue"])
                self.assertNotIn("--requeue", argv)
                export = [a for a in argv if a.startswith("--export=")]
                self.assertEqual(len(export), 1)
                fields = dict(kv.split("=", 1) for kv in export[0][len("--export="):].split(",")[1:])
                self.assertEqual(export[0].split(",")[0], "--export=ALL")
                command = shlex.split(fields["COMMAND"])
                local = spec("local", profile)["argv"]
                self.assertEqual(command, local, "COMMAND is the local command")
                self.assertTrue(argv[-1].endswith("/ray.sub"))


class TestBaseModelPassThrough(unittest.TestCase):
    """F-17: the training stage starts from the same base-model directory the gate hashes,
    otherwise lineage names the wrong parent. `launch_spec` takes the path explicitly and emits
    `policy.model_name=<path>` once, for every launcher and profile."""

    def test_explicit_parameter(self):
        import inspect

        from gatedtrain.backends.nemo_rl import launch_spec

        self.assertIn("base_model", inspect.signature(launch_spec).parameters)

    def test_passed_through(self):
        from gatedtrain.backends.nemo_rl import launch_spec

        for launcher in ("local", "docker", "sbatch"):
            for profile in ("smoke", "long"):
                for path in ("/work/tierb/base-model", "/data/other/base"):
                    with self.subTest(launcher=launcher, profile=profile, path=path):
                        s = launch_spec(STAGE, nemo_rl_dir=ARGS["nemo_rl_dir"], launcher=launcher,
                                        out_dir=ARGS["out_dir"], export_dir=ARGS["export_dir"], profile=profile,
                                        image=ARGS["image"], work_dir=ARGS["work_dir"], hf_cache=ARGS["hf_cache"],
                                        base_model=path)
                        argv = s["argv"]
                        if launcher == "sbatch":
                            export = [a for a in argv if a.startswith("--export=")][0]
                            argv = shlex.split(export.split("COMMAND=", 1)[1].split(",CONTAINER=", 1)[0])
                        hits = [a for a in argv if a.startswith("policy.model_name=")]
                        self.assertEqual(hits, [f"policy.model_name={path}"])


class TestGenerationsFromConfig(unittest.TestCase):
    """G-7, draft section 2 item 6: the count comes from the config, not a constant."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-gens-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_from_checkpoint_config(self):
        from gatedtrain.backends.nemo_rl import samples_per_position_from_checkpoint

        for gens in (4, 8):
            with self.subTest(gens=gens):
                root = build_stage_out(self.tmp / f"g{gens}", gens=gens, steps=5, save_period=5)
                self.assertEqual(samples_per_position_from_checkpoint(root / "checkpoints" / "step_5"), gens)

    def test_constant_removed(self):
        import gatedtrain.backends.nemo_rl as mod

        self.assertFalse(hasattr(mod, "NUM_GENERATIONS_PER_PROMPT"), "draft section 2 item 6")

    def test_unknown_profile(self):
        with self.assertRaises(ValueError):
            spec("local", "huge")


if __name__ == "__main__":
    unittest.main()
