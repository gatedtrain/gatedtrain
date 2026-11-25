"""One name, one place for every constant that used to be copied and the pinned values of
the trust constants (class A): a change to either fails here, so it cannot go unnoticed.
"""

from __future__ import annotations

import inspect
import math
import unittest
from pathlib import Path
from unittest import mock

from gatedtrain import TAGS, cli, gates, items, registration, tierb
from gatedtrain.ainative import alpha_investing as ai
from gatedtrain.ainative import batch, runtime, scorers, sealed_eval
from gatedtrain.ainative.keeper import inference, protocol, texts
from gatedtrain.ainative.trust import FixedAlphaLedger
from gatedtrain.backends import nemo_evaluator, nemo_rl
from gatedtrain.ainative import backend, hostsetup, project
from gatedtrain.ext import conformance, executor, launch, smoke
from gatedtrain.families import nemo_rl as family


class TrustConstantsArePinned(unittest.TestCase):
    """Class A: these must not be tunable and must not drift. Each is the value of design 05 (AN-15, AN-6, AN-18)."""

    def test_alpha_investing(self) -> None:
        self.assertEqual(ai.TARGET_ALPHA, 0.05)
        self.assertEqual(ai.INITIAL_WEALTH, 0.0475)
        self.assertEqual(ai.PAYOUT, 0.05)
        self.assertEqual(ai.MIN_BID, 0.005)
        self.assertEqual(ai.PLANNING_ALPHA, 0.02375)

    def test_sealed_set_planning(self) -> None:
        self.assertEqual(sealed_eval.DEFAULT_MDE, 0.10)
        self.assertEqual(sealed_eval.DEFAULT_POWER, 0.8)
        self.assertEqual(sealed_eval.PILOT_SIGMA, math.sqrt(0.5))
        self.assertEqual(sealed_eval.MAX_SEALED_FRACTION, 0.5)

    def test_statistical_and_refusal_constants(self) -> None:
        """The gate's numerical limits and the registration's refusal rules (design 05 §3, REG-*, AN-26): none is a
        parameter of any entry point."""
        from gatedtrain import canonical, constraints
        from gatedtrain.ainative import trust_stages
        self.assertEqual(canonical.MAX_SAFE_INT, 2 ** 53 - 1)
        self.assertEqual((registration.DEFAULT_MIN_PAIRED, registration.DEFAULT_MAX_UNPAIRED), (10, 0))
        self.assertEqual(registration.SCORER_ALLOWLIST, ("numeric_match",))
        self.assertEqual(constraints.N_MAX, 1_000_000)
        self.assertEqual(gates.ZERO_WIDTH_REL_TOL, 1e-12)
        self.assertEqual(gates.RATIO_PART_LIMIT, 1e12)
        self.assertEqual(sealed_eval.CONTROL_SEED_LIMIT, 2 ** 31)
        self.assertEqual(trust_stages.MAX_MARGIN, 0.5)

    def test_smoke_verdict_thresholds(self) -> None:
        """EXT-29: the verdict is recomputed from the record with these rules."""
        self.assertEqual((smoke.MEAN_MARGIN, smoke.MODE_SHARE, smoke.OFF_BOUND_SHARE, smoke.OFF_BOUND_MIN),
                         (0.05, 0.5, 0.25, 2))

    def test_keeper_protocol_limits(self) -> None:
        self.assertEqual(protocol.MAX_REQUEST_BYTES, 96 * 1024 * 1024)
        self.assertEqual(protocol.MAX_RESPONSE_BYTES, 96 * 1024 * 1024)
        self.assertEqual(batch.MAX_PREDICTIONS_BYTES, 256 * 1024 * 1024)

    def test_alpha_investing_values_are_not_parameters(self) -> None:
        """No public entry point takes the alpha-investing constants as an argument."""
        for fn in (ai.bid, ai.can_bid, ai.settle, ai.replay):
            names = set(inspect.signature(fn).parameters)
            self.assertFalse(names & {"initial_wealth", "payout", "min_bid", "target_alpha"}, fn.__name__)


class SignedTextIsByteIdentical(unittest.TestCase):
    """The guarantee texts are signed into reports and the host script is what an owner reads before running it as root:
    their bytes are those of the release the hashes were taken from (the texts: 0.6.0, unchanged in 0.7.0; the script: 0.7.0)."""

    def sha(self, text: str) -> str:
        import hashlib
        return hashlib.sha256(text.encode()).hexdigest()

    def test_alpha_texts(self) -> None:
        self.assertEqual(self.sha(ai.GUARANTEE), "6ccdb16d2013bfee045ac58823cb7ca6fe485026d2549ee304fa8536990e2560")
        self.assertEqual(self.sha(texts.LIMIT_ALPHA_RULE), "15b7b4a83aa3d27360e5e43fa46163b084e5d16c4e8e8d228f6959c928fef727")

    def test_host_script(self) -> None:
        """The `setup host` script of gatedtrain 0.7.0 with its version number replaced by a placeholder (the script
        prints the version it was made by); the hashes were taken from 0.7.0."""
        import gatedtrain
        from gatedtrain.ainative import hostsetup
        plain = hostsetup.host_script(owner="alice", owner_uid=1234)["script"]
        with_image = hostsetup.host_script(owner="alice", owner_uid=1234,
                                           inference_ref="x/y@sha256:" + "a" * 64)["script"]
        self.assertEqual(self.sha(plain.replace(gatedtrain.__version__, "<version>")), "e91f994a0610174b91c9e458f6d09ec50f0feea038bb8460b90d1990e7fc1949")
        self.assertEqual(self.sha(with_image.replace(gatedtrain.__version__, "<version>")), "0704b38eb3362981a8757dc1020d0fb22e4adbe7310a2608ca9af190b84bcc51")


class AlphaHasOneSource(unittest.TestCase):
    def test_derived_from_the_target(self) -> None:
        self.assertEqual(ai.INITIAL_WEALTH, ai.TARGET_ALPHA * (1 - ai.TARGET_ALPHA))
        self.assertEqual(ai.PAYOUT, ai.TARGET_ALPHA)
        self.assertEqual(ai.PLANNING_ALPHA, ai.INITIAL_WEALTH / 2)

    def test_other_modules_use_the_target(self) -> None:
        self.assertEqual(sealed_eval.DEFAULT_ALPHA, ai.TARGET_ALPHA)
        default = inspect.signature(FixedAlphaLedger.__init__).parameters["alpha"].default
        self.assertEqual(default, ai.TARGET_ALPHA)

    def test_texts_quote_the_target(self) -> None:
        level = f"{ai.TARGET_ALPHA:g}"
        self.assertIn(f"at level {level}", ai.GUARANTEE)
        self.assertIn(f"at level {level} over those attempts", texts.LIMIT_ALPHA_RULE)
        self.assertEqual(level, "0.05")


class TimeoutsAddUp(unittest.TestCase):
    def test_long_operation_timeout_is_the_sum_of_its_parts(self) -> None:
        self.assertEqual(protocol.LONG_OP_TIMEOUT_S,
                         runtime.GPU_WAIT_S + 2 * batch.MAX_GENERATOR_TIME_LIMIT_S + batch.EVAL_SLACK_S)
        self.assertEqual(protocol.LONG_OP_TIMEOUT_S, 3900.0)

    def test_the_cap_and_the_default_are_separate_numbers(self) -> None:
        self.assertEqual(batch.MAX_GENERATOR_TIME_LIMIT_S, 1200)
        self.assertEqual(protocol.LONG_OP_TIMEOUT_S,
                         batch.evaluation_wait_bound(runtime.GPU_WAIT_S, batch.MAX_GENERATOR_TIME_LIMIT_S))
        self.assertEqual(batch.evaluation_wait_bound(900, 100), 900 + 200 + batch.EVAL_SLACK_S)

    def test_the_slack_has_one_home(self) -> None:
        self.assertIs(inference.EVAL_SLACK_S, batch.EVAL_SLACK_S)


class SharedMemoryHasOneSource(unittest.TestCase):
    def test_one_value(self) -> None:
        self.assertEqual(launch.SHM_SIZE, f"--shm-size={launch.SHM_SIZE_VALUE}")
        self.assertEqual(inference.SHM_SIZE, launch.SHM_SIZE_VALUE)
        self.assertEqual(launch.SHM_SIZE_VALUE, "8g")
        default = [f for f in runtime.RunSpec.__dataclass_fields__.values() if f.name == "shm_size"][0].default
        self.assertEqual(default, launch.SHM_SIZE_VALUE)


class NamedOnce(unittest.TestCase):
    def test_gpu_cap(self) -> None:
        self.assertIs(smoke.GPU_CAP_H, launch.GPU_CAP_H)

    def test_evaluator_commit(self) -> None:
        self.assertIs(tierb.NEMO_EVALUATOR_COMMIT, nemo_evaluator.NEMO_EVALUATOR_COMMIT)

    def test_vocabularies(self) -> None:
        self.assertIs(registration.BOOTSTRAP_VARIANTS, gates.BOOTSTRAP_VARIANTS)
        self.assertIs(conformance.DIRECTIONS, items.DIRECTIONS)
        self.assertIs(scorers.DIRECTIONS, items.DIRECTIONS)
        self.assertIs(nemo_rl.TAGS, TAGS)

    def test_rows_per_shard_default_is_the_functions(self) -> None:
        """The command line names no default of its own: without `--rows-per-shard` the function's applies."""
        parser = cli.build_parser()
        args = parser.parse_args(["export-gsm8k", "--out", "x"])
        self.assertIsNone(args.rows_per_shard)
        self.assertEqual(inspect.signature(tierb.export_gsm8k).parameters["rows_per_shard"].default,
                         tierb.DEFAULT_ROWS_PER_SHARD)
        with mock.patch.object(tierb, "export_gsm8k", return_value=Path("out")) as export:
            cli.cmd_export(args)
        self.assertEqual(export.call_args.kwargs["rows_per_shard"], tierb.DEFAULT_ROWS_PER_SHARD)
        args = parser.parse_args(["export-gsm8k", "--out", "x", "--rows-per-shard", "7"])
        with mock.patch.object(tierb, "export_gsm8k", return_value=Path("out")) as export:
            cli.cmd_export(args)
        self.assertEqual(export.call_args.kwargs["rows_per_shard"], 7)


class RootlessSocketHasOneSource(unittest.TestCase):
    def test_the_helper_composes_the_documented_path(self) -> None:
        self.assertEqual(launch.rootless_docker_host(1234), "unix:///run/user/1234/docker.sock")
        self.assertEqual(launch.rootless_docker_host("$KUID"), "unix:///run/user/$KUID/docker.sock")
        self.assertEqual(launch.rootless_docker_host(7, run_user_dir="/x"), "unix:///x/7/docker.sock")

    def test_it_agrees_with_the_path_the_launch_rules_check(self) -> None:
        cfg = launch.HostConfig(host_work="/w", output_root="/o", rootless=True, owner_uid=1234)
        self.assertEqual(launch.rootless_docker_host(1234), "unix://" + launch.rootless_socket(cfg))

    def test_no_module_spells_the_path_itself(self) -> None:
        root = Path(launch.__file__).resolve().parents[1]
        offenders = []
        for path in sorted(root.rglob("*.py")):
            if path.name == "launch.py" or "testing" in path.parts:
                continue
            for n, line in enumerate(path.read_text().splitlines(), 1):
                code = line.split("#")[0]
                if "/run/user" in code and "docker.sock" in code and "<keeper uid>" not in code:
                    offenders.append(f"{path.relative_to(root)}:{n}")
        self.assertEqual(offenders, [])

    def test_the_host_script_is_unchanged(self) -> None:
        script = hostsetup.host_script(owner="alice", owner_uid=1234)["script"]
        self.assertIn("unix:///run/user/1234/docker.sock", script)
        with_image = hostsetup.host_script(
            owner="alice", owner_uid=1234, inference_ref="x/y@sha256:" + "a" * 64)["script"]
        self.assertIn("unix:///run/user/$KUID/docker.sock", with_image)
        self.assertIn("unix:///run/user/${KUID}/docker.sock", with_image)
        self.assertIn("XDG_RUNTIME_DIR=/run/user/$OWNER_UID", script)


class NemoRlPinsAreWrittenOnce(unittest.TestCase):
    def test_the_short_commit_is_the_full_ones_prefix(self) -> None:
        self.assertEqual(family.NEMO_RL_COMMIT_SHORT, family.NEMO_RL_COMMIT[:7])
        self.assertIs(nemo_rl.NEMO_RL_COMMIT, family.NEMO_RL_COMMIT_SHORT)
        self.assertEqual(nemo_rl.DEFAULT_IMAGE, f"nemo-rl:{family.NEMO_RL_COMMIT_SHORT}")
        self.assertEqual(tierb.CHECKPOINT_COMPLETENESS["nemo_rl_commit"], family.NEMO_RL_COMMIT_SHORT)

    def test_the_source_directory_has_one_home(self) -> None:
        self.assertIs(nemo_rl.CONTAINER_SRC, family.NEMO_RL_SRC)
        self.assertEqual(family.NEMO_RL_RUN_OPTIONS[-2:], ("-w", family.NEMO_RL_SRC))


class ExecutorLimitsAgree(unittest.TestCase):
    """Class A (B18: model-written code runs under exactly these limits): the option list is derived from the table."""

    def test_values_are_pinned(self) -> None:
        self.assertEqual(executor.EXECUTOR_LIMITS,
                         {"cpus": 8, "memory": "8g", "memory_swap": "8g", "pids": 256, "nofile": "1024:1024", "core": 0})
        self.assertEqual(executor.EXECUTOR_TMPFS_OPTIONS, "rw,nosuid,nodev,noexec,size=4g")

    def test_the_options_are_the_tables(self) -> None:
        lim = executor.EXECUTOR_LIMITS
        self.assertEqual(executor.EXECUTOR_LIMIT_OPTIONS,
                         ("--cpus", str(lim["cpus"]), "--memory", lim["memory"], "--memory-swap", lim["memory_swap"],
                          "--pids-limit", str(lim["pids"]), "--ulimit", f"nofile={lim['nofile']}",
                          "--ulimit", f"core={lim['core']}"))


class JobDefaultsAreNamed(unittest.TestCase):
    def test_defaults(self) -> None:
        fields = backend.Job.__dataclass_fields__
        self.assertEqual(fields["time_limit_s"].default, backend.DEFAULT_JOB_TIME_LIMIT_S)
        self.assertEqual(fields["max_attempts"].default, backend.DEFAULT_MAX_ATTEMPTS)
        self.assertEqual((backend.DEFAULT_JOB_TIME_LIMIT_S, backend.DEFAULT_MAX_ATTEMPTS), (21600, 3))

    def test_one_poll_interval(self) -> None:
        for cls in (backend.LocalBackend, backend.StageBackend):
            self.assertEqual(inspect.signature(cls.wait).parameters["poll_s"].default, backend.JOB_POLL_S)
        self.assertEqual(runtime.GPU_POLL_S, backend.JOB_POLL_S)


class ProjectFileHasOneName(unittest.TestCase):
    def test_the_name(self) -> None:
        self.assertEqual(project.PROJECT_FILE, "project.yaml")

    def test_no_code_spells_it(self) -> None:
        root = Path(project.__file__).resolve().parents[1]
        offenders = [f"{p.relative_to(root)}" for p in sorted(root.rglob("*.py"))
                     if p.name != "project.py" and '"project.yaml"' in p.read_text()]
        self.assertEqual(offenders, [])
        self.assertEqual(Path(project.__file__).read_text().count('"project.yaml"'), 1)


class TierBLauncherNamesNoMachine(unittest.TestCase):
    """launch_spec carries no host path, no GPU architecture of the author's machine."""

    STAGE = {"id": "rl"}

    def spec(self, **kw):
        return nemo_rl.launch_spec(self.STAGE, nemo_rl_dir="/x/nemo-rl", launcher="docker", **kw)

    def test_the_docker_launcher_needs_the_host_directories(self) -> None:
        with self.assertRaisesRegex(ValueError, "work_dir"):
            self.spec(hf_cache="/c")
        with self.assertRaisesRegex(ValueError, "hf_cache"):
            self.spec(work_dir="/w")

    def test_the_directories_are_the_callers(self) -> None:
        argv = self.spec(work_dir="/w", hf_cache="/c")["argv"]
        self.assertIn("/c:/hf_home", argv)
        self.assertIn("/w:/work", argv)

    def test_no_gpu_architecture_unless_asked(self) -> None:
        argv = self.spec(work_dir="/w", hf_cache="/c")["argv"]
        self.assertFalse([a for a in argv if "CUDA_ARCH_LIST" in a])
        argv = self.spec(work_dir="/w", hf_cache="/c", cuda_arch="9.0")["argv"]
        self.assertIn("TORCH_CUDA_ARCH_LIST=9.0", argv)
        self.assertIn("FLASHINFER_CUDA_ARCH_LIST=9.0", argv)

    def test_shared_memory_is_a_parameter(self) -> None:
        self.assertIn("--shm-size=32g", self.spec(work_dir="/w", hf_cache="/c")["argv"])
        self.assertIn("--shm-size=64g", self.spec(work_dir="/w", hf_cache="/c", shm_size="64g")["argv"])

    def test_other_launchers_need_no_host_directory(self) -> None:
        nemo_rl.launch_spec(self.STAGE, nemo_rl_dir="/x/nemo-rl", launcher="local")
        nemo_rl.launch_spec(self.STAGE, nemo_rl_dir="/x/nemo-rl", launcher="sbatch", image="img")

    def test_no_machine_default_is_left(self) -> None:
        for name in ("DEFAULT_WORK_DIR", "DEFAULT_HF_CACHE"):
            self.assertFalse(hasattr(nemo_rl, name), name)


if __name__ == "__main__":
    unittest.main()
