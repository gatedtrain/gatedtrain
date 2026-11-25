"""Citations: every key and flag the Tier B helpers emit exists at the pins (TIER-B-SPEC section 3).

Reads the pinned clones (read only); each class skips with a clear reason when its clone is
absent. `TestReference*` checks the golden files against the clones and passes
without gatedtrain; `TestImplementation*` checks what gatedtrain actually emits.

- NeMo RL 4f39fd8: the recipe examples/configs/grpo_smoke.yaml layered on grpo_math_1B.yaml
  (grpo_smoke.yaml:6); Hydra struct mode needs `++` for keys the recipe lacks
  (nemo_rl/utils/config.py:188); ResponseDataset's arguments (response_dataset.py:32-43);
  Automodel's `save_consolidated` modes (Automodel components/checkpoint/config.py:60-85);
  docker flags (docs/about/installation.md:21-37; ray.sub nofile 65535).
- NeMo Evaluator ea35ba3: the config schema (config/eval_config.py:59-66; config/services.py:25-44,
  :82-105, :240-259; config/benchmarks.py:34-60; config/solvers.py SimpleSolver; config/output.py:41-49),
  the CLI (cli/eval.py:71-88), the results path (orchestration/orchestrator.py:1046-1047) and the
  snapshot file name (config/snapshot.py:39).
"""

from __future__ import annotations

import ast
import inspect
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from _support import NEMO_EVALUATOR_DIR, NEMO_RL_DIR, SKEL_ROOT, describe, run_cli
from _tierb import TIERB

NRL = NEMO_RL_DIR
NEL = NEMO_EVALUATOR_DIR / "src" / "nemo_evaluator"
GOLDEN = TIERB / "golden"


def need(path: Path) -> Path:
    if not path.exists():
        raise unittest.SkipTest(f"pinned clone not available: {path}")
    return path


def merged_recipe() -> dict:
    def merge(a, b):
        out = dict(a)
        for k, v in b.items():
            out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        return out
    cfg_dir = need(NRL / "examples" / "configs")
    smoke = yaml.safe_load((cfg_dir / "grpo_smoke.yaml").read_text(encoding="utf-8"))
    assert smoke["defaults"] == "grpo_math_1B.yaml"
    base = yaml.safe_load((cfg_dir / smoke.pop("defaults")).read_text(encoding="utf-8"))
    return merge(base, smoke)


def has_key(cfg: dict, dotted: str) -> bool:
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return True


def class_fields(path: Path, name: str) -> set[str]:
    tree = ast.parse(need(path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return {n.target.id for n in node.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
    raise AssertionError(f"class {name} not found in {path}")


def check_overrides(test: unittest.TestCase, overrides: list[str]) -> None:
    recipe = merged_recipe()
    rd = (NRL / "nemo_rl" / "data" / "datasets" / "response_datasets" / "response_dataset.py").read_text(encoding="utf-8")
    init_args = set(re.findall(r"^\s{8}(\w+):", rd.split("def __init__", 1)[1].split("):", 1)[0], re.M))
    automodel_cfg = (NRL / "3rdparty" / "Automodel-workspace" / "Automodel" / "nemo_automodel" / "components" /
                     "checkpoint" / "config.py")
    for o in overrides:
        with test.subTest(override=o):
            key, value = o.split("=", 1)
            if key.startswith("++"):
                key = key[2:]
                test.assertFalse(has_key(recipe, key), f"{key} exists in the recipe; `++` is not needed")
                test.assertEqual(key.rsplit(".", 1)[0], "data.train")
                test.assertIn(key.rsplit(".", 1)[1], init_args, "ResponseDataset argument (response_dataset.py:32-43)")
            else:
                test.assertTrue(has_key(recipe, key), f"{key} is not in grpo_smoke.yaml + grpo_math_1B.yaml; "
                                                      "struct mode (utils/config.py:188) needs `++`")
            if key == "policy.automodel_cfg.checkpoint.save_consolidated":
                test.assertIn(f'= "{value}"', need(automodel_cfg).read_text(encoding="utf-8"))
            if key == "data.train.dataset_name":
                init = (NRL / "nemo_rl" / "data" / "datasets" / "response_datasets" / "__init__.py").read_text()
                test.assertIn(f'"{value}":', init)
            if key == "checkpointing.keep_top_k":
                test.assertEqual(value, "null")
                ck = (NRL / "nemo_rl" / "utils" / "checkpoint.py").read_text(encoding="utf-8")
                test.assertIn("keep_top_k: NotRequired[int | None]", ck)


def check_docker(test: unittest.TestCase, argv: list[str]) -> None:
    doc = need(NRL / "docs" / "about" / "installation.md").read_text(encoding="utf-8")
    ray_sub = need(NRL / "ray.sub").read_text(encoding="utf-8")
    dockerfile = need(NRL / "docker" / "Dockerfile").read_text(encoding="utf-8")
    joined = " ".join(argv)
    for flag in ("--runtime=nvidia", "--gpus all", "--ulimit memlock=-1", "-w /opt/nemo-rl", "-e HF_HOME=/hf_home"):
        test.assertIn(flag, joined)
        test.assertIn(flag.replace("-e HF_HOME=/hf_home", "-e HF_HOME=/hf_home"), doc)
    test.assertIn("--shm-size=", joined)
    test.assertIn("--shm-size=", doc)
    test.assertIn("--ulimit nofile=65535:65535", joined)
    test.assertIn("65535", ray_sub)
    test.assertIn("TORCH_CUDA_ARCH_LIST", dockerfile)


NEL_SCHEMA = {
    (): ("config/eval_config.py", "EvalConfig"),
    ("services", "*"): ("config/services.py", "ExternalApiService"),
    ("services", "*", "generation"): ("config/services.py", "GenerationConfig"),
    ("services", "*", "proxy"): ("config/services.py", "ProxyConfig"),
    ("benchmarks", "#"): ("config/benchmarks.py", "BenchmarkConfig"),
    ("benchmarks", "#", "solver"): ("config/solvers.py", "SimpleSolver"),
    ("output",): ("config/output.py", "OutputConfig"),
}


def check_nel_config(test: unittest.TestCase, cfg: dict) -> None:
    def walk(node, path):
        if isinstance(node, list):
            for item in node:
                walk(item, path + ("#",))
            return
        if not isinstance(node, dict):
            return
        if path in NEL_SCHEMA:
            rel, cls = NEL_SCHEMA[path]
            fields = class_fields(NEL / rel, cls)
            test.assertLessEqual(set(node), fields, f"{'.'.join(path) or '<root>'}: keys not in {cls} ({rel})")
            for k, v in node.items():
                if k == "extra_body":
                    continue
                walk(v, path + (k,))
        elif path == ("services",):
            for v in node.values():
                walk(v, path + ("*",))
    walk(cfg, ())
    services = need(NEL / "config" / "services.py").read_text(encoding="utf-8")
    proto = re.search(r'^Protocol = Literal\[(.*)\]', services, re.M).group(1)
    test.assertIn(f'"{cfg["services"]["model"]["protocol"]}"', proto)
    test.assertEqual(cfg["services"]["model"]["type"], "api")
    test.assertTrue(cfg["services"]["model"]["url"].endswith("/chat/completions"),
                    "ExternalApiService.url is the full URL (services.py:240-245)")


class TestReferenceNemoRL(unittest.TestCase):
    def test_golden_overrides_exist_at_pin(self):
        golden = json.loads((GOLDEN / "build_command.json").read_text(encoding="utf-8"))
        overrides = golden["common_overrides"] + [o for p in golden["profiles"].values() for o in p["overrides"]]
        check_overrides(self, overrides)

    def test_paths_exist(self):
        for rel in ("examples/run_grpo.py", "examples/configs/grpo_smoke.yaml", "ray.sub"):
            self.assertTrue(need(NRL / rel).is_file(), rel)

    def test_golden_docker_flags_documented(self):
        golden = json.loads((GOLDEN / "build_command.json").read_text(encoding="utf-8"))
        flags = [t for g in golden["docker_flag_groups"] for t in g] + ["--shm-size=32g"]
        check_docker(self, flags)

    def test_answer_extraction_at_pin(self):
        src = need(NRL / "nemo_rl" / "data" / "datasets" / "response_datasets" / "gsm8k.py").read_text(encoding="utf-8")
        self.assertIn('return text.split("####")[1].strip()', src)


class TestReferenceBaseModel(unittest.TestCase):
    """F-17: `policy.model_name` is in the recipe chain and NeMo RL loads it with Hugging Face
    `from_pretrained`, which accepts a local directory."""

    def test_key_in_recipe_chain(self):
        cfg_dir = need(NRL / "examples" / "configs")
        base = yaml.safe_load((cfg_dir / "grpo_math_1B.yaml").read_text(encoding="utf-8"))
        smoke = yaml.safe_load((cfg_dir / "grpo_smoke.yaml").read_text(encoding="utf-8"))
        self.assertIn("model_name", base["policy"])                 # grpo_math_1B.yaml:155
        self.assertEqual(smoke["policy"]["model_name"], "Qwen/Qwen3-0.6B")   # grpo_smoke.yaml:15
        self.assertTrue(has_key(merged_recipe(), "policy.model_name"))
        # The tokenizer follows the model unless overridden: `name: ${policy.model_name}` (grpo_math_1B.yaml:157).
        self.assertEqual(base["policy"]["tokenizer"]["name"], "${policy.model_name}")

    def test_loaded_with_from_pretrained(self):
        setup = need(NRL / "nemo_rl" / "models" / "automodel" / "setup.py").read_text(encoding="utf-8")
        # setup.py:343, :764 read config["model_name"]; :374-375 AutoConfig.from_pretrained(model_name, ...);
        # :941-942 model_class.from_pretrained(model_name, ...).
        self.assertIn('model_name = config["model_name"]', setup)
        self.assertRegex(setup, r"AutoConfig\.from_pretrained\(\s*model_name,")
        self.assertRegex(setup, r"model_class\.from_pretrained\(\s*model_name,")
        grpo = need(NRL / "nemo_rl" / "algorithms" / "grpo.py").read_text(encoding="utf-8")
        # grpo.py:1426: the vLLM generation worker gets the same model (vllm_worker.py:505, :775).
        self.assertIn('generation_config["model_name"] = policy_config["model_name"]', grpo)

    def test_local_directory_documented(self):
        doc = need(NRL / "docs" / "guides" / "grpo-deepscaler.md").read_text(encoding="utf-8")
        self.assertRegex(doc, r"policy\.model_name=/\S+")      # grpo-deepscaler.md:12-13


class TestReferenceHfOffline(unittest.TestCase):
    """F-22: `HF_HUB_OFFLINE=1` is honoured by the huggingface_hub version NeMo RL locks at 4f39fd8."""

    def locked_version(self) -> str:
        lock = need(NRL / "uv.lock").read_text(encoding="utf-8")
        m = re.search(r'^name = "huggingface-hub"\nversion = "([^"]+)"', lock, re.M)
        self.assertIsNotNone(m, "huggingface-hub entry in uv.lock")
        return m.group(1)

    def test_lock_entry(self):
        # uv.lock:2487-2488 at 4f39fd8: name = "huggingface-hub", version = "1.32.0".
        self.assertEqual(self.locked_version(), "1.32.0")

    def test_nemo_rl_documents_the_variable(self):
        doc = need(NRL / "docs" / "guides" / "grpo-audio.md").read_text(encoding="utf-8")
        self.assertIn("HF_HUB_OFFLINE=1", doc)                     # grpo-audio.md:153

    def test_constant_in_locked_version(self):
        """Reads an unpacked copy of the locked version from the local uv cache, if one exists;
        the file is checked against its own dist-info RECORD hash."""
        import base64
        import glob
        import hashlib

        ver = self.locked_version()
        hits = sorted(glob.glob(str(Path.home() / ".cache" / "uv" / "archive-v0" / "*" /
                                    f"huggingface_hub-{ver}.dist-info")))
        if not hits:
            self.skipTest(f"cannot verify: no local copy of huggingface_hub {ver} (not vendored in the clone)")
        dist = Path(hits[0])
        const = dist.parent / "huggingface_hub" / "constants.py"
        record = (dist / "RECORD").read_text(encoding="utf-8")
        want = re.search(r"^huggingface_hub/constants\.py,sha256=([^,]+),", record, re.M).group(1)
        got = base64.urlsafe_b64encode(hashlib.sha256(const.read_bytes()).digest()).rstrip(b"=").decode()
        self.assertEqual(got, want, "constants.py matches the 1.32.0 RECORD")
        text = const.read_text(encoding="utf-8")
        self.assertIn('HF_HUB_OFFLINE = _is_true(os.environ.get("HF_HUB_OFFLINE")', text)   # constants.py:194
        self.assertIn("return HF_HUB_OFFLINE", text)                                            # is_offline_mode()


class TestReferenceNemoEvaluator(unittest.TestCase):
    def test_golden_configs_match_schema(self):
        for r in (0, 1):
            with self.subTest(repeat=r):
                check_nel_config(self, yaml.safe_load((GOLDEN / f"nel-gsm8k-rep{r}.yaml").read_text(encoding="utf-8")))

    def test_cli_and_output_paths(self):
        cli = need(NEL / "cli" / "eval.py").read_text(encoding="utf-8")
        self.assertIn('@eval_cmd.command("run")', cli)
        self.assertIn('"--override", "-O"', cli)
        self.assertIn('"--output-dir", "-o"', cli)
        orch = need(NEL / "orchestration" / "orchestrator.py").read_text(encoding="utf-8")
        self.assertIn("task_dir = output_dir / _safe_name(bench_name)", orch)
        art = need(NEL / "engine" / "artifacts.py").read_text(encoding="utf-8")
        self.assertIn('rp = out / "results.jsonl"', art)
        snap = need(NEL / "config" / "snapshot.py").read_text(encoding="utf-8")
        self.assertIn('SNAPSHOT_FILENAME = "full_config.yaml"', snap)

    def test_no_dataset_revision_key(self):
        """G-17: the hf:// loader takes split and config only, so no revision can be pinned in the config."""
        custom = need(NEL / "environments" / "custom.py").read_text(encoding="utf-8")
        loader = custom.split("def _load_hf", 1)[1].split("\ndef ", 1)[0]
        self.assertNotIn("revision", loader)


class TestImplementationEmitsPinnedKeys(unittest.TestCase):
    def test_build_command_overrides(self):
        from gatedtrain.backends.nemo_rl import launch_spec

        golden = json.loads((GOLDEN / "build_command.json").read_text(encoding="utf-8"))
        a = golden["args"]
        stage = {"id": "rl", "uses": "rl.nemo_rl@0.1"}
        for profile in ("smoke", "long"):
            argv = launch_spec(stage, nemo_rl_dir=a["nemo_rl_dir"], launcher="local", out_dir=a["out_dir"],
                               export_dir=a["export_dir"], profile=profile, image=a["image"], work_dir=a["work_dir"],
                               hf_cache=a["hf_cache"], cuda_arch=a["cuda_arch"], base_model=a["base_model"])["argv"]
            overrides = [x for x in argv if "=" in x and not x.startswith("-")]
            self.assertTrue(overrides)
            check_overrides(self, overrides)
        docker = launch_spec(stage, nemo_rl_dir=a["nemo_rl_dir"], launcher="docker", out_dir=a["out_dir"],
                             export_dir=a["export_dir"], profile="smoke", image=a["image"], work_dir=a["work_dir"],
                             hf_cache=a["hf_cache"], cuda_arch=a["cuda_arch"], base_model=a["base_model"])["argv"]
        check_docker(self, docker[:docker.index(a["image"])])

    def test_nel_config_keys(self):
        tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-cite-"))
        try:
            reg = tmp / "registration.yaml"
            reg.write_text((TIERB / "registration.template.yaml").read_text(encoding="utf-8")
                           .replace("{prompt_set_hash}", "sha256:" + "0" * 64), encoding="utf-8")
            proc = run_cli(["nel-config", "--registration", str(reg), "--model-url", "http://localhost:8000/v1",
                            "--model-id", "candidate", "--eval-root", "/work/out/eval/candidate", "--out",
                            str(tmp / "cfg")], cwd=SKEL_ROOT, timeout=60)
            self.assertEqual(proc.returncode, 0, describe(proc))
            for p in sorted((tmp / "cfg").glob("*.yaml")):
                with self.subTest(file=p.name):
                    check_nel_config(self, yaml.safe_load(p.read_text(encoding="utf-8")))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
