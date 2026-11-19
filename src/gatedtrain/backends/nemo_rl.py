"""Tier B trainer backend `nemo_rl` (MVP-SPEC section 4; TIER-B-SPEC H-5).

Builds, but never runs, the launch command for one NeMo RL recipe on
unmodified NeMo RL at 4f39fd8. Citations are file:line at 4f39fd8.

- **Recipe:** `examples/configs/grpo_smoke.yaml`. It layers on
  `grpo_math_1B.yaml` (grpo_smoke.yaml:6), whose `cluster.gpus_per_node: 1`
  and `num_nodes: 1` are at grpo_math_1B.yaml:690-692. Model
  `Qwen/Qwen3-0.6B` (grpo_smoke.yaml:15).
- **Launch script:** `examples/run_grpo.py` with `--config` plus Hydra
  overrides (run_grpo.py:53-63, :86-87). Hydra struct mode is on
  (nemo_rl/utils/config.py:188), so keys the recipe lacks need `++`.

Overrides (`COMMON_OVERRIDES`):

| Override | Why |
|---|---|
| `data_plane.enabled=true` | selects `grpo_train_sync` (nemo_rl/data_plane/factory.py:51-55), whose outputs the ledger reads (grpo_sync.py:1323, :1353, :1407-1414, :1458-1460) |
| `data.shuffle=false` | file order (grpo_math_1B.yaml:617; StatefulDataLoader, grpo.py:883-890) |
| `checkpointing.enabled=true` | the smoke recipe turns checkpoints off (grpo_smoke.yaml:12-13); `consumed_samples` lives in them (grpo_sync.py:1353) |
| `checkpointing.keep_top_k=null` | keep every cut for the ledger (checkpoint.py:234 `NotRequired[int \\| None]`; deletion at :590-631) |
| `checkpointing.checkpoint_dir`, `logger.log_dir` | the stage's output directory (grpo_math_1B.yaml:138, :665) |
| `data.train.dataset_name=ResponseDataset` | local JSONL loader (response_datasets/__init__.py:99; response_dataset.py:32-61) |
| `++data.train.data_path`, `++data.train.input_key`, `++data.train.output_key` | ResponseDataset arguments absent from the recipe, so `++` (response_dataset.py:42-48) |
| `data.train.split_validation_size=0` | no validation split reorders rows (response_dataset.py:76) |
| `data.default.prompt_file` | NeMo Evaluator's GSM8K prompt, written by `export-gsm8k` (G-15; processors.py:574-576) |
| `policy.tokenizer.chat_template_kwargs={enable_thinking:false}` | Qwen3 thinking off (G-15; grpo_math_1B.yaml:158; nemo_rl/algorithms/utils.py:465-496) |
| `policy.automodel_cfg.checkpoint.save_consolidated=final` | HF safetensors under `step_<last>/policy/weights/model/consolidated` for serving (grpo_math_1B.yaml:178-181; Automodel config.py:64 `FINAL = "final"`) |

Profiles (`PROFILES`): `smoke` keeps the recipe's 10 steps of 2 x 4
(grpo_smoke.yaml:8-10); `long` is 150 steps of 32 x 8 with a global batch of
256 (an estimate, not a NeMo RL recipe; docs/tier-b-runbook.md step 7, the long run).
`samples_per_position` is the profile's `grpo.num_generations_per_prompt`;
after a run, read it from the checkpoint with
`samples_per_position_from_checkpoint` (utils/checkpoint.py:414-419), never
from a constant (interim rule G-7).

Launchers:

- `local`: absolute script and recipe, run with cwd `nemo_rl_dir`, because the
  recipe has relative paths (grpo_math_1B.yaml:632);
- `docker`: `docker run` with the documented flags
  (docs/about/installation.md:21-37), `nofile=65535` for Ray (ray.sub:161-182),
  `TORCH_CUDA_ARCH_LIST=8.9` and `FLASHINFER_CUDA_ARCH_LIST=8.9` because the
  image targets 9.0 and 10.0 (Dockerfile:231), the image's own source at
  `/opt/nemo-rl` with no checkout mounted over it (the fingerprint check,
  nemo_rl/__init__.py:105), and `-w /opt/nemo-rl`;
- `sbatch`: `sbatch --no-requeue` around NeMo RL's `ray.sub`, driven by
  COMMAND, CONTAINER and MOUNTS (ray.sub:110-112; docs/cluster.md:10-24),
  with COMMAND equal to the local argv. R9-OI-1
  exposure: an administrator or an automated health-check service can still requeue the job.

Outputs are tagged `unpromotable`. The nemo_gym-responses path writes no step
log (grpo_sync.py:1428) and is unsupported.
"""

from __future__ import annotations

import shlex
from pathlib import Path

import yaml

NEMO_RL_COMMIT = "4f39fd8"
RECIPE = "examples/configs/grpo_smoke.yaml"
LAUNCH_SCRIPT = "examples/run_grpo.py"
CONTAINER_SRC = "/opt/nemo-rl"
TAGS = ("prototype", "unpromotable")
LAUNCHERS = ("local", "docker", "sbatch")

# Documented defaults for the keywords added in TIER-B-SPEC H-5.
DEFAULT_PROFILE = "smoke"
DEFAULT_IMAGE = "nemo-rl:4f39fd8"
DEFAULT_WORK_DIR = "/data/gatedtrain-tierb"       # host directory mounted at /work by the docker launcher
DEFAULT_HF_CACHE = "/data/hf-cache"        # host Hugging Face cache mounted at /hf_home
DEFAULT_SHM_SIZE = "32g"                   # keep at or below host RAM (docs/about/installation.md:25 uses 64g)

COMMON_OVERRIDES = (
    "data_plane.enabled=true",
    "data.shuffle=false",
    "checkpointing.enabled=true",
    "checkpointing.keep_top_k=null",
    "checkpointing.checkpoint_dir={out}/checkpoints",
    "logger.log_dir={out}/logs",
    "data.train.dataset_name=ResponseDataset",
    "++data.train.data_path={export}/train.jsonl",
    "++data.train.input_key=question",
    "++data.train.output_key=answer",
    "data.train.split_validation_size=0",
    "data.default.prompt_file={export}/nel_gsm8k_prompt.txt",
    "policy.tokenizer.chat_template_kwargs={{enable_thinking:false}}",
    "policy.automodel_cfg.checkpoint.save_consolidated=final",
)

PROFILES = {
    "smoke": {"max_num_steps": 10, "save_period": 5, "num_prompts_per_step": 2, "num_generations_per_prompt": 4,
              "extra": ()},
    "long": {"max_num_steps": 150, "save_period": 25, "num_prompts_per_step": 32, "num_generations_per_prompt": 8,
             "extra": ("policy.train_global_batch_size=256", "policy.train_micro_batch_size=8")},
}


def overrides(*, out_dir: str, export_dir: str, profile: str, base_model: str | None = None) -> list[str]:
    """Hydra overrides for run_grpo.py: the common set, `policy.model_name` (F-17) and the profile's."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; use one of {sorted(PROFILES)}")
    p = PROFILES[profile]
    out = [o.format(out=out_dir, export=export_dir) for o in COMMON_OVERRIDES]
    if base_model is not None:
        # F-17: train from the base-model directory whose manifest the gate records. The key is in
        # the recipe (grpo_math_1B.yaml:155; grpo_smoke.yaml:15); the tokenizer follows it
        # (`name: ${policy.model_name}`, grpo_math_1B.yaml:157). NeMo RL loads it with from_pretrained
        # (nemo_rl/models/automodel/setup.py:374-375, :941-942) and passes it to vLLM (grpo.py:1426).
        out.append(f"policy.model_name={base_model}")
    out += [f"grpo.max_num_steps={p['max_num_steps']}", f"checkpointing.save_period={p['save_period']}",
            f"grpo.num_prompts_per_step={p['num_prompts_per_step']}",
            f"grpo.num_generations_per_prompt={p['num_generations_per_prompt']}", *p["extra"]]
    return out


def launch_spec(stage: dict, *, nemo_rl_dir: str, launcher: str, out_dir: str | None = None,
                export_dir: str = "/work/export", profile: str = DEFAULT_PROFILE, image: str = DEFAULT_IMAGE,
                work_dir: str = DEFAULT_WORK_DIR, hf_cache: str = DEFAULT_HF_CACHE, base_model: str | None = None,
                account: str = "<account>", partition: str = "<partition>", time_limit: str = "01:00:00") -> dict:
    """{"argv", "cwd", "samples_per_position"} for one Tier B stage. Never executed by gatedtrain.

    `out_dir` defaults to `<nemo_rl_dir>/results/gatedtrain/<stage id>`. For the docker launcher,
    `out_dir`, `export_dir` and `base_model` are paths inside the container (under /work).
    `base_model` is the spec's `base` directory as the trainer sees it; it becomes
    `policy.model_name` (F-17). Pass it for every gated run. `None`, the default kept for older
    callers, leaves the recipe's `Qwen/Qwen3-0.6B`, and the gate's baseline then is not the
    trained parent.
    """
    if launcher not in LAUNCHERS:
        raise ValueError(f"unknown launcher {launcher!r}; use one of {LAUNCHERS} (NeMo-Run is not supported)")
    stage_id = str(stage.get("id", "stage"))
    out = out_dir or f"{nemo_rl_dir}/results/gatedtrain/{stage_id}"
    ov = overrides(out_dir=out, export_dir=export_dir, profile=profile, base_model=base_model)
    spp = PROFILES[profile]["num_generations_per_prompt"]
    local = ["uv", "run", "python", f"{nemo_rl_dir}/{LAUNCH_SCRIPT}", "--config", f"{nemo_rl_dir}/{RECIPE}", *ov]
    if launcher == "local":
        return {"argv": local, "cwd": nemo_rl_dir, "samples_per_position": spp}
    if launcher == "docker":
        # HF_HUB_OFFLINE=1 (F-22, interim rule G-20): after F-17 the base model (policy.model_name) and the
        # data (export/) are local, so the training container needs no Hub access and must not use it.
        flags = [
            "--rm", "--runtime=nvidia", "--gpus", "all", f"--shm-size={DEFAULT_SHM_SIZE}",
            "--ulimit", "memlock=-1", "--ulimit", "nofile=65535:65535",
            "-e", "TORCH_CUDA_ARCH_LIST=8.9", "-e", "FLASHINFER_CUDA_ARCH_LIST=8.9",
            "-e", "HF_HOME=/hf_home", "-e", "HF_HUB_OFFLINE=1", "-v", f"{hf_cache}:/hf_home", "-v", f"{work_dir}:/work",
            "-w", CONTAINER_SRC,
        ]
        program = ["uv", "run", "python", LAUNCH_SCRIPT, "--config", RECIPE]
        return {"argv": ["docker", "run", *flags, image, *program, *ov], "cwd": CONTAINER_SRC,
                "samples_per_position": spp}
    command = shlex.join(local)
    mounts = f"{nemo_rl_dir}:{nemo_rl_dir}"
    if "," in command + image + mounts:
        raise ValueError("a comma in the command, image or mounts breaks sbatch --export")
    env = f"ALL,COMMAND={command},CONTAINER={image},MOUNTS={mounts}"
    argv = ["sbatch", "--no-requeue", "--nodes=1", "--gres=gpu:1", f"--account={account}",
            f"--partition={partition}", f"--time={time_limit}", f"--job-name=gatedtrain-{stage_id}-unpromotable",
            f"--export={env}", f"{nemo_rl_dir}/ray.sub"]
    return {"argv": argv, "cwd": nemo_rl_dir, "samples_per_position": spp}


def build_command(stage: dict, *, nemo_rl_dir: str, launcher: str, **kwargs) -> list[str]:
    """argv only; see `launch_spec` for the keywords and their defaults."""
    return launch_spec(stage, nemo_rl_dir=nemo_rl_dir, launcher=launcher, **kwargs)["argv"]


def samples_per_position_from_checkpoint(step_dir: Path) -> int:
    """`grpo.num_generations_per_prompt` from `step_<N>/config.yaml`, which NeMo RL writes from the
    run config (utils/checkpoint.py:414-419). Lines per loader position for the ledger (G-7)."""
    cfg = yaml.safe_load((Path(step_dir) / "config.yaml").read_text(encoding="utf-8"))
    try:
        n = cfg["grpo"]["num_generations_per_prompt"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{step_dir}/config.yaml has no grpo.num_generations_per_prompt") from exc
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError(f"{step_dir}/config.yaml: grpo.num_generations_per_prompt = {n!r}")
    return n
