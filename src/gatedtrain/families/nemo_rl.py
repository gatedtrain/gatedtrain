"""NeMo RL family: the pinned framework, its images and the output layout NeMo RL and the recipes' cut writers share.

Used by the recipes that train with NeMo RL (CAD SFT and GRPO, VLM GRPO, agentic GRPO). Values are the pins of the
Tier B NeMo RL image (nemo-rl 4f39fd8) and of its uid-1000 derivative.
"""

from __future__ import annotations

import re

NEMO_RL_COMMIT = "4f39fd86289a2ffd5aaf5d202702f7fd7185c924"  # Tier B image nemo-rl:4f39fd8
NEMO_RL_IMAGE = "sha256:8cd5b714ea91ec7ea9a976c75b0ae6f6932140cf7a8aea803100de6331a2113a"
# The NeMo RL images a recipe may run in, each pinned by its image ID: the Tier B image, and gatedtrain-a6-nemo-rl:1 (the Tier
# B image's layers plus one that hands the ten vLLM files NeMo RL patches at start-up, and their directories, to uid
# 1000; a trainer running as 1000 has NeMo RL write `<file>.patch_lock` beside them).
NEMO_RL_UID1000_IMAGE = "sha256:7a4078e70b28cc6de622fa3cf25c7596c52337fa3f1779488b402c5fa1c74e69"
NEMO_RL_IMAGES = (NEMO_RL_IMAGE, NEMO_RL_UID1000_IMAGE)
NEMO_RL_PYTHON = "/opt/nemo_rl_venv/bin/python"               # nemo-rl docker/Dockerfile:227, :621
VLLM_WORKER_PYTHON = "/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python"
# NeMo RL in the Tier B image (docs/tier-b-runbook.md step 6): Ray's object store needs a larger /dev/shm than the
# default 8g, NeMo RL's locked memory and open files, its source tree as cwd.
NEMO_RL_RUN_OPTIONS = ("--shm-size=16g", "--ulimit", "memlock=-1", "--ulimit", "nofile=65535:65535", "-w",
                       "/opt/nemo-rl")

# Output layout: the recipes' cut writers publish cuts/checkpoint-<N>/ (cut.json last) after NeMo RL saves step_<N>;
# NeMo RL's own logs lie under nemo/logs/exp_<n>/ with one train_data_step<N>.jsonl per step.
CUTS_DIR = "cuts"
CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")        # under cuts/; `.checkpoint-<N>.partial` never
CUT_MARKER = "cut.json"
SHA_RE = re.compile(r"sha256:[0-9a-f]{64}")
FREEZE_FILE = "env-freeze.txt"
FREEZE_COMMIT_RE = re.compile(r"# nemo-rl commit ([0-9a-f]{40})")
NEMO_LOGS = ("nemo", "logs")
EXP_DIR_RE = re.compile(r"exp_\d+")
TRAIN_DATA_RE = re.compile(r"train_data_step(\d+)\.jsonl")
