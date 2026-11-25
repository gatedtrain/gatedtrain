"""NeMo RL family: the pinned framework, its images and the output layout NeMo RL and the recipes' cut writers share.

Used by the recipes that train with NeMo RL (CAD SFT and GRPO, VLM GRPO, agentic GRPO). Values are the pins of the
Tier B NeMo RL image (nemo-rl 4f39fd8) and of its uid-1000 derivative.
"""

from __future__ import annotations

import re

NEMO_RL_COMMIT = "4f39fd86289a2ffd5aaf5d202702f7fd7185c924"  # Tier B image nemo-rl:4f39fd8
# The NeMo RL images a recipe may run in are pinned by image ID in DATA, the image lock (`gatedtrain.imagelock`,
# `data/image-lock.json`), under the logical names below; no ID is written in code. The module attributes
# `NEMO_RL_IMAGE`, `NEMO_RL_UID1000_IMAGE` and `NEMO_RL_IMAGES` stay for the published packs that import them: they are
# computed from the SHIPPED lock only (maintainer entries, Tier B first). New code calls `accepted_images(logical)`,
# which also includes the IDs the owner added with `gatedtrain image add`.
IMAGE_TIERB = "nemo-rl/tierb"                    # the Tier B image, built from NeMo RL's own docker/Dockerfile
IMAGE_UID1000 = "nemo-rl/tierb-uid1000"          # its layers plus one that hands the ten vLLM files NeMo RL patches to uid 1000
IMAGE_ROOTLESS = "nemo-rl/tierb-rootless"        # the same layer for a rootless Docker host (root-owned files)
LOGICAL_IMAGES = (IMAGE_TIERB, IMAGE_UID1000, IMAGE_ROOTLESS)


def accepted_images(logical: str, **kwargs) -> list[str]:
    """The accepted image IDs of a logical image: the shipped lock, then the owner's config additions."""
    from .. import imagelock
    return imagelock.accepted_ids(logical, **kwargs)


def __getattr__(name: str):
    """The legacy image constants, computed from the shipped lock (maintainer entries only)."""
    if name in ("NEMO_RL_IMAGE", "NEMO_RL_UID1000_IMAGE", "NEMO_RL_IMAGES"):
        from .. import imagelock
        lock = imagelock.load_lock()
        tierb_ids = imagelock.maintainer_ids(IMAGE_TIERB, lock=lock)
        uid1000_ids = imagelock.maintainer_ids(IMAGE_UID1000, lock=lock)
        if not (tierb_ids and uid1000_ids):          # an AttributeError, so `getattr(m, name, default)` and hasattr work
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}: the shipped image lock lists no "
                                 "maintainer image for it")
        tierb, uid1000 = tierb_ids[0], uid1000_ids[0]
        return {"NEMO_RL_IMAGE": tierb, "NEMO_RL_UID1000_IMAGE": uid1000, "NEMO_RL_IMAGES": (tierb, uid1000)}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


NEMO_RL_COMMIT_SHORT = NEMO_RL_COMMIT[:7]                     # how tags and records name it
NEMO_RL_SRC = "/opt/nemo-rl"                                  # NeMo RL's source tree inside its image (the cwd)
NEMO_RL_PYTHON = "/opt/nemo_rl_venv/bin/python"               # nemo-rl docker/Dockerfile:227, :621
VLLM_WORKER_PYTHON = "/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python"
# NeMo RL in the Tier B image (docs/tier-b-runbook.md step 6): Ray's object store needs a larger /dev/shm than the
# default 8g, NeMo RL's locked memory and open files, its source tree as cwd.
NEMO_RL_RUN_OPTIONS = ("--shm-size=16g", "--ulimit", "memlock=-1", "--ulimit", "nofile=65535:65535", "-w",
                       NEMO_RL_SRC)

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
