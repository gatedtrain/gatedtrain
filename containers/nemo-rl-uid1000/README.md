# NeMo RL image for a trainer running as uid 1000 (`gatedtrain-a6-nemo-rl:1`)

The NeMo RL family's derived trainer image, pinned in `gatedtrain.families.nemo_rl` as `NEMO_RL_UID1000_IMAGE`
(`sha256:7a4078e7...`; `NEMO_RL_IMAGES` holds it next to the Tier B image `NEMO_RL_IMAGE`). The NeMo RL recipes (CAD,
VLM, agentic) run their trainer and generation containers in it.

It is the Tier B image `nemo-rl:4f39fd8` plus one layer: the ten vLLM files NeMo RL patches at start-up, and their
directories, belong to uid 1000. NeMo RL writes `<file>.patch_lock` beside them, and gatedtrain refuses a trainer running as
root (MM-34), so without this layer the first GRPO step fails with PermissionError. Nothing is added or upgraded.

## The base image `nemo-rl:4f39fd8`

`nemo-rl:4f39fd8` is a local tag, not a published image. It is built from the public NeMo RL repository
(<https://github.com/NVIDIA-NeMo/RL>) at commit `4f39fd86289a2ffd5aaf5d202702f7fd7185c924`, with that repository's own
`docker/Dockerfile` (`gatedtrain.backends.nemo_rl.DEFAULT_IMAGE` defaults to this tag):

    git clone https://github.com/NVIDIA-NeMo/RL.git nemo-rl
    git -C nemo-rl checkout 4f39fd86289a2ffd5aaf5d202702f7fd7185c924
    git -C nemo-rl submodule update --init --recursive
    cd nemo-rl
    docker buildx build --target release --build-context nemo-rl=. -f docker/Dockerfile \
      --build-arg SKIP_TRTLLM_BUILD=1 --build-arg SKIP_SGLANG_BUILD=1 \
      --build-arg NEMO_RL_COMMIT=4f39fd86289a2ffd5aaf5d202702f7fd7185c924 \
      --build-arg MAX_JOBS=<CPU cores / 2> --tag nemo-rl:4f39fd8 --load .

`docs/tier-b-runbook.md` step 2 has the checks (the Automodel submodule commit, the commit inside the image) and the
build lessons (`--target release`, memory limits). The reference build had image ID `sha256:8cd5b714...`; an image
built elsewhere may get a different ID, which then has to be recorded deliberately (`NEMO_RL_IMAGE`).

## Build this image

From this directory, on the Docker daemon that will run the containers (no network needed):

    docker build -t gatedtrain-a6-nemo-rl:1 .

Pass `--context <name>` when that daemon is not the default Docker context (the reference host used a dedicated
context named `gatedtrain`). Check the image ID against `NEMO_RL_UID1000_IMAGE` after the build; a different ID is a
different image, which `ext-launch` refuses until the constant is changed deliberately.

The Containerfile ends with `USER root` (the user needed for the `chown`). It is kept because it is part of the image
configuration behind the pinned image ID; ext-launch always passes an explicit non-root `--user` (MM-34), so the
image's default user is never used by gatedtrain.
