# NeMo RL trainer image for a rootless Docker host (logical name `nemo-rl/tierb-rootless`)

The Tier B image (`containers/nemo-rl-tierb`) plus one layer: the ten vLLM files NeMo RL patches at start-up, and their
eight directories, are owned by root (0:0). It is `containers/nemo-rl-uid1000` with another owner; the parity of the two
file lists is a test (`tests/test_image_recipes.py`).

## Why

NeMo RL rewrites those vLLM files and creates `<file>.patch_lock` beside them. In the Tier B image they belong to uid
65532 (`nemo-runtime`). Under rootless Docker the container's root is the owner on the host, and the training container
drops every capability (no `DAC_OVERRIDE`), so only a file owned by uid 0 can be written: without this layer the first
GRPO step fails with PermissionError on `ray_executor.py.patch_lock`. Nothing is added or upgraded.

## Build

On the owner's rootless daemon; no network is needed:

    docker --context <ctx> build --build-arg BASE=<the Tier B image tag or ID> -t <tag> \
      -f containers/nemo-rl-rootless/Containerfile containers/nemo-rl-rootless

`BASE` has no default on purpose: it must name an image that is accepted for `nemo-rl/tierb`
(`gatedtrain image check nemo-rl/tierb --ref <BASE>`). The ID of your build is host specific, so the maintainer lock
holds none for this logical name. Record yours:

    gatedtrain image add nemo-rl/tierb-rootless --id <ID> --recipe nemo-rl-rootless --recipe-commit <commit> \
      --containerfile-sha256 sha256:<hash of containers/nemo-rl-rootless/Containerfile> --base-id <ID of BASE>

Reports then flag the image as USER-ADDED (it is not in the maintainer lock). That is the honest statement: a result
says which image the driver declared and that it is not in the maintainer lock (it is the driver's declaration from job
records the agent can write, not proof of what ran).

`containers/` is not shipped in the wheel or sdist: this recipe needs a source checkout of gatedtrain.
