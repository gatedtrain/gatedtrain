# NeMo RL trainer image for a trainer running as uid 1000 (logical name `nemo-rl/tierb-uid1000`)

The NeMo RL family's derived trainer image. The IDs gatedtrain accepts for it are data in the image lock
(`gatedtrain image list`; the shipped file is `src/gatedtrain/data/image-lock.json`), not code. The NeMo RL recipes
(CAD, VLM, agentic) run their trainer and generation containers in it.

`containers/` is not shipped in the wheel or sdist: the build recipes need a source checkout of gatedtrain.

It is the Tier B image (`containers/nemo-rl-tierb`) plus one layer: the ten vLLM files NeMo RL patches at start-up, and
their directories, belong to uid 1000. NeMo RL writes `<file>.patch_lock` beside them, and gatedtrain refuses a trainer
running as root (MM-34), so without this layer the first GRPO step fails with PermissionError. Nothing is added or
upgraded. The rootless-host variant (owner 0:0) is `containers/nemo-rl-rootless`.

## The base image

The base is the Tier B image, a local build, not a published image. `containers/nemo-rl-tierb/README.md` explains how to
build it and how an image of your own is recorded; the source repository and commit are in the lock
(`gatedtrain --json image list`). `docs/tier-b-runbook.md` step 2 has the checks and the build lessons.

## Build this image

From this directory, on the Docker daemon that will run the containers (no network needed):

    docker build --build-arg BASE=<the Tier B image tag or ID> -t <tag> .

Pass `--context <name>` when that daemon is not the default Docker context. `BASE` has no default on purpose. Then:

    gatedtrain image check nemo-rl/tierb-uid1000 --ref <tag> [--docker-context <name>]

The maintainer lock records no Containerfile hash for the reference build of this image: it was built from the file
before it took `BASE` as a build argument (same chown list, same layers), so the hash of the file shipped here would
claim a build that did not happen.

A different ID is a different image: `ext-launch` refuses it until it is accepted, either by the maintainers (a new
lock entry in a release) or by you with `gatedtrain image add` (a recorded act; reports flag it as USER-ADDED).

The Containerfile ends with `USER root` (the user needed for the `chown`). It is kept because it is part of the image
configuration behind the accepted image ID; ext-launch always passes an explicit non-root `--user` (MM-34), so the
image's default user is never used by gatedtrain.
