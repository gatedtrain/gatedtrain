# The Tier B NeMo RL image (logical name `nemo-rl/tierb`)

The trainer image of the NeMo RL recipes: NeMo RL's own `docker/Dockerfile` (release stage) built at a pinned commit.
It is not published anywhere; it is built locally. The `containers/` directory is not shipped in the wheel or sdist
(the package carries only the lock), so these recipes need a source checkout of gatedtrain. gatedtrain pins it by image ID, and the IDs are data, not code: the
maintainer lock shipped with gatedtrain (`gatedtrain image list`), plus the IDs you add yourself.

## What the lock says, and what a rebuild gives you

The lock names the source repository, the commit and the build date of the reference build
(`gatedtrain --json image list`, field `recipe`). Read the source and commit from there; this directory hard-codes none.

The ID of a source build is host and time specific: the same commit, built again or elsewhere, usually gets another ID.
NeMo RL's `docker/Dockerfile` pulls its base images by tag, so the build is not bit-reproducible and the ID says nothing
beyond "this build on this host at this time". The provenance (the source, the commit, the sha256 of `docker/Dockerfile`
the script records as `containerfile_sha256`), not the ID, is the audit trail. So the lock holds the reference build's
ID, and a build of yours is accepted only after you record it:

    gatedtrain image check nemo-rl/tierb --ref <your tag> [--docker-context <ctx>]    # exit 0 accepted, 1 not
    gatedtrain image add nemo-rl/tierb --id <ID> --recipe <name> --recipe-commit <commit> ...

`image add` writes your own config (`$XDG_CONFIG_HOME/gatedtrain/images.json`; no sudo). It is an explicit, recorded act,
but be clear about what it is. The config is NOT a security boundary against an agent that runs as your user: your
environment (`HOME`, `XDG_CONFIG_HOME`) locates the file, and a same-user agent controls both, so it can run
`image add --yes` or write the file itself. What stays reliable is the keeper's flag: the keeper checks the declared
images against its own copy of the maintainer lock, the signed report flags one that is not in that lock as USER-ADDED
and carries the sha256 of that lock, and `gatedtrain verify` prints the hash and warns
(`--require-maintainer-images` fails on such a report, and on one that lists no image at all). The flag holds when the
keeper's code is outside the agent's reach (a separate-user keeper). The image block is the driver's declaration read
from job records the agent can write, not proof of what ran. Also, `image add` is recorded for provenance only: published
packs still read the shipped lock, so it does not by itself make a pack accept the image (packs adopt
`accepted_images(logical)` in a later release).

## Build

1. Clone the source named by the lock, check out its commit and initialise the submodules (the script checks all three
   and refuses otherwise):

       git clone <source> nemo-rl
       git -C nemo-rl checkout <commit>
       git -C nemo-rl submodule update --init --recursive

2. Run the script on the daemon that will run the containers. It takes everything from its arguments, checks read-only
   first, builds only when those pass, does nothing when the tag already exists (idempotent; `--rebuild` overrides),
   and never writes your gatedtrain config:

       containers/nemo-rl-tierb/build.sh --context <ctx> --source-dir nemo-rl --commit <commit> --tag <tag> \
         --max-jobs <N> --dry-run          # the checks and the plan only
       containers/nemo-rl-tierb/build.sh --context <ctx> --source-dir nemo-rl --commit <commit> --tag <tag> --max-jobs <N>

   It builds with `--target release` (without it the build stops at a 10-byte stage), skips the TensorRT-LLM and
   SGLang builds (`SKIP_TRTLLM_BUILD=1`, `SKIP_SGLANG_BUILD=1`), and passes the commit as `NEMO_RL_COMMIT`. These are
   the build arguments of `docs/tier-b-runbook.md` step 2, which also has the build lessons: about 2-5 hours, and
   memory: `--max-jobs` alone does not bound it on a 32 GB host (uv builds several CUDA packages at once).

3. The script ends by printing the exact `gatedtrain image add nemo-rl/tierb ...` command with the ID, the recipe
   commit and the hash of the Dockerfile it built from. Run it if you accept the image. A checkout with local changes
   (untracked and ignored files count, since the whole checkout is the build context) is refused unless
   `--allow-modified`, and then the printed command says so in its note. With an existing tag and no `--rebuild` it
   says "image tag exists; not rebuilt" and prints the command for the ID found WITHOUT `--recipe-commit` and
   `--containerfile-sha256`, with the note "tag existed; not built by this run": this run did not build that image, so
   it claims no provenance for it, and `gatedtrain image add` refuses the command until you supply the true recipe
   commit or Containerfile hash.

## Derived images

`containers/nemo-rl-uid1000` (`nemo-rl/tierb-uid1000`, trainer running as uid 1000) and `containers/nemo-rl-rootless`
(`nemo-rl/tierb-rootless`, rootless Docker host) add one layer on this image and take it as `--build-arg BASE=<tag>`.
