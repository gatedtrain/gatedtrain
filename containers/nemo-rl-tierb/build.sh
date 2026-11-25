#!/bin/bash
# Build the Tier B NeMo RL image from a checkout of the NeMo RL source, then print the command that records it.
# Everything comes from the arguments: no path, URL, tag or commit is built into this script. The checks before the
# build only read; the build runs only without --dry-run; the script never writes gatedtrain's image config (recording
# the image is the owner's explicit `gatedtrain image add`, printed at the end).
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: build.sh --context CTX --source-dir DIR --commit SHA --tag TAG --max-jobs N [options]

  --context CTX        the Docker context of the daemon that will hold the image (required)
  --source-dir DIR     a checkout of the NeMo RL source with its submodules (required; `gatedtrain --json image list`
                       names the source and commit the maintainer lock was built from)
  --commit SHA         the full 40-hex commit the checkout must be at (required)
  --tag TAG            the local tag to give the image (required)
  --max-jobs N         parallel compile jobs, about half the CPU cores; memory bound, see README.md (required)
  --recipe-name NAME   recipe name to print in the add command (default: nemo-rl-source-build)
  --allow-modified     build from a checkout with local changes; the printed command then says so in the note
  --rebuild            build even when the tag already exists on the daemon
  --dry-run            run the read-only checks and print the plan; build nothing
  -h, --help           this text
USAGE
}

fail() { echo "build.sh: $*" >&2; exit 1; }
need() { [ -n "${2:-}" ] || { echo "build.sh: missing $1" >&2; usage >&2; exit 2; }; }
# An option that takes a value must be followed by one (`shift 2` would otherwise fail without a word).
value() { [ "$2" -ge 2 ] || { echo "build.sh: $1 needs a value" >&2; usage >&2; exit 2; }; }

context="" source_dir="" commit="" tag="" max_jobs="" recipe="nemo-rl-source-build"
allow_modified=0 rebuild=0 dry=0
while [ $# -gt 0 ]; do
  case "$1" in
    --context) value "$1" $#; context="${2:-}"; shift 2 ;;
    --source-dir) value "$1" $#; source_dir="${2:-}"; shift 2 ;;
    --commit) value "$1" $#; commit="${2:-}"; shift 2 ;;
    --tag) value "$1" $#; tag="${2:-}"; shift 2 ;;
    --max-jobs) value "$1" $#; max_jobs="${2:-}"; shift 2 ;;
    --recipe-name) value "$1" $#; recipe="${2:-}"; shift 2 ;;
    --allow-modified) allow_modified=1; shift ;;
    --rebuild) rebuild=1; shift ;;
    --dry-run) dry=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "build.sh: unknown argument $1" >&2; usage >&2; exit 2 ;;
  esac
done
need --context "$context"; need --source-dir "$source_dir"; need --commit "$commit"; need --tag "$tag"
need --max-jobs "$max_jobs"
[[ "$commit" =~ ^[0-9a-f]{40}$ ]] || { echo "build.sh: --commit is not 40 lowercase hex digits" >&2; exit 2; }
[[ "$max_jobs" =~ ^[1-9][0-9]*$ ]] || { echo "build.sh: --max-jobs is not a whole number >= 1" >&2; exit 2; }
# A tag and a context name are docker names: they start with a letter or digit, so neither can be read as an option
# (`-v`, `--host=...`); the docker commands below also put `--` before the tag.
[[ "$tag" =~ ^[A-Za-z0-9][A-Za-z0-9_.:/@-]*$ ]] || { echo "build.sh: --tag is not an image reference (it must start with a letter or digit)" >&2; exit 2; }
[[ "$context" =~ ^[A-Za-z0-9][A-Za-z0-9_.+-]*$ ]] || { echo "build.sh: --context is not a Docker context name (it must start with a letter or digit)" >&2; exit 2; }

# Read-only checks.
command -v docker >/dev/null || fail "the docker CLI is not on PATH"
command -v git >/dev/null || fail "git is not on PATH"
docker --context "$context" context inspect -- "$context" >/dev/null 2>&1 || fail "no Docker context named $context"
[ -f "$source_dir/docker/Dockerfile" ] || fail "$source_dir has no docker/Dockerfile: not a NeMo RL checkout"
head="$(git -C "$source_dir" rev-parse HEAD 2>/dev/null)" || fail "$source_dir is not a git checkout"
[ "$head" = "$commit" ] || fail "the checkout's HEAD is $head, not the requested commit $commit"
if git -C "$source_dir" submodule status --recursive | grep -q '^-'; then
  fail "submodules are not initialised: git -C $source_dir submodule update --init --recursive"
fi
modified=no
# The build context is the whole checkout: untracked AND ignored files (--ignored) end up in it too, so they count as
# local changes, in the checkout and in each submodule.
local_changes="$(git -C "$source_dir" status --porcelain --ignored; git -C "$source_dir" submodule --quiet foreach --recursive 'git status --porcelain --ignored')"
if [ -n "$local_changes" ]; then
  [ "$allow_modified" -eq 1 ] || fail "the checkout has local changes; commit or discard them, or pass --allow-modified"
  modified=yes
fi

dockerfile_hash="sha256:$(sha256sum "$source_dir/docker/Dockerfile" | cut -d' ' -f1)"
build=(docker --context "$context" buildx build --target release --build-context "nemo-rl=$source_dir"
       -f "$source_dir/docker/Dockerfile"
       --build-arg SKIP_TRTLLM_BUILD=1 --build-arg SKIP_SGLANG_BUILD=1
       --build-arg "NEMO_RL_COMMIT=$commit" --build-arg "MAX_JOBS=$max_jobs"
       --tag "$tag" --load "$source_dir")

existing="$(docker --context "$context" image inspect -f '{{.Id}}' -- "$tag" 2>/dev/null || true)"
echo "plan: build $tag on context $context from $source_dir at $commit (locally modified: $modified)"
built=yes
if [ -n "$existing" ] && [ "$rebuild" -eq 0 ]; then
  built=no
  echo "image tag exists; not rebuilt: $tag is already on the daemon as $existing (pass --rebuild to build again)"
  [ "$dry" -eq 1 ] && echo "dry run: nothing was built"
elif [ "$dry" -eq 1 ]; then
  echo "dry run: would run: ${build[*]}"
  echo "dry run: nothing was built"
  exit 0
else
  "${build[@]}"
fi

id="$(docker --context "$context" image inspect -f '{{.Id}}' -- "$tag")"
# The note records what the recorded image is NOT: a tag found on the daemon was not built by this run, so nothing in
# this run vouches for what it holds.
notes=()
[ "$built" = no ] && notes+=("tag existed; not built by this run")
[ "$modified" = yes ] && notes+=("built from a locally modified checkout")
note=""
if [ "${#notes[@]}" -gt 0 ]; then
  joined="$(printf '%s; ' "${notes[@]}")"
  note=" --note $(printf %q "${joined%; }")"
fi
echo
if [ "$built" = yes ]; then echo "built image ID: $id"; else echo "image ID (not rebuilt): $id"; fi
echo "The ID of a source build is specific to this host and time (base images are pulled by tag, so the build is not"
echo "bit-reproducible; the recipe provenance below, not the ID, is the audit trail). To accept the image (a recorded,"
echo "owner-side act; reports flag it as user-added), run:"
if [ "$built" = yes ]; then
  echo "  gatedtrain image add nemo-rl/tierb --id $(printf %q "$id") --recipe $(printf %q "$recipe") --recipe-commit $(printf %q "$commit") --containerfile-sha256 $(printf %q "$dockerfile_hash")$note"
else
  # This run did not build the image, so nothing here vouches for its source or Containerfile: the command claims
  # neither. `gatedtrain image add` refuses it until you supply the true --recipe-commit or --containerfile-sha256.
  echo "  gatedtrain image add nemo-rl/tierb --id $(printf %q "$id") --recipe $(printf %q "$recipe")$note"
  echo "This run did not build the image, so the command above names no recipe commit or Containerfile hash, and"
  echo "'image add' refuses it until you supply the true --recipe-commit COMMIT (or --containerfile-sha256 HASH)."
fi
