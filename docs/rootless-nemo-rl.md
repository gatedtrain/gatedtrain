# NeMo RL on a rootless host: lessons

This page is for someone who runs a NeMo RL pack or recipe on a host where Docker runs rootless (the setup of the
[quickstart without sudo](sudo-free-quickstart.md), or the hardened setup in [AI-native projects](ai-native.md)). It
collects what went wrong on the first real runs and what to do about it, so you do not meet each one on a GPU hour.

**How to read the labels.** *Verified* means it was observed in a run of ours on one host: a single machine with one
RTX 4090 (24 GB), rootless Docker for the owner and for the keeper, NeMo RL at commit `4f39fd8`. The runs are the
VLM scenario pack (Qwen3.5-0.8B on CLEVR-CoGenT, 399 sealed items) and the text pack (Qwen3, GSM8K rows); the run notes
are the maintainers' own and are not published. *General advice* is inference from those runs or from how the pieces
work, and was not tested on other hardware. Where the Docker daemon is rootful, or the GPU or the model differs, check
the numbers again.

## 1. Image ownership under rootless Docker (verified)

Under rootless Docker the container's root is you on the host, and the training container drops all capabilities. In
the Tier B image the ten vLLM files that NeMo RL rewrites at start-up (and their eight directories) belong to uid
65532. NeMo RL creates a `<file>.patch_lock` beside each, and without `DAC_OVERRIDE` only a file owned by uid 0 can be
written, so the first GRPO step fails with a `PermissionError` on `ray_executor.py.patch_lock`.

The fix is a derived image with one small layer (about 380 kB) that makes those files owned by root:
`containers/nemo-rl-rootless/` in the gatedtrain source (logical name `nemo-rl/tierb-rootless`). Nothing is added or
upgraded. It is `containers/nemo-rl-uid1000` with another owner; a test keeps the two file lists equal. Build it on
your own rootless daemon with `--build-arg BASE=<the Tier B image>`; `BASE` has no default on purpose. Every rootless
NeMo RL pack needs this image, or an equivalent.

## 2. The image lock (verified for the mechanism; the build recipe is yours to run)

gatedtrain does not pin images in code. It refers to logical names and keeps the accepted image IDs in an image lock
shipped with the release, plus a file of your own for images you built.

- `gatedtrain image list` shows every accepted image, its origin (`maintainer-built` or `user-built`) and its recipe. It also prints the sha256 of the maintainer lock it read (`lock_sha256` under `gatedtrain --json image list`); this is informational until release hashes are published to compare it with.
- `gatedtrain image check <logical> --ref <ref> [--docker-context <ctx>]` (or `--id sha256:...`) says whether an image
  is accepted (exit 0) or not (exit 1, with the `image add` command to run).
- `gatedtrain image add <logical> --id sha256:... --recipe <name> --recipe-commit <commit> ...` records your build in
  `$XDG_CONFIG_HOME/gatedtrain/images.json`. It needs no sudo, asks you to type `yes` at a terminal (or takes `--yes`),
  and only adds.
- The build recipes are in `containers/`: `nemo-rl-tierb/` (the image built from NeMo RL's own Dockerfile at the
  pinned commit, with `build.sh`), `nemo-rl-uid1000/` and `nemo-rl-rootless/` (the derived layers).

Things to know:

- **The ID of a source build is host and time specific.** The same commit built again usually gets another ID. The
  maintainer lock holds the ID of the reference build, so a build of yours is accepted only after you record it with
  `image add`. No registry holds the reference image, so each user builds it themselves (a build of NeMo RL's release
  stage takes hours).
- **A user-built image is flagged in the report.** Reports list each trainer image with its origin and recipe, as the
  driver declares it, and the keeper checks that against its own copy of the maintainer lock. An image you added shows
  as `USER-ADDED, not in the maintainer lock`, and `gatedtrain verify --require-maintainer-images` exits 1 on it. That
  is the honest statement: you, not the maintainers, vouched for that image. The declaration is the driver's; the
  keeper does not see which image a container really ran.
- **The config is not a barrier against an agent running as you.** No environment variable, command-line flag or
  project file supplies an accepted ID, and the config file is refused if it is a link, is not yours, or is writable by
  group or others. But `HOME` and `XDG_CONFIG_HOME` locate the file, and an agent that runs as you controls both and
  can run `image add --yes` or write the file itself. The reliable signal is the keeper's `user_added` flag in the
  signed report, together with the sha256 of the keeper's shipped lock (`verify` prints it; it is informational until
  release hashes are published). That holds when the keeper's code is outside the agent's reach (separate-user mode).
- **`image add` changes nothing for gating yet.** It is recorded for provenance; packs adopt `accepted_images(logical)`
  in a later release, and until then published packs read the shipped lock only.
- **What the report's image block is.** The job records it is read from are in the agent-writable project, so it is the
  driver's declaration, not proof of what ran. A report whose jobs have no image block makes `verify` warn "trainer
  images not declared by the driver", and `--require-maintainer-images` exit 1 (also for a report that lists no image).
  `containers/` is not shipped in the wheel or sdist, so the build recipes need a source checkout of gatedtrain.
- **Packs are not migrated yet.** The published scenario packs still use their own image lists (the pack for the VLM
  recipe reads an environment variable of its own for extra IDs) until they adopt the lock's accepted-images call. The
  old module attributes of gatedtrain keep working for them, computed from the maintainer entries of the lock.

## 3. `memlock` (verified on one host)

The rootful NeMo RL path runs with `--ulimit memlock=-1`. A rootless daemon refuses it, because it cannot raise the
hard limit above the one it inherited. The VLM pack leaves the option out and its GPU run passed. `gatedtrain setup
check` shows your hard limit as an informational row. If a workload needs more locked memory, an administrator has to
raise the hard limit for your user; that is a prerequisite outside gatedtrain. Whether other models need it is not
known.

## 4. vLLM is not deterministic by default, so greedy decoding needs `max_num_seqs=1` (verified; the cause is inference)

Registered greedy decoding (see [Greedy decoding](ai-native.md#greedy-decoding)) is checked by the keeper (AN-26):
every item is generated under the registered seed and under a control seed, and any difference refuses the attempt
after it spent its alpha. With vLLM's default batching the two passes differ a lot, because batch composition changes
the numerics and a last-bit difference in bf16 becomes another text over hundreds of tokens.

| Run (one RTX 4090, two passes, two seeds in one container) | Items that differ between the passes |
|---|---|
| VLM pack, Qwen3.5-0.8B, 512 new tokens, base model, default batching | 224/399 (56.1 %); a second process 217/399 (54.4 %) |
| Same, trained model | 121/399 (30.3 %) |
| Same, base model, `max_num_seqs=1` | 0/399 |
| Same, trained model, `max_num_seqs=1` | 0/399 |
| Text pack, Qwen3, 1,024 new tokens, base model, default batching | 239/399 (59.9 %) |
| Same, trained model | 205/399 (51.4 %) |
| Same, both models, `max_num_seqs=1` | 0/399 |

`VLLM_BATCH_INVARIANT=1` did not help for Qwen3.5: the engine does not start ("batch_invariant mode is not supported
for GDN_ATTN"). Running one request at a time with prefix caching off (`max_num_seqs=1`) made the passes identical in
these runs. General advice: a pack must pass a determinism probe (two passes, different seeds, on items that are not
sealed items) before it registers greedy decoding, and again after any change of the generator, vLLM or GPU. The
alternative is `decoding: sampled`.

## 5. Time limits (verified on the RTX 4090; slower setups are inference)

- **The keeper's batch-generator time limit is capped at 1,200 s** (AN-25). Greedy decoding with `max_num_seqs=1` is
  slow: the text pack needed 17 min 42 s and 17 min 33 s for the two passes over 399 items (about 10 % under the cap),
  and the VLM pack 435 s and 277 s. A slower GPU, longer outputs or more items will exceed the cap, and an evaluation
  that hits it is a crash that has already spent its alpha. Estimate before the sealed run: time a probe of a few
  items, scale to all items, and double the generation for the two passes.
- **A GPU attempt waits up to 900 s for a free GPU** and is refused when the project's committed GPU hours plus its
  estimate pass the budget; a running attempt stops at its time limit or the budget. See [the local
  backend](ai-native.md#local-backend).

## 6. `keeper confirm` needs a terminal (verified)

`gatedtrain keeper confirm` asks you to type `yes`. Through some agent shells there is no stdin, so run it in a terminal
or pass `--yes` after reading the sentence. It is a step for you, not for an agent, in both keeper modes: an agent with
your shell could run it too, and the keeper records only that the confirmation came from the owner uid.

## 7. Paths: no root-owned roots for your data (verified)

Do not put your own data under a root-owned directory such as `/srv/...`: you cannot write there without sudo. Keep it
in a tree you own. In separate-user mode the keeper reads model files from a hand-off directory, so every directory on
the way must be traversable by the keeper's uid (`o+x` on directories, `o+r` on files, no links); in same-user mode the
keeper is your own user and this does not apply.

## 8. GPU hours are logged, hand probes are not (verified)

Every job the local backend runs adds a row to its GPU log, and the budget counts them. A probe you start by hand (a
determinism probe, a timing test) is not in that log. Keep your own tally of those hours next to the log, so that the
total you report is the real one.
