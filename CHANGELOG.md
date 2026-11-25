# Changelog

Release notes, newest first. Each release lists what is new and what changes for users of the earlier release. The
"Rules" line at the end of a release names the rules behind the changes; the [test reference](tests/README.md) maps
each rule to its tests.

## Changes in 0.6.0

The analysis code version stays `gatedtrain 0.3.0`: `gates.py` did not change, and the one change in `registration.py`
is an opt-in mode that is refused unless a path asks for it, so registrations made with 0.3.0 to 0.5.0 stay valid.

Registered greedy decoding for the AI-native path (design revision 26, rules REG-GREEDY and AN-26; see [Greedy
decoding](docs/ai-native.md#greedy-decoding)):

- Greedy is an **explicit opt-in registration mode**: `sampler.decoding: greedy` with `temperature: 0`, a `top_p` that
  is absent or 1, one evaluation seed and one sample per item. `temperature: 0` alone is still refused, and so is
  greedy decoding on every path that cannot check it (`ext-*`, Tier B, `check_contents` without `greedy_allowed`).
  Violations are refused with `REG-GREEDY`.
- **AN-26, the keeper's determinism check.** `evaluate_sealed` (the keeper and the in-process path) generates every
  greedy item once more under a control seed that the registration does not name, drawn by the keeper per arm and
  attempt, and refuses the attempt if the predictions differ. The check runs after the alpha grant: a refused attempt
  settles as not passed and costs its alpha. The refusal states no count and no prediction. The keeper compares the
  predictions itself and takes nothing about determinism from the container.
- No change to the gate statistics, the power plan or alpha-investing: the item is the resampled unit, so one sample per
  item is the K = 1 case of the same estimator. A greedy score is the greedy decoder's score; it is not comparable with
  a score at temperature above 0.

Known limits (read before registering greedy decoding):

- **vLLM's default batching is not deterministic.** On a GPU run of the VLM pack (Qwen3.5-0.8B, 512-token chain of
  thought, 399 items), 54 to 56 % of the items of the base model and 30 % of those of the trained model differed between
  two greedy passes with the default batching; the same seed in two processes differed as much, so this is batch-
  dependent numerics, not the seed. With `max_num_seqs=1` (one request at a time) the same runs gave 0 of 399 differing
  items for both models. vLLM's batch-invariant mode does not support Qwen3.5 (`GDN_ATTN`). AN-26 therefore fails
  closed: **a pack must pass a determinism probe (two passes, different seeds, on items outside the sealed set) before
  it registers greedy decoding**, or its sealed evaluations will be refused after spending alpha.
- **The text pack (`gatedtrain-pack-text-gsm8k`) has not been probed** and still needs `gatedtrain<0.6`; it does not
  register greedy decoding.
- The control pass runs in the same process as the registered pass, so differences between processes are not seen.
  A sampler whose outputs all agree under two seeds on the sealed set passes the check.
- **Reports and the owner's metric confirmation do not yet state the decoding mode.** It is in the signed registration,
  whose hash the report names. Stating it needs a change of the keeper's report and confirmation texts.
- A keeper older than 0.6.0 refuses a greedy registration (`REG-FIELD`, unknown sampler field `decoding`): upgrade the
  keeper's environment together with the driver and the pack.
- Packs that pin `gatedtrain>=0.5,<0.6` need a release that allows 0.6.

Rules: REG-GREEDY, AN-26.

## Changes in 0.5.0

The analysis code version stays `gatedtrain 0.3.0` (the gate statistics and registration semantics did not change),
so registrations made with 0.3.0 or 0.4.0 stay valid.

AI-native step 3a, a preview on Linux (see [Rootless containers, the local backend and the keeper's container (step
3a)](docs/ai-native.md#rootless-containers-the-local-backend-and-the-keepers-container-step-3a)):

- Rootless host setup: `gatedtrain setup host` prints one root script that sets up a rootless Docker daemon for the
  owner and one for the keeper user and removes the owner from the rootful `docker` group. gatedtrain never runs it.
- The local job backend: pack stages run their jobs in containers on the owner's rootless daemon (`ctx.backend`),
  with GPU accounting against the project's budget and a row per job in the host's GPU log; `gatedtrain project job
  list | logs | cancel` show and stop them.
- The keeper's container: `keeper serve --inference-image ...` runs a scenario pack's batch generator in a container
  of the keeper's own rootless daemon, on a copy of the model files the keeper hashed.
- `ext-launch` on a rootless host (`rootless = true` in the host file): the launch rules accept the owner's rootless
  daemon and print `--device nvidia.com/gpu=all` in place of `--runtime=nvidia --gpus all`.

Behaviour that changes for 0.4.0 users:

- `verify` exits 1 for a report signed by a 0.4.0 keeper: that report states the 0.4.0 guarantee, which was stated
  before rootful Docker access was checked. It also exits 1 against a 0.4.0 keeper (its `hello` does not state
  `owner_has_root_docker`), and for a report of a keeper that runs as the owner's own user (a test or development
  setup; before, only `--keeper` against such a keeper exited 1). `verify` exits 0 only when every report states the
  separate-user guarantee.
- Upgrade the keeper's environment (the venv `setup keeper` installed) together with the driver: the driver now sends
  the project with `hello`, and a 0.4.0 keeper states no rootful-access check.
- A scenario pack scorer whose id is also an ext evaluator row id is refused; before, the row shadowed the
  scorer at registration and `preregister` refused with an unclear message. Give the scorer its own id.

Fixes:

- The credential file-name scan no longer walks a read-only source that is itself one of the adapter's `scan_skip`
  directories (for example a job that mounts a runtime's `.venv` directly); the source's own name is still checked.
- `keeper confirm` without `--yes` and with no answer on stdin (a shell started without input) now says no answer was
  read and to rerun with `--yes` after reading the sentence.
- Importing a scenario pack or ext adapter plugin that uses `gatedtrain.testing` no longer makes CLI processes ignore
  the host file (`~/.config/gatedtrain/host.toml`); only the test helpers (`gatedtrain.testing.support`, `.ext`)
  default an unset `GATEDTRAIN_HOST_CONFIG` to an empty host file.

Rules: EXT-263, AN-17.

## Changes in 0.4.0

The analysis code version stays `gatedtrain 0.3.0`. A registration records `analysis_code_version` and is refused
when it differs from the running code; from this release that version names the gate statistics and registration
semantics, not the package release (`gatedtrain.ANALYSIS_VERSION`; design 05 section 3.7). It changes only when that
code's behaviour changes, so a registration made with 0.3.0 passes the analysis code version check under 0.4.0,
and one whose evaluators are Tier B, external-run or recipe evaluators stays valid. A Tier A stand-in (demo)
registration made with 0.3.0 does not: the stand-in evaluator's identity hash covers the stand-in model code, which
changed after 0.3.0 (oneDNN turned off in the CPU demo), so it pins an old hash and gates MISSING
(`evaluator_hash_mismatch`); register it again. The stand-in hash names the analysis version, not the package
release, so later releases do not change it by themselves. Lockfile backends, the ledger `writer_id`, the keeper's
`keeper_version`, the `setup keeper` install pin `gatedtrain==<version>` and `--version` still name the package
release that wrote them.

New required dependency: `cryptography>=50.0.2` (Ed25519 signatures of the AI-native keeper).

New in this release:

- Resumable external runs: `ext-collect` reads a resumed run as a chain of attempts with
  their cuts, keeps the records after a non-final attempt's last committed cut in the ledger as `orphaned` (consumption
  ledger schema 1.1; a 1.0 ledger reads as kept rows), and `impact --json` adds an `orphaned` section for resumed runs.
- AI-native projects, a preview: step 1, `gatedtrain project` (in process, CPU only), and step 2, the keeper daemon
  that holds the sealed items, registrations and alpha ledger and signs its records (`keeper`, `setup keeper`,
  `verify`, Linux only). See [AI-native projects (preview)](docs/ai-native.md). AI-native code is new in 0.4.0; a
  registration made with the library scorers of `gatedtrain.ainative.scorers` on unreleased main before 0.4.0 pins
  an older evaluator hash, which no longer matches the scorer: register it again (a new `preregister`).

Breaking for arm writers and for `ext-collect` ([MULTIMODAL-SPEC](MULTIMODAL-SPEC.md); [findings](docs/findings.md),
"Review of the 3.12 path fix" and "Review of PR 2"):

- A symbolic link anywhere in an `output_ref` path, or as an arm's `items.jsonl`, `eval_meta.json` or `merge.json`,
  is refused, also one that stays inside the arm. Write copies or hard links.
- `output_ref` paths must be in canonical form: relative, UTF-8, no NUL, no `.`, `..` or empty component
  (`outputs/x.png`, not `./outputs//x.png`).
- A merged arm's sources (`merge.json` `sources[].path`, or `run_dir` for `ext-from-nel`) must be absolute paths in
  normal form under the host's output root, or under the directory given to `ext-collect --sources-root` (a relative
  one is taken from the working directory). Arms that 0.3.0 merged from paths spelled with `.` or `..` record such
  paths and must be merged again.
- `ext-from-nel` reads each NeMo Evaluator run once, without following links: a `full_config.yaml` or
  `results.jsonl` that is a symbolic link, a FIFO or another non-regular file is refused, and the hashes recorded in
  `merge.json` are those of the bytes checked.
- Tests run as any non-root user (the launch fixtures run the container as the running user and its primary group).
  With `GATEDTRAIN_REQUIRE_CONTAINER_USER=1`, as in CI, the tests that need such a user fail instead of skipping.

Rules: EXT-RES-1 to EXT-RES-7 (resumable runs), MM-1 and MM-5 (arm writers).

Earlier releases are listed by their [git tags](https://github.com/gatedtrain/gatedtrain/tags).
