# Changelog

Release notes, newest first. Each release lists what is new and what changes for users of the earlier release. The
"Rules" line at the end of a release names the rules behind the changes; the [test reference](tests/README.md) maps
each rule to its tests.

## Changes in 0.7.1

On top of 0.7.0. Implements design revision 28 (tag `rev-28` of gatedtrain-design).

Owner-side settings that used to be code constants or environment variables are now keys of the host file. Defaults
and every recorded hash are unchanged unless a key is set:

- **Host file keys.** `gpu_mem_limit_gb` and `gpu_cap_h` (numbers above 0), `input_roots`, `interpreter_trees`,
  `interpreter`, `mirror_root`, and `[packs.<name>]` tables of absolute paths. `smoke_verdict` and `smoke_problems`
  take the bounds as parameters; `ext-launch` and `ext-collect` pass the host's. The numbers and the pack tables enter
  `host_config_sha256` only when set.
- **The two numbers are owner-side and bounded.** `gpu_mem_limit_gb` is at most 1024 and `gpu_cap_h` at most 100000
  (`launch.MAX_GPU_MEM_LIMIT_GB`, `launch.MAX_GPU_CAP_H`; fixed, not settings): a larger value is refused, because a
  bound no real card or node can reach checks nothing. Only the owner's host file sets them, no variable or flag does,
  and `host_config_sha256` binds them. `GATEDTRAIN_GPU_CAP_H` in the printed command is now the exact value (it was
  rounded to six digits, so 1000.4 was enforced as 1000), in plain decimal.
- **Deprecated variables.** The `GATEDTRAIN_EXT_*` layout variables are deprecated and removed in 0.9.0, and so are a
  pack's runtime variables (for example `GATEDTRAIN_PACK_TEXT_GSM8K_RUNTIME`). Every `ext-*` command prints a stderr
  line naming each one that is set and the host file key that replaces it. A variable and its host file key
  (`input_roots`, `interpreter_trees`, `interpreter`, `mirror_root`, `[packs.<name>] runtime_dir`) set to different
  values is refused (EXT-HOST): the variable never outranks what the owner wrote. Equal values, or either alone, work.
  `GATEDTRAIN_EXT_HOST_WORK` and `GATEDTRAIN_EXT_OUTPUT_ROOT` still override the file, as in 0.7.0.
- **Keeper options.** `keeper serve` and `setup keeper` take `--gpu-wait-s`, `--gpu-busy-mib` and `--eval-python`; a
  non-default value is in the signed environment record. The wait is bounded by the client's deadline less the slack,
  and the busy threshold to 1 TiB (a threshold above the GPU's memory still turns the check off: it is signed when
  set). A bad flag value is reported under its own flag. The three layout rows 0.7.0 reserved for them (`[gpu] wait_s`,
  `[gpu] busy_mib`, `[inference] keeper_eval_python`) are supported in both `setup keeper` modes, validated like the
  flags and printed into the unit; a flag beats the layout; they need `--inference-image`.
  `rowkit.runtime_dir(..., pack=)` reads the host file's `[packs.<name>] runtime_dir`. `launch.pack_setting` reads any
  pack setting.
- **Downgrading.** gatedtrain 0.7.0 refuses a host file that holds any key new in 0.7.1 (it lists the keys it knows);
  remove the new keys before going back.

Constants that were copied or tied to one machine now have one name and one place. Nothing here changes a registration,
a signature, a ledger or a report, except where marked as a change:

- **One source for each shared constant.** The alpha-investing starting wealth and payout are computed from
  `TARGET_ALPHA`, and the guarantee texts quote it; `PILOT_SIGMA` is the scorers' `PLANNING_SD_BOUND`; the keeper's long
  operation timeout is the sum of `GPU_WAIT_S`, two generator time limits and `EVAL_SLACK_S`; the 64 MiB data limit, the
  96 MiB request cap derived from it and the predictions cap live in `ainative/limits.py`; the container shared-memory
  size, the GPU-hour ceiling, the NeMo Evaluator commit, the bootstrap variants, the metric directions, the prototype
  tags and the default rows per shard are each defined once. `tests/test_single_source.py` pins the trust constants and
  checks every one of these.
- **Tier B launcher (`backends.nemo_rl.launch_spec`, prototype).** The docker launcher no longer defaults `work_dir` and
  `hf_cache` to directories of one host: both are required and a call without them raises `ValueError`. The GPU
  architecture list is no longer fixed at `8.9`: pass `cuda_arch` (for example `"8.9"`) to export `TORCH_CUDA_ARCH_LIST`
  and `FLASHINFER_CUDA_ARCH_LIST`; without it the command has neither. `shm_size` is a parameter (default `32g`).
  `DEFAULT_WORK_DIR` and `DEFAULT_HF_CACHE` are removed.
- The NeMo RL commit is written once (`families.nemo_rl.NEMO_RL_COMMIT`; the short form, the image tag and the source
  directory derive from it); the executor's option list is derived from its limit table; job defaults (`6 h`, 3 attempts,
  poll interval) and the rootless socket address (`launch.rootless_docker_host`) have one name each; `PROJECT_FILE` names
  `project.yaml`. The text of `gatedtrain setup host` is byte-identical to 0.7.0 apart from the version number it prints (checked for both
  script variants).
- `docs/configuration.md` lists the fixed trust values and why they are not settings.
- `gatedtrain export-gsm8k` takes its default rows per shard from `tierb.DEFAULT_ROWS_PER_SHARD`; the value is unchanged.

Rules: AN-35, EXT-264, AN-34 (three rows supported).

## Changes in 0.7.0

Implements design revision 27 (tag `rev-27` of gatedtrain-design). Rules AN-27 to AN-34 and IMG-1 to IMG-9.

The AI-native path no longer needs sudo by default, and a result states the keeper mode that produced it (see [Keeper
modes](docs/ai-native.md#keeper-modes) and the [quickstart without sudo](docs/sudo-free-quickstart.md)):

- **Behaviour change: `gatedtrain setup keeper` now prints a same-user setup by default.** The commands run without sudo:
  a keeper under your own OS user, a state directory under your XDG data directory (mode 0700), a systemd user unit and
  the socket under `$XDG_RUNTIME_DIR`. The old output (the keeper user, the root-owned environment, `sudo`) is
  `gatedtrain setup keeper --mode separate-user`, unchanged. `--user` and `--venv` are refused in same-user mode
  (AN-30), `--packages` is ignored there with a warning (the keeper runs from the current environment). Scripts that
  parsed the old default output must pass `--mode separate-user`; without `--mode` the command adds one warning saying
  so (JSON `warnings`, stderr in text mode), and an explicit `--mode` has none. The printed same-user block is one
  script: its first line is `set -e` and every check exits non-zero, so a state directory that is shared or not yours
  stops it before the unit is written and `systemctl --user enable --now` runs (run it with `sh FILE`, or paste it
  whole). The `gatedtrain` command the unit runs is the one next to the Python that ran the setup (the PATH is not
  searched, another environment's command could be there); `--executable PATH` (layout: `[same_user] executable`)
  names another one, for example a link at a plain path when the venv's path holds a space, `%` or `$`. Only the
  paths that reach `ExecStart=` must be plain: a home or `XDG_CONFIG_HOME` with a space no longer blocks the unit
  directory. A path or Docker host with a non-printable character (escape, NUL, a line or bidi separator) is refused in
  both setup commands and the layout file, and the printed commands escape any that remains.
- **The keeper mode is an observed fact, signed and sticky.** The keeper derives `separate_user` or `same_user` from
  its uids, its state directory and the uid of each peer, never from an option (AN-27); the mode is in `hello`
  (`keeper_mode`, `mode_facts`, `modes_seen`; `same_user` stays), in the signed environment record and in every
  registration, claim and report. A sealed set only ever gets weaker: once served in same-user mode it stays so, and a
  set served in both modes says so. The client reads the kernel's credentials of the listener on the socket
  (`SO_PEERCRED`) and a claim of separate-user mode never upgrades a result without it; a keeper that claims another uid
  while the kernel names the client's own is refused (AN-29).
- **New guarantee text, legacy text still recognised.** A same-user keeper states "the keeper ran under the same OS user
  as the owner (same-user mode). It followed the procedure: ... these records show that the procedure was followed, not
  that the sealed data stayed unseen. ..." instead of "(a test or development setup)". Reports with the old text still
  verify and still warn. Every surface prints a mode line (`keeper fingerprint`, `project create --keeper`, `project
  plan`, every stage of `project run`, `project status`, `keeper confirm`, `report.md`, `project report`, `verify`) and,
  in the JSON envelope, `data.keeper_mode`; in same-user mode `project run`, `project status`, `project report`,
  `project plan` and `project create` also add the mode line to `warnings`, and a same-user PASS (a gate line or a
  report headline) carries the qualifier. The intake plan no longer says the data "will be sealed" without it.
- **`--layout FILE` for `setup host` and `setup keeper` (AN-34).** One TOML file replaces the path flags (`gatedtrain
  setup layout` prints a complete, commented one with the effective defaults). A flag beats the file, the file beats the
  default; the printed commands start with the file's path and SHA-256 and the source of each non-default value, and
  `data.layout` records the same. `schema_version = 1` is required; the file is read without following symlinks, has a
  size cap, and is only printed, never read by the keeper. The `[gpu]` table (`wait_s`, `busy_mib`) and
  `[inference] keeper_eval_python` are reserved: a layout that sets them is refused until the keeper has those options.
  `[host] eval_python_root` is new (layout only). `setup host`'s flags no longer show their defaults as argparse
  defaults; the output is unchanged.
- **Report `schema_version` 3.** It adds `keeper_mode`, `modes_seen`, a `keeper_mode` per attempt and
  `require_keeper_mode`. Version 2 reports are still verified, their mode read from the signed guarantee text; a
  version 3 report whose mode contradicts its text is refused (AN-14).
- **`project create --require-keeper-mode separate-user`** (default `any`) fixes, in the registration, that the project
  accepts a separate-user keeper only. It is refused up front on a same-user keeper, enforced by the keeper at
  `evaluate_sealed`, `gate` and `register`, and by `verify` (exit 3), also under `--accept-mode` (AN-28).
- **`gatedtrain verify --accept-mode same-user`.** Exit 0 stays reserved for a separate-user result. A same-user result
  still exits 1; with the option, exactly that result (not a rootful-access, revision 23 or unknown text) exits 0 and
  ends with `ACCEPTED WEAKER MODE: same-user (...)` and keeps the consistency warnings of the strict path (an anchor
  that is not the keeper still warns that deleted records cannot be seen); the JSON data has `accepted_weaker_mode`,
  `keeper_mode` (the effective mode, never the keeper's unverified claim), `effective_mode`, `mode_basis` and `mode_sources`
  (`keeper_claim`, `kernel_attestation`, `report`, `receipts`). The effective mode is the weakest of all of them: the
  `keeper_mode` of every signed registration and claim receipt counts too, a report stronger than the receipts it
  covers or with contradicting `modes_seen` or attempts is refused (AN-14), and a project with keeper receipts but no
  stated mode says "mode unknown" and exits 1. The project commands print the weaker of the keeper's statement and
  this client's check; the keeper's set mode also reads the `keeper_mode` of its signed claims and
  registrations (so deleting `mode_seen.json` cannot make a set stronger) and `intake` enforces `--require-keeper-mode`.
- **`gatedtrain setup check`** names the outside prerequisites (rootless Docker, `uidmap`, subordinate ids, user
  namespaces, rootless networking, the NVIDIA driver and CDI spec, the systemd user session, lingering, the memlock
  limit, `ptrace_scope`), says which ones an administrator has to provide, reads the host only and exits 1 when a
  required one is missing or unknown (AN-32, AN-33).
- **The image lock replaces hard-coded image IDs.** Code names logical images (`nemo-rl/tierb`,
  `nemo-rl/tierb-uid1000`, `nemo-rl/tierb-rootless`); the accepted IDs live in `data/image-lock.json`, shipped with the
  release, plus your own `$XDG_CONFIG_HOME/gatedtrain/images.json` that can only add user-built images; no environment
  variable, flag or project file supplies an accepted ID, and the config cannot name an ID the lock lists (IMG-1 to
  IMG-4). The module attributes the published packs import
  (`NEMO_RL_IMAGE`, `NEMO_RL_IMAGES`, `NEMO_RL_UID1000_IMAGE`) are kept, computed from the lock; `accepted_images(logical)`
  also reads your config. Registrations that name the two earlier IDs keep verifying.
- **`gatedtrain --json image list` prints the sha256 of the lock it read** (`lock_sha256`), informational until
  release hashes are published to compare it with. (`--json` is a top-level option: `image list --json` is a usage
  error; the image READMEs and the build script's usage said so wrongly and are corrected, and a test now parses every
  documented `gatedtrain` command line with the real parser.) The `nemo-rl/tierb-uid1000` lock entry records no
  Containerfile hash: the maintainer image was built from the file before it took `BASE` as an argument (same chown
  list), so the shipped file's hash would claim a build that did not happen. `containers/nemo-rl-tierb/build.sh` prints
  the `image add` command for a tag that already existed WITHOUT `--recipe-commit` and `--containerfile-sha256`
  (nothing in that run vouches for the image), so `image add` refuses it until you supply the true provenance.
- **Report requests degrade one thing at a time.** A project directory under `jobs/` without its `job.json` (a submit
  cut short) is no job and no longer fails the image list; when only the trainer images cannot be read the job totals
  are still sent, with a warning; the retry without `images` happens only for an older keeper's exact `compute`
  refusal (a keeper that predates `images`), never for another AN-10 refusal. Warnings raised below a command are
  queued in `output.notice_scope()` (entered by `cli.main`) and carried by `emit`; a library caller outside a scope gets
  them on stderr at once, or reads them with `output.drain_notices()` inside its own scope.
- **`gatedtrain image list|check|add`** (IMG-5, IMG-6). `image add` records an image you built, at a terminal or with
  `--yes`, and only adds. It is recorded for provenance only: packs adopt `accepted_images(logical)` in a later release,
  so for now it changes nothing for gating.
- **Image provenance in the report.** The driver declares each trainer image of the project's jobs, the keeper checks it
  against its own copy of the maintainer lock and signs both, `report.md` has a line per image, and `verify` warns about
  a user-added image or for jobs without a declared image block; `--require-maintainer-images` makes either exit 1, and
  also exits 1 for a report that lists no image at all (no backend totals, zero jobs, an empty list). The keeper signs
  no row that contradicts itself (an unlisted ID is never `maintainer-built` or `source: lock`; a repeated ID is
  refused), `verify` refuses a malformed image entry (AN-14) instead of crashing, and the report carries the sha256 of
  the keeper's shipped lock, the hash of the very bytes it checked, which `verify` prints with the note that in
  same-user mode the file is writable by you and no release hash list is published yet (IMG-7 to IMG-9).
- **Build recipes** in `containers/`: `nemo-rl-tierb` (a script and notes to build the Tier B image), `nemo-rl-rootless`
  (the derived image for rootless Docker) and the updated `nemo-rl-uid1000`. New pages: [Quickstart without
  sudo](docs/sudo-free-quickstart.md) and [NeMo RL on a rootless host](docs/rootless-nemo-rl.md).

Known limits (read before relying on a same-user result):

- **Same-user mode is meaningfully weaker.** Against an agent with a shell, it does not protect the sealed items, the
  labels, the alpha ledger or the signing key; it shows that the procedure was followed, not that the data stayed
  unseen. The reports and every surface say so; harden with `--mode separate-user` and new data.
- **The kernel check (AN-29) holds for the reader's own account and the account that owns the project files.** It
  compares the listener's uid with the verifier's own and with the uid that owns the project directory and
  `trust/keeper.json` (the account that wrote the records): a keeper of either uid that claims separate-user mode is
  refused. A reader under a third account (an admin, CI) must anchor with a fingerprint taken from the real keeper.
- **Offline, `separate_user` means "stated by the key holder".** With `--fingerprint` or no anchor, the mode is what the
  signed records say; on a same-user keeper the key holder is the user's own account, so the statement can be forged by
  the account it is meant to guard against, and a fingerprint kept away from the agent does not prevent it (the key is
  readable by the user). Such a result exits 0 with the warning "the mode is as stated by the holder of the key", the
  line `mode as stated in the signed records: separate_user` and `data.mode_basis` (`kernel_attestation`,
  `signed_records` or `report_text`). `keeper fingerprint` prints this limit. Use `--keeper` against a separate-user
  keeper.
- **Kernel attestation is Linux-only.** Where the kernel's answer about the listener is unavailable, a keeper's claim of
  separate-user mode never upgrades the result: it counts as same-user, noted as unavailable.
- **Image provenance is the driver's declaration, checked only against the keeper's own lock.** The keeper recognises
  an image ID (it is in its lock or it is not); it never saw which image ran, and the job records are the driver's.
  `report.md` and `verify` say "not observed: job records are the driver's". The lock's sha256 that `verify` and
  `image list` print is informational until release hashes are published.
- **The owner image config is not a security boundary against an agent running as the owner.** `HOME` and
  `XDG_CONFIG_HOME` locate it, so a same-user agent can run `image add --yes` or write it. The reliable signal is the
  keeper's `user_added` flag in the signed report (with the lock's sha256), which holds when the keeper's code is outside
  the agent's reach (a separate-user keeper). `--require-maintainer-images` is the strict reading.
- **A report without an image block is only a warning unless `--require-maintainer-images` is given.** The warning is
  for a report that counts jobs but lists no image; a report that ran no jobs is silent without the flag (a warning on
  every such report would carry no information) and fails with it. The job records are in the agent-writable project;
  the image block is the driver's declaration, not proof of what ran. A project with no report at all fails
  `--require-maintainer-images` ("no report states any trainer image").
- **The build recipes need a source checkout.** `containers/` is not shipped in the wheel or sdist (only the lock is).
- **A source build of the trainer image is not bit-reproducible.** Base images are pulled by tag, so its ID is host and
  time specific; the recipe provenance, not the ID, is the audit trail.
- **`NoNewPrivileges=yes` in the same-user keeper unit is unverified against rootless Docker** (no Docker was available
  to try it); drop the line if the keeper cannot start its inference containers.
- **The packs are not migrated yet.** The published scenario packs keep their own image lists (and the environment
  variable one of them reads) until they adopt the lock's accepted-images call.
- **The `gatedtrain-agent` MCP server and operator skill text are not updated.** They do not yet tell the agent to quote
  the keeper mode, and still say that setup needs sudo.
- **The separate-user path could not be tested with a real second OS user in CI.** The tests run the keeper as the test
  user and inject the kernel's answer; the separate-user property is checked by hand on a host with a real keeper user.
- **What the same-user guarantee covers about the model under test.** A batch generator's model runs in a container
  without network access that is given the inputs only, never the labels. A per-item generator runs inside the keeper
  process, without that confinement; the signed text says so. Both are inside the user's account in same-user mode.
- **Batch packs need `--inference-image` in same-user mode too.** Without `setup keeper --inference-image sha256:...`
  (and a rootless Docker for the user) the keeper refuses a batch generator with AN-8; per-item packs run without it.
- **Version skew.** A 0.6.0 driver refuses a version 3 keeper report (AN-10), and a 0.6.0 `verify` fails with INTERNAL
  on a project made by this release: upgrade the clients (the driver, `verify`) before or together with the keeper. A
  new driver binds to a 0.6.0 keeper (the default requirement sends no argument), but `--require-keeper-mode
  separate-user` is refused against it ("keeper predates AN-28; upgrade it").
- **Upgrading an existing keeper needs a new setup.** A separate-user keeper's virtual environment is root-owned, so
  `gatedtrain setup keeper --mode separate-user` has to be run again for the new code; nothing upgrades it in place.
- **Unfinished legacy projects.** A project whose receipts state no keeper mode (made before this release) and that has
  no report now exits 1 with "mode unknown" on an offline `verify --fingerprint`; verify it with `--keeper` against the
  keeper, or finish it so that a report states the mode.
- **Legacy sealed sets take the current observation.** A set made before this release has no `mode_seen.json`, so its
  mode starts from what the keeper observes now. Do not move a same-user legacy state to a separate-user keeper: the
  set would be called separate-user though it was held by a keeper the user's account could read. Make a new project
  and take the data in again.

Rules: AN-27 to AN-33, AN-14 (extended), AN-12 (extended), IMG-1 to IMG-9.

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
