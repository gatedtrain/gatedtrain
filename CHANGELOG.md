# Changelog

## Changes in 0.5.0

The analysis code version stays `gatedtrain 0.3.0` (the gate statistics and registration semantics did not change),
so registrations made with 0.3.0 or 0.4.0 stay valid.

AI-native step 3a, a preview on Linux (see "Rootless containers, the local backend and the keeper's container (step
3a)" below):

- Rootless host setup: `gatedtrain setup host` prints one root script that sets up a rootless Docker daemon for the
  owner and one for the keeper user and removes the owner from the rootful `docker` group. gatedtrain never runs it.
- The local job backend: pack stages run their jobs in containers on the owner's rootless daemon (`ctx.backend`),
  with GPU accounting against the project's budget and a row per job in the host's GPU log; `gatedtrain project job
  list | logs | cancel` show and stop them.
- The keeper's container: `keeper serve --inference-image ...` runs a scenario pack's batch generator in a container
  of the keeper's own rootless daemon, on a copy of the model files the keeper hashed.
- `ext-launch` on a rootless host (`rootless = true` in the host file): the launch rules accept the owner's rootless
  daemon and print `--device nvidia.com/gpu=all` in place of `--runtime=nvidia --gpus all` (EXT-263).

Behaviour that changes for 0.4.0 users:

- `verify` exits 1 for a report signed by a 0.4.0 keeper: that report states the 0.4.0 guarantee, which was stated
  before rootful Docker access was checked. It also exits 1 against a 0.4.0 keeper (its `hello` does not state
  `owner_has_root_docker`), and for a report of a keeper that runs as the owner's own user (a test or development
  setup; before, only `--keeper` against such a keeper exited 1). `verify` exits 0 only when every report states the
  separate-user guarantee.
- Upgrade the keeper's environment (the venv `setup keeper` installed) together with the driver: the driver now sends
  the project with `hello`, and a 0.4.0 keeper states no rootful-access check.
- A scenario pack scorer whose id is also an ext evaluator row id is refused (AN-17); before, the row shadowed the
  scorer at registration and `preregister` refused with an unclear message. Give the scorer its own id.

Fixes:

- The credential file-name scan no longer walks a read-only source that is itself one of the adapter's `scan_skip`
  directories (for example a job that mounts a runtime's `.venv` directly); the source's own name is still checked.
- `keeper confirm` without `--yes` and with no answer on stdin (a shell started without input) now says no answer was
  read and to rerun with `--yes` after reading the sentence.
- Importing a scenario pack or ext adapter plugin that uses `gatedtrain.testing` no longer makes CLI processes ignore
  the host file (`~/.config/gatedtrain/host.toml`); only the test helpers (`gatedtrain.testing.support`, `.ext`)
  default an unset `GATEDTRAIN_HOST_CONFIG` to an empty host file.

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

- Resumable external runs (EXT-RES-1 to EXT-RES-7): `ext-collect` reads a resumed run as a chain of attempts with
  their cuts, keeps the records after a non-final attempt's last committed cut in the ledger as `orphaned` (consumption
  ledger schema 1.1; a 1.0 ledger reads as kept rows), and `impact --json` adds an `orphaned` section for resumed runs.
- AI-native projects, a preview: step 1, `gatedtrain project` (in process, CPU only), and step 2, the keeper daemon
  that holds the sealed items, registrations and alpha ledger and signs its records (`keeper`, `setup keeper`,
  `verify`, Linux only). See "AI-native projects (preview)" below. AI-native code is new in 0.4.0; a registration
  made with the library scorers of `gatedtrain.ainative.scorers` on unreleased main before 0.4.0 pins an older
  evaluator hash, which no longer matches the scorer: register it again (a new `preregister`).

Breaking for arm writers and for `ext-collect` (MULTIMODAL-SPEC MM-1, MM-5; docs/findings.md,
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


Earlier releases are listed by their git tags.
