# gatedtrain

gatedtrain records what went into a model training run and tests whether the resulting checkpoint meets a
pre-registered evaluation criterion. It works with external trainers and evaluators. You can query the records
to find which checkpoints may have consumed a training shard, for example after finding a problem in the dataset.

This is a prototype. Outputs carry the tags `prototype` and `unpromotable`.

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

## Install and run the demo

From a checkout with Python 3.12 or 3.13, create a virtual environment and validate the demo spec.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
gatedtrain validate demo/spec.yaml
```

To run the demo end to end, install the `tier-a` extra for PyTorch. The example trains a small model on two-digit
addition. Run these commands from the repository root with a fresh `demo/baseline` path.

```sh
python -m pip install -e '.[tier-a]'
gatedtrain warmup demo/warmup.yaml --out demo/baseline
gatedtrain run demo/spec.yaml --runs-root demo/runs
gatedtrain impact demo/data/train/shard-00003.jsonl --runs-root demo/runs
```

The `run` command prints its output directory and gate report. See the [CPU quickstart](docs/quickstart.md) for
the report fields and control runs, or the [Tier B runbook](docs/tier-b-runbook.md) for a GPU run with NeMo RL.

## Commands

`gatedtrain` (or `python -m gatedtrain`) has these subcommands:

| Command | What it does |
|---|---|
| `validate`, `warmup`, `run` | Check the spec and registration, build a baseline, or run the CPU demo. |
| `export-gsm8k`, `verify-export`, `nel-config`, `merge-eval`, `tierb-collect` | Prepare data and evaluation for a NeMo RL run and collect its results. |
| `ext-launch` | Check an external run request, write the launch record and print the command for the operator. Use `--recheck` to verify the record before running it. |
| `ext-from-nel`, `ext-merge-eval`, `ext-arm-fill` | Build and verify the per-item results for each side of the evaluation. |
| `ext-collect` | Compare the candidate and baseline on the same evaluation items, then write the gate report, consumption ledger and checkpoint lineage. |
| `project` | AI-native projects (preview): `create`, `plan`, `run`, `status`, `report`. See below. |
| `keeper`, `setup keeper`, `verify` | AI-native keeper (preview, Linux only): `keeper serve`, `keeper fingerprint`, `keeper confirm`; print the install commands; check a keeper project's signed records. See below. |
| `impact` | Look up a shard across recorded runs. Each answer reports its precision as `row`, `iteration` or `stage`. |

Gates support superiority, non-inferiority and equivalence. The registration fixes the margin, alpha, target
power and evaluation seeds before the run. The report includes the decision, achieved power and an
`underpowered` flag.

## AI-native projects (preview)

`gatedtrain project` runs one goal as a project directory: a fixed pipeline of five trust stages (`intake_seal`,
`preregister`, `evaluate_sealed`, `gate`, `report`) around the stages a pipeline file plugs in. Step 1 is a preview
for CPU-only work and the synthetic scenario pack in `gatedtrain.testing`, which is not registered as an entry point.

```sh
gatedtrain project create ./proj --goal "..." --data labelled.jsonl --pipeline pipeline.yaml [--gpu-hours H]
gatedtrain project plan ./proj <stage>                       # what a stage would do; changes nothing
gatedtrain project run ./proj <stage> [--decision file.json] # stages run in pipeline order
gatedtrain project status ./proj
gatedtrain project report ./proj
```

`--pipeline` is the path of a pipeline YAML. `--gpu-hours` is recorded in
`project.yaml` as the budget; step 1 enforces nothing. `run` of the `gate` stage exits 0 on PASS and 1 otherwise;
a refusal exits 3. Re-running a stage with unchanged inputs returns its recorded result.

Without a keeper (step 1, in process), a project scores its sealed items once:
after the first scoring `evaluate_sealed` refuses to score again, for the same model reference too (a rerun with
unchanged inputs returns the recorded scores). A model reference must content-address its weights (for example
carry their hash); step 1 does not check this, so a reference to a mutable checkpoint path lets the gate records
describe weights the reference no longer points to after a retrain. The gate spends the project's single alpha: a second gate attempt is refused. A project bound to a keeper
follows the keeper's alpha rule instead (below). A top-level `--json` prints one result object per call.

**Trust boundary in step 1.** Stages run in the operator's process. CLI outputs carry only hashes of the sealed
items, but in step 1 they sit in plain files under `<root>/trust/` that the operator and any stage can read and
rewrite, as can the attempt log; the report's `sealed_guarantee` says so ("unsealed (step 1): the operator process
can read the sealed items, and every stage can read and rewrite the trust records and the attempt log").
A project bound to a keeper (below) moves these records out of the operator's reach. In step 1 the alpha is per project: a new project on the same data
would get a fresh alpha and the same sealed split, so a new test needs new sealed data. The report's known limits
list these points, the model-reference rule, that the metric confirmation is the decision flag `metric_confirmed` set by whoever runs the
stages, and how many `evaluate_sealed` attempts the project made.

`project create` copies the pipeline into `<root>/pipeline.yaml` and records its sha256; the project refuses a copy
that was changed. To use another pipeline, create a new project. A stage that refuses in `run` before it writes any
record, spends alpha or claims a trust record is logged as `refused` and changes no stage state; a refusal after any
of these, or from the output checks, is `failed`.

### The keeper (step 2)

The keeper is a daemon under its own OS user (Linux only: a Unix socket with a peer-credential check). It alone holds
the sealed items, the registrations, the claims, the per-item records, the alpha ledger and an Ed25519 signing key;
the stages of a bound project become requests to it, and it signs every record it returns. Only the owner's uid and
its own may connect.

```sh
gatedtrain setup keeper [--owner-uid UID] [--packages NAME==VERSION ...]   # prints sudo commands; runs nothing
gatedtrain keeper fingerprint --socket /run/gatedtrain-keeper/keeper.sock    # keep the fingerprint out of band
gatedtrain project create ./proj --goal "..." --data labelled.jsonl --pipeline pipeline.yaml \
    --keeper /run/gatedtrain-keeper/keeper.sock
gatedtrain keeper confirm ./proj --decision preregister.json [--yes]          # before `run ./proj preregister`
gatedtrain verify ./proj --keeper /run/gatedtrain-keeper/keeper.sock          # or --fingerprint sha256:...
```

`setup keeper` prints the commands that create the keeper user, its 0700 state directory, a root-owned virtual
environment with gatedtrain and the scenario packs pinned to versions, and a systemd unit running `gatedtrain keeper
serve`; it writes nothing. The keeper refuses to start when its code, environment, state directory or socket
directory can be changed by another user (AN-13); a path holding `%`, `$`, whitespace, a quote or a backslash is
refused by `setup keeper`, since systemd would rewrite it in the unit. Known limit of that check: a module first
imported after the start-up check that lies outside gatedtrain, the installed scenario packs and gatedtrain's runtime
dependencies is not checked, and site-packages is not walked as a whole. `project create --keeper` binds the project to the key the keeper serves (`trust/keeper.json`,
`trust/keeper.pub`); a bound project never falls back to step 1: an unreachable keeper (AN-12) or one serving another
key (AN-14) is refused. `keeper confirm` shows the sentence that says what the scorer counts and the margin, and
records it in the keeper only after you type `yes` (or pass `--yes`; under `--json` it never prompts); `preregister`
refuses a registration without a matching confirmation (AN-18). The scorer comes from the library (`exact_match`,
`numeric_match`, `normalised_match`, `set_f1`, `field_f1`) or from a pack by reference (AN-17).

**Alpha rule.** Under the keeper the alpha is not per project but per sealed set, by alpha-investing (Foster-Stine):
the initial wealth is W0 = 0.05 * (1 - 0.05) = 0.0475; attempt j bids alpha_j = W_{j-1} / 2 (at most
W_{j-1} / (1 + W_{j-1}), so a bid never costs more than the wealth; this binds only above a wealth of 1); a FAIL, INCONCLUSIVE,
MISSING or crashed attempt costs alpha_j / (1 - alpha_j); a PASS earns the payout 0.05; an attempt whose bid would
be below 0.005 is refused. Intake plans power at W0 / 2 = 0.02375. Each `evaluate_sealed` spends its alpha before any
item is scored, so a project may gate several candidates (rerun `train`, then `evaluate_sealed` and `gate`)
until one passes; a PASS ends the project's attempts. A new project on byte-identical data shares the sealed set and its wealth;
other data holding a sealed item are refused (AN-16). The report lists every attempt with its alpha, cost and payout
and states the guarantee: "alpha-investing (Foster-Stine) over the gate attempts on this sealed set controls the
marginal false discovery rate (mFDR) at level 0.05". Its known limits qualify this: it assumes each test is valid given
the earlier outcomes; the sealed items are reused across attempts and only each attempt's status is revealed. A report
closes the sealed set: no further attempt on it, from any project, and a new project on the same data is refused at
intake.

**Oracles.** The keeper answers some requests about sealed data with counts and does not limit how often they are
asked: opening a project is free and a refused intake stores nothing, so anything running as the user can learn from
intake refusals how many inputs of a candidate data file were sealed before (and, repeated, which); a registration
refused for its scorer names how many sealed labels that scorer cannot take. The report lists this in its known
limits.

**Guarantee.** With the keeper under its own user, the report's `sealed_guarantee` reads: "the sealed items,
registrations, claims, ledger and per-item records are held by a separate OS user running code the user's account
cannot change, and every record is signed; the candidate runs inside the keeper process, not in a container (a model
format that executes code is refused); the original data file and the metric confirmation are only as strong as the
user's own account; a root user or the keeper's own account can break all of it." A keeper running as the owner's own
user (as in the tests) says so instead: "the keeper runs as the owner's own OS user, so anything running as the user
can read its state and signing key: the records are signed, but nothing is sealed (a test or development setup)".

**`verify`.** `verify ./proj --keeper SOCKET` checks every signed record in the project (the registration, each
receipt in `trust/keeper/records.jsonl`, the gate result, `report.json` and `report.md`) against the key the keeper
serves, and that the project holds every record the keeper signed for it; `--fingerprint` checks against a
fingerprint held out of band, offline, and cannot detect a deleted record (it says so). It exits 0 when every record
checks, 3 on a bad signature, an edited record, a replaced `trust/keeper.pub` or another fingerprint (AN-14), and 1
without either anchor or against a keeper that runs as the owner's own uid (consistency only: the owner can read that
keeper's key). The driver keeps every envelope the keeper signed for the project: after each `evaluate_sealed`, `gate`
and `report`, and on `project report`, it fetches the keeper's list and adds what it lacks (for example the claim of an
evaluation it stopped waiting for), each verified first. A record the keeper signed that the project still lacks fails
`verify --keeper` with exit 3.

## Adapter plugins

Each external training recipe has an adapter: a pure-Python class that reads the run's outputs (checkpoints,
consumption logs, evaluation records) without importing the training framework, plus registry rows that describe the
run (inputs, container paths, pins, evaluators). A recipe package registers them through the Python entry point group
`gatedtrain.adapters`:

```toml
[project.entry-points."gatedtrain.adapters"]
my_adapter = "my_recipe.my_adapter_rows:PLUGIN"
```

where `PLUGIN = {"adapters": {name: "module:Class"}, "trainer_rows": {name: row}, "evaluator_rows": {id: row}}`
(`gatedtrain.ext.registry`). Adapters use the helpers of `gatedtrain.ext.sdk` and `gatedtrain.ext.rowkit`.

**Conformance kit.** A recipe repository can run the conformance checks with its own tests, so that a plugin that
registers but would break `ext-launch`, `ext-arm-fill` or `ext-collect` fails before any GPU run:

```python
import unittest
from gatedtrain.ext.conformance import AdapterConformance

class TestConformance(AdapterConformance, unittest.TestCase):
    ADAPTER = "my_adapter"
```

`gatedtrain.testing` provides the shared test helpers (canonical JSON and manifest hashes, CLI subprocess runs) and a
synthetic adapter, `gatedtrain.testing.synthetic`: a complete, small example of the adapter interface that
gatedtrain's own tests run on. It is registered only when a test asks for it (`install()` in process, or
`python -m gatedtrain.testing.synthetic <gatedtrain arguments>` for a CLI run), never by an installed distribution.

## Recipe repositories

Each recipe ships its own adapter and runtime instructions.

| Task | Recipe |
|---|---|
| Math reasoning | [TRL GRPO on GSM8K](https://github.com/gatedtrain/gatedtrain-recipe-trl-grpo-gsm8k) |
| Drone control | [Stable-Baselines3 PPO](https://github.com/gatedtrain/gatedtrain-recipe-sb3-ppo-drone-hover) |
| Image generation | [DDPO with SD1.5](https://github.com/gatedtrain/gatedtrain-recipe-ddpo-sd15) |
| Image generation for OCR | [Flow-GRPO with SD3.5](https://github.com/gatedtrain/gatedtrain-recipe-flow-grpo-sd35-ocr) |
| Video generation | [Flow-GRPO with Wan](https://github.com/gatedtrain/gatedtrain-recipe-flow-grpo-wan-video) |
| Speech recognition | [Whisper SFT on AMI](https://github.com/gatedtrain/gatedtrain-recipe-whisper-sft-ami) |
| Text-to-speech | [CosyVoice DPO](https://github.com/gatedtrain/gatedtrain-recipe-cosyvoice-dpo-tts) |
| CAD | [NeMo RL for CAD](https://github.com/gatedtrain/gatedtrain-recipe-nemo-rl-cad) |
| Visual reasoning | [NeMo RL on CLEVR](https://github.com/gatedtrain/gatedtrain-recipe-nemo-rl-vlm-clevr) |
| Robot manipulation | [LeRobot SmolVLA on LIBERO](https://github.com/gatedtrain/gatedtrain-recipe-lerobot-smolvla-libero) |
| Agentic math | [NeMo RL for agentic math](https://github.com/gatedtrain/gatedtrain-recipe-nemo-rl-agentic-math) |

## Configuration

gatedtrain has no built-in host path. The external-run commands read the host layout from a TOML host file,
`~/.config/gatedtrain/host.toml` or the file named by `GATEDTRAIN_HOST_CONFIG`:

```toml
host_work = "/srv/gatedtrain/work"        # holds bin/gpu-step.sh (the operator's GPU wrapper) and gpu-log.md
output_root = "/srv/gatedtrain/out"       # launches write strictly under this directory
```

`GATEDTRAIN_EXT_HOST_WORK`, `GATEDTRAIN_EXT_OUTPUT_ROOT` and `GATEDTRAIN_EXT_INPUT_ROOTS` override it. Fixed defaults in
`gatedtrain.ext.launch` that a host may need to know about:

- `DOCKER_CONTEXT = "gatedtrain"`: the printed sidecar commands use `docker --context gatedtrain`; create a Docker
  context of that name for the daemon that runs gatedtrain's containers.
- `DAEMON_CONFIGS`: `/etc/docker-gatedtrain/daemon.json` (a second daemon dedicated to gatedtrain, where a host runs
  one) and `/etc/docker/daemon.json`; only their `data-root` key is read, so that no engine data directory is ever
  mounted.
- `SOCKET_PATHS` / `DOCKER_SOCKET_GLOBS`: daemon sockets, including `/run/docker-gatedtrain.sock`; a container never
  runs in a socket's group.
- A host whose daemons keep their configuration or socket elsewhere lists them in the host file; they are refused
  like the built-in ones and enter the launch record's `host_config_sha256`:

  ```toml
  extra_daemon_configs = ["/etc/docker-other/daemon.json"]   # each one's data-root is never mounted
  extra_daemon_sockets = ["/run/docker-other.sock"]          # never mounted; a container never runs in its group
  ```
- `GPU_CAP_H = 1000`: the GPU-hour budget ceiling passed to `gpu-step.sh` (`GATEDTRAIN_GPU_CAP_H`), high enough not to
  limit runs; hours are still logged in `gpu-log.md` and every full run is preceded by a smoke.
- A host whose containers run on the owner's rootless Docker daemon (the context `gatedtrain` pointing at
  `unix:///run/user/<owner uid>/docker.sock`) says so in the host file. The launch rules then require the container
  user `0:0` (the owner on the host), an output directory owned by the owner, and an owner whose primary group owns no
  rootful Docker socket, and print `--device nvidia.com/gpu=all` in place of `--runtime=nvidia --gpus all` (EXT-263).
  The value is a TOML boolean, set in the file only (no environment override); `true` enters `host_config_sha256` and
  the launch record (`"rootless": true`), and a record written under the other setting is refused:

  ```toml
  rootless = true
  ```

`bin/gpu-step.sh` is not part of gatedtrain, and the reference host's script is not published: an operator writes
their own. `docs/gpu-step.md` gives the interface gatedtrain relies on: how the printed command invokes it, the
`WORK=` and `LOG=` lines that `ext-launch` and `ext-collect` check, the script's hash pin, and the formats of
`gpu-guard.log` and `gpu-log.md`.

## Tests

From the repository root, with the package and its `tier-a` extra installed in `.venv` (the tests run the CLI with
the interpreter that runs the suite):

```sh
.venv/bin/python -m unittest discover -s tests -t tests
```

The full suite takes about 13 minutes on a CPU; `.github/workflows/test.yml` runs it on Python 3.12 and 3.13 for
every push and pull request, with the design set at `rev-19`. Some tests read external sources and skip, naming the
variable, when it is unset:

| Variable | Contents |
|---|---|
| `GATEDTRAIN_DESIGN_DIR` | the design set `gatedtrain-design` |
| `GATEDTRAIN_NEMO_EVALUATOR_DIR` | a NeMo Evaluator clone at `ea35ba36756abd59de201f60af8a2feba014744d` |
| `GATEDTRAIN_NEMO_RL_DIR` | a NeMo RL clone at `4f39fd86289a2ffd5aaf5d202702f7fd7185c924` |
| `GATEDTRAIN_TEST_TIERB_ROOT` | the Tier B host work directory of a host that ran the Tier B run (`bin/gpu-step.sh`, `tierb/`, `runs/`), read only |
| `GATEDTRAIN_TEST_MM_ROOT` | the runtime and output root of a host that ran the multimodal recipes, read only (for the recipes' tests) |

`GATEDTRAIN_SKIP_SLOW=1` skips the slow end-to-end runs. The suite uses a synthetic adapter. See the
[test reference](tests/README.md) for coverage and fixture details.

## Design references

This version implements revision 19 of [gatedtrain-design](https://github.com/gatedtrain/gatedtrain-design).
The local specifications cover the [CPU demo](MVP-SPEC.md), [Tier B helpers](TIER-B-SPEC.md) and
[external adapters](MULTIMODAL-SPEC.md). The [findings log](docs/findings.md) records design questions and
regression fixes.

### Unpublished references

Some specifications and test notes cite development decisions, interim rules and review findings by number.
[Section 9 of the test reference](tests/README.md#9-design-gaps-found-while-writing-the-tests) lists G-1 through G-11.

Paths under `<mm-root>` refer to the host layout used for the recipe runs. The matching instructions are in
`runtime/README.md` in each [recipe repository](#recipe-repositories).

| Host directory | Request area | Recipe |
|---|---|---|
| `area1-trl` | 1 | TRL GRPO on GSM8K |
| `area2-drones` | 4 | Stable-Baselines3 PPO |
| `area3-sd15` | 2, row 2a | DDPO with SD1.5 |
| `area3-video` | 3 | Flow-GRPO with Wan |
| `area4-whisper` | 5 | Whisper SFT on AMI |
| `area5-sd35` | 2, row 2b | Flow-GRPO with SD3.5 |
| `area6-cad` | 6 | NeMo RL for CAD |
| `area7-vlm` | 7 | NeMo RL on CLEVR |
| `area8-vla` | 8 | LeRobot SmolVLA on LIBERO |
| `area10-agentic` | 10 | NeMo RL for agentic math |
| `area11-tts` | 11 | CosyVoice DPO |

## Data and licenses

- GSM8K (`openai/gsm8k`, MIT, © 2021 OpenAI): downloaded by `export-gsm8k` at a pinned dataset commit. The fixtures
  under `tests/fixtures/tierb/` are synthetic rows in GSM8K's shape; `tests/fixtures/r11/longrun/raw_samples.json`
  quotes one GSM8K test question (see `THIRD_PARTY_NOTICES.md`).
- NeMo RL and NeMo Evaluator (Apache-2.0): used unchanged as external tools at the pinned commits above; the NeMo RL
  container image is built from NeMo RL's own Dockerfile (`containers/nemo-rl-uid1000/README.md`). gatedtrain quotes
  NeMo Evaluator's GSM8K prompt text so that training and evaluation prompts agree (see `THIRD_PARTY_NOTICES.md`).
- Qwen3-0.6B (Apache-2.0, Qwen team): the base model of the Tier B run; downloaded by the operator, not
  redistributed (a few of its responses are quoted in `raw_samples.json`).
- Checkpoints: the external-run commands hash model files and read safetensors headers and JSON; they never unpickle
  a checkpoint. The Tier A stand-in model loads its own checkpoint with `torch.load(..., weights_only=True)`; load
  only checkpoints you produced.

## License

gatedtrain is licensed under the Apache License, Version 2.0 (`LICENSE`). See `NOTICE` for the copyright notice and
`THIRD_PARTY_NOTICES.md` for the third-party material it includes.

Names of third-party projects and products are trademarks of their owners; gatedtrain is not affiliated with or
endorsed by them.
