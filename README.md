# gatedtrain

gatedtrain records what went into a model training run and tests whether the resulting checkpoint meets a
pre-registered evaluation criterion. It works with external trainers and evaluators. You can query the records
to find which checkpoints may have consumed a training shard, for example after finding a problem in the dataset.

This is a prototype. Outputs carry the tags `prototype` and `unpromotable`.

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
| `impact` | Look up a shard across recorded runs. Each answer reports its precision as `row`, `iteration` or `stage`. |

Gates support superiority, non-inferiority and equivalence. The registration fixes the margin, alpha, target
power and evaluation seeds before the run. The report includes the decision, achieved power and an
`underpowered` flag.

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
