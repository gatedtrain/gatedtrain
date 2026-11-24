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

## Documentation

- [Commands](docs/commands.md)
- [AI-native projects (preview)](docs/ai-native.md)
- [External runs: adapter plugins and recipe repositories](docs/external-runs.md)
- [Configuration](docs/configuration.md)
- [Development: tests and design references](docs/development.md)
- [Changelog](CHANGELOG.md)

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
