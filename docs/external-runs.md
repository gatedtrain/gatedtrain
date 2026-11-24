# External runs

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
