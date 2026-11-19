"""External-run adapters (MULTIMODAL-SPEC.md MM-1 to MM-39; tests/README.md section 11.18 and later).

A framework that trains outside gatedtrain (TRL, HF Trainer, NeMo RL, ...) is turned into the same run directory,
lockfile, ledger, gate report and impact answer as `tierb-collect` gives, through one generic command
(`ext-collect`) and one small adapter per framework. Adapters are plugins: a recipe distribution registers its adapter
class and registry rows through the entry point group `gatedtrain.adapters` (`registry`); gatedtrain itself ships none, and its
tests use the synthetic adapter `gatedtrain.testing.synthetic`. Nothing in this package imports a training framework,
torch or a pickle loader (MM-12); adapters read JSON, YAML, text logs and safetensors headers, and hash everything
else.

Modules (names: the maintainers' choice):

- `registry`: the adapter registry (the plugins' MULTIMODAL-SPEC section 6 rows as data, MM-37), the disclosure table
  and the evaluator identity hash (EXT-6);
- `conformance`: what the commands require of an adapter plugin (`adapter_problems`, the `AdapterConformance` mixin a
  recipe's tests use);
- `sdk`, `rowkit`: the helpers and row blocks adapters share (run arguments, safetensors checks, mount binding, the
  runtime directory rows of a `venv_in_container` recipe);
- `executor`: the executor-image limits of a sidecar (B18);
- `arm`: the per-item arm reader shared by `ext-collect` and `ext-merge-eval` (MM-1 to MM-3);
- `launch`: `ext-launch` (record and command, never run; MM-31 to MM-36) and `gpu_seconds`;
- `smoke`: the smoke verdict (MM-38, MM-39);
- `collect`, `merge`, `from_nel`, `fill`: the `ext-collect`, `ext-merge-eval`, `ext-from-nel` and
  `ext-arm-fill` commands.

The framework families' shared adapter bases are in `gatedtrain.families` (NeMo RL, diffusers LoRA, Flow-GRPO).

prototype, unpromotable.
"""
