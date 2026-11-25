# Tier B runbook: the skeleton on a single-GPU Linux host

- **Status:** prototype, unpromotable. An operator runs every command below by hand on a single-GPU Linux host. The commands are written for one Ada-generation GPU with 24 GB (compute capability 8.9, for example one RTX 4090); another GPU changes the `*_CUDA_ARCH_LIST` values and the memory settings.
- **Pins:** NeMo RL `4f39fd86289a2ffd5aaf5d202702f7fd7185c924` (4f39fd8) and NeMo Evaluator `ea35ba36756abd59de201f60af8a2feba014744d` (ea35ba3). Use both unchanged: no hooks, no shims, no run contract. Every `file:line` below refers to these commits.
- **Outputs:** tagged `unpromotable` (01 section 8.1). The registration is unsigned until a named team signs it.
- **What gatedtrain does on Tier B:** it prepares inputs and reads outputs. It never launches a GPU job.
  - `gatedtrain run` refuses a `rl.nemo_rl@0.1` stage, as before.
  - The Tier B path is the helpers:
    - `export-gsm8k` and `verify-export` (data);
    - `launch_spec` and `build_command` (the training command);
    - `nel-config` and `merge-eval` (evaluation);
    - `tierb-collect` (gate report, ledger and impact answer).
- **Notation:**
  - `<HOST_WORK>`: the host work directory, for example `/data/gatedtrain-tierb`, mounted at `/work` in containers.
  - `<HF_CACHE>`: the host Hugging Face cache, mounted at `/hf_home`.
  - `<HOST_VENV>`: the torch-free host venv of step 3b.
  - `gatedtrain`: `PYTHONPATH=<repo>/src <HOST_VENV>/bin/python -m gatedtrain` on the host.
  - `<repo>`: a checkout of this repository on the host.
  - **Estimate** marks a number that nothing measured.
  - **Unverified** marks a fact the pinned sources could not confirm; check it on the host.
  - Development rule and decision IDs are described in the [design references](development.md#design-references).
- **Rule for every step:** run each command on its own, read its output, and stop at the first stop condition. Record what you saw; do not work around it.

## 1. GPU-hour budget and GPU-time log

- **Choose a GPU-hour budget before step 3.** The spec template carries one as `budget: {gpu_hours: 8}` (interim rule G-10); change it to the budget you choose.
- **Stop** when the log below would pass the budget. Do not resubmit a failed GPU step until you have read its output and decided on the change.
- **Expected total (estimate):** about 3-5 GPU-hours (section 6).

Keep this log, one row per GPU step, with wall-clock times from the host:

| Step | Started (UTC) | Ended (UTC) | GPU-hours | Running total | Outcome |
|---|---|---|---|---|---|
| 3 container checks | | | | | |
| 5 reward check | | | | | |
| 6 smoke run | | | | | |
| 7 long run | | | | | |
| 8-9 baseline serve and evaluation | | | | | |
| 8-10 candidate serve and evaluation | | | | | |

## 2. Facts this runbook relies on

These facts, checked against the pinned sources, drive the commands.

| Fact | Evidence |
|---|---|
| The image is built from `docker/Dockerfile` (release stage) with the local checkout as the `nemo-rl` build context. No NGC image is published per commit. | docker/README.md:2-7; docs/docker.md:5-31; Dockerfile:47-49, :637; README.md:56, :66 |
| Source extensions target 9.0 and 10.0 only (`TORCH_CUDA_ARCH_LIST="9.0 10.0"`). Transformer Engine has no sm_89 kernels. | Dockerfile:191-196, :230-231 |
| A fingerprint check compares the source with the image. Never mount a checkout over `/opt/nemo-rl`. | nemo_rl/__init__.py:105 |
| Documented `docker run` flags: `--runtime=nvidia --gpus all --shm-size --ulimit memlock=-1`, `-e HF_HOME`, `-w /opt/nemo-rl`. Ray wants `nofile` ≥ 65535. | docs/about/installation.md:21-37; ray.sub:161-182 |
| The recipe is `examples/configs/grpo_smoke.yaml` on `grpo_math_1B.yaml`: `Qwen/Qwen3-0.6B`, one GPU, Automodel, vLLM colocated, `max_total_sequence_length: 512`. | grpo_smoke.yaml:1-20; grpo_math_1B.yaml:166, :176-177, :587, :690-692 |
| `data_plane.enabled=true` selects `grpo_train_sync`, the trainer whose outputs the ledger reads. | nemo_rl/data_plane/factory.py:51-55; grpo_sync.py:1323, :1353, :1407-1414, :1458-1460 |
| Struct mode needs `++` for keys the recipe lacks (`data.train.data_path`, `input_key`, `output_key`). | nemo_rl/utils/config.py:188; response_dataset.py:42-48 |
| `ResponseDataset` reads a local JSONL in file order. `StatefulDataLoader(..., drop_last=True)` reads rows in order when `data.shuffle=false`. | response_dataset.py:42-61; grpo.py:883-890 (`drop_last=True` on :888) |
| NeMo RL's GSM8K keeps the answer after `####` with thousands commas. NeMo Evaluator strips commas. | response_datasets/gsm8k.py:22-25; NeMo Evaluator benchmarks/gsm8k.py:29-32 |
| NeMo Evaluator's GSM8K prompt asks for `The answer is`, and its scorer reads that line. NeMo RL's default prompt asks for `\boxed{}`. | benchmarks/gsm8k.py:22-26, :42-44; examples/prompts/cot.txt:1 |
| The training reward wraps the ground truth as `\boxed{gt}` and checks the response with math-verify. | nemo_rl/environments/math_environment.py:93-99, :133-137 |
| `save_consolidated: final` writes HF safetensors to `step_<last>/policy/weights/model/consolidated`; the recipe default writes none. | grpo_math_1B.yaml:178-181; Automodel config.py:64; Automodel checkpointing.py:415-421 |
| `keep_top_k: 3` deletes older checkpoints; `null` keeps every cut. | nemo_rl/utils/checkpoint.py:234, :590-631 |
| NeMo RL writes `logs/exp_<NNN>/` and `checkpoints/step_<N>/{training_info.json, config.yaml, train_dataloader.pt, policy/...}`. | examples/run_grpo.py:106; utils/logger.py:1993-2025; utils/checkpoint.py:404-419; grpo_sync.py:1383-1414 |
| Without validation metrics NeMo RL only warns, and still saves the checkpoint. | grpo_sync.py:1356-1366 (warning), :1378-1386 (save) |
| NeMo Evaluator writes `<output dir>/gsm8k/results.jsonl` and a `full_config.yaml` snapshot. | orchestration/orchestrator.py:1046-1047, :1081; config/snapshot.py:39; executors/local_executor.py:69 |
| NeMo Evaluator sends one `seed` per run, copied into every request. It has no `top_k`. | orchestrator.py:881-889; config/services.py:38-44 |
| NeMo Evaluator's `hf://` loader takes no dataset revision. | environments/custom.py:126-148 |

## 3. Host layout

```
<HOST_WORK>/                     mounted at /work
  nemo-rl/                       checkout at 4f39fd8, used only to build the image
  tierb/                         a copy of <repo>/demo/tierb
    spec.yaml  registration.yaml  trainer.yaml  reward_check.py
    export/                      gatedtrain export-gsm8k output
    base-model/                  Qwen3-0.6B, downloaded with --local-dir (plain files)
    nel/                         gatedtrain nel-config output
  out/<smoke|long>/rl/           NeMo RL checkpoints/ and logs/
  out/<smoke|long>/eval/<arm>/   NeMo Evaluator runs rep<r>/ and the merged arm
  runs/                          gatedtrain tierb-collect run directories
```

## 4. Steps

### Step 0. Prerequisites (no GPU)

1. Choose the GPU-hour budget, set it in the spec template, and start the log in section 1.
2. Make sure no other workload shares the GPU during the runs.

**Stop if** no budget is set.

### Step 1. Preflight (no GPU work; about 15 min, estimate)

```sh
nvidia-smi
```

- **Expect** the GPU (for an RTX 4090: `NVIDIA GeForce RTX 4090`, 24564 MiB) and a `CUDA Version` of at least 13.2. The image uses CUDA 13.2 user space (Dockerfile:30). The exact minimum driver is **unverified**; check the CUDA 13.2 release notes.
- **Stop if** the CUDA version is below 13.2, or another process holds the GPU.

```sh
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv
```

**Expect** `compute_cap` `8.9`. Record `driver_version`.

```sh
docker run --rm --gpus all nvcr.io/nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

**Expect** the same GPU from inside a container; any small CUDA 13 base image works. **Stop if** the output says `could not select device driver`, which means the Container Toolkit is missing.

```sh
df -h /var/lib/docker
```

```sh
df -h <HOST_WORK>
```

**Expect** at least 250 GB free for Docker and at least 100 GB in `<HOST_WORK>` (estimates; checkpoints are about 5-10 GB each). **Stop if** below.

```sh
free -g
```

**Expect** at least 64 GB RAM (estimate). **Stop if** below 32 GB.

```sh
curl -sI https://huggingface.co/api/models/Qwen/Qwen3-0.6B
```

```sh
curl -sI https://huggingface.co/api/datasets/openai/gsm8k
```

**Expect** `HTTP/2 200` for both; both are public. **Stop if** blocked.

```sh
ldd --version
```

**Expect** glibc 2.28 or later; the host wheels are manylinux_2_28 (step 3b). **Stop if** below.

```sh
uv --version
```

**Expect** a version string. Steps 3b and 9 use `uv` on the host. **Stop if** it is missing; install it from https://docs.astral.sh/uv/.

### Step 2. Build the image at 4f39fd8 (no GPU; about 2-5 h on the first build, estimate; network)

```sh
git clone https://github.com/NVIDIA-NeMo/RL.git <HOST_WORK>/nemo-rl
```

```sh
git -C <HOST_WORK>/nemo-rl checkout 4f39fd86289a2ffd5aaf5d202702f7fd7185c924
```

```sh
git -C <HOST_WORK>/nemo-rl submodule update --init --recursive
```

```sh
git -C <HOST_WORK>/nemo-rl submodule status 3rdparty/Automodel-workspace/Automodel
```

**Expect** ` b916107a5ce8e77a3f18682f8b0eaa46a4b54389 3rdparty/Automodel-workspace/Automodel`, with no `+` or `-` prefix. **Stop if** it differs.

With `<HOST_WORK>/nemo-rl` as the working directory:

```sh
docker buildx build --target release --build-context nemo-rl=. -f docker/Dockerfile --build-arg SKIP_TRTLLM_BUILD=1 --build-arg SKIP_SGLANG_BUILD=1 --build-arg NEMO_RL_COMMIT=4f39fd86289a2ffd5aaf5d202702f7fd7185c924 --build-arg MAX_JOBS=<CPU cores / 2> --tag nemo-rl:4f39fd8 --load .
```

- **Sources:** docs/docker.md:28-30, :53-65; Dockerfile:193, :650-658. vLLM must not be skipped: the data-plane trainer needs it (nemo_rl/distributed/actor_environments.py:101).
- **Expect** the build to end with the release stage's offline `uv sync` check passing (Dockerfile:842-843).
- **Stop if** the build fails. Report the failing stage; do not edit the Dockerfile.

```sh
docker image inspect nemo-rl:4f39fd8 --format '{{.Id}}'
```

Record the image ID. It is the `--image-id` of step 11. The image IDs gatedtrain accepts are data, not code: `gatedtrain image list` shows them. A source build gets an ID of its own, which is accepted only after `gatedtrain image check nemo-rl/tierb --ref nemo-rl:4f39fd8` and `gatedtrain image add` (see `containers/nemo-rl-tierb/README.md`); reports then flag it as user-added.

```sh
docker run --rm nemo-rl:4f39fd8 git -C /opt/nemo-rl rev-parse HEAD
```

**Expect** `4f39fd86289a2ffd5aaf5d202702f7fd7185c924`. **Stop if** it differs.

### Step 3. Container checks (GPU; about 10 min, estimate)

Open an interactive container. It has the same flags as the training launcher, plus `-it` and the serving port:

```sh
docker run --rm -it --runtime=nvidia --gpus all --shm-size=32g --ulimit memlock=-1 --ulimit nofile=65535:65535 -e TORCH_CUDA_ARCH_LIST=8.9 -e FLASHINFER_CUDA_ARCH_LIST=8.9 -e HF_HOME=/hf_home -v <HF_CACHE>:/hf_home -v <HOST_WORK>:/work -w /opt/nemo-rl -p 127.0.0.1:8000:8000 nemo-rl:4f39fd8 bash
```

- `--shm-size=32g` must be at or below host RAM; the documentation uses 64g (docs/about/installation.md:25).
- The two `*_CUDA_ARCH_LIST` variables are a precaution for runtime JIT builds on sm_89. Whether FlashInfer reads them at vLLM 0.29 is **unverified**.

Inside the container:

```sh
ls /opt/ray_venvs
```

**Expect** actor venvs including `...AutomodelPolicyWorker`, `...VllmGenerationWorker` and `...SyncRolloutActor` (actor_environments.py:48-60, :101). **Stop if** any is missing.

```sh
/opt/ray_venvs/nemo_rl.models.policy.workers.automodel_policy_worker.AutomodelPolicyWorker/bin/python -c "import torch, flash_attn; from flash_attn import flash_attn_func; q=torch.randn(1,128,8,64,device='cuda',dtype=torch.bfloat16); print(torch.cuda.get_device_capability(), flash_attn.__version__, flash_attn_func(q,q,q,causal=True).shape)"
```

- **Expect** `(8, 9) 2.8.1 torch.Size([1, 128, 8, 64])`.
- The venv directory name is **unverified**; take it from the `ls` output.
- **Stop if** the output says `no kernel image is available`.

Exit the container.

### Step 3a. Host directories, as the host user (no GPU; 1 min)

**Why.** The NeMo RL image runs as root (`USER root`, docker/Dockerfile:846 at 4f39fd8), and the launcher does not pass `--user`. Every file a container writes under `/work` is therefore owned by root on the host. A directory that a container creates first is root-owned too, and a later host step cannot write into it. So the host user creates every directory that a container writes into, before any container does.

Copy the templates first; `cp -R` into an existing `tierb/` would nest the copy as `tierb/tierb`:

```sh
cp -R <repo>/demo/tierb <HOST_WORK>/tierb
```

Then pre-create the directories:

```sh
mkdir -p <HOST_WORK>/tierb/base-model <HOST_WORK>/out/{smoke,long}/{rl,eval}
```

- **`tierb/base-model`:** the step 4 download runs in a container and writes into it.
- **`out/{smoke,long}/rl`:** the training container writes `checkpoints/` and `logs/` into it.
- **`out/{smoke,long}/eval`:** the host-side NeMo Evaluator runs and `merge-eval` write here in steps 9 and 10. If the training container created `out/<run>/` first, root would own it, and the host could not create `eval/`.
- **`tierb/export` is not pre-created on purpose.** `gatedtrain export-gsm8k` runs on the host as the host user and refuses an existing `--out` directory, so it creates `tierb/export` itself, host-owned (finding F-24).

Root-owned files under `out/*/rl` and inside `tierb/base-model` are expected. The host tools only read them: the step 6 and 7 checks, `vllm serve` (inside a container), `tierb-collect` and `impact`. No host step writes into or deletes from those directories; the `.cache` removal in step 4 runs inside the container.

### Step 3b. Linux host install of gatedtrain, as the host user (no GPU; about 5 min, estimate; network)

`uv.lock` covers macOS only (`[tool.uv] environments` in pyproject.toml; interim rule G-11), so `uv sync --locked` (the macOS Tier A install) does not work on the Linux host. The host-side Tier B commands need no torch. Install them into a plain venv from exact pins equal to `uv.lock` (finding F-27):

```sh
uv venv --python 3.13 <HOST_VENV>
```

```sh
uv pip install --python <HOST_VENV>/bin/python -r <repo>/requirements-tierb-host.txt --only-binary :all: --require-hashes
```

- **Pins:** numpy 2.5.4, scipy 1.18.1, pyarrow 26.0.1, duckdb 2.0.0 and pyyaml 6.0.3. scipy's only locked dependency is numpy.
- The host requirements resolve to binary wheels for x86-64 Linux with CPython 3.13 and glibc 2.28 or later. The resolver check used `uv pip compile requirements-tierb-host.txt --python-platform x86_64-manylinux_2_28 --python-version 3.13 --only-binary :all:`.
- The host requirements include distribution hashes generated with `uv pip compile --generate-hashes`. Regenerate them whenever a pin changes. The install checks those hashes with `--require-hashes` and requires wheels with `--only-binary :all:`.
- **No torch on the host:** `gatedtrain run`, `gatedtrain warmup` and the Tier A tests are not available. The host runs `validate`, `export-gsm8k`, `verify-export`, `nel-config`, `merge-eval`, `tierb-collect` and `impact`.
- From here on, `gatedtrain` means `PYTHONPATH=<repo>/src <HOST_VENV>/bin/python -m gatedtrain`.

Check the install:

```sh
PYTHONPATH=<repo>/src <HOST_VENV>/bin/python -m gatedtrain validate <HOST_WORK>/tierb/spec.yaml
```

**Expect** `ok`: the template validates before its `prompt_set_hash` is filled in (the template keeps the revision 10 names, which design revision 11 still reads; `eval_distribution: {item_set_hash: ...}` is the new form). **Stop if** it fails with an import error.

### Step 4. Export GSM8K and download the base model (no GPU; about 10 min, estimate; network)

The templates were copied in step 3a. Export the dataset (H-1). Without `--source` the command downloads `openai/gsm8k` config `main` from Hugging Face at the current dataset commit and records that commit as the revision (G-17).

```sh
gatedtrain export-gsm8k --out <HOST_WORK>/tierb/export
```

- The download path is not covered by tests. The parquet file names it fetches (`main/{train,test}-00000-of-00001.parquet`) are **unverified**.
- **If the download fails**, fetch the two splits by other means as JSONL rows `{question, answer}`. Then pass them with `--source <dir> --revision <dataset commit hash>`; with `--source`, `--revision` is required.

```sh
gatedtrain verify-export <HOST_WORK>/tierb/export
```

**Expect** `ok` and this layout:

- `export/train/shard-00000.jsonl` to `shard-00014.jsonl`: 7,473 training rows, 14 × 500 + 473, in split order;
- `export/train.jsonl`: the concatenation NeMo RL reads. It sits **outside** the shard directory, never inside it.
  - The SHA-256 of the concatenated `train.jsonl` is recorded in `export/export.json` as `concat_sha256`.
  - `verify-export` checks that hash against a fresh concatenation of the shards in file-name order and against `train.jsonl` itself (L-5).
  - Run `verify-export` again right before steps 6, 7 and 11.
- `export/heldout/prompts.jsonl`: the 1,319 test rows, the gate's item set. Its manifest hash is the registration's `prompt_set_hash` (revision 11 name: `eval_distribution.item_set_hash`).
- `export/nel_gsm8k_prompt.txt`: NeMo Evaluator's GSM8K prompt with `{}` in place of `{question}`, and no trailing newline (G-15).
- `export/manifests/{training_shards,prompt_set}.json`: the content-addressed manifests (canonical JSON per RFC 8785).
- `export/export.json`: the dataset revision and all hashes.

Answers keep thousands separators, for example `3,082`, as NeMo RL does (G-19). **Stop if** the row counts differ or `verify-export` refuses.

Download the base model once, at a pinned commit, into a **plain directory** with `--local-dir`. This directory is used three times:

- it is the training start point: `launch_spec(..., base_model="/work/tierb/base-model")` emits `policy.model_name=/work/tierb/base-model`, and the tokenizer follows it (`name: ${policy.model_name}`, grpo_math_1B.yaml:157; F-17);
- it is the baseline that is served and evaluated (step 8);
- it is the baseline whose manifest the gate records.

The manifest refuses symbolic links (manifest rule H2), and a Hugging Face cache snapshot is a tree of symlinks, so do not point at the cache. First read the commit:

```sh
curl -s https://huggingface.co/api/models/Qwen/Qwen3-0.6B
```

Record the `sha` field as `<QWEN_REV>` in your run notes. Then, inside a container from step 3:

```sh
uv run huggingface-cli download Qwen/Qwen3-0.6B --revision <QWEN_REV> --local-dir /work/tierb/base-model
```

The CLI name in the image (`huggingface-cli` or `hf`) is **unverified**; use whichever `uv run ... --help` accepts.

Still inside the same container, delete the download metadata. `--local-dir` leaves `base-model/.cache/`, which would be hashed into the baseline manifest, and `tierb-collect` refuses it (TIERB-BASELINE). The files are root-owned, so the host user cannot delete them; delete them in the container:

```sh
rm -r /work/tierb/base-model/.cache
```

Then, on the host (read only):

```sh
find <HOST_WORK>/tierb/base-model -type l
```

**Expect** no output. **Stop if** any symlink is listed, or if `config.json`, the tokenizer files or `*.safetensors` are missing from `base-model/`.

### Step 5. Registration, spec and the G-15/G-19 reward check (no GPU, then about 2 min of GPU, estimate)

1. Edit `<HOST_WORK>/tierb/registration.yaml`. Replace the `prompt_set_hash` placeholder with `prompt_set_hash` from `export/export.json`. Keep the other values, or change them all before any run:
   - K = 4 and `eval_seeds`;
   - the sampler: `temperature`, `top_p`, `max_new_tokens`, and `enable_thinking: false`;
   - the margin and the planning σ, which are `internal_pilot` placeholders;
   - `evaluators[0].scorer: numeric_match` (interim rule G-21). The template is the v2 amendment (05:142) that adds it: NeMo Evaluator's default GSM8K scorer gave Qwen3-0.6B a mean reward of about 0 on the smoke baseline because of answer formatting, so the gate reads the `numeric_match` scorer instead (`scoring/pattern.py:39-43` at ea35ba3). `numeric_match` is the only allowed value; any other is refused (`REG-SCORER`). A scorer is allowed only on the Tier B evaluator `gsm8k.nemo_evaluator@0.1`; on any other evaluator (the Tier A demo `addition_em.cpu_stub@0.1` included) `validate` refuses it (`REG-SCORER`, decision D-10). Without the line, the gate reads the default `reward`. Either way, every per-record score the gate reads must be a finite number in [0, 1]; `tierb-collect` refuses NaN, infinities and values outside [0, 1] (`TIERB-RESULTS`, decision D-11).
   - `evaluators[0].manifest_hash`: the resolved evaluator identity hash (05:151, design revision 10). The template carries the value for `numeric_match`, `sha256:2c196d9d…`; without a scorer it is `sha256:032ad6d4…`. A null or absent hash is refused (`REG-FIELD`), and the fix line names the hash to use. If the hash differs from the one `tierb-collect` resolves, the gate is MISSING with reason `evaluator_hash_mismatch`.
   - Optional fields (05 §3.4, §3.6, §3.7): `min_paired` (default 10: fewer complete pairs is INCONCLUSIVE), `pairing.max_unpaired` (default 0; an integer count or a fraction in [0, 1) of the expected items; above it the gate is MISSING), `amends: sha256:<prior registration hash>` for an amendment, and `post_hoc: true` when the amendment was made after any candidate-arm result was seen (the gate then cannot PASS: a PASS is reported as INCONCLUSIVE with reason `post_hoc`).
2. Put the registration's new SHA-256 into `spec.yaml`:

   ```sh
   sha256sum <HOST_WORK>/tierb/registration.yaml
   ```

   Edit the `@sha256:` part of the registration uri in `<HOST_WORK>/tierb/spec.yaml`.
3. Validate:

   ```sh
   gatedtrain validate <HOST_WORK>/tierb/spec.yaml
   ```

   **Expect** `ok`. **Stop if** it refuses; every error line names the rule, the field and the fix.
4. For the smoke run, set `profile: smoke` in `<HOST_WORK>/tierb/trainer.yaml`. Set it back to `long` before step 7.

**G-15/G-19 reward check.** Training uses NeMo Evaluator's prompt with Qwen3 thinking off (G-15). The NeMo RL reward must therefore accept `The answer is N`, including a thousands-separator ground truth (`3,082` against `The answer is 3082`, G-19). Inside a container from step 3:

```sh
uv run python /work/tierb/reward_check.py
```

- The script builds the reward as `math_environment.py:93-99` does and wraps the ground truth as `:133-137` does.
- **Expect** six `ok` lines and `G-15/G-19 reward check: PASS`, exit 0.
- **Stop if** any case prints `FAIL`. Report the lines; the gate would otherwise measure a format mismatch.

Log the GPU time.

### Step 6. Smoke run: 10 steps of 2 × 4 (GPU; about 10-20 min including first-run compilation, estimate)

Print the exact command (H-5), on the host:

```sh
PYTHONPATH=<repo>/src <HOST_VENV>/bin/python -c "import shlex; from gatedtrain.backends.nemo_rl import launch_spec; s = launch_spec({'id': 'rl'}, nemo_rl_dir='/opt/nemo-rl', launcher='docker', out_dir='/work/out/smoke/rl', export_dir='/work/tierb/export', profile='smoke', image='nemo-rl:4f39fd8', work_dir='<HOST_WORK>', hf_cache='<HF_CACHE>', cuda_arch='8.9', base_model='/work/tierb/base-model'); print(shlex.join(s['argv'])); print('samples_per_position', s['samples_per_position'])"
```

It prints this command and `samples_per_position 4`.

**Offline mode.** The command sets `-e HF_HUB_OFFLINE=1` (F-22, interim rule G-20). The training container gets no Hugging Face Hub access. The base model (`/work/tierb/base-model`, step 4) and the data (`/work/tierb/export`, step 4) must already be local before the run. A run that tries to fetch anything from the Hub fails instead of silently using another revision.

Run the command on the host:

```sh
docker run --rm --runtime=nvidia --gpus all --shm-size=32g --ulimit memlock=-1 --ulimit nofile=65535:65535 -e TORCH_CUDA_ARCH_LIST=8.9 -e FLASHINFER_CUDA_ARCH_LIST=8.9 -e HF_HOME=/hf_home -e HF_HUB_OFFLINE=1 -v <HF_CACHE>:/hf_home -v <HOST_WORK>:/work -w /opt/nemo-rl nemo-rl:4f39fd8 uv run python examples/run_grpo.py --config examples/configs/grpo_smoke.yaml data_plane.enabled=true data.shuffle=false checkpointing.enabled=true checkpointing.keep_top_k=null checkpointing.checkpoint_dir=/work/out/smoke/rl/checkpoints logger.log_dir=/work/out/smoke/rl/logs data.train.dataset_name=ResponseDataset ++data.train.data_path=/work/tierb/export/train.jsonl ++data.train.input_key=question ++data.train.output_key=answer data.train.split_validation_size=0 data.default.prompt_file=/work/tierb/export/nel_gsm8k_prompt.txt 'policy.tokenizer.chat_template_kwargs={enable_thinking:false}' policy.automodel_cfg.checkpoint.save_consolidated=final policy.model_name=/work/tierb/base-model grpo.max_num_steps=10 checkpointing.save_period=5 grpo.num_prompts_per_step=2 grpo.num_generations_per_prompt=4
```

**Expected log lines:**

- `Running synchronous GRPO training (TransferQueue)` (factory.py:54);
- `Auto-setting force_hf=True` (nemo_rl/models/automodel/setup.py:179-185). The policy runs Hugging Face Qwen3 with flash-attention 2, which avoids Transformer Engine on sm_89;
- `Loaded training dataset ... with 7473 samples`;
- the warning `You asked to save checkpoints based on accuracy but no val metrics were collected`. It is expected: NeMo RL warns and still saves (grpo_sync.py:1356-1366, :1378-1386).

**Expected outputs:**

- `/work/out/smoke/rl/logs/exp_001/train_data_step1.jsonl` to `train_data_step10.jsonl`, with 8 lines each (2 prompts × 4 generations);
- `checkpoints/step_5` and `checkpoints/step_10`, each with `training_info.json` (`consumed_samples` 10 and 20, counting prompts, G-7), `config.yaml` and `train_dataloader.pt`;
- `checkpoints/step_10/policy/weights/model/consolidated/` with `*.safetensors` and `config.json`.

**Checks, on the host:**

```sh
grep -h -o '"rewards": [0-9.]*' <HOST_WORK>/out/smoke/rl/logs/exp_001/train_data_step*.jsonl
```

**Expect** at least one reward above 0 over the 10 steps.

```sh
grep -c "The answer is" <HOST_WORK>/out/smoke/rl/logs/exp_001/train_data_step10.jsonl
```

**Expect** a count above 0. `content` holds the raw assistant text (grpo_sync.py:1454-1457).

These host checks only read the root-owned outputs. **Stop if** either `grep` reports `Permission denied`. The host-side checks assume the container wrote world-readable files (root's default umask 022); that is **unverified** for this image (finding F-25).

**Stop if** any of these happen:

- a CUDA OOM, or a vLLM error that free memory is below the requested utilization. Retry once with `policy.generation.vllm_cfg.gpu_memory_utilization=0.4` (grpo_math_1B.yaml:587), then stop. Any further fallback is a deliberate change to the recipe; record it before rerunning:
  - `policy.train_micro_batch_size=4`;
  - `policy.automodel_cfg.activation_checkpointing=true`;
  - `policy.generation.vllm_cfg.enforce_eager=true`;
  - `policy.automodel_cfg.cpu_offload=true`.
- a Transformer Engine or FlashInfer `no kernel image` error;
- a data-plane or TransferQueue error. One GPU with checkpointing on is **unverified** upstream: NeMo RL's own test uses 2 GPUs (tests/functional/grpo_dp_simple.sh:25-41);
- every reward over the 10 steps is 0;
- `consolidated/` is missing;
- more than one `exp_*` directory exists (a manual restart; `tierb-collect` refuses it).

Log the GPU time. Collect the smoke run with steps 8-11 before starting the long run; it checks the whole path cheaply.

### Step 7. Long run: 150 steps of 32 × 8 (GPU; about 2-3.5 h, estimate)

Set `profile: long` in `<HOST_WORK>/tierb/trainer.yaml`, then run `verify-export` again. Print the command with `profile='long'` and `out_dir='/work/out/long/rl'` (step 6 snippet). It prints `samples_per_position 8` and this command:

```sh
docker run --rm --runtime=nvidia --gpus all --shm-size=32g --ulimit memlock=-1 --ulimit nofile=65535:65535 -e TORCH_CUDA_ARCH_LIST=8.9 -e FLASHINFER_CUDA_ARCH_LIST=8.9 -e HF_HOME=/hf_home -e HF_HUB_OFFLINE=1 -v <HF_CACHE>:/hf_home -v <HOST_WORK>:/work -w /opt/nemo-rl nemo-rl:4f39fd8 uv run python examples/run_grpo.py --config examples/configs/grpo_smoke.yaml data_plane.enabled=true data.shuffle=false checkpointing.enabled=true checkpointing.keep_top_k=null checkpointing.checkpoint_dir=/work/out/long/rl/checkpoints logger.log_dir=/work/out/long/rl/logs data.train.dataset_name=ResponseDataset ++data.train.data_path=/work/tierb/export/train.jsonl ++data.train.input_key=question ++data.train.output_key=answer data.train.split_validation_size=0 data.default.prompt_file=/work/tierb/export/nel_gsm8k_prompt.txt 'policy.tokenizer.chat_template_kwargs={enable_thinking:false}' policy.automodel_cfg.checkpoint.save_consolidated=final policy.model_name=/work/tierb/base-model grpo.max_num_steps=150 checkpointing.save_period=25 grpo.num_prompts_per_step=32 grpo.num_generations_per_prompt=8 policy.train_global_batch_size=256 policy.train_micro_batch_size=8
```

- The long profile is an estimate, not a NeMo RL recipe. 150 × 32 = 4,800 prompts, under one epoch of 7,473 rows, so no epoch wraps. The ledger mapping assumes NeMo RL's drop_last epochs (decision D-1; grpo.py:883-890).
- **Expect** 150 step files of 256 lines each, and checkpoints at steps 25, 50, …, 150 with `consumed_samples` = 32 × step. `consolidated/` appears under `step_150`. The mean reward per step should rise within about 50 steps (estimate).
- **Wall time:** about 40-70 s per step on an RTX 4090, plus about 10 min start-up (estimate).

**Stop if** any of these happen:

- the run would pass the GPU-hour budget;
- the reward stays at 0, or falls for more than 50 steps;
- an OOM (the step 6 fallbacks apply; record any rerun);
- the disk is above 90 %.

Log the GPU time.

### Step 8. Serve one model at a time (GPU; about 2-5 min to start, estimate)

Use a fresh container from step 3; the training container must have exited.

Baseline, served from the same local directory that `tierb-collect` hashes:

```sh
uv run --extra vllm vllm serve /work/tierb/base-model --served-model-name baseline --generation-config vllm --max-model-len 2048 --gpu-memory-utilization 0.85 --seed 0 --port 8000
```

Candidate, from the consolidated weights of the last checkpoint:

```sh
uv run --extra vllm vllm serve /work/out/<smoke|long>/rl/checkpoints/step_<N>/policy/weights/model/consolidated --tokenizer /work/out/<smoke|long>/rl/checkpoints/step_<N>/policy/tokenizer --served-model-name candidate --generation-config vllm --max-model-len 2048 --gpu-memory-utilization 0.85 --seed 0 --port 8000
```

- `--generation-config vllm` keeps Qwen3's `generation_config.json` from changing the sampler; NeMo Evaluator cannot send `top_k` (config/services.py:38-44). Whether the flag exists at vLLM 0.29 is **unverified**; check `vllm serve --help`.
- `uv run --extra vllm` installs from the image's offline cache (docs/docker.md:75; docs/guides/use-custom-vllm.md:17).
- `--max-model-len 2048` must hold the prompt plus `max_new_tokens` (1024 in the template).

From the host:

```sh
curl -s http://localhost:8000/v1/models
```

**Expect** the served name. Record `vllm --version`.

**Stop if** the server fails to start, `--generation-config` is rejected, or `uv` tries a network download.

### Step 9. Evaluate the baseline (GPU through the server; about 10-15 min for K = 4, estimate)

Install NeMo Evaluator at ea35ba3 on the host. It has no lockfile, so record `uv pip freeze`:

```sh
git clone https://github.com/NVIDIA-NeMo/Evaluator.git <HOST_WORK>/nemo-evaluator
```

```sh
git -C <HOST_WORK>/nemo-evaluator checkout ea35ba36756abd59de201f60af8a2feba014744d
```

```sh
uv venv --python 3.12 <HOST_WORK>/nel-venv
```

```sh
uv pip install --python <HOST_WORK>/nel-venv/bin/python <HOST_WORK>/nemo-evaluator
```

Write the K configs from the registration (H-4). One run per registered seed, with `repeats: 1` (G-16) and `enable_thinking: false` (G-15):

```sh
gatedtrain nel-config --registration <HOST_WORK>/tierb/registration.yaml --model-url http://localhost:8000/v1 --model-id baseline --eval-root <HOST_WORK>/out/<run>/eval/baseline --out <HOST_WORK>/tierb/nel/<run>-baseline
```

**Expect** `nel-gsm8k-rep0.yaml` to `nel-gsm8k-rep3.yaml`. Each one carries:

- `services.model.generation`: `temperature`, `top_p`, `max_tokens` and `seed` = `eval_seeds[r]`;
- `proxy.extra_body.chat_template_kwargs.enable_thinking: false`;
- one `gsm8k` benchmark with `repeats: 1`;
- when the registration names a scorer (G-21), `scoring: {metrics: [{type: scorer, name: numeric_match}]}` under that benchmark (config/benchmarks.py:63, config/scoring.py:24-30). NeMo Evaluator then writes `scorer_rewards["scorer:numeric_match"]` next to the default `reward` (engine/eval_loop.py:785-794);
- `output.dir` = `.../eval/baseline/rep<r>`.

**Never** use NeMo Evaluator's `--resume` after the scorer changes (added, removed or renamed): records cached from the earlier run are reused as they are, without `scorer_rewards` for the new scorer (engine/eval_loop.py:319-331), and the resume check does not notice, because `config_hash` leaves the scorers out (engine/step_log.py:50-59). `tierb-collect` would then refuse the arm (`TIERB-RESULTS`). Run the changed configs into a fresh output directory instead.

No dataset revision key exists at ea35ba3 (environments/custom.py:126-148). The export and the report record the revision (G-17). Run each config, r = 0..3, one at a time:

```sh
<HOST_WORK>/nel-venv/bin/nel eval run <HOST_WORK>/tierb/nel/<run>-baseline/nel-gsm8k-rep0.yaml
```

**Expect** `.../eval/baseline/rep0/gsm8k/results.jsonl` with 1,319 lines (`problem_idx` 0..1318, `repeat` 0) and `.../rep0/full_config.yaml`. Then merge (H-3):

```sh
gatedtrain merge-eval --registration <HOST_WORK>/tierb/registration.yaml --out <HOST_WORK>/out/<run>/eval/baseline/merged <HOST_WORK>/out/<run>/eval/baseline/rep0 <HOST_WORK>/out/<run>/eval/baseline/rep1 <HOST_WORK>/out/<run>/eval/baseline/rep2 <HOST_WORK>/out/<run>/eval/baseline/rep3
```

- **Expect** `merged/results.jsonl` with 4 × 1,319 records, `repeat` r with `metadata.eval_seed` = `eval_seeds[r]`, `merged/configs/rep<r>.yaml` (a byte copy of each run's `full_config.yaml`), and `merged/merge.json` with the sources (each with its `config` and `config_sha256`) and the G-16 note.
- `merge-eval` refuses (`error[MERGE-EVAL]`) a missing or empty run, `repeats` ≠ 1, a seed out of registered order, or a problem set that differs between runs. With a registered scorer it also refuses a run whose `full_config.yaml` does not declare `{type: scorer, name: numeric_match}` under the gsm8k benchmark (G-21).
- NeMo Evaluator's `problem_idx` is the original test-split index (engine/eval_loop.py:132-134), the same order as `export/heldout/prompts.jsonl`.

**Stop if** any of these happen:

- `merge-eval` refuses;
- a record has a `scoring_details` error;
- the mean reward is below 0.2 (estimate; suggests a template or `enable_thinking` problem). With a registered scorer, read the mean of `scorer_rewards["scorer:numeric_match"]`, the value the gate uses (G-21);
- two repeats are identical across all problems (the seed was not applied).

Log the GPU time.

### Step 10. Evaluate the candidate (about 10-15 min, estimate)

1. Stop the baseline server (Ctrl-C) and start the candidate server (step 8).
2. Run `nel-config` with `--model-id candidate`, `--eval-root .../eval/candidate` and `--out .../nel/<run>-candidate`.
3. Run the four `nel eval run` commands.
4. Run `merge-eval` into `.../eval/candidate/merged`.

**Stop if** any setting differs from the baseline run other than the model and the output directory; compare the four `full_config.yaml` files of each arm. Log the GPU time.

### Step 11. Collect, gate and impact (no GPU; about 5 min, estimate)

Run `verify-export` once more. Then (H-6):

```sh
gatedtrain tierb-collect <HOST_WORK>/tierb/spec.yaml --stage-out <HOST_WORK>/out/<run>/rl --eval-baseline <HOST_WORK>/out/<run>/eval/baseline/merged --eval-candidate <HOST_WORK>/out/<run>/eval/candidate/merged --image-id <image ID from step 2> --nemo-rl-commit 4f39fd86289a2ffd5aaf5d202702f7fd7185c924 --runs-root <HOST_WORK>/runs --nemo-rl-modified <yes|no> --image-build-notes "<how the image was built>"
```

`--nemo-rl-modified yes|no` states whether the NeMo RL image was built from a locally modified checkout (for example a changed `MAX_JOBS` in `pyproject.toml`), and `--image-build-notes` describes the change. Both go to the lockfile's `backend.trainer` as `nemo_rl_modified` and `image_build_notes`. Without the flag, `nemo_rl_modified` is recorded as `"unknown"`, never as false; any other value is a usage error (exit 2).

**What it does:**

- runs the `verify-export` checks on `<HOST_WORK>/tierb/export`;
- reads `samples_per_position` from `step_<last>/config.yaml` (G-7);
- writes a run directory like `gatedtrain run`:
  - the manifests;
  - a lockfile with the image ID, the NeMo RL commit, the NeMo RL config hash, the dataset revision, and the checkpoint completeness rule the adapter declares for NeMo RL 4f39fd8 (`backend.trainer.checkpoint_completeness`: only `step_<N>` directories are cuts; a `tmp_step_<N>` directory is an interrupted write and is ignored; design revision 11, R11-15);
  - the ledger at precision `iteration`, `derivation: reconstructed` (the shard of each step is derived from the step sample counts and the loader order), mapped to the exported shards;
  - the lineage, the gate and the report, with `dataset_revision` and the G-16 note.
- gates on the complete pairs and counts the missing items of each arm against the registered item set (all 1,319 held-out problems) times K (`missing_baseline`, `missing_candidate`, and `missing_reasons` per reason: `absent`, or `no_seed` for a record without `metadata.eval_seed`); above `pairing.max_unpaired` the gate is MISSING with reason `missing_above_bound` (05 §3.4). The report uses item names (`n_items`, `samples_per_item`, `eval_distribution`, `direction`, `natural_units`; design revision 11). It also gives `k_actual` (repeats per item in each arm; fewer than K are missing items, a repeat outside range(K) is MISSING with reason `k_mismatch`), checks each record's `metadata.eval_seed` against `eval_seeds[repeat]` (MISSING, `seed_mismatch`), and reports `p_value` (one-sided, by inverting the BCa interval; TOST for equivalence), `min_paired`, `status_reasons`, `registration_amends`, `post_hoc` and `evaluator_hash_registered`;
- with a registered scorer (G-21), gates on `scorer_rewards["scorer:numeric_match"]` instead of `reward`; the report says so in a `metric_source: scorer_rewards[scorer:numeric_match]` line (`metric_source: reward` without a scorer) and adds a note on `numeric_match`'s trailing-zero false positives (25/5,276 = 0.47% on the smoke baseline). The scorer is part of `evaluator_hash` and of the lockfile's `backend.evaluator`.

**Refusals.** Exit 3, and nothing is written:

| Rule | Cause |
|---|---|
| `TIERB-RESULTS` | an arm's `results.jsonl` is missing or empty. This covers pointing at the parent `eval/` directory or at an unmerged `rep<r>` run, which used to give a silent MISSING gate. Without a scorer: a record has no `reward`, or its `reward` is not a finite number in [0, 1] (D-11). With a registered scorer (G-21), the scorer value is required and `reward` is not read: a record has no numeric `scorer_rewards["scorer:numeric_match"]`, that value is not a finite number in [0, 1] (D-11), its `scoring_details["scorer:numeric_match"]` holds an `error`, or its `scorer_rewards` or `scoring_details` is not a JSON object (D-12). |
| `TIERB-SCORER` | a scorer is registered, but the arm's `configs/rep<r>.yaml` (copied by `merge-eval`) are missing, unreadable (not valid UTF-8 or YAML, D-12) or do not all declare it under the gsm8k benchmark (G-21) |
| `TIERB-DATA` | the `data` section of `step_<last>/config.yaml` would break the ledger's step-to-shard mapping: `data.shuffle` is not `false` (absent included), `data.train` is not exactly one dataset, its `data_path` is not `<dir>/<export dir name>/train.jsonl`, or `split_validation_size` is not 0 (design revision 10, review finding 7) |
| `TIERB-STAGE` | more than one `exp_*` directory, or a failed `consumed_samples` check |
| `TIERB-BASE` | the baseline directory contains symlinks, or `policy.model_name` in `step_<last>/config.yaml` is not the spec's base-model directory, for example a Hub id or another directory (L-1, F-17) |
| `TIERB-READ` | a file or directory under the base model or a saved checkpoint cannot be read by the host user (F-25) |
| `TIERB-BASELINE` | `base-model/.cache` exists; delete it |
| `TIERB-PROMPTS` | a problem outside the registered held-out set was evaluated, or the answers or questions differ from it (R4). A held-out problem missing from an arm is a missing item, not a refusal (design revision 11) |
| `TIERB-PROFILE` | `step_<last>/config.yaml` is not a mapping, or does not match the trainer.yaml `profile` |
| `ARTIFACT-HASH` | a non-registration `@sha256:` pin in the spec does not match |
| `EXPORT-*` | the export changed |
| `REG-*` | the registration changed |

Unpaired or missing items are no longer a refusal (the rev-9 `TIERB-PAIRING` rule is retired by design revision 10, R10-8): `tierb-collect` writes the run and a MISSING report, exit 1, when they exceed `pairing.max_unpaired`.

**Expect** `run_dir: <path>` and the text gate report. Exit 0 is PASS; exit 1 is FAIL, INCONCLUSIVE or MISSING; the `status_reasons:` line says why. The 1,319-prompt test set exceeds the template's planned n of 546. Then ask which checkpoints possibly consumed a shard:

```sh
gatedtrain impact <HOST_WORK>/tierb/export/train/shard-00000.jsonl --run-dir <run_dir> --json
```

**Expect** every checkpoint at or after the first step that read shard 0, at precision `iteration` and `derivation: reconstructed`, with the answer "possibly consumed", and `counts: {orphaned, superseded}`: records of steps after the last saved checkpoint that read the shard (they enter no result), and 0 superseded (one attempt). `ranges` gives the same orphaned records as step ranges per attempt and shard (`superseded` empty); the text output prints them as `orphaned_range: attempt <a> shard <shard> steps <first>-<last>` or `orphaned_ranges: none` (design revision 11, R11-16).

**Stop if** `tierb-collect` refuses. Report the error lines; do not edit outputs to get past a check.

## 5. Without the export (G-14 fallback)

If the GSM8K split cannot be exported to local shard files, the step-to-shard mapping is impossible.

- Train on the recipe's own `dataset_name: gsm8k` instead.
- Record the stage at precision `stage` (05:348): only that the stage used the dataset snapshot.
- Skip the ledger. `tierb-collect` refuses a spec whose dataset is not an export, so record the gate result and the `stage` edge by hand in your run notes.
- The impact answer is then "possibly consumed" for every checkpoint of the stage.

## 6. GPU-hour estimate (all estimates, nothing measured)

| Item | GPU-hours |
|---|---|
| Step 3 checks and step 5 reward check | 0.05-0.1 |
| Smoke run, including first-run compilation and start-up | 0.2-0.4 |
| Long run (150 × 32 × 8) | 2.0-3.5 |
| 4 evaluations: baseline and candidate for smoke and long, each K = 4 × 1,319, plus server start-up | 0.7-1.2 |
| **Total** | **about 3-5** |

The template's 8 GPU-hour budget leaves room for one memory-fallback rerun of the smoke run. The baseline evaluation can be reused for both comparisons if the sampler, server and seeds are identical; that saves about 0.3 GPU-hours. The image build uses no GPU but several CPU-hours.

## 7. Risks

- **R1, format and sampler.** Training uses NeMo Evaluator's prompt with thinking off (G-15). The step 5 reward check must pass before any training. Part of any gain may be answer-format learning; the report says so.
  - **Training sampler:** temperature 1.0, top_p 1.0, and `max_new_tokens = max_total_sequence_length = 512` (grpo_math_1B.yaml:166, :479-481).
  - **Gate sampler:** the registered deployment sampler, temperature 0.7 with up to 1,024 new tokens.
  - 05 section 3.3 allows the difference: training-sampler scores are reported but do not decide, and the gate runs under the deployment sampler.
  - **Truncated training responses are not masked in this recipe.** NeMo RL zeroes the loss weight of truncated responses only when `grpo.overlong_filtering` is true (grpo_sync.py:934-937). The recipe sets `overlong_filtering: false` (grpo_math_1B.yaml:18); grpo_smoke.yaml does not change it, and neither does `build_command`. Decided: `overlong_filtering` stays false, as the recipe ships it (interim rule G-20; finding F-23).
  - So a response cut at 512 tokens stays in the batch and the loss with the reward the math environment gives it, which is usually 0 when no answer was written. It is neither masked nor dropped.
- **R2, sm_89.** Transformer Engine has no sm_89 kernels. The Qwen3 path avoids it through `force_hf`. FlashInfer JIT architecture handling is **unverified**.
- **R3, one-GPU data plane.** Upstream tests it on 2 GPUs without checkpointing only.
- **R4, dataset revisions.** The export records the GSM8K revision (G-17), but NeMo Evaluator fetches the test split without one (benchmarks/gsm8k.py:37). `tierb-collect` therefore checks the evaluated set against the registered held-out set before it writes anything (TIERB-PROMPTS, HIGH-2). Every evaluated `problem_idx` must be one of the 1,319 held-out problems (a held-out problem that was not evaluated is a missing item of the gate). Each `expected_answer` must equal the held-out answer with commas removed (benchmarks/gsm8k.py:29-31). Each held-out question must appear in the record's `prompt`. A Hub revision change between export and evaluation is refused, not gated.
- **R5, sampler drift.** Without `--generation-config vllm`, Qwen3's `generation_config.json` sets `top_k` and `top_p` on the server. `merge-eval` checks each run's temperature, top_p, max_tokens, model and `enable_thinking` against the registration and the other runs (LOW-1). The training sampler differs from the gate's by design (R1).
- **R6, common random numbers.** Per-repeat seeds need K separate runs plus `merge-eval` (G-16). vLLM seeded sampling is reproducible only up to batch numerics.
- **R7, disk.** `keep_top_k=null` keeps every checkpoint, about 5-10 GB each (estimate).
- **R8, effect size.** The pins hold no reference curve for Qwen3-0.6B, so the long run may be INCONCLUSIVE.
- **R9, restarts.** There is no scheduler requeue on a local host, but a manual restart writes a second `exp_*` directory. `tierb-collect` refuses it; stop.
- **R10, container hardening.** The `docker run` commands above are the ones the reference run used: the container keeps the default network (needed in step 2 and for the server port in step 8), the default capabilities and a writable root filesystem, and runs as the image's user. `ext-launch` runs external containers with `--network none`, `--user <uid>:<gid>`, `--cap-drop ALL`, `--security-opt no-new-privileges` and read-only mounts; these flags were not tried with this NeMo RL image (Ray and vLLM start local services). Run Tier B only on a host you control, with nothing secret mounted into `<HOST_WORK>` or `<HF_CACHE>`; for steps 6 and 7 (no network needed, `HF_HUB_OFFLINE=1`) adding `--network none --security-opt no-new-privileges` is the first hardening to try, and its effect is untested.

## 8. Lessons from a first run on one RTX 4090 (single-GPU host)

Found on one RTX 4090 (24 GB) in a machine with about 32 GB of RAM and swap enabled. Each item names the step it changes.

- **Step 2, `--target release`.** Without it the build stops at the last Dockerfile stage (`trtllm-cache-export-status`, `FROM scratch`, Dockerfile:982-984) and tags a 10-byte image. The command above now passes it.
- **Step 2, build memory.** On a host with 32 GB RAM, `MAX_JOBS` alone does not bound the build: uv builds several CUDA packages at once, and flash-attn pins `MAX_JOBS=4` with `--threads 4` (pyproject.toml:609-617). The build that finished used `MAX_JOBS=2`, `NVTE_BUILD_MAX_JOBS=2`, `NVTE_BUILD_THREADS_PER_JOB=1` and a local one-line edit `flash-attn = { MAX_JOBS = "2", NVCC_THREADS = "2" }` (a recorded deviation; pass `--nemo-rl-modified` to `tierb-collect`). About 3 h 40 min.
- **Step 2, commit check.** `/opt/nemo-rl` belongs to UID 65532, so `git rev-parse` as root fails with "dubious ownership". Use `--entrypoint git nemo-rl:4f39fd8 -c safe.directory=/opt/nemo-rl -C /opt/nemo-rl rev-parse HEAD`.
- **All GPU steps, `--runtime=nvidia`.** With the NVIDIA Container Toolkit in CDI mode, `--gpus all` alone is refused.
- **All container steps, networking.** A Docker daemon without a bridge needs `--network host`; drop `-p 127.0.0.1:8000:8000` and add `--host 127.0.0.1` to `vllm serve`. The vLLM API is unauthenticated: publish it on the loopback interface only, never on all interfaces.
- **All container steps, `--shm-size`.** Must not exceed host RAM; 16g on a host with 32 GB RAM.
- **Steps 6-7, host RAM.** Ray's memory monitor killed workers at 95 % RAM (the CPU-offloaded policy holds about 18 GB). The workaround used: `-e RAY_memory_monitor_refresh_ms=0`, let the kernel swap, stop above 60 GB of swap. Peak swap in the long run: about 23 GB.
- **Step 7, GPU memory.** The first long attempt hit a CUDA OOM in `log_softmax` (fp32 logits, 4,251 tokens × 151,936 vocabulary × 4 B = 2.58 GB). `policy.automodel_cfg.activation_checkpointing=true` fixed it (1.22 GPU-h for 150 steps). Next fallback: `policy.sequence_packing.train_mb_tokens=2048`.
- **Step 5, reward check.** `rewards` is a list (`"rewards": [1.0]`); read it with a JSON parser, not `grep`.
- **Step 8, serving binary.** `uv run --extra vllm vllm` reinstalls the extra in every fresh container. Use the image's worker venv: `--entrypoint /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/vllm` (vLLM 0.29.0, the same venv training generation used). `vllm serve --help` needs a visible GPU.
- **Step 8, port.** Check the port is free first; if 8000 is taken, pick a free port, for example 8011, and pass it to `nel-config` (`--model-url http://localhost:8011/v1`).
- **Step 9, evaluator cache.** The training container creates root-owned files in a shared `<HF_CACHE>`; the host evaluator then fails with `PermissionError`. Give the host evaluator its own cache (`HF_HOME=<HOST_WORK>/hf-home-host`).
- **Step 9, scorer.** Gate on `numeric_match` (registration v2, G-21), not the default extraction, which scored the base model degenerately.

## Appendix A. Slurm form

For a Slurm cluster instead of a single-GPU host:

```python
from gatedtrain.backends.nemo_rl import launch_spec
s = launch_spec({"id": "rl"}, nemo_rl_dir="/path/to/nemo-rl", launcher="sbatch", out_dir="/path/to/out/rl",
                export_dir="/path/to/tierb/export", profile="smoke", image="<image@digest>",
                account="<account>", partition="<partition>")
```

`s["argv"]` has this form:

```sh
sbatch --no-requeue --nodes=1 --gres=gpu:1 --account=<account> --partition=<partition> --time=01:00:00 --job-name=gatedtrain-rl-unpromotable --export=ALL,COMMAND=<the local launcher's command>,CONTAINER=<image>,MOUNTS=<nemo-rl>:<nemo-rl> <nemo-rl>/ray.sub
```

- `COMMAND`, `CONTAINER` and `MOUNTS` drive NeMo RL's `ray.sub` (ray.sub:110-112; docs/cluster.md:10-24). `sbatch --export` splits on commas, so the builder refuses a comma in any of them.
- `--no-requeue` is the skeleton launcher. NeMo-Run is not used, because it implies `--requeue` (X21).
- **R9-OI-1 exposure:** an administrator or an automated health-check service can still requeue or restart the job. A second attempt writes a second `exp_*` directory, which `tierb-collect` refuses.
- The GPU-hour budget and every stop condition above apply unchanged.
