# gatedtrain quickstart (CPU, Tier A)

The CPU demo trains a character-level transformer of about 450,000 parameters on two-digit addition and
compares the trained checkpoint with its baseline. The demo backends, marked `skeleton-stub`, write the same
file formats the ledger and pairing code read from NeMo RL (`4f39fd8`) and NeMo Evaluator (`ea35ba3`).

## 1. Install

Use the pip setup in the [repository README](../README.md#try-it), including the `tier-a` extra.
On macOS you can also use the locked environment. From the repository root, run the following command with
[`uv`](https://docs.astral.sh/uv/). It downloads a managed CPython 3.13 if needed.

```sh
uv sync --python 3.13 --python-preference only-managed --locked --extra tier-a
```

`uv sync` installs gatedtrain in editable mode into `.venv`. Activate it from the repository root.

```sh
. .venv/bin/activate
```

## 2. First run

1. Train the baseline checkpoint. The short warm-up leaves room to measure the effect of fine-tuning.

   ```sh
   cd demo
   gatedtrain warmup warmup.yaml --out baseline
   ```

   It takes about 50 s on a recent laptop. It refuses if `baseline/` already exists.
2. Check the spec and its registration without running anything:

   ```sh
   gatedtrain validate spec.yaml        # prints "ok"
   ```

3. Run the whole path: plan, baseline evaluation, training, candidate evaluation, manifests, lockfile, ledger, gate and report.

   ```sh
   gatedtrain run spec.yaml --runs-root runs
   ```

   It takes about 1 minute. It prints `run_dir: <path>` and then the text gate report. Each run gets a new directory under `runs/`; an earlier run is never overwritten.
4. Ask which checkpoints possibly consumed a training shard:

   ```sh
   gatedtrain impact data/train/shard-00003.jsonl --runs-root runs
   gatedtrain impact data/unused/shard-00000.jsonl --runs-root runs --json   # empty result
   ```

   The answer is "possibly consumed" at precision `iteration`: the ledger knows which shards each training step read, not which rows (05 section 11.2).

Exit codes: 0 success (for `run`, the gate passed); 1 the run finished but the gate did not pass; 2 usage error; 3 refused before anything ran.

## 3. Reading a gate report

The gate report is in `<run_dir>/report/gate_report.json`, with a text copy in `gate_report.txt`. The key lines:

| Line | Meaning |
|---|---|
| `status` | PASS, FAIL, INCONCLUSIVE or MISSING. Only PASS opens a blocking gate. |
| `status_reasons` | Why a result is MISSING (`no_pairs`, `missing_above_bound`, `k_mismatch`, `seed_mismatch`, `malformed_record`, `unknown_item`, `model_hash_mismatch`, `evaluator_hash_mismatch`) or INCONCLUSIVE (`min_paired`, `zero_width_interval`, `post_hoc`, `underpowered`); empty for an unqualified result. |
| `p_value` | One-sided p-value for the gate's null, by inverting the BCa interval (`bca_inversion`); TOST for equivalence; null for a zero-width interval. `p_value: <= <p>` (JSON `p_value_at_floor: true`) when no bootstrap mean lies in the tail, so the value is a bound set by the number of replicates. The decision still comes from the interval. |
| `hypothesis_type` | superiority, non_inferiority or equivalence (05 section 3.1). |
| `direction` | `higher_is_better` or `lower_is_better`, from the clause (05 section 3.1). Scores of a lower-is-better metric are negated, so Δ > 0 always favours the candidate. |
| `delta_hat` | Mean over items of (candidate mean − baseline mean), oriented. The repeats present in both arms (K = 4 in the demo) are averaged per item first, on raw exact-match scores. |
| `natural_units` | The difference candidate − baseline and its interval in the metric's own units (equal to `delta_hat` and `interval` for a higher-is-better metric). |
| `interval` | Two-sided (1 − 2α) item-clustered BCa bootstrap interval. Its lower end is the one-sided (1 − α) bound that the pass rule uses. |
| `margin` | δ_sup for superiority (0 in the demo), or δ for non-inferiority and equivalence. |
| `achieved_power`, `mde` | Power at the planned effect, and the minimum detectable effect, both from the observed SD of the per-item differences (05 section 3.6; TOST power for equivalence). |
| `underpowered` | true when the achieved power is below target or the item set is smaller than the planned n (`plan.n_required`). An underpowered pass is still PASS, flagged. An underpowered non-rejection is INCONCLUSIVE, never FAIL. |
| `plan` | `n_required` under the registered σ plan (`ucb80`: σ_U = s·√(df/χ²₀.₂₀,df)), the MDE at the available n, and `underpowered` known before the run. |
| `registration_hash` | SHA-256 of the registration file. The run refuses if the file changed after the spec recorded its hash. |
| `independence_tier`, `transfer_test`, `contamination`, `family_alpha` | `not-assessed (skeleton)`: the skeleton has no data for these. |
| `sd_diff` | Sample standard deviation (ddof = 1) of the per-item paired differences; the input to `achieved_power` and `mde`. |
| `sd_zero` | true when every per-item difference is equal, as with identical models. The reported achieved power is the formula's limiting value of 1.0. |
| `unpaired_baseline`, `unpaired_candidate` | Items present in one arm only. They are left out of the gate and the scores, and a `warning:` line appears when either is above 0. |
| `n_items`, `samples_per_item`, `unit` | Complete pairs (items with at least one repeat present in both arms), the registered K, and `item`. |
| `missing_baseline`, `missing_candidate`, `missing_reasons`, `max_unpaired` | Items of the registered design (the registered item set times K) missing from each arm, the count per reason (`absent`, an evaluator error class, `out_of_range`, `no_seed`), and the registered bound (default 0). Above it the gate is MISSING (05 section 3.4). |
| `min_paired` | Registered minimum of complete pairs (default 10); fewer is INCONCLUSIVE. |
| `k_actual` | [min, max] repeats per item in each arm. Fewer than K are missing items; a repeat outside range(K) is MISSING (`k_mismatch`). |
| `eval_distribution`, `item_set_hash` | The registered evaluation distribution (`item_set_hash`, `env_version`, `env_params_version`) and the hash of the evaluated item set. |
| `registration_amends`, `post_hoc` | The prior registration's hash for an amendment, and whether it followed candidate results (then a PASS is reported as INCONCLUSIVE). |
| `evaluator_hash_registered` | The registration's `evaluators[0].manifest_hash`; when it differs from `evaluator_hash` the gate is MISSING. |
| `disjoint_from` | `not-assessed (skeleton)`: the spec declares held-out data disjoint from training data, but the skeleton does not check it. |
| `tags` | `prototype`, `unpromotable`, `registration: unsigned`. |

## 4. The demo controls

| Spec | What it shows | Expected status |
|---|---|---|
| `spec.yaml` | Fine-tuning on 5 clean shards | PASS |
| `spec-no-train.yaml` | Same with lr = 0: identical weights, identical samples | INCONCLUSIVE (Δ̂ = 0, `sd_zero: true`, zero-width interval, 05 section 3.6) |
| `spec-no-train-few-prompts.yaml` | lr = 0 on the 30-prompt set | INCONCLUSIVE (underpowered, `sd_zero: true`, zero-width interval) |
| `spec-few-prompts.yaml` | 30 prompts, below the planned n = 172 | underpowered; PASS or INCONCLUSIVE, never FAIL |
| `spec-bad-train.yaml` | Training on shuffled labels | FAIL |
| `spec-bad-train-ni.yaml` | Same, non-inferiority gate with δ = 0.10 | FAIL |

## 5. Errors

Every spec or registration error is one line naming the rule, the field and the fix:

```
error[REG-HASH] artifacts.registration.uri: registration hash mismatch: ... | fix: ...
```

If you edit a registration, put its new SHA-256 in every spec that names it.

## 6. Tier B (GPU, run by an operator)

`gatedtrain run` covers Tier A only. It refuses a Tier B spec (`rl.nemo_rl@0.1`) and never launches a GPU job. The Tier B path, for one RTX 4090 with NeMo RL 4f39fd8 and NeMo Evaluator ea35ba3, is in `docs/tier-b-runbook.md`, with templates in `demo/tierb/`. On the Linux host, install the torch-free helpers from `requirements-tierb-host.txt` (runbook step 3b), not with `uv sync --locked`. Its helpers are:

| Command | Purpose |
|---|---|
| `gatedtrain export-gsm8k`, `gatedtrain verify-export` | GSM8K as local shard files under a manifest, and a re-check before each run |
| `gatedtrain nel-config` | K NeMo Evaluator configs, one per registered seed |
| `gatedtrain merge-eval` | K single-seed evaluator runs merged into one arm |
| `gatedtrain tierb-collect` | recorded NeMo RL and NeMo Evaluator outputs turned into a run directory: gate report, ledger, lineage. `gatedtrain impact` then works on it. |

## 7. External runs (adapters, GPU, run by an operator)

`ext-collect` turns the outputs of a framework that trains outside gatedtrain into the same run directory as `tierb-collect` (MULTIMODAL-SPEC.md; tests/README.md section 11.18). The adapters are plugins of the recipe repositories; the example below is the TRL recipe's adapter `trl_grpo` (`gatedtrain-recipe-trl-grpo-gsm8k`: TRL GRPO on HF checkpoints, trainer `rl.trl_grpo@0.1`, evaluator `gsm8k.nemo_evaluator_items@0.1`). Nothing here launches a process.

| Command | Purpose |
|---|---|
| `gatedtrain ext-launch --adapter A --profile smoke\|full --request REQ.yaml --gpu-log <HOST_WORK>/gpu-log.md [--smoke-record smoke.json] --out-record launch.json` | checks the request (pinned image, read-only inputs, one read-write output, no host-work or Docker-socket mount, `--network none` unless a reason is given, non-root user, `HF_HUB_OFFLINE=1`, no credential), writes the launch record and prints the `gpu-step.sh` command; the operator runs it. `full` needs a PASS smoke record. |
| `gatedtrain ext-from-nel --registration R --adapter A --evaluator E --items DIR --checkpoint DIR --out ARM RUN_0 ... RUN_{K-1}` | K single-seed NeMo Evaluator runs (registered seed order) to one per-item arm (`items.jsonl`, `eval_meta.json`, `merge.json`) |
| `gatedtrain ext-merge-eval --registration R --out ARM RUN...` | single- or multi-seed per-item arms merged into one arm, repeats relabelled by registered seed order |
| `gatedtrain ext-collect SPEC --adapter A --stage-out DIR --launch-record launch.json --smoke-record smoke.json --eval-baseline ARM --eval-candidate ARM [--runs-root R]` | refuses before writing on any broken input (exit 3), else writes the run directory and the gate report (exit 0 PASS, 1 otherwise); `gatedtrain impact` then works on it, across runs with `--runs-root` |

The trainer config is `{backend: trl_grpo, profile: smoke | full}` and must match the launch record's profile. The run's output directory must hold `env-freeze.txt` (the launcher's `pip freeze` pre-step), whose `trl` version equals the record's. The TRL checkpoint layout the adapter reads is documented in the recipe's `runtime/README.md` and its adapter module.
