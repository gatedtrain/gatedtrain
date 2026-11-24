# Walking-skeleton prototype: specification

- See the [design references](docs/development.md#design-references) for the rule and decision IDs used here.
- **Decision:** maintainers' decision of 2026-11-09 to build a walking-skeleton prototype.
- **Shape source:** verification-plan E2E-SKEL, L2-E-skel-compat and UX-03.
- **Design sources:**
  - 04 §3 (workflow spec) and §4.2 (gates);
  - 05 §3 (gate library), §11 (lineage model, §11.2 precision, §11.3 impact query) and §12.2 (`ConsumptionRecord`);
  - the format conventions (canonical JSON per RFC 8785, content-addressed manifests).

- The [test reference](tests/README.md) links the public interface to its design rules. Section 7 lists the acceptance evidence.
- **Not in scope:** no NeMo RL or NeMo Evaluator change, no run contract, no R, no hooks, no claim of row-level lineage.

## 1. Layout and environment
- **Root:** the repository root. Code is in `src/gatedtrain/`, tests are in `tests/`, docs are in `docs/`, and the CLI is `gatedtrain`, which is `python -m gatedtrain`.
- **Environment:** its own virtual environment, `.venv` at the repository root, created with `uv`. A `pyproject.toml` lists the dependencies, and `uv.lock` pins them.
  - Allowed dependencies: `numpy`, `scipy`, `pyarrow`, `duckdb`, `pyyaml`, and CPU-only `torch` for the stand-in trainer.
  - Nothing else without a maintainers' decision.
  - Only one `uv` command at a time in this environment.
- **Python 3.12 or 3.13.**
- **Tests use `unittest`.** The whole Tier A suite runs on CPU in under 15 minutes.

## 2. The path (`gatedtrain run <spec.yaml>`)
1. **Spec.**
   - Parse a workflow spec that is a strict syntactic subset of the 04 §3 YAML: one stage and one gate, no field the 04 §3 grammar lacks.
   - Also check the 04 §4.2 required-field rules for a one-stage, one-gate workflow.
   - Errors name the rule, the field and the fix.
2. **Registration.**
   - Read the registration file named by the spec. It covers clause, metric, margin δ, α, power, σ plan, unit, K, evaluation seeds, prompt-set hash, and analysis-code version (05 §3.7).
   - Record its SHA-256.
   - The run refuses to start if the registration is missing, a blocking gate has no margin (05 §3.1), or the registration hash given in the spec does not match the file.
3. **Plan.** `plan()` computes the required n, the MDE and the design effect under the σ plan (05 §3.6). It reports `underpowered` up front if the prompt set is smaller than n.
4. **Baseline evaluation.** Evaluate the baseline checkpoint on the prompt set, with K samples per prompt and the registered evaluation seeds (common random numbers, 05 §3.3).
5. **Training stage.** Run the stage's trainer backend on the training shards.
6. **Candidate evaluation.** Evaluate the trained checkpoint on the same prompts, with the same K and the same seeds.
7. **Gate.**
   - Pair records by `(problem_idx, repeat)` as NeMo Evaluator's `load_paired_records` does (`src/nemo_evaluator/engine/comparison.py:360` at `ea35ba3`). Reuse that function when it can be imported without heavy dependencies; otherwise reimplement it and add a test that both agree on a fixture.
   - Compute per-prompt paired differences of raw scores, with K averaged per prompt (05 §3.3).
   - `evaluate()` returns Δ̂, the interval (prompt-clustered bootstrap, 05 §3.4; seedable), the α used, achieved power, MDE, `underpowered`, and the status.
   - The statuses are PASS, FAIL, INCONCLUSIVE and MISSING, with the 05 §3.1 rule for underpowered results.
   - Superiority, non-inferiority and equivalence are all implemented; the demo spec uses superiority.
8. **Manifests and lockfile.**
   - Content-addressed manifests (canonical JSON per RFC 8785, SHA-256) exist for the baseline checkpoint, the trained checkpoint, the training shards, the prompt set, the registration, the evaluation results and the gate report.
   - A lockfile records the backend identity (stand-in: package version and torch version; Tier B: container digest and NeMo RL commit) and the config hash, with no allow-list checks.
9. **Ledger.** A Parquet ledger of `ConsumptionRecord`-shaped rows (05 §12.2) at precision `iteration` (05 §11.2).
   - It is built only from what the trainer already writes: per-step `train_data_step<N>.jsonl` and `consumed_samples` in the checkpoint state, the same names and shapes as NeMo RL `grpo_sync.py:1323, :1353, :1408-1414, :1458-1460` at `4f39fd8`.
   - It maps each step to the training shard it drew from.
   - There are no row IDs.
10. **Impact query.**
    - `gatedtrain impact <shard-hash or shard path>` runs a DuckDB query over the ledger and the lineage edges (05 §11.3).
    - It returns every checkpoint that possibly consumed the shard, with precision `iteration`, and nothing for a shard that was not used.
11. **Report.**
    - A gate report (JSON and a short text rendering) with every 05 §3.7 field the skeleton has.
    - Fields it cannot fill (independence tier, contamination, family α) are printed as `not-assessed (skeleton)`.
    - Tags: `prototype`, `unpromotable`, `registration: unsigned`.

The run writes everything under one run directory and never overwrites an earlier run.

## 3. Tier A stand-ins (`skeleton-stub`, CPU)
Every stand-in is marked `skeleton-stub` in its code, its output and the docs.

- **Task.** A small sequence task with an exact-match score per sample, for example 2-digit addition as character strings.
  - The training data is split into shard files.
  - Held-out evaluation prompts are disjoint from the training data.
  - There are at least 200 prompts.
- **Model.** A tiny character-level transformer in torch, CPU only, a few hundred thousand parameters.
- **Baseline.** A checkpoint from a short warm-up run, so the baseline is partly capable and neither 0% nor 100%.
- **Trainer backend `cpu_stub`.**
  - Supervised fine-tuning on the stage's shards.
  - Deterministic under a seed.
  - Writes `train_data_step<N>.jsonl` and a checkpoint whose state holds `consumed_samples` and the data-loader state, as NeMo RL does.
  - Each step's records identify the shard read. NeMo RL's file has no dataset row ID, so the stand-in must not add one; it may add a shard field only if the ledger writer would get the same information from NeMo RL's data-loader state. Otherwise it maps steps to shards from `consumed_samples` and the shard order. A test covers whichever way is chosen, and the spec's "no row-level lineage" rule holds.
- **Evaluator backend `cpu_stub`.**
  - Samples K completions per prompt with temperature sampling under the registered evaluation seeds.
  - Writes `results.jsonl` in NeMo Evaluator's per-sample format (`src/nemo_evaluator/engine/artifacts.py:98` at `ea35ba3`), keyed by `problem_idx` and `repeat`.

## 4. Tier B (GPU, run by hand; code and runbook only)
- **Trainer backend `nemo_rl`:**
  - an `sbatch --no-requeue` wrapper, or a plain command on a single GPU host, around one NeMo RL recipe at `4f39fd8`;
  - outputs tagged `unpromotable`;
  - R9-OI-1 exposure noted.

  Pick a recipe that runs on one GPU with a small model and cite it.
- **Evaluator backend `nemo_evaluator`:** a NeMo Evaluator run on the baseline and on the trained checkpoint.
- **Runbook `docs/tier-b-runbook.md`:** host prerequisites, commands, expected outputs, and a GPU-hour budget.
- The test suite never runs Tier B. Its tests cover command construction and parsing of recorded sample outputs only.

## 5. Acceptance tests (Tier A, all on CPU)
- **E2E-pass.** From a clean run directory, `gatedtrain run demo/spec.yaml` completes with exit 0:
  - the trained model's exact-match rate beats the baseline's;
  - the superiority gate is PASS, with Δ̂ > 0, a lower bound > δ_sup and `underpowered` false;
  - the report has every required field.
- **Negative control: no training.** The same spec with learning rate 0 gives a gate that is not PASS: FAIL, or INCONCLUSIVE when underpowered.
- **Negative control: too few prompts.** A prompt set below the planned n gives `underpowered: true`. A non-rejection is INCONCLUSIVE, never FAIL.
- **Negative control: bad training.** A stage that damages the model (for example, training on shuffled labels) gives FAIL on superiority, and FAIL on a non-inferiority variant.
- **Registration integrity.** If the registration changes after the spec records its hash, the run refuses with a clear error.
- **Impact query.**
  - A shard used by the stage returns the trained checkpoint with precision `iteration`.
  - An unused shard returns nothing.
  - The baseline checkpoint is never returned for a stage shard.
- **Determinism.** Two runs with the same seeds give identical gate statistics and identical content hashes for checkpoints, results and ledger, ignoring wall-clock fields, which must be excluded from the hashes.
- **Statistics unit tests.** For each of the three gate types, check against hand-computed or SciPy reference values:
  - MDE and the σ_U factors from 05 §3.6, for example 1.56 s at df = 4;
  - bootstrap interval coverage on synthetic data, within tolerance, with a seeded random-number generator;
  - the underpowered PASS and INCONCLUSIVE rules.
- **L2-E-skel-compat fixture.** Written before the code: the demo spec files satisfy the 04 §3 subset and §4.2 rules.
- **Errors.** Each spec and registration error names the rule, the field and the fix.
- **Stub marking.** Every stand-in module and output carries `skeleton-stub`.

## 6. Docs (CPU path)
- `docs/quickstart.md`: install, first run, reading a gate report.
- `docs/tier-b-runbook.md`.
- A findings log, `docs/findings.md`, for design gaps met while building (interim rules are listed in `tests/README.md` section 9). A gap means stop and record it as a design gap; do not invent a rule.

## 7. Evidence
- Hashes of all code files.
- Three clean runs of the full suite.
- Mutants:
  - the gate ignores pairing;
  - the gate uses group-normalized scores;
  - the underpowered rule inverted;
  - the registration hash check removed;
  - the ledger omits a step;
  - the impact query drops the precision filter.

  Each must be killed.
- The E2E-pass gate report, attached.
