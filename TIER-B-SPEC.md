# Tier B helpers for the RTX 4090 run: specification

- See the [design references](README.md#design-references) for the rule and decision IDs used here.
- **Inputs:**
  - a research draft (since folded into `docs/tier-b-runbook.md`), with facts verified at NeMo RL `4f39fd8` and NeMo Evaluator `ea35ba3`;
  - the interim rules G-7 (`tests/README.md` section 9) and G-14 to G-17 (maintainers' working material, not published);
  - the maintainers' decision of 2026-11-09 to run Tier B on one RTX 4090.
- **Goal:** an operator on the GPU host goes from GPU outputs to a gate report and an impact answer with one command. Every Tier B mismatch in the draft §2 is fixed.

  The existing 158 tests keep passing. No GPU runs.

## 1. Helpers
- **H-1 `gatedtrain export-gsm8k --out DIR [--revision REV] [--source DIR]`.** Writes:
  - training shard files under `DIR/train/`, in split order;
  - the concatenation `DIR/train.jsonl`, outside the shard directory;
  - the held-out item set from the test split (`heldout/prompts.jsonl`; the spec's gate role `items`, type `item_set`, or the legacy role `prompts`, type `prompt_dataset`, since design revision 11; `export.json` keeps its key `prompt_set_hash`, which the registration records as `eval_distribution.item_set_hash`);
  - a content-addressed manifest (canonical JSON per RFC 8785);
  - a hashes file holding the concatenation's SHA-256 and the dataset revision (G-17).

  Rows are `{question, answer}` with the answer extracted after `####` (draft §1.4). `--source` reads a local copy of the dataset; the tests use a small fixture, never the network.
- **H-2 `gatedtrain verify-export DIR`.** Re-concatenates the shards in file-name order and checks the hash against the recorded one and against `train.jsonl`. Exit 3 on a mismatch.
- **H-3 `merge-eval`.** Merges K single-seed NeMo Evaluator runs (`<run>/gsm8k/results.jsonl`, draft §1.7) into one `results.jsonl`, relabelling `repeat` by registered seed order (G-16). It refuses when:
  - a run is missing or empty;
  - a run has mismatched problem sets;
  - a run has `repeats` ≠ 1;
  - the seed is not the one registered.
- **H-4.** Writes K NeMo Evaluator config files from the registration: benchmark, seed, temperature, `top_p`, `max_tokens`, `enable_thinking: false` (G-15), and the dataset revision where supported. Every key must exist at `ea35ba3`; cite each.
- **H-5 `build_command`.** Extended with:
  - the data overrides with `++` prefixes (draft §1.4);
  - the evaluator prompt and `enable_thinking: false` (G-15);
  - `policy.automodel_cfg.checkpoint.save_consolidated=final` (draft §1.6);
  - `keep_top_k=null`;
  - the working directory;
  - a `docker` launcher for a local host with the draft's run flags.

  The generations-per-prompt count is read from the config, not hard-coded, and the ledger uses it (G-7). Two profiles: `smoke` (10 steps, 2 × 4) and `long` (150 steps, 32 × 8; draft §1.5).
- **H-6 `gatedtrain tierb-collect <spec.yaml> --stage-out DIR --eval-baseline DIR --eval-candidate DIR [--runs-root R]`.** Produces a run directory like `gatedtrain run`:
  - manifests;
  - a lockfile with the image ID and the NeMo RL commit;
  - the ledger from the NeMo RL logs and checkpoints, against the exported shards;
  - the gate;
  - the report.

  It refuses with exit 3, before creating anything, when results are missing, empty or unpaired beyond the registered tolerance; this fixes the silent MISSING. After it, `gatedtrain impact` works on the run.
- **H-7.** Spec and registration support for the Tier B evaluator `gsm8k.nemo_evaluator@0.1` and trainer `rl.nemo_rl@0.1`, `budget.gpu_hours > 0`, and the sampler fields `top_p` and `enable_thinking`. The Tier A behaviour is unchanged.
- **H-8.** Templates `demo/tierb/spec.yaml` and `demo/tierb/registration.yaml`: GSM8K test set, K = 4, superiority, a margin and planning σ justified in comments as `internal_pilot` placeholders, and a GPU-hour budget of 8 (`budget: {gpu_hours: 8}`, interim rule G-10; operators set their own).

## 2. Docs
- Replace `docs/tier-b-runbook.md` with the single-GPU runbook built from the draft §4, with the helpers in place of the "blocked" notes.
- Keep the facts and citations, and mark every estimate.
- Keep the Slurm form as an appendix.
- Delete the draft once its content has moved.

## 3. Tests (written first)
- **Fixture tests for each helper:**
  - export from a tiny local GSM8K-shaped fixture, then verify, then tamper, after which verify must refuse;
  - merge across K fixture runs with the refusal cases;
  - an evaluator-config golden file;
  - `build_command` goldens for the `local`, `docker` and `sbatch` launchers and for both profiles;
  - the ledger with 8 generations per prompt;
  - `tierb-collect` end to end on recorded fixture outputs (NeMo RL-shaped logs and checkpoints, plus NeMo Evaluator-shaped results at `<run>/gsm8k/results.jsonl`), ending in PASS for one fixture and in a refusal for missing or empty results;
  - impact after collect.
- **The silent-MISSING regression:** pointing collect at the parent directory gives a refusal, not MISSING.
- **Citations:** each key and flag used in the commands and configs is checked against the pinned clones by a test that reads the clone files, and skips with a clear reason when the clones are absent.
