# Commands

This page lists every `gatedtrain` subcommand with one line on what it does. Use it to find the right command; the
linked pages explain each group, and `gatedtrain <command> --help` shows all flags.

`gatedtrain` (or `python -m gatedtrain`) has these subcommands:

| Command | What it does |
|---|---|
| `validate`, `warmup`, `run` | Check the spec and registration, build a baseline, or run the CPU demo. |
| `export-gsm8k`, `verify-export`, `nel-config`, `merge-eval`, `tierb-collect` | Prepare data and evaluation for a NeMo RL run and collect its results. |
| `ext-launch` | Check an external run request, write the launch record and print the command for the operator. Use `--recheck` to verify the record before running it. |
| `ext-from-nel`, `ext-merge-eval`, `ext-arm-fill` | Build and verify the per-item results for each side of the evaluation. |
| `ext-collect` | Compare the candidate and baseline on the same evaluation items, then write the gate report, consumption ledger and checkpoint lineage. |
| `project` | AI-native projects (preview): `create`, `plan`, `run`, `status`, `report`, and `job list`, `job logs`, `job cancel` for the jobs of the local backend. See [AI-native projects](ai-native.md). |
| `keeper`, `setup keeper`, `setup host`, `verify` | AI-native keeper (preview, Linux only): `keeper serve`, `keeper fingerprint`, `keeper confirm`; print the install commands of the keeper and of the rootless Docker host; check a keeper project's signed records. See [the keeper](ai-native.md#the-keeper-step-2) and [rootless containers](ai-native.md#rootless-containers-the-local-backend-and-the-keepers-container-step-3a). |
| `impact` | Look up a shard across recorded runs. Each answer reports its precision as `row`, `iteration` or `stage`. |

Gates support superiority, non-inferiority and equivalence. The registration fixes the margin, alpha, target
power and evaluation seeds before the run. The report includes the decision, achieved power and an
`underpowered` flag.

The ideas behind gates and `impact` are in [Concepts](concepts.md); the `ext-*` commands in
[External runs](external-runs.md).
