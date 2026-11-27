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
| `keeper`, `setup keeper`, `setup host`, `setup eval-tree`, `setup check`, `verify` | AI-native keeper (preview, Linux only): `keeper serve`, `keeper fingerprint`, `keeper confirm`; print the install commands of the keeper (`setup keeper`: same-user mode without sudo by default, `--mode separate-user` for the hardened install) and of the rootless Docker host; `--layout FILE` on `setup host` and `setup keeper` takes the paths from a TOML file (`setup layout` prints one with the defaults; a flag overrides the file; AN-34, see [the setup layout file](configuration.md#the-setup-layout-file)); `setup check` names the outside prerequisites and changes nothing; `verify` checks a keeper project's signed records (`--accept-mode same-user` accepts a same-user result, `--require-maintainer-images` fails on a user-added trainer image). `project create --require-keeper-mode separate-user` refuses a same-user keeper. See [Keeper modes](ai-native.md#keeper-modes), the [quickstart without sudo](sudo-free-quickstart.md) and [rootless containers](ai-native.md#rootless-containers-the-local-backend-and-the-keepers-container-step-3a). |
| `image` | The image lock: `image list` shows the container images gatedtrain accepts, `image check LOGICAL --ref REF` (or `--id`) says whether an image is accepted, `image add LOGICAL --id ID --recipe NAME ...` records an image you built in your own config. See [Trainer images and the image lock](ai-native.md#trainer-images-and-the-image-lock) and [NeMo RL on a rootless host](rootless-nemo-rl.md). |
| `impact` | Look up a shard across recorded runs. Each answer reports its precision as `row`, `iteration` or `stage`. |

`--json` is a top-level option and goes before the subcommand: `gatedtrain --json setup check`.

Gates support superiority, non-inferiority and equivalence. The registration fixes the margin, alpha, target
power and evaluation seeds before the run. The report includes the decision, achieved power and an
`underpowered` flag.

The ideas behind gates and `impact` are in [Concepts](concepts.md); the `ext-*` commands in
[External runs](external-runs.md).
