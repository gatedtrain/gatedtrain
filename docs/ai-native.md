# AI-native projects (preview)

`gatedtrain project` runs one goal as a project directory: a fixed pipeline of five trust stages (`intake_seal`,
`preregister`, `evaluate_sealed`, `gate`, `report`) around the stages a pipeline file plugs in. Step 1 is a preview
for CPU-only work and the synthetic scenario pack in `gatedtrain.testing`, which is not registered as an entry point.

```sh
gatedtrain project create ./proj --goal "..." --data labelled.jsonl --pipeline pipeline.yaml [--gpu-hours H]
gatedtrain project plan ./proj <stage>                       # what a stage would do; changes nothing
gatedtrain project run ./proj <stage> [--decision file.json] # stages run in pipeline order
gatedtrain project status ./proj
gatedtrain project report ./proj
```

`--pipeline` is the path of a pipeline YAML. `--gpu-hours` is recorded in
`project.yaml` as the budget; step 1 enforces nothing. `run` of the `gate` stage exits 0 on PASS and 1 otherwise;
a refusal exits 3. Re-running a stage with unchanged inputs returns its recorded result.

Without a keeper (step 1, in process), a project scores its sealed items once:
after the first scoring `evaluate_sealed` refuses to score again, for the same model reference too (a rerun with
unchanged inputs returns the recorded scores). A model reference must content-address its weights (for example
carry their hash); step 1 does not check this, so a reference to a mutable checkpoint path lets the gate records
describe weights the reference no longer points to after a retrain. The gate spends the project's single alpha: a second gate attempt is refused. A project bound to a keeper
follows the keeper's alpha rule instead (below). A top-level `--json` prints one result object per call.

**Trust boundary in step 1.** Stages run in the operator's process. CLI outputs carry only hashes of the sealed
items, but in step 1 they sit in plain files under `<root>/trust/` that the operator and any stage can read and
rewrite, as can the attempt log; the report's `sealed_guarantee` says so ("unsealed (step 1): the operator process
can read the sealed items, and every stage can read and rewrite the trust records and the attempt log").
A project bound to a keeper (below) moves these records out of the operator's reach. In step 1 the alpha is per project: a new project on the same data
would get a fresh alpha and the same sealed split, so a new test needs new sealed data. The report's known limits
list these points, the model-reference rule, that the metric confirmation is the decision flag `metric_confirmed` set by whoever runs the
stages, and how many `evaluate_sealed` attempts the project made.

`project create` copies the pipeline into `<root>/pipeline.yaml` and records its sha256; the project refuses a copy
that was changed. To use another pipeline, create a new project. A stage that refuses in `run` before it writes any
record, spends alpha or claims a trust record is logged as `refused` and changes no stage state; a refusal after any
of these, or from the output checks, is `failed`.

## The keeper (step 2)

The keeper is a daemon under its own OS user (Linux only: a Unix socket with a peer-credential check). It alone holds
the sealed items, the registrations, the claims, the per-item records, the alpha ledger and an Ed25519 signing key;
the stages of a bound project become requests to it, and it signs every record it returns. Only the owner's uid and
its own may connect.

```sh
gatedtrain setup keeper [--owner-uid UID] [--packages NAME==VERSION ...]   # prints sudo commands; runs nothing
gatedtrain keeper fingerprint --socket /run/gatedtrain-keeper/keeper.sock    # keep the fingerprint out of band
gatedtrain project create ./proj --goal "..." --data labelled.jsonl --pipeline pipeline.yaml \
    --keeper /run/gatedtrain-keeper/keeper.sock
gatedtrain keeper confirm ./proj --decision preregister.json [--yes]          # before `run ./proj preregister`
gatedtrain verify ./proj --keeper /run/gatedtrain-keeper/keeper.sock          # or --fingerprint sha256:...
```

`setup keeper` prints the commands that create the keeper user, its 0700 state directory, a root-owned virtual
environment with gatedtrain and the scenario packs pinned to versions, and a systemd unit running `gatedtrain keeper
serve`; it writes nothing. The keeper refuses to start when its code, environment, state directory or socket
directory can be changed by another user (AN-13); a path holding `%`, `$`, whitespace, a quote or a backslash is
refused by `setup keeper`, since systemd would rewrite it in the unit. Known limit of that check: a module first
imported after the start-up check that lies outside gatedtrain, the installed scenario packs and gatedtrain's runtime
dependencies is not checked, and site-packages is not walked as a whole. `project create --keeper` binds the project to the key the keeper serves (`trust/keeper.json`,
`trust/keeper.pub`); a bound project never falls back to step 1: an unreachable keeper (AN-12) or one serving another
key (AN-14) is refused. `keeper confirm` shows the sentence that says what the scorer counts and the margin, and
records it in the keeper only after you type `yes` (or pass `--yes`; under `--json` it never prompts); `preregister`
refuses a registration without a matching confirmation (AN-18). The scorer comes from the library (`exact_match`,
`numeric_match`, `normalised_match`, `set_f1`, `field_f1`) or from a pack by reference (AN-17).

**Alpha rule.** Under the keeper the alpha is not per project but per sealed set, by alpha-investing (Foster-Stine):
the initial wealth is W0 = 0.05 * (1 - 0.05) = 0.0475; attempt j bids alpha_j = W_{j-1} / 2 (at most
W_{j-1} / (1 + W_{j-1}), so a bid never costs more than the wealth; this binds only above a wealth of 1); a FAIL, INCONCLUSIVE,
MISSING or crashed attempt costs alpha_j / (1 - alpha_j); a PASS earns the payout 0.05; an attempt whose bid would
be below 0.005 is refused. Intake plans power at W0 / 2 = 0.02375. Each `evaluate_sealed` spends its alpha before any
item is scored, so a project may gate several candidates (rerun `train`, then `evaluate_sealed` and `gate`)
until one passes; a PASS ends the project's attempts. A new project on byte-identical data shares the sealed set and its wealth;
other data holding a sealed item are refused (AN-16). The report lists every attempt with its alpha, cost and payout
and states the guarantee: "alpha-investing (Foster-Stine) over the gate attempts on this sealed set controls the
marginal false discovery rate (mFDR) at level 0.05". Its known limits qualify this: it assumes each test is valid given
the earlier outcomes; the sealed items are reused across attempts and only each attempt's status is revealed. A report
closes the sealed set: no further attempt on it, from any project, and a new project on the same data is refused at
intake.

**Oracles.** The keeper answers some requests about sealed data with counts and does not limit how often they are
asked: opening a project is free and a refused intake stores nothing, so anything running as the user can learn from
intake refusals how many inputs of a candidate data file were sealed before (and, repeated, which); a registration
refused for its scorer names how many sealed labels that scorer cannot take. The report lists this in its known
limits.

**Guarantee.** With the keeper under its own user, the report's `sealed_guarantee` reads: "the sealed items,
registrations, claims, ledger and per-item records are held by a separate OS user running code the user's account
cannot change, and every record is signed; the keeper found no rootful Docker or containerd access by the user's
account, nor the account in the docker, lxd, incus-admin or disk group, when this was stated or at any earlier check
on this sealed set (sudo is not checked); a scenario pack's per-item generator runs the candidate inside the keeper
process, and a batch generator runs it in a container of the keeper's own rootless Docker with no network, a read-only
root file system and the predictions as its only output; either way the candidate is a copy of the model files the
keeper hashed, and a model format that executes code is refused; an escape from that container through a bug in the
inference code or the kernel gives the keeper's account; the original data file and the metric confirmation are only
as strong as the user's own account; a root user, also one reached through sudo, or the keeper's own account can break
all of it." A keeper running as the owner's own user (as in the tests) says so instead: "the keeper runs as the owner's
own OS user, so anything running as the user can read its state and signing key: the records are signed, but nothing
is sealed (a test or development setup)".

**`verify`.** `verify ./proj --keeper SOCKET` checks every signed record in the project (the registration, each
receipt in `trust/keeper/records.jsonl`, the gate result, `report.json` and `report.md`) against the key the keeper
serves, and that the project holds every record the keeper signed for it; `--fingerprint` checks against a
fingerprint held out of band, offline, and cannot detect a deleted record (it says so). It exits 0 when every record
checks, 3 on a bad signature, an edited record, a replaced `trust/keeper.pub` or another fingerprint (AN-14), and 1
without either anchor or against a keeper that runs as the owner's own uid (consistency only: the owner can read that
keeper's key), and also, under any anchor, when a report states any guarantee but the separate-user one above (the
same-user text, the downgraded text of the rootful-access check below, or the 0.4.0 text), with a warning naming it. The driver keeps every envelope the keeper signed for the project: after each `evaluate_sealed`, `gate`
and `report`, and on `project report`, it fetches the keeper's list and adds what it lacks (for example the claim of an
evaluation it stopped waiting for), each verified first. A record the keeper signed that the project still lacks fails
`verify --keeper` with exit 3.

## Rootless containers, the local backend and the keeper's container (step 3a)

Step 3a runs a project's jobs and the keeper's sealed evaluations in containers on rootless Docker (preview, Linux
only). The owner and the keeper each run their own rootless Docker daemon, and the owner leaves the rootful `docker`
group, through which anything running as the owner could otherwise read the keeper's state.

```sh
gatedtrain setup host [--inference-image REF@sha256:...] [--eval-requirements FILE --eval-python PATH] \
    [--confirm-group-removal]                     # prints one root script; runs and writes nothing
sudo bash gatedtrain-host-setup.sh                # after reading it; delete it afterwards
gatedtrain project job list ./proj                # also: job logs ./proj JOB_ID [--tail N], job cancel ./proj JOB_ID
```

**Host.** `setup host` prints one bash script for root. All its checks come before its first change: it runs as root,
the owner has the uid it was printed for, an NVIDIA CDI specification of `nvidia.com/gpu` exists in `/etc/cdi` or
`/var/run/cdi`, `slirp4netns` or `pasta` is installed, no subordinate id range of the owner or the keeper user
covers the other's uid or gid, and the parents of the directories it creates are root-owned and
not writable by others. It then installs `uidmap` and `docker-ce-rootless-extras` at the installed `docker-ce`
version, adds the AppArmor profile rootlesskit needs on Ubuntu 24.04, creates the keeper user (as `setup keeper`) and
the subordinate id ranges, removes the owner from the `docker` group before any of the owner's user services start
(`--confirm-group-removal` pauses first), enables lingering, starts one rootless daemon per user with its own data
root (`/srv/gatedtrain/docker` for the owner, `/var/lib/gatedtrain-keeper-docker` for the keeper) and Docker's native
CDI, and creates the owner's Docker context `gatedtrain`. Jobs write the models the keeper reads under the hand-off
directory `/srv/gatedtrain/out` (the owner's, mode 0755). With `--inference-image` the keeper's daemon pulls that image
and a drop-in of the keeper's systemd unit starts `keeper serve` with its image ID; with `--eval-requirements`
(hash-pinned, binary wheels only) the script builds a root-owned eval environment (`/opt/gatedtrain-keeper/eval-venv`)
from an interpreter under `/opt/gatedtrain-keeper/python`, mounted read-only into the keeper's containers. A login
session keeps the `docker` group until it ends, so log out everywhere afterwards; when the owner's user manager
was already running (the script warns), restart it once those sessions have ended (`sudo systemctl restart
user@<uid>.service`, or reboot), since the owner's rootless daemon started under it with the group. Then set in the host file (see
"Configuration"):

```toml
rootless = true
output_root = "/srv/gatedtrain/out"
```

**Local backend.** A pack stage reaches the project's job backend as `ctx.backend` (AN-22 to AN-24):
`new_output_dir(name, key=...)`, `submit(job)`, `status`, `wait`, `logs`, `cancel`, `artifacts` and `hand_off`. A
`Job` (an image pinned by digest, the argv, read-only input mounts, one output directory, environment names the
adapter allows, GPU or not, a time limit and a GPU-hour estimate) is checked by the launch rules `ext-launch` applies
to a request of its adapter before anything starts, and runs detached on the owner's rootless daemon (the context
`gatedtrain`). The backend refuses a daemon that is not rootless or whose socket is not the owner's, and an image the
daemon does not hold (no job pulls one). The project keeps the records under `<root>/jobs/`: the job (environment
names only), one row per attempt start, end, stop and cancel, and the tail of each attempt's log. Each attempt writes
into a fresh `<output dir>/a<n>`. `failed`, `preempted` and `lost` attempts are retried up to `max_attempts` (default
3); `oom`, `timed_out` and `cancelled` ones are not. A GPU attempt waits up to 900 s for a free GPU and is refused when
the project's committed GPU-hours plus its estimate pass the budget (`--gpu-hours`); a running attempt is stopped at
its time limit or at the budget. Every job that ends adds one row to `<host_work>/gpu-log.md` (`docs/gpu-step.md`). A
stage run again with unchanged inputs gets the same job, so after the driver died while a job ran, `project run
<stage>` reattaches to the running container instead of starting another one. `hand_off(job_id, paths)` returns
`{path, sha256}` for model files the keeper user can read (regular files with `o+r`, directories with `o+x`, no link
on the way), the `files` of a model reference.

**The keeper's container.** `keeper serve --inference-image sha256:<64 hex> [--eval-tree DIR ...] [--docker-host
URL] [--docker PATH] [--nvidia-smi PATH]` runs a scenario pack's batch generator in a container of the keeper's own rootless daemon, one container per arm
for all registered seeds, with `--user 0:0 --cap-drop ALL --security-opt no-new-privileges --read-only --network none
--log-driver none` and Docker's default seccomp profile. Before any alpha is spent the keeper copies the model files a
reference lists into its own state and checks their hashes; the container gets only that copy, read-only at
`/in/model`, and the sealed inputs (never the labels) in a file. Its `/out` must then hold exactly one well-formed
`predictions.jsonl`; anything else is a crash, and the attempt has spent its alpha (AN-9). The report's `compute`
block states the keeper's containers and their seconds from the keeper's signed records, and the backend's job totals
the driver sends with the report request, marked "the driver's job records (cooperative, not checked by the
keeper)".

**Rootful access.** At every `hello`, intake, sealed summary, evaluate, gate and report request the keeper checks
whether the owner's account can reach a rootful Docker or containerd daemon: membership of `docker`, `lxd`, `incus-admin` or `disk` or of a rootful socket's group, a process of
an old session that still carries such a group, a rootful socket writable by everyone or carrying an access control
list, a listener on TCP port 2375 or 2376. A check it cannot complete counts as positive, and so does a process
table it cannot fully see (`/proc` mounted with `hidepid`, or no PID 1 and no process of another user). `keeper fingerprint` shows
`owner_has_root_docker` and the reasons. While the check is positive, and for good on every sealed set in use at a
positive check, a separate-user keeper states instead: "the keeper runs as a separate OS user, but the user's account
can reach a rootful Docker or containerd daemon or holds a root-equivalent group (docker, lxd, incus-admin or disk), or
did at a check while this sealed set was in use, and through that can read the keeper's state and signing key: the
records are signed, but nothing is sealed from the user's account; remove the user from those groups, end the sessions
that still carry them, and take in new data". `verify` exits 1 against a keeper whose `hello` says
`owner_has_root_docker`, and on a report that states any text but the separate-user guarantee.

**What is cooperative.** The job records, the GPU budget, the GPU log and the `ctx.backend` rules (a GPU job only from
a stage that declares `needs: gpu`, no job and no output directory from `plan`) hold for drivers and stages that use
the backend: anything running as the owner can start containers on the owner's daemon itself or edit the project's
records. The budget is enforced while a driver polls the job. `rootless = true` is the host file's word: the backend
checks the daemon it reaches, while `ext-launch` only prints. The keeper's own container time is the checked figure,
and the rootful-access check does not cover sudo.

## Scenario packs

A scenario pack is a distribution that registers its stages in the entry point group `gatedtrain.stages` (one class
per stage name; the five trust stage names are reserved) and one object in `gatedtrain.scenarios` with `name`,
`generators` (per-item generation functions, by the name a model reference gives in `generator`), `pipeline` (the
pipeline file), `scorers` (by reference) and `batch_generators` (`gatedtrain.ainative.batch.BatchGenerator`
declarations, by name). A name is either a per-item or a batch generator. A pack that has only batch generators still
exposes an empty `generators` mapping. A per-item generator runs inside the keeper process; a batch generator runs
only in the keeper's inference container (the in-process path and a keeper started without `--inference-image`
refuse it, AN-8). `gatedtrain.testing.synthetic_pack` is a small example of both (`synthetic_pack.yaml`,
`synthetic_pack_batch.yaml`); it is not registered as an entry point.

A `BatchGenerator` names its `script` (an absolute path inside the pack's installed package), the `interpreter` that
runs it (an absolute container path: the image's own, or one inside the keeper's eval trees with `eval_trees=True`),
`gpu` and `time_limit_s` (default 1200). The keeper runs `<interpreter> -I /gen/<script> --items /in/items.jsonl
--sampler /in/sampler.json --model /in/model --out /out`. The script reads one `{"item_id", "input"}` line per item and
`{"sampler", "seeds"}`, and writes only `/out/predictions.jsonl`: one line `{"item_id", "prediction", "seed"}` per item
and seed, each written as `json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
allow_nan=False)` followed by a newline. That is the keeper's canonical form only when `prediction` is a str, an int
within ±(2**53-1), a bool or null: a float, a larger int, a missing, extra or duplicate line, or any other file in
`/out` is a crash (AN-9).
