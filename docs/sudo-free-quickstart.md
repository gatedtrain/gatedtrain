# Quickstart without sudo

This page is for someone who wants to run an AI-native project on a Linux machine without running anything as root
for gatedtrain itself. It lists what must already be on the machine, checks it, sets up the keeper under your own user
(the default, same-user mode), creates a first project, shows how to read the mode line in the result, and says how to
harden the setup later. It assumes you have read what a gate and a sealed test set are in [Concepts](concepts.md).

**What you get, stated first.** In same-user mode the keeper runs as you. It follows the procedure (the registration is
signed before training, each sealed attempt spends its alpha before it is scored, a batch generator's model runs in a
container without network access that is given the inputs only; a per-item generator runs inside the keeper process,
without that confinement), but it does not seal anything from your account: anything running as you, such as an agent
with a shell, can read the sealed items, the labels, the alpha ledger and the keeper's signing key. The
reports say so, in the same words every time. For the guarantee that the sealed data stayed unseen you need the
separate-user mode (one sudo; see [Hardening](#hardening-separate-user-mode)). The two modes are compared in
[Keeper modes](ai-native.md#keeper-modes).

## Prerequisites

gatedtrain needs no sudo for this setup. These items are not gatedtrain's to install; on a machine that lacks one,
someone with administrator rights has to provide it once.

| Prerequisite | What it is for | Who provides it on a machine that lacks it |
|---|---|---|
| Rootless Docker for your user (the `docker` CLI and a daemon running as you, socket at `$XDG_RUNTIME_DIR/docker.sock` or named by `DOCKER_HOST`) | Jobs and the model under test run in containers | You can install it as yourself (`dockerd-rootless-setuptool.sh install`) once the next two rows are in place |
| `uidmap` (setuid `newuidmap` and `newgidmap`) and a subordinate id range for your user in `/etc/subuid` and `/etc/subgid` (at least 65536 ids) | Rootless Docker maps container ids into your user namespace | An administrator |
| Unprivileged user namespaces enabled (and, where the OS restricts them, a profile that lets rootlesskit use them) | Same | An administrator |
| Rootless networking (`slirp4netns` or `pasta`) | Container networking for a rootless daemon | An administrator |
| For GPU jobs only: an NVIDIA driver and an NVIDIA CDI specification naming `nvidia.com/gpu` in `/etc/cdi`, `/var/run/cdi` or `$XDG_CONFIG_HOME/cdi` | GPU access from a rootless container | An administrator |
| A systemd user session (`$XDG_RUNTIME_DIR` set) | The keeper runs as a user service. Without one, run `keeper serve` in a terminal | You |

On a managed or shared machine the first four rows may already be there; `gatedtrain setup check` tells you.

## Check the machine

```sh
gatedtrain setup check            # or: gatedtrain --json setup check; add --gpu or --no-gpu to override the GPU detection
```

The check reads the machine and changes nothing; it never runs sudo. It prints one row per prerequisite with a status
(`ok`, `missing`, `unknown` or `not_applicable`), whether it is required, who usually fixes it (`you` or
`administrator`) and what to do. Exit status 0 means every required item is ok; 1 means a required item is missing or
could not be determined (the rows are listed). The rows marked as informational (`userns_apparmor`, `systemd_user`,
`linger`, `memlock`, `ptrace_scope`) never fail the check. The last line says that gatedtrain itself needs no sudo for the same-user setup,
and that items marked "administrator" are Docker, OS or NVIDIA prerequisites that need root on a machine that lacks
them. A file or directory that exists but that you cannot read, and subordinate ids that `/etc/nsswitch.conf` also takes
from SSSD or LDAP when the local files are short, are `unknown`, not `missing`: the check cannot rule them out. The
AppArmor restriction on user namespaces (Ubuntu 24.04) is the informational row `userns_apparmor`: `unknown`, because
whether rootlesskit may run depends on a profile the check does not evaluate. A context named by `currentContext` in
the Docker config (`$DOCKER_CONFIG` or `~/.docker`) is noted like `DOCKER_CONTEXT`. Rule: AN-32, AN-33.

## Set up the keeper (same-user mode)

Install gatedtrain, and the scenario pack you will use, in the Python environment you run `gatedtrain` from: the
keeper runs from that environment.

```sh
gatedtrain setup keeper
```

This is the default (`--mode same-user`). It prints commands and runs nothing. The first line says what you are
getting: `# same-user mode: no sudo; run these as YOUR user, not with sudo. The keeper runs under your own OS user, so it does NOT seal data from your account
(an agent with your shell can read it). To harden: gatedtrain setup keeper --mode separate-user (one sudo).` The
commands create the state directory (`${XDG_DATA_HOME:-$HOME/.local/share}/gatedtrain-keeper`, mode 0700), write a
systemd user unit under `${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/`, and enable it with `systemctl --user`. The
socket is `$XDG_RUNTIME_DIR/gatedtrain-keeper/keeper.sock`. Read the commands, then run them yourself: the printed
block is one script that starts with `set -e` and stops at the first failing step (a state directory that already
exists but is shared or not yours stops it before the unit is written), so save it and run it with `sh FILE`, or paste
it whole. Leaving out `--mode` adds one warning, because the default was the separate-user output before 0.7.0. The
unit runs the `gatedtrain` command next to the Python that printed it (the PATH is not searched; the command is
refused when it is missing). When that path holds a space, `%` or `$`, make a link at a plain path and pass it with
`--executable PATH` (layout file: `[same_user] executable`). With no
systemd user session, the output ends with the `keeper serve` command to run in a terminal instead. To keep the keeper
running after you log out, `loginctl enable-linger` (printed as an optional comment) may ask an administrator. To
apply changed options to a unit that is already running, run the printed comment `systemctl --user restart
gatedtrain-keeper.service` (enabling a running unit does not restart it).

The last printed command shows the keeper's key fingerprint. Keep it somewhere the agent cannot write, such as a note
on another device: `gatedtrain verify <project> --fingerprint <fp>` checks a project against it later.

```sh
gatedtrain keeper fingerprint --socket "$XDG_RUNTIME_DIR/gatedtrain-keeper/keeper.sock"
```

The output includes the mode line and a `harden:` line.

**Batch generators need an inference image.** A scenario pack with a batch generator (the VLM pack) runs it in a
container of the keeper's own rootless Docker, in same-user mode as well. Run `gatedtrain setup keeper --inference-image
sha256:<64 hex>` (and see `gatedtrain setup check` for rootless Docker) to print a unit that starts the keeper with
it; a keeper without one refuses a batch generator (AN-8). Per-item packs need none.

**Customising paths.** To change the state directory or the socket without flags, put them in a layout file
(`gatedtrain setup layout > layout.toml`, edit, then `gatedtrain setup keeper --layout layout.toml`). The file is only
printed into the commands; see [the setup layout file](configuration.md#the-setup-layout-file).

## A first project

```sh
gatedtrain project create ./proj --goal "..." --data labelled.jsonl --pipeline pipeline.yaml \
    --keeper "$XDG_RUNTIME_DIR/gatedtrain-keeper/keeper.sock"
```

Before any data is taken in, `project create --keeper` prints the mode line of the keeper it is bound to:

```text
keeper mode: same-user (NOT sealed from the user's account: the procedure is followed, the sealed data is not protected from anything running as the user; to harden: gatedtrain setup keeper --mode separate-user)
```

Then run the stages (`gatedtrain project run ./proj <stage>`; see [AI-native projects](ai-native.md#how-a-project-runs)).
The `intake` stage prints the mode line again when the keeper takes in the sealed data, and `keeper confirm` prints it
before the metric sentence you confirm. `keeper confirm` needs a terminal or `--yes`, and is a step for you, not for an
agent.

## Reading the result

The report (`gatedtrain project report ./proj`, and `report.md` in the project) begins with the mode line, and in
same-user mode the headline says what the PASS covers, for example `PASS (same-user keeper: procedure followed; sealed
data not protected from the user's account)`. Quote that line in any summary. `report.json` carries `keeper_mode`, the
modes the sealed set was ever served in (`modes_seen`), and each attempt's mode.

`gatedtrain verify ./proj --keeper "$XDG_RUNTIME_DIR/gatedtrain-keeper/keeper.sock"` checks the signed records. For a
same-user keeper it exits 1 with a warning that names the mode, because the owner can read that keeper's key: the
records are consistent, but they prove nothing about the sealed data. If you have decided that a same-user result is
enough for your purpose, say so per call:

```sh
gatedtrain verify ./proj --keeper "$XDG_RUNTIME_DIR/gatedtrain-keeper/keeper.sock" --accept-mode same-user
```

This exits 0 and ends with `ACCEPTED WEAKER MODE: same-user (procedure followed; sealed data not protected from the
user's account)`. A project created with `--require-keeper-mode separate-user` refuses it, whatever the option says.

## Hardening: separate-user mode

When an agent with your shell is part of the picture and the sealed data must stay unseen, run the keeper under its own
OS user:

```sh
gatedtrain setup keeper --mode separate-user      # prints the sudo commands; read them, then run them once
gatedtrain setup host                             # the rootless Docker host script for you and the keeper; see AI-native projects
```

Hardening means a new keeper (its own state directory, socket and key) and new projects bound to it with new data: the
sealed items of a same-user keeper were readable by the agent and must be treated as exposed, and no state or key moves
between keepers. Old projects keep verifying with `--fingerprint`, reported as same-user. Add
`--require-keeper-mode separate-user` to `project create` to make a project refuse any weaker keeper. The details and
the migration rules are in [Keeper modes](ai-native.md#keeper-modes).

If you run NeMo RL jobs on rootless Docker, read [NeMo RL on a rootless host](rootless-nemo-rl.md) before the first GPU run.
