# Configuration

This page is for the operator of a host that runs external training runs or AI-native project jobs. It explains the
host file that tells gatedtrain where its work and output directories are, the environment variables that override
it, and the Docker settings gatedtrain assumes. The CPU demo needs none of this.

## The host file

gatedtrain has no built-in host path. The external-run commands read the host layout from a TOML host file,
`~/.config/gatedtrain/host.toml` or the file named by `GATEDTRAIN_HOST_CONFIG`:

```toml
host_work = "/srv/gatedtrain/work"        # holds bin/gpu-step.sh (the operator's GPU wrapper) and gpu-log.md
output_root = "/srv/gatedtrain/out"       # launches write strictly under this directory
```

`GATEDTRAIN_EXT_HOST_WORK`, `GATEDTRAIN_EXT_OUTPUT_ROOT` and `GATEDTRAIN_EXT_INPUT_ROOTS` override it.

**Rootless hosts.** A host whose containers run on the owner's rootless Docker daemon (the context `gatedtrain` pointing
at `unix:///run/user/<owner uid>/docker.sock`) says so in the host file. The launch rules then require the container
user `0:0` (the owner on the host), an output directory owned by the owner, and an owner whose primary group owns no
rootful Docker socket, and print `--device nvidia.com/gpu=all` in place of `--runtime=nvidia --gpus all`. The value is a
TOML boolean, set in the file only (no environment override); `true` enters `host_config_sha256` and the launch record
(`"rootless": true`), and a record written under the other setting is refused:

```toml
rootless = true
```

Rules: EXT-263 (rootless hosts; see the [test reference](../tests/README.md)).

## The owner image config

gatedtrain refers to container images by logical names (`nemo-rl/tierb`, ...), and the accepted image IDs are data. Two
files hold them: the image lock shipped with the release, which the maintainers write, and one file of your own,
`$XDG_CONFIG_HOME/gatedtrain/images.json` (`~/.config/gatedtrain/images.json` when the variable is unset), written by
`gatedtrain image add`. The environment names where your file is (see the limit below); it supplies no accepted ID.

- The file can only **add** IDs, each with origin `user-built` and a recipe (a commit or a Containerfile hash), to a
  logical image the shipped lock already names. It cannot add a logical image, or edit or remove a lock entry.
- It carries `schema` `gatedtrain.image-config` and `schema_version` 1. A different version, an unknown or missing key,
  a duplicate key, a malformed ID or a file over 256 KiB is refused, not read as best effort.
- It is read without following links and must be a regular file owned by the running user and not writable by group
  or others; a file that breaks this is refused (IMG-3), not ignored.
- The file cannot name an ID the shipped lock lists, under any logical image (IMG-2), so it can never re-label or
  re-attribute a shipped image. Every free-text field in it is one printable line.
- **It is not a security boundary against an agent that runs as your user.** Your environment (`HOME`,
  `XDG_CONFIG_HOME`) decides where the file is, and a same-user agent controls both: it can run `gatedtrain image add
  --yes` or write the file itself. The file records your decision, nothing more. What stays reliable is the keeper's
  flag in the signed report (`user_added`): the keeper checks the images in a report against its own copy of the
  maintainer lock, flags an image that is not in it, and signs the sha256 of that lock, which `gatedtrain verify`
  prints so you can compare it with a release. That holds when the keeper's code is outside the agent's reach (a
  separate-user keeper).
- `--built-utc` must be a real date; two `image add` runs at the same time keep both entries (an advisory lock on
  `images.json.lock` beside the file, which stays there); a file nested too deeply for JSON is refused (IMG-2).
- `image add` is **recorded for provenance only**: published packs read the shipped lock, so adding an image changes
  nothing for gating until packs adopt `accepted_images(logical)` in a later release.

Rules: IMG-1 to IMG-4 (see the [test reference](../tests/README.md)).

## The setup layout file

`gatedtrain setup host` and `gatedtrain setup keeper` take their paths from flags. Instead of a dozen flags the owner
can pass one TOML file, `--layout FILE`. It is **printed input**: gatedtrain reads it, prints the setup commands with
its values, and writes nothing; the keeper never reads it, and nothing finds it for you (there is no default location
and no environment variable; the owner names the file on every run).

`gatedtrain setup layout [--for host|keeper] [--mode same-user|separate-user]` prints a complete, commented, valid
layout with the effective defaults. `--for keeper` is the default and `--mode` defaults to `same-user`, as in `setup
keeper`.

**Precedence.** An explicit flag beats the layout file, which beats the built-in default; a flag that disagrees with
the file is not an error. The printed output starts with `# layout: <path as given> sha256:<hash of the bytes read>`
and a `# source of each value` block that lists every value that is not a default, with where it came from (`flag` or
`layout`). With `--json`, `data.layout` is `{path, sha256, keys}` where `keys` maps each key to `flag`, `layout` or
`default`. The unit text itself is unchanged: `ExecStart=` carries the paths.

**Schema.** `schema_version = 1` is required (exactly 1; any other value is refused, never read as an older or newer
layout). Unknown tables and keys, a wrong type, an invalid TOML file, a symlink or non-regular file, and a file over
64 KiB are refused (rule AN-34); every problem found in the document is reported at once. Values pass the same checks
as the flags (a path or value with a non-printable character is refused, and so is TOML nested too deeply to read). A table that does not apply to the command or mode is refused with the mode named: `[keeper]` is for
`setup host` and `setup keeper --mode separate-user`; `[same_user]` is for `setup keeper --mode same-user`; `[host]`
is for `setup host`. Defaults are the built-in ones of each command (`gatedtrain setup layout` shows them; the
same-user ones come from your XDG directories).

| Table | Key | Replaces | Applies to | Default |
|---|---|---|---|---|
| `[keeper]` | `user` | `--keeper-user` (host), `--user` | host, separate-user | the keeper's built-in system user |
| `[keeper]` | `state`, `socket`, `venv` | `--state`, `--socket`, `--venv` | host, separate-user | the built-in separate-user locations |
| `[same_user]` | `state`, `socket` | `--state`, `--socket` | same-user | under your XDG data and runtime directories |
| `[same_user]` | `executable` | `--executable` | same-user | the `gatedtrain` command next to the Python that printed the setup; only a path without a space, `%` or `$` can go into `ExecStart=` |
| `[host]` | `keeper_home`, `keeper_data_root`, `owner_data_root`, `handoff`, `eval_venv` | the flag of the same name | host | built-in |
| `[host]` | `eval_python_root` | none (layout only) | host | built-in; the root-owned tree the eval interpreter must be under |
| `[inference]` | `image` | `--inference-image` | host | none; a reference pinned by digest (`name@sha256:...`) |
| `[inference]` | `eval_requirements`, `eval_python` | the flag of the same name | host | none; `eval_requirements` is an absolute path |
| `[inference]` | `eval_trees` (list) | `--eval-tree` | separate-user, same-user | none |
| `[gpu]` | `wait_s`, `busy_mib` | `--gpu-wait-s`, `--gpu-busy-mib` | reserved | not supported by this version |
| `[inference]` | `keeper_eval_python` | `--eval-python` of `setup keeper` | reserved | not supported by this version |

A reserved key is refused with "not supported by this version (the keeper has no such option yet)". Flags that name
who runs the setup or a switch for one run (`--owner`, `--owner-uid`, `--packages`, `--inference-image` of `setup
keeper`, `--docker-host`, `--confirm-group-removal`, `--mode`) have no layout key. Rule: AN-34.

## Fixed defaults

Fixed defaults in `gatedtrain.ext.launch` that a host may need to know about:

- `DOCKER_CONTEXT = "gatedtrain"`: the printed sidecar commands use `docker --context gatedtrain`; create a Docker
  context of that name for the daemon that runs gatedtrain's containers.
- `DAEMON_CONFIGS`: `/etc/docker-gatedtrain/daemon.json` (a second daemon dedicated to gatedtrain, where a host runs
  one) and `/etc/docker/daemon.json`; only their `data-root` key is read, so that no engine data directory is ever
  mounted.
- `SOCKET_PATHS` / `DOCKER_SOCKET_GLOBS`: daemon sockets, including `/run/docker-gatedtrain.sock`; a container never
  runs in a socket's group.
- A host whose daemons keep their configuration or socket elsewhere lists them in the host file; they are refused
  like the built-in ones and enter the launch record's `host_config_sha256`:

  ```toml
  extra_daemon_configs = ["/etc/docker-other/daemon.json"]   # each one's data-root is never mounted
  extra_daemon_sockets = ["/run/docker-other.sock"]          # never mounted; a container never runs in its group
  ```
- `GPU_CAP_H = 1000`: the GPU-hour budget ceiling passed to `gpu-step.sh` (`GATEDTRAIN_GPU_CAP_H`), high enough not to
  limit runs; hours are still logged in `gpu-log.md` and every full run is preceded by a smoke.

## The GPU wrapper script

`bin/gpu-step.sh` is not part of gatedtrain, and the reference host's script is not published: an operator writes
their own. [gpu-step.md](gpu-step.md) gives the interface gatedtrain relies on: how the printed command invokes it, the
`WORK=` and `LOG=` lines that `ext-launch` and `ext-collect` check, the script's hash pin, and the formats of
`gpu-guard.log` and `gpu-log.md`.
