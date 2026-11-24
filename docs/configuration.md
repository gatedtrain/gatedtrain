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
