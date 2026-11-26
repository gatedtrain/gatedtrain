"""Validate and record a container launch.

`ext-launch` checks a request against the host and adapter configuration, writes
the canonical launch record, and prints the command for the operator. The
operator's `gpu-step.sh` wrapper runs the trainer and logs GPU time. Some
adapters also need a sidecar, so the printed commands start it before the
trainer and stop it afterward.

The launch record binds the resolved mounts, interpreter, training script and
effective host configuration. It stores environment names only. The command
label includes a checksum of the record so the collector can match it to the
wrapper's log. Identical requests at the same GPU total share a label, and
their attempts are counted together. The checksum detects accidental changes.

`--recheck` and `ext-collect` validate the record against the current host.
The wrapper interface is documented in `docs/gpu-step.md`. See
`MULTIMODAL-SPEC.md` MM-31 to MM-39 and `tests/README.md` for the record format
and validation rules.
"""

from __future__ import annotations

import datetime as _dt
import fnmatch
import json
import math
import os
import posixpath
import re
import shlex
import stat
import unicodedata
import dataclasses
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

import yaml

from .. import TAGS
from ..canonical import canonical_json, file_sha256, plain_decimal, sha256_bytes, text_lines
from ..errors import Problem, Refused
from ..paths import realpath as _realpath
from ..renamed import old_schema_note

RECORD_SCHEMA = "gatedtrain.launch_record"
SCHEMA_VERSION = "2.0"       # EXT-95: 1.0 records predate the interpreter binding
PROFILES = ("smoke", "full")
# The GPU-hour budget ceiling passed to the operator's gpu-step.sh wrapper (`GATEDTRAIN_GPU_CAP_H`). 1000 is high enough
# not to limit runs: GPU hours are still logged in gpu-log.md, and every full run is preceded by a smoke.
GPU_CAP_H = 1000
RUNTIME_KINDS = ("container", "venv_in_container")
# Docker reference grammar (distribution/reference, reference.go): an optional domain (alphanumeric first, an
# optional port), path components `[a-z0-9]+(?:(?:[._]|__|[-]*)[a-z0-9]+)*`, an optional tag, then the digest.
# Nothing in it can start with `-`, so docker never reads the image as an option (EXT-39).
_DOMAIN = r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)*(?::[0-9]+)?/)?"
_COMPONENT = r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*"
_TAG = r"(?::[A-Za-z0-9_][A-Za-z0-9_.-]{0,127})?"
# The grammar patterns are used with `re.fullmatch` (EXT-54): `$` would accept a trailing newline, `\d` any Unicode
# digit. CREDENTIAL_RE is searched.
IMAGE_RE = re.compile(rf"sha256:[0-9a-f]{{64}}|{_DOMAIN}{_COMPONENT}(?:/{_COMPONENT})*{_TAG}@sha256:[0-9a-f]{{64}}")
SHA_RE = re.compile(r"sha256:[0-9a-f]{64}")
USER_RE = re.compile(r"([0-9]+):([0-9]+)")
# Match whole underscore-delimited components or a TOKEN/KEY suffix.
# This catches HFTOKEN without blocking TOKENIZERS_PARALLELISM (EXT-55, EXT-71).
CREDENTIAL_RE = re.compile(r"(?:^|_)(?:TOKENS?|SECRETS?|PASSWORDS?|PASSWD|KEYS?|CREDENTIALS?|APIKEYS?|PATS?|AUTH|"
                           r"PASSPHRASES?)(?:_|\Z)|(?:TOKENS?|KEYS?)\Z")
# Keep UIDs and GIDs in the signed 32-bit range for Docker compatibility (EXT-74).
ID_MAX = 2**31 - 1
LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}")
LABEL_DIGEST_HEX = 12
HASHED_LABEL_RE = re.compile(rf"(?P<base>[A-Za-z0-9][A-Za-z0-9_.-]{{0,99}})-(?P<digest>[0-9a-f]{{{LABEL_DIGEST_HEX}}})")
ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
NETWORK_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")       # Docker object names
HUB_OFFLINE = "HF_HUB_OFFLINE"
# EXT-40: Docker daemon sockets on this host and the directories that hold daemon sockets.
# A host adds its own with the host file's `extra_daemon_sockets`.
SOCKET_PATHS = ("/var/run/docker.sock", "/run/docker.sock", "/run/docker-gatedtrain.sock")
SOCKET_DIRS = ("/run", "/var/run")
# EXT-51: kernel file systems whose entries are views of other processes or of the host (`/proc/1/root` is the
# host's `/` for dockerd); a host path under them is never mounted.
KERNEL_DIRS = ("/proc", "/sys", "/dev")
# EXT-62: the container engines' data directories (images, container file systems, volumes).
ENGINE_DATA_DIRS = ("/var/lib/docker", "/var/lib/containerd")
# EXT-73: daemon configurations whose `data-root` key (only that key is read) names another engine data directory
# (`/etc/docker-gatedtrain/daemon.json`: a second daemon's data directory on a host that runs one for gatedtrain).
# A host whose daemons keep their configuration elsewhere names them in the host file's `extra_daemon_configs`.
DAEMON_CONFIGS = ("/etc/docker-gatedtrain/daemon.json", "/etc/docker/daemon.json")
# EXT-66: system directories that hold credentials or logs; a mount source is never one of them, inside one, or
# an ancestor of one.
SECRET_DIRS = ("/etc", "/root", "/var/log")
HOME_ROOT = "/home"
# EXT-63, EXT-66, EXT-67: the host layout (MULTIMODAL-SPEC MM-34, MM-36), read from the host file: TOML with
# `host_work` (the directory holding bin/gpu-step.sh and gpu-log.md) and `output_root` (where launches write), both
# absolute paths. No built-in value: a machine path in the code would bind every host to one machine's layout.
ENV_HOST_CONFIG = "GATEDTRAIN_HOST_CONFIG"
DEFAULT_HOST_CONFIG = "~/.config/gatedtrain/host.toml"
HOST_FILE_KEYS = ("host_work", "output_root")
# Optional host file lists of absolute paths: more daemon configurations whose `data-root` is refused (EXT-73) and
# more daemon sockets that are never mounted and whose group a container never runs in (EXT-40, EXT-51).
HOST_FILE_LIST_KEYS = ("extra_daemon_configs", "extra_daemon_sockets")
# Optional host file booleans: `rootless` (EXT-263), the launch rules of a host whose containers run on the owner's
# rootless Docker daemon. No environment variable overrides it.
HOST_FILE_BOOL_KEYS = ("rootless",)
# Optional host file numbers, above zero and finite, owner-side: the smoke GPU-memory bound in GB (10^9 bytes) and the
# GPU-hour ceiling passed to gpu-step.sh. Unset, the defaults below apply. No environment variable sets them; a value
# enters `host_config_sha256` (so the launch record binds it) only when the file sets it.
HOST_FILE_NUMBER_KEYS = ("gpu_mem_limit_gb", "gpu_cap_h")
GPU_MEM_LIMIT_GB = 22.5          # the smoke bound for a 24 GB card (EXT-29); the host file's `gpu_mem_limit_gb` replaces it
# The largest value the host file may give each number (trust constants, fixed; owner decision to widen). Without a top,
# `1e308` would switch the smoke check off while still passing validation. 1024 GB: a terabyte is about 3.5 times the
# memory of the largest single accelerator of 2026 (288 GB), so no real card needs more and a bound above it checks
# nothing. 100000 GPU hours: more than one 8-GPU node can burn in a year (8 x 8760 h = 70080 h), so a ceiling above it
# is no ceiling for a node-scale host. `HOST_FILE_NUMBER_MAX` is the one table the validation reads.
MAX_GPU_MEM_LIMIT_GB = 1024
MAX_GPU_CAP_H = 100000
HOST_FILE_NUMBER_MAX = {"gpu_mem_limit_gb": MAX_GPU_MEM_LIMIT_GB, "gpu_cap_h": MAX_GPU_CAP_H}
# Optional host file paths and path lists that used to be settable only by `GATEDTRAIN_EXT_*` variables: the layout the
# launch rules check against (EXT-79, EXT-112, EXT-113).
HOST_FILE_PATH_KEYS = ("interpreter", "mirror_root")
HOST_FILE_PATHLIST_KEYS = ("input_roots", "interpreter_trees")
# Optional host file table `[packs.<pack name>]`: the settings a scenario pack asks the owner for, each an absolute path
# (for example `runtime_dir`). A pack names its own settings; the host file only validates their shape.
HOST_FILE_PACKS_KEY = "packs"
_PACK_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_PACK_KEY = re.compile(r"[a-z][a-z0-9_]*")
OUTPUT_ROOT_TOKEN = "{output_root}"      # a registry row's prefix for a directory under the output root
# The adapter supplies input roots unless GATEDTRAIN_EXT_INPUT_ROOTS is set (EXT-79).
ENV_PREFIX = "GATEDTRAIN_EXT_"
ENV_HOST_WORK = "GATEDTRAIN_EXT_HOST_WORK"
ENV_OUTPUT_ROOT = "GATEDTRAIN_EXT_OUTPUT_ROOT"
ENV_INPUT_ROOTS = "GATEDTRAIN_EXT_INPUT_ROOTS"         # os.pathsep-separated
# Tests can substitute the root for identity mounts and the pinned interpreter paths (EXT-112, EXT-113).
ENV_MIRROR_ROOT = "GATEDTRAIN_EXT_MIRROR_ROOT"
ENV_INTERPRETER = "GATEDTRAIN_EXT_INTERPRETER"
ENV_INTERPRETER_TREES = "GATEDTRAIN_EXT_INTERPRETER_TREES"   # os.pathsep-separated
# The variables above are a temporary workaround, not the design: each has a host file key (below), and a variable that
# is set prints a deprecation line naming it. They go away in `ENV_REMOVED_IN`; set the file key instead.
ENV_REMOVED_IN = "0.9.0"
ENV_TO_FILE_KEY = {ENV_HOST_WORK: "host_work", ENV_OUTPUT_ROOT: "output_root", ENV_INPUT_ROOTS: "input_roots",
                   ENV_INTERPRETER_TREES: "interpreter_trees", ENV_INTERPRETER: "interpreter",
                   ENV_MIRROR_ROOT: "mirror_root"}
# Protect system directories from mounts that would shadow the image's files.
# A read-only identity mount under a configured input root can preserve its host path (EXT-112).
SYSTEM_CONTAINER_DIRS = ("/etc", "/usr", "/bin", "/sbin", "/opt", "/proc", "/sys", "/dev", "/run", "/var", "/root",
                         "/home")
# Keep homes and caches under writable paths to avoid loading cached objects from inputs (EXT-114).
ENV_PATH_ROOTS = ("/tmp", "/out")
GPU_STEP = ("bin", "gpu-step.sh")
GPU_LOG_MD = "gpu-log.md"
# EXT-51: the daemon sockets whose group may not be the container's gid (patterns, `glob` syntax).
DOCKER_SOCKET_GLOBS = ("/run/docker.sock", "/var/run/docker.sock", "/run/docker-*.sock")
# Fixed docker options of every printed command (EXT-24; the TRL recipe's README
# "Command shape"; docs/tier-b-runbook.md section 8). The record keeps tmpfs paths only (EXT-39); the options
# are code, never request values.
GPU_RUNTIME = "--runtime=nvidia"      # with --gpus: the gatedtrain daemon's NVIDIA toolkit in CDI mode refuses --gpus alone
ROOTFUL_GPU_OPTIONS = (GPU_RUNTIME, "--gpus", "all")
# EXT-263: a rootless daemon gets the GPU through Docker's native CDI, and its container root is the owner on the host.
CDI_GPU_OPTIONS = ("--device", "nvidia.com/gpu=all")
ROOTLESS_USER = "0:0"
# The owner's runtime directories; a rootless daemon's socket is `<RUN_USER_DIR>/<uid>/docker.sock` (EXT-263).
RUN_USER_DIR = "/run/user"
ROOTLESS_SOCKET = "docker.sock"
SHM_SIZE_VALUE = "8g"               # the shared memory of a container; must stay at or below host RAM
SHM_SIZE = f"--shm-size={SHM_SIZE_VALUE}"
TMPFS_OPTIONS = "rw,exec,nosuid,nodev,size=8g"   # MM-34: Triton and torch.compile load compiled objects from /tmp
                                                 # (Docker's default is noexec); TRL recipe README
# Scan read-only inputs by NFKC-normalized, case-folded name (EXT-80, EXT-93).
# The adapter supplies scan exclusions. The stdlib secrets module is exempt in declared interpreter trees.
CREDENTIAL_NAME_GLOBS = (".env", ".env.*", "*credential*", "*secret*", "*.pem", "*.key", "id_rsa*", "id_ed25519*",
                         "id_ecdsa*", "id_dsa*", "token", "stored_tokens", ".netrc", ".git-credentials", ".npmrc",
                         ".pypirc", "*.p12", "*.pfx", ".s3cfg", ".boto")
CREDENTIAL_DIR_NAMES = ((".docker", "config.json"), (".kube", "config"), ("gh", "hosts.yml"))   # (directory, name)
SCAN_LIMIT = 200_000
# Bounds for resolving the pinned Python interpreter (EXT-91).
INTERPRETER_RE = re.compile(r"python(?:3(?:\.[0-9]+)?)?")
LINK_HOPS = 40
_STDLIB_DIR_RE = re.compile(r"python3\.[0-9]+")
_STDLIB_SECRETS_PYC_RE = re.compile(r"secrets\.cpython-[0-9]+(?:\.opt-[12])?\.pyc")

REQUEST_KEYS = ("area", "label", "host_work", "runtime", "framework", "training_seed", "gpus", "network",
                "network_reason", "user", "env", "mounts", "tmpfs", "output_dir", "gpu_est_h", "argv")
# Legacy requests may still include these keys with null values (EXT-97).
RETIRED_REQUEST_KEYS = ("network_host_reason",)
RECORD_KEYS = ("schema", "schema_version", "tags", "adapter", "area", "profile", "label", "host_work", "runtime",
               "framework", "training_seed", "gpus", "argv", "mounts", "tmpfs", "network", "network_reason",
               "user", "env_names", "gpu_est_h", "gpu_used_h", "output_dir",
               "output_root", "gpu_step_sha256",                                   # EXT-63, EXT-67
               "allow_network", "script_path", "script_sha256", "host_config_sha256",  # EXT-81, EXT-83, EXT-85
               "interpreter_path", "interpreter_sha256")                           # EXT-91
# Paired sidecars are available to adapters that declare them in the registry (EXT-190).
OPTIONAL_REQUEST_KEYS = ("sidecar",)
OPTIONAL_RECORD_KEYS = ("sidecar", "rootless")      # EXT-263: `"rootless": true` only on a rootless host
SIDECAR_REQUEST_KEYS = ("runtime", "argv", "mounts", "env", "network", "user", "tmpfs", "read_only", "cap_drop",
                        "security_opt", "gpus", "limits")
SIDECAR_RECORD_KEYS = ("runtime", "argv", "mounts", "env_names", "network", "user", "tmpfs", "read_only",
                       "cap_drop", "security_opt", "gpus", "limits", "interpreter_sha256", "script_sha256")
# The sidecar's isolation flags, exactly these values (executor/run-executor.sh COMMON); the printed command takes them
# from code, never from the request.
SIDECAR_FLAGS = {"read_only": True, "cap_drop": ["ALL"], "security_opt": ["no-new-privileges"], "gpus": False}
SIDECAR_LIMIT_KEYS = ("cpus", "memory", "memory_swap", "pids", "nofile", "core")
DOCKER_CONTEXT = "gatedtrain"                                   # MM-31: the gatedtrain daemon (`docker --context gatedtrain`)


@dataclass(frozen=True)
class HostConfig:
    """The host layout the rules check against (EXT-63, EXT-66, EXT-67). `None` fields are read on this host:
    home directories (`host_home_dirs`), engine data directories (`engine_data_dirs`), a uid's primary group
    (`passwd_gid`). `input_roots` None means the adapter's (`adapter_config`, EXT-79); without an adapter, none.
    The same holds for the environment allow-list `env_names` (EXT-92), the scan's `scan_skip` directories and the
    `interpreter_trees` whose stdlib `secrets` module is exempt (EXT-93): None is the adapter's, else none."""
    host_work: str = ""                  # "" = not configured (host_config refuses, EXT-HOST)
    output_root: str = ""
    input_roots: tuple[str, ...] | None = None
    home_dirs: tuple[str, ...] | None = None
    engine_dirs: tuple[str, ...] | None = None
    primary_gid: Callable[[int], int | None] | None = None
    env_names: tuple[str, ...] | None = None
    scan_skip: tuple[str, ...] | None = None
    interpreter_trees: tuple[str, ...] | None = None
    # None defers to the adapter for mount paths, interpreter and path-valued environment names.
    # mirror_root defaults to / on the host for identity mounts (EXT-112 to EXT-114).
    container_paths: Mapping[str, Any] | None = None
    interpreter: str | None = None
    env_paths: tuple[str, ...] | None = None
    mirror_root: str | None = None
    # Per-script image interpreters and sidecar settings default to the adapter's registry row (EXT-190, EXT-192).
    image_interpreters: Mapping[str, str] | None = None
    sidecar: Mapping[str, Any] | None = None
    # EXT-199: script file name -> the interpreter option allowed before it (`-I`), and the executor-image layout.
    interpreter_flags: Mapping[str, str] | None = None
    b18: Mapping[str, Any] | None = None
    # The host file's `extra_daemon_configs` and `extra_daemon_sockets`, beside DAEMON_CONFIGS and SOCKET_PATHS.
    extra_daemon_configs: tuple[str, ...] = ()
    extra_daemon_sockets: tuple[str, ...] = ()
    # EXT-263: the host file's `rootless`; the owner whose rootless daemon runs the containers (None: this process's uid).
    rootless: bool = False
    owner_uid: int | None = None
    # The host file's owner-side numbers (None: the defaults GPU_MEM_LIMIT_GB and GPU_CAP_H) and pack settings.
    gpu_mem_limit_gb: float | None = None
    gpu_cap_h: float | None = None
    packs: Mapping[str, Mapping[str, str]] | None = None


def gpu_mem_limit_gb(cfg: HostConfig) -> float:
    """The smoke GPU-memory bound of this host: its file's `gpu_mem_limit_gb`, else `GPU_MEM_LIMIT_GB`."""
    return GPU_MEM_LIMIT_GB if cfg.gpu_mem_limit_gb is None else cfg.gpu_mem_limit_gb


def gpu_cap_h(cfg: HostConfig) -> float:
    """The GPU-hour ceiling of this host: its file's `gpu_cap_h`, else `GPU_CAP_H`."""
    return GPU_CAP_H if cfg.gpu_cap_h is None else cfg.gpu_cap_h


def host_file(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The host file's `host_work` and `output_root`, the optional lists `extra_daemon_configs` and
    `extra_daemon_sockets`, and the optional boolean `rootless` ({} when the file does not exist); Refused when it does
    not parse, has other keys, a layout value is not an absolute path, a list key is not a list of absolute paths, or
    `rootless` is not a TOML boolean (EXT-263)."""
    import tomllib

    env = os.environ if environ is None else environ
    path = Path(os.path.expanduser(env.get(ENV_HOST_CONFIG) or DEFAULT_HOST_CONFIG))
    fix = f"write {path} as TOML with absolute paths `host_work` and `output_root`"
    if not path.exists():
        return {}
    try:
        doc = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise Refused([Problem("EXT-HOST", str(path), f"does not parse ({exc.__class__.__name__})", fix)])
    known = (HOST_FILE_KEYS + HOST_FILE_LIST_KEYS + HOST_FILE_BOOL_KEYS + HOST_FILE_NUMBER_KEYS + HOST_FILE_PATH_KEYS
             + HOST_FILE_PATHLIST_KEYS + (HOST_FILE_PACKS_KEY,))
    bad = sorted(set(doc) - set(known))
    if bad:
        raise Refused([Problem("EXT-HOST", str(path), f"has keys {bad} besides {list(known)}", fix)])
    for k, v in doc.items():
        if k in HOST_FILE_NUMBER_KEYS:
            if not _positive_number(v, HOST_FILE_NUMBER_MAX[k]):
                top = HOST_FILE_NUMBER_MAX[k]
                raise Refused([Problem("EXT-HOST", f"{path}.{k}", f"{v!r} is not a number above 0 and at most {top:g}",
                                       f"write `{k}` in {path} as a TOML number above zero and at most {top:g}, for "
                                       f"example `{k} = 22.5`")])
        elif k == HOST_FILE_PACKS_KEY:
            _check_packs(v, path)
        elif k in HOST_FILE_LIST_KEYS or k in HOST_FILE_PATHLIST_KEYS:
            if not (isinstance(v, list) and all(isinstance(x, str) and PurePosixPath(x).is_absolute() for x in v)):
                raise Refused([Problem("EXT-HOST", f"{path}.{k}", f"{v!r} is not a list of absolute paths",
                                       f"write `{k}` in {path} as a TOML array of absolute paths")])
        elif k in HOST_FILE_BOOL_KEYS:
            if not isinstance(v, bool):
                raise Refused([Problem("EXT-HOST", f"{path}.{k}", f"{v!r} is not true or false (EXT-263)",
                                       f"write `{k} = true` or `{k} = false` in {path} (a TOML boolean, unquoted)")])
        elif not (isinstance(v, str) and PurePosixPath(v).is_absolute()):
            raise Refused([Problem("EXT-HOST", f"{path}.{k}", f"{v!r} is not an absolute path", fix)])
    return dict(doc)


def _positive_number(v: Any, top: float) -> bool:
    """A finite number above 0 and at most `top` (not a bool; an integer too large for a float is not finite enough)."""
    try:
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0 < v <= top
    except OverflowError:
        return False


def _check_packs(value: Any, path: Path) -> None:
    """The `[packs.<name>]` tables: names and keys of plain words, values absolute paths; Refused (EXT-HOST) otherwise."""
    fix = f"write `[packs.<pack name>]` tables in {path} whose values are absolute paths"
    if not isinstance(value, dict):
        raise Refused([Problem("EXT-HOST", f"{path}.packs", f"{value!r} is not a table of pack tables", fix)])
    for name, table in value.items():
        if not (_PACK_NAME.fullmatch(name) and isinstance(table, dict)):
            raise Refused([Problem("EXT-HOST", f"{path}.packs.{name}", "is not a table named by a plain pack name", fix)])
        for key, v in table.items():
            if not (_PACK_KEY.fullmatch(key) and isinstance(v, str) and PurePosixPath(v).is_absolute()):
                raise Refused([Problem("EXT-HOST", f"{path}.packs.{name}.{key}",
                                       f"{v!r} is not an absolute path under a plain key", fix)])


def owner_gpu_mem_limit_gb(environ: Mapping[str, str] | None = None) -> float:
    """The smoke GPU-memory bound the owner's host file sets (`gpu_mem_limit_gb`), else `GPU_MEM_LIMIT_GB`; it needs no
    host layout, so a stage can ask for it where no launch is configured. A scenario pack takes this as the ceiling of
    any bound a pipeline asks for. Refused (EXT-HOST) when the file is malformed."""
    value = host_file(environ).get("gpu_mem_limit_gb")
    return GPU_MEM_LIMIT_GB if value is None else value


def owner_gpu_bounds(environ: Mapping[str, str] | None = None) -> dict[str, float]:
    """The smoke bounds the owner's host file sets, as `smoke_problems` keywords (`gpu_mem_limit_gb`, `gpu_cap_h`; each
    the default when the file does not set it). They need no host layout. Refused (EXT-HOST) when the file is malformed:
    a bad file is never read as the defaults."""
    doc = host_file(environ)
    return {"gpu_mem_limit_gb": doc.get("gpu_mem_limit_gb", GPU_MEM_LIMIT_GB), "gpu_cap_h": doc.get("gpu_cap_h", GPU_CAP_H)}


def pack_setting(pack: str, key: str, environ: Mapping[str, str] | None = None) -> str | None:
    """The owner's setting `key` of the scenario pack `pack` from the host file's `[packs.<pack>]` table (an absolute
    path), or None when the file, the table or the key is missing. Refused (EXT-HOST) when the file is malformed."""
    table = (host_file(environ).get(HOST_FILE_PACKS_KEY) or {}).get(pack) or {}
    return table.get(key)


def env_conflict(var: str, key: str, from_env: Any, from_file: Any, where: str) -> Refused:
    """The refusal (EXT-HOST) for a deprecated variable and the owner's host file key that name different values: a
    variable set by whoever controls the process environment must not silently outrank what the owner wrote."""
    return Refused([Problem("EXT-HOST", key, f"{var} is {from_env!r} but the host file {where} sets `{key}` to "
                            f"{from_file!r}; the two disagree and neither outranks the other",
                            f"remove the variable {var} or the key `{key}` from the host file {where}")])


def env_or_file(var: str, key: str, from_env: Any, from_file: Any, where: str) -> Any:
    """The value of the setting `key`: `from_env` (the deprecated variable `var`) or `from_file`; both set and unequal
    is Refused (`env_conflict`), so the host file's key is never outranked silently. An unset side is falsy."""
    if from_env and from_file and from_env != from_file:
        raise env_conflict(var, key, from_env, from_file, where)
    return from_env or from_file


def _env_or_file(env: Mapping[str, str], var: str, file: Mapping[str, Any], from_env: Any) -> Any:
    key = ENV_TO_FILE_KEY[var]
    from_file = tuple(file.get(key, ())) if key in HOST_FILE_PATHLIST_KEYS else file.get(key)
    where = os.path.expanduser(env.get(ENV_HOST_CONFIG) or DEFAULT_HOST_CONFIG)
    value = env_or_file(var, key, from_env, from_file, where)
    return value if key in HOST_FILE_PATHLIST_KEYS else (value or None)


def host_config(environ: Mapping[str, str] | None = None) -> HostConfig:
    """The host file's layout, or the `GATEDTRAIN_EXT_HOST_WORK`, `GATEDTRAIN_EXT_OUTPUT_ROOT` and
    `GATEDTRAIN_EXT_INPUT_ROOTS` overrides (tests; another host). Without the input-roots override the adapter's roots
    apply (EXT-79). Refused (EXT-HOST) when host_work or the output root is configured nowhere. The host file's
    daemon lists apply under the overrides too: they only add refusals. `rootless` comes from the host file only
    (EXT-263)."""
    env = os.environ if environ is None else environ
    file = host_file(env)
    roots = _env_or_file(env, ENV_INPUT_ROOTS, file, tuple(r for r in env.get(ENV_INPUT_ROOTS, "").split(os.pathsep) if r))
    trees = _env_or_file(env, ENV_INTERPRETER_TREES, file,
                         tuple(r for r in env.get(ENV_INTERPRETER_TREES, "").split(os.pathsep) if r))   # EXT-113
    layout = {"host_work": env.get(ENV_HOST_WORK) or file.get("host_work"),
              "output_root": env.get(ENV_OUTPUT_ROOT) or file.get("output_root")}
    missing = [k for k, v in layout.items() if not v]
    if missing:
        where = os.path.expanduser(env.get(ENV_HOST_CONFIG) or DEFAULT_HOST_CONFIG)
        raise Refused([Problem("EXT-HOST", k, f"the host layout names no {k}",
                               f"set `{k}` in {where} (TOML, an absolute path) or {ENV_PREFIX}{k.upper()}")
                       for k in missing])
    return HostConfig(host_work=layout["host_work"], output_root=layout["output_root"],
                      input_roots=roots or None, interpreter_trees=trees or None,
                      interpreter=_env_or_file(env, ENV_INTERPRETER, file, env.get(ENV_INTERPRETER) or None),
                      mirror_root=_env_or_file(env, ENV_MIRROR_ROOT, file, env.get(ENV_MIRROR_ROOT) or None),
                      extra_daemon_configs=tuple(file.get("extra_daemon_configs", ())),
                      extra_daemon_sockets=tuple(file.get("extra_daemon_sockets", ())),
                      rootless=file.get("rootless", False), gpu_mem_limit_gb=file.get("gpu_mem_limit_gb"),
                      gpu_cap_h=file.get("gpu_cap_h"), packs=file.get(HOST_FILE_PACKS_KEY))


def adapter_config(cfg: HostConfig, adapter: str) -> HostConfig:
    """EXT-79: `cfg` with the adapter's registry input roots when it names none (no override); EXT-92, EXT-93:
    likewise the environment allow-list, the scan skip list and the interpreter trees, each filled on its own (the
    input-roots override keeps them). Unchanged for an unknown adapter (refused elsewhere)."""
    from . import registry

    try:
        row = {"input_roots": registry.input_roots(adapter), "env_names": registry.env_names(adapter),
               "scan_skip": registry.scan_skip(adapter), "interpreter_trees": registry.interpreter_trees(adapter),
               "container_paths": registry.container_paths(adapter), "interpreter": registry.interpreter(adapter),
               "env_paths": registry.env_paths(adapter),                                  # EXT-112 to EXT-114
               "image_interpreters": registry.image_interpreters(adapter),                 # EXT-192
               "sidecar": registry.sidecar(adapter),                                      # EXT-190
               "interpreter_flags": registry.interpreter_flags(adapter), "b18": registry.b18(adapter)}   # EXT-199
    except KeyError:
        return cfg
    # A row may name a directory under the host's output root as `{output_root}/<dir>` (a recipe's earlier runs, read
    # by a later phase); it is filled in from this layout, so the row names no host path.
    for k in ("input_roots", "scan_skip"):
        row[k] = tuple(p.replace(OUTPUT_ROOT_TOKEN, cfg.output_root) if p.startswith(OUTPUT_ROOT_TOKEN) else p
                       for p in row[k])
    return dataclasses.replace(cfg, **{k: v for k, v in row.items() if getattr(cfg, k) is None})


def host_config_sha256(cfg: HostConfig) -> str:
    """EXT-85: SHA-256 of the canonical JSON {host_work, output_root, input_roots, mirror_root, interpreter,
    interpreter_trees} of the effective layout. Resolve adapter defaults before hashing so changes to overrides
    can invalidate a recorded layout (EXT-117). An
    adapter with image interpreters adds `image_interpreters`; the other adapters' hashes are unchanged. Likewise a
    host file with `extra_daemon_configs` or `extra_daemon_sockets` adds that list, and `rootless = true` adds
    `"rootless": true` (EXT-263; a rootful layout's hash is unchanged)."""
    doc = {"host_work": cfg.host_work, "output_root": cfg.output_root, "input_roots": list(cfg.input_roots or ()),
           "mirror_root": cfg.mirror_root, "interpreter": cfg.interpreter,
           "interpreter_trees": list(cfg.interpreter_trees or ())}
    if cfg.image_interpreters:
        doc["image_interpreters"] = dict(cfg.image_interpreters)
    for k in HOST_FILE_LIST_KEYS:
        if getattr(cfg, k):
            doc[k] = list(getattr(cfg, k))
    if cfg.rootless:
        doc["rootless"] = True
    for k in HOST_FILE_NUMBER_KEYS:                       # owner-side numbers: bound only when the file sets them
        if getattr(cfg, k) is not None:
            doc[k] = getattr(cfg, k)
    if cfg.packs:
        doc["packs"] = {name: dict(table) for name, table in sorted(cfg.packs.items())}
    return sha256_bytes(canonical_json(doc))


def override_warnings(environ: Mapping[str, str] | None = None) -> list[str]:
    """EXT-85: one stderr line per `GATEDTRAIN_EXT_*` variable set: each loosens or moves the host layout the
    rules check against."""
    env = os.environ if environ is None else environ
    def line(name: str) -> str:
        text = (f"warning: {name} overrides the host layout ({env[name]}); the launch record's host_config_sha256 "
                "binds the layout it was written under (EXT-85)")
        if name in ENV_TO_FILE_KEY:
            text += (f". The variable is deprecated: set `{ENV_TO_FILE_KEY[name]}` in the host file instead; it is "
                     f"removed in {ENV_REMOVED_IN}")
        return text

    return [line(name) for name in sorted(env) if name.startswith(ENV_PREFIX)]


def passwd_gid(uid: int) -> int | None:
    """The primary group of `uid` in this host's user database, or None."""
    import pwd

    try:
        return pwd.getpwuid(uid).pw_gid
    except (KeyError, OverflowError):
        return None


def host_home_dirs() -> tuple[str, ...]:
    """EXT-66: `/root`, every directory directly under `/home`, and every user database home under `/home`."""
    import glob
    import pwd

    homes = {"/root"}
    for pw in pwd.getpwall():
        if _abs(pw.pw_dir) and _is_parent(HOME_ROOT, pw.pw_dir):
            homes.add(_norm(pw.pw_dir))
    homes.update(_norm(p) for p in glob.glob(f"{HOME_ROOT}/*") if os.path.isdir(p))
    return tuple(sorted(homes))


def engine_data_dirs(configs: tuple[str, ...] = DAEMON_CONFIGS) -> tuple[str, ...]:
    """EXT-62, EXT-73: ENGINE_DATA_DIRS plus the `data-root` of each readable daemon configuration. Only that
    key is read; the file is never printed."""
    dirs = list(ENGINE_DATA_DIRS)
    for p in configs:
        try:
            root = json.loads(Path(p).read_text(encoding="utf-8")).get("data-root")
        except (OSError, ValueError, AttributeError):
            continue
        if _abs(root):
            dirs.append(_norm(root))
    return tuple(dict.fromkeys(dirs))


def gpu_step_path(host_work: str) -> Path:
    return Path(host_work, *GPU_STEP)


def _plain_path(p: str) -> bool:
    """No `:` or `,` (docker's `-v` and `--tmpfs` separators) and no control character (EXT-39)."""
    return not any(c in p for c in ":,") and all(ord(c) >= 32 and c != "\x7f" for c in p)


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _abs(p: Any) -> bool:
    return isinstance(p, str) and PurePosixPath(p).is_absolute()


def _norm(p: str) -> str:
    """Lexical normal form; a leading `//` (kept by POSIX `normpath`) is read as `/`, as the kernel does
    (EXT-56)."""
    n = posixpath.normpath(p)
    return "/" + n.lstrip("/") if n.startswith("//") else n


def _real(p: str) -> str:
    """`realpath` of the normal form, lenient (the path need not exist)."""
    return _realpath(_norm(p))


def _socket_groups(paths: tuple[str, ...]) -> list[tuple[str, int]]:
    """(path, group) of every existing path the `glob` patterns `paths` match (the target's group for a link)."""
    import glob

    found: list[tuple[str, int]] = []
    for pattern in paths:
        for p in sorted(glob.glob(pattern)):
            try:
                found.append((p, os.stat(p).st_gid))
            except OSError:
                pass
    return found


def docker_socket_gids(paths: tuple[str, ...] | None = None) -> set[int]:
    """EXT-51: the groups of the Docker daemon sockets that exist on this host (`DOCKER_SOCKET_GLOBS`)."""
    return {gid for _, gid in _socket_groups(DOCKER_SOCKET_GLOBS if paths is None else paths)}


# ---------------------------------------------------------------------------
# EXT-263: the launch rules on a rootless host
# ---------------------------------------------------------------------------

def owner_uid(cfg: HostConfig) -> int:
    """EXT-263: the owner, whose rootless daemon runs the containers (`cfg.owner_uid`, else this process's uid)."""
    return os.getuid() if cfg.owner_uid is None else cfg.owner_uid


def _primary_gid(cfg: HostConfig, uid: int) -> int | None:
    return (passwd_gid if cfg.primary_gid is None else cfg.primary_gid)(uid)


def gpu_options(rootless: bool) -> tuple[str, ...]:
    """The GPU options of a GPU step: Docker's native CDI under a rootless daemon (EXT-263), else the NVIDIA runtime."""
    return CDI_GPU_OPTIONS if rootless else ROOTFUL_GPU_OPTIONS


def container_user(cfg: HostConfig) -> str:
    """The container's `uid:gid`: `0:0` under a rootless daemon (the owner on the host, EXT-263), else the owner's uid
    and primary group (EXT-66). Raises ValueError for an owner without a user database entry."""
    if cfg.rootless:
        return ROOTLESS_USER
    uid = owner_uid(cfg)
    gid = _primary_gid(cfg, uid)
    if gid is None:
        raise ValueError(f"uid {uid} is not a user of this host")
    return f"{uid}:{gid}"


def rootless_docker_host(uid: int | str, *, run_user_dir: str | None = None) -> str:
    """The `unix://` address of the rootless Docker daemon of `uid` (a number, or the shell variable a generated script
    names it by): `<run_user_dir>/<uid>/docker.sock`. Only a number or a literal shell variable belongs here: the value is
    not quoted."""
    return "unix://" + posixpath.join(RUN_USER_DIR if run_user_dir is None else run_user_dir, str(uid), ROOTLESS_SOCKET)


def rootless_socket(cfg: HostConfig, *, run_user_dir: str | None = None) -> str:
    """EXT-263: the owner's rootless daemon socket, `<run_user_dir>/<owner uid>/docker.sock` (the context `gatedtrain`)."""
    return posixpath.join(RUN_USER_DIR if run_user_dir is None else run_user_dir, str(owner_uid(cfg)), ROOTLESS_SOCKET)


def rootful_socket_gids(cfg: HostConfig, *, globs: tuple[str, ...] | None = None,
                        run_user_dir: str | None = None) -> set[int]:
    """EXT-263: the groups of the Docker sockets `docker_socket_gids` finds (`globs`, default DOCKER_SOCKET_GLOBS, plus
    the host file's `extra_daemon_sockets`), except a socket whose resolved path lies under the owner's runtime
    directory `<run_user_dir>/<owner uid>/` (the owner's own rootless daemon; a link there to another socket counts)."""
    own = _real(posixpath.join(RUN_USER_DIR if run_user_dir is None else run_user_dir, str(owner_uid(cfg))))
    paths = (DOCKER_SOCKET_GLOBS if globs is None else tuple(globs)) + tuple(cfg.extra_daemon_sockets)
    return {gid for p, gid in _socket_groups(paths) if not _within(_real(p), own)}


def rootless_daemon_problem(cfg: HostConfig, *, run_user_dir: str | None = None) -> str | None:
    """EXT-263, `ext-launch --recheck`: why the owner's rootless daemon socket is not in place, or None (also None on a
    rootful host). A stat only: the socket exists, is a socket (not a link) and is owned by the owner; no process is
    started (the local backend asks the daemon itself, AN-22)."""
    import stat

    if not cfg.rootless:
        return None
    path, uid = rootless_socket(cfg, run_user_dir=run_user_dir), owner_uid(cfg)
    try:
        st = os.lstat(path)
    except OSError as exc:
        return (f"the owner's rootless Docker socket {path} cannot be read ({exc.__class__.__name__}): the rootless "
                f"daemon is not running for uid {uid} (EXT-263)")
    if not stat.S_ISSOCK(st.st_mode):
        return f"{path} is not a socket; the owner's rootless Docker daemon listens there (EXT-263)"
    if st.st_uid != uid:
        return (f"{path} is owned by uid {st.st_uid}, not the owner's uid {uid}: it is not the owner's rootless Docker "
                "daemon (EXT-263)")
    return None


def option_value(v: str) -> str:
    """EXT-39: `v` as a printed option value; ValueError when docker would read it as an option (empty or `-...`)."""
    if not v or v.startswith("-"):
        raise ValueError(f"{v!r} would be read by docker as an option")
    return v


def mount_source(host: str) -> str:
    """EXT-69: the `-v` source of the mount of `host`, its strict `realpath` on this host; ValueError when it is an
    option, does not resolve, or resolves to a path holding `:`, `,` or a control character."""
    option_value(host)
    real = _strict(host)
    if real is None:
        raise ValueError(f"mount source {host!r} does not resolve on this host")
    if not _plain_path(real):
        raise ValueError(f"mount source {host!r} resolves to {real!r}, which holds `:`, `,` or a control character")
    return real


def _kernel_problem(p: str, engine_dirs: tuple[str, ...] = ENGINE_DATA_DIRS) -> str | None:
    for d in KERNEL_DIRS:
        if _within(p, d):
            return f"{p} lies under {d}, a kernel file system that exposes other processes and the host"
    for d in engine_dirs:                                # EXT-62, EXT-73
        if _within(p, d):
            return f"{p} is or lies under {d}, a container engine's data directory"
        if _is_parent(p, d):
            return f"{p} contains {d}, a container engine's data directory (EXT-73)"
    return None


def _host_problem(host: str, engine_dirs: tuple[str, ...] = ENGINE_DATA_DIRS) -> tuple[str | None, str | None]:
    """EXT-51: (why `host` may not be mounted, its strict `realpath`). The lexical path and the path the
    host resolves it to are both judged; a path that does not resolve for this user is refused, since dockerd
    (root) may resolve it elsewhere (`/proc/1/root`)."""
    lexical = _norm(host)
    why = _kernel_problem(lexical, engine_dirs)
    if why:
        return why, None
    try:
        real = os.path.realpath(lexical, strict=True)
    except (OSError, ValueError) as exc:
        return (f"{host} does not resolve on this host ({exc.__class__.__name__}); a mount source must exist and "
                "be readable here, never resolved by the daemon alone"), None
    why = _kernel_problem(real, engine_dirs)
    if why:
        return f"{host} resolves to {real}; {why}", None
    return None, real


def _strict(p: str) -> str | None:
    try:
        return os.path.realpath(_norm(p), strict=True)
    except (OSError, ValueError):
        return None


def _secret_problem(host: str, real: str | None, homes: tuple[str, ...]) -> str | None:
    """EXT-66: why a mount source could expose credentials, judged on the lexical and the resolved path: it is,
    contains or lies under `/etc`, `/root` or `/var/log`; it is `/home` or a home directory or contains one; it
    is or lies under a dot-directory directly in a home directory (`~/.ssh`, `~/.docker`, `~/.cache`, ...)."""
    for p in dict.fromkeys(x for x in (_norm(host), real) if x is not None):
        for d in SECRET_DIRS:
            if _within(p, d) or _is_parent(p, d):
                return f"{host} is, contains or lies under {d}"
        if _within(HOME_ROOT, p):
            return f"{host} is or contains {HOME_ROOT}"
        for h in homes:
            if _within(h, p):
                return f"{host} is or contains the home directory {h}"
            if _is_parent(h, p) and PurePosixPath(p).relative_to(h).parts[0].startswith("."):
                return f"{host} is or lies under a dot-directory of the home directory {h}"
    return None


def _uid_gid(user: Any) -> tuple[int, int] | None:
    m = USER_RE.fullmatch(user) if isinstance(user, str) else None
    return (int(m.group(1)), int(m.group(2))) if m else None


def _is_parent(parent: str, child: str) -> bool:
    """Whether `parent` is a proper ancestor of `child` (lexical, POSIX paths)."""
    pp, cc = PurePosixPath(_norm(parent)), PurePosixPath(_norm(child))
    return pp != cc and pp in cc.parents


def _within(path: str, root: str) -> bool:
    """Whether `path` is `root` or lies under it (lexical, POSIX paths)."""
    return _norm(path) == _norm(root) or _is_parent(root, path)


def _socket_problem(host: str, real: str | None = None, extra: tuple[str, ...] = ()) -> str | None:
    """EXT-40: why a mount of `host` could reach a Docker daemon socket, or None. The lexical path and its
    `realpath` (symbolic links resolved on this host; strict, EXT-51) are both judged."""
    import stat

    for p in dict.fromkeys((_norm(host), real if real is not None else _real(host))):
        if os.path.basename(p).endswith(".sock") or "docker.sock" in p:
            return f"{host} is a socket path"
        for sock in SOCKET_PATHS + tuple(extra):
            if _within(sock, p):
                return f"{host} is or contains the Docker socket {sock}"
        for d in SOCKET_DIRS:
            if _within(p, d) or _is_parent(p, d):
                return f"{host} is, contains or lies under {d}, which holds daemon sockets"
        try:
            if stat.S_ISSOCK(os.stat(p).st_mode):
                return f"{host} is a socket"
        except OSError:
            pass
    return None


def _fold(name: str) -> str:
    """EXT-93: the name as matched: NFKC-normalised (fullwidth `ＩＤ_rsa` is `ID_rsa`), then case-folded."""
    return unicodedata.normalize("NFKC", name).casefold()


def _variants(p: str) -> set[str]:
    """The lexical normal form of `p` and its strict `realpath` when it resolves."""
    return {x for x in (_norm(p), _strict(p)) if x is not None}


def _stdlib_secrets(name: str, top: str, trees: tuple[str, ...] = ()) -> bool:
    """EXT-80, EXT-93: the standard library's `secrets` module of a declared interpreter tree: `top` is exactly
    `<tree>/lib/python3.<N>` and `name` is `secrets.py`, or `top` is its `__pycache__` and `name` the module's byte
    code. Nothing is exempt without a tree."""
    for tree in trees:
        for t in _variants(tree):
            if not _is_parent(t, top):
                continue
            parts = PurePosixPath(top).relative_to(t).parts
            if name == "secrets.py":
                if len(parts) == 2 and parts[0] == "lib" and _STDLIB_DIR_RE.fullmatch(parts[1]):
                    return True
            elif (len(parts) == 3 and parts[0] == "lib" and _STDLIB_DIR_RE.fullmatch(parts[1])
                  and parts[2] == "__pycache__" and _STDLIB_SECRETS_PYC_RE.fullmatch(name)):
                return True
    return False


def _credential_glob(name: str, top: str, trees: tuple[str, ...] = ()) -> str | None:
    low, parent = _fold(name), _fold(os.path.basename(top.rstrip("/")))
    for d, n in CREDENTIAL_DIR_NAMES:                                       # EXT-93
        if low == n and parent == d:
            return f"{d}/{n}"
    for g in CREDENTIAL_NAME_GLOBS:
        if fnmatch.fnmatchcase(low, g) and not _stdlib_secrets(name, top, trees):
            return g
    return None


def credential_name_problem(root: str, limit: int = SCAN_LIMIT, *, skip: tuple[str, ...] = (),
                            interpreter_trees: tuple[str, ...] = ()) -> str | None:
    """EXT-80, EXT-93: why the read-only source `root` may hold a credential file, judged on names only (no file
    is opened): an entry whose normalised name matches CREDENTIAL_NAME_GLOBS or CREDENTIAL_DIR_NAMES, a directory
    that cannot be listed, or more than `limit` entries. Only the directories in `skip` (exact paths, lexical or
    resolved) are not descended into, and a `root` that is itself a skip path is not walked (its own name is still
    matched); links are not followed. The stdlib `secrets` module is exempt in
    `interpreter_trees` only."""
    root = str(root)
    skipped = set().union(*(_variants(d) for d in skip)) if skip else set()
    trees = tuple(interpreter_trees)
    glob = _credential_glob(os.path.basename(root.rstrip("/")) or "/", os.path.dirname(root.rstrip("/")), trees)
    if glob:
        return f"{root} has a credential-like name (`{glob}`; EXT-80)"
    if not os.path.isdir(root):
        return None
    if _variants(root) & skipped:
        return None
    errors: list[OSError] = []
    seen = 0
    for top, dirs, files in os.walk(root, onerror=errors.append):
        dirs[:] = sorted(d for d in dirs if _norm(os.path.join(top, d)) not in skipped)
        for name in sorted(dirs) + sorted(files):
            seen += 1
            if seen > limit:
                return (f"{root} holds more than {limit} entries; the credential file-name scan stops there, so "
                        "the source is not accepted (mount a narrower directory; EXT-80)")
            glob = _credential_glob(name, top, trees)
            if glob:
                return (f"{root} holds {os.path.join(top, name)}, a credential-like file name (`{glob}`); "
                        "only names are read, never contents (EXT-80, EXT-93)")
        if errors:
            break
    if errors:
        e = errors[0]
        return (f"cannot list {e.filename} ({e.__class__.__name__}) under {root}; the credential file-name scan "
                "cannot rule out a credential file there (EXT-80)")
    return None


def _ro_table(mounts: Any) -> list[tuple[str, str]]:
    """(container path, strict host realpath) of each well-formed read-only mount whose source resolves."""
    table = []
    for m in mounts if isinstance(mounts, list) else []:
        if isinstance(m, dict) and m.get("mode") == "ro" and _abs(m.get("host")) and _abs(m.get("container")):
            real = _strict(m["host"])
            if real is not None:
                table.append((_norm(m["container"]), real))
    return table


def _host_of(cpath: str, table: list[tuple[str, str]]) -> str | None:
    """The host path behind the container path `cpath`: inside exactly one read-only mount (nested mounts are
    refused, EXT-94, so two would be ambiguous), else None."""
    hits = [(c, h) for c, h in table if _within(cpath, c)]
    if len(hits) != 1:
        return None
    c, h = hits[0]
    rel = PurePosixPath(cpath).relative_to(c)
    return str(PurePosixPath(h, rel)) if rel.parts else h


def container_file(path: Any, mounts: Any) -> tuple[str, str] | None:
    """EXT-91: (resolved container path, host file) of the absolute container path `path` as the container
    resolves it: component by component, each inside a read-only mount (or a proper ancestor of a mount's container
    path, a directory docker creates); a link is read on the host and its target taken as a container path
    (absolute) or relative to the link's directory, at most LINK_HOPS links. The result is a regular file in a
    read-only mount; None otherwise (a path through the read-write output, a tmpfs or the image, a dangling or
    looping link, a directory)."""
    if not _abs(path):
        return None
    table = _ro_table(mounts)
    pending = list(PurePosixPath("/" + path.lstrip("/")).parts[1:])
    cur, hops = "/", 0
    while pending:
        comp = pending.pop(0)
        if comp in ("", "."):
            continue
        if comp == "..":
            cur = posixpath.dirname(cur)
            continue
        nxt = posixpath.join(cur, comp)
        host = _host_of(nxt, table)
        if host is None:
            if any(_is_parent(nxt, c) for c, _ in table):
                cur = nxt
                continue
            return None
        if os.path.islink(host):
            hops += 1
            if hops > LINK_HOPS:
                return None
            try:
                target = os.readlink(host)
            except OSError:
                return None
            if target.startswith("/"):
                cur = "/"
            pending = [p for p in PurePosixPath(target).parts if p != "/"] + pending
            continue
        cur = nxt
    host = _host_of(cur, table)
    try:
        if host is None or not stat.S_ISREG(os.lstat(host).st_mode):
            return None
    except OSError:
        return None
    return cur, host


@dataclass(frozen=True)
class ArgvBinding:
    """EXT-91: the interpreter (resolved container path and host file) and the script (argv[1] as written and its
    host file) the record binds. EXT-192: `interpreter_host` is None for an interpreter inside the pinned image."""
    interpreter_path: str
    interpreter_host: str | None
    script_path: str
    script_host: str


def _in_tree(host: str, trees: tuple[str, ...]) -> bool:
    """EXT-113: whether the host file's resolved path lies strictly inside one of `trees` (lexical or resolved). The
    path resolves strictly: one that does not resolve on this host (a loop, an unreadable link) is in no tree (S6)."""
    real = _strict(host)
    if real is None:
        return False
    return any(_is_parent(t, real) for tree in trees for t in _variants(tree))


def bind_argv(argv: Any, mounts: Any, *, interpreter: str | None = None, trees: tuple[str, ...] | None = None,
              image_interpreters: Mapping[str, str] | None = None,
              interpreter_flags: Mapping[str, str] | None = None) -> tuple[ArgvBinding | None, str | None]:
    """EXT-83, EXT-91: (the binding, None) when argv[0] is an interpreter and argv[1] a script, both regular files
    in read-only mounts as the container resolves them (`container_file`), the interpreter named `python`,
    `python3` or `python3.<N>`, the script ending in `.py`, and argv[1] not an option; else (None, why).
    EXT-113: with `interpreter`, argv[0] is exactly that string (CPython finds the venv from the unresolved argv[0],
    so another bin directory linking to the same file would swap site-packages); with `trees`, the file argv[0]
    resolves to lies inside one of them on the host (a renamed copy of another program does not). ext-launch,
    `--recheck` and ext-collect pass the adapter's.
    EXT-192: with `image_interpreters` (script file name -> interpreter inside the image), argv[0] may instead be the
    interpreter mapped to argv[1]'s file name; it lies in no mount (a mount there would replace it) and is pinned by the
    image ID, so the binding has no host file for it. EXT-210: with `interpreter_flags`, the script may follow the one
    option the adapter allows for it (argv[2])."""
    fix = " (EXT-91)"
    if not (isinstance(argv, list) and len(argv) >= 2 and all(isinstance(a, str) for a in argv[:2])):
        return None, "argv must be [<interpreter>, <script.py>, <script arguments>...]" + fix
    images = dict(image_interpreters or {})
    if argv[0] in images.values():
        # EXT-210: the one interpreter option the adapter allows before that script (`-I`), as for a host interpreter.
        si = script_index(argv, interpreter_flags)
        want = images.get(posixpath.basename(argv[si]))
        if want != argv[0]:
            return None, (f"argv[0] {argv[0]!r} is the image's interpreter of "
                          f"{sorted(s for s, i in images.items() if i == argv[0])}, not of {argv[si]!r} (EXT-192)")
        if _host_of(_norm(argv[0]), _ro_table(mounts)) is not None or any(
                _within(_norm(argv[0]), _norm(m["container"]))
                for m in (mounts if isinstance(mounts, list) else []) if isinstance(m, dict)
                and _abs(m.get("container"))):
            return None, f"argv[0] {argv[0]!r} lies in a mount; the image's own interpreter is never replaced (EXT-192)"
        if argv[si].startswith("-"):
            return None, f"argv[{si}] {argv[si]!r} is an option; the script follows the interpreter directly" + fix
        script = container_file(argv[si], mounts)
        if script is None or not (argv[si].endswith(".py") and script[0].endswith(".py")):
            return None, (f"argv[{si}] {argv[si]!r} does not resolve, as the container sees it, to a `.py` file in a "
                          "read-only mount" + fix)
        return ArgvBinding(argv[0], None, argv[si], script[1]), None
    if interpreter is not None and argv[0] != interpreter:
        return None, (f"argv[0] {argv[0]!r} is not the adapter's interpreter {interpreter!r}"
                      + (f" (or the image's {sorted(set(images.values()))} for their scripts)" if images else "")
                      + ", written exactly so: the venv the interpreter loads is found from argv[0] as written (EXT-113)")
    interp = container_file(argv[0], mounts)
    if interp is None:
        return None, (f"argv[0] {argv[0]!r} does not resolve, as the container sees it, to a file in a read-only mount; "
                      "the interpreter is an absolute container path of a read-only input" + fix)
    if not INTERPRETER_RE.fullmatch(posixpath.basename(interp[0])):
        return None, (f"argv[0] {argv[0]!r} resolves to {interp[0]}, not a python interpreter (python, python3, "
                      "python3.<N>)" + fix)
    if trees is not None and not _in_tree(interp[1], tuple(trees)):
        return None, (f"argv[0] {argv[0]!r} resolves to {interp[0]} ({_realpath(interp[1])} on this host), "
                      f"which lies in no interpreter tree of the adapter ({', '.join(trees) or 'none'}) (EXT-113)")
    si = script_index(argv, interpreter_flags)                                 # EXT-199: `python -I <script>`
    if argv[si].startswith("-"):
        return None, (f"argv[{si}] {argv[si]!r} is an option; the script follows the interpreter directly, so `-c`, "
                      "`-m`, `-` and interpreter options are refused" + fix)
    script = container_file(argv[si], mounts)
    if script is None:
        return None, (f"argv[{si}] {argv[si]!r} does not resolve, as the container sees it, to a file in a read-only "
                      "mount; the training script is an absolute container path of a read-only input" + fix)
    if not (argv[si].endswith(".py") and script[0].endswith(".py")):
        return None, f"argv[{si}] {argv[si]!r} (resolved {script[0]}) is not a `.py` script" + fix
    return ArgvBinding(interp[0], interp[1], argv[si], script[1]), None


def script_index(argv: list, flags: Mapping[str, str] | None) -> int:
    """EXT-199: the index of the script in `argv`: 2 when argv[1] is the one interpreter option the adapter allows for
    the script argv[2] names (`interpreter_flags`, e.g. `-I` for the executor's scripts), else 1."""
    flags = dict(flags or {})
    if (len(argv) > 2 and isinstance(argv[1], str) and isinstance(argv[2], str)
            and flags.get(posixpath.basename(argv[2])) == argv[1]):
        return 2
    return 1


def launch_label(record: dict) -> str:
    """EXT-46: `<request label>-<first 12 hex of SHA-256(canonical record)>`; `record["label"]` holds the
    request label."""
    return f"{record['label']}-{sha256_bytes(canonical_json(record))[len('sha256:'):][:LABEL_DIGEST_HEX]}"


def label_problem(record: dict) -> str | None:
    """ext-collect's re-check of the record's hashed label (EXT-46), or None."""
    label = record.get("label")
    m = HASHED_LABEL_RE.fullmatch(label) if isinstance(label, str) else None
    if not m:
        return (f"label {label!r} is not `<request label>-<{LABEL_DIGEST_HEX} hex>`; ext-launch writes one label "
                "per launch record")
    try:
        want = launch_label(dict(record, label=m.group("base")))
    except (TypeError, ValueError):
        return "the record cannot be hashed"
    if want != label:
        return (f"label {label!r} is not this record's checksum ({want}): the record changed after ext-launch "
                "wrote it. The checksum is unkeyed; it detects accidental edits only, not a record rewritten on "
                "purpose")
    return None


# ---------------------------------------------------------------------------
# MM-32, MM-34: rules shared by the request (ext-launch) and the record (ext-collect)
# ---------------------------------------------------------------------------

def _output_problem(output: str, host_work: str | None, cfg: HostConfig, *, uid: int | None,
                    fresh: bool) -> str | None:
    """EXT-63: why `output` cannot be the run's own new output directory, or None. It resolves (strict) to a
    directory strictly under the configured output root's resolved path, outside host_work (lexical and
    resolved), owned by the container's uid `uid` (on a rootless host the owner's, which callers pass; EXT-263);
    with `fresh`, it is empty."""
    import stat

    root = _strict(cfg.output_root)
    if root is None:
        return f"the output root {cfg.output_root} does not resolve on this host"
    real = _strict(output)
    if real is None:
        return f"output_dir {output} does not resolve on this host; create it (empty, as the container's user) first"
    if not _is_parent(root, real):
        return (f"output_dir {output} resolves to {real}, not strictly under the output root {cfg.output_root} "
                f"({root})")
    if host_work is not None and (_within(output, host_work)
                                  or _within(real, _strict(host_work) or _real(host_work))):
        return f"output_dir {output} lies in host_work {host_work}, which holds gpu-step.sh and the GPU logs"
    try:
        st = os.stat(real)
        if not stat.S_ISDIR(st.st_mode):
            return f"output_dir {output} is not a directory"
        if uid is not None and st.st_uid != uid:
            if cfg.rootless:
                return (f"output_dir {output} is owned by uid {st.st_uid}, not the owner's uid {uid} (the container's "
                        "uid 0 under the rootless daemon; EXT-263)")
            return f"output_dir {output} is owned by uid {st.st_uid}, not the container's uid {uid}"
        if fresh:
            with os.scandir(real) as entries:
                if any(True for _ in entries):
                    return (f"output_dir {output} is not empty; a new run writes into a new, empty directory, so "
                            "nothing of an earlier run is collected as this one's")
    except OSError as exc:
        return f"output_dir {output} cannot be read ({exc.__class__.__name__})"
    return None


def _system_container_dir(c: str) -> str | None:
    """EXT-112: the system directory the normal-form container path `c` is or lies under (`/` itself, every `/lib*`
    directory, SYSTEM_CONTAINER_DIRS), or None."""
    if c == "/":
        return "/"
    first = PurePosixPath(c).parts[1]
    if first.startswith("lib"):
        return f"/{first}"
    return next((d for d in SYSTEM_CONTAINER_DIRS if _within(c, d)), None)


def _identity_host(c: str, mirror_root: str | None) -> str:
    """EXT-112: the host path an identity mount of the normal-form container path `c` has: `c` itself, or `c` under
    the mirror root (tests root the host side under a test directory)."""
    if not mirror_root or _norm(mirror_root) == "/":
        return c
    return _norm(_norm(mirror_root) + c)


def _container_problem(c: str, mode: Any, identity: bool, rules: Mapping[str, Any] | None) -> str | None:
    """EXT-112: why a mount may not sit at the normal-form container path `c`, or None. `identity`: the mount is at its
    own host path (`_identity_host`) and that lies in an input root. A system directory is refused whatever `rules`
    (the adapter's `container_paths`) say, except for a read-only identity mount; without rules every path is."""
    ro_identity = mode == "ro" and identity
    sys_dir = _system_container_dir(c)
    if sys_dir is not None and not ro_identity:
        return (f"container path {c} is or lies under {sys_dir}, a directory of the image's system (loader "
                "configuration, entrypoint, interpreters, libraries) that no mount replaces (EXT-112)")
    if rules is None:
        return f"container path {c}: no container paths are configured without an adapter (EXT-112)"
    if mode == "ro":
        under = tuple(rules.get("ro_under") or ())
        if (rules.get("ro_identity") and ro_identity) or any(_is_parent(u, c) for u in under):
            return None
        return (f"container path {c} of a read-only mount is neither its own host path inside an input root nor under "
                f"{', '.join(u.rstrip('/') + '/' for u in under) or 'an allowed directory'} (EXT-112)")
    if mode == "rw" and not any(c == _norm(x) for x in rules.get("rw") or ()):
        return (f"container path {c} of the read-write mount is not {', '.join(rules.get('rw') or ()) or 'allowed'} "
                "(EXT-112)")
    return None


def _env_path_problem(value: str) -> str | None:
    """EXT-114: why a path-valued environment value is refused (the value is never shown), or None: it is absolute and
    its normal form is or lies under ENV_PATH_ROOTS."""
    if not value.startswith("/"):
        return "is not an absolute path"
    if not any(_within(_norm(value), r) for r in ENV_PATH_ROOTS):
        return f"does not lie at or under {' or '.join(ENV_PATH_ROOTS)} (in normal form)"
    return None


def env_value_problems(env: Mapping[str, Any], path_names: tuple[str, ...]) -> list[tuple[str, str]]:
    """EXT-114: [(`env.<name>`, message)] for each path-valued name in `env` whose value is not a string at or under
    /tmp or /out. Messages never hold the value."""
    out = []
    for name in sorted(n for n in env if n in path_names):
        why = _env_path_problem(env[name]) if isinstance(env[name], str) else "is not a string"
        if why:
            out.append((f"env.{name}", f"{name} names a home or cache directory, and its value {why}; a home or cache "
                        "lies in the tmpfs /tmp or the output /out, never in an input (the value is not shown; "
                        "EXT-114)"))
    return out


def _env_refusal(name: str, allowed: frozenset[str]) -> str:
    """EXT-92: the refusal of an environment name outside the adapter's allow-list (the value is never shown)."""
    return (f"{name} is not in the adapter's environment allow-list ({len(allowed)} names"
            f"{': ' + ', '.join(sorted(allowed)) if allowed else '; none without an adapter'}); any other name, "
            "loader, interpreter and shell variables included (LD_*, PYTHON*, BASH_ENV, PATH), is refused (EXT-92)")


def isolation_problems(doc: dict, *, kind: str, framework: str | None = None,
                       socket_gids: set[int] | frozenset[int] | None = None, config: HostConfig | None = None,
                       fresh_output: bool | None = None, allow_network: bool = False) -> list[tuple[str, str]]:
    """[(field, message)] for every broken rule; `kind` is "request" (env values present) or "record" (env
    names only). Field names are request or record paths (`mounts[1].mode`, `env.HF_TOKEN`, ...).
    `socket_gids` are the groups of the Docker daemon sockets (EXT-51); None reads them on this host. `config`
    is the host layout (None: `host_config()`; input roots None: none, `adapter_config` fills the adapter's);
    `fresh_output` (default: for a request) also requires the output directory to be empty (EXT-63).
    `allow_network` is ext-launch's `--allow-network` for a request; a record carries `allow_network` (EXT-81)."""
    out: list[tuple[str, str]] = []
    bad = lambda f, m: out.append((f, m))
    cfg = host_config() if config is None else config
    fresh = (kind == "request") if fresh_output is None else fresh_output
    engines = (engine_data_dirs(DAEMON_CONFIGS + cfg.extra_daemon_configs) if cfg.engine_dirs is None
               else tuple(cfg.engine_dirs))
    homes = host_home_dirs() if cfg.home_dirs is None else tuple(cfg.home_dirs)
    roots = tuple(cfg.input_roots or ())                                    # EXT-79
    allowed_env = frozenset(cfg.env_names or ())                            # EXT-92
    ids = _uid_gid(doc.get("user"))
    spool = _spool(doc, cfg)                                                # EXT-190: (container, host) or None
    spool_mounts: list[int] = []

    # Runtime pinning (MM-32).
    rt = doc.get("runtime")
    if not isinstance(rt, dict):
        bad("runtime", "runtime must be a mapping {kind, image, ...}")
        rt = {}
    if rt.get("kind") not in RUNTIME_KINDS:
        bad("runtime.kind", f"runtime kind {rt.get('kind')!r} is not one of {', '.join(RUNTIME_KINDS)}")
    if not (isinstance(rt.get("image"), str) and IMAGE_RE.fullmatch(rt["image"])):
        bad("runtime.image", f"image {rt.get('image')!r} is not pinned: give the image ID sha256:<64 hex> or a "
                             "<ref>@sha256:<64 hex> digest, never a mutable tag")
    if rt.get("kind") == "venv_in_container":
        if kind == "request":
            lock = rt.get("venv_lock")
            if not (isinstance(lock, str) and Path(lock).is_file()):
                bad("runtime.venv_lock", f"venv_in_container needs `venv_lock`, the committed lock file of the "
                                         f"host venv; {lock!r} is not a file")
        elif not (isinstance(rt.get("venv_lock_sha256"), str) and SHA_RE.fullmatch(rt["venv_lock_sha256"])):
            bad("runtime.venv_lock_sha256", "venv_in_container records the lock file's SHA-256")

    # Output, host work and mounts (MM-34).
    output = doc.get("output_dir")
    if not _abs(output):
        bad("output_dir", f"output_dir {output!r} is not an absolute path")
        output = None
    host_work = doc.get("host_work")
    if not _abs(host_work):
        bad("host_work", f"host_work {host_work!r} is not an absolute path")
        host_work = None
    elif _norm(host_work) != _norm(cfg.host_work):                       # EXT-67
        bad("host_work", f"host_work {host_work} is not this host's work directory {cfg.host_work}, whose "
                         f"bin/gpu-step.sh and gpu-guard.log account the GPU time")
    if output is not None:
        why = _output_problem(output, host_work, cfg, uid=owner_uid(cfg) if cfg.rootless else ids[0] if ids else None,
                              fresh=fresh)                                  # EXT-263: the owner's on a rootless host
        if why:
            bad("output_dir", why)
    mounts = doc.get("mounts")
    if not isinstance(mounts, list):
        bad("mounts", "mounts must be a list of {host, container, mode}")
        mounts = []
    rw_output = 0
    containers: dict[str, int] = {}
    placed: list[tuple[int, str, Any]] = []                                 # EXT-88: (index, container, mode)
    sources: dict[int, tuple[str, str | None]] = {}                         # EXT-112: index -> (host, resolved)
    for t in doc.get("tmpfs") if isinstance(doc.get("tmpfs"), list) else []:
        if _abs(t):
            containers.setdefault(_norm(t), -1)
    for i, m in enumerate(mounts):
        f = f"mounts[{i}]"
        if not isinstance(m, dict) or set(m) != {"host", "container", "mode"}:
            bad(f, "a mount is exactly {host, container, mode}")
            continue
        host, mode = m["host"], m["mode"]
        before = len(out)
        if not _abs(host):
            bad(f"{f}.host", f"host path {host!r} is not absolute")
            host = None
        elif not _plain_path(host):
            bad(f"{f}.host", f"host path {host!r} holds `:`, `,` or a control character, which change the meaning "
                             "of `-v host:container:mode` (EXT-39)")
            host = None
        if not _abs(m["container"]):
            bad(f"{f}.container", f"container path {m['container']!r} is not absolute")
        elif not _plain_path(m["container"]):
            bad(f"{f}.container", f"container path {m['container']!r} holds `:`, `,` or a control character "
                                  "(EXT-39)")
        else:                                                              # EXT-74
            c = _norm(m["container"])
            placed.append((i, c, mode))
            if c in containers:
                prev = "a tmpfs" if containers[c] < 0 else f"mounts[{containers[c]}]"
                bad(f"{f}.container", f"container path {c} is also {prev}; each container path is mounted once")
            else:
                containers[c] = i
        if mode not in ("ro", "rw"):
            bad(f"{f}.mode", f"mode {mode!r} is neither ro nor rw")
        if host is None:
            continue
        why, real = _host_problem(host, engines)
        sources[i] = (host, real)
        if why:
            bad(f"{f}.host", f"{why} (EXT-51)")
        elif not _plain_path(real):
            bad(f"{f}.host", f"{host} resolves to {real!r}, which holds `:`, `,` or a control character; the printed "
                             "command mounts the resolved path (EXT-39, EXT-69)")
        sock = _socket_problem(host, real, cfg.extra_daemon_sockets)
        if sock:
            bad(f"{f}.host", f"a Docker socket is never mounted: {sock} (EXT-40)")
        why = _secret_problem(host, real, homes)                          # EXT-66
        if why:
            bad(f"{f}.host", f"a credential store is never mounted: {why} (EXT-66)")
        if mode == "ro" and not any(_within(host, r) and (real is None or _within(real, _strict(r) or _norm(r)))
                                    for r in roots):
            bad(f"{f}.host", f"{host} lies under no input root ({', '.join(roots) or 'none configured'}); read-only "
                             "inputs come from the adapter's input roots only (EXT-66, EXT-79)")
        # MM-34 on the lexical and on the resolved paths (EXT-56): `//x` is `/x`, a link is its target.
        if host_work is not None and (_norm(host) == _norm(host_work)
                                      or (real is not None and real == _real(host_work))):
            bad(f"{f}.host", "the host work directory is never mounted (MM-34)")
        elif host_work is not None and (_is_parent(host, host_work)
                                        or (real is not None and _is_parent(real, _real(host_work)))):
            bad(f"{f}.host", f"{host} is a parent of the host work directory, which is never mounted (EXT-62)")
        elif output is not None and (_is_parent(host, output)
                                     or (real is not None and _is_parent(real, _real(output)))):
            bad(f"{f}.host", f"{host} is a parent of the output directory; mount the output itself")
        elif mode == "ro" and output is not None and (_within(host, output)
                                                      or (real is not None and _within(real, _real(output)))):
            bad(f"{f}.host", f"{host} is the output directory {output} or lies inside it; a run never reads its own "
                             "output as an input (EXT-104)")
        is_output = output is not None and (_norm(host) == _norm(output)
                                            or (real is not None and real == _strict(output)))
        is_spool = (spool is not None and mode == "rw" and not is_output and _abs(m["container"])
                    and _norm(m["container"]) == spool[0] and real is not None and real == spool[1])
        if mode == "rw":
            if is_output:
                rw_output += 1
            elif is_spool:                                                  # EXT-190: the sidecar's spool
                spool_mounts.append(i)
            else:
                bad(f"{f}.mode", f"{host} is an input and must be mounted read-only; only the run's own output "
                                 "directory is read-write"
                                 + (f" (and the sidecar's spool at {spool[0]}, the directory the sidecar mounts there)"
                                    if spool is not None else ""))
        if mode == "ro" and real is not None and len(out) == before:          # EXT-80: names only, last
            why = credential_name_problem(real, skip=tuple(cfg.scan_skip or ()),
                                          interpreter_trees=tuple(cfg.interpreter_trees or ()))
            if why:
                bad(f"{f}.host", f"a credential file is never mounted: {why}")
    if output is not None and rw_output != 1:
        bad("output_dir", f"exactly one read-write mount must be the output directory {output}; found {rw_output}")
    # Apply the adapter's mount allow-list and system directory restrictions (EXT-112).
    rules = cfg.container_paths
    for i, c, mode in placed:
        host, real = sources.get(i, (None, None))
        # The mount is at its own host path, as written or as resolved (a link to the directory is mounted by its
        # target, `build_command`), inside an input root.
        identity = (host is not None and _identity_host(c, cfg.mirror_root) in {_norm(host), real}
                    and any(_within(host, r) and (real is None or _within(real, _strict(r) or _norm(r)))
                            for r in roots))
        why = None if i in spool_mounts else _container_problem(c, mode, identity, rules)
        if why:
            bad(f"mounts[{i}].container", why)
    tmpfs = doc.get("tmpfs")
    if not (isinstance(tmpfs, list) and all(_abs(t) and _plain_path(t) for t in tmpfs)):
        bad("tmpfs", "tmpfs must be a list of absolute container paths without `:` or `,` (no mount options)")
    elif any(_norm(t) == "/" for t in tmpfs):                                # EXT-88
        bad("tmpfs", "a tmpfs at / would hide the image's file system; mount scratch at a directory such as /tmp")
    else:
        seen: set[str] = set()
        for t in tmpfs:
            n = _norm(t)
            if n in seen:                                                    # EXT-115
                bad("tmpfs", f"tmpfs path {n} is listed more than once ({t!r}); each tmpfs path is mounted once "
                             "(EXT-115)")
            seen.add(n)
            sys_dir = _system_container_dir(n)
            if sys_dir is not None:                                          # EXT-112
                bad("tmpfs", f"tmpfs path {n} is or lies under {sys_dir}, a directory of the image's system (EXT-112)")
            elif rules is None or not any(n == _norm(x) for x in rules.get("tmpfs") or ()):
                bad("tmpfs", f"tmpfs path {n} is not "
                             f"{', '.join((rules or {}).get('tmpfs') or ()) or 'allowed without an adapter'} (EXT-112)")
    # Reject nested mounts in either direction so one mount cannot shadow another's files (EXT-88, EXT-94).
    spots = [(f"mounts[{j}].container", c, f"mounts[{j}] ({mode})") for j, c, mode in placed]
    spots += [("tmpfs", _norm(t), "the tmpfs") for t in (tmpfs if isinstance(tmpfs, list) else [])
              if _abs(t) and _norm(t) != "/"]
    for f_in, c_in, _ in spots:
        for _, c_out, what in spots:
            if _is_parent(c_out, c_in):
                bad(f_in, f"container path {c_in} lies inside {c_out}, {what}; mounts never nest, in either "
                          "direction (EXT-88, EXT-94)")

    # Network and user (MM-34).
    network = doc.get("network")
    allowed = allow_network if kind == "request" else doc.get("allow_network")
    if kind == "record" and not isinstance(allowed, bool):
        bad("allow_network", f"allow_network {allowed!r} is not true or false (EXT-81)")
    if not isinstance(network, str) or not network:
        bad("network", "network must be named (`none` unless a reason is given)")
    elif not NETWORK_RE.fullmatch(network):
        bad("network", f"network {network!r} is not a Docker network name (EXT-39)")
    elif network == "host":                                                  # EXT-81
        bad("network", "network `host` is never used: the container would share the host's network namespace, "
                       "its loopback services included (EXT-81)")
    elif network != "none":
        if allowed is not True:
            bad("network", f"network {network!r} needs the operator's `ext-launch --allow-network`; the default is "
                           "`none` (EXT-81)")
        if not (isinstance(doc.get("network_reason"), str) and doc["network_reason"].strip()):
            bad("network_reason", f"network {network!r} needs a `network_reason`; the default is `none`")
    user = doc.get("user")
    if cfg.rootless:                                                         # EXT-263
        if user != ROOTLESS_USER:
            bad("user", f"user {user!r} is not {ROOTLESS_USER}: under a rootless daemon the container runs as "
                        f"{ROOTLESS_USER}, which is the owner on the host (any other uid is a subordinate uid that "
                        "cannot write the owner's output; EXT-263)")
        uid = owner_uid(cfg)
        gids = rootful_socket_gids(cfg) if socket_gids is None else set(socket_gids)
        primary = _primary_gid(cfg, uid)
        if primary is None:
            bad("user", f"the owner uid {uid} is not a user of this host (EXT-263)")
        elif primary in gids:
            bad("user", f"the owner's primary group {primary} owns a rootful Docker socket; container gid 0 is that "
                        "group on the host, so the container could drive that daemon (EXT-263)")
    elif ids is None:
        bad("user", f"user {user!r} is not `<uid>:<gid>` (ASCII digits)")
    elif ids[0] > ID_MAX or ids[1] > ID_MAX:                                # EXT-74
        bad("user", f"uid or gid above {ID_MAX} in {user!r}; the kernel's ids are 32 bits and larger values wrap")
    elif ids[0] == 0:
        bad("user", "the container runs as root (uid 0); use a non-root uid:gid")
    else:
        uid, gid = ids
        gids = (docker_socket_gids(DOCKER_SOCKET_GLOBS + cfg.extra_daemon_sockets) if socket_gids is None
                else set(socket_gids))
        primary = (passwd_gid if cfg.primary_gid is None else cfg.primary_gid)(uid)
        if gid == 0:
            bad("user", "the container's group is root (gid 0); use a non-root gid (EXT-51)")
        elif gid in gids:
            bad("user", f"gid {gid} owns a Docker daemon socket on this host; a container in that group can drive "
                        "the daemon through any socket it reaches (EXT-51)")
        elif primary is None:
            bad("user", f"uid {uid} is not a user of this host; the container runs as a host user (EXT-66)")
        elif gid != primary:
            bad("user", f"gid {gid} is not the primary group {primary} of uid {uid}; a supplementary group (adm, "
                        "shadow, docker, ...) would open files the run must not read (EXT-66)")

    # Environment: names only in the record, values never printed (MM-34, MM-35).
    if kind == "request":
        env = doc.get("env")
        if not isinstance(env, dict) or not all(isinstance(k, str) for k in env):
            bad("env", "env must map variable names to values")
            env = {}
        for name in sorted(env):
            if not ENV_NAME_RE.fullmatch(name):
                bad(f"env.{name}", "an environment variable name is `[A-Za-z_][A-Za-z0-9_]*` (EXT-39)")
            elif CREDENTIAL_RE.search(name.upper()):
                bad(f"env.{name}", "credentials are never passed into a container (the value is not shown)")
            elif name not in allowed_env:                                       # EXT-92
                bad(f"env.{name}", _env_refusal(name, allowed_env))
            elif not isinstance(env[name], str):
                bad(f"env.{name}", "an environment value must be a string")
            elif any(c in env[name] for c in "\n\r\x00"):                       # EXT-88
                bad(f"env.{name}", "an environment value holds a line break or NUL (the value is not shown)")
        for f, msg in env_value_problems(env, tuple(cfg.env_paths or ())):     # EXT-114
            if isinstance(env.get(f[len("env."):]), str):
                bad(f, msg)
        if env.get(HUB_OFFLINE) != "1" and not _b18_script(doc, cfg):           # EXT-199: no model loaded
            bad(f"env.{HUB_OFFLINE}", f"{HUB_OFFLINE} must be \"1\" in training and evaluation")
    else:
        names = doc.get("env_names")
        if not (isinstance(names, list) and all(isinstance(n, str) for n in names)):
            bad("env_names", "env_names must be a list of variable names")
            names = []
        for name in names:
            if not ENV_NAME_RE.fullmatch(name):
                bad("env_names", f"{name!r} is not an environment variable name (EXT-39)")
            elif CREDENTIAL_RE.search(name.upper()):
                bad("env_names", f"{name} names a credential; credentials are never passed into a container")
            elif name not in allowed_env:                                       # EXT-92
                bad("env_names", _env_refusal(name, allowed_env))
        if HUB_OFFLINE not in names and not _b18_script(doc, cfg):
            bad("env_names", f"{HUB_OFFLINE} is not set; training and evaluation run offline")
        if "rootless" in doc and doc["rootless"] is not True:                 # EXT-263
            bad("rootless", f"rootless is {doc['rootless']!r}: a record written on a rootless host carries "
                            "`\"rootless\": true` and any other omits the key")
        elif doc.get("rootless", False) != cfg.rootless:
            bad("rootless", f"the record was written under rootless={str(doc.get('rootless', False)).lower()}, this "
                            f"host is rootless={str(cfg.rootless).lower()} (the host file's `rootless`; EXT-263)")

    # Plain fields.
    if kind == "request":
        if not (isinstance(doc.get("label"), str) and LABEL_RE.fullmatch(doc["label"])):
            bad("label", f"label {doc.get('label')!r} is not a gpu-step.sh label (letters, digits, _ . -; first a "
                         "letter or digit; at most 100 characters)")
    else:
        why = label_problem(doc)
        if why:
            bad("label", why)
    if not _int(doc.get("area")):
        bad("area", "area must be an integer")
    if not _int(doc.get("training_seed")):
        bad("training_seed", "training_seed must be an integer (MM-29)")
    if not isinstance(doc.get("gpus"), bool):
        bad("gpus", "gpus must be true or false")
    if doc.get("gpus") is False:                                                # EXT-105 (MM-31)
        if not (_num(doc.get("gpu_est_h")) and doc["gpu_est_h"] >= 0):
            bad("gpu_est_h", "gpu_est_h must be a number >= 0 for a CPU-only step (gpus false), which is logged with "
                             "0 GPU-hours (MM-31)")
    elif not (_num(doc.get("gpu_est_h")) and doc["gpu_est_h"] > 0):              # EXT-67
        bad("gpu_est_h", "gpu_est_h must be a number > 0 (gpu-step.sh checks the cap against it)")
    argv = doc.get("argv")
    if not (isinstance(argv, list) and argv and all(isinstance(a, str) for a in argv)):
        bad("argv", "argv must be a non-empty list of strings")
    elif not all(argv):                                                      # EXT-88
        bad("argv", "argv holds an empty string, which the printed command would quote as ''")
    fw = doc.get("framework")
    if not (isinstance(fw, dict) and set(fw) == {"name", "version", "commit"} and isinstance(fw.get("name"), str)
            and isinstance(fw.get("version"), str) and (fw["commit"] is None or isinstance(fw["commit"], str))):
        bad("framework", "framework must be {name, version, commit}")
    elif framework is not None and fw["name"] != framework:
        bad("framework.name", f"framework {fw['name']!r} is not the adapter's {framework!r}")
    out += sidecar_problems(doc, kind=kind, cfg=cfg, socket_gids=socket_gids, fresh=fresh,   # EXT-190
                            spool_mounts=len(spool_mounts))
    return out


def _b18_script(doc: Any, cfg: HostConfig) -> str | None:
    """EXT-199: the script file name when the document runs one of the adapter's executor-image scripts, else None."""
    argv = doc.get("argv") if isinstance(doc, dict) else None
    if not (cfg.b18 and isinstance(argv, list) and len(argv) > 1 and all(isinstance(a, str) for a in argv)):
        return None
    name = posixpath.basename(argv[script_index(argv, cfg.interpreter_flags)])
    return name if name in tuple(cfg.b18.get("scripts") or ()) else None


def _spool(doc: dict, cfg: HostConfig) -> tuple[str, str] | None:
    """EXT-190: (container path, strict host realpath) of the sidecar's read-write spool when the adapter declares a
    sidecar and the document carries one with exactly one read-write mount at the row's spool path; else None."""
    row, side = cfg.sidecar, doc.get("sidecar")
    if not row or not isinstance(side, dict) or not isinstance(side.get("mounts"), list):
        return None
    rw = [m for m in side["mounts"] if isinstance(m, dict) and m.get("mode") == "rw"]
    if len(rw) != 1 or not _abs(rw[0].get("host")) or not _abs(rw[0].get("container")) \
            or _norm(rw[0]["container"]) != _norm(row["spool"]):
        return None
    real = _strict(rw[0]["host"])
    return (_norm(row["spool"]), real) if real is not None else None


def _strict_type_equal(a: Any, b: Any) -> bool:
    """Equal JSON values with equal types (`True` is not `1`, `8` is not `"8"`, `1.0` is not `1`)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return set(a) == set(b) and all(_strict_type_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_strict_type_equal(x, y) for x, y in zip(a, b))
    return a == b


def sidecar_problems(doc: Any, *, kind: str, cfg: HostConfig, socket_gids: set[int] | frozenset[int] | None = None,
                     fresh: bool = False, spool_mounts: int | None = None) -> list[tuple[str, str]]:
    """EXT-190: [(field, message)] for the paired sidecar of a request (`kind` "request") or record ("record"), judged
    against the adapter's registry row `cfg.sidecar`. A document without `sidecar` gives
    nothing here (whether one is required is the adapter's `launch_problems`); one with a sidecar for an adapter that
    declares none is refused. `fresh` (ext-launch, `--recheck`) also requires the spool to be empty. `spool_mounts` is
    the number of trainer mounts isolation_problems accepted as the spool (None: not counted here)."""
    out: list[tuple[str, str]] = []
    bad = lambda f, m: out.append((f, m))
    if not isinstance(doc, dict) or "sidecar" not in doc:
        return out
    side = doc["sidecar"]
    row = cfg.sidecar
    if not row:
        bad("sidecar", "this adapter declares no paired sidecar container (registry `sidecar`); remove `sidecar`")
        return out
    if not isinstance(side, dict):
        bad("sidecar", f"sidecar is {side!r}, not a mapping; a launch without a sidecar omits the key")
        return out
    keys = SIDECAR_REQUEST_KEYS if kind == "request" else SIDECAR_RECORD_KEYS
    for k in keys:
        if k not in side:
            bad(f"sidecar.{k}", f"missing sidecar key {k} (the sidecar states every isolation setting, EXT-190)")
    for k in sorted(set(side) - set(keys), key=str):
        bad(f"sidecar.{k}", f"unknown sidecar key {k!r}; allowed: {', '.join(keys)}")
    # An image_interpreter is pinned by the executor image digest. A mounted venv uses its lock (MM-32, EXT-249).
    rt = side.get("runtime") if isinstance(side.get("runtime"), dict) else {}
    image_py = row.get("image_interpreter")
    if image_py:
        rt_keys = ("kind", "image") if kind == "request" else ("kind", "image", "venv_lock_sha256")
        if set(rt) != set(rt_keys) or rt.get("kind") != "container" or (
                kind == "record" and rt.get("venv_lock_sha256") is not None):
            bad("sidecar.runtime", f"runtime must be {{{', '.join(rt_keys)}}} with kind container (the image's own "
                                   f"interpreter {image_py}; no venv lock)")
    else:
        rt_keys = ("kind", "image", "venv_lock") if kind == "request" else ("kind", "image", "venv_lock_sha256")
        if set(rt) != set(rt_keys) or rt.get("kind") != "venv_in_container":
            bad("sidecar.runtime", f"runtime must be {{{', '.join(rt_keys)}}} with kind venv_in_container")
    if rt.get("image") != row["image"]:
        bad("sidecar.runtime.image", f"image {rt.get('image')!r} is not the pinned executor image {row['image']}")
    if not image_py:
        lock = _file_sha_or_none(rt.get("venv_lock")) if kind == "request" else rt.get("venv_lock_sha256")
        if lock != row["venv_lock_sha256"]:
            bad("sidecar.runtime.venv_lock", f"the executor venv lock is {lock!r}, not the pinned executor lock "
                                             f"{row['venv_lock_sha256']} (MM-32)")
    # Isolation flags, network, user, limits, tmpfs (executor/run-executor.sh COMMON).
    for key, want in SIDECAR_FLAGS.items():
        if key in side and not _strict_type_equal(side[key], want):
            bad(f"sidecar.{key}", f"{key} is {side[key]!r}, not {want!r}: the executor container runs with a read-only "
                                  "root file system, no capabilities, no-new-privileges and no GPU (B18)")
    if side.get("network") != "none":
        bad("sidecar.network", f"network {side.get('network')!r}: the executor runs model-written code and has network "
                               "none, always (decision \"Area 6 decisions\")")
    if side.get("user") != doc.get("user"):
        bad("sidecar.user", f"user {side.get('user')!r} is not the trainer's {doc.get('user')!r}: the executor binds its "
                            "socket with umask 077, so only the same uid can connect")
    limits = side.get("limits")
    if "limits" in side and not _strict_type_equal(limits, dict(row["limits"])):
        bad("sidecar.limits", f"limits {limits!r} are not the executor's {dict(row['limits'])!r} (cpus, memory, swap "
                              "equal to memory, pids, open files, no core dumps; B18)")
    if "tmpfs" in side and not _strict_type_equal(side.get("tmpfs"), list(row["tmpfs"])):
        bad("sidecar.tmpfs", f"tmpfs {side.get('tmpfs')!r} is not {list(row['tmpfs'])!r} (mounted "
                             f"{row['tmpfs_options']})")
    # Mounts: identity read-only mounts of exactly the row's directories, and one read-write spool.
    mounts = side.get("mounts") if isinstance(side.get("mounts"), list) else []
    if not isinstance(side.get("mounts"), list):
        bad("sidecar.mounts", "mounts must be a list of {host, container, mode}")
    engines = (engine_data_dirs(DAEMON_CONFIGS + cfg.extra_daemon_configs) if cfg.engine_dirs is None
               else tuple(cfg.engine_dirs))
    homes = host_home_dirs() if cfg.home_dirs is None else tuple(cfg.home_dirs)
    roots = tuple(cfg.input_roots or ())
    ids = _uid_gid(doc.get("user"))
    output = doc.get("output_dir") if _abs(doc.get("output_dir")) else None
    ro_seen: list[str] = []
    spool_host = None
    for i, m in enumerate(mounts):
        f = f"sidecar.mounts[{i}]"
        if not (isinstance(m, dict) and set(m) == {"host", "container", "mode"} and _abs(m.get("host"))
                and _abs(m.get("container")) and _plain_path(m["host"]) and _plain_path(m["container"])):
            bad(f, "a sidecar mount is exactly {host, container, mode} with absolute paths without `:` or `,`")
            continue
        host, c, mode = m["host"], _norm(m["container"]), m["mode"]
        why, real = _host_problem(host, engines)
        if why:
            bad(f"{f}.host", f"{why} (EXT-51)")
            continue
        sock = _socket_problem(host, real, cfg.extra_daemon_sockets)
        if sock:
            bad(f"{f}.host", f"a Docker socket is never mounted: {sock} (EXT-40)")
        secret = _secret_problem(host, real, homes)
        if secret:
            bad(f"{f}.host", f"a credential store is never mounted: {secret} (EXT-66)")
        if mode == "rw":
            if c != _norm(row["spool"]):
                bad(f"{f}.mode", f"{host} is mounted read-write at {c}; the sidecar's only read-write mount is its spool "
                                 f"at {row['spool']}, every other mount is read-only")
                continue
            if spool_host is not None:
                bad(f"{f}.mode", "the sidecar has one read-write mount, its spool")
                continue
            spool_host = real
            why = _output_problem(host, doc.get("host_work") if _abs(doc.get("host_work")) else None, cfg,
                                  uid=owner_uid(cfg) if cfg.rootless else ids[0] if ids else None,   # EXT-263
                                  fresh=fresh)
            if why:
                bad(f"{f}.host", f"the spool: {why.replace('output_dir', 'spool')}")
            if output is not None and (_within(real, _strict(output) or _norm(output)) or _within(output, host)
                                       or _within(_strict(output) or _norm(output), real)):
                bad(f"{f}.host", f"the spool {host} is, holds or lies inside the trainer's output {output}; the trainer "
                                 "and the sidecar share only the spool")
            elif output is not None and real != posixpath.join(posixpath.dirname(_strict(output) or _norm(output)),
                                                                 posixpath.basename(row["spool"])):
                bad(f"{f}.host", f"the spool {host} is not {posixpath.basename(row['spool'])}/ beside the trainer's output "
                                 f"{output} (expected <run>/train and <run>/spool)")
            for j, tm in enumerate(doc.get("mounts") if isinstance(doc.get("mounts"), list) else []):
                if not (isinstance(tm, dict) and _abs(tm.get("host"))) or (
                        tm.get("mode") == "rw" and _abs(tm.get("container"))
                        and _norm(tm["container"]) == _norm(row["spool"])):
                    continue                            # the trainer's own spool mount (checked below)
                th = _strict(tm["host"]) or _norm(tm["host"])
                if _within(th, real) or _within(real, th):
                    bad(f"{f}.host", f"the trainer's mounts[{j}] {tm['host']} is, holds or lies inside the spool {host}; "
                                     "the spool reaches the trainer only at its own mount")
        elif mode == "ro":
            identity = _identity_host(c, cfg.mirror_root) in {_norm(host), real}
            if c not in {_norm(x) for x in row["ro_mounts"]} or not identity:
                bad(f"{f}.container", f"read-only mount {host} at {c}: the sidecar mounts exactly "
                                      f"{', '.join(row['ro_mounts'])}, each at its own host path")
                continue
            if not any(_within(host, r) and _within(real, _strict(r) or _norm(r)) for r in roots):
                bad(f"{f}.host", f"{host} lies under no input root ({', '.join(roots) or 'none configured'}) (EXT-79)")
                continue
            if c in ro_seen:
                bad(f"{f}.container", f"{c} is mounted twice")
            ro_seen.append(c)
            cred = credential_name_problem(real, skip=tuple(cfg.scan_skip or ()),
                                           interpreter_trees=tuple(cfg.interpreter_trees or ()))
            if cred:
                bad(f"{f}.host", f"a credential file is never mounted: {cred}")
        else:
            bad(f"{f}.mode", f"mode {mode!r} is neither ro nor rw")
    missing = sorted({_norm(x) for x in row["ro_mounts"]} - set(ro_seen))
    if isinstance(side.get("mounts"), list) and missing:
        bad("sidecar.mounts", f"the sidecar mounts no {', '.join(missing)} read-only at its own host path (the "
                              "executor's interpreter and code)")
    if isinstance(side.get("mounts"), list) and spool_host is None:
        bad("sidecar.mounts", f"the sidecar has no read-write spool at {row['spool']} (the directory holding its socket)")
    elif spool_host is not None:
        trainer = [m for m in doc.get("mounts") or [] if isinstance(m, dict) and m.get("mode") == "rw"
                   and _abs(m.get("container")) and _norm(m["container"]) == _norm(row["spool"])
                   and _abs(m.get("host")) and _strict(m["host"]) == spool_host]
        if len(trainer) != 1 or (spool_mounts is not None and spool_mounts != 1):
            bad("sidecar.mounts", f"the trainer mounts the sidecar's spool {spool_host} read-write at {row['spool']} "
                                  f"{len(trainer)} times, not once: the two containers share exactly the spool")
    # Environment: the row's names exactly; values only in a request, never recorded (MM-35).
    names = sorted(row["env_names"])
    if kind == "request":
        env = side.get("env")
        if not (isinstance(env, dict) and all(isinstance(k, str) for k in env)):
            bad("sidecar.env", "env must map variable names to values")
            env = {}
        for k in sorted(env):
            if not ENV_NAME_RE.fullmatch(k):
                bad(f"sidecar.env.{k}", "an environment variable name is `[A-Za-z_][A-Za-z0-9_]*` (EXT-39)")
            elif CREDENTIAL_RE.search(k.upper()):
                bad(f"sidecar.env.{k}", "credentials are never passed into a container (the value is not shown)")
            elif k not in names:
                bad(f"sidecar.env.{k}", _env_refusal(k, frozenset(names)))
            elif not isinstance(env[k], str) or any(ch in env[k] for ch in "\n\r\x00"):
                bad(f"sidecar.env.{k}", "an environment value is a string without a line break or NUL")
        for k in names:
            if k not in env:
                bad(f"sidecar.env.{k}", f"{k} is not set; the sidecar sets {', '.join(names)} "
                                        "(executor/run-executor.sh)")
        for f, msg in env_value_problems({k: v for k, v in env.items() if k in names}, tuple(row["env_paths"])):
            bad(f"sidecar.{f}", msg)
    elif side.get("env_names") != names:
        bad("sidecar.env_names", f"env_names {side.get('env_names')!r} are not {names}")
    # argv: exactly the row's (the serve command), bound to files in the sidecar's read-only mounts.
    argv = side.get("argv")
    if not _strict_type_equal(argv, list(row["argv"])):
        bad("sidecar.argv", f"argv {argv!r} is not {list(row['argv'])!r} (executor/run-executor.sh serve; no other "
                            "option, no other limit)")
        return out
    ro = [m for m in mounts if isinstance(m, dict) and m.get("mode") == "ro"]
    if image_py:                                        # EXT-249: the image's interpreter (argv[0] is the row's)
        script = container_file(argv[2], ro)
        if script is None:
            bad("sidecar.argv", f"{argv[2]} does not resolve, as the container sees it, to a file in the sidecar's "
                                "read-only mounts (EXT-91)")
            return out
        if kind == "record":
            if side.get("interpreter_sha256") is not None:
                bad("sidecar.interpreter_sha256", f"interpreter_sha256 is {side.get('interpreter_sha256')!r}: the "
                                                  f"image's interpreter {image_py} is pinned by the image ID (null)")
            now = _file_sha_or_none(script[1])
            if side.get("script_sha256") != now:
                bad("sidecar.script_sha256", f"{script[0]} ({script[1]}) is {now!r}, not the "
                                             f"{side.get('script_sha256')!r} ext-launch recorded: it changed after the "
                                             "launch (EXT-83)")
        return out
    interp, script = container_file(argv[0], ro), container_file(argv[2], ro)
    if interp is None or script is None:
        bad("sidecar.argv", f"{argv[0]} or {argv[2]} does not resolve, as the container sees it, to a file in the "
                            "sidecar's read-only mounts (EXT-91)")
        return out
    if not INTERPRETER_RE.fullmatch(posixpath.basename(interp[0])) or (
            cfg.interpreter_trees is not None and not _in_tree(interp[1], tuple(cfg.interpreter_trees))):
        bad("sidecar.argv", f"{argv[0]} resolves to {interp[0]}, not a python interpreter in the adapter's interpreter "
                            "trees (EXT-113)")
    if kind == "record":
        for key, (path, host) in (("interpreter_sha256", interp), ("script_sha256", script)):
            now = _file_sha_or_none(host)
            if side.get(key) != now:
                bad(f"sidecar.{key}", f"{path} ({host}) is {now!r}, not the {side.get(key)!r} ext-launch recorded: it "
                                      "changed after the launch (EXT-83)")
    return out


def _file_sha_or_none(path: Any) -> str | None:
    try:
        return file_sha256(Path(path)) if isinstance(path, str) and Path(path).is_file() else None
    except OSError:
        return None


def record_problems(record: Any, *, adapter: str, stage_out: Path | None, framework: str | None = None,
                    flag: str = "--launch-record", socket_gids: set[int] | None = None,
                    config: HostConfig | None = None, fresh_output: bool = False) -> list[Problem]:
    """ext-collect's re-check of a launch record (EXT-26): schema, the isolation rules, the adapter, and the
    one read-write mount being `--stage-out` (skipped when `stage_out` is None: `--recheck`). EXT-63, EXT-67:
    the recorded output root and GPU wrapper hash must agree with the host. The current input roots, host layout,
    interpreter and training script must match the record. `fresh_output` also requires an empty output directory."""
    cfg = adapter_config(host_config() if config is None else config, adapter)
    fix = "launch with the command `gatedtrain ext-launch` prints, then collect with its record"
    if not isinstance(record, dict):
        return [Problem("EXT-LAUNCH", flag, "the launch record is not a JSON object", fix)]
    why = schema_version_problem(record)                                    # EXT-95: never upgraded
    if why:
        return [Problem("EXT-LAUNCH", f"{flag}.schema_version", why, "launch again with the current ext-launch "
                        "(new record, new output directory); an older record is never upgraded")]
    out: list[Problem] = []
    for k in RECORD_KEYS:
        if k not in record:
            out.append(Problem("EXT-LAUNCH", f"{flag}.{k}", "missing launch record key (MM-35)", fix))
    for k in sorted(set(record) - set(RECORD_KEYS) - set(OPTIONAL_RECORD_KEYS)):
        out.append(Problem("EXT-LAUNCH", f"{flag}.{k}", "unknown launch record key (MM-35)", fix))
    if record.get("schema") != RECORD_SCHEMA:
        out.append(Problem("EXT-LAUNCH", f"{flag}.schema", f"schema is {record.get('schema')!r}, not {RECORD_SCHEMA}"
                           f"{old_schema_note(record.get('schema'))}", fix))
    if record.get("adapter") != adapter:
        out.append(Problem("EXT-LAUNCH", f"{flag}.adapter",
                           f"the record launched adapter {record.get('adapter')!r}, not --adapter {adapter!r}", fix))
    if record.get("profile") not in PROFILES:
        out.append(Problem("EXT-LAUNCH", f"{flag}.profile", f"profile {record.get('profile')!r} is not smoke or full",
                           fix))
    for f, msg in isolation_problems(record, kind="record", framework=framework, socket_gids=socket_gids,
                                     config=cfg, fresh_output=fresh_output):
        out.append(Problem("EXT-LAUNCH", f"{flag}.{f}", msg, fix))
    output = record.get("output_dir")
    if stage_out is not None and _abs(output) and _realpath(output) != _realpath(stage_out):
        out.append(Problem("EXT-LAUNCH", f"{flag}.output_dir",
                           f"the record's read-write output {output} is not --stage-out {stage_out}", fix))
    if "output_root" in record and (not _abs(record["output_root"])
                                    or _norm(record["output_root"]) != _norm(cfg.output_root)):
        out.append(Problem("EXT-LAUNCH", f"{flag}.output_root", f"the record's output root "
                           f"{record['output_root']!r} is not this host's {cfg.output_root} (EXT-63)", fix))
    hw = record.get("host_work")
    if "gpu_step_sha256" in record and _abs(hw):
        script = gpu_step_path(hw)
        try:
            now = file_sha256(script)
        except OSError as exc:
            now = f"unreadable ({exc.__class__.__name__})"
        if record["gpu_step_sha256"] != now:
            out.append(Problem("EXT-GPU", f"{flag}.gpu_step_sha256",
                               f"{script} is {now}, not the {record['gpu_step_sha256']!r} ext-launch recorded: the "
                               "wrapper that accounted the GPU time changed after the launch (EXT-67)",
                               "restore the recorded gpu-step.sh, or launch again with the current one"))
    if "host_config_sha256" in record and record["host_config_sha256"] != host_config_sha256(cfg):   # EXT-85
        out.append(Problem("EXT-LAUNCH", f"{flag}.host_config_sha256",
                           f"the record's host layout hash {record['host_config_sha256']!r} is not this host's "
                           f"{host_config_sha256(cfg)} (host_work {cfg.host_work}, output root {cfg.output_root}, "
                           f"input roots {', '.join(cfg.input_roots or ()) or 'none'}): the record was written under "
                           "another layout or other GATEDTRAIN_EXT_* overrides (EXT-85)",
                           "check with the layout ext-launch used, or launch again"))
    binding, why = bind_argv(record.get("argv"), record.get("mounts"), interpreter=cfg.interpreter,   # EXT-83, EXT-91,
                             trees=cfg.interpreter_trees, image_interpreters=cfg.image_interpreters,  # EXT-113, EXT-192
                             interpreter_flags=cfg.interpreter_flags)                                  # EXT-199
    if binding is None:
        out.append(Problem("EXT-LAUNCH", f"{flag}.argv", why, fix))
    else:
        for role, key, path, host in (("interpreter", "interpreter", binding.interpreter_path, binding.interpreter_host),
                                      ("training script", "script", binding.script_path, binding.script_host)):
            if f"{key}_path" not in record or f"{key}_sha256" not in record:
                continue                                                    # reported as missing keys
            if record[f"{key}_path"] != path:
                out.append(Problem("EXT-LAUNCH", f"{flag}.{key}_path",
                                   f"the record's {key}_path {record[f'{key}_path']!r} is not the {role} the argv "
                                   f"names ({path!r}) (EXT-83, EXT-91)", fix))
                continue
            if host is None:                                                # EXT-192: pinned by the image ID
                if record[f"{key}_sha256"] is not None:
                    out.append(Problem("EXT-LAUNCH", f"{flag}.{key}_sha256", f"the {role} {path} lies in the image, "
                                       f"which pins it; the record's {key}_sha256 is null, not "
                                       f"{record[f'{key}_sha256']!r} (EXT-192)", fix))
                continue
            try:
                now = file_sha256(Path(host))
            except OSError as exc:
                now = f"unreadable ({exc.__class__.__name__})"
            if now != record[f"{key}_sha256"]:
                out.append(Problem("EXT-LAUNCH", f"{flag}.{key}_sha256",
                                   f"the {role} {path} ({host}) is {now}, not the {record[f'{key}_sha256']!r} "
                                   "ext-launch recorded: it changed after the launch (EXT-83, EXT-91)",
                                   f"restore the recorded {role}, or launch again with the current one"))
    out += adapter_launch_problems(record, adapter=adapter, kind="record", flag=flag, fix=fix)       # EXT-190 to EXT-193
    return out


def adapter_launch_problems(doc: dict, *, adapter: str, kind: str, flag: str, fix: str) -> list[Problem]:
    """EXT-190 to EXT-193: the adapter's own launch rules (`launch_problems`: runtime per script, which launch pairs a
    sidecar), for the adapters that declare them; none otherwise."""
    from . import registry

    try:
        ad = registry.adapter(adapter)
    except KeyError:
        return []
    if not hasattr(ad, "launch_problems"):
        return []
    return [Problem("EXT-LAUNCH", f"{flag}.{f}", msg, fix) for f, msg in ad.launch_problems(doc, kind=kind)]


def schema_version_problem(record: dict) -> str | None:
    """Explain an unsupported launch record version (EXT-95)."""
    got = record.get("schema_version")
    if got == SCHEMA_VERSION:
        return None
    return (f"launch record schema_version {got!r} is not {SCHEMA_VERSION}: the record predates the current launch "
            "rules for interpreter binding, environment allow-lists, credential scans and nested mounts (EXT-95)")


# ---------------------------------------------------------------------------
# MM-36: GPU accounting
# ---------------------------------------------------------------------------

def _ts(token: str) -> _dt.datetime:
    return _dt.datetime.strptime(token, "%Y-%m-%dT%H:%M:%SZ")


def gpu_seconds(log_path: Path, label: str) -> int:
    """Seconds of GPU time of `label` in gpu-step.sh's log (EXT-28): each `START` of the label (third token
    exactly the label) is paired with the next `END` or `PREEMPTED` of the label, and the pair durations are
    summed over every invocation and attempt. Raises ValueError when the label is absent, has no `END`, or
    has a `START` without a terminator (design choice 9). Other events add nothing.

    EXT-46: gpu-step.sh exits with the container's rc after `END ... rc=<rc>`, and a failed invocation may be
    relaunched under the same label. The run finished only if the label's last `END` has `rc=0`; an `END` with
    another rc is accepted only when a later `END` of the label has `rc=0`. An `END` without `rc=<int>` is
    refused. EXT-53: the label's last terminal event (`END` or `PREEMPTED`) must be that `END rc=0`; a
    relaunch preempted after a finished run leaves the outputs of an unfinished attempt."""
    log_path = Path(log_path)
    total = 0
    open_start: _dt.datetime | None = None
    seen = ended = False
    last_rc: int | None = None
    last_event: str | None = None
    for n, line in enumerate(text_lines(log_path.read_text(encoding="utf-8")), start=1):
        tok = line.split()
        if len(tok) < 3 or tok[2] != label or tok[1] not in ("START", "END", "PREEMPTED"):
            continue
        seen = True
        try:
            t = _ts(tok[0])
        except ValueError:
            raise ValueError(f"{log_path}:{n}: timestamp {tok[0]!r} is not %Y-%m-%dT%H:%M:%SZ") from None
        if tok[1] == "START":
            if open_start is not None:
                raise ValueError(f"{log_path}:{n}: START of {label} while an earlier attempt is unterminated")
            open_start = t
        else:
            if open_start is None:
                raise ValueError(f"{log_path}:{n}: {tok[1]} of {label} without a START")
            total += int((t - open_start).total_seconds())
            open_start = None
            last_event = tok[1]
            if tok[1] == "END":
                ended = True
                rcs = [x[3:] for x in tok[3:] if x.startswith("rc=")]
                if len(rcs) != 1 or not re.fullmatch(r"-?[0-9]+", rcs[0]):
                    raise ValueError(f"{log_path}:{n}: END of {label} has no rc=<int>")
                last_rc = int(rcs[0])
    if not seen:
        raise ValueError(f"{log_path} has no START/END/PREEMPTED line for label {label!r}")
    if open_start is not None:
        raise ValueError(f"{log_path}: the last START of {label} has no END or PREEMPTED")
    if not ended:
        raise ValueError(f"{log_path}: label {label!r} has no END line (the run never finished)")
    if last_event != "END":
        raise ValueError(f"{log_path}: the last terminal event of {label!r} is {last_event}, after its last END; "
                         "a relaunch did not finish (EXT-53)")
    if last_rc != 0:
        raise ValueError(f"{log_path}: the last END of {label!r} has rc={last_rc}; the run failed (a failed "
                         "invocation counts only when a later END of the label has rc=0)")
    return total


def first_start(log_path: Path, label: str) -> float:
    """EXT-64: the POSIX time of the label's first `START` line in gpu-step.sh's log (UTC). Raises ValueError
    when there is none or its timestamp does not parse."""
    for n, line in enumerate(text_lines(Path(log_path).read_text(encoding="utf-8")), start=1):
        tok = line.split()
        if len(tok) >= 3 and tok[1] == "START" and tok[2] == label:
            try:
                return _ts(tok[0]).replace(tzinfo=_dt.timezone.utc).timestamp()
            except ValueError:
                raise ValueError(f"{log_path}:{n}: timestamp {tok[0]!r} is not %Y-%m-%dT%H:%M:%SZ") from None
    raise ValueError(f"{log_path} has no START line for label {label!r}")


GPU_STEP_WORK_RE = re.compile(r"""WORK=(?P<q>["']?)(?P<work>[^"'\s]+)(?P=q)\s*""")
GPU_STEP_LOG_LINE = 'LOG="$WORK/gpu-guard.log"'
# EXT-57: an assignment to WORK or LOG anywhere in a line (`X=`, `export X=`, `a; X=`, `${X:=..}`, `${X=..}`).
GPU_STEP_ASSIGN_RE = re.compile(r"(?<![A-Za-z0-9_$])(?P<name>WORK|LOG)(?:\+=|:?=)")


def gpu_step_assignments(text: str) -> dict[str, list[int]]:
    """EXT-57: line numbers of the assignments to WORK and LOG in gpu-step.sh's text (a lexical scan, not a
    shell parser: `read`, `for`, `eval` and the like are not seen)."""
    out: dict[str, list[int]] = {"WORK": [], "LOG": []}
    for n, line in enumerate(text_lines(text), start=1):
        code = line.split("#", 1)[0] if line.lstrip().startswith("#") else line
        for m in GPU_STEP_ASSIGN_RE.finditer(code):
            out[m.group("name")].append(n)
    return out


def gpu_step_log_problem(host_work: Path) -> str | None:
    """EXT-46: `<host_work>/bin/gpu-step.sh` writes its log to `$WORK/gpu-guard.log` with `WORK` hardcoded
    (`WORK=<host_work>` in the host's script), not to the
    host_work the record names. The log ext-collect reads, `<host_work>/gpu-guard.log`, is the script's log only
    when the script's `WORK=` line names host_work and its `LOG=` line is the stock one. None when it does.

    EXT-57: WORK and LOG are each assigned exactly once (a second `LOG=` after the stock one would move the log).
    This guards against a misconfigured host_work; a script forged under host_work passes, since ext-collect
    reads the script and does not run it."""
    script = Path(host_work) / "bin" / "gpu-step.sh"
    try:
        text = script.read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        return f"cannot read {script} ({exc.__class__.__name__}); it decides where the GPU log is written"
    assigned = gpu_step_assignments(text)
    for name, lines in assigned.items():
        if len(lines) != 1:
            return f"{script} assigns {name} {len(lines)} times (lines {lines}), not once (EXT-57)"
    works = [m.group("work") for m in map(GPU_STEP_WORK_RE.fullmatch, text_lines(text)) if m]
    if len(works) != 1:
        return f"{script} has {len(works)} `WORK=<path>` lines, not one"
    if GPU_STEP_LOG_LINE not in [l.strip() for l in text_lines(text)]:
        return f"{script} has no `{GPU_STEP_LOG_LINE}` line"
    if _norm(works[0]) != _norm(str(host_work)):
        return (f"{script} logs to {works[0]}/gpu-guard.log, not {_norm(str(host_work))}/gpu-guard.log; "
                "launch with host_work set to the script's WORK")
    return None


def gpu_used_h(gpu_log_md: Path) -> float:
    """The last `Running total` of the table in `<HOST_WORK>/gpu-log.md` (MM-31). Raises ValueError."""
    lines = text_lines(Path(gpu_log_md).read_text(encoding="utf-8"))
    col = None
    last = None
    for line in lines:
        s = line.strip()
        if not s.startswith("|"):
            col = None                                   # a table ends at its first non-table line
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if col is None:
            if "Running total" in cells:
                col = cells.index("Running total")
            continue
        if all(set(c) <= set("-: ") for c in cells):
            continue
        if col < len(cells):
            try:
                last = float(cells[col])
            except ValueError:
                raise ValueError(f"{gpu_log_md}: running total {cells[col]!r} is not a number") from None
    if last is None:
        raise ValueError(f"{gpu_log_md} has no table row with a `Running total`")
    return last


# ---------------------------------------------------------------------------
# ext-launch
# ---------------------------------------------------------------------------

def _g(x: float) -> str:
    """A number printed into the gpu-step.sh command: the exact shortest value, never rounded (`%g` kept six digits, so a
    ceiling of 1000.4 was enforced as 1000) and never in exponent form (`canonical.plain_decimal`)."""
    return plain_decimal(x)


def build_command(record: dict, env: dict[str, str], *, est_h: float, gpu_cap_h: float = GPU_CAP_H) -> str:
    """The gpu-step.sh command (EXT-24): tokens shell-quoted, env sorted by name, mounts in record order;
    `--runtime=nvidia` before `--gpus all`, `--shm-size=8g` after the security options, and each tmpfs path
    mounted with TMPFS_OPTIONS.
    Before the image, docker parses options: every token there is a fixed option or the value of the fixed
    option before it, and no value starts with `-` (EXT-39; raises ValueError, which the request checks
    prevent).
    EXT-69: each `-v` source is the mount's strict `realpath` on this host (raises ValueError when it does not
    resolve), so the daemon mounts the directory the rules judged, not whatever a symbolic link names later;
    `ext-launch --recheck` re-runs the rules right before the command.
    EXT-263: a record with `"rootless": true` gets the GPU through Docker's native CDI (`gpu_options`)."""
    value, source = option_value, mount_source
    from . import registry

    # EXT-191: the adapter's fixed options for this script (code, never request values); a `--shm-size=` among them
    # replaces the default, and a row's tmpfs options replace TMPFS_OPTIONS.
    adapter = record.get("adapter", "")
    script = (posixpath.basename(record["argv"][script_index(record["argv"], registry.interpreter_flags(adapter))])
              if len(record["argv"]) > 1 else "")
    b18 = registry.b18(adapter)
    if b18 and script in b18["scripts"]:                                      # EXT-199: run-executor.sh's layout
        tok = [f"GATEDTRAIN_GPU_CAP_H={_g(gpu_cap_h)}", f"GATEDTRAIN_GPU_USED_H={_g(record['gpu_used_h'])}",
               f"GATEDTRAIN_GPU_STEP_EST_H={_g(est_h)}", f"{record['host_work'].rstrip('/')}/bin/gpu-step.sh",
               value(record["label"]), "--rm", "--network", value(record["network"]), "--user", value(record["user"]),
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--read-only"]
        for t in record["tmpfs"]:
            tok += ["--tmpfs", value(f"{t}:{b18['tmpfs_options']}")]
        tok += list(b18["limit_options"])
        for var in sorted(env):
            tok += ["-e", value(f"{var}={env[var]}")]
        for m in record["mounts"]:
            tok += ["-v", value(f"{source(m['host'])}:{m['container']}:{m['mode']}")]
        tok.append(value(record["runtime"]["image"]))
        tok += list(record["argv"])
        return " ".join(shlex.quote(t) for t in tok)
    options, tmpfs_options = registry.run_options(adapter, script)
    shm = [o for o in options if o.startswith("--shm-size=")]
    tok = [f"GATEDTRAIN_GPU_CAP_H={_g(gpu_cap_h)}", f"GATEDTRAIN_GPU_USED_H={_g(record['gpu_used_h'])}",
           f"GATEDTRAIN_GPU_STEP_EST_H={_g(est_h)}", f"{record['host_work'].rstrip('/')}/bin/gpu-step.sh",
           value(record["label"]), "--rm"]
    if record["gpus"]:
        tok += list(gpu_options(record.get("rootless", False)))
    tok += ["--network", value(record["network"]), "--user", value(record["user"]), "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", *(shm or [SHM_SIZE])]
    tok += [o for o in options if not o.startswith("--shm-size=")]
    for name in sorted(env):
        tok += ["-e", value(f"{name}={env[name]}")]
    for m in record["mounts"]:
        tok += ["-v", value(f"{source(m['host'])}:{m['container']}:{m['mode']}")]
    for t in record["tmpfs"]:
        tok += ["--tmpfs", value(f"{t}:{tmpfs_options or TMPFS_OPTIONS}")]
    tok.append(value(record["runtime"]["image"]))
    tok += list(record["argv"])
    return " ".join(shlex.quote(t) for t in tok)


def sidecar_name(record: dict, row: Mapping[str, Any]) -> str:
    """Build the sidecar name from the adapter's prefix and the hashed launch label."""
    return f"{row['name_prefix']}{record['label']}"


def sidecar_commands(side: dict, env: dict[str, str], row: Mapping[str, Any], name: str) -> tuple[str, str]:
    """EXT-190: (start, stop) of the paired sidecar, as executor/run-executor.sh serve prints it: `docker --context gatedtrain
    run -d --name <name> --rm`, network none, the record's user, no capabilities, no-new-privileges, read-only root
    file system, the row's tmpfs options and limits, the environment, the mounts by their strict `realpath` (EXT-69),
    the image and argv; then `docker --context gatedtrain stop <name>` after the trainer. Every isolation option is code.
    The sidecar never gets a GPU (`SIDECAR_FLAGS`), so no GPU option is printed on a rootful or a rootless host."""
    value = option_value
    lim = side["limits"]
    tok = ["docker", "--context", DOCKER_CONTEXT, "run", "-d", "--name", value(name), "--rm", "--network", "none",
           "--user", value(side["user"]), "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--read-only"]
    for t in side["tmpfs"]:
        tok += ["--tmpfs", value(f"{t}:{row['tmpfs_options']}")]
    tok += ["--cpus", value(str(lim["cpus"])), "--memory", value(str(lim["memory"])), "--memory-swap",
            value(str(lim["memory_swap"])), "--pids-limit", value(str(lim["pids"])), "--ulimit",
            value(f"nofile={lim['nofile']}"), "--ulimit", value(f"core={lim['core']}")]
    for var in sorted(env):
        tok += ["-e", value(f"{var}={env[var]}")]
    for m in side["mounts"]:
        tok += ["-v", value(f"{mount_source(m['host'])}:{m['container']}:{m['mode']}")]
    tok.append(value(side["runtime"]["image"]))
    tok += list(side["argv"])
    stop = ["docker", "--context", DOCKER_CONTEXT, "stop", value(name)]
    return " ".join(shlex.quote(t) for t in tok), " ".join(shlex.quote(t) for t in stop)




def spec_binding_problems(spec: Path, mounts: Any, ad: Any, *, flag: str, argv: Any = None) -> list[Problem]:
    """EXT-65: with `--spec`, the read-only mounts bind the spec's base model (MM-21) and its export directory,
    the parent of the training shards (adapter `mount_binding_problems`). An adapter with `spec_binding_argv` also
    gets the launch's argv, to bind only the launches that need it (lerobot_smolvla: training, not evaluation)."""
    from ..spec import load_spec

    try:
        wf = load_spec(Path(spec))
    except Refused as exc:
        return exc.problems
    names = wf.names
    extra = {"argv": argv} if getattr(ad, "spec_binding_argv", False) else {}
    return ad.mount_binding_problems(mounts if isinstance(mounts, list) else [], base=wf.path_of("model"),
                                     data=wf.path_of("data"), base_field=f"artifacts.{names['model']}.uri",
                                     data_field=f"artifacts.{names['data']}.uri", flag=flag, **extra)


def _clause_evaluators(spec: Path) -> list[tuple[str, dict]] | None:
    """R18-8: (evaluator id, registry row) of every clause when the spec's registration has several clauses, else
    None (an unreadable spec is refused by `spec_binding_problems`)."""
    from . import registry
    from ..spec import load_spec

    try:
        wf = load_spec(Path(spec))
    except Refused:
        return None
    if wf.clause is not None:
        return None
    return [(c["evaluator"], registry.EVALUATOR_ROWS[c["evaluator"]]) for c in wf.clauses
            if c["evaluator"] in registry.EVALUATOR_ROWS] or None


def _adapter(adapter: str):
    from . import registry

    try:
        return registry.adapter(adapter)
    except KeyError:
        raise Refused([Problem("EXT-ADAPTER", "--adapter", f"no adapter {adapter!r} is registered",
                               f"use one of {', '.join(sorted(registry.ADAPTERS))}")])


def _spec_required(ad: Any, adapter: str) -> Problem:
    """EXT-84: the adapter binds the spec's base and export to read-only mounts, so `--spec` is required."""
    return Problem("EXT-SPEC", "--spec", f"adapter {adapter} binds the spec's base model and export directories to "
                   "read-only mounts (EXT-65), so the spec is required (EXT-84)",
                   "pass --spec SPEC, the spec the run will be collected with")


def ext_launch(*, adapter: str, profile: str, request: Path, gpu_log: Path, out_record: Path,
               smoke_record: Path | None = None, spec: Path | None = None,
               config: HostConfig | None = None, allow_network: bool = False) -> str:
    """Check the request, write the launch record and return the command line (never run). Raises Refused,
    and then writes nothing. `allow_network` is `--allow-network` (EXT-81)."""
    from . import registry
    from .smoke import smoke_problems

    cfg = adapter_config(host_config() if config is None else config, adapter)
    problems: list[Problem] = []
    fix = "fix the request and run ext-launch again"
    ad = _adapter(adapter)
    if spec is None and getattr(ad, "binds_spec", False):                 # EXT-84
        problems.append(_spec_required(ad, adapter))
    out_record = Path(out_record)
    if out_record.exists():
        problems.append(Problem("EXT-LAUNCH", "--out-record", f"{out_record} already exists; a record is never "
                                "overwritten", "pass a new --out-record path"))
    if profile not in PROFILES:
        problems.append(Problem("EXT-PROFILE", "--profile", f"profile {profile!r} is not smoke or full",
                                "pass --profile smoke or --profile full"))
    try:
        req = yaml.safe_load(Path(request).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise Refused(problems + [Problem("EXT-LAUNCH", "--request", f"cannot read {request}: "
                                          f"{exc.__class__.__name__}", "pass the launch request YAML")])
    if not isinstance(req, dict):
        raise Refused(problems + [Problem("EXT-LAUNCH", "--request", "the request is not a mapping",
                                          "write the request as a YAML mapping")])
    for k in REQUEST_KEYS:
        if k not in req:
            problems.append(Problem("EXT-LAUNCH", f"--request.{k}", "missing request key", fix))
    for k in sorted(set(req) - set(REQUEST_KEYS) - set(RETIRED_REQUEST_KEYS) - set(OPTIONAL_REQUEST_KEYS)):
        problems.append(Problem("EXT-LAUNCH", f"--request.{k}", "unknown request key",
                                f"remove it; allowed: {', '.join(REQUEST_KEYS)}"))
    for k in RETIRED_REQUEST_KEYS:                                          # EXT-97
        if k in req and req[k] is not None:
            problems.append(Problem("EXT-LAUNCH", f"--request.{k}", f"{k} is retired: network `host` is never used "
                                    "(EXT-81), so a reason for it has no meaning; a null value is accepted and not "
                                    "recorded (EXT-97)", f"remove {k}"))
    for f, msg in isolation_problems(req, kind="request", framework=ad.framework_name, config=cfg,
                                     allow_network=allow_network):
        problems.append(Problem("EXT-LAUNCH", f"--request.{f}", msg, fix))
    binding, why = bind_argv(req.get("argv"), req.get("mounts"), interpreter=cfg.interpreter,   # EXT-83, EXT-91,
                             trees=cfg.interpreter_trees, image_interpreters=cfg.image_interpreters,  # EXT-113, EXT-192
                             interpreter_flags=cfg.interpreter_flags)                                  # EXT-199
    problems += adapter_launch_problems(req, adapter=adapter, kind="request", flag="--request", fix=fix)   # EXT-190
    if binding is None:
        problems.append(Problem("EXT-LAUNCH", "--request.argv", f"{why}; the record binds the code the run executes",
                                "run [<python in a read-only mount>, <script .py in a read-only mount>, <script "
                                "arguments>...], e.g. /venv/bin/python /code/train_grpo.py --out /out"))
    # EXT-67: the configured host_work's GPU log and wrapper.
    want_log = Path(cfg.host_work) / GPU_LOG_MD
    if _norm(str(Path(gpu_log).absolute())) != _norm(str(want_log)):
        problems.append(Problem("EXT-GPU", "--gpu-log", f"{gpu_log} is not {want_log}, the running total "
                                "gpu-step.sh checks the cap against", f"pass --gpu-log {want_log}"))
    where = gpu_step_log_problem(Path(cfg.host_work))
    if where:
        problems.append(Problem("EXT-GPU", "--request.host_work", where,
                                "launch through the gpu-step.sh that logs to <host_work>/gpu-guard.log"))
    try:
        step_sha = file_sha256(gpu_step_path(cfg.host_work))
    except OSError:
        step_sha = None                                  # reported by gpu_step_log_problem
    try:
        used = gpu_used_h(gpu_log)
    except (OSError, ValueError) as exc:
        used = None
        problems.append(Problem("EXT-GPU", "--gpu-log", f"cannot read the GPU-time log: {exc}",
                                "pass <HOST_WORK>/gpu-log.md with its running-total table"))
    for msg in (ad.argv_problems(req.get("argv"), profile=profile) if hasattr(ad, "argv_problems")    # EXT-133,
                else []):                                                                             # EXT-156
        problems.append(Problem("EXT-LAUNCH", "--request.argv", msg, fix))
    if spec is not None:                                 # EXT-65
        problems += spec_binding_problems(spec, req.get("mounts"), ad, flag="--request", argv=req.get("argv"))
    est = req.get("gpu_est_h")
    if profile == "full":
        # MM-31, MM-39 (EXT-27): a full run needs a PASS smoke record; its projection is the step estimate.
        smoke = None
        if smoke_record is None:
            problems.append(Problem("EXT-SMOKE", "--smoke-record", "a full run needs the area's PASS smoke record",
                                    "run the smoke profile first and pass its smoke.json"))
        else:
            try:
                smoke = json.loads(Path(smoke_record).read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                problems.append(Problem("EXT-SMOKE", "--smoke-record", f"cannot read {smoke_record}: "
                                        f"{exc.__class__.__name__}", "pass the smoke run's smoke.json"))
        if smoke is not None:
            # EXT-105: the smoke baseline is scored by the adapter's first (deciding) evaluator. R18-8: with `--spec`
            # naming a registration with several clauses, by every clause's evaluator.
            row = registry.EVALUATOR_ROWS[ad.evaluator_ids[0]]
            why = smoke_problems(smoke, adapter=adapter, score_range=row["score_range"], binary=row["binary"],
                                 direction=row.get("direction"),
                                 evaluators=_clause_evaluators(spec) if spec is not None else None,
                                 gpu_mem_limit_gb=gpu_mem_limit_gb(cfg), gpu_cap_h=gpu_cap_h(cfg))
            for msg in why:
                problems.append(Problem("EXT-SMOKE", "--smoke-record", msg,
                                        "fix the cause and repeat the smoke run; a FAIL stops the area (MM-39)"))
            if not why:
                est = smoke.get("projected_gpu_h")
                if req.get("gpus") is False:                                    # EXT-105 (MM-31)
                    if not (_num(est) and est >= 0):
                        problems.append(Problem("EXT-SMOKE", "--smoke-record", f"projected_gpu_h {est!r} is not a "
                                                "number >= 0 (CPU-only request, gpus false; MM-31)",
                                                "repeat the smoke run"))
                elif not (_num(est) and est > 0):                               # EXT-67
                    problems.append(Problem("EXT-SMOKE", "--smoke-record", f"projected_gpu_h {est!r} is not a "
                                            "number > 0", "repeat the smoke run"))
    if problems:
        raise Refused(problems)

    rt = req["runtime"]
    record = {
        "schema": RECORD_SCHEMA, "schema_version": SCHEMA_VERSION, "tags": list(TAGS), "adapter": adapter,
        "area": req["area"], "profile": profile, "label": req["label"], "host_work": req["host_work"],
        "runtime": {"kind": rt["kind"], "image": rt["image"],
                    "venv_lock_sha256": file_sha256(Path(rt["venv_lock"])) if rt["kind"] == "venv_in_container"
                    else None},
        "framework": req["framework"], "training_seed": req["training_seed"], "gpus": req["gpus"],
        "argv": req["argv"], "mounts": req["mounts"], "tmpfs": req["tmpfs"], "network": req["network"],
        "network_reason": req["network_reason"],
        "user": req["user"], "env_names": sorted(req["env"]), "gpu_est_h": est, "gpu_used_h": used,
        "output_dir": req["output_dir"], "output_root": cfg.output_root, "gpu_step_sha256": step_sha,
        "allow_network": bool(allow_network and req["network"] != "none"),                       # EXT-81
        "script_path": binding.script_path, "script_sha256": file_sha256(Path(binding.script_host)),   # EXT-83
        "host_config_sha256": host_config_sha256(cfg),                                            # EXT-85
        "interpreter_path": binding.interpreter_path,                                             # EXT-91
        "interpreter_sha256": (file_sha256(Path(binding.interpreter_host)) if binding.interpreter_host is not None
                               else None),                                                         # EXT-192
    }
    side = req.get("sidecar")
    if side is not None:                                 # EXT-190: checked by isolation_problems (sidecar_problems)
        ro = [m for m in side["mounts"] if m["mode"] == "ro"]
        image_py = (cfg.sidecar or {}).get("image_interpreter")     # EXT-249: pinned by the image ID, no lock
        record["sidecar"] = {
            "runtime": {"kind": side["runtime"]["kind"], "image": side["runtime"]["image"],
                        "venv_lock_sha256": None if image_py else file_sha256(Path(side["runtime"]["venv_lock"]))},
            "argv": side["argv"], "mounts": side["mounts"], "env_names": sorted(side["env"]),
            "network": side["network"], "user": side["user"], "tmpfs": side["tmpfs"], "read_only": side["read_only"],
            "cap_drop": side["cap_drop"], "security_opt": side["security_opt"], "gpus": side["gpus"],
            "limits": side["limits"],
            "interpreter_sha256": None if image_py else file_sha256(Path(container_file(side["argv"][0], ro)[1])),
            "script_sha256": file_sha256(Path(container_file(side["argv"][2], ro)[1]))}
    if cfg.rootless:                                     # EXT-263: only on a rootless host (rootful records unchanged)
        record["rootless"] = True
    record["label"] = launch_label(record)               # EXT-46: one label per launch record
    command = build_command(record, req["env"], est_h=est, gpu_cap_h=gpu_cap_h(cfg))
    if side is not None:                                 # EXT-190: sidecar first, trainer, then stop the sidecar
        start, stop = sidecar_commands(record["sidecar"], side["env"], cfg.sidecar, sidecar_name(record, cfg.sidecar))
        command = "\n".join((start, command, stop))
    created = False
    try:
        out_record.parent.mkdir(parents=True, exist_ok=True)
        with out_record.open("xb") as f:                 # never overwrite (checked above; race-safe)
            created = True
            f.write(canonical_json(record))
    except OSError as exc:                               # EXT-75: a failed write is a refusal
        if created:
            try:
                out_record.unlink()
            except OSError:
                pass
        raise Refused([Problem("EXT-WRITE", "--out-record", f"writing {out_record} failed "
                               f"({exc.__class__.__name__}: {exc})",
                               "fix the cause (disk space, permissions) and run ext-launch again")]) from exc
    return command


def recheck(*, adapter: str, record_path: Path, spec: Path | None = None, config: HostConfig | None = None) -> str:
    """EXT-69: `ext-launch --recheck <record>`. Re-validates a launch record against the filesystem now: every
    isolation rule (mount sources resolved again, the output directory still empty), the recorded output root,
    the gpu-step.sh checksum and log, the label checksum, and with `--spec` the base and export mounts. The
    operator runs it immediately before launch. The GPU total must match the log (EXT-82). Adapters that bind spec
    mounts require `--spec` (EXT-84). On a rootless host the owner's rootless daemon socket must be in place, by a
    stat only (EXT-263). Return the confirmation line or raise Refused (exit 3)."""
    cfg = adapter_config(host_config() if config is None else config, adapter)
    ad = _adapter(adapter)
    fix = "do not run the printed command; fix the cause and launch again"
    if spec is None and getattr(ad, "binds_spec", False):                 # EXT-84
        raise Refused([_spec_required(ad, adapter)])
    try:
        record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Refused([Problem("EXT-LAUNCH", "--recheck", f"cannot read {record_path}: {exc.__class__.__name__}",
                               "pass the launch.json ext-launch wrote")])
    problems = record_problems(record, adapter=adapter, stage_out=None, framework=ad.framework_name,
                               flag="--recheck", config=cfg, fresh_output=True)
    if isinstance(record, dict) and schema_version_problem(record):         # EXT-95: nothing else is judged
        raise Refused(problems)
    if isinstance(record, dict) and _abs(record.get("host_work")):
        where = gpu_step_log_problem(Path(record["host_work"]))
        if where:
            problems.append(Problem("EXT-GPU", "--recheck.host_work", where, fix))
    if spec is not None and isinstance(record, dict):
        problems += spec_binding_problems(spec, record.get("mounts"), ad, flag="--recheck", argv=record.get("argv"))
    if isinstance(record, dict) and hasattr(ad, "argv_problems"):         # EXT-133
        problems += [Problem("EXT-LAUNCH", "--recheck.argv", msg, fix)                                  # EXT-156
                     for msg in ad.argv_problems(record.get("argv"), profile=record.get("profile"))]
    if isinstance(record, dict):                                           # EXT-82
        log = Path(cfg.host_work) / GPU_LOG_MD
        try:
            now = gpu_used_h(log)
        except (OSError, ValueError) as exc:
            problems.append(Problem("EXT-GPU", "--recheck.gpu_used_h", f"cannot read the GPU-time log: {exc}", fix))
        else:
            if not (_num(record.get("gpu_used_h")) and record["gpu_used_h"] == now):
                problems.append(Problem("EXT-GPU", "--recheck.gpu_used_h",
                                        f"the record's gpu_used_h {record.get('gpu_used_h')!r} is not the running total "
                                        f"{_g(now)} of {log} now: GPU time was logged after ext-launch, so the printed "
                                        "GATEDTRAIN_GPU_USED_H is stale and gpu-step.sh would check the cap against it (EXT-82)",
                                        "launch again (new record, new output directory) so the command carries the "
                                        "current running total"))
    why = rootless_daemon_problem(cfg)                                     # EXT-263: stat only, no process
    if why:
        problems.append(Problem("EXT-LAUNCH", "--recheck.daemon", why,
                                "start the owner's rootless daemon (`systemctl --user start docker`) and run the "
                                "printed command through the context `gatedtrain`"))
    if problems:
        raise Refused(problems)
    return f"recheck: ok {record['label']}"
