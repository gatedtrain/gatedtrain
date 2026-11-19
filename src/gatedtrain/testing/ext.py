"""Fixture builders for external-run tests (MULTIMODAL-SPEC sections 5 and 6; tests/README.md section 11.18)."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

from .support import canonical_bytes

IMAGE_ID = "sha256:" + "ab" * 32
# The external-run host layout overrides ext-launch and ext-collect read (EXT-63, EXT-66, EXT-67, EXT-79).
ENV_HOST_WORK, ENV_OUTPUT_ROOT, ENV_INPUT_ROOTS = ("GATEDTRAIN_EXT_HOST_WORK", "GATEDTRAIN_EXT_OUTPUT_ROOT",
                                                   "GATEDTRAIN_EXT_INPUT_ROOTS")
# Fixture overrides for the identity mount root and interpreter paths (EXT-112, EXT-113).
ENV_MIRROR_ROOT, ENV_INTERPRETER, ENV_INTERPRETER_TREES = ("GATEDTRAIN_EXT_MIRROR_ROOT", "GATEDTRAIN_EXT_INTERPRETER",
                                                           "GATEDTRAIN_EXT_INTERPRETER_TREES")


def _jcs_ready(obj: Any) -> Any:
    """Integral floats as integers, as RFC 8785 prints them (1.0 -> 1); other floats keep repr."""
    if isinstance(obj, float) and math.isfinite(obj) and obj == int(obj):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _jcs_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jcs_ready(v) for v in obj]
    return obj


def jcs(obj: Any) -> bytes:
    return canonical_bytes(_jcs_ready(obj))


def sha_of(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(jcs(obj)).hexdigest()


def write_canonical(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(jcs(obj))
    return path


def launch_label(record: dict[str, Any]) -> str:
    """EXT-46: the gpu-step label ext-launch writes, `<request label>-<first 12 hex of SHA-256(JCS(record with
    label = the request label))>`; `record["label"]` holds the request label."""
    return f"{record['label']}-{hashlib.sha256(jcs(record)).hexdigest()[:12]}"


def host_config_sha256(host_work: Path | str, output_root: Path | str, input_roots: list, *,
                       mirror_root: Path | str | None = None, interpreter: str | None = None,
                       interpreter_trees: list = (), extra_daemon_configs: list = (),
                       extra_daemon_sockets: list = ()) -> str:
    """EXT-85, EXT-117: SHA-256 of the canonical JSON {host_work, output_root, input_roots, mirror_root, interpreter,
    interpreter_trees} of the effective host layout (null mirror root: `/`), plus the host file's non-empty
    `extra_daemon_configs` and `extra_daemon_sockets`."""
    doc = {"host_work": str(host_work), "output_root": str(output_root),
           "input_roots": [str(r) for r in input_roots],
           "mirror_root": None if mirror_root is None else str(mirror_root), "interpreter": interpreter,
           "interpreter_trees": [str(t) for t in interpreter_trees]}
    for k, v in (("extra_daemon_configs", extra_daemon_configs), ("extra_daemon_sockets", extra_daemon_sockets)):
        if v:
            doc[k] = [str(x) for x in v]
    return sha_of(doc)


def gpu_step_script(host_work: Path, *, work: Path | str | None = None) -> Path:
    """<HOST_WORK>/bin/gpu-step.sh as far as ext-collect reads it (EXT-46): the `WORK=` and `LOG=` lines of the
    real script. Never executed."""
    p = host_work / "bin" / "gpu-step.sh"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("#!/usr/bin/env bash\n# fixture stand-in; never executed\nset -u\n\n"
                 f"WORK={work if work is not None else host_work}\n"
                 'LOG="$WORK/gpu-guard.log"\nexit 99\n', encoding="utf-8")
    return p


def safetensors_bytes(tag: str) -> bytes:
    """A minimal valid safetensors file: 8-byte little-endian header length, JSON header, data (one F32[2])."""
    header = json.dumps({"__metadata__": {"format": "pt"}, "w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}},
                        separators=(",", ":")).encode("utf-8")
    header += b" " * ((8 - len(header) % 8) % 8)
    return struct.pack("<Q", len(header)) + header + hashlib.sha256(tag.encode()).digest()[:8]
