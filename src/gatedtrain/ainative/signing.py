"""AN-14: Ed25519 signatures on trust records.

Every record the keeper or the driver hands across a trust boundary travels in an envelope
`{schema, schema_version, kind, key_fingerprint, record, signature}`. The signed bytes are the RFC 8785 canonical JSON
of `{"kind", "key_fingerprint", "record"}`, so key order and number form never change a signature, and a record
cannot be replayed under another kind or another key. The key fingerprint is `sha256:` plus the SHA-256 of the raw
32-byte public key. `verify` raises `Refused` with rule AN-14 on every deviation and returns the record otherwise.

The private key file is a PKCS8 PEM without a passphrase, created exclusively with mode 0600; loading refuses a file
that any group or other bit can reach.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
import stat
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from ..canonical import canonical_json, sha256_bytes
from ..errors import Problem, Refused

SIGNED_SCHEMA = "gatedtrain.ainative.signed"
SCHEMA_VERSION = 1
KINDS = ("registration", "metric_confirmation", "claim", "ledger_entry", "gate_record", "gate_summary", "report",
         "environment")

_ENVELOPE_KEYS = frozenset({"schema", "schema_version", "kind", "key_fingerprint", "record", "signature"})
_FP_PREFIX = "sha256:"


def _refuse(field: str, message: str, fix: str) -> Refused:
    return Refused([Problem("AN-14", field, message, fix)])


def generate_key(path: Path) -> None:
    """Write a new unencrypted PKCS8 PEM private key; exclusive create, mode 0600, never over a file or a link."""
    pem = Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError:
        raise _refuse("key file", f"{path} already exists",
                      "choose a new path; a key is never overwritten") from None
    except OSError as exc:
        raise _refuse("key file", f"cannot create {path}: {exc.strerror}",
                      "use a writable path in a directory you own") from None
    with os.fdopen(fd, "wb") as handle:
        handle.write(pem)
    os.chmod(path, 0o600)


def load_private_key(path: Path) -> Ed25519PrivateKey:
    """Load a private key file; refuses a missing file, any group or other mode bit, or a key that is not Ed25519."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise _refuse("key file", f"cannot open {path}: {exc.strerror}",
                      "pass the path of the keeper key file") from None
    with os.fdopen(fd, "rb") as handle:
        mode = stat.S_IMODE(os.fstat(handle.fileno()).st_mode)
        if mode & 0o077:
            raise _refuse("key file", f"{path} has mode {mode:04o}; group and other bits must be clear",
                          f"run chmod 600 {path}")
        data = handle.read()
    try:
        key = serialization.load_pem_private_key(data, password=None)
    except (ValueError, TypeError):
        raise _refuse("key file", f"{path} is not an unencrypted PEM private key",
                      "generate a key with the keeper tooling") from None
    if not isinstance(key, Ed25519PrivateKey):
        raise _refuse("key file", f"{path} is not an Ed25519 key", "generate a key with the keeper tooling")
    return key


def public_pem(key: Ed25519PrivateKey | Ed25519PublicKey) -> bytes:
    """SubjectPublicKeyInfo PEM of the public half."""
    public = key.public_key() if isinstance(key, Ed25519PrivateKey) else key
    return public.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)


def load_public_pem(pem: bytes) -> Ed25519PublicKey:
    try:
        key = serialization.load_pem_public_key(pem)
    except (ValueError, TypeError):
        raise _refuse("public key", "not a PEM public key", "pass the SubjectPublicKeyInfo PEM of the key") from None
    if not isinstance(key, Ed25519PublicKey):
        raise _refuse("public key", "not an Ed25519 public key", "pass an Ed25519 public key")
    return key


def fingerprint(key: Ed25519PrivateKey | Ed25519PublicKey) -> str:
    """`sha256:<64 hex>` of the raw 32-byte public key."""
    public = key.public_key() if isinstance(key, Ed25519PrivateKey) else key
    raw = public.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return _FP_PREFIX + hashlib.sha256(raw).hexdigest()


def _signed_bytes(kind: str, key_fp: str, record: Any) -> bytes:
    try:
        return canonical_json({"kind": kind, "key_fingerprint": key_fp, "record": record})
    except (ValueError, TypeError, RecursionError) as exc:
        raise _refuse("record", f"record is not canonical JSON: {exc}",
                      "use strings, booleans, null, finite numbers, lists and string-keyed objects") from None


def sign(key: Ed25519PrivateKey, kind: str, record: dict) -> dict:
    if kind not in KINDS:
        raise _refuse("kind", f"unknown record kind {kind!r}", f"use one of: {', '.join(KINDS)}")
    if not isinstance(record, dict):
        raise _refuse("record", "a signed record must be a JSON object", "pass a dict")
    key_fp = fingerprint(key)
    signature = key.sign(_signed_bytes(kind, key_fp, record))
    return {"schema": SIGNED_SCHEMA, "schema_version": SCHEMA_VERSION, "kind": kind, "key_fingerprint": key_fp,
            "record": record, "signature": base64.b64encode(signature).decode("ascii")}


def verify(envelope: object, public: Ed25519PublicKey, *, kind: str | None = None) -> dict:
    """Return the record of a valid envelope; raise `Refused` (AN-14) on anything else."""
    if not isinstance(envelope, dict) or set(envelope) != _ENVELOPE_KEYS:
        raise _refuse("envelope", "not a signed envelope (missing or extra keys)",
                      "the record was altered or is not a signed record; reject it")
    if envelope["schema"] != SIGNED_SCHEMA or envelope["schema_version"] != SCHEMA_VERSION \
            or isinstance(envelope["schema_version"], bool):
        raise _refuse("schema", "unsupported signed-record schema or version",
                      f"expected {SIGNED_SCHEMA} version {SCHEMA_VERSION}")
    env_kind = envelope["kind"]
    if env_kind not in KINDS:
        raise _refuse("kind", "unknown record kind", f"use one of: {', '.join(KINDS)}")
    if kind is not None and env_kind != kind:
        raise _refuse("kind", f"record kind is {env_kind!r}, expected {kind!r}",
                      "a record of one kind cannot stand in for another")
    key_fp = envelope["key_fingerprint"]
    if key_fp != fingerprint(public):
        raise _refuse("key_fingerprint", "the record was signed by another key",
                      "verify against the key whose fingerprint the record names")
    record = envelope["record"]
    if not isinstance(record, dict):
        raise _refuse("record", "a signed record must be a JSON object", "reject the record")
    body = _signed_bytes(env_kind, key_fp, record)
    sig_text = envelope["signature"]
    try:
        if not isinstance(sig_text, str):
            raise ValueError("signature is not text")
        signature = base64.b64decode(sig_text.encode("ascii"), validate=True)
        # One canonical text per signature: reject unused trailing bits and wrong lengths (malleability).
        if len(signature) != 64 or base64.b64encode(signature).decode("ascii") != sig_text:
            raise ValueError("signature is not canonical base64 of 64 bytes")
        public.verify(signature, body)
    except (ValueError, binascii.Error, InvalidSignature):
        raise _refuse("signature", "the signature does not match the record",
                      "the record was altered or signed by another key; reject it") from None
    return record


def envelope_sha256(envelope: dict) -> str:
    return sha256_bytes(canonical_json(envelope))
