"""Keeper wire protocol (AN-12): JSON lines over a Unix socket, with the peer checked by `SO_PEERCRED`.

Request line: `{"op": str, "project": "sha256:<64 hex>" | null, "args": object}`, nothing else. Response line:
`{"ok": bool, "result": object | null, "problems": [{rule, field, message, fix}]}` (the AN-1 problem shape). One
request per connection. Lines are capped while they are read, and a refusal never echoes what the peer sent.

A refusal after the keeper granted alpha (a sealed evaluation whose scoring failed) still carries the signed claim
and ledger envelopes in `result` alongside `ok: false` (preflight finding 9); the client raises `RefusedAfterSpend`
holding them, so the driver logs the spend.

Client deadlines: the keeper serves one connection at a time, and a client's deadline starts before it connects. The
ops that split, score or gate (`LONG_OPS`: intake of up to 64 MiB, evaluate, gate, report) get `LONG_OP_TIMEOUT_S`,
longer than any request the keeper is expected to spend on the synthetic pack and on a CPU scorer; every other op
keeps `SOCKET_TIMEOUT_S`. A client that gives up while
its request waits behind another is not served: the keeper drops a request whose peer has already closed. A client
that gives up during its own evaluation can send the identical request again and gets the stored claim.

The server reads the caps and the timeout from this module at call time, so a test can patch them for an in-process
server.
"""

from __future__ import annotations

import json
import re
import socket
import struct
import time
from dataclasses import asdict
from pathlib import Path

from ...errors import Problem, Refused

MAX_REQUEST_BYTES = 96 * 1024 * 1024          # a 64 MiB data file in base64 plus the envelope
MAX_RESPONSE_BYTES = 96 * 1024 * 1024
SOCKET_TIMEOUT_S = 60.0
LONG_OPS = frozenset({"intake", "evaluate", "gate", "report"})
LONG_OP_TIMEOUT_S = 3600.0
MAX_SOCKET_PATH = 107                         # sun_path is 108 bytes including the terminating NUL

RULE = "AN-12"
_PROJECT_ID = re.compile(r"sha256:[0-9a-f]{64}")
_REQUEST_KEYS = frozenset({"op", "project", "args"})
_RESPONSE_KEYS = frozenset({"ok", "result", "problems"})
_PROBLEM_KEYS = ("rule", "field", "message", "fix")
_RECV_CHUNK = 1 << 16


def refused(field: str, message: str, fix: str) -> Refused:
    return Refused([Problem(RULE, field, message, fix)])


class RefusedAfterSpend(Refused):
    """A refusal after the keeper granted an attempt: `result` holds the signed envelopes of that spend."""

    def __init__(self, problems: list[Problem], result: dict) -> None:
        super().__init__(problems)
        self.result = result


def read_line(sock: socket.socket, cap: int, *, deadline: float | None = None) -> bytes:
    """One line without its `\\n`. Refused (AN-12) as soon as more than `cap` bytes precede the newline (never buffers
    more than `cap + 1` bytes) or at EOF before the newline. With `deadline` (a `time.monotonic()` value) the whole
    line must arrive by then, else `TimeoutError`; without it the socket's own timeout applies to each read."""
    buf = bytearray()
    while True:
        if deadline is not None:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("the line did not arrive in time")
            sock.settimeout(left)
        chunk = sock.recv(min(_RECV_CHUNK, cap + 1 - len(buf)))
        if not chunk:
            raise refused("request", "the connection closed before the end of the line",
                          "send one JSON object followed by a newline")
        nl = chunk.find(b"\n")
        buf += chunk if nl < 0 else chunk[:nl]
        if len(buf) > cap:
            raise refused("request", f"the line is longer than the cap of {cap} bytes",
                          "send a smaller request (a data file of at most 64 MiB)")
        if nl >= 0:
            return bytes(buf)


def peer_uid(conn: socket.socket) -> int:
    """The uid of the process at the other end of a connected Unix socket (Linux `SO_PEERCRED`: pid, uid, gid)."""
    size = struct.calcsize("3i")
    _pid, uid, _gid = struct.unpack("3i", conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, size))
    return uid


def peer_allowed(uid: int, *, owner_uid: int, keeper_uid: int) -> bool:
    return uid == owner_uid or uid == keeper_uid


# ---- lines --------------------------------------------------------------------------------------------------------

def _no_constant(_name: str):
    raise ValueError("not JSON")


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    doc = dict(pairs)
    if len(doc) != len(pairs):
        raise ValueError("duplicate key")
    return doc


def _loads(line: bytes) -> object:
    return json.loads(line.decode("utf-8"), object_pairs_hook=_unique_keys, parse_constant=_no_constant)


def _dumps(doc: dict) -> bytes:
    return json.dumps(doc, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"


def parse_request(line: bytes) -> tuple[str, str | None, dict]:
    """(op, project, args) of a request line, or Refused (AN-12) naming the field, never its content."""
    try:
        doc = _loads(line)
    except (ValueError, RecursionError):
        raise refused("request", "the line is not one JSON object (UTF-8, no duplicate keys, no NaN)",
                      'send {"op": ..., "project": ..., "args": {...}} on one line') from None
    if not isinstance(doc, dict):
        raise refused("request", "the request is not a JSON object", 'send {"op": ..., "project": ..., "args": {...}}')
    if set(doc) - _REQUEST_KEYS:
        raise refused("request", "the request has keys other than op, project and args", "send only those keys")
    op, project, args = doc.get("op"), doc.get("project", ...), doc.get("args")
    if not isinstance(op, str):
        raise refused("op", "op is missing or not a string", "name the operation, for example `hello`")
    if project is ... or not (project is None or (isinstance(project, str) and _PROJECT_ID.fullmatch(project))):
        raise refused("project", 'project is missing or is neither null nor "sha256:<64 lowercase hex>"',
                      "send the project id the driver computes, or null for an op outside a project")
    if not isinstance(args, dict):
        raise refused("args", "args is missing or not a JSON object", "send args as an object, {} when empty")
    return op, project, args


def encode_request(op: str, project: str | None, args: dict) -> bytes:
    return _dumps({"op": op, "project": project, "args": args})


def ok_response(result: object) -> dict:
    return {"ok": True, "result": result, "problems": []}


def problem_response(problems: list[Problem], result: dict | None = None) -> dict:
    return {"ok": False, "result": result, "problems": [asdict(p) for p in problems]}


def encode_response(resp: dict) -> bytes:
    return _dumps(resp)


def parse_response(line: bytes) -> dict:
    """A response line checked for its shape; Refused (AN-12) when it is not a keeper response."""
    bad = refused("response", "the keeper sent a malformed response", "check that the socket is a gatedtrain keeper's")
    try:
        doc = _loads(line)
    except (ValueError, RecursionError):
        raise bad from None
    if not isinstance(doc, dict) or set(doc) != _RESPONSE_KEYS or not isinstance(doc["ok"], bool):
        raise bad
    problems = doc["problems"]
    if not isinstance(problems, list) or not all(
            isinstance(p, dict) and set(p) == set(_PROBLEM_KEYS) and all(isinstance(p[k], str) for k in _PROBLEM_KEYS)
            for p in problems):
        raise bad
    if not doc["ok"] and (not problems or not (doc["result"] is None or isinstance(doc["result"], dict))):
        raise bad
    return doc


# ---- client -------------------------------------------------------------------------------------------------------

class KeeperClient:
    """One connection per call. Every failure is `Refused` (AN-12), never a traceback. Without an explicit `timeout`
    an op in `LONG_OPS` waits `LONG_OP_TIMEOUT_S` and any other op `SOCKET_TIMEOUT_S`."""

    def __init__(self, socket_path: Path, *, timeout: float | None = None) -> None:
        self.socket_path = Path(socket_path)
        self.timeout = timeout

    def timeout_for(self, op: str) -> float:
        if self.timeout is not None:
            return self.timeout
        return LONG_OP_TIMEOUT_S if op in LONG_OPS else SOCKET_TIMEOUT_S

    def call(self, op: str, project: str | None, args: dict) -> object:
        line = encode_request(op, project, args)
        if len(line) - 1 > MAX_REQUEST_BYTES:
            raise refused("request", f"the request is longer than the cap of {MAX_REQUEST_BYTES} bytes",
                          "send a smaller request (a data file of at most 64 MiB)")
        timeout = self.timeout_for(op)
        deadline = time.monotonic() + timeout
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            try:
                sock.connect(str(self.socket_path))
            except (OSError, ValueError):
                raise refused("socket", "the keeper is not reachable at the socket",
                              "start it (`gatedtrain keeper serve`, see `gatedtrain setup keeper`) or pass its "
                              "socket path") from None
            try:
                try:
                    sock.sendall(line)
                except (BrokenPipeError, ConnectionResetError):
                    pass                     # the keeper may refuse (a foreign peer) before reading: read its answer
                raw = read_line(sock, MAX_RESPONSE_BYTES, deadline=deadline)
            except Refused:
                raise refused("response", "the keeper closed the connection without a complete response",
                              "check the keeper's state; retry the call") from None
            except TimeoutError:
                raise refused("response", f"the keeper did not answer within {timeout:g} s",
                              "the keeper serves one request at a time and may be busy with a sealed evaluation; "
                              "check that it is running, then retry the call (an identical evaluate returns the "
                              "stored claim)") from None
            except OSError:
                raise refused("response", "the connection to the keeper failed",
                              "check that the keeper is running; retry the call") from None
        doc = parse_response(raw)
        if not doc["ok"]:
            problems = [Problem(*(p[k] for k in _PROBLEM_KEYS)) for p in doc["problems"]]
            if doc["result"] is not None:
                raise RefusedAfterSpend(problems, doc["result"])
            raise Refused(problems)
        return doc["result"]
