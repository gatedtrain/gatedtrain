"""Keeper serve loop and op dispatch (AN-12).

`serve` binds the Unix socket (mode 0666: the `SO_PEERCRED` uid check is the gate, not the file mode) and handles one
connection at a time, one request per connection. The peer's uid is checked before any byte is read; a refused peer
gets one problem line and the connection closes. Each request must arrive whole within `SOCKET_TIMEOUT_S` and within
`MAX_REQUEST_BYTES`. Nothing a peer sends is logged. `SIGTERM` stops the loop, closes and unlinks the socket.

`OPS` maps an op name to its handler `(keeper, project, args, peer_uid) -> result`; this module registers `hello`,
later modules register the state and attempt ops.
"""

from __future__ import annotations

import os
import signal
import socket
import stat
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from ... import __version__
from ...errors import Problem, Refused
from . import protocol

_ACCEPT_POLL_S = 0.2             # how often the idle loop checks for a stop request


class Keeper:
    def __init__(self, state_dir: Path, owner_uid: int) -> None:
        self.state_dir = Path(state_dir)
        self.owner_uid = owner_uid
        self.keeper_uid = os.geteuid()

    def handle(self, line: bytes, peer: int) -> dict:
        """The response to one request line. Never raises: an unexpected exception becomes a problem with rule
        `INTERNAL` that names the exception class only (its message may hold a sealed input)."""
        try:
            op, project, args = protocol.parse_request(line)
            handler = OPS.get(op)
            if handler is None:
                raise protocol.refused("op", "unknown op", "send `hello` to see the keeper version; check the "
                                       "client and the keeper are the same gatedtrain version")
            return protocol.ok_response(handler(self, project, args, peer))
        except Refused as exc:
            return protocol.problem_response(exc.problems)
        except Exception as exc:
            return protocol.problem_response([_internal(exc)])


def _internal(exc: BaseException) -> Problem:
    return Problem("INTERNAL", "op", f"unexpected {type(exc).__name__}; this is a bug", "report it")


def _hello(keeper: Keeper, project: str | None, args: dict, peer: int) -> dict:
    return {
        "keeper_version": __version__,
        "fingerprint": None,                 # the signing key (Task 8)
        "public_pem": None,
        "sealed_guarantee": None,
        "environment_sha256": None,          # the environment record (Task 7)
        "same_user": keeper.keeper_uid == keeper.owner_uid,
    }


OPS: dict[str, Callable[[Keeper, str | None, dict, int], object]] = {"hello": _hello}


# ---- transport ----------------------------------------------------------------------------------------------------

def _send(conn: socket.socket, data: bytes) -> None:
    try:
        conn.sendall(data)
    except OSError:                          # the peer left or stopped reading: nothing more to do for it
        pass


def _send_problem(conn: socket.socket, problem: Problem) -> None:
    _send(conn, protocol.encode_response(protocol.problem_response([problem])))


def _respond(keeper: Keeper, line: bytes, peer: int) -> bytes:
    resp = keeper.handle(line, peer)
    try:
        data = protocol.encode_response(resp)
    except (TypeError, ValueError, RecursionError) as exc:      # a handler result that is not JSON
        return protocol.encode_response(protocol.problem_response([_internal(exc)]))
    if len(data) - 1 > protocol.MAX_RESPONSE_BYTES:
        return protocol.encode_response(protocol.problem_response([Problem(
            protocol.RULE, "response", f"the response is longer than the cap of {protocol.MAX_RESPONSE_BYTES} bytes",
            "report it; the keeper never sends a response over the cap")]))
    return data


def _serve_connection(keeper: Keeper, conn: socket.socket) -> None:
    timeout = protocol.SOCKET_TIMEOUT_S
    conn.settimeout(timeout)
    deadline = time.monotonic() + timeout
    try:
        uid = protocol.peer_uid(conn)
    except OSError:
        return
    if not protocol.peer_allowed(uid, owner_uid=keeper.owner_uid, keeper_uid=keeper.keeper_uid):
        _send_problem(conn, Problem(protocol.RULE, "peer", "the connecting uid is neither the owner uid nor the "
                                    "keeper's uid", "connect as the user the keeper was set up for"))
        return
    try:
        line = protocol.read_line(conn, protocol.MAX_REQUEST_BYTES, deadline=deadline)
    except Refused as exc:                   # over the cap, or EOF before the newline
        _send(conn, protocol.encode_response(protocol.problem_response(exc.problems)))
        return
    except TimeoutError:
        _send_problem(conn, Problem(protocol.RULE, "request", f"no complete request line within {timeout} s",
                                    "send the whole request at once, followed by a newline"))
        return
    except OSError:
        return
    conn.settimeout(timeout)                 # the reply gets its own full timeout
    _send(conn, _respond(keeper, line, uid))


def _socket_is_live(path: Path) -> bool:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
        probe.settimeout(1.0)
        try:
            probe.connect(str(path))
        except OSError:
            return False
    return True


def _prepare_socket_path(socket_path: Path) -> None:
    """Refuse a path that is too long or holds something other than a socket; remove a stale socket (left by a
    keeper that crashed), but never one a live keeper still listens on."""
    if len(os.fsencode(socket_path)) > protocol.MAX_SOCKET_PATH:
        raise protocol.refused("socket", f"the socket path is longer than {protocol.MAX_SOCKET_PATH} bytes",
                               "use a shorter path, such as a directory under /run")
    try:
        st = os.lstat(socket_path)
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(st.st_mode):
        raise protocol.refused("socket", "the socket path exists and is not a socket",
                               "remove it or choose another socket path")
    if _socket_is_live(socket_path):
        raise protocol.refused("socket", "another keeper is serving on the socket path",
                               "stop that keeper first or choose another socket path")
    os.unlink(socket_path)


def serve(state_dir: Path, socket_path: Path, owner_uid: int, *, stop: threading.Event | None = None) -> int:
    """Serve until SIGTERM (or until `stop` is set, for an in-process server); 0 on a clean stop. A start that cannot
    proceed raises Refused (AN-12)."""
    if sys.platform != "linux":
        raise protocol.refused("platform", "the keeper runs on Linux only (Unix socket, SO_PEERCRED)",
                               "run the keeper on a Linux host")
    socket_path = Path(socket_path)
    _prepare_socket_path(socket_path)
    stop = stop if stop is not None else threading.Event()
    keeper = Keeper(state_dir, owner_uid)

    previous = None
    if threading.current_thread() is threading.main_thread():
        previous = signal.signal(signal.SIGTERM, lambda _sig, _frame: stop.set())
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    bound = None
    try:
        try:
            listener.bind(str(socket_path))
        except OSError as exc:
            raise protocol.refused("socket", f"cannot bind the socket ({type(exc).__name__})",
                                   "create the socket's directory, writable by the keeper user") from None
        bound = os.lstat(socket_path)
        os.chmod(socket_path, 0o666)
        listener.listen(16)
        listener.settimeout(_ACCEPT_POLL_S)
        while not stop.is_set():
            try:
                conn, _ = listener.accept()
            except TimeoutError:
                continue
            except InterruptedError:
                continue
            with conn:
                try:
                    _serve_connection(keeper, conn)
                except Exception:            # one bad connection never stops the keeper; nothing it sent is logged
                    pass
    finally:
        listener.close()
        if bound is not None:
            try:
                st = os.lstat(socket_path)
                if (st.st_dev, st.st_ino) == (bound.st_dev, bound.st_ino):
                    os.unlink(socket_path)
            except OSError:
                pass
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
    return 0
