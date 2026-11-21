"""AN-12: keeper protocol, transport and peer credentials (tests/README.md section 13).

One JSON request per line, one JSON response line with problems in the AN-1 shape; only the owner uid or the keeper's
own uid may connect; a line over the cap, non-JSON, a JSON array, an unknown op or a silent peer gets one problem
response or a closed connection and the keeper keeps serving; an unreachable socket is a refusal without a traceback.

Cases that need small caps or short timeouts run against an in-process `serve` thread (a patched module constant does
not reach a subprocess); the rest run against the keeper subprocess of `tests/_keeper.py`.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import socket
import stat
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from _keeper import KeeperProcess, skip_unless_linux

from gatedtrain.ainative.keeper import protocol, server
from gatedtrain.ainative.keeper.protocol import (MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES, MAX_SOCKET_PATH,
                                                 SOCKET_TIMEOUT_S, KeeperClient, peer_allowed, read_line)
from gatedtrain.errors import Refused

PROBLEM_KEYS = {"rule", "field", "message", "fix"}


def raw_roundtrip(path: Path, raw: bytes, *, timeout: float = 10.0) -> dict | None:
    """Send `raw` as is and read one response line; None when the keeper closed the connection without one."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(str(path))
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            s.sendall(raw)
        buf = b""
        while not buf.endswith(b"\n"):
            try:
                chunk = s.recv(65536)
            except ConnectionResetError:
                break
            if not chunk:
                break
            buf += chunk
    if not buf.endswith(b"\n"):
        return None
    return json.loads(buf)


@contextlib.contextmanager
def serving(**caps):
    """An in-process keeper (`serve` in a thread) on a fresh socket, with `protocol` constants patched by `caps`."""
    tmp = Path(tempfile.mkdtemp(prefix="gk-"))
    state, sock = tmp / "state", tmp / "keeper.sock"
    state.mkdir(mode=0o700)
    stop = threading.Event()
    errors: list[BaseException] = []

    def run():
        try:
            server.serve(state, sock, os.getuid(), stop=stop)
        except BaseException as exc:                        # surfaced by the context below
            errors.append(exc)

    with contextlib.ExitStack() as stack:
        stack.callback(shutil.rmtree, tmp, True)
        for name, value in caps.items():
            stack.enter_context(mock.patch.object(protocol, name, value))
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 10
        while True:                                         # the socket file exists before `listen`: ask hello
            if errors or time.monotonic() > deadline:
                stop.set()
                raise AssertionError(f"the in-process keeper did not start: {errors!r}")
            if sock.exists():
                with contextlib.suppress(Refused):
                    KeeperClient(sock, timeout=2).call("hello", None, {})
                    break
            time.sleep(0.01)
        try:
            yield sock
        finally:
            stop.set()
            thread.join(10)
            if thread.is_alive():
                raise AssertionError("the in-process keeper did not stop")
            if errors:
                raise errors[0]


def problem_of(resp: dict) -> dict:
    assert set(resp) == {"ok", "result", "problems"}, resp
    assert resp["ok"] is False and resp["result"] is None, resp
    assert len(resp["problems"]) == 1 and set(resp["problems"][0]) == PROBLEM_KEYS, resp
    return resp["problems"][0]


class TestConstants(unittest.TestCase):
    def test_values(self):
        self.assertEqual(MAX_REQUEST_BYTES, 96 * 1024 * 1024)
        self.assertEqual(MAX_RESPONSE_BYTES, 96 * 1024 * 1024)
        self.assertEqual(SOCKET_TIMEOUT_S, 60.0)
        self.assertEqual(MAX_SOCKET_PATH, 107)


class TestPeerAllowed(unittest.TestCase):
    def test_peer_allowed(self):
        self.assertTrue(peer_allowed(1000, owner_uid=1000, keeper_uid=999))
        self.assertTrue(peer_allowed(999, owner_uid=1000, keeper_uid=999))
        self.assertFalse(peer_allowed(1001, owner_uid=1000, keeper_uid=999))
        self.assertFalse(peer_allowed(0, owner_uid=1000, keeper_uid=999))


class TestReadLine(unittest.TestCase):
    def pair(self):
        a, b = socket.socketpair()
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        return a, b

    def test_line_at_the_cap_is_read(self):
        a, b = self.pair()
        a.sendall(b"x" * 10 + b"\n")
        self.assertEqual(read_line(b, 10), b"x" * 10)

    def test_line_over_the_cap_refused(self):
        a, b = self.pair()
        a.sendall(b"x" * 11 + b"\n")
        with self.assertRaises(Refused) as cm:
            read_line(b, 10)
        self.assertEqual(cm.exception.problems[0].rule, "AN-12")

    def test_over_the_cap_refused_before_the_newline_arrives(self):
        a, b = self.pair()
        a.sendall(b"x" * 11)                                 # no newline yet: the reader must not wait for one
        b.settimeout(5)
        with self.assertRaises(Refused):
            read_line(b, 10)

    def test_eof_before_newline_refused(self):
        a, b = self.pair()
        a.sendall(b"{}")
        a.shutdown(socket.SHUT_WR)
        with self.assertRaises(Refused) as cm:
            read_line(b, 10)
        self.assertEqual(cm.exception.problems[0].rule, "AN-12")


class TestHandle(unittest.TestCase):
    """`Keeper.handle` on lines, without a socket: every refusal is AN-12 and never echoes the request."""

    def setUp(self):
        tmp = Path(tempfile.mkdtemp(prefix="gk-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        self.keeper = server.Keeper(tmp, os.getuid())

    def test_hello_result(self):
        resp = self.keeper.handle(b'{"op":"hello","project":null,"args":{}}', os.getuid())
        self.assertEqual(set(resp), {"ok", "result", "problems"})
        self.assertIs(resp["ok"], True)
        self.assertEqual(resp["problems"], [])
        self.assertEqual(set(resp["result"]), {"keeper_version", "fingerprint", "public_pem", "sealed_guarantee",
                                               "environment_sha256", "same_user"})
        self.assertIs(resp["result"]["same_user"], True)       # the test keeper runs as the owner uid

    def test_bad_lines(self):
        cases = {
            b"not json": "request",
            b"\xff\xfe": "request",
            b"[1]": "request",
            b'{"op":"hello","project":null,"args":{},"op":"hello"}': "request",   # duplicate key
            b'{"op":"hello","project":null,"args":{},"extra":1}': "request",
            b'{"op":"hello","args":{}}': "project",
            b'{"op":"hello"}': "project",
            b'{"op":"hello","project":"sha256:SECRET","args":{}}': "project",
            b'{"op":"hello","project":null,"args":[]}': "args",
            b'{"op":1,"project":null,"args":{}}': "op",
            b'{"op":"nope","project":null,"args":{}}': "op",    # unknown op, nothing else wrong
            b'{"op":"hello","project":null,"args":{"x":NaN}}': "request",
        }
        for line, field in cases.items():
            with self.subTest(line=line):
                p = problem_of(self.keeper.handle(line, os.getuid()))
                self.assertEqual((p["rule"], p["field"]), ("AN-12", field))
                self.assertNotIn("SECRET", json.dumps(p))
                self.assertNotIn("nope", json.dumps(p))

    def test_project_id_accepted(self):
        resp = self.keeper.handle(json.dumps({"op": "hello", "project": "sha256:" + "a" * 64, "args": {}}).encode(),
                                  os.getuid())
        self.assertIs(resp["ok"], True, resp)

    def test_unexpected_exception_is_internal_with_class_name_only(self):
        def boom(keeper, project, args, peer):
            raise ValueError("sealed item: the secret label")

        with mock.patch.dict(server.OPS, {"boom": boom}):
            p = problem_of(self.keeper.handle(b'{"op":"boom","project":null,"args":{}}', os.getuid()))
        self.assertEqual(p["rule"], "INTERNAL")
        self.assertIn("ValueError", p["message"])
        self.assertNotIn("secret", json.dumps(p))

    def test_handler_refusal_is_passed_through(self):
        from gatedtrain.errors import Problem

        def refuse(keeper, project, args, peer):
            raise Refused([Problem("AN-16", "data", "two rows were sealed before", "use new data")])

        with mock.patch.dict(server.OPS, {"refuse": refuse}):
            p = problem_of(self.keeper.handle(b'{"op":"refuse","project":null,"args":{}}', os.getuid()))
        self.assertEqual((p["rule"], p["field"]), ("AN-16", "data"))

    def test_handler_receives_project_args_and_peer(self):
        seen = []
        with mock.patch.dict(server.OPS, {"echo": lambda k, pr, a, peer: seen.append((k, pr, a, peer)) or {"n": 1}}):
            resp = self.keeper.handle(json.dumps({"op": "echo", "project": "sha256:" + "b" * 64,
                                                  "args": {"k": [1]}}).encode(), 4242)
        self.assertEqual(resp, {"ok": True, "result": {"n": 1}, "problems": []})
        self.assertEqual(seen, [(self.keeper, "sha256:" + "b" * 64, {"k": [1]}, 4242)])


@skip_unless_linux
class TestInProcessServer(unittest.TestCase):
    def test_bad_requests_keep_the_keeper_serving(self):        # Review Focus 5
        cap = 1024
        with serving(MAX_REQUEST_BYTES=cap, SOCKET_TIMEOUT_S=5.0) as sock:
            over = raw_roundtrip(sock, b"x" * (cap + 1) + b"\n")
            if over is not None:                               # a closed connection is acceptable too
                self.assertEqual(problem_of(over)["rule"], "AN-12")
            over_no_newline = raw_roundtrip(sock, b"{" * (8 * cap))
            if over_no_newline is not None:
                self.assertEqual(problem_of(over_no_newline)["rule"], "AN-12")
            for raw, field in ((b"not json\n", "request"), (b"[1]\n", "request"),
                               (b'{"op":"nope","project":null,"args":{}}\n', "op"),
                               (b'{"op":"hello"}\n', "project")):
                with self.subTest(raw=raw):
                    p = problem_of(raw_roundtrip(sock, raw))
                    self.assertEqual((p["rule"], p["field"]), ("AN-12", field))
            eof = raw_roundtrip(sock, b'{"op":"hello"')         # EOF before the newline
            self.assertEqual(problem_of(eof)["rule"], "AN-12")
            self.assertIn("keeper_version", KeeperClient(sock, timeout=5).call("hello", None, {}))

    def test_line_at_the_cap_is_served(self):
        line = b'{"op":"hello","project":null,"args":{}}'
        with serving(MAX_REQUEST_BYTES=len(line)) as sock:
            resp = raw_roundtrip(sock, line + b"\n")
            self.assertIs(resp["ok"], True, resp)

    def test_foreign_uid_refused_end_to_end(self):
        with serving(SOCKET_TIMEOUT_S=5.0) as sock, \
                mock.patch.object(protocol, "peer_uid", lambda conn: os.getuid() + 1), \
                mock.patch.object(server.Keeper, "handle", side_effect=AssertionError("read a refused peer")) as h:
            with self.assertRaises(Refused) as cm:
                KeeperClient(sock, timeout=5).call("hello", None, {})
            p = cm.exception.problems[0]
            self.assertEqual((p.rule, p.field), ("AN-12", "peer"))
            # the refusal is sent before anything is read: a peer that sends nothing still gets it at once
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(3)
                s.connect(str(sock))
                self.assertEqual(problem_of(json.loads(read_line(s, 1 << 20)))["field"], "peer")
            h.assert_not_called()

    def test_idle_client_times_out(self):
        with serving(SOCKET_TIMEOUT_S=0.5) as sock:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as idle:
                idle.settimeout(10)
                idle.connect(str(sock))
                time.sleep(0.1)                                # the keeper is now holding the idle peer
                t0 = time.monotonic()
                self.assertIn("keeper_version", KeeperClient(sock, timeout=10).call("hello", None, {}))
                self.assertLess(time.monotonic() - t0, 5.0)
                got = idle.recv(65536)                          # a timeout problem line, or the closed connection
                if got:
                    self.assertEqual(problem_of(json.loads(got))["rule"], "AN-12")

    def test_trickling_client_times_out(self):
        with serving(SOCKET_TIMEOUT_S=0.5) as sock:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as slow:
                slow.settimeout(10)
                slow.connect(str(sock))
                t0, closed = time.monotonic(), False
                while time.monotonic() - t0 < 4.0:             # one byte every 0.1 s never finishes a line
                    try:
                        slow.sendall(b" ")
                    except (BrokenPipeError, ConnectionResetError):
                        closed = True
                        break
                    time.sleep(0.1)
                self.assertTrue(closed, "the keeper kept a trickling peer past its request deadline")
            self.assertIn("keeper_version", KeeperClient(sock, timeout=5).call("hello", None, {}))

    def test_response_over_the_cap_refused(self):
        # the cap check is on the server side: a big result becomes one AN-12 problem line
        # the cap leaves room for `hello` (fingerprint, public key, environment hash), which the helper waits on
        with serving(MAX_RESPONSE_BYTES=1024) as sock, \
                mock.patch.dict(server.OPS, {"big": lambda k, pr, a, peer: {"x": "y" * 4000}}):
            p = problem_of(raw_roundtrip(sock, b'{"op":"big","project":null,"args":{}}\n'))
            self.assertEqual((p["rule"], p["field"]), ("AN-12", "response"))

    def test_socket_mode_and_second_serve_refused(self):
        with serving() as sock:
            self.assertTrue(stat.S_ISSOCK(os.lstat(sock).st_mode))
            self.assertEqual(stat.S_IMODE(os.lstat(sock).st_mode), 0o666)
            with self.assertRaises(Refused) as cm:              # a live keeper's socket is not taken over
                server.serve(sock.parent / "state", sock, os.getuid())
            self.assertEqual(cm.exception.problems[0].rule, "AN-12")
            self.assertIn("keeper_version", KeeperClient(sock, timeout=5).call("hello", None, {}))
        self.assertFalse(sock.exists(), "serve leaves its socket behind")


@skip_unless_linux
class TestServeRefusals(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gk-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.state = self.tmp / "state"
        self.state.mkdir(mode=0o700)

    def test_socket_path_too_long(self):
        sock = self.tmp / ("s" * (MAX_SOCKET_PATH + 1 - len(os.fsencode(self.tmp)) - 1))
        self.assertEqual(len(os.fsencode(sock)), MAX_SOCKET_PATH + 1)
        with self.assertRaises(Refused) as cm:
            server.serve(self.state, sock, os.getuid())
        self.assertEqual((cm.exception.problems[0].rule, cm.exception.problems[0].field), ("AN-12", "socket"))
        self.assertFalse(sock.exists())

    def test_existing_non_socket_refused(self):
        sock = self.tmp / "keeper.sock"
        sock.write_text("keep me", encoding="utf-8")
        with self.assertRaises(Refused) as cm:
            server.serve(self.state, sock, os.getuid())
        self.assertEqual(cm.exception.problems[0].rule, "AN-12")
        self.assertEqual(sock.read_text(encoding="utf-8"), "keep me")

    def test_socket_directory_missing_refused(self):
        with self.assertRaises(Refused) as cm:
            server.serve(self.state, self.tmp / "no-such-dir" / "keeper.sock", os.getuid())
        self.assertEqual((cm.exception.problems[0].rule, cm.exception.problems[0].field), ("AN-12", "socket"))

    def test_symlink_at_the_socket_path_refused(self):
        target = self.tmp / "elsewhere.sock"
        sock = self.tmp / "keeper.sock"
        sock.symlink_to(target)
        with self.assertRaises(Refused) as cm:
            server.serve(self.state, sock, os.getuid())
        self.assertEqual(cm.exception.problems[0].rule, "AN-12")
        self.assertTrue(sock.is_symlink())
        self.assertFalse(target.exists())

    def test_socket_directory_writable_by_another_uid_refused(self):   # final review 6a (AN-13)
        run = self.tmp / "run"
        run.mkdir()
        os.chmod(run, 0o777)                                # other-write: anyone could replace the socket
        stop = threading.Event()
        stop.set()                                          # never serve, even if the refusal were missing
        with self.assertRaises(Refused) as cm:
            server.serve(self.state, run / "keeper.sock", os.getuid(), stop=stop)
        problems = cm.exception.problems
        self.assertEqual({p.rule for p in problems}, {"AN-13"})
        self.assertTrue(any(str(run) in p.message for p in problems), problems)
        self.assertEqual(list(self.state.iterdir()), [])     # refused before the state is opened
        self.assertEqual(list(run.iterdir()), [])


class TestClient(unittest.TestCase):
    def test_unreachable_socket(self):
        tmp = Path(tempfile.mkdtemp(prefix="gk-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        with self.assertRaises(Refused) as cm:
            KeeperClient(tmp / "missing.sock").call("hello", None, {})
        p = cm.exception.problems[0]
        self.assertEqual((p.rule, p.message), ("AN-12", "the keeper is not reachable at the socket"))
        self.assertNotIn("Traceback", str(cm.exception))

    def test_request_over_the_cap_refused_before_sending(self):
        tmp = Path(tempfile.mkdtemp(prefix="gk-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        with mock.patch.object(protocol, "MAX_REQUEST_BYTES", 64), self.assertRaises(Refused) as cm:
            KeeperClient(tmp / "missing.sock").call("hello", None, {"x": "y" * 100})
        self.assertEqual((cm.exception.problems[0].rule, cm.exception.problems[0].field), ("AN-12", "request"))

    def test_intake_waits_with_the_long_op_deadline(self):     # final review 6c: a 64 MiB split is not a short op
        self.assertEqual(protocol.LONG_OPS, frozenset({"intake", "evaluate", "gate", "report"}))
        client = KeeperClient(Path("unused.sock"))
        self.assertEqual(client.timeout_for("intake"), protocol.LONG_OP_TIMEOUT_S)
        self.assertEqual(client.timeout_for("hello"), protocol.SOCKET_TIMEOUT_S)


@skip_unless_linux
class TestKeeperProcess(unittest.TestCase):
    def test_hello_over_the_socket(self):
        with KeeperProcess() as k:
            result = k.client.call("hello", None, {})
            self.assertIn("keeper_version", result)
            self.assertIs(result["same_user"], True)

    def test_bad_requests_keep_the_keeper_serving(self):
        with KeeperProcess() as k:
            for raw, field in ((b"not json\n", "request"), (b"[1]\n", "request"),
                               (b'{"op":"nope","project":null,"args":{}}\n', "op"),
                               (b'{"op":"hello"}\n', "project")):
                p = problem_of(raw_roundtrip(k.socket, raw))
                self.assertEqual((p["rule"], p["field"]), ("AN-12", field))
            self.assertTrue(k.client.call("hello", None, {}))
            self.assertNotIn(b"nope", k.log_path.read_bytes())   # nothing a peer sent is logged

    def test_sigterm_unlinks_the_socket_and_a_killed_keeper_restarts(self):
        with KeeperProcess() as k:
            self.assertEqual(k.stop(), 0, k.output())
            self.assertFalse(k.socket.exists())
            k.start()
            k.kill()                                            # a crash leaves a stale socket file behind
            self.assertTrue(k.socket.exists())
            k.restart()
            self.assertIn("keeper_version", k.client.call("hello", None, {}))

    def test_socket_path_too_long_exits_3(self):
        from _support import EXIT_REFUSED, describe, run_cli
        tmp = Path(tempfile.mkdtemp(prefix="gk-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / "state").mkdir(mode=0o700)
        proc = run_cli(["--json", "keeper", "serve", "--state", str(tmp / "state"), "--socket",
                        str(tmp / ("s" * 120)), "--owner-uid", str(os.getuid())], cwd=tmp, timeout=60)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        doc = json.loads(proc.stdout)
        self.assertEqual((doc["command"], doc["problems"][0]["rule"]), ("keeper serve", "AN-12"))


if __name__ == "__main__":
    unittest.main()
