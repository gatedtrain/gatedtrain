"""Test fixture: the keeper as a subprocess of the test user (tests/README.md section 13).

`KeeperProcess` makes a fresh temporary directory holding `state/` (mode 0700) and `keeper.sock`, starts
`tests/fixtures/ainative/run_cli.py keeper serve --state ... --socket ... --owner-uid <os.getuid()>` (so the synthetic
pack is injected as in step 1) and waits for `hello`. Leaving the context stops the process and removes the directory;
a start that fails cleans up before it raises.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from _support import FIXTURES, cli_env

from gatedtrain.ainative.keeper.protocol import KeeperClient
from gatedtrain.errors import Refused

RUN_CLI = FIXTURES / "ainative" / "run_cli.py"
START_TIMEOUT_S = 15.0
STOP_TIMEOUT_S = 10.0


def skip_unless_linux(case):
    return unittest.skipUnless(sys.platform == "linux", "the keeper is Linux only (Unix socket, SO_PEERCRED)")(case)


class KeeperStartError(AssertionError):
    """The keeper process exited, or did not answer `hello`, within the start timeout."""

    def __init__(self, message: str, returncode: int | None, output: str) -> None:
        super().__init__(f"{message} (exit {returncode})\n--- output ---\n{output[-4000:]}")
        self.returncode, self.output = returncode, output


class KeeperProcess:
    def __init__(self, *, owner_uid: int | None = None, env: dict[str, str] | None = None) -> None:
        self.owner_uid = os.getuid() if owner_uid is None else owner_uid
        self.env = dict(env or {})
        self.tmp: Path | None = None
        self.proc: subprocess.Popen | None = None
        self._log = None

    # paths -----------------------------------------------------------------------------------------------------
    @property
    def state(self) -> Path:
        return self.tmp / "state"

    @property
    def socket(self) -> Path:
        return self.tmp / "keeper.sock"

    @property
    def log_path(self) -> Path:
        return self.tmp / "keeper.log"

    @property
    def client(self) -> KeeperClient:
        return KeeperClient(self.socket)

    def output(self) -> str:
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    # lifecycle -------------------------------------------------------------------------------------------------
    def __enter__(self) -> KeeperProcess:
        self.tmp = Path(tempfile.mkdtemp(prefix="gk-"))
        try:
            self.state.mkdir(mode=0o700)
            os.chmod(self.state, 0o700)
            self.start()
        except BaseException:
            self._cleanup()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self._cleanup()

    def start(self) -> None:
        if self.running():
            raise AssertionError("the keeper is already running")
        self._close_log()
        self._log = open(self.log_path, "ab")
        cmd = [sys.executable, str(RUN_CLI), "keeper", "serve", "--state", str(self.state),
               "--socket", str(self.socket), "--owner-uid", str(self.owner_uid)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=self._log, stderr=subprocess.STDOUT,
                                     env={**cli_env(), **self.env}, cwd=str(self.tmp))
        deadline = time.monotonic() + START_TIMEOUT_S
        probe = KeeperClient(self.socket, timeout=2.0)
        while True:
            code = self.proc.poll()
            if code is not None:
                self.proc.wait()
                raise KeeperStartError("the keeper exited before answering hello", code, self.output())
            try:
                probe.call("hello", None, {})
                return
            except Refused:
                pass
            if time.monotonic() > deadline:
                self.kill()
                raise KeeperStartError("the keeper did not answer hello in time", None, self.output())
            time.sleep(0.05)

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> int | None:
        """SIGTERM and wait; SIGKILL if it does not exit in time. Returns the exit code."""
        if self.proc is None:
            return None
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        return self.proc.returncode

    def kill(self) -> int | None:
        """SIGKILL (a crash: the socket file stays behind). Returns the exit code."""
        if self.proc is None:
            return None
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        return self.proc.returncode

    def restart(self) -> None:
        self.stop()
        self.start()

    def _close_log(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None

    def _cleanup(self) -> None:
        try:
            self.stop()
        finally:
            self._close_log()
            if self.tmp is not None:
                shutil.rmtree(self.tmp, ignore_errors=True)
