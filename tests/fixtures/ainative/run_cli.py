"""Test-only entry point: run `gatedtrain.cli.main` with the synthetic scenario pack injected.

The pack is deliberately not registered through entry points, so production code has no hook that loads it. This
wrapper patches the two entry point lookups the same way `tests/test_an_synthetic_pack.py::inject_pack` does, then
calls the real CLI and exits with its code.

Test-only generator switch (keeper tests, tests/test_an_keeper_attempts.py): the environment variable
`GATEDTRAIN_TEST_GENERATOR` wraps the pack's generator inside this process (so inside a keeper subprocess too). A model
whose `label` is `base` is the baseline; every other model is a candidate.

- `count:<file>`: append one byte per call to `<file>` (`b` for the baseline, `c` for a candidate);
- `raise`: a candidate call raises `ValueError` whose message is the item's input (it must never leave the keeper);
- `nan`: a candidate call returns a float NaN (not canonical JSON);
- `block:<file>`: the first candidate call creates `<file>`, then sleeps (the test kills the process meanwhile);
- `slow:<seconds>`: every candidate call sleeps that long first;
- `write_error`: the generator is the pack's, but the keeper cannot write a candidate's per-item records (`OSError`),
  an exception after the grant that is not a refusal.

Test-only backend switch (AN-24, tests/test_an_backend_stages.py): `GATEDTRAIN_TEST_BACKEND_RUNTIME=fake:<state dir>`
makes `machine.compute_backend` build its backend on `FakeRuntime(state_dir=<state dir>)`, whose containers of every
image run `subprocess_handler`; the outcome queue is the one the test stored in the state directory (no outcomes are
passed here), so no Docker daemon is touched.

Test-only host view switch (AN-21, tests/test_an_rootful.py, tests/_keeper.py): `GATEDTRAIN_TEST_HOST_VIEW=<dir>` makes
`keeper serve` run its rootful-access detection on the fake tree `<dir>` (`HostView(root=<dir>, root_uid=<this uid>)`)
instead of the real host. Only this wrapper reads the variable; production code has no such switch.

Test-only keeper runtime switch (AN-25, tests/test_an_keeper_inference.py):
`GATEDTRAIN_TEST_KEEPER_RUNTIME=fake:<image id>[:<state dir>]` makes `keeper serve` started with an inference image
run its containers on `FakeRuntime` instead of Docker: the fake daemon holds `<image id>` under the configured image
reference (so a different `--inference-image` is refused at start), its socket is the `unix://` path of
`--docker-host` (owned by this uid), containers of that image run `subprocess_handler` (the pack's script on the
host), and with `<state dir>` the containers and the run log live there for the test to read. Only this wrapper reads
the variable.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from gatedtrain import cli
from gatedtrain.ainative import stage as stage_mod, trust_stages
from gatedtrain.testing import synthetic_pack as pack


class _StageEP:
    def __init__(self, cls: type) -> None:
        self.name, self.value, self._cls = cls.name, f"gatedtrain.testing.synthetic_pack:{cls.__name__}", cls

    def load(self) -> type:
        return self._cls


class _ScenarioEP:
    name = "synthetic"
    value = "gatedtrain.testing.synthetic_pack:SCENARIO"

    def load(self):
        return _scenario()


TEST_GENERATOR_ENV = "GATEDTRAIN_TEST_GENERATOR"
TEST_BACKEND_RUNTIME_ENV = "GATEDTRAIN_TEST_BACKEND_RUNTIME"
TEST_HOST_VIEW_ENV = "GATEDTRAIN_TEST_HOST_VIEW"
TEST_KEEPER_RUNTIME_ENV = "GATEDTRAIN_TEST_KEEPER_RUNTIME"


class _EveryImage(dict):
    """A handler table that runs `subprocess_handler` for any image."""

    def get(self, key, default=None):
        from gatedtrain.testing.fake_runtime import subprocess_handler

        return subprocess_handler


def _fake_backend_runtime(mode: str) -> None:
    from gatedtrain.ainative import machine
    from gatedtrain.testing.fake_runtime import FakeRuntime

    kind, _, arg = mode.partition(":")
    if kind != "fake" or not arg:
        raise SystemExit(f"{TEST_BACKEND_RUNTIME_ENV}: expected fake:<state dir>, got {mode!r}")
    real = machine.compute_backend

    def compute_backend(project, *, runtime=None):
        rt = FakeRuntime(state_dir=Path(arg))
        rt.handlers = _EveryImage()
        return real(project, runtime=rt)

    machine.compute_backend = compute_backend


def _fake_host_view(root: str) -> None:
    from gatedtrain.ainative.keeper import rootful, server

    real = server.serve
    view = rootful.HostView(root=Path(root), root_uid=os.getuid())

    def serve(*args, **kwargs):
        return real(*args, **{**kwargs, "host_view": view})

    server.serve = serve


def _fake_keeper_runtime(mode: str) -> None:
    import re

    from gatedtrain.ainative.keeper import server
    from gatedtrain.testing.fake_runtime import FakeRuntime, subprocess_handler

    m = re.fullmatch(r"fake:(sha256:[0-9a-f]{64})(?::(.+))?", mode)
    if m is None:
        raise SystemExit(f"{TEST_KEEPER_RUNTIME_ENV}: expected fake:<image id>[:<state dir>], got {mode!r}")
    image, state_dir = m.group(1), m.group(2)
    real = server.serve

    def serve(*args, **kwargs):
        inference = kwargs.get("inference")
        if inference is not None:
            host = inference.docker_host
            sock = os.path.realpath(host[len("unix://"):]) if host.startswith("unix://") else None
            ref = inference.image_ref or inference.image_id
            kwargs["runtime"] = FakeRuntime({image: subprocess_handler}, image_ids={ref: image, image: image},
                                            state_dir=Path(state_dir) if state_dir else None, socket=sock)
        return real(*args, **kwargs)

    server.serve = serve


def _test_generator(mode: str):
    kind, _, arg = mode.partition(":")
    if kind not in ("count", "raise", "nan", "block", "slow"):
        raise SystemExit(f"{TEST_GENERATOR_ENV}: unknown mode {kind!r}")

    def generate(model_ref: dict, item: dict, **kw):
        candidate = model_ref.get("label") != "base"
        if kind == "count":
            fd = os.open(arg, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                os.write(fd, b"c" if candidate else b"b")
            finally:
                os.close(fd)
        elif candidate and kind == "raise":
            raise ValueError(item["input"])
        elif candidate and kind == "nan":
            return float("nan")
        elif candidate and kind == "block":
            open(arg, "a").close()
            time.sleep(3600)
        elif candidate and kind == "slow":
            time.sleep(float(arg))
        return pack.generate(model_ref, item, **kw)

    return generate


def _scenario():
    mode = os.environ.get(TEST_GENERATOR_ENV)
    if not mode or mode == "write_error":
        return pack.SCENARIO
    return pack.Scenario(pack.SCENARIO.name, {pack.GENERATOR: _test_generator(mode)}, pack.SCENARIO.pipeline,
                         pack.SCENARIO.scorers)


def _fail_candidate_record_writes() -> None:
    from gatedtrain.ainative.keeper import state

    original = state.KeeperState._write_arm

    def write_arm(self, rel: str, model_sha: str, records: list) -> None:
        if "/records/" in rel:
            raise OSError("test: the candidate's records cannot be written")
        original(self, rel, model_sha, records)

    state.KeeperState._write_arm = write_arm


def main(argv: list[str]) -> int:
    if os.environ.get(TEST_GENERATOR_ENV) == "write_error":
        _fail_candidate_record_writes()
    if os.environ.get(TEST_BACKEND_RUNTIME_ENV):
        _fake_backend_runtime(os.environ[TEST_BACKEND_RUNTIME_ENV])
    if os.environ.get(TEST_HOST_VIEW_ENV):
        _fake_host_view(os.environ[TEST_HOST_VIEW_ENV])
    if os.environ.get(TEST_KEEPER_RUNTIME_ENV):
        _fake_keeper_runtime(os.environ[TEST_KEEPER_RUNTIME_ENV])
    eps = [_StageEP(c) for c in (pack.ProfileData, pack.ChooseTaskAndBase, pack.BuildTrainSet, pack.WriteRecipe,
                                 pack.Smoke, pack.Train)]
    stage_mod._plugin_entry_points = lambda: list(eps)
    trust_stages._scenario_entry_points = lambda: [_ScenarioEP()]
    trust_stages._reset_scenario_snapshot()
    return cli.main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
