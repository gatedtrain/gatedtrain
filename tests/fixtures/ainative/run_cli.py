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
"""

from __future__ import annotations

import os
import sys
import time

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
    eps = [_StageEP(c) for c in (pack.ProfileData, pack.ChooseTaskAndBase, pack.BuildTrainSet, pack.WriteRecipe,
                                 pack.Smoke, pack.Train)]
    stage_mod._plugin_entry_points = lambda: list(eps)
    trust_stages._scenario_entry_points = lambda: [_ScenarioEP()]
    trust_stages._reset_scenario_snapshot()
    return cli.main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
