"""Test-only entry point: run `gatedtrain.cli.main` with the synthetic scenario pack injected.

The pack is deliberately not registered through entry points, so production code has no hook that loads it. This
wrapper patches the two entry point lookups the same way `tests/test_an_synthetic_pack.py::inject_pack` does, then
calls the real CLI and exits with its code.
"""

from __future__ import annotations

import sys

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
        return pack.SCENARIO


def main(argv: list[str]) -> int:
    eps = [_StageEP(c) for c in (pack.ProfileData, pack.ChooseTaskAndBase, pack.BuildTrainSet, pack.WriteRecipe,
                                 pack.Smoke, pack.Train)]
    stage_mod._plugin_entry_points = lambda: list(eps)
    trust_stages._scenario_entry_points = lambda: [_ScenarioEP()]
    trust_stages._reset_scenario_snapshot()
    return cli.main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
