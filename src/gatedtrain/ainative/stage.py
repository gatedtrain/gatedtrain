"""AN-3: the stage protocol and the stage registry.

A pipeline is a list of stages. Each stage declares the artifacts it reads (`inputs`, name -> schema id), the
artifacts it writes (`outputs`), the resources it needs (`needs`) and the params it accepts (`params`). `plan`
proposes what the stage would do and has no side effects; `run` executes a confirmed decision and returns the
artifacts it wrote; `check` re-validates the outputs and returns refusals in the standard problem form.

The five trust stages are built-in names reserved by the registry: a plugin that registers one of them is
refused (`AN-3`), so an agent cannot swap a trust stage for its own implementation. In step 1 the built-ins are
placeholders; the real trust stages replace them without changing the reserved names.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import TYPE_CHECKING, Any, Protocol

from ..errors import Problem, Refused
from .project import ArtifactRef, Project

if TYPE_CHECKING:  # defined by the trust module
    from .trust import AlphaLedger, TrustBackend

STAGE_ENTRY_POINT_GROUP = "gatedtrain.stages"
TRUST_STAGES = ("intake_seal", "preregister", "evaluate_sealed", "gate", "report")
NEEDS = frozenset({"gpu", "llm", "network", "user"})


@dataclass(frozen=True)
class Proposal:
    stage: str
    summary: str
    choices: dict
    estimate: dict
    needs_user: str | None = None


@dataclass
class StageContext:
    project: Project
    inputs: dict[str, ArtifactRef]
    params: dict
    trust: "TrustBackend"
    ledger: "AlphaLedger"


class Stage(Protocol):
    name: str
    version: str
    inputs: dict[str, str]        # artifact name -> schema id
    outputs: dict[str, str]
    needs: frozenset[str]         # subset of NEEDS
    params: frozenset[str]        # accepted param names

    def plan(self, ctx: StageContext) -> Proposal: ...

    def run(self, ctx: StageContext, decision: dict) -> dict[str, ArtifactRef]: ...

    def check(self, ctx: StageContext, outputs: dict[str, ArtifactRef]) -> list[Problem]: ...


class _TrustPlaceholder:
    """Built-in entry that reserves a trust stage name until the real implementation lands."""

    name = ""
    version = "0"
    inputs: dict[str, str] = {}
    outputs: dict[str, str] = {}
    needs: frozenset[str] = frozenset()
    params: frozenset[str] = frozenset()

    def plan(self, ctx: StageContext) -> Proposal:
        raise NotImplementedError(f"trust stage {self.name!r} is not implemented yet")

    def run(self, ctx: StageContext, decision: dict) -> dict[str, ArtifactRef]:
        raise NotImplementedError(f"trust stage {self.name!r} is not implemented yet")

    def check(self, ctx: StageContext, outputs: dict[str, ArtifactRef]) -> list[Problem]:
        raise NotImplementedError(f"trust stage {self.name!r} is not implemented yet")


def _placeholder(name: str) -> type:
    return type(f"{name.title().replace('_', '')}Placeholder", (_TrustPlaceholder,), {"name": name})


def builtin_stages() -> dict[str, type]:
    return {name: _placeholder(name) for name in TRUST_STAGES}


def _plugin_entry_points() -> list[Any]:
    return list(metadata.entry_points(group=STAGE_ENTRY_POINT_GROUP))


def stage_registry() -> dict[str, type]:
    """Built-ins plus plugins from the `gatedtrain.stages` entry-point group; a plugin named like a trust stage
    is refused."""
    registry = builtin_stages()
    problems: list[Problem] = []
    for ep in _plugin_entry_points():
        if ep.name in TRUST_STAGES:
            problems.append(Problem(
                "AN-3", f"entry point {ep.name}",
                f"plugin {ep.value!r} registers the reserved trust stage name {ep.name!r}",
                "rename the plugin stage; the five trust stages are built in and cannot be replaced"))
            continue
        if ep.name in registry:
            problems.append(Problem("AN-3", f"entry point {ep.name}", f"stage name {ep.name!r} is registered twice",
                                    "give each plugin stage a unique name"))
            continue
        registry[ep.name] = ep.load()
    if problems:
        raise Refused(problems)
    return registry
