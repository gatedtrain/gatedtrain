"""AN-14, AN-27, AN-28, AN-29: the keeper mode in `gatedtrain verify`.

`verify` states the mode as a fact with its sources (`mode_sources`): the keeper's claim, the kernel's answer about the
process that listens on the keeper's socket (AN-29), and each report's own signed `keeper_mode`. The effective mode is
the weakest of them, so no source can upgrade another. Exit 0 stays reserved for the separate-user result. A same-user
result exits 1, unless the reader opts in with `--accept-mode same-user`, which turns exactly that result (and no other
weaker one: not a rootful-access report, not a revision 23 or unknown text) into exit 0 with the statement printed. A
project that registered `require_keeper_mode: separate_user` is refused (AN-28, exit 3) whatever the option says.
"""

from __future__ import annotations

from typing import Any

from ..errors import Problem, Refused
from .keeper import modes
from .sealed_eval import REPORT_SCHEMA_VERSION
from .keeper.texts import KEEPER_GUARANTEE, KEEPER_OWNER_RUN_GUARANTEE, KEEPER_SAME_USER_GUARANTEE

# The only value `--accept-mode` takes. Accepting a weaker mode is the reader's own decision, taken per call; there is
# no environment variable or project file for it (an agent controls its own environment and project).
ACCEPT_SAME_USER = "same-user"
ACCEPT_MODES = (ACCEPT_SAME_USER,)

OWNER_RUN_REPORT = ("the report states that the keeper ran as the owner's own OS user (same-user mode): the procedure "
                    "was followed, but nothing was sealed from the user's account")
ACCEPTED_WEAKER_MODE = ("ACCEPTED WEAKER MODE: same-user (procedure followed; sealed data not protected from the "
                        "user's account)")

# The report texts a same-user result may carry and still be accepted: the strong text (a project whose records were
# served by both kinds of keeper) and the two same-user texts. Any other text is a different weaker statement.
_ACCEPTABLE_TEXTS = (KEEPER_GUARANTEE, KEEPER_SAME_USER_GUARANTEE, KEEPER_OWNER_RUN_GUARANTEE)


def _bad(field: str, message: str) -> Refused:
    return Refused([Problem("AN-14", field, message, "the project's records were altered or come from another keeper; "
                                                      "do not trust its report")])


def receipt_modes(receipts: list[dict]) -> tuple[list[str], dict[str, str]]:
    """`(modes, claim modes by attempt id)`: the `keeper_mode` of every signed registration and claim receipt that
    states one. A record that predates AN-27 states none (skipped: the mode stays undetermined unless another source
    gives it); a value that is not a mode is refused (AN-14)."""
    found: list[str] = []
    by_attempt: dict[str, str] = {}
    for env in receipts:
        if env.get("kind") not in ("registration", "claim"):
            continue
        record = env.get("record") or {}
        if "keeper_mode" not in record:
            continue
        mode = record["keeper_mode"]
        if mode not in modes.MODES:
            raise _bad(f"receipts.{env['kind']}.keeper_mode", f"a signed {env['kind']} states a keeper mode that is "
                       "not one of " + ", ".join(modes.MODES))
        found.append(mode)
        if env["kind"] == "claim":
            by_attempt[record.get("attempt_id")] = mode
    return found, by_attempt


def _listed(rel: str, field: str, items: Any, key: str) -> list[str]:
    """The valid modes of a version 3 report's list (`attempts` or `modes_seen`); refused (AN-14) when it is not a
    list of records that each name a mode."""
    if not isinstance(items, list) or any(not isinstance(i, dict) or i.get(key) not in modes.MODES for i in items):
        raise _bad(f"{rel}.{field}", f"the report's {field} is not a list that names a keeper mode ({key}) for each "
                   "entry")
    return [i[key] for i in items]


def report_mode(rel: str, report: dict, receipts: tuple[list[str], dict[str, str]] = ([], {})) -> str | None:
    """The mode a checked report states: its signed `keeper_mode` (version 3), refused (AN-14) when it is not a mode,
    contradicts the report's signed guarantee text, is stronger than the signed registration and claim receipts it
    covers or than its own `attempts` and `modes_seen`, or when those lists name no mode or contradict the receipts
    (an attempt's mode is its claim's); for a version 2 report the mode its text states (also refused when stronger
    than the receipts). None when the text is unknown (the unknown-text warning covers it)."""
    seen_in_receipts, by_attempt = receipts
    from_text = modes.mode_of_text(report.get("sealed_guarantee"))
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        claimed = from_text
    else:
        claimed = report.get("keeper_mode")
        if claimed not in modes.MODES:
            raise _bad(f"{rel}.keeper_mode", "the report's keeper mode is not one of " + ", ".join(modes.MODES))
        if from_text is not None and from_text != claimed:
            raise _bad(f"{rel}.keeper_mode", "the report's keeper mode contradicts the guarantee text it signs")
        attempts = report.get("attempts")
        in_attempts = _listed(rel, "attempts", attempts, "keeper_mode")
        for a in attempts:
            stated = by_attempt.get(a.get("attempt_id"))
            if stated is not None and stated != a["keeper_mode"]:
                raise _bad(f"{rel}.attempts", "an attempt's keeper mode differs from the one its signed claim states")
        in_seen = _listed(rel, "modes_seen", report.get("modes_seen"), "mode")
        if in_seen and seen_in_receipts and modes.rank(modes.weakest(*in_seen)) > \
                modes.rank(modes.weakest(*seen_in_receipts)):
            raise _bad(f"{rel}.modes_seen", "the report's modes_seen omit a weaker mode that a signed receipt states")
        weaker = [*in_attempts, *in_seen]
        if weaker and modes.rank(claimed) > modes.rank(modes.weakest(*weaker)):
            raise _bad(f"{rel}.keeper_mode", "the report's keeper mode is stronger than its own attempts or "
                       "modes_seen")
    if claimed is not None and seen_in_receipts and modes.rank(claimed) > modes.rank(modes.weakest(*seen_in_receipts)):
        raise _bad(f"{rel}.keeper_mode", "the report's keeper mode is stronger than a signed receipt it covers")
    return claimed


def required_mode(receipts: list[dict], report_requirements: list[Any]) -> str:
    """The policy of the project (AN-28): `separate_user` when a signed registration or a report names it."""
    named = [e["record"].get("require_keeper_mode") for e in receipts if e.get("kind") == "registration"]
    return modes.SEPARATE_USER if modes.SEPARATE_USER in (*named, *report_requirements) else modes.REQUIRE_ANY


def sources(keeper_claim: str | None, attested: str | None, reports: list[str | None],
            receipts: list[str] | None = None) -> dict:
    return {"keeper_claim": keeper_claim, "kernel_attestation": attested, "report": reports,
            "receipts": receipts or []}


def effective(keeper_mode: str | None, report_modes: list[str | None], receipt_modes: list[str] = ()) -> str | None:
    """The weakest known mode among the keeper's effective mode, the reports' modes and the signed receipts' modes;
    None when none is known."""
    known = [m for m in (keeper_mode, *report_modes, *receipt_modes) if m in modes.MODES]
    return modes.weakest(*known) if known else None


def check_policy(require: str, effective_mode: str | None) -> None:
    """AN-28: refused when the project requires a stronger mode than the one the evidence shows."""
    if effective_mode is None and require != modes.REQUIRE_ANY:
        raise Refused([Problem(
            modes.RULE_POLICY, "keeper_mode", f"the project's registration requires a {require} keeper, but no signed "
            "record or keeper states the mode, so it cannot be shown", "verify against the keeper (`--keeper`), or "
            "run the project under a keeper in separate-user mode (`gatedtrain setup keeper --mode separate-user`)")])
    if effective_mode is not None and not modes.satisfies(effective_mode, require):
        raise Refused([Problem(
            modes.RULE_POLICY, "keeper_mode", f"the project's registration requires a {require} keeper, but the "
            f"evidence shows {effective_mode}", "this result does not meet the registered requirement; run the "
            "project under a keeper in separate-user mode (`gatedtrain setup keeper --mode separate-user`)")])


def accepted_weaker(out: dict, accept: str | None) -> bool:
    """True when `--accept-mode same-user` applies: the result is same-user and nothing else weaker. A keeper that
    is same-user states nothing about rootful access that matters (nothing is sealed anyway), so its rootful flag
    does not block; a rootful or unknown report text does."""
    if accept != ACCEPT_SAME_USER or out["anchor"] == "none" or out.get("effective_mode") is None:
        return False
    found = out.get("report_guarantees") or []
    if any(g not in _ACCEPTABLE_TEXTS for g in found):
        return False
    keeper_same = out["anchor"] == "keeper" and out["same_user"] is True
    if out["anchor"] == "keeper" and not keeper_same and out["root_docker"]:
        return False
    return keeper_same or out.get("effective_mode") == modes.SAME_USER
