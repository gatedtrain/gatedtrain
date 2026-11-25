"""AN-27 to AN-29: the keeper's mode is an observed fact, never a flag.

There are two modes. `separate_user`: the keeper runs under another OS user than the owner and holds its state out of
the owner's reach. `same_user`: it does not, so anything running as the owner (an agent with a shell) can read the
sealed data and the signing key. The mode is never chosen by an option, an environment variable or a project file:
`observe` derives it from facts the process reads from the kernel and the file system, so a caller cannot weaken or
strengthen the statement by saying so. These names, the policy values and the facts below are trust constants: they
live here once, and `tests/test_an_mode_observe.py` pins them.

`observe` (AN-27) takes values and does no I/O. The mode is `separate_user` only when every condition holds; any failed
condition gives `same_user` (fail toward the weaker mode, as the rootful-access detection AN-21 does):

1. the keeper's effective uid is not the owner uid it serves;
2. the state directory belongs to the keeper's uid and is closed to group and others;
3. the peer of this connection, when there is one, is not the keeper's own uid.

Condition 3 closes the misdeclared owner: a process started by the agent as `keeper serve --owner-uid <any other
number>` has the agent's uid, and the agent reaches it as a peer with the keeper's own uid, so that peer sees `same_user`
whatever `--owner-uid` says. `--owner-uid` is only the peer allow-list, never the definition of the mode.

`weakest` combines modes (a set only ever gets weaker, AN-27); `satisfies` applies the registered policy
`require_keeper_mode` (AN-28); `effective_client_mode` is what a client that reads the kernel's credentials of the
listening process may believe (AN-29); `mode_of_text` derives the mode of an older report from its signed guarantee
text.
"""

from __future__ import annotations

import stat
from collections.abc import Iterable
from typing import Any

from ...errors import Problem, Refused

SEPARATE_USER = "separate_user"
SAME_USER = "same_user"
# Ordered from the weakest to the strongest statement; the rank of a mode is its index. `same_user` is the weaker
# mode: it is what any doubt resolves to.
MODES = (SAME_USER, SEPARATE_USER)

REQUIRE_ANY = "any"
# The registration policy `require_keeper_mode` (AN-28): the weakest mode the owner accepts for a project. Fixed when
# the project is opened and signed into its registration, so an agent cannot loosen it afterwards.
REQUIREMENTS = (REQUIRE_ANY, SEPARATE_USER)

# Permission bits of group and others. A state directory with any of them set is readable or writable by an account
# that is not the keeper's, so the keeper is not separate from it.
OTHERS_AND_GROUP_BITS = 0o077

RULE_OBSERVE = "AN-27"
RULE_POLICY = "AN-28"
RULE_ATTESTATION = "AN-29"


def rank(mode: str) -> int:
    """The strength of a mode: 0 for `same_user`, 1 for `separate_user`; ValueError for anything else."""
    if mode not in MODES:
        raise ValueError(f"not a keeper mode: {mode!r}")
    return MODES.index(mode)


def weakest(*modes: str) -> str:
    """The weakest of the given modes (at least one)."""
    if not modes:
        raise ValueError("no mode given")
    return min(modes, key=rank)


def satisfies(mode: str, require: str) -> bool:
    """True when `mode` meets the policy `require` (one of `REQUIREMENTS`)."""
    if require not in REQUIREMENTS:
        raise ValueError(f"not a keeper mode policy: {require!r}")
    return require == REQUIRE_ANY or rank(mode) >= rank(require)


def observe(keeper_euid: int, owner_uid: int, state_dir_stat: Any, peer_uid: int | None = None) -> dict:
    """AN-27: `{"mode", "facts"}` from observed values. `state_dir_stat` has `st_uid` and `st_mode` (an
    `os.stat_result`). `facts`: `keeper_euid`, `owner_uid`, `peer_uid` (None when not per connection),
    `state_dir_uid`, `state_dir_mode` (octal text) and `reasons`, the failed conditions in words."""
    reasons = []
    if keeper_euid == owner_uid:
        reasons.append("the keeper runs under the owner's own uid")
    if state_dir_stat.st_uid != keeper_euid:
        reasons.append("the state directory does not belong to the keeper's uid")
    perm = stat.S_IMODE(state_dir_stat.st_mode)
    if perm & OTHERS_AND_GROUP_BITS:
        reasons.append("the state directory is open to group or others")
    if peer_uid is not None and peer_uid == keeper_euid:
        reasons.append("this peer has the keeper's own uid, so it can read the keeper's state")
    return {"mode": SAME_USER if reasons else SEPARATE_USER,
            "facts": {"keeper_euid": keeper_euid, "owner_uid": owner_uid, "peer_uid": peer_uid,
                      "state_dir_uid": state_dir_stat.st_uid, "state_dir_mode": f"{perm:04o}", "reasons": reasons}}


def claimed_mode(hello: Any) -> str:
    """The mode a keeper's `hello` claims. A keeper that predates `keeper_mode` is read from its `same_user` flag
    (a flag that is not exactly false counts as same-user); a value that is not a mode counts as `same_user`."""
    claim = hello.get("keeper_mode") if isinstance(hello, dict) else None
    if claim in MODES:
        return claim
    if claim is None and isinstance(hello, dict) and hello.get("same_user") is False:
        return SEPARATE_USER
    return SAME_USER


def attested_mode(attested_uid: int | None, own_euid: int, same_account: Iterable[int] = ()) -> str | None:
    """What the kernel's answer about the listener says: `same_user` when its uid is the client's own or one of the
    `same_account` uids, `separate_user` for any other uid, None when the kernel gave none."""
    if attested_uid is None:
        return None
    return SAME_USER if attested_uid == own_euid or attested_uid in same_account else SEPARATE_USER


def effective_client_mode(claimed: str, attested_uid: int | None, own_euid: int,
                          same_account: Iterable[int] = ()) -> tuple[str, str]:
    """AN-29: `(mode, note)` for a client that asked the kernel who listens on the keeper's socket
    (`protocol.attested_listener_uid`). A claim never upgrades: `separate_user` stands only when the claim is
    `separate_user` and the kernel names a listener that is neither the client's own uid nor in `same_account` (the
    uids of the accounts that wrote the project's files: whoever wrote the records can run a keeper of its own, so a
    reader under a third account, an admin or CI, must not take that listener for a separate user). A listener of
    one of those uids that claims `separate_user` is a contradiction (a process of the client's, or of the project
    owner's, own user posing as the keeper) and is refused."""
    if claimed == SAME_USER or claimed not in MODES:
        return SAME_USER, "the keeper states same-user mode"
    if attested_uid is None:
        return SAME_USER, "kernel attestation unavailable: the keeper's claim cannot be confirmed"
    if attested_uid == own_euid:
        raise Refused([Problem(
            RULE_ATTESTATION, "keeper", "the keeper claims to run under another OS user, but the kernel says the "
            "process serving this socket has your own uid",
            "this socket is served by a process of your own uid; it is not the keeper you were bound to: start the "
            "real keeper")])
    if attested_uid in same_account:
        raise Refused([Problem(
            RULE_ATTESTATION, "keeper", "the keeper claims to run under another OS user, but the kernel says the "
            "process serving this socket has the uid of the account that owns the project's files",
            "this socket is served by a process of the account that wrote the project (an agent can run a keeper "
            "of its own with a key it holds); it is not the keeper you were bound to: start the real keeper, or "
            "anchor with a fingerprint taken from the real keeper")])
    return SEPARATE_USER, "the keeper states separate-user mode and the kernel names another uid as its listener"


def mode_of_text(guarantee: Any) -> str | None:
    """The mode a signed guarantee text states, for a report that carries no `keeper_mode`: the two same-user texts
    give `same_user`; the separate-user texts (the current one, the rootful-access one, the revision 23 one) give
    `separate_user` (the rootful flag is separate, see `verify`); any other text gives None."""
    from . import texts          # texts names the modes, so this module cannot import it at the top

    if guarantee in (texts.KEEPER_SAME_USER_GUARANTEE, texts.KEEPER_OWNER_RUN_GUARANTEE):
        return SAME_USER
    if guarantee in (texts.KEEPER_GUARANTEE, texts.KEEPER_ROOT_DOCKER_GUARANTEE, texts.KEEPER_GUARANTEE_V1):
        return SEPARATE_USER
    return None

