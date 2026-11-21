"""The keeper's guarantee and known-limit texts (design 05 §18.9; spec §5.2, §5.3, §5.7).

`sealed_guarantee` picks the text a keeper states: `KEEPER_GUARANTEE` when it runs as its own OS user, and
`KEEPER_SAME_USER_GUARANTEE` when its uid is the owner's (tests, CI, a development setup), since then nothing is
sealed from the user's account.
"""

from __future__ import annotations

KEEPER_GUARANTEE = ("the sealed items, registrations, claims, ledger and per-item records are held by a separate OS "
                    "user running code the user's account cannot change, and every record is signed; the candidate "
                    "runs inside the keeper process, not in a container (a model format that executes code is "
                    "refused); the original data file and the metric confirmation are only as strong as the user's "
                    "own account; a root user or the keeper's own account can break all of it.")

KEEPER_SAME_USER_GUARANTEE = ("the keeper runs as the owner's own OS user, so anything running as the user can read "
                              "its state and signing key: the records are signed, but nothing is sealed (a test or "
                              "development setup)")

LIMIT_ORIGINAL_DATA = ("the original data file is the user's and stays readable to anything running as the user; run "
                       "intake before handing the project to an agent and keep the original out of its reach")

LIMIT_KEEPER_CONFIRMATION = ("an agent with the user's shell can run `gatedtrain keeper confirm` too; the keeper "
                             "records that the confirmation came from the owner uid, not that a human typed it")


def sealed_guarantee(keeper_uid: int, owner_uid: int) -> str:
    """The guarantee a keeper running as `keeper_uid` for the user `owner_uid` states."""
    return KEEPER_SAME_USER_GUARANTEE if keeper_uid == owner_uid else KEEPER_GUARANTEE
