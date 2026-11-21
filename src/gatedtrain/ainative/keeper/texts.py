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


# ---- report v2 known limits (spec §5.4; preflight finding 16) -----------------------------------------------------

LIMIT_KEEPER_MODEL_REFERENCE = ("a model reference must content-address its weights; the keeper hashes the model files "
                                "a reference lists (`files`) when it scores, not weights the reference reaches in any "
                                "other way")

LIMIT_ALPHA_RULE = ("the test level of each attempt is set by alpha-investing (Foster-Stine) over every sealed "
                    "evaluation on this sealed set, from every project that uses it; the guarantee is the marginal "
                    "false discovery rate (mFDR) at level 0.05 over those attempts, not an error rate per attempt")


def sealed_evaluations_text(n: int) -> str:
    return (f"the sealed set was scored in {n} sealed evaluation{'' if n == 1 else 's'} across the projects that use "
            "it (crashed ones included); each spent its alpha before it was scored, and this report closes the set")


def known_limits(guarantee: str, n_attempts: int) -> list[dict]:
    """The known limits of a keeper report (`schema_version` 2)."""
    return [{"id": "sealed_guarantee", "text": guarantee},
            {"id": "original_data", "text": LIMIT_ORIGINAL_DATA},
            {"id": "metric_confirmation", "text": LIMIT_KEEPER_CONFIRMATION},
            {"id": "model_reference", "text": LIMIT_KEEPER_MODEL_REFERENCE},
            {"id": "alpha_rule", "text": LIMIT_ALPHA_RULE},
            {"id": "sealed_evaluations", "attempts": n_attempts, "text": sealed_evaluations_text(n_attempts)}]
