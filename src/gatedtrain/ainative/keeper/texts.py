"""The keeper's guarantee and known-limit texts (design 05 §18.9; spec §5.2, §5.3, §5.7).

`sealed_guarantee` picks the text a keeper states: `KEEPER_SAME_USER_GUARANTEE` when its uid is the owner's (tests,
CI, a development setup), since then nothing is sealed from the user's account, whatever the rootful-access detection
says; else `KEEPER_ROOT_DOCKER_GUARANTEE` when the detection (AN-21) is positive now or was at any check on the sealed
set; else `KEEPER_GUARANTEE`. `KEEPER_GUARANTEE_V1` is the revision 23 text (gatedtrain 0.4.0), stated before rootful
access was checked: no keeper states it any more, and `verify` exits 1 on a report that carries it (AN-14).
"""

from __future__ import annotations

KEEPER_GUARANTEE_V1 = ("the sealed items, registrations, claims, ledger and per-item records are held by a separate OS "
                       "user running code the user's account cannot change, and every record is signed; the candidate "
                       "runs inside the keeper process, not in a container (a model format that executes code is "
                       "refused); the original data file and the metric confirmation are only as strong as the user's "
                       "own account; a root user or the keeper's own account can break all of it.")

# design 05 §18.9 (revision 25), verbatim: the separate-user text, with the rootful-access check and the keeper's
# inference container.
KEEPER_GUARANTEE = ("the sealed items, registrations, claims, ledger and per-item records are held by a separate OS "
                    "user running code the user's account cannot change, and every record is signed; the keeper "
                    "found no rootful Docker or containerd access by the user's account, nor the account in the "
                    "docker, lxd, incus-admin or disk group, when this was stated or at any earlier check on this "
                    "sealed set (sudo is not checked); a scenario pack's per-item generator runs the candidate "
                    "inside the keeper process, and a batch generator runs it in a container of the keeper's own "
                    "rootless Docker with no network, a read-only root file system and the predictions as its only "
                    "output; either way the candidate is a copy of the model files the keeper hashed, and a model "
                    "format that executes code is refused; an escape from that container through a bug in the "
                    "inference code or the kernel gives the keeper's account; the original data file and the metric "
                    "confirmation are only as strong as the user's own account; a root user, also one reached "
                    "through sudo, or the keeper's own account can break all of it.")

KEEPER_ROOT_DOCKER_GUARANTEE = ("the keeper runs as a separate OS user, but the user's account can reach a rootful "
                                "Docker or containerd daemon or holds a root-equivalent group (docker, lxd, "
                                "incus-admin or disk), or did at a check while this sealed set was in use, and through "
                                "that can read the keeper's state and signing key: the records are signed, but nothing "
                                "is sealed from the user's account; remove the user from those groups, end the "
                                "sessions that still carry them, and take in new data")

KEEPER_SAME_USER_GUARANTEE = ("the keeper runs as the owner's own OS user, so anything running as the user can read "
                              "its state and signing key: the records are signed, but nothing is sealed (a test or "
                              "development setup)")

COMPUTE_BACKEND_SOURCE = "the driver's job records (cooperative, not checked by the keeper)"   # AN-10 `compute`

LIMIT_ORIGINAL_DATA = ("the original data file is the user's and stays readable to anything running as the user; run "
                       "intake before handing the project to an agent and keep the original out of its reach")

LIMIT_KEEPER_CONFIRMATION = ("an agent with the user's shell can run `gatedtrain keeper confirm` too; the keeper "
                             "records that the confirmation came from the owner uid, not that a human typed it")


def sealed_guarantee(keeper_uid: int, owner_uid: int, *, root_docker: bool = False) -> str:
    """The guarantee a keeper running as `keeper_uid` for the user `owner_uid` states; `root_docker`: the rootful-access
    detection (AN-21) is positive now or was at a check on the sealed set."""
    if keeper_uid == owner_uid:
        return KEEPER_SAME_USER_GUARANTEE
    return KEEPER_ROOT_DOCKER_GUARANTEE if root_docker else KEEPER_GUARANTEE


# ---- report v2 known limits (spec §5.4; preflight finding 16) -----------------------------------------------------

LIMIT_KEEPER_MODEL_REFERENCE = ("a model reference must content-address its weights: the keeper copies the files a "
                                "reference lists (`files`) into its own state, hashes the copy and runs only the copy, "
                                "so the hash in the claim is the hash of what ran; weights the reference reaches in "
                                "any other way are not covered")

LIMIT_ALPHA_RULE = ("the test level of each attempt is set by alpha-investing (Foster-Stine) over every sealed "
                    "evaluation on this sealed set, from every project that uses it; the guarantee is the marginal "
                    "false discovery rate (mFDR) at level 0.05 over those attempts, not an error rate per attempt, "
                    "assuming each test is valid given the earlier outcomes; the sealed items are reused across "
                    "attempts and only each attempt's status is revealed")

LIMIT_ORACLES = ("the keeper answers some requests about sealed data with counts, and does not limit how often they are "
                 "asked: opening a project is free and a refused intake stores nothing, so anything running as the "
                 "user can send candidate data files to new projects and learn from the refusals how many of their "
                 "inputs were sealed before (repeated, this narrows down which); a registration refused for its scorer "
                 "names how many sealed labels that scorer cannot take, which reveals counts of label types in the "
                 "sealed part")


def sealed_evaluations_text(n: int) -> str:
    return (f"the sealed set was scored in {n} sealed evaluation{'' if n == 1 else 's'} across the projects that use "
            "it (crashed ones included); each spent its alpha before it was scored, and this report closes the set")


def known_limits(guarantee: str, n_attempts: int) -> list[dict]:
    """The known limits of a keeper report (`schema_version` 2)."""
    return [{"id": "sealed_guarantee", "text": guarantee},
            {"id": "original_data", "text": LIMIT_ORIGINAL_DATA},
            {"id": "oracles", "text": LIMIT_ORACLES},
            {"id": "metric_confirmation", "text": LIMIT_KEEPER_CONFIRMATION},
            {"id": "model_reference", "text": LIMIT_KEEPER_MODEL_REFERENCE},
            {"id": "alpha_rule", "text": LIMIT_ALPHA_RULE},
            {"id": "sealed_evaluations", "attempts": n_attempts, "text": sealed_evaluations_text(n_attempts)}]
