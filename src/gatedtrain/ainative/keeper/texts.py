"""The keeper's guarantee and known-limit texts (design 05 §18.9; spec §5.2, §5.3, §5.7).

`sealed_guarantee` picks the text a keeper states: `KEEPER_SAME_USER_GUARANTEE` when its uid is the owner's (tests,
CI, a development setup), since then nothing is sealed from the user's account, whatever the rootful-access detection
says; else `KEEPER_ROOT_DOCKER_GUARANTEE` when the detection (AN-21) is positive now or was at any check on the sealed
set; else `KEEPER_GUARANTEE`. `KEEPER_GUARANTEE_V1` is the revision 23 text (gatedtrain 0.4.0), stated before rootful
access was checked: no keeper states it any more, and `verify` exits 1 on a report that carries it (AN-14).
"""

from __future__ import annotations

from .modes import SAME_USER, SEPARATE_USER
from ..alpha_investing import TARGET_ALPHA

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

# AN-27 (the same-user mode as a supported tier): what a keeper under the owner's own OS user states from now on. It
# says what is still true (the procedure) and what is not (the data is not sealed from the user's account). The
# text above stays a recognised legacy text, so reports made before this one still verify and still warn.
KEEPER_OWNER_RUN_GUARANTEE = (
    "the keeper ran under the same OS user as the owner (same-user mode). It followed the procedure: the "
    "registration was signed before training, every sealed attempt spent its alpha before it was scored, and a "
    "batch generator's model ran in a container without network access that was given the inputs only, never the "
    "labels (a per-item generator runs inside the keeper process, without that confinement). It does not protect "
    "the sealed items, the labels, the alpha ledger or its signing key from anything running as that user, such as "
    "an agent with a shell, so these records show that the procedure was followed, not that the sealed data stayed "
    "unseen. Run the keeper under a separate OS user (`gatedtrain setup keeper --mode separate-user`, one sudo) for "
    "that guarantee.")

# The one line every surface prints for the observed mode (AN-27); a surface never words a weaker mode like a stronger.
MODE_LINE = {
    SEPARATE_USER: "keeper mode: separate-user (the sealed data and the signing key are held by another OS user)",
    SAME_USER: "keeper mode: same-user (NOT sealed from the user's account: the procedure is followed, the sealed "
               "data is not protected from anything running as the user; to harden: gatedtrain setup keeper --mode "
               "separate-user)"}

# The limit of any statement of the mode that is read without the keeper listening: "separate-user" is then what the key
# holder states (AN-27), and on a same-user keeper the holder is the user's own account.
MODE_STATED_NOTE = ("note: the mode is as stated by the holder of the key (a same-user keeper's key is readable by the "
                    "user, and a signature proves only who wrote the records); the kernel's check of the listener "
                    "(AN-29) holds for the reader's own account and the account that owns the project files; a reader "
                    "under a third account must anchor with a fingerprint taken from the real keeper")

# Appended to a report headline in same-user mode, so "PASS" is never read next to "sealed" without the qualifier.
HEADLINE_SUFFIX_SAME_USER = " (same-user keeper: procedure followed; sealed data not protected from the user's account)"

# Appended to the intake plan's "will be sealed" sentence in same-user mode: the plan must not promise a seal the mode does
# not give.
PLAN_SEALED_SAME_USER = ("The keeper is in same-user mode: the sealed part is kept apart by the procedure only, not "
                         "protected from anything running as your user.")

COMPUTE_BACKEND_SOURCE = "the driver's job records (cooperative, not checked by the keeper)"   # AN-10 `compute`

# The limit of the image provenance in a report and in `verify` (IMG-7): the keeper recognises an image ID, it never saw
# which image ran.
LIMIT_IMAGES_NOT_OBSERVED = ("not observed: job records are the driver's (agent-writable); the keeper recognises an image "
                             "ID but never saw which image ran")

LIMIT_ORIGINAL_DATA = ("the original data file is the user's and stays readable to anything running as the user; run "
                       "intake before handing the project to an agent and keep the original out of its reach")

LIMIT_KEEPER_CONFIRMATION = ("an agent with the user's shell can run `gatedtrain keeper confirm` too; the keeper "
                             "records that the confirmation came from the owner uid, not that a human typed it")


def sealed_guarantee(keeper_uid: int, owner_uid: int, *, root_docker: bool = False, mode: str | None = None) -> str:
    """The guarantee a keeper running as `keeper_uid` for the user `owner_uid` states; `root_docker`: the rootful-access
    detection (AN-21) is positive now or was at a check on the sealed set. `mode` (AN-27): the observed mode; when
    given it decides same-user or not (a same-user keeper's text takes precedence over the rootful one); without it
    the uids decide, as before."""
    same_user = (keeper_uid == owner_uid) if mode is None else mode == SAME_USER
    if same_user:
        return KEEPER_OWNER_RUN_GUARANTEE
    return KEEPER_ROOT_DOCKER_GUARANTEE if root_docker else KEEPER_GUARANTEE


# ---- report v2 known limits (spec §5.4) ---------------------------------------------------------------------------

LIMIT_KEEPER_MODEL_REFERENCE = ("a model reference must content-address its weights: the keeper copies the files a "
                                "reference lists (`files`) into its own state, hashes the copy and runs only the copy, "
                                "so the hash in the claim is the hash of what ran; weights the reference reaches in "
                                "any other way are not covered")

LIMIT_ALPHA_RULE = ("the test level of each attempt is set by alpha-investing (Foster-Stine) over every sealed "
                    "evaluation on this sealed set, from every project that uses it; the guarantee is the marginal "
                    f"false discovery rate (mFDR) at level {TARGET_ALPHA:g} over those attempts, not an error rate "
                    "per attempt, assuming each test is valid given the earlier outcomes; the sealed items are reused "
                    "across attempts and only each attempt's status is revealed")

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
