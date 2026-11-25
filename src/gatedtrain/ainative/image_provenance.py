"""Image provenance in the signed report (IMG-7 .. IMG-9).

A result should say which trainer image produced it and where that image comes from. The driver reads the image ID of
each job record (`job_image_ids`) and declares, per distinct ID, what the lock files say (`driver_images`: logical
name, origin, recipe, source `lock` | `owner_config` | `unlisted`). The keeper does not trust that declaration
(IMG-8): `check_images` validates its shape and re-derives `in_maintainer_lock` from the keeper's OWN shipped lock,
which lies in its root-owned code; an ID outside it is `user_added`. The signed report holds the entries marked
`declared_by` (the keeper for an ID of its lock, whose logical name, origin and recipe it fills in itself; the driver
otherwise) with the keeper's check beside them, and the sha256 of the keeper's shipped lock (`images_lock_sha256`,
the hash of the very bytes the check used).

The signed row never contradicts itself. For an ID the keeper's lock does not list, the only driver declaration kept is
`source: owner_config` with `origin: user-built`, a logical name and a recipe (what `driver_images` writes for an owner
config entry); anything else (a claim of `source: lock` or `maintainer-built`, a missing recipe, ...) is reduced to
`source: unlisted` with a null logical name, origin and recipe. A driver whose lock is newer than the keeper's thus
shows its image as unlisted, never as maintainer-built. The same ID twice is refused (IMG-8). `validate_report_images`
checks these rules again on the reading side (`verify`, `report.md`): a row that breaks them is AN-14, never printed.

LIMIT: the job records are in the agent-writable project, so this is the driver's declaration of the images, not proof
of what ran; the reliable part is the keeper's flag, which holds when the keeper's code is outside the agent's reach.
IMG-7: `render_lines` prints one line per image in `report.md`. IMG-9: `verify` warns about each user-added image and
about a report whose jobs have no image list, prints the lock hash, and `--require-maintainer-images` turns either
warning into exit 1 and also fails a report that states no images at all (no `compute`, no backend, zero jobs or an
empty list: none of them states any provenance): the flag passes only when every checked report lists at least one
image and every listed image is in the keeper's lock.
"""

from __future__ import annotations

from typing import Any

from .. import imagelock
from ..errors import Problem, Refused
from ..output import notice

MAX_IMAGES = 64                     # distinct trainer images in one project; far above any real project
SOURCES = ("lock", "owner_config", "unlisted")
DECLARED_BY = "driver"
KEEPER_LOCK = "keeper"                  # `declared_by` of an entry the keeper filled in from its own lock
_ROW_KEYS = frozenset({"id", "logical", "origin", "recipe", "in_maintainer_lock", "source"})
_RECIPE_KEYS = frozenset({"name", "source", "commit", "containerfile_sha256", "build"})


def _job_records(project) -> list[dict]:
    from .backend import JOB_FILE, JOBS_DIR, LocalBackend      # backend imports keeper_trust, which imports this lazily

    be = LocalBackend(project)
    return [be._record(job_id) for job_id in be._job_ids()
            if (project.root / JOBS_DIR / job_id / JOB_FILE).exists()]     # a submit cut short has no record yet, as in `jobs()`


def job_image_ids(project) -> list[str]:
    """The distinct image IDs (sha256:<64 hex>) the project's job records ran in, sorted; a runtime image that is not
    an ID is not listed (jobs are pinned by ID, MM-32)."""
    ids = set()
    for rec in _job_records(project):
        image = ((rec.get("job") or {}).get("runtime") or {}).get("image")
        if isinstance(image, str) and imagelock.ID_RE.fullmatch(image):
            ids.add(image)
    return sorted(ids)


def provenance_rows(ids: list[str], *, lock: dict | None = None, config: dict | None | bool = False) -> list[dict]:
    return [imagelock.provenance(i, lock=lock, config=config) for i in ids]


def driver_images(project, *, ids: list[str] | None = None) -> list[dict]:
    """The driver's declaration for the project's trainer images. An owner config that cannot be read never blocks the
    report: the images then show as `unlisted` unless the shipped lock names them."""
    found = job_image_ids(project) if ids is None else ids
    try:
        return provenance_rows(found)
    except (Refused, RecursionError) as exc:
        why = ", ".join(p.rule for p in exc.problems) if isinstance(exc, Refused) else type(exc).__name__
        notice(f"the owner's image config could not be read ({why}); trainer images outside the maintainer lock are "
               "shown as unlisted (run `gatedtrain image list` for the reason)")
        return provenance_rows(found, config=None)


def _bad(message: str) -> Refused:
    return Refused([Problem("IMG-8", "compute.images", message, "send the images `backend_compute` builds, or leave "
                            "images out")])


LOCK_SHA_KEY = "images_lock_sha256"          # in `compute.backend`: the keeper's shipped lock, for a verifier to compare
NOT_DECLARED = ("no image provenance stated: the report carries no list of trainer images (trainer images not declared "
                "by the driver; job records are in the agent-writable project, and an image block is the driver's "
                "declaration, not proof of what ran)")
NOT_DECLARED_REQUIRED = "images not declared: --require-maintainer-images cannot be confirmed for this report"


def _opt_str(v: Any) -> bool:
    """None, or one printable line (no control character: the text reaches `report.md` and CLI output)."""
    return v is None or imagelock.is_text_line(v)


def _opt_match(v: Any, rx) -> bool:
    return v is None or (isinstance(v, str) and rx.fullmatch(v) is not None)


def _valid_recipe(r: Any) -> bool:
    return r is None or (isinstance(r, dict) and set(r) == _RECIPE_KEYS and _opt_str(r["name"]) and r["name"] is not None
                         and _opt_str(r["source"]) and _opt_str(r["build"]) and _opt_match(r["commit"], imagelock.COMMIT_RE)
                         and _opt_match(r["containerfile_sha256"], imagelock.ID_RE))


def _lock_entry(image_id: str, lock: dict) -> tuple[str, dict] | None:
    for logical, body in lock["images"].items():
        for e in body["accepted"]:
            if e["id"] == image_id:
                return logical, e
    return None


def _driver_row_ok(row: Any) -> bool:
    return (isinstance(row, dict) and set(row) == _ROW_KEYS and isinstance(row["id"], str)
            and imagelock.ID_RE.fullmatch(row["id"]) is not None
            and _opt_match(row["logical"], imagelock.NAME_RE) and _opt_str(row["logical"])
            and row["origin"] in (None, imagelock.MAINTAINER, imagelock.USER)
            and isinstance(row["in_maintainer_lock"], bool) and row["source"] in SOURCES
            and _valid_recipe(row["recipe"]))


def _unlisted(image_id: str) -> dict:
    return {"id": image_id, "logical": None, "origin": None, "recipe": None, "source": "unlisted",
            "in_maintainer_lock": False, "user_added": True, "declared_by": DECLARED_BY,
            "checked_by_keeper": {"in_maintainer_lock": False}}


def _check(images: Any, lk: dict) -> list[dict]:
    if not isinstance(images, list) or len(images) > MAX_IMAGES:
        raise _bad(f"images is not a list of at most {MAX_IMAGES} entries")
    out, seen = [], set()
    for row in images:
        if not (isinstance(row, dict) and set(row) == _ROW_KEYS):
            raise _bad("an image entry does not have exactly {id, logical, origin, recipe, in_maintainer_lock, source}")
        if not _driver_row_ok(row):
            raise _bad("an image entry has a malformed value")
        if row["id"] in seen:
            raise _bad("an image ID is listed twice")
        seen.add(row["id"])
        found = _lock_entry(row["id"], lk)
        if found is not None:
            logical, e = found
            out.append({"id": row["id"], "logical": logical, "origin": e["origin"], "recipe": dict(e["recipe"]),
                        "source": "lock", "in_maintainer_lock": True, "user_added": False, "declared_by": KEEPER_LOCK,
                        "checked_by_keeper": {"in_maintainer_lock": True}})
            continue
        entry = _unlisted(row["id"])
        if (row["source"] == "owner_config" and row["origin"] == imagelock.USER and row["logical"] is not None
                and row["recipe"] is not None):
            entry.update(logical=row["logical"], origin=row["origin"], recipe=row["recipe"], source="owner_config")
        out.append(entry)
    return out


def check_images(images: Any, *, lock: dict | None = None) -> list[dict]:
    """IMG-8: validate the driver's `images` (exact keys, caps, one printable line per string, no duplicate ID) and
    return the report entries. An ID in the keeper's OWN shipped lock gets its logical name, origin and recipe from
    that lock (what the driver declared is overwritten; `declared_by: "keeper"`). An ID outside it is `user_added`
    and keeps only a declaration that cannot contradict that (see the module docstring): the owner-config pair
    (`owner_config`, `user-built`, logical name, recipe), else `unlisted`. `in_maintainer_lock` is always the
    keeper's own answer."""
    return _check(images, lock if lock is not None else imagelock.load_lock())


def checked_images(images: Any, *, lock: dict | None = None) -> tuple[list[dict], str | None]:
    """`check_images` together with the sha256 of the lock bytes it used (None for a `lock` passed in): the one read
    that makes the report's `images_lock_sha256` the hash of the lock the entries were derived from."""
    if lock is not None:
        return _check(images, lock), None
    lk, sha = imagelock.load_lock_with_sha()
    return _check(images, lk), sha


_REPORT_KEYS = _ROW_KEYS | {"user_added", "declared_by", "checked_by_keeper"}


def _bad_report(message: str) -> Refused:
    return Refused([Problem("AN-14", "compute.backend.images", message, "the report's image block is malformed; do "
                            "not trust the report, check the keeper")])


def validate_report_images(images: Any) -> list[dict]:
    """AN-14: the shape and the self-consistency of the image entries of a report the keeper signed. Refuses a block
    that is not a list, a row with another key set or a wrong type, a duplicate ID, a row that contradicts itself
    (`user_added` with `in_maintainer_lock`, a lock row not declared by the keeper or without `source: lock`, an
    unlisted row with a name, an origin or a recipe) and text that is not one printable line. Returns `images`."""
    if not isinstance(images, list) or len(images) > MAX_IMAGES:
        raise _bad_report(f"images is not a list of at most {MAX_IMAGES} entries")
    seen = set()
    for row in images:
        if not (isinstance(row, dict) and set(row) == _REPORT_KEYS and _driver_row_ok({k: row[k] for k in _ROW_KEYS})):
            raise _bad_report("an image entry is malformed")
        locked, check = row["in_maintainer_lock"], row["checked_by_keeper"]
        if not (isinstance(row["user_added"], bool) and row["declared_by"] in (DECLARED_BY, KEEPER_LOCK)
                and isinstance(check, dict) and check == {"in_maintainer_lock": locked}):
            raise _bad_report("an image entry is malformed")
        if row["id"] in seen:
            raise _bad_report("an image ID is listed twice")
        seen.add(row["id"])
        if locked:
            ok = (not row["user_added"] and row["declared_by"] == KEEPER_LOCK and row["source"] == "lock"
                  and row["origin"] == imagelock.MAINTAINER and row["logical"] is not None and row["recipe"] is not None)
        else:
            plain = row["logical"] is None and row["origin"] is None and row["recipe"] is None
            ok = (row["user_added"] and row["declared_by"] == DECLARED_BY and
                  ((row["source"] == "unlisted" and plain) or
                   (row["source"] == "owner_config" and row["origin"] == imagelock.USER
                    and row["logical"] is not None and row["recipe"] is not None)))
        if not ok:
            raise _bad_report("an image entry contradicts itself (origin, source, user_added and the keeper's check "
                              "do not agree)")
    return images


def render_lines(images: list[dict]) -> list[str]:
    """IMG-7: one `report.md` line per image, read from the report alone. A malformed block is refused (AN-14), never
    rendered: `verify` re-renders every report, and a signed but malformed entry must not crash it."""
    lines = []
    for im in validate_report_images(images):
        recipe = im.get("recipe")
        built = f"{recipe['name']} {recipe['commit'] or '-'}" if isinstance(recipe, dict) else "unknown"
        where = ("in the maintainer lock: the keeper recognises this ID" if im["in_maintainer_lock"]
                 else "USER-ADDED, not in the maintainer lock")
        lines.append(f"- trainer image {im['logical'] or 'unlisted'}: {im['origin'] or 'unknown'} {im['id']} built from "
                     f"recipe {built} ({where}; the driver declares that jobs ran in it)")
    return lines


def user_added_images(report: dict) -> list[str]:
    """The IDs a report flags as not in the maintainer lock (none for a report without images)."""
    backend = ((report.get("compute") or {}).get("backend")) if isinstance(report, dict) else None
    imgs = backend.get("images") if isinstance(backend, dict) else None
    if imgs is None:
        return []
    return [i["id"] for i in validate_report_images(imgs) if i["user_added"]]


def user_added_warnings(ids: list[str]) -> list[str]:
    """IMG-9: the `verify` warning for each user-added image."""
    return [f"trainer image {i} is user-added (not in the maintainer lock)" for i in ids]


def _backend(report: Any) -> dict | None:
    b = ((report.get("compute") or {}).get("backend")) if isinstance(report, dict) and isinstance(report.get("compute"), dict) else None
    return b if isinstance(b, dict) else None


IMAGES_STATED, IMAGES_NOT_STATED, IMAGES_JOBS_UNDECLARED = 0, 1, 2     # levels of `images_undeclared`, worst is largest


def images_undeclared(report: dict) -> int:
    """How well a report states its trainer images (a level, so `verify` keeps the worst of its reports).
    `IMAGES_STATED` (0): the backend totals carry a non-empty `images` list (each entry is checked by
    `validate_report_images`). `IMAGES_JOBS_UNDECLARED` (2): the totals count jobs but the list is missing or empty
    (an older driver, or one that left it out): the report ran trainer jobs and says nothing about their images.
    `IMAGES_NOT_STATED` (1): any other report without a list (no `compute`, no backend, zero jobs): nothing is stated,
    and nothing ran in the project's backend, so it only matters to `--require-maintainer-images`."""
    b = _backend(report)
    imgs = b.get("images") if b else None
    if isinstance(imgs, list) and imgs:
        return IMAGES_STATED
    jobs = b.get("jobs") if b else None
    return IMAGES_JOBS_UNDECLARED if isinstance(jobs, int) and not isinstance(jobs, bool) and jobs > 0 \
        else IMAGES_NOT_STATED


def report_lock_sha256(report: dict) -> str | None:
    """The sha256 of the keeper's shipped lock a report states (None when it states none or something malformed)."""
    b = _backend(report)
    v = b.get(LOCK_SHA_KEY) if b else None
    return v if isinstance(v, str) and imagelock.ID_RE.fullmatch(v) else None


def image_check(out: dict, require_maintainer: bool) -> tuple[list[str], bool]:
    """IMG-9: the `verify` warnings for the images of the checked reports, and whether `--require-maintainer-images`
    makes the exit 1: a user-added image, or any report without a non-empty keeper-checked image list. Without the
    flag a report that counts jobs but lists no image warns; a report that ran no jobs stays silent (a warning on
    every such report would be a constant, so it would carry no information)."""
    added, level = out.get("user_added_images") or [], int(out.get("images_undeclared") or 0)
    warnings = user_added_warnings(added)
    if level >= IMAGES_JOBS_UNDECLARED or (require_maintainer and level):
        warnings.append(NOT_DECLARED)
    if require_maintainer and level:
        warnings.append(NOT_DECLARED_REQUIRED)
    return warnings, require_maintainer and bool(added or level)


def lock_line(out: dict) -> str:
    """The `verify` line naming the keeper's lock hash(es) of the checked reports ('' when none)."""
    hashes = out.get("image_lock_sha256") or []
    if not hashes:
        return ""
    return ("keeper's image lock sha256: " + ", ".join(hashes) + "\n"
            "  in same-user mode the lock file is writable by the user; no release hash list is published yet - "
            "compare with a trusted copy\n")
