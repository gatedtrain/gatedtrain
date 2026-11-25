"""`gatedtrain image list|check|add`: the image lock from the command line (IMG-5, IMG-6).

`list` shows every accepted image and where it came from (the shipped lock or the owner's config). `check` compares a
local image's ID with the accepted set. `add` records an image the owner built as an explicit, owner-side act in the
owner's config; it only adds (there is no remove command: the file is the owner's), it needs a terminal or `--yes`,
and it never reads the environment for paths. The keeper validates against its own view of the shipped lock and the
report flags a user-added image (`gatedtrain.ainative.image_provenance`). Adding here is recorded for provenance
only: published packs read the shipped lock, so it changes nothing for gating until packs adopt
`accepted_images(logical)`. The config is the owner's file but not a boundary against a same-user agent
(`gatedtrain.imagelock`).

Rules: IMG-5 the image ID could not be computed (docker failed or printed no ID); IMG-6 `add` was refused (no terminal
and no `--yes`, a missing recipe fact, an invalid value or an untrustworthy existing config).
"""

from __future__ import annotations

import datetime
import re
import subprocess
import sys

from . import __version__, imagelock
from .errors import Problem, Refused
from .output import Result, emit

INSPECT_TIMEOUT_S = 60           # `docker image inspect` reads the local store; a longer wait is a hung daemon
EXIT_OK, EXIT_NOT_PASS, EXIT_USAGE, EXIT_REFUSED = 0, 1, 2, 3
USER_ADDED = "user-added: the keeper's report will flag this image as not in the maintainer lock"
GATING_NOTE = ("recorded for provenance; packs adopt accepted_images(logical) in a later release (published packs "
               "read the shipped lock only, so this changes nothing for gating yet)")
LOCK_HASH_NOTE = ("sha256 of the maintainer lock this version read; informational until release hashes are published "
                  "to compare it with")
RECIPES_HINT = ("the build recipes are under containers/ in the gatedtrain source repository, not in the installed "
                "package (README.md there)")
SOURCE_ONLY = ("containers/ is not shipped in the wheel or sdist: the build recipes need a source checkout of "
               "gatedtrain")
# A value handed to `docker` as a reference or a context name must not be read as an option (`-v`, `--host=...`):
# docker names start with a letter or digit, so a leading `-` is never a name.
NOT_AN_OPTION = re.compile(r"[^-\s].*")


def _run(argv: list[str], timeout: float) -> tuple[int, str]:
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return 127, ""
    return p.returncode, p.stdout


RUNNER = _run                        # replaced by tests


def _stdin_is_tty() -> bool:
    return sys.stdin.isatty()


def _ask(prompt: str) -> str:
    return input(prompt)


def _finish(args, data: dict, human: str = "", *, ok: bool = True, code: int = EXIT_OK) -> int:
    return emit(Result(f"image {args.image_cmd}", ok, code, data, [], []), as_json=bool(getattr(args, "json", False)),
                human=human)


def _refuse(args, problems: list[Problem], code: int = EXIT_REFUSED) -> int:
    return emit(Result(f"image {args.image_cmd}", False, code, {}, list(problems), []),
                as_json=bool(getattr(args, "json", False)), human="")


def _row(a: imagelock.Accepted) -> dict:
    return {"logical": a.logical, "id": a.id, "origin": a.origin, "recipe": a.recipe, "base_id": a.base_id,
            "built_utc": a.built_utc, "added_by": a.added_by, "note": a.note, "source": a.source}


def _rows(lock: dict) -> list[dict]:
    cfg = imagelock.load_user_config(lock=lock)
    return [_row(a) for logical in lock["images"] for a in imagelock.resolve(logical, lock=lock, config=cfg)]


def cmd_list(args) -> int:
    try:
        lock, lock_sha = imagelock.load_lock_with_sha()
        rows = _rows(lock)
    except Refused as exc:
        return _refuse(args, exc.problems)
    # The hash of the bytes just parsed. Informational: a release does not publish its lock hash yet, so there is
    # nothing authoritative to compare it with (the keeper's report carries the same hash, `verify` prints it).
    lines = [f"lock: {lock_sha} ({LOCK_HASH_NOTE})"]
    for logical, body in lock["images"].items():
        mine = [r for r in rows if r["logical"] == logical]
        lines.append(f"{logical}: {body['purpose']}")
        for r in mine:
            rec = r["recipe"]
            lines.append(f"  {r['id']}  {r['origin']}  recipe {rec['name']} {rec['commit'] or '-'}  [{r['source']}]")
        if not mine:
            lines.append("  (no accepted image)")
    return _finish(args, {"images": rows, "lock_sha256": lock_sha}, "\n".join(lines) + "\n")


def _image_id(args) -> str:
    """The ID to check: `--id`, or the one `docker image inspect` reports for `--ref` (IMG-5)."""
    if args.id is not None:
        return args.id
    for field, value in (("ref", args.ref), ("docker_context", args.docker_context)):
        if value is not None and not NOT_AN_OPTION.fullmatch(value):
            raise Refused([Problem("IMG-5", field, f"{value!r} is empty or starts with '-', which docker would read as "
                                   "an option", "pass an image reference or Docker context name, or use --id")])
    argv = ["docker"] + (["--context", args.docker_context] if args.docker_context else []) + \
        ["image", "inspect", "-f", "{{.Id}}", "--", args.ref]
    rc, out = RUNNER(argv, INSPECT_TIMEOUT_S)
    found = out.strip()
    if rc != 0 or not imagelock.ID_RE.fullmatch(found):
        raise Refused([Problem("IMG-5", "ref", f"docker could not report an image ID for {args.ref!r}",
                               "check the reference and the Docker context (`--docker-context`), or pass `--id`")])
    return found


def _add_command(logical: str, image_id: str) -> str:
    return (f"gatedtrain image add {logical} --id {image_id} --recipe <RECIPE-NAME> --recipe-commit <COMMIT> "
            "[--containerfile-sha256 <HASH>]")


def cmd_check(args) -> int:
    if (args.id is None) == (args.ref is None):
        return _refuse(args, [Problem("USAGE", "argv", "pass exactly one of --ref and --id", "see `image check --help`")],
                       EXIT_USAGE)
    try:
        if args.id is not None and not imagelock.ID_RE.fullmatch(args.id):
            raise Refused([Problem("IMG-5", "id", "is not sha256:<64 lowercase hex>", "pass the full image ID")])
        image_id = _image_id(args)
        accepted = imagelock.accepted_ids(args.logical)
    except Refused as exc:
        return _refuse(args, exc.problems)
    data = {"logical": args.logical, "id": image_id, "accepted": image_id in accepted}
    if data["accepted"]:
        return _finish(args, data, f"accepted: {image_id} is an accepted {args.logical} image\n")
    build = next((e["recipe"]["build"] for e in imagelock.load_lock()["images"][args.logical]["accepted"]
                  if e["recipe"]["build"]), None)
    data["add_command"] = _add_command(args.logical, image_id)
    human = (f"NOT accepted: {image_id} is not an accepted {args.logical} image\n"
             f"to accept an image you built (a recorded, owner-side act): {data['add_command']}\n"
             f"recipe: {build + '/README.md in a source checkout' if build else RECIPES_HINT}\n"
             f"{SOURCE_ONLY}\n")
    return _finish(args, data, human, ok=False, code=EXIT_NOT_PASS)


def cmd_add(args) -> int:
    try:
        lock = imagelock.load_lock()
        if args.logical not in lock["images"]:
            raise imagelock._unknown(args.logical)
        if imagelock.in_maintainer_lock(args.id, lock=lock):
            return _finish(args, {"logical": args.logical, "id": args.id, "user_added": False, "written": False},
                           f"nothing to add: {args.id} is already in the maintainer lock\n")
        with imagelock.locked_config():
            return _add_locked(args, lock)
    except Refused as exc:
        probs = [Problem("IMG-6", p.field, p.message, p.fix) if p.rule == "IMG-2" and p.field != "file" else p
                 for p in exc.problems]
        return _refuse(args, probs)
    except OSError as exc:
        return _refuse(args, [Problem("IMG-6", "file", f"cannot lock the owner's image config ({type(exc).__name__})",
                                      "check the permissions of its directory")])


def _add_locked(args, lock: dict) -> int:
    """The read-modify-write of `add`, under `imagelock.locked_config`."""
    try:
        cfg = imagelock.load_user_config(lock=lock)
        entry = {"id": args.id, "origin": imagelock.USER,
                 "recipe": {"name": args.recipe, "source": None, "commit": args.recipe_commit,
                            "containerfile_sha256": args.containerfile_sha256, "build": None},
                 "base_id": args.base_id,
                 "built_utc": args.built_utc or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
                 "added_by": f"gatedtrain {__version__} image add", "note": args.note}
        body = cfg["images"].setdefault(args.logical, {"accepted": []})
        if any(e["id"] == args.id for e in body["accepted"]):
            return _finish(args, {"logical": args.logical, "id": args.id, "user_added": True, "written": False},
                           f"already recorded: {args.id}\n{USER_ADDED}\n")
        body["accepted"].append(entry)
        imagelock.validate_config(cfg, lock)                           # refuse before asking anything
        summary = (f"image {args.id}\n  logical image: {args.logical}\n  recipe: {args.recipe} "
                   f"{args.recipe_commit or '-'} {args.containerfile_sha256 or ''}".rstrip() + "\n"
                   f"  file: {imagelock.user_config_path()}\n")
        if not args.yes:
            if not _stdin_is_tty():
                raise Refused([Problem("IMG-6", "yes", "adding an accepted image needs a person at a terminal, and "
                                       "stdin is not one", "run it in a terminal, or pass --yes to confirm")])
            if _ask(summary + f"{USER_ADDED}\nType yes to record it: ").strip() != "yes":
                return _finish(args, {"logical": args.logical, "id": args.id, "user_added": False, "written": False},
                               "not recorded\n", ok=False, code=EXIT_NOT_PASS)
        path = imagelock.write_user_config(cfg, lock=lock)
    except Refused as exc:
        probs = [Problem("IMG-6", p.field, p.message, p.fix) if p.rule == "IMG-2" and p.field != "file" else p
                 for p in exc.problems]
        return _refuse(args, probs)
    return _finish(args, {"logical": args.logical, "id": args.id, "user_added": True, "written": True,
                          "path": str(path), "entry": entry, "gating_note": GATING_NOTE},
                   f"recorded in {path}:\n{summary}{USER_ADDED}\n{GATING_NOTE}\n")


def register(sub) -> None:
    p = sub.add_parser("image", help="the image lock: list, check or add the container images gatedtrain accepts")
    s = p.add_subparsers(dest="image_cmd", required=True)
    c = s.add_parser("list", help="every accepted image, with its origin and recipe")
    c.set_defaults(func=cmd_list)
    c = s.add_parser("check", help="is this image accepted for a logical image? exit 0 yes, 1 no")
    c.add_argument("logical", help="a logical image name (see `image list`)")
    c.add_argument("--ref", metavar="REF", help="an image reference the Docker daemon holds")
    c.add_argument("--id", metavar="ID", help="an image ID (sha256:<64 hex>)")
    c.add_argument("--docker-context", metavar="CTX", help="the Docker context to ask for --ref")
    c.set_defaults(func=cmd_check)
    c = s.add_parser("add", help="record an image you built in your own config (adds only; reports flag it)")
    c.add_argument("logical")
    c.add_argument("--id", required=True, metavar="ID")
    c.add_argument("--recipe", required=True, metavar="NAME", help="the recipe the image was built from")
    c.add_argument("--recipe-commit", metavar="COMMIT", help="the 40-hex commit the recipe's source was built at")
    c.add_argument("--containerfile-sha256", metavar="HASH", help="sha256:<64 hex> of the Containerfile used")
    c.add_argument("--base-id", metavar="ID")
    c.add_argument("--built-utc", metavar="YYYY-MM-DD", help="default: today (UTC)")
    c.add_argument("--note", metavar="TEXT")
    c.add_argument("--yes", action="store_true", help="record without the terminal prompt")
    c.set_defaults(func=cmd_add)
