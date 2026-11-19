"""Names the framework wrote under its earlier names, `ptc-skel` (0.1.0) and `ptc-hub` (0.2.0).

Version 0.2.0 renamed every persisted identifier of `ptc-skel`, and version 0.3.0 renamed them again, to
`gatedtrain`: the schema ids (`gatedtrain.launch_record`, `gatedtrain.item_eval`, ...), the ledger `writer_id`
(`gatedtrain.ledger@<version>`) and `ANALYSIS_CODE_VERSION` (`gatedtrain <version>`). This code reads only the new
names; nothing here translates an old one. The functions below only say, in a refusal, that a value is an old name
and how to move on:

- a registration that records an old analysis code version is brought forward by an amendment (05 section 3.7):
  a new registration that references the prior one by hash (`amends`) and states when it was written
  (`created_at`); `REGISTRATION_AMENDMENT` is the fix text. When it changes nothing but the version (from one of
  `PRE_RENAME_VERSIONS` to this one), it is a version-only amendment and is not post hoc (05 section 3.7, R17-1);
- a record written before a rename (launch record, smoke record, arm metadata, merge document) is read by the
  commit that wrote it; a new run writes new records.
"""

from __future__ import annotations

from . import ANALYSIS_CODE_VERSION, PACKAGE

# The framework's earlier names, oldest first: `ptc-skel` before 0.2.0, `ptc-hub` from 0.2.0 to before 0.3.0.
OLD_PACKAGES = ("ptc-skel", "ptc-hub")
# Their schema id prefixes, e.g. ptc_skel.launch_record, ptc_hub.launch_record, keyed to the package that wrote them.
OLD_SCHEMA_PREFIXES = {"ptc_skel.": "ptc-skel", "ptc_hub.": "ptc-hub"}
# Analysis code versions the framework recorded before a rename. An amendment that moves a registration from one
# of these to ANALYSIS_CODE_VERSION and changes nothing else is a version-only amendment (05 section 3.7, R17-1).
PRE_RENAME_VERSIONS = ("ptc-skel 0.1.0", "ptc-hub 0.2.0")
# The fields a version-only amendment may change. `evaluators[*].manifest_hash` may change too, when `rename_proof`
# shows that the scorer behind it changed only by the name map below (R17-3).
VERSION_ONLY_FIELDS = ("analysis_code_version", "amends", "created_at", "post_hoc", "rename_proof")
# The published old -> new name map, applied in this order to bytes (R17-3: a pre-rename scorer file mapped by it
# must give the renamed scorer byte for byte). The first five pairs are the 0.2.0 rename (`ptc-skel` -> `ptc-hub`),
# the rest the 0.3.0 rename (`ptc-hub` -> `gatedtrain`), so a file of either earlier version maps to this one.
NAME_MAP = (
    (b"ptc-skel 0.1.0", b"ptc-hub 0.2.0"),
    (b"ptc-skel.ledger@0.1.0", b"ptc-hub.ledger@0.2.0"),
    (b"PTC_SKEL_", b"PTC_HUB_"),
    (b"ptc_skel", b"ptc_hub"),
    (b"ptc-skel", b"ptc-hub"),
    (b"ptc-hub 0.2.0", b"gatedtrain 0.3.0"),
    (b"ptc-hub.ledger@0.2.0", b"gatedtrain.ledger@0.3.0"),
    (b"PTC_HUB_", b"GATEDTRAIN_"),
    (b"PTC_", b"GATEDTRAIN_"),
    (b"ptc_recipe_", b"gatedtrain_recipe_"),
    (b"ptc-recipe-", b"gatedtrain-recipe-"),
    (b"ptc_hub", b"gatedtrain"),
    (b"ptc-hub", b"gatedtrain"),
)


def apply_name_map(data: bytes) -> bytes:
    for old, new in NAME_MAP:
        data = data.replace(old, new)
    return data


REGISTRATION_AMENDMENT = (
    "amend the registration (05 section 3.7): write a new registration file, equal to this one except "
    f"`analysis_code_version: {ANALYSIS_CODE_VERSION}`, `amends: sha256:<SHA-256 of this file>`, `created_at` "
    "(RFC 3339 UTC, when you write it) and `post_hoc: false` (a version-only amendment is not post hoc, 05 "
    "section 3.7); keep the prior file next to it, where validation compares the two; then put the new file's "
    "sha256 in the spec's registration uri")


def is_old_analysis_code_version(value: object) -> bool:
    return isinstance(value, str) and value.startswith(tuple(name + " " for name in OLD_PACKAGES))


def old_schema_package(value: object) -> str | None:
    """The earlier package that wrote schema id `value`, or None when `value` is not an old schema id."""
    if isinstance(value, str):
        for prefix, package in OLD_SCHEMA_PREFIXES.items():
            if value.startswith(prefix):
                return package
    return None


def is_old_schema(value: object) -> bool:
    return old_schema_package(value) is not None


def old_schema_note(value: object) -> str:
    """'' unless `value` is a schema id written before a rename; else a sentence to append to the refusal."""
    package = old_schema_package(value)
    if package is None:
        return ""
    return (f" ({value!r} was written by {package}, before the rename to {ANALYSIS_CODE_VERSION}, which reads only "
            f"{PACKAGE}.* ids: read a record written before the rename with the commit that wrote it, or write it "
            "again with this version)")
