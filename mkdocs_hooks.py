"""MkDocs hooks for the documentation site.

The site is built from the same files GitHub shows: the pages under `docs/`, plus `README.md` as the home page and
`CHANGELOG.md` as the changelog, added at build time so that nothing is copied into `docs/`. Links are written for
GitHub, so they are rewritten here for the site: a link to a page of the site becomes a link to that page, and a link
to any other file of the repository becomes a link to that file on GitHub.
"""

from __future__ import annotations

import posixpath
import re
from pathlib import Path

from mkdocs.structure.files import File

ROOT = Path(__file__).resolve().parent
BLOB = "https://github.com/gatedtrain/gatedtrain/blob/main/"

# Repository files that become site pages, by their path in the repository and their path in the site's docs.
GENERATED = {"README.md": "index.md", "CHANGELOG.md": "changelog.md"}

LINK = re.compile(r"\]\((?P<target>[^)\s]+)(?P<title>\s+\"[^\"]*\")?\)")


def on_files(files, config):
    for repo_path, docs_path in GENERATED.items():
        files.append(File.generated(config, docs_path, content=(ROOT / repo_path).read_text(encoding="utf-8")))
    return files


def _repo_path(page_src: str) -> str:
    """The repository path of the page whose docs path is `page_src`."""
    for repo_path, docs_path in GENERATED.items():
        if page_src == docs_path:
            return repo_path
    return f"docs/{page_src}"


def _docs_path(repo_path: str) -> str | None:
    """The docs path of a repository file that is a site page, else None."""
    if repo_path in GENERATED:
        return GENERATED[repo_path]
    if repo_path.startswith("docs/") and repo_path.endswith(".md") and (ROOT / repo_path).is_file():
        return repo_path.removeprefix("docs/")
    return None


def rewrite(target: str, page_src: str) -> str:
    """The site's form of a link `target` written in the page whose docs path is `page_src`."""
    if target.startswith(BLOB):
        rest = target.removeprefix(BLOB)
    elif re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith(("#", "/")):
        return target
    else:
        base = posixpath.dirname(_repo_path(page_src))
        rest = posixpath.normpath(posixpath.join(base, target))
    path, sep, anchor = rest.partition("#")
    if path.startswith("../"):
        return target
    docs_path = _docs_path(path)
    if docs_path is None:
        return BLOB + rest
    rel = posixpath.relpath(docs_path, posixpath.dirname(page_src) or ".")
    return rel + sep + anchor


def on_page_markdown(markdown, page, config, files):
    src = page.file.src_uri
    return LINK.sub(lambda m: f"]({rewrite(m['target'], src)}{m['title'] or ''})", markdown)
