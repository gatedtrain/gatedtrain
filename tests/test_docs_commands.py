"""Doc-vs-CLI: every `gatedtrain ...` command line the sudo-free docs, the image READMEs and the build script's usage
text promise is parsed with the real argument parser, so a documented flag that does not exist (for example
`image list --json`, where `--json` is a top-level option) fails here instead of in a user's terminal.

The extraction is deliberately simple: a fenced-block line or an inline code span that starts with `gatedtrain`,
with the usual documentation notation reduced to plain arguments (`[optional]` dropped, `<placeholder>` replaced,
`a|b` takes the first alternative, a trailing comment, redirect or `...` removed). A line that holds `...` and an
inline span in prose (which often names a command without its operands) may leave out arguments the parser requires;
any other problem (unknown option, wrong placement, bad choice) fails.
"""

from __future__ import annotations

import re
import shlex
import unittest
from pathlib import Path

from gatedtrain import cli

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ("docs/sudo-free-quickstart.md", "docs/rootless-nemo-rl.md", "docs/configuration.md",
           "containers/nemo-rl-tierb/README.md", "containers/nemo-rl-tierb/build.sh",
           "containers/nemo-rl-uid1000/README.md", "containers/nemo-rl-rootless/README.md")
SUBCOMMANDS = ("image", "setup", "keeper", "verify", "project")      # the commands this release's docs introduce

FENCE = re.compile(r"^(?P<ind>[ ]*)```[^\n]*\n(?P<body>.*?)^(?P=ind)```[ ]*$", re.S | re.M)
INLINE = re.compile(r"`([^`\n]+(?:\n[^`\n]+)*)`")
REQUIRED = "the following arguments are required"


def normalise(raw: str) -> tuple[list[str], bool] | None:
    """(argv after `gatedtrain`, whether the line had an ellipsis) or None when `raw` is not a command line."""
    text = " ".join(raw.replace("\\\n", " ").split())
    text = re.sub(r"^\$ ", "", text)
    if not text.startswith("gatedtrain "):
        return None
    text = re.sub(r"\$\([^)]*\)", "X", text)                    # a shell substitution
    text = re.sub(r"\s#.*$", "", text)                          # a trailing comment
    text = re.split(r"\s(?:>|;|&&|\|\|?)\s", text)[0]          # a redirect or a second command
    text = re.sub(r"\[[^\]]*\]", "", text)                      # optional groups
    text = re.sub(r"<[^>]*>", "X", text)                        # placeholders
    text = re.sub(r"\b(?:or|then)\b:?.*$", "", text) if " or: " in text else text
    ellipsis = "..." in text
    text = text.replace("...", "")
    text = re.sub(r"(?<=\s)(\S+)\|\S+", r"\1", text)            # `host|keeper` -> `host`
    text = text.rstrip(" .,)")
    try:
        argv = shlex.split(text)[1:]
    except ValueError:
        return None
    first = next((t for t in argv if not t.startswith("-")), None)
    return (argv, ellipsis) if first in SUBCOMMANDS else None


def command_lines() -> list[tuple[str, list[str], bool, str]]:
    found = []
    for name in SOURCES:
        text = (ROOT / name).read_text(encoding="utf-8")
        prose = FENCE.sub("", text)
        spans = [m.group("body") for m in FENCE.finditer(text)]
        lines = []
        for body in spans:
            logical = ""
            for line in body.splitlines():
                logical += line.strip().rstrip("\\") + (" " if line.rstrip().endswith("\\") else "")
                if not line.rstrip().endswith("\\"):
                    lines.append(logical)
                    logical = ""
        for raw in lines:
            got = normalise(raw)
            if got is not None:
                found.append((name, got[0], got[1], raw.strip()))
        for m in INLINE.finditer(prose):                          # prose: operands may be left out
            got = normalise(m.group(1))
            if got is not None:
                found.append((name, got[0], True, m.group(1).strip()))
    return found


class DocumentedCommandsTest(unittest.TestCase):
    def test_the_extraction_finds_the_commands_the_docs_promise(self):
        seen = {" ".join(a[:2]) for _, a, _, _ in command_lines()}
        for expected in ("image list", "image check", "image add", "setup keeper", "setup check", "setup layout",
                         "keeper fingerprint", "verify ./proj", "project create"):
            self.assertIn(expected, seen)
        self.assertGreater(len(command_lines()), 25)

    def test_every_documented_command_line_parses(self):
        for name, argv, ellipsis, raw in command_lines():
            with self.subTest(file=name, line=raw[:90]):
                parser = cli.build_parser()
                try:
                    cli._Parser.json_errors = True              # raise instead of printing and exiting
                    parser.parse_args(argv)
                except cli._UsageError as exc:
                    if not (ellipsis and str(exc).startswith(REQUIRED)):
                        self.fail(f"{raw!r} -> {argv}: {exc}")
                finally:
                    cli._Parser.json_errors = False

    def test_json_is_documented_as_a_top_level_option_only(self):
        for name in SOURCES:
            text = (ROOT / name).read_text(encoding="utf-8")
            for m in re.finditer(r"gatedtrain (?:image|setup|keeper|verify|project)\b[^`\n#]*--json", text):
                self.fail(f"{name}: `{m.group(0)}`: --json goes before the subcommand (`gatedtrain --json ...`)")

    def test_the_checker_rejects_the_old_wrong_form(self):
        argv, _ = normalise("gatedtrain image list --json")
        cli._Parser.json_errors = True
        try:
            with self.assertRaises(cli._UsageError):
                cli.build_parser().parse_args(argv)
        finally:
            cli._Parser.json_errors = False


if __name__ == "__main__":
    unittest.main()
