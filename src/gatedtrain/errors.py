"""Readable refusal errors: every error
names the rule, the field and the fix. Printed one per line as

    error[<RULE>] <field>: <message> | fix: <fix>

(tests/README.md section 3).
"""

from __future__ import annotations

from dataclasses import dataclass


def printable(text: str, keep: str = "") -> str:
    """`text` with every non-printable character (escape, bell, NUL, bidi and line separators, ...) written as its
    Python escape (`\\x1b`), so a value that came from a user file or the environment cannot move the cursor, colour
    the terminal or forge a line. Printable text, accents and CJK included, is unchanged, and so is any character named in `keep`."""
    return "".join(c if c.isprintable() or c in keep else c.encode("unicode_escape").decode("ascii") for c in text)


@dataclass(frozen=True)
class Problem:
    rule: str
    field: str
    message: str
    fix: str

    def line(self) -> str:
        # One line, whatever the parts hold: a field or message can carry text from a file the user controls.
        msg = printable(" ".join(self.message.split()))
        fix = printable(" ".join(self.fix.split()))
        return f"error[{printable(self.rule)}] {printable(self.field)}: {msg} | fix: {fix}"


class Refused(Exception):
    """The command refuses to start (exit code 3)."""

    def __init__(self, problems: list[Problem]) -> None:
        super().__init__("; ".join(p.line() for p in problems))
        self.problems = list(problems)
