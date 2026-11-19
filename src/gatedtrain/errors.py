"""Readable refusal errors: every error
names the rule, the field and the fix. Printed one per line as

    error[<RULE>] <field>: <message> | fix: <fix>

(tests/README.md section 3).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Problem:
    rule: str
    field: str
    message: str
    fix: str

    def line(self) -> str:
        msg = " ".join(self.message.split())
        fix = " ".join(self.fix.split())
        return f"error[{self.rule}] {self.field}: {msg} | fix: {fix}"


class Refused(Exception):
    """The command refuses to start (exit code 3)."""

    def __init__(self, problems: list[Problem]) -> None:
        super().__init__("; ".join(p.line() for p in problems))
        self.problems = list(problems)
