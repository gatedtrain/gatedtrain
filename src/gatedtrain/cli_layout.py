"""AN-34: `--layout FILE` on `setup host` / `setup keeper`, and `gatedtrain setup layout` (kept out of cli.py; the
schema and the resolution are in ainative/layout.py)."""

from __future__ import annotations

import argparse
import os

from .ainative import layout
from .errors import Problem, Refused


def add_layout_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--layout", metavar="FILE",
                        help="a TOML layout file (see `gatedtrain setup layout`) that replaces the path flags; a flag "
                             "overrides it, it overrides the built-in default; printed into the output, not read by "
                             "the keeper (AN-34)")


def add_parser(ss) -> None:
    c = ss.add_parser("layout", help="print a complete, commented layout file showing the effective defaults "
                                     "(AN-34); writes nothing")
    c.add_argument("--for", dest="for_command", choices=("host", "keeper"), default="keeper",
                   help="the command the layout is for (default: keeper)")
    c.add_argument("--mode", choices=(layout.SAME, layout.SEPARATE), default=None,
                   help="with --for keeper: the keeper mode (default: same-user, as `setup keeper`)")
    c.set_defaults(func=cmd_setup_layout)


def cmd_setup_layout(args) -> int:
    from .cli import _kfinish, _krefuse

    if args.for_command == "host":
        if args.mode is not None:
            return _krefuse(args, [Problem(layout.RULE, "--mode", "--mode applies to --for keeper; `setup host` has "
                                           "no mode", "drop --mode, or use --for keeper")])
        scope = layout.HOST
    else:
        scope = args.mode or layout.SAME
    try:
        text = layout.render(scope, env=os.environ)
    except Refused as exc:
        return _krefuse(args, exc.problems)
    return _kfinish(args, {"scope": scope, "toml": text}, text)
