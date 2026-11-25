"""AN-33: the `gatedtrain setup check` command (kept out of cli.py; the checks are in ainative/setup_check.py)."""

from __future__ import annotations

import argparse

from .ainative import setup_check as sc


def make_probe() -> sc.HostProbe:
    return sc.SystemProbe()


def human_table(checks: list[dict]) -> str:
    width = max(len(c["id"]) for c in checks)
    lines = [f"{'check'.ljust(width)}  status          required  owner          summary"]
    for c in checks:
        lines.append(f"{c['id'].ljust(width)}  {c['status'].ljust(14)}  {str(c['required']).ljust(8)}  "
                     f"{c['owner'].ljust(13)}  {c['summary']}")
    todo = [c for c in checks if c["status"] in ("missing", "unknown")]
    if todo:
        lines += ["", "to fix:"] + [f"  {c['id']} ({c['owner']}): {c['fix']}" for c in todo]
    lines += ["", sc.CLOSING_LINE]
    return "\n".join(lines) + "\n"


def cmd_setup_check(args) -> int:
    from .cli import EXIT_NOT_PASS, EXIT_OK, _kfinish

    probe = make_probe()
    gpu = sc.gpu_default(probe) if args.gpu is None else args.gpu
    checks = sc.run_checks(probe, gpu=gpu)
    failing = sc.failing_ids(checks)
    admin = [c["id"] for c in checks if c["id"] in failing and c["owner"] == "administrator"]
    data = {"checks": checks, "gpu": gpu, "failing": failing, "needs_administrator": admin}
    return _kfinish(args, data, human_table(checks), ok=not failing,
                    exit_code=EXIT_NOT_PASS if failing else EXIT_OK)


def add_parser(ss) -> None:
    c = ss.add_parser("check", help="name the outside prerequisites (rootless Docker, GPU CDI, ...); changes nothing")
    c.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=None,
                   help="check the NVIDIA driver and CDI spec (default: when /proc/driver/nvidia exists)")
    c.set_defaults(func=cmd_setup_check)
