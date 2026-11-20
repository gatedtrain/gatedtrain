"""`gatedtrain` command line (tests/README.md section 2).

Exit codes: 0 success (for `run`, the gate passed); 1 `run` completed but the
gate is FAIL, INCONCLUSIVE or MISSING; 2 usage error; 3 refused (spec or
registration error, hash mismatch, existing `--out`, unknown impact target).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import STUB_MARK, __version__
from .errors import Problem, Refused
from .paths import resolve

EXIT_OK, EXIT_NOT_PASS, EXIT_USAGE, EXIT_REFUSED = 0, 1, 2, 3


def _refuse(problems: list[Problem]) -> int:
    for p in problems:
        print(p.line(), file=sys.stderr)
    return EXIT_REFUSED


def cmd_validate(args) -> int:
    from .spec import load_spec

    try:
        load_spec(Path(args.spec))
    except Refused as exc:
        return _refuse(exc.problems)
    print("ok")
    return EXIT_OK


def cmd_run(args) -> int:
    from .runner import run

    try:
        run_dir, report, text = run(Path(args.spec), Path(args.runs_root))
    except Refused as exc:
        return _refuse(exc.problems)
    print(f"run_dir: {run_dir}")
    print(text, end="")
    return EXIT_OK if report["status"] == "PASS" else EXIT_NOT_PASS


def cmd_warmup(args) -> int:
    from .stubs import trainer

    out = resolve(args.out)
    if out.exists():
        return _refuse([Problem("WARMUP-OUT", "--out", f"{out} already exists; a warm-up never overwrites",
                                "pass a new --out directory")])
    try:
        trainer.load_config(Path(args.config), kind="warmup")
        trainer.warmup(Path(args.config), out)
    except trainer.ConfigError as exc:
        return _refuse([Problem("TRAINER-CONFIG", exc.field, exc.message, exc.fix)])
    print(f"baseline: {out} ({STUB_MARK} warm-up checkpoint)")
    return EXIT_OK


def cmd_impact(args) -> int:
    from .impact import UnknownTarget, impact_counts, impact_query, impact_ranges, resolve_target

    try:
        target = resolve_target(args.shard)
        where = {"run_dir": Path(args.run_dir) if args.run_dir else None,
                 "runs_root": Path(args.runs_root) if args.runs_root else None}
        results = impact_query(target, **where)
        counts = impact_counts(target, **where)
        ranges = impact_ranges(target, **where)
    except (UnknownTarget, FileNotFoundError) as exc:
        return _refuse([Problem("IMPACT-TARGET", "shard", str(exc),
                                "pass sha256:<64 hex> or the path of an existing shard file, and a run directory")])
    if args.json:
        query = {"shard": args.shard, "shard_sha256": target, "run_dir": args.run_dir, "runs_root": args.runs_root,
                 "outcome": "consumed", "include_exposed": False}
        print(json.dumps({"query": query, "results": results, "counts": counts, "ranges": ranges}, indent=1,
                         sort_keys=True))
    else:
        if not results:
            print("no checkpoint consumed this shard")
        for r in results:
            print(f"{r['answer']}: {r['checkpoint_hash']} run {r['run_id']} stage {r['stage_id']} "
                  f"last_step_included {r['last_step_included']} (first step {r['first_step'] if r['first_step'] is not None else 'null'}, "
                  f"precision {r['precision']}, derivation {r['derivation'] or 'null'})")
        print(f"orphaned: {counts['orphaned']}, superseded: {counts['superseded']}")
        for kind in ("orphaned", "superseded"):              # R11A-19: ranges for the membership question
            if not ranges[kind]:
                print(f"{kind}_ranges: none")
            for g in ranges[kind]:
                print(f"{kind}_range: attempt {g['attempt_id']} shard {g['shard']} steps "
                      f"{g['first_step']}-{g['last_step']}")
    return EXIT_OK


def _tierb_call(fn) -> int:
    try:
        fn()
    except Refused as exc:
        return _refuse(exc.problems)
    return EXIT_OK


def cmd_export(args) -> int:
    from .tierb import export_gsm8k

    def go():
        out = export_gsm8k(Path(args.out), source=Path(args.source) if args.source else None,
                           revision=args.revision, rows_per_shard=args.rows_per_shard)
        print(f"export: {out}")
    return _tierb_call(go)


def cmd_verify_export(args) -> int:
    from .tierb import verify_export

    return _tierb_call(lambda: (verify_export(Path(args.dir)), print("ok")))


def cmd_merge_eval(args) -> int:
    from .tierb import merge_eval

    return _tierb_call(lambda: print(f"merged: {merge_eval(Path(args.registration), Path(args.out), [Path(r) for r in args.runs])}"))


def cmd_nel_config(args) -> int:
    from .tierb import nel_configs

    def go():
        for p in nel_configs(Path(args.registration), model_url=args.model_url, model_id=args.model_id,
                             eval_root=args.eval_root, out=Path(args.out)):
            print(p)
    return _tierb_call(go)


def cmd_tierb_collect(args) -> int:
    from .tierb import collect

    try:
        run_dir, report, text = collect(Path(args.spec), stage_out=Path(args.stage_out),
                                        eval_baseline=Path(args.eval_baseline), eval_candidate=Path(args.eval_candidate),
                                        image_id=args.image_id, nemo_rl_commit=args.nemo_rl_commit,
                                        runs_root=Path(args.runs_root),
                                        nemo_rl_modified={"yes": True, "no": False, None: "unknown"}[args.nemo_rl_modified],
                                        image_build_notes=args.image_build_notes)
    except Refused as exc:
        return _refuse(exc.problems)
    print(f"run_dir: {run_dir}")
    print(text, end="")
    return EXIT_OK if report["status"] == "PASS" else EXIT_NOT_PASS


def _ext_warnings() -> None:
    """EXT-85: a stderr line per active `GATEDTRAIN_EXT_*` host-layout override."""
    from .ext.launch import override_warnings

    for line in override_warnings():
        print(line, file=sys.stderr)


def cmd_ext_collect(args) -> int:
    from .ext.collect import collect

    _ext_warnings()
    try:
        run_dir, report, text = collect(Path(args.spec), adapter=args.adapter, stage_out=Path(args.stage_out),
                                        launch_record=Path(args.launch_record), smoke_record=Path(args.smoke_record),
                                        eval_baseline=[Path(a) for a in args.eval_baseline],
                                        eval_candidate=[Path(a) for a in args.eval_candidate],
                                        runs_root=Path(args.runs_root),
                                        sources_root=Path(args.sources_root).absolute() if args.sources_root else None)
    except Refused as exc:
        return _refuse(exc.problems)
    print(f"run_dir: {run_dir}")
    print(text, end="")
    return EXIT_OK if report["status"] == "PASS" else EXIT_NOT_PASS


def cmd_ext_merge_eval(args) -> int:
    from .ext.merge import merge_eval

    return _tierb_call(lambda: print(f"merged: {merge_eval(Path(args.registration), Path(args.out), [Path(r) for r in args.runs])}"))


def cmd_ext_launch(args) -> int:
    from .ext.launch import ext_launch, recheck

    _ext_warnings()
    spec = Path(args.spec) if args.spec else None
    if args.recheck:                                     # EXT-69: re-validate a record right before the command
        try:
            print(recheck(adapter=args.adapter, record_path=Path(args.recheck), spec=spec))
        except Refused as exc:
            return _refuse(exc.problems)
        return EXIT_OK
    missing = [f"--{n.replace('_', '-')}" for n in ("profile", "request", "gpu_log", "out_record")
               if getattr(args, n) is None]
    if missing:
        print(f"gatedtrain ext-launch: error: {', '.join(missing)} required (or --recheck RECORD)", file=sys.stderr)
        return EXIT_USAGE
    try:
        command = ext_launch(adapter=args.adapter, profile=args.profile, request=Path(args.request),
                             gpu_log=Path(args.gpu_log), out_record=Path(args.out_record),
                             smoke_record=Path(args.smoke_record) if args.smoke_record else None, spec=spec,
                             allow_network=args.allow_network)
    except Refused as exc:
        return _refuse(exc.problems)
    print(command)                                       # stdout is exactly the command (EXT-22, EXT-24)
    return EXIT_OK


def cmd_ext_from_nel(args) -> int:
    from .ext.from_nel import from_nel

    return _tierb_call(lambda: print(f"arm: {from_nel(registration=Path(args.registration), adapter=args.adapter, evaluator=args.evaluator, items=Path(args.items), checkpoint=Path(args.checkpoint), out=Path(args.out), runs=[Path(r) for r in args.runs])}"))


def cmd_ext_arm_fill(args) -> int:
    """EXT-110: fill an external arm's model_hash, item_set_hash and sampler_hash, or verify them."""
    from .ext.fill import arm_fill

    def go():
        out = arm_fill(adapter=args.adapter, arm=Path(args.arm), checkpoint=Path(args.checkpoint),
                       items=Path(args.items), base=Path(args.base) if args.base else None)
        for key, value in out["values"].items():
            how = "filled" if key in out["filled"] else "converted" if key in out["converted"] else "verified"
            print(f"{how} {key}: {value}")

    return _tierb_call(go)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="gatedtrain", description=f"walking-skeleton prototype {__version__} "
                                 f"(prototype, unpromotable; Tier A stand-ins are {STUB_MARK})")
    ap.add_argument("--version", action="version", version=f"gatedtrain {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="check a spec and its registration")
    p.add_argument("spec")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("run", help="run the whole path and print the gate report")
    p.add_argument("spec")
    p.add_argument("--runs-root", default="runs")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("warmup", help="train the baseline checkpoint (skeleton-stub)")
    p.add_argument("config")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_warmup)

    p = sub.add_parser("impact", help="checkpoints that possibly consumed a shard (05 section 11.3)")
    p.add_argument("shard", help="sha256:<hex> or a shard file path")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir")
    g.add_argument("--runs-root")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_impact)

    # Tier B helpers (TIER-B-SPEC H-1 to H-4, H-6). They never launch a GPU job.
    p = sub.add_parser("export-gsm8k", help="Tier B: export GSM8K to shard files under a manifest (H-1)")
    p.add_argument("--out", required=True)
    p.add_argument("--source", help="local copy with train.jsonl and test.jsonl (openai/gsm8k main rows)")
    p.add_argument("--revision", help="Hugging Face dataset commit hash (required with --source; a download needs a full 40-hex commit "
                   "and defaults to tierb.DATASET_REVISION; G-17)")
    p.add_argument("--rows-per-shard", type=int, default=500)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("verify-export", help="Tier B: re-check an export's hashes (H-2)")
    p.add_argument("dir")
    p.set_defaults(func=cmd_verify_export)

    p = sub.add_parser("merge-eval", help="Tier B: merge K single-seed NeMo Evaluator runs (H-3)")
    p.add_argument("--registration", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("runs", nargs="+")
    p.set_defaults(func=cmd_merge_eval)

    p = sub.add_parser("nel-config", help="Tier B: write K NeMo Evaluator configs from the registration (H-4)")
    p.add_argument("--registration", required=True)
    p.add_argument("--model-url", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--eval-root", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_nel_config)

    p = sub.add_parser("tierb-collect", help="Tier B: recorded outputs to a gate report and ledger (H-6)")
    p.add_argument("spec")
    p.add_argument("--stage-out", required=True)
    p.add_argument("--eval-baseline", required=True)
    p.add_argument("--eval-candidate", required=True)
    p.add_argument("--image-id", required=True)
    p.add_argument("--nemo-rl-commit", required=True)
    p.add_argument("--runs-root", default="runs")
    # R10A-11: whether the NeMo RL image carries a local modification; absent is recorded as "unknown".
    p.add_argument("--nemo-rl-modified", choices=("yes", "no"),
                   help="the NeMo RL image carries a local modification (yes) or none (no); absent: unknown")
    p.add_argument("--image-build-notes", help="free text on how the image was built (recorded in the lockfile)")
    p.set_defaults(func=cmd_tierb_collect)

    # External-run adapters (MULTIMODAL-SPEC MM-5, MM-9, MM-31; tests/README.md section 11.18). They never launch a
    # process: ext-launch prints the command for the operator.
    p = sub.add_parser("ext-collect", help="external run: recorded outputs to a gate report and ledger (MM-9)")
    p.add_argument("spec")
    p.add_argument("--adapter", required=True)
    p.add_argument("--stage-out", required=True)
    p.add_argument("--launch-record", required=True)
    p.add_argument("--smoke-record", required=True)
    # Design revision 18 (R18-1): once per clause, in the order of the gate's evaluators.
    p.add_argument("--eval-baseline", required=True, action="append",
                   help="the baseline arm; with several clauses, once per clause in the gate's evaluator order")
    p.add_argument("--eval-candidate", required=True, action="append",
                   help="the candidate arm; with several clauses, once per clause in the gate's evaluator order")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--sources-root", help="the directory a merged arm's sources (merge.json) must lie under "
                   "(MM-5); default: the host's output root")
    p.set_defaults(func=cmd_ext_collect)

    p = sub.add_parser("ext-merge-eval", help="external run: merge per-item arm runs into one arm (MM-5)")
    p.add_argument("--registration", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("runs", nargs="+")
    p.set_defaults(func=cmd_ext_merge_eval)

    p = sub.add_parser("ext-launch", help="external run: write the launch record and print the command (MM-31)")
    p.add_argument("--adapter", required=True)
    p.add_argument("--profile", choices=("smoke", "full"), help="required unless --recheck")
    p.add_argument("--request", help="required unless --recheck")
    p.add_argument("--gpu-log", help="<HOST_WORK>/gpu-log.md (its last running total); required unless --recheck")
    p.add_argument("--smoke-record", help="the PASS smoke.json; required with --profile full")
    p.add_argument("--out-record", help="required unless --recheck")
    p.add_argument("--spec", help="the spec: its base and export directories must be read-only mounts (EXT-65); "
                                  "required when the adapter binds them (EXT-84)")
    p.add_argument("--allow-network", action="store_true",
                   help="allow the request's named network other than none and host, recorded with its reason "
                        "(EXT-81)")
    p.add_argument("--recheck", metavar="RECORD",
                   help="re-validate a launch record against the filesystem now; run it right before the printed "
                        "command (EXT-69)")
    p.set_defaults(func=cmd_ext_launch)

    p = sub.add_parser("ext-from-nel", help="external run: K NeMo Evaluator runs to one per-item arm (EXT-35)")
    p.add_argument("--registration", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--evaluator", required=True)
    p.add_argument("--items", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("runs", nargs="+")
    p.set_defaults(func=cmd_ext_from_nel)

    p = sub.add_parser("ext-arm-fill", help="external run: fill and verify an arm's gatedtrain hashes (EXT-110)")
    p.add_argument("--adapter", required=True)
    p.add_argument("--checkpoint", required=True, help="the evaluated cut directory")
    p.add_argument("--items", required=True, help="the item set directory the arm evaluated")
    p.add_argument("--base", help="the plain base directory a LoRA cut was trained on (EXT-126, MM-18)")
    p.add_argument("arm", help="the arm directory (items.jsonl, eval_meta.json); eval_meta.json is rewritten")
    p.set_defaults(func=cmd_ext_arm_fill)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":                                # EXT-89: `python -m gatedtrain.cli` runs the CLI
    sys.exit(main())
