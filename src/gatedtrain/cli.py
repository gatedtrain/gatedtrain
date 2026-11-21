"""`gatedtrain` command line (tests/README.md section 2).

Exit codes: 0 success (for `run`, the gate passed); 1 `run` completed but the
gate is FAIL, INCONCLUSIVE or MISSING; 2 usage error; 3 refused (spec or
registration error, hash mismatch, existing `--out`, unknown impact target).
Under the top-level `--json` an argparse usage error is the envelope with exit 2 (rule `USAGE`) and an uncaught
exception is the envelope with exit 1 (rule `INTERNAL`, the exception class only; no traceback), the code the text
mode gives for a traceback. Without `--json` both behave as argparse and Python always did.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from . import STUB_MARK, __version__
from .errors import Problem, Refused
from .output import Result, emit
from .paths import resolve

EXIT_OK, EXIT_NOT_PASS, EXIT_USAGE, EXIT_REFUSED = 0, 1, 2, 3


def _as_json(args) -> bool:
    return bool(getattr(args, "json", False))


def _finish(args, data: dict, human: str = "", *, ok: bool = True, exit_code: int = EXIT_OK,
            warnings: list[str] | None = None) -> int:
    """Emit a finished command: `human` on stdout in text mode, one JSON object with `--json`."""
    return emit(Result(args.command, ok, exit_code, data, [], list(warnings or [])), as_json=_as_json(args), human=human)


def _refuse(args, problems: list[Problem], warnings: list[str] | None = None) -> int:
    return emit(Result(args.command, False, EXIT_REFUSED, {}, list(problems), list(warnings or [])),
                as_json=_as_json(args), human="")


def _gate_result(args, run_dir, report, text, warnings: list[str] | None = None) -> int:
    ok = report["status"] == "PASS"
    return _finish(args, {"run_dir": str(run_dir), "report": report}, f"run_dir: {run_dir}\n{text}", ok=ok,
                   exit_code=EXIT_OK if ok else EXIT_NOT_PASS, warnings=warnings)


def cmd_validate(args) -> int:
    from .spec import load_spec

    try:
        load_spec(Path(args.spec))
    except Refused as exc:
        return _refuse(args, exc.problems)
    return _finish(args, {}, "ok\n")


def cmd_run(args) -> int:
    from .runner import run

    try:
        run_dir, report, text = run(Path(args.spec), Path(args.runs_root))
    except Refused as exc:
        return _refuse(args, exc.problems)
    return _gate_result(args, run_dir, report, text)


def cmd_warmup(args) -> int:
    from .stubs import trainer

    out = resolve(args.out)
    if out.exists():
        return _refuse(args, [Problem("WARMUP-OUT", "--out", f"{out} already exists; a warm-up never overwrites",
                                      "pass a new --out directory")])
    try:
        trainer.load_config(Path(args.config), kind="warmup")
        trainer.warmup(Path(args.config), out)
    except trainer.ConfigError as exc:
        return _refuse(args, [Problem("TRAINER-CONFIG", exc.field, exc.message, exc.fix)])
    return _finish(args, {"baseline": str(out)}, f"baseline: {out} ({STUB_MARK} warm-up checkpoint)\n")


def cmd_impact(args) -> int:
    from .impact import UnknownTarget, impact_counts, impact_orphaned, impact_query, impact_ranges, resolve_target

    try:
        target = resolve_target(args.shard)
        where = {"run_dir": Path(args.run_dir) if args.run_dir else None,
                 "runs_root": Path(args.runs_root) if args.runs_root else None}
        results = impact_query(target, **where)
        counts = impact_counts(target, **where)
        ranges = impact_ranges(target, **where)
        orphaned = impact_orphaned(target, **where)
    except (UnknownTarget, FileNotFoundError) as exc:
        return _refuse(args, [Problem("IMPACT-TARGET", "shard", str(exc),
                                      "pass sha256:<64 hex> or the path of an existing shard file, and a run directory")])
    query = {"shard": args.shard, "shard_sha256": target, "run_dir": args.run_dir, "runs_root": args.runs_root,
             "outcome": "consumed", "include_exposed": False}
    obj = {"query": query, "results": results, "counts": counts, "ranges": ranges}
    if orphaned:                                             # EXT-RES-5: only a resumed ext run's orphaned rows
        obj["orphaned"] = orphaned
    if _as_json(args):                                       # the AN-1 envelope; `data` is the impact object
        return _finish(args, obj)
    lines: list[str] = []
    if args.impact_json:                                     # the legacy `impact ... --json` object (R10A-9, R11A-19)
        lines.append(json.dumps(obj, indent=1, sort_keys=True))
    else:
        if not results:
            lines.append("no checkpoint consumed this shard")
        for r in results:
            lines.append(f"{r['answer']}: {r['checkpoint_hash']} run {r['run_id']} stage {r['stage_id']} "
                         f"last_step_included {r['last_step_included']} (first step {r['first_step'] if r['first_step'] is not None else 'null'}, "
                         f"precision {r['precision']}, derivation {r['derivation'] or 'null'})")
        lines.append(f"orphaned: {counts['orphaned']}, superseded: {counts['superseded']}")
        for kind in ("orphaned", "superseded"):              # R11A-19: ranges for the membership question
            if not ranges[kind]:
                lines.append(f"{kind}_ranges: none")
            for g in ranges[kind]:
                lines.append(f"{kind}_range: attempt {g['attempt_id']} shard {g['shard']} steps "
                             f"{g['first_step']}-{g['last_step']}")
        for attempt_id, runs in (orphaned.get("attempts") or {}).items():   # EXT-RES-5: the JSON section's runs
            lines.append(f"orphaned_steps: attempt {attempt_id} steps "
                         + ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in runs))
    return _finish(args, obj, "".join(line + "\n" for line in lines))


def _tierb_call(args, fn, warnings: list[str] | None = None) -> int:
    """Run `fn() -> (data, human)`; a refusal becomes the refusal result."""
    try:
        data, human = fn()
    except Refused as exc:
        return _refuse(args, exc.problems, warnings)
    return _finish(args, data, human, warnings=warnings)


def cmd_export(args) -> int:
    from .tierb import export_gsm8k

    def go():
        out = export_gsm8k(Path(args.out), source=Path(args.source) if args.source else None,
                           revision=args.revision, rows_per_shard=args.rows_per_shard)
        return {"export": str(out)}, f"export: {out}\n"
    return _tierb_call(args, go)


def cmd_verify_export(args) -> int:
    from .tierb import verify_export

    def go():
        verify_export(Path(args.dir))
        return {}, "ok\n"
    return _tierb_call(args, go)


def cmd_merge_eval(args) -> int:
    from .tierb import merge_eval

    def go():
        merged = merge_eval(Path(args.registration), Path(args.out), [Path(r) for r in args.runs])
        return {"merged": str(merged)}, f"merged: {merged}\n"
    return _tierb_call(args, go)


def cmd_nel_config(args) -> int:
    from .tierb import nel_configs

    def go():
        paths = list(nel_configs(Path(args.registration), model_url=args.model_url, model_id=args.model_id,
                                 eval_root=args.eval_root, out=Path(args.out)))
        return {"configs": [str(p) for p in paths]}, "".join(f"{p}\n" for p in paths)
    return _tierb_call(args, go)


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
        return _refuse(args, exc.problems)
    return _gate_result(args, run_dir, report, text)


def _ext_warnings(args) -> list[str]:
    """EXT-85: a line per active `GATEDTRAIN_EXT_*` host-layout override. In text mode they go to stderr at once,
    before anything else the command prints (as before AN-1), and an empty list is returned so `emit` does not print
    them again; with `--json` they are returned for the envelope's `warnings`. Either way they are kept on `args`, so
    `main` puts them in the envelope of an uncaught exception too."""
    from .ext.launch import override_warnings

    warnings = list(override_warnings())
    args.ext_warnings = warnings
    if _as_json(args):
        return warnings
    for line in warnings:
        print(line, file=sys.stderr)
    return []


def cmd_ext_collect(args) -> int:
    from .ext.collect import collect

    warnings = _ext_warnings(args)
    try:
        run_dir, report, text = collect(Path(args.spec), adapter=args.adapter, stage_out=Path(args.stage_out),
                                        launch_record=Path(args.launch_record), smoke_record=Path(args.smoke_record),
                                        eval_baseline=[Path(a) for a in args.eval_baseline],
                                        eval_candidate=[Path(a) for a in args.eval_candidate],
                                        runs_root=Path(args.runs_root),
                                        sources_root=Path(args.sources_root).absolute() if args.sources_root else None)
    except Refused as exc:
        return _refuse(args, exc.problems, warnings)
    return _gate_result(args, run_dir, report, text, warnings)


def cmd_ext_merge_eval(args) -> int:
    from .ext.merge import merge_eval

    def go():
        merged = merge_eval(Path(args.registration), Path(args.out), [Path(r) for r in args.runs])
        return {"merged": str(merged)}, f"merged: {merged}\n"
    return _tierb_call(args, go)


def cmd_ext_launch(args) -> int:
    from .ext.launch import ext_launch, recheck

    warnings = _ext_warnings(args)
    spec = Path(args.spec) if args.spec else None
    if args.recheck:                                     # EXT-69: re-validate a record right before the command
        try:
            checked = recheck(adapter=args.adapter, record_path=Path(args.recheck), spec=spec)
        except Refused as exc:
            return _refuse(args, exc.problems, warnings)
        return _finish(args, {"recheck": checked}, f"{checked}\n", warnings=warnings)
    missing = [f"--{n.replace('_', '-')}" for n in ("profile", "request", "gpu_log", "out_record")
               if getattr(args, n) is None]
    if missing:
        message = f"{', '.join(missing)} required (or --recheck RECORD)"
        if _as_json(args):
            problem = Problem("USAGE", ", ".join(missing), message, "pass the missing options, or --recheck RECORD")
            return emit(Result(args.command, False, EXIT_USAGE, {}, [problem], warnings), as_json=True, human="")
        print(f"gatedtrain ext-launch: error: {message}", file=sys.stderr)
        return EXIT_USAGE
    try:
        command = ext_launch(adapter=args.adapter, profile=args.profile, request=Path(args.request),
                             gpu_log=Path(args.gpu_log), out_record=Path(args.out_record),
                             smoke_record=Path(args.smoke_record) if args.smoke_record else None, spec=spec,
                             allow_network=args.allow_network)
    except Refused as exc:
        return _refuse(args, exc.problems, warnings)
    return _finish(args, {"command": command}, f"{command}\n", warnings=warnings)   # stdout is exactly the command (EXT-22, EXT-24)


def cmd_ext_from_nel(args) -> int:
    from .ext.from_nel import from_nel

    def go():
        arm = from_nel(registration=Path(args.registration), adapter=args.adapter, evaluator=args.evaluator,
                       items=Path(args.items), checkpoint=Path(args.checkpoint), out=Path(args.out),
                       runs=[Path(r) for r in args.runs])
        return {"arm": str(arm)}, f"arm: {arm}\n"
    return _tierb_call(args, go)


def cmd_ext_arm_fill(args) -> int:
    """EXT-110: fill an external arm's model_hash, item_set_hash and sampler_hash, or verify them."""
    from .ext.fill import arm_fill

    def go():
        out = arm_fill(adapter=args.adapter, arm=Path(args.arm), checkpoint=Path(args.checkpoint),
                       items=Path(args.items), base=Path(args.base) if args.base else None)
        lines = []
        for key, value in out["values"].items():
            how = "filled" if key in out["filled"] else "converted" if key in out["converted"] else "verified"
            lines.append(f"{how} {key}: {value}\n")
        return {"filled": dict(out["values"])}, "".join(lines)

    return _tierb_call(args, go)


# AN-11: `gatedtrain project ...`, thin wrappers over `ainative.machine` and `ainative.project`.

_GATE_STAGE = "gate"


def _project_command(args) -> str:
    return f"project {args.project_cmd}"


def _pfinish(args, data: dict, human: str = "", *, ok: bool = True, exit_code: int = EXIT_OK) -> int:
    return emit(Result(_project_command(args), ok, exit_code, data, [], []), as_json=_as_json(args), human=human)


def _prefuse(args, problems: list[Problem]) -> int:
    return emit(Result(_project_command(args), False, EXIT_REFUSED, {}, list(problems), []),
                as_json=_as_json(args), human="")


def _ref_data(refs: dict) -> dict:
    return {name: {"schema": r.schema, "sha256": r.sha256, "path": r.path} for name, r in sorted(refs.items())}


def _load_decision(path: str | None) -> dict:
    if path is None:
        return {}
    try:
        doc = json.loads(Path(path).read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise Refused([Problem("AN-11", "--decision", f"cannot read the decision file ({type(exc).__name__})",
                               "pass a JSON file holding one object")]) from None
    if not isinstance(doc, dict):
        raise Refused([Problem("AN-11", "--decision", "the decision file must hold one JSON object",
                               "pass a JSON file holding one object, e.g. {\"metric_confirmed\": true}")])
    return doc


def cmd_project_create(args) -> int:
    from .ainative.pipeline import load_pipeline
    from .ainative.project import create_project

    pipeline = Path(args.pipeline).absolute()
    try:
        load_pipeline(pipeline)                                  # syntax only; stage types are checked on `plan`/`run`
        project = create_project(Path(args.root).absolute(), goal=args.goal, data=Path(args.data).absolute(),
                                 pipeline=str(pipeline), budget={"gpu_hours": args.gpu_hours, "llm_usd": 0},
                                 pipeline_bytes=pipeline.read_bytes())     # the project keeps its own copy
    except Refused as exc:
        return _prefuse(args, exc.problems)
    return _pfinish(args, {"root": str(project.root), "pipeline": project.pipeline,
                           "pipeline_sha256": project.pipeline_sha256, "budget": project.budget},
                    f"project: {project.root}\n")


def cmd_project_plan(args) -> int:
    from .ainative.machine import plan_stage
    from .ainative.project import Project

    try:
        proposal = plan_stage(Project.load(Path(args.root)), args.stage)
    except Refused as exc:
        return _prefuse(args, exc.problems)
    data = {"stage": proposal.stage, "summary": proposal.summary, "choices": proposal.choices,
            "estimate": proposal.estimate, "needs_user": proposal.needs_user}
    human = f"{proposal.stage}: {proposal.summary}\n" + (f"needs_user: {proposal.needs_user}\n" if proposal.needs_user else "")
    return _pfinish(args, data, human)


def cmd_project_run(args) -> int:
    from .ainative.machine import run_stage
    from .ainative.project import Project

    try:
        project = Project.load(Path(args.root))
        result = run_stage(project, args.stage, _load_decision(args.decision))
        data = {"stage": result.stage, "key": result.key, "attempt": result.attempt, "reused": result.reused,
                "outputs": _ref_data(result.outputs)}
        status = None
        if result.stage == _GATE_STAGE and "gate_result" in result.outputs:
            status = project.read_artifact(result.outputs["gate_result"]).get("status")
            data["status"] = status
    except Refused as exc:
        return _prefuse(args, exc.problems)
    human = f"{result.stage}: {'reused' if result.reused else 'done'}\n" + (f"gate: {status}\n" if status else "")
    if result.stage == _GATE_STAGE:                              # `run` of the gate: EXIT_OK on PASS only
        ok = status == "PASS"
        return _pfinish(args, data, human, ok=ok, exit_code=EXIT_OK if ok else EXIT_NOT_PASS)
    return _pfinish(args, data, human)


def _recorded_gate(project):
    """(gate status recorded in the attempt log or None, the alpha ledger's entries): evidence apart from the
    pipeline state, so a stale or unreadable derived state never hides a recorded verdict."""
    from .ainative.machine import recorded_output
    from .ainative.trust import FixedAlphaLedger

    rec = recorded_output(project, _GATE_STAGE, "gate_result")
    return (rec.get("status") if isinstance(rec, dict) else None), FixedAlphaLedger(project).entries()


def cmd_project_status(args) -> int:
    from .ainative.machine import done_output, status
    from .ainative.project import Project

    try:
        project = Project.load(Path(args.root))
        stages = status(project)
        gate = done_output(project, _GATE_STAGE, "gate_result")
        recorded, ledger = _recorded_gate(project)
    except Refused as exc:
        return _prefuse(args, exc.problems)
    gate_status = gate.get("status") if isinstance(gate, dict) else None
    data = {"goal": project.goal, "stages": {n: {"status": s["status"],
                                                 "last_event": (s["last_attempt"] or {}).get("event"),
                                                 "last_refusal": s["last_refusal"]}
                                             for n, s in stages.items()},
            "gate": gate_status, "passed": gate_status == "PASS", "gate_recorded": recorded,
            "alpha_ledger": ledger}
    lines = []
    for n, s in data["stages"].items():
        ref = s["last_refusal"]
        note = f" (last attempt refused: {', '.join(ref['rules'] or [])} {ref['reason']})" if ref else ""
        lines.append(f"{n}: {s['status']}{note}\n")
    if gate_status:
        lines.append(f"gate: {gate_status}\n")
    elif recorded:
        lines.append(f"gate: recorded {recorded}, but the gate stage is {stages[_GATE_STAGE]['status']}\n")
    elif ledger:
        lines.append("gate: alpha spent, no gate result recorded\n")
    else:
        lines.append("gate: no gate result recorded and no alpha spent\n")
    return _pfinish(args, data, "".join(lines))


def cmd_project_report(args) -> int:
    from .ainative.machine import done_output, status
    from .ainative.project import Project

    try:
        project = Project.load(Path(args.root))
        report = done_output(project, "report", "report")
        if report is None:
            state = status(project)["report"]["status"]
            recorded, ledger = _recorded_gate(project)
    except Refused as exc:
        return _prefuse(args, exc.problems)
    if report is None:
        evidence = (f"a gate result is recorded ({recorded}) and the alpha ledger has {len(ledger)} entries"
                    if recorded else f"no gate result is recorded; the alpha ledger has {len(ledger)} entries")
        return _prefuse(args, [Problem("AN-11", "report", f"there is no current report: the report stage is {state}; "
                                       f"{evidence}", "run the stages in order up to `report`; `project status` "
                                       "shows where it stopped")])
    ok = report.get("status") == "PASS"
    md = project.root / "stages" / "report" / "report.md"
    human = md.read_text(encoding="utf-8") if md.is_file() else f"status: {report.get('status')}\n"
    return _pfinish(args, {"report": report, "report_md": str(md)}, human, ok=ok,
                    exit_code=EXIT_OK if ok else EXIT_NOT_PASS)


def _gpu_hours(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(f"{text!r} must be a finite number, at least 0")
    return value


def _add_project(sub) -> None:
    p = sub.add_parser("project", help="AI-native project (step 1 preview): create, plan, run, status, report")
    ps = p.add_subparsers(dest="project_cmd", required=True)

    c = ps.add_parser("create", help="create a project directory for one goal")
    c.add_argument("root", help="a new or empty directory")
    c.add_argument("--goal", required=True, help="what the project is for, in words")
    c.add_argument("--data", required=True, help="the labelled data (a file or directory); its hash is recorded")
    c.add_argument("--pipeline", required=True, metavar="FILE",
                   help="the pipeline spec: the path of a pipeline YAML (the loader resolves a path, not a registry "
                        "name). The project keeps a copy and its sha256; a changed copy is refused, and another "
                        "pipeline needs a new project with new sealed data")
    c.add_argument("--gpu-hours", type=_gpu_hours, default=0.0, metavar="H",
                   help="the GPU-hour budget; step 1 only records it in project.yaml and does not enforce it")
    c.set_defaults(func=cmd_project_create)

    c = ps.add_parser("plan", help="show what a stage would do, without running it")
    c.add_argument("root")
    c.add_argument("stage", help="a stage name of the pipeline")
    c.set_defaults(func=cmd_project_plan)

    c = ps.add_parser("run", help="run one stage; `gate` exits 0 on PASS and 1 otherwise")
    c.add_argument("root")
    c.add_argument("stage", help="a stage name of the pipeline")
    c.add_argument("--decision", metavar="FILE", help="a JSON object with the decision for this stage")
    c.set_defaults(func=cmd_project_run)

    c = ps.add_parser("status", help="the status of every stage and the gate result, if any")
    c.add_argument("root")
    c.set_defaults(func=cmd_project_status)

    c = ps.add_parser("report", help="print the report; refused when the report stage is not done")
    c.add_argument("root")
    c.set_defaults(func=cmd_project_report)


class _UsageError(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    """Under the top-level `--json`, a usage error is raised (and wrapped by `main`) instead of printed and exited."""

    json_errors = False

    def error(self, message: str):
        if _Parser.json_errors:
            raise _UsageError(message)
        super().error(message)


def _json_requested(argv: list[str]) -> tuple[bool, str]:
    """(top-level `--json` given, the subcommand token): only options before the first positional count."""
    as_json = False
    for tok in argv:
        if tok == "--json":
            as_json = True
        elif not tok.startswith("-"):
            return as_json, tok
    return as_json, "gatedtrain"


def build_parser() -> argparse.ArgumentParser:
    ap = _Parser(prog="gatedtrain", description=f"walking-skeleton prototype {__version__} "
                                 f"(prototype, unpromotable; Tier A stand-ins are {STUB_MARK})")
    ap.add_argument("--version", action="version", version=f"gatedtrain {__version__}")
    ap.add_argument("--json", action="store_true",
                    help="print one JSON object (schema gatedtrain.cli_result) instead of text; goes before the "
                         "subcommand (AN-1)")
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
    p.add_argument("--json", action="store_true", dest="impact_json",
                   help="the legacy impact object (after the subcommand); with the top-level --json, the AN-1 object")
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

    _add_project(sub)
    return ap


def main(argv: list[str] | None = None) -> int:
    from . import runner

    argv = list(sys.argv[1:] if argv is None else argv)
    as_json, first = _json_requested(argv)
    _Parser.json_errors = as_json
    try:
        args = build_parser().parse_args(argv)
    except _UsageError as exc:                           # AN-1: a usage error is the envelope too (exit 2)
        problem = Problem("USAGE", "argv", str(exc), "see `gatedtrain --help`")
        return emit(Result(first, False, EXIT_USAGE, {}, [problem], []), as_json=True, human="")
    finally:
        _Parser.json_errors = False
    runner.QUIET = _as_json(args)                        # AN-1: no progress lines on stderr under --json
    try:
        return args.func(args)
    except Exception as exc:
        if not _as_json(args):
            raise
        name = _project_command(args) if args.command == "project" else args.command
        problem = Problem("INTERNAL", "", f"unexpected {type(exc).__name__}; this is a bug", "report it")
        return emit(Result(name, False, EXIT_NOT_PASS, {}, [problem], list(getattr(args, "ext_warnings", []))),
                    as_json=True, human="")
    finally:
        runner.QUIET = False


if __name__ == "__main__":                                # EXT-89: `python -m gatedtrain.cli` runs the CLI
    sys.exit(main())
