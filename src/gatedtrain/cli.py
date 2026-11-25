"""`gatedtrain` command line (tests/README.md section 2).

Exit codes: 0 success (for `run`, the gate passed); 1 `run` completed but the
gate is FAIL, INCONCLUSIVE or MISSING, and `verify` checked consistency only (no
keeper and no fingerprint as anchor); 2 usage error; 3 refused (spec or
registration error, hash mismatch, existing `--out`, unknown impact target, a
bad signature or a record that differs from the anchor in `verify`).
Under the top-level `--json` an argparse usage error is the envelope with exit 2 (rule `USAGE`) and an uncaught
exception is the envelope with exit 1 (rule `INTERNAL`, the exception class only; no traceback), the code the text
mode gives for a traceback. Without `--json` both behave as argparse and Python always did.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

from . import STUB_MARK, __version__
from .errors import Problem, Refused, printable
from .output import Result, emit, notice_scope
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
    if args.project_cmd == "job":
        return f"project job {args.job_cmd}"
    return f"project {args.project_cmd}"


def _pfinish(args, data: dict, human: str = "", *, ok: bool = True, exit_code: int = EXIT_OK,
             warnings: list[str] | None = None) -> int:
    warnings = list(warnings or [])
    if not _as_json(args):                  # the human text already carries the mode line: not a second time on stderr
        from .ainative.keeper import texts

        warnings = [w for w in warnings if w not in texts.MODE_LINE.values()]
    return emit(Result(_project_command(args), ok, exit_code, data, [], warnings), as_json=_as_json(args), human=human)


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


def _keeper_modes():
    from .ainative.keeper import modes

    return modes


def _keeper_hello(socket: Path) -> dict:
    """The keeper's `hello`, its key checked (AN-14). It waits with the long-op deadline: on a keeper busy with another
    client's evaluation the wait is queueing; a keeper that is not running fails at the connect (AN-12)."""
    from .ainative.keeper import protocol
    from .ainative.keeper_trust import _hello_key

    from .ainative.keeper import modes

    asking = protocol.KeeperClient(socket, timeout=protocol.LONG_OP_TIMEOUT_S)
    hello = asking.call("hello", None, {})
    _hello_key(hello)
    # AN-29: the mode to state is the keeper's claim weakened by the kernel's answer; a contradiction is refused
    mode, note = modes.effective_client_mode(modes.claimed_mode(hello), asking.last_attested_uid, os.geteuid())
    return {**hello, "effective_mode": mode, "mode_note": note}


def _remove_unbound_project(root: Path, existed: bool) -> None:
    """Undo `create_project` after a binding that failed: everything in `root` was written by it (it refuses a root
    that holds anything), so the contents go, and the directory too when it did not exist before."""
    for child in list(root.iterdir()):
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    if not existed:
        root.rmdir()


def cmd_project_create(args) -> int:
    from .ainative.keeper import modes, texts
    from .ainative.keeper_trust import bind_project
    from .ainative.pipeline import load_pipeline
    from .ainative.project import create_project

    pipeline = Path(args.pipeline).absolute()
    socket = Path(args.keeper).absolute() if args.keeper else None
    require = args.require_keeper_mode.replace("-", "_")
    hello = None
    try:
        load_pipeline(pipeline)                                  # syntax only; stage types are checked on `plan`/`run`
        if socket is not None:
            hello = _keeper_hello(socket)                        # an unreachable keeper: nothing is created
            if require != modes.REQUIRE_ANY and "keeper_mode" not in hello:   # a keeper before AN-27 cannot hold a policy
                raise Refused([Problem(modes.RULE_POLICY, "--require-keeper-mode", "the keeper does not state a mode "
                                       "and cannot enforce a requirement: keeper predates AN-28; upgrade it",
                                       "upgrade the keeper (`gatedtrain setup keeper`, then restart it), or leave out "
                                       "--require-keeper-mode")])
            if not modes.satisfies(hello["effective_mode"], require):         # AN-28: refuse before anything is made
                raise Refused([Problem(modes.RULE_POLICY, "--require-keeper-mode", "the bound keeper is in "
                                       f"{hello['effective_mode'].replace('_', '-')} mode, which does not meet the "
                                       f"requirement {args.require_keeper_mode}", "bind a keeper under a separate OS "
                                       "user (`gatedtrain setup keeper --mode separate-user`), or leave out "
                                       "--require-keeper-mode")])
        elif require != modes.REQUIRE_ANY:
            raise Refused([Problem(modes.RULE_POLICY, "--require-keeper-mode", "a keeper requirement needs a keeper",
                                   "pass --keeper SOCKET, or leave out --require-keeper-mode")])
        root = Path(args.root).absolute()
        existed = root.exists()                                  # create_project accepts only an empty existing root
        project = create_project(root, goal=args.goal, data=Path(args.data).absolute(),
                                 pipeline=str(pipeline), budget={"gpu_hours": args.gpu_hours, "llm_usd": 0},
                                 pipeline_bytes=pipeline.read_bytes())     # the project keeps its own copy
        try:
            binding = bind_project(project, socket, require) if socket is not None else None
        except BaseException:
            _remove_unbound_project(root, existed)               # a refused binding leaves no half-made project
            raise
    except Refused as exc:
        return _prefuse(args, exc.problems)
    data = {"root": str(project.root), "pipeline": project.pipeline, "pipeline_sha256": project.pipeline_sha256,
            "budget": project.budget}
    human = f"project: {project.root}\n"
    warnings: list[str] = []
    if binding is not None:
        mode = hello["effective_mode"]
        warnings = _same_user_warnings(mode)
        data["keeper"] = {"fingerprint": binding.fingerprint, "project_id": binding.project_id, "keeper_mode": mode,
                          "mode_facts": hello.get("mode_facts"), "require_keeper_mode": require}
        human += (f"keeper: {binding.socket} (key {binding.fingerprint})\n{texts.MODE_LINE[mode]}\n"
                  + (f"requirement: {args.require_keeper_mode}\n" if require != modes.REQUIRE_ANY else ""))
    return _pfinish(args, data, human, warnings=warnings)


def cmd_project_plan(args) -> int:
    from .ainative.machine import plan_stage
    from .ainative.project import Project

    try:
        project = Project.load(Path(args.root))
        proposal = plan_stage(project, args.stage)
        mode, mode_warnings = _mode_surface(project)
    except Refused as exc:
        return _prefuse(args, exc.problems)
    data = {"stage": proposal.stage, "summary": proposal.summary, "choices": proposal.choices,
            "estimate": proposal.estimate, "needs_user": proposal.needs_user}
    human = f"{proposal.stage}: {proposal.summary}\n" + (f"needs_user: {proposal.needs_user}\n" if proposal.needs_user else "")
    human = _with_mode(data, human, mode)
    return _pfinish(args, data, human, warnings=mode_warnings)


def _effective_mode(project, stated: str | None, *, tolerate_unreachable: bool = False) -> tuple[str | None, str | None]:
    """AN-27, AN-29: `(mode, warning)` to print for a keeper project: the weakest of the keeper's stated mode (from
    its summary or report) and this client's own check of the keeper (`KeeperTrust.keeper_mode`: its claim weakened
    by the kernel's answer), never the keeper's statement alone. A keeper that contradicts the kernel is refused
    (AN-29); with `tolerate_unreachable` an unreachable keeper leaves the stated mode and a warning. Without a stated
    mode (a stage that is not the intake, `status`, `plan`) a keeper-bound project is judged by the check alone;
    a project without a keeper has no mode (None)."""
    from .ainative.keeper import modes
    from .ainative.keeper_trust import KeeperTrust, load_binding

    if stated is None and load_binding(project) is None:        # no keeper: there is no mode to state
        return None, None
    known = [stated] if stated in modes.MODES else []
    try:
        known.append(KeeperTrust(project).keeper_mode)
    except Refused as exc:
        if not tolerate_unreachable or any(p.rule == modes.RULE_ATTESTATION for p in exc.problems):
            raise
        return (modes.weakest(*known) if known else modes.SAME_USER,
                "the keeper could not be asked for its mode now; the mode shown is the one the records state, read "
                "from the project files and not verified against the keeper: "
                + exc.problems[0].line())
    return modes.weakest(*known) if known else modes.SAME_USER, None


def _same_user_warnings(mode: str | None) -> list[str]:
    """AN-27: one warning with the mode line for a same-user result, so a consumer of the JSON envelope that reads
    only `ok` and `warnings` still sees that the data are not sealed from the user's account."""
    from .ainative.keeper import modes, texts

    return [texts.MODE_LINE[mode]] if mode == modes.SAME_USER else []


def _mode_surface(project, stated: str | None = None) -> tuple[str | None, list[str]]:
    """`(mode, warnings)` for a command that prints a result for `project`: `_effective_mode` (an unreachable keeper
    leaves a warning and the weaker mode), plus the same-user warning. `(None, [])` for a project without a keeper."""
    mode, warning = _effective_mode(project, stated, tolerate_unreachable=True)
    return mode, ([warning] if warning else []) + _same_user_warnings(mode)


def _with_mode(data: dict, human: str, mode: str | None, *, first: bool = False) -> str:
    """Put the mode in the JSON data and its line (AN-27) in the human text, after it or `first`."""
    if mode is None:
        return human
    from .ainative.keeper import texts

    data["keeper_mode"] = mode
    return f"{texts.MODE_LINE[mode]}\n{human}" if first else f"{human}{texts.MODE_LINE[mode]}\n"


def _gate_line(status: str, mode: str | None) -> str:
    """The gate result line; in same-user mode it carries the headline qualifier (AN-27), as the report's does."""
    from .ainative.keeper import modes, texts

    return f"gate: {status}" + (texts.HEADLINE_SUFFIX_SAME_USER if mode == modes.SAME_USER else "") + "\n"


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
        summary = project.read_artifact(result.outputs["sealed_summary"]) if "sealed_summary" in result.outputs \
            else None
        # AN-27: every stage of a keeper project states the mode: the intake's own statement, weakened by this client's check
        mode, mode_warnings = _mode_surface(project, summary.get("keeper_mode") if isinstance(summary, dict) else None)
    except Refused as exc:
        return _prefuse(args, exc.problems)
    human = f"{result.stage}: {'reused' if result.reused else 'done'}\n" + (_gate_line(status, mode) if status else "")
    human = _with_mode(data, human, mode)
    if result.stage == _GATE_STAGE:                              # `run` of the gate: EXIT_OK on PASS only
        ok = status == "PASS"
        return _pfinish(args, data, human, ok=ok, exit_code=EXIT_OK if ok else EXIT_NOT_PASS, warnings=mode_warnings)
    return _pfinish(args, data, human, warnings=mode_warnings)


def _recorded_gate(project) -> tuple[str | None, list | None, list[str]]:
    """(gate status recorded in the attempt log or None, the alpha ledger's entries of this project, warnings):
    evidence apart from the pipeline state, so a stale or unreadable derived state never hides a recorded verdict.
    The ledger is read through `machine.backends` (the keeper's for a bound project); when it cannot be read (an
    unreachable keeper, AN-12) the entries are None and a warning says why."""
    from .ainative.machine import backends, recorded_output

    rec = recorded_output(project, _GATE_STAGE, "gate_result")
    recorded = rec.get("status") if isinstance(rec, dict) else None
    try:
        _, ledger = backends(project)
        project_entries = getattr(ledger, "project_entries", None)     # the keeper's ledger spans the sealed set
        entries = project_entries() if project_entries is not None else ledger.entries()
    except Refused as exc:
        return recorded, None, [f"the alpha ledger cannot be read: {p.line()}" for p in exc.problems]
    return recorded, entries, []


def cmd_project_status(args) -> int:
    from .ainative.machine import done_output, status
    from .ainative.project import Project

    try:
        project = Project.load(Path(args.root))
        stages = status(project)
        gate = done_output(project, _GATE_STAGE, "gate_result")
        recorded, ledger, warnings = _recorded_gate(project)
        mode, mode_warnings = _mode_surface(project)
        warnings = [*warnings, *mode_warnings]
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
        lines.append(_gate_line(gate_status, mode))
    elif recorded:
        lines.append(_gate_line(f"recorded {recorded}, but the gate stage is {stages[_GATE_STAGE]['status']}", mode))
    elif ledger:
        lines.append("gate: alpha spent, no gate result recorded\n")
    elif ledger is None:
        lines.append("gate: no gate result recorded; the alpha ledger cannot be read\n")
    else:
        lines.append("gate: no gate result recorded and no alpha spent\n")
    return _pfinish(args, data, _with_mode(data, "".join(lines), mode, first=True), warnings=warnings)


def cmd_project_report(args) -> int:
    from .ainative.machine import done_output, status
    from .ainative.project import Project

    try:
        project = Project.load(Path(args.root))
        warnings = _sync_receipts(project)
        report = done_output(project, "report", "report")
        if report is None:
            state = status(project)["report"]["status"]
            recorded, ledger, _ = _recorded_gate(project)
    except Refused as exc:
        return _prefuse(args, exc.problems)
    if report is None:
        count = f"has {len(ledger)} entries" if ledger is not None else "cannot be read"
        evidence = (f"a gate result is recorded ({recorded}) and the alpha ledger {count}"
                    if recorded else f"no gate result is recorded; the alpha ledger {count}")
        return _prefuse(args, [Problem("AN-11", "report", f"there is no current report: the report stage is {state}; "
                                       f"{evidence}", "run the stages in order up to `report`; `project status` "
                                       "shows where it stopped")])
    ok = report.get("status") == "PASS"
    md = project.root / "stages" / "report" / "report.md"
    human = md.read_text(encoding="utf-8") if md.is_file() else f"status: {report.get('status')}\n"
    stated = report.get("keeper_mode")                           # AN-27: a keeper report (version 3) names its mode
    try:
        mode, mode_warnings = _mode_surface(project, stated)
    except Refused as exc:
        return _prefuse(args, exc.problems)
    warnings = [*warnings, *mode_warnings]
    if mode is not None and (mode != stated or not human.startswith("keeper mode:")):
        from .ainative.keeper import texts

        human = f"{texts.MODE_LINE[mode]}\n" + (f"(the report states {stated or 'no mode'}; this check of the keeper "
                                                 f"shows {mode})\n" if mode != stated else "") + human
    return _pfinish(args, {"report": report, "report_md": str(md), "keeper_mode": mode, "report_keeper_mode": stated},
                    human, ok=ok, exit_code=EXIT_OK if ok else EXIT_NOT_PASS, warnings=warnings)


def _job_call(args, op):
    """AN-24: run `op(backend)` on the project's backend (`machine.compute_backend`); a refusal, or a daemon error
    (the operation and exit code only), is the refusal envelope."""
    from .ainative.machine import compute_backend
    from .ainative.project import Project
    from .ainative.runtime import RuntimeFailure

    try:
        return op(compute_backend(Project.load(Path(args.root)))), None
    except Refused as exc:
        return None, _prefuse(args, exc.problems)
    except RuntimeFailure as exc:
        return None, _prefuse(args, [Problem("AN-22", "daemon", str(exc),
                                             "check that the daemon runs: `docker --context gatedtrain info`")])


def _job_line(j) -> str:
    return (f"{j.job_id} {j.state} attempt {j.attempt} rc {j.rc} gpu_seconds {j.gpu_seconds}"
            + (f" ({j.reason})" if j.reason else "") + "\n")


def cmd_project_job_list(args) -> int:
    import dataclasses

    jobs, refused = _job_call(args, lambda b: b.jobs())          # the records only; no runtime
    if refused is not None:
        return refused
    return _pfinish(args, {"jobs": [dataclasses.asdict(j) for j in jobs]},
                    "".join(_job_line(j) for j in jobs) or "no jobs\n")


def cmd_project_job_logs(args) -> int:
    log, refused = _job_call(args, lambda b: b.logs(args.job_id, tail=args.tail))
    if refused is not None:
        return refused
    return _pfinish(args, {"job_id": args.job_id, "log": log}, log)


def cmd_project_job_cancel(args) -> int:
    import dataclasses

    job, refused = _job_call(args, lambda b: b.cancel(args.job_id))
    if refused is not None:
        return refused
    return _pfinish(args, {"job": dataclasses.asdict(job)}, _job_line(job))


def _sync_receipts(project) -> list[str]:
    """A keeper project's receipts get every envelope the keeper signed for it that they lack (`KeeperTrust.sync`);
    the report is printed either way, with a warning when the keeper cannot be asked."""
    from .ainative.keeper_trust import KeeperTrust, load_binding

    try:
        if load_binding(project) is None:
            return []
        KeeperTrust(project).sync()
    except Refused as exc:
        return [f"the receipts were not synced with the keeper: {p.line()}" for p in exc.problems]
    return []


def _gpu_hours(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(f"{text!r} must be a finite number, at least 0")
    return value


def _add_project(sub) -> None:
    p = sub.add_parser("project", help="AI-native project (step 1 preview): create, plan, run, status, report, job")
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
    c.add_argument("--keeper", metavar="SOCKET",
                   help="bind the project to the keeper at this socket (trust/keeper.json, trust/keeper.pub); the "
                        "keeper then seals, scores and gates (sealed from your account only in separate-user mode) "
                        "(AN-12, AN-14)")
    c.add_argument("--require-keeper-mode", choices=[m.replace("_", "-") for m in _keeper_modes().REQUIREMENTS],
                   default=_keeper_modes().REQUIRE_ANY,
                   help="the weakest keeper mode this project accepts, fixed now and signed into its registration "
                        "(AN-28); `separate-user` refuses a same-user keeper")
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

    c = ps.add_parser("job", help="the project's backend jobs (AN-24): list, logs, cancel")
    js = c.add_subparsers(dest="job_cmd", required=True)
    j = js.add_parser("list", help="every job's state, from the job records only")
    j.add_argument("root")
    j.set_defaults(func=cmd_project_job_list)
    j = js.add_parser("logs", help="a job's current attempt log")
    j.add_argument("root")
    j.add_argument("job_id", help="a job id from `project job list`")
    j.add_argument("--tail", type=int, default=200, metavar="N", help="the last N lines (default 200)")
    j.set_defaults(func=cmd_project_job_logs)
    j = js.add_parser("cancel", help="cancel a job: stop its running container; the end is recorded `cancelled`")
    j.add_argument("root")
    j.add_argument("job_id", help="a job id from `project job list`")
    j.set_defaults(func=cmd_project_job_cancel)


# AN-12, AN-14, AN-18, AN-19: `gatedtrain keeper ...`, `gatedtrain setup keeper`, `gatedtrain verify`.

def _command_name(args) -> str:
    if args.command == "project":
        return _project_command(args)
    sub = {"keeper": "keeper_cmd", "setup": "setup_cmd", "image": "image_cmd"}.get(args.command)
    return f"{args.command} {getattr(args, sub)}" if sub else args.command


def _kfinish(args, data: dict, human: str = "", *, ok: bool = True, exit_code: int = EXIT_OK,
             warnings: list[str] | None = None) -> int:
    return emit(Result(_command_name(args), ok, exit_code, data, [], list(warnings or [])), as_json=_as_json(args),
                human=human)


def _krefuse(args, problems: list[Problem]) -> int:
    return emit(Result(_command_name(args), False, EXIT_REFUSED, {}, list(problems), []), as_json=_as_json(args),
                human="")


def _inference_config(args):
    """AN-25: the keeper's inference options as an `InferenceConfig`, None without `--inference-image`; Refused
    (AN-13) for a malformed value or an inference option without the image."""
    from .ainative.keeper.inference import DEFAULT_DOCKER, DEFAULT_NVIDIA_SMI, InferenceConfig

    given = [name for name, value in (("--inference-ref", args.inference_ref), ("--eval-tree", args.eval_tree),
                                      ("--docker-host", args.docker_host), ("--docker", args.docker),
                                      ("--nvidia-smi", args.nvidia_smi))
             if value not in (None, [])]
    if args.inference_image is None:
        if given:
            raise Refused([Problem("AN-13", "inference_image", f"{given[0]} needs --inference-image",
                                   "pass the inference image ID (sha256:<64 hex>) the keeper's daemon holds")])
        return None
    try:
        return InferenceConfig(image_id=args.inference_image,
                               docker_host=args.docker_host or f"unix:///run/user/{os.geteuid()}/docker.sock",
                               eval_trees=tuple(Path(t) for t in args.eval_tree or ()), image_ref=args.inference_ref,
                               docker=Path(args.docker) if args.docker else DEFAULT_DOCKER,
                               nvidia_smi=Path(args.nvidia_smi) if args.nvidia_smi else DEFAULT_NVIDIA_SMI)
    except ValueError as exc:
        raise Refused([Problem("AN-13", "inference_image", str(exc),
                               "pass --inference-image sha256:<64 hex>, absolute --eval-tree, --docker and "
                               "--nvidia-smi paths, and --inference-ref pinned by digest")]) from None


def cmd_keeper_serve(args) -> int:
    from .ainative.keeper.server import serve

    try:
        inference = _inference_config(args)
        code = serve(Path(args.state).absolute(), Path(args.socket).absolute(), args.owner_uid, inference=inference)
    except Refused as exc:                               # AN-12, AN-13: the keeper does not start
        return _krefuse(args, exc.problems)
    return _kfinish(args, {}, exit_code=code)


def cmd_keeper_fingerprint(args) -> int:
    try:
        hello = _keeper_hello(Path(args.socket).absolute())
    except Refused as exc:
        return _krefuse(args, exc.problems)
    from .ainative.keeper import modes, texts

    mode = hello["effective_mode"]
    data = {"fingerprint": hello["fingerprint"], "sealed_guarantee": hello.get("sealed_guarantee"),
            "owner_has_root_docker": hello.get("owner_has_root_docker"),
            "root_docker_reasons": hello.get("root_docker_reasons"), "keeper_mode": mode,
            "mode_facts": hello.get("mode_facts")}
    reasons = "".join(f"  {r}\n" for r in data["root_docker_reasons"] or [])
    harden = "harden: gatedtrain setup keeper --mode separate-user (one sudo)\n" if mode == modes.SAME_USER else ""
    return _kfinish(args, data, f"fingerprint: {data['fingerprint']}\n{texts.MODE_LINE[mode]}\n{harden}{texts.MODE_STATED_NOTE}\n"
                                f"guarantee: {data['sealed_guarantee']}\n"
                                f"owner has rootful access (AN-21): {data['owner_has_root_docker']}\n{reasons}")


CONFIRM_PROMPT = "Type yes to record this confirmation in the keeper: "


def cmd_keeper_confirm(args) -> int:
    """AN-18: show the plan sentence of the decision's scorer, scorer_params and margin, take `yes` on stdin (or
    `--yes`), then record the confirmation in the keeper. The prompt goes to stderr, so stdout stays the result."""
    from .ainative import signing
    from .ainative.keeper import texts
    from .ainative.keeper_trust import KeeperTrust
    from .ainative.project import Project
    from .ainative.trust_stages import keeper_confirmation

    try:
        trust = KeeperTrust(Project.load(Path(args.root)))
        decision = _load_decision(args.decision)
        summary = trust.sealed_summary()
        values, sentence = keeper_confirmation(decision, summary.get("n"))
        if not args.yes and _as_json(args):          # AN-1: under --json stdout is one object and stderr stays empty
            raise Refused([Problem("AN-18", "--yes", f"--json cannot prompt; the sentence to confirm is: {sentence}",
                                   "rerun without --json to read the sentence and type yes, or pass --yes")])
        if not args.yes:
            print(texts.MODE_LINE[trust.keeper_mode], file=sys.stderr)       # AN-27: before the sentence
            print(sentence, file=sys.stderr)
            print(CONFIRM_PROMPT, end="", file=sys.stderr, flush=True)
            answer = sys.stdin.readline()
            if answer == "":                         # EOF: nothing was typed (stdin ended before an answer)
                raise Refused([Problem("AN-18", "stdin", "no answer was read: stdin ended before an answer",
                                       "read the sentence above and rerun with --yes to confirm it")])
            if answer.strip() != "yes":
                raise Refused([Problem("AN-18", "stdin", "the confirmation was not given (the answer was not `yes`)",
                                       "read the sentence and answer yes to confirm it, or change --decision "
                                       "first")])
        env = trust.confirm_metric(values, sentence)
    except Refused as exc:
        return _krefuse(args, exc.problems)
    sha = signing.envelope_sha256(env)
    return _kfinish(args, {"sentence": sentence, "envelope_sha256": sha, "keeper_mode": trust.keeper_mode},
                    f"{texts.MODE_LINE[trust.keeper_mode]}\n{sentence}\nconfirmed: {sha}\n",
                    warnings=_same_user_warnings(trust.keeper_mode) if _as_json(args) else None)   # the text has the line


def cmd_setup_keeper(args) -> int:
    """AN-19: print the privileged commands and the systemd unit; run and write nothing."""
    import os

    from .ainative import layout
    from .ainative.keeper import setup_user
    from .ainative.keeper.setup import installed_packs, setup_commands

    owner = os.getuid() if args.owner_uid is None else args.owner_uid
    defaulted = args.mode is None                         # the default changed in 0.7.0: say so when it was not chosen
    args.mode = args.mode or "same-user"
    try:                                                  # an empty value is a typo in both modes, never the default
        setup_user.refuse_empty(user=args.user, state=args.state, socket=args.socket, venv=args.venv,
                                executable=args.executable)
        lay = layout.resolve(args.mode, args, args.layout)           # AN-34: flag > layout file > default
    except Refused as exc:
        return _krefuse(args, exc.problems)
    if args.mode == "same-user":                          # AN-30, AN-31: the default, user-level commands only
        return _cmd_setup_keeper_same_user(args, owner, lay, [setup_user.DEFAULT_MODE_WARNING] if defaulted else [])
    try:
        if args.executable is not None:
            raise setup_user.executable_mode_problem()
        out = setup_commands(user=lay.values["user"], state=lay.values["state"], socket=lay.values["socket"],
                             venv=lay.values["venv"], owner_uid=owner,
                             packages=args.packages, inference_image=args.inference_image,
                             eval_trees=lay.values["eval_trees"], docker_host=args.docker_host)
    except Refused as exc:
        return _krefuse(args, exc.problems)
    out["commands"] = [*lay.header, *out["commands"]]
    warnings = []
    if args.packages is None and not installed_packs():
        warnings.append("no scenario pack is installed here; pass --packages NAME==VERSION for each pack the keeper "
                        "must run")
    human = printable("# run these once, as a user who may use sudo; gatedtrain runs none of them\n"
                      + "".join(c + "\n" for c in out["commands"]), keep="\n\t")
    return _kfinish(args, {**out, "mode": "separate_user", **({"layout": lay.info} if lay.info else {})}, human,
                    warnings=warnings)


def _cmd_setup_keeper_same_user(args, owner: int, lay, warnings: list[str]) -> int:
    """AN-30, AN-31: print user-level commands and a systemd user unit; no sudo, run and write nothing."""
    from .ainative.keeper import setup_user

    try:
        for field in ("user", "venv"):
            if getattr(args, field) is not None:
                raise setup_user.mode_option_problem(f"--{field}")
        out = setup_user.setup_commands_same_user(owner_uid=owner, state=lay.values["state"],
                                                  socket=lay.values["socket"], executable=lay.values["executable"],
                                                  inference_image=args.inference_image,
                                                  eval_trees=lay.values["eval_trees"], docker_host=args.docker_host)
    except Refused as exc:
        return _krefuse(args, exc.problems)
    if args.packages is not None:
        warnings = [*warnings, "--packages is ignored in same-user mode: the keeper runs from the current "
                    "environment, with the packs installed here"]
    out["commands"] = [*lay.header, *out["commands"]]
    # Defence in depth: the validators refuse non-printable values; whatever slips through is shown escaped.
    human = printable(setup_user.HEADER + "\n" + "".join(c + "\n" for c in out["commands"]), keep="\n\t")
    return _kfinish(args, {**out, **({"layout": lay.info} if lay.info else {})}, human, warnings=warnings)


def cmd_setup_host(args) -> int:
    """AN-20: print the root script that sets up rootless Docker for the owner and the keeper; run and write
    nothing. Plain mode prints the script only, so stdout is the file."""
    import os
    import pwd

    from .ainative import layout
    from .ainative.hostsetup import RULE, host_script

    try:
        lay = layout.resolve("host", args, args.layout)             # AN-34: flag > layout file > default
        v = lay.values
        if args.owner is None:
            owner, uid = pwd.getpwuid(os.getuid()).pw_name, os.getuid() if args.owner_uid is None else args.owner_uid
        else:
            owner, uid = args.owner, args.owner_uid
            if uid is None:
                try:
                    uid = pwd.getpwnam(owner).pw_uid
                except KeyError:
                    raise Refused([Problem(RULE, "--owner-uid", f"there is no user {owner!r} on this host to take "
                                           "the uid from", "pass --owner-uid")]) from None
        requirements = None
        if v["eval_requirements"] is not None:
            try:
                requirements = Path(v["eval_requirements"]).read_bytes()
            except OSError as exc:
                raise Refused([Problem(RULE, "--eval-requirements", f"the requirements file cannot be read "
                                       f"({type(exc).__name__})", "pass a readable requirements file")]) from None
        out = host_script(owner=owner, owner_uid=uid, keeper_user=v["user"], keeper_home=v["keeper_home"],
                          keeper_data_root=v["keeper_data_root"], owner_data_root=v["owner_data_root"],
                          handoff=v["handoff"], state=v["state"], socket=v["socket"], venv=v["venv"],
                          inference_ref=v["image"], eval_requirements=requirements,
                          eval_python=v["eval_python"], eval_venv=v["eval_venv"],
                          eval_python_root=v["eval_python_root"], confirm_group_removal=args.confirm_group_removal)
    except Refused as exc:
        return _krefuse(args, exc.problems)
    if lay.info:                                   # the header goes after the shebang
        first, _, rest = out["script"].partition("\n")
        out = {**out, "script": first + "\n" + "".join(h + "\n" for h in lay.header) + rest, "layout": lay.info}
    return _kfinish(args, out, printable(out["script"], keep="\n\t"))


def cmd_verify(args) -> int:
    """AN-14: exit 0 when the project's signed records check against an anchor (the keeper's key or --fingerprint),
    1 when there is no anchor and only their consistency was checked (no anchor given, a keeper that runs as the
    owner's own uid, or one whose owner reaches a rootful daemon (AN-21), either of which can read its key), 1 also
    under any anchor when a report states any guarantee but the separate-user one (`report_warnings`), 3 on a bad
    signature or a mismatch. A result without the keeper as the anchor says its mode is as stated by the holder of
    the key (`MODE_STATED_NOTE`, a warning, never the exit code)."""
    from .ainative import image_provenance, verify_mode
    from .ainative.keeper import texts
    from .ainative.verify import (CONSISTENCY_ONLY, MODE_UNKNOWN_HEADLINE, NO_REPORT_IMAGES,
                                  OFFLINE_LIMIT, ROOT_DOCKER_KEEPER, SAME_USER_KEEPER, report_warnings, verify_project)

    try:
        out = verify_project(Path(args.root), keeper=Path(args.keeper) if args.keeper else None,
                             fingerprint=args.fingerprint)
    except Refused as exc:
        return _krefuse(args, exc.problems)
    reports = report_warnings(out)
    has_report = bool(out.get("report_guarantees"))
    if has_report:
        img_warn, img_fail = image_provenance.image_check(out, args.require_maintainer_images)      # IMG-9
    else:               # a project with no report states no image: the flag cannot be met (it must not pass vacuously)
        img_warn, img_fail = ([NO_REPORT_IMAGES] if args.require_maintainer_images else []), args.require_maintainer_images
    mode = out.get("effective_mode")
    stated = [texts.MODE_STATED_NOTE] if out["anchor"] != "keeper" and mode is not None else []     # AN-27

    def fin(args, data, human="", *, ok=True, exit_code=EXIT_OK, warnings=None):
        """User-added or undeclared trainer images warn; with --require-maintainer-images they also make the exit 1."""
        if img_fail and ok:
            ok, exit_code = False, EXIT_NOT_PASS
        limit = f"{texts.LIMIT_IMAGES_NOT_OBSERVED}\n" if has_report and out.get("images_undeclared") == \
            image_provenance.IMAGES_STATED else ""
        return _kfinish(args, data, human + image_provenance.lock_line(out) + limit, ok=ok, exit_code=exit_code,
                        warnings=[*(warnings or []), *stated, *img_warn])

    if verify_mode.accepted_weaker(out, args.accept_mode):          # AN-27: the reader's own, per-call decision
        mode_warnings = [w for w in (SAME_USER_KEEPER if out["same_user"] else None,
                                     OFFLINE_LIMIT if out["anchor"] != "keeper" else None, *reports) if w]
        return fin(args, {**out, "accepted_weaker_mode": True},
                   f"accepted (weaker mode): {out['checked']} records under key {out['fingerprint']} checked against "
                   f"the {out['anchor']}\n{verify_mode.ACCEPTED_WEAKER_MODE}\n", warnings=mode_warnings)
    if out["anchor"] == "none":
        return fin(args, out, f"consistent: {out['checked']} records under key {out['fingerprint']}\n",
                        ok=False, exit_code=EXIT_NOT_PASS, warnings=[CONSISTENCY_ONLY, OFFLINE_LIMIT, *reports])
    if out["anchor"] == "keeper" and out["same_user"]:
        return fin(args, out, f"consistent with a same-user keeper: {out['checked']} records under key "
                                   f"{out['fingerprint']}\n", ok=False, exit_code=EXIT_NOT_PASS,
                        warnings=[SAME_USER_KEEPER, *reports])
    if out["anchor"] == "keeper" and out["root_docker"]:
        return fin(args, out, f"consistent with a keeper whose owner reaches a rootful daemon: {out['checked']} "
                                   f"records under key {out['fingerprint']}\n", ok=False, exit_code=EXIT_NOT_PASS,
                        warnings=[ROOT_DOCKER_KEEPER, *reports])
    warnings = [] if out["anchor"] == "keeper" else [OFFLINE_LIMIT]       # only the keeper sees a deleted record
    if out.get("mode_unknown"):               # no mode in any record (a project written before 0.7.0): not a report fault
        return fin(args, out, f"verified against the {out['anchor']}, but {MODE_UNKNOWN_HEADLINE}: {out['checked']} "
                                   f"records under key {out['fingerprint']}\n", ok=False, exit_code=EXIT_NOT_PASS,
                        warnings=[*warnings, *reports])
    if reports:
        return fin(args, out, f"verified against the {out['anchor']}, but the report's guarantee does not hold: "
                                   f"{out['checked']} records under key {out['fingerprint']}\n", ok=False,
                        exit_code=EXIT_NOT_PASS, warnings=[*warnings, *reports])
    line = "" if mode is None else (texts.MODE_LINE[mode] if out["anchor"] == "keeper" else
                                    f"mode as stated in the signed records: {mode}") + "\n"
    return fin(args, out, f"verified against the {out['anchor']}: {out['checked']} records under key "
                               f"{out['fingerprint']}\n{line}", warnings=warnings)


def verify_mode_choices() -> list[str]:
    from .ainative.verify_mode import ACCEPT_MODES

    return list(ACCEPT_MODES)


def _add_keeper(sub) -> None:
    from .ainative.keeper import setup

    p = sub.add_parser("keeper", help="AI-native keeper (Linux only): serve, fingerprint, confirm")
    ks = p.add_subparsers(dest="keeper_cmd", required=True)
    c = ks.add_parser("serve", help="serve the keeper on a Unix socket until SIGTERM")
    c.add_argument("--state", required=True, metavar="DIR", help="the keeper's state directory")
    c.add_argument("--socket", required=True, metavar="PATH", help="the Unix socket path (at most 107 bytes)")
    c.add_argument("--owner-uid", required=True, type=int, metavar="UID",
                   help="the uid allowed to connect besides the keeper's own")
    c.add_argument("--inference-image", metavar="ID",
                   help="run batch generators in containers of this image ID (sha256:<64 hex>) on the keeper's own "
                        "rootless Docker daemon (AN-25); without it a batch generator is refused")
    c.add_argument("--inference-ref", metavar="REF",
                   help="the reference (pinned by digest) the daemon holds the image under (default: the ID)")
    c.add_argument("--eval-tree", action="append", metavar="DIR",
                   help="a root-owned tree mounted read-only at its own path into the container (repeatable)")
    c.add_argument("--docker-host", metavar="URL",
                   help="the keeper's rootless daemon (default unix:///run/user/<keeper uid>/docker.sock)")
    c.add_argument("--docker", metavar="PATH", help="the Docker CLI's absolute path (default /usr/bin/docker)")
    c.add_argument("--nvidia-smi", metavar="PATH",
                   help="nvidia-smi's absolute path, for a GPU batch generator (default /usr/bin/nvidia-smi)")
    c.set_defaults(func=cmd_keeper_serve)

    c = ks.add_parser("fingerprint", help="print the fingerprint of the key the keeper serves; keep it out of band")
    c.add_argument("--socket", required=True, metavar="PATH")
    c.set_defaults(func=cmd_keeper_fingerprint)

    c = ks.add_parser("confirm", help="show the deciding metric's sentence and record its confirmation (AN-18)")
    c.add_argument("root", help="the project directory, bound to a keeper")
    c.add_argument("--decision", metavar="FILE",
                   help="the preregister decision (JSON): its scorer, scorer_params and margin are confirmed")
    c.add_argument("--yes", action="store_true", help="confirm without reading `yes` from stdin")
    c.set_defaults(func=cmd_keeper_confirm)

    p = sub.add_parser("setup", help="print the commands that install a component; runs nothing (AN-19)")
    ss = p.add_subparsers(dest="setup_cmd", required=True)
    c = ss.add_parser("keeper", help="print the commands and the systemd unit that install the keeper")
    c.add_argument("--mode", choices=("same-user", "separate-user"), default=None,
                   help="same-user (the default since 0.7.0, with a warning when left out): your own user, no sudo, "
                        "no sealing from your account; separate-user: a keeper user, sudo commands, the sealed-data "
                        "guarantee")
    from . import cli_layout

    cli_layout.add_layout_option(c)
    c.add_argument("--user", help=f"separate-user: the keeper's system user (default {setup.DEFAULT_USER})")
    c.add_argument("--state", metavar="DIR", help="the keeper's state directory (default: per mode)")
    c.add_argument("--socket", metavar="PATH", help="the keeper's socket (default: per mode)")
    c.add_argument("--venv", metavar="DIR", help=f"separate-user: the root-owned virtual environment (default "
                                                f"{setup.DEFAULT_VENV})")
    c.add_argument("--executable", metavar="PATH",
                   help="same-user: the gatedtrain command the unit runs (default: the one next to this Python; give "
                        "a link at a plain path when that path holds a space, `%%` or `$`)")
    c.add_argument("--owner-uid", type=int, metavar="UID", help="the user the keeper serves (default: yours)")
    c.add_argument("--packages", nargs="+", metavar="SPEC",
                   help="scenario packs as NAME==VERSION (default: the installed packs at their versions)")
    c.add_argument("--inference-image", metavar="ID",
                   help="the inference image ID (sha256:<64 hex>) printed into ExecStart= (AN-25)")
    c.add_argument("--eval-tree", action="append", metavar="DIR",
                   help="a root-owned eval tree printed into ExecStart= (repeatable; needs --inference-image)")
    c.add_argument("--docker-host", metavar="URL",
                   help="the keeper's rootless daemon printed into ExecStart= (needs --inference-image)")
    c.set_defaults(func=cmd_setup_keeper)

    from .ainative import hostsetup

    c = ss.add_parser("host", help="print the root script that sets up rootless Docker for you and the keeper "
                                   "(AN-20); runs nothing")
    c.add_argument("--owner", metavar="NAME", help="the user the keeper serves (default: you)")
    c.add_argument("--owner-uid", type=int, metavar="UID", help="the owner's uid (default: the owner's uid here)")
    cli_layout.add_layout_option(c)
    c.add_argument("--keeper-user", metavar="NAME", help="the keeper's system user")
    c.add_argument("--keeper-home", metavar="DIR")
    c.add_argument("--keeper-data-root", metavar="DIR",
                   help="the keeper's rootless Docker data root")
    c.add_argument("--owner-data-root", metavar="DIR",
                   help="the owner's rootless Docker data root")
    c.add_argument("--handoff", metavar="DIR",
                   help="the hand-off directory (the owner's output_root)")
    c.add_argument("--state", metavar="DIR", help="the keeper's state (as setup keeper)")
    c.add_argument("--socket", metavar="PATH", help="the keeper's socket (as setup "
                                                                                  "keeper)")
    c.add_argument("--venv", metavar="DIR", help="the keeper's venv (as setup keeper)")
    c.add_argument("--inference-image", metavar="REF@sha256:...",
                   help="the inference base image, pinned by digest, the keeper's daemon pulls; its ID goes into a "
                        "drop-in of the keeper's unit")
    c.add_argument("--eval-requirements", metavar="FILE",
                   help="hash-pinned requirements of the keeper's eval venv (needs --eval-python)")
    c.add_argument("--eval-python", metavar="PATH",
                   help=f"the interpreter the eval venv is created from, under {hostsetup.DEFAULT_EVAL_PYTHON_ROOT}")
    c.add_argument("--eval-venv", metavar="DIR",
                   help="the keeper's root-owned eval venv")
    c.add_argument("--confirm-group-removal", action="store_true",
                   help="pause before removing you from the docker group")
    c.set_defaults(func=cmd_setup_host)

    from . import cli_setup_check

    cli_setup_check.add_parser(ss)
    cli_layout.add_parser(ss)

    p = sub.add_parser("verify", help="check a keeper project's signed records against the keeper or a fingerprint "
                                      "(AN-14)")
    p.add_argument("root", help="the project directory")
    p.add_argument("--keeper", metavar="SOCKET", help="anchor on the key the keeper at this socket serves")
    p.add_argument("--fingerprint", metavar="FP", help="anchor on a key fingerprint held out of band")
    p.add_argument("--accept-mode", choices=verify_mode_choices(),
                   help="accept a same-user result (exit 0, stated in the output); a stricter project policy still "
                        "refuses it (AN-27, AN-28)")
    p.add_argument("--require-maintainer-images", action="store_true",
                   help="exit 1 unless a report states its trainer images and every one is in the maintainer lock: a "
                        "project with no report, a report with no image list (undeclared) or a user-added image "
                        "fails (IMG-9)")
    p.set_defaults(func=cmd_verify)


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

    from . import cli_image                              # IMG-5, IMG-6: `gatedtrain image list|check|add`
    cli_image.register(sub)

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
    _add_keeper(sub)
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
        with notice_scope():          # warnings raised below the command are queued for `emit` (also for INTERNAL)
            try:
                return args.func(args)
            except Exception as exc:
                if not _as_json(args):
                    raise
                name = _command_name(args)
                problem = Problem("INTERNAL", "", f"unexpected {type(exc).__name__}; this is a bug", "report it")
                return emit(Result(name, False, EXIT_NOT_PASS, {}, [problem],
                                   list(getattr(args, "ext_warnings", []))), as_json=True, human="")
    finally:
        runner.QUIET = False


if __name__ == "__main__":                                # EXT-89: `python -m gatedtrain.cli` runs the CLI
    sys.exit(main())
