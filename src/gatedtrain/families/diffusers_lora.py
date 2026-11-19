"""DiffusersLoraAdapter: shared by several recipes' adapters (moved verbatim from ddpo_sd15.DdpoSd15Adapter)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..canonical import canonical_json, entries_hash, manifest_entries, sha256_bytes
from ..errors import Problem
from ..ext.registry import EVALUATOR_ROWS
from ..ext.sdk import (compact_sha256, is_int as _int, read_cut_marker, read_json as _read_json, read_jsonl,
                       read_run_args, runtime_script_problems, sha_or_none as _sha_or_none, single_shard as _shard)

CUT_MARKER = "cut.json"                                       # written last (the launchers' save_cut)
CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")        # `.checkpoint-<E>.partial` never


class DiffusersLoraAdapter:
    """Family base (from ddpo_sd15.DdpoSd15Adapter): the members several recipes' adapters share. A
    subclass sets the layout constants these read through `self` and implements the hooks they call."""

    # Layout defaults of the diffusers LoRA launchers (train_ddpo.py, train_flow_grpo.py and its Wan port).
    BASE_REF = "base_ref.json"
    CHECKPOINT_DIR_RE = CHECKPOINT_DIR_RE
    REVISION_RE = re.compile(r"^[0-9a-f]{40}$")

    def __init__(self) -> None:
        self._base_cache: dict[str, list[dict]] = {}

    def metric(self, evaluator_id: str) -> dict[str, Any]:
        if evaluator_id not in self.evaluator_ids:
            raise KeyError(evaluator_id)
        row = EVALUATOR_ROWS[evaluator_id]
        return {"metric": row["metric"], "direction": row["direction"], "score_range": tuple(row["score_range"]),
                "error_classes": tuple(row["error_classes"]), "score_outcomes": tuple(row["score_outcomes"]),
                "scorer": dict(row["scorer"]), "binary": row["binary"], "tier": row["tier"]}

    @staticmethod
    def is_cut(d: Path) -> bool:
        """Whether `d` is (meant to be) a LoRA cut: it holds cut.json. The plain base directory never does."""
        return (Path(d) / CUT_MARKER).exists() or (Path(d) / CUT_MARKER).is_symlink()

    def needs_base(self, d: Path) -> bool:
        """ext-arm-fill: a cut's model hash composes the base (MM-18), so `--base` is required."""
        return self.is_cut(d)

    def _cut_files_problem(self, d: Path) -> str | None:
        """Report a checkpoint directory as incomplete if one of its files is unreadable (EXT-154)."""
        try:
            return self._cut_files_problem_unguarded(d)
        except OSError as exc:
            return (f"{Path(exc.filename).name if exc.filename else Path(d).name} cannot be read "
                    f"({exc.__class__.__name__}: {exc.strerror or exc})")

    def model_dir_problem(self, d: Path) -> str | None:
        """None when `d` is a complete LoRA cut (it holds cut.json) or a plain SD1.5 base directory."""
        d = Path(d)
        if d.is_symlink() or not d.is_dir():
            return f"{d} is not a plain directory"
        if self.is_cut(d):
            return self._cut_files_problem(d)
        return self.base_dir_problem(d)

    def cut_problem(self, d: Path, epoch: int) -> str | None:
        d = Path(d)
        if d.is_symlink() or not d.is_dir():
            return f"{d} is not a plain directory"
        bad = self._cut_files_problem(d)
        if bad:
            return bad
        doc, _ = read_cut_marker(d)
        if doc.get("epoch") != epoch:
            return f"{self.CUT_MARKER} epoch is {doc.get('epoch')!r}, not {epoch}"
        return None

    def checkpoint_dirs(self, stage_out: Path) -> list[tuple[int, Path]]:
        """Every `checkpoint-<E>` directory, complete or not, in epoch order; `.checkpoint-<E>.partial` is never one."""
        stage_out = Path(stage_out)
        if not stage_out.is_dir():
            return []
        out = []
        for p in stage_out.iterdir():
            m = self.CHECKPOINT_DIR_RE.match(p.name)
            if m and p.is_dir() and not p.is_symlink():
                out.append((int(m.group(1)), p))
        return sorted(out)

    def find_cuts(self, stage_out: Path) -> tuple[list[tuple[int, Path]], list[Path]]:
        """(complete [(global_step, path)] in epoch order, incomplete [path]): a cut's last step included is the
        optimizer step of its cut.json (05 section 12.4), the ledger's `step` (EXT-119)."""
        complete, incomplete = [], []
        for epoch, d in self.checkpoint_dirs(stage_out):
            if self.cut_problem(d, epoch) is None:
                complete.append((read_cut_marker(d)[0]["global_step"], d))
            else:
                incomplete.append(d)
        return complete, incomplete

    def identity(self, cut: Path, base: Any = None, *, entries: list[dict] | None = None) -> dict[str, Any]:
        """EXT-120 (MM-17, MM-18). The base alone: `model_hash` = `cut_hash` = its manifest hash. A cut: `adapter_hash`
        over the deployable files, `model_hash` = sha256(canonical_json({base, adapter})), `cut_hash` over the whole
        cut, `base_ref` = {repo, revision (from base_ref.json), manifest_hash of the base}. Raises ValueError for a cut
        without `base`."""
        is_cut = self.is_cut(cut)
        if entries is None and not is_cut:                # EXT-135: the base is walked once per adapter instance
            entries = self._base_entries(Path(cut))
        if entries is None:
            entries = manifest_entries(Path(cut))
        if not is_cut:
            self._base_cache.setdefault(str(Path(cut).resolve()), entries)    # collect walks the base once
            h = entries_hash(entries)
            return {"model_hash": h, "cut_hash": h, "components": [e["path"] for e in entries], "base_ref": None}
        if base is None:
            raise ValueError(f"{cut} is a LoRA cut: its model hash composes the base directory's hash (MM-18), so the "
                             "base directory is needed")
        comps = [e for e in entries if e["path"] in self.COMPONENTS]
        adapter = entries_hash(comps)
        base_hash = entries_hash(self._base_entries(Path(base)))
        ref, _ = _read_json(Path(cut) / self.BASE_REF)
        ref = ref if isinstance(ref, dict) else {}
        return {"model_hash": sha256_bytes(canonical_json({"base": base_hash, "adapter": adapter})),
                "cut_hash": entries_hash(entries), "adapter_hash": adapter,
                "components": [e["path"] for e in comps],
                "base_ref": {"repo": ref.get("repo"), "revision": ref.get("revision"), "manifest_hash": base_hash}}

    def consumption_problems(self, stage_out: Path, data: Path, cuts: list) -> list[str]:
        """EXT-123: why the observed logs cannot give the per-sample consumption up to the last cut."""
        return self._ledger_plan(stage_out, data, cuts)[1]

    def training_payloads(self, stage_out: Path, base: Path, cuts: list, *,
                          data: Path | None = None) -> tuple[set[str] | None, str | None]:
        """Every prompt text the run could train on: the spec's training shard (`data`) and every sampled prompt of
        samples.jsonl up to the last cut's epoch. (None, why) when a file cannot be read."""
        out: set[str] = set()
        if data is not None:
            shard = _shard(data)
            if shard is None:
                return None, f"{data} holds no single training shard"
            rows, why = read_jsonl(Path(data) / shard)
            if rows is None:
                return None, why
            out |= {r["prompt"] for r in rows if isinstance(r.get("prompt"), str)}
        last_epoch = 0
        for _, d in cuts:
            doc, why = read_cut_marker(d)
            if doc is None or not _int(doc.get("epoch")):
                return None, f"{Path(d).name}: {why or 'cut.json has no epoch'}"
            last_epoch = max(last_epoch, doc["epoch"])
        samples, why = read_jsonl(Path(stage_out) / self.SAMPLES)
        if samples is None:
            return None, why
        out |= {x["prompt"] for x in samples if isinstance(x.get("prompt"), str) and _int(x.get("epoch"))
                and x["epoch"] <= last_epoch}
        return out, None

    def script_problems(self, stage_out: Path, launch: dict) -> list[Problem]:
        """EXT-83: env-runtime.json `script_sha256` equals the record's (train_ddpo.py writes it as area 1 does)."""
        return runtime_script_problems(stage_out, launch)

    def mount_binding_problems(self, mounts: Any, *, base: Path, data: Path, base_field: str, data_field: str,
                               flag: str = "--launch-record") -> list[Problem]:
        """`ext-launch --spec` (optional for this adapter): the spec's base directory and its prompt directory lie
        inside read-only mounts (EXT-121, EXT-122)."""
        from ..ext.launch import _ro_table, _within

        table = _ro_table(mounts)
        out = []
        if not any(_within(str(Path(base).resolve()), host) for _, host in table):
            out.append(Problem("EXT-BASE", base_field, f"{flag} mounts no read-only directory that is or holds the "
                               f"spec's base directory {base}", self.BASE_FIX))
        if not any(_within(str(Path(data).resolve()), host) for _, host in table):
            out.append(Problem("EXT-DATA", data_field, f"{flag} mounts no read-only directory that is or holds the "
                               f"spec's training prompts {data}", self.DATA_FIX))
        return out

    def _base_files(self, base: Path) -> dict[str, str]:
        """{relative path: "sha256:<hex>"} of the base directory, base_ref.json's `files` shape."""
        return {e["path"]: "sha256:" + e["sha256"] for e in self._base_entries(base)}

    def checks(self, *, last_cut: Path, base: Path, launch: dict, base_field: str, stage_out: Path | None = None,
               data: Path | None = None, data_field: str = "data", **_: Any) -> list[Problem]:
        """EXT-121 base binding, EXT-122 data binding, EXT-125 seed, profile, last cut, script and sampler code."""
        from ..ext.launch import container_file

        stage_out = Path(stage_out) if stage_out is not None else Path(last_cut).parent
        out: list[Problem] = []
        args, why = read_run_args(stage_out)
        if args is None:
            return [Problem("EXT-BASE", base_field, f"{why}, so the run cannot be bound to the spec's base (EXT-121)",
                            self.BASE_FIX)]
        mounts = launch.get("mounts")
        cuts = [d for _, d in self.find_cuts(stage_out)[0]] or [Path(last_cut)]
        # EXT-121 (MM-21): the run loaded the spec's base through a read-only mount and re-hashed it; every cut's
        # base_ref.json names that base by a 40-hex revision and its files' SHA-256; adapter_config.json agrees.
        got = args.get("base_model")
        hit = container_file(f"{got.rstrip('/')}/model_index.json", mounts) if isinstance(got, str) and got else None
        if hit is None or Path(hit[1]).parent.resolve() != Path(base).resolve():
            out.append(Problem("EXT-BASE", base_field, f"run_args.json base_model {got!r} is not, through a read-only "
                               f"mount, the spec's base directory {base} (MM-21)", self.BASE_FIX))
        rb = args.get("base_ref") if isinstance(args.get("base_ref"), dict) else {}
        if rb.get("verified") is not True:
            out.append(Problem("EXT-BASE", base_field, f"run_args.json base_ref.verified is {rb.get('verified')!r}: the "
                               "run did not re-hash the base files against MODELS.json (--no-verify-base)",
                               "train without --no-verify-base"))
        try:
            files = self._base_files(base)
        except (OSError, ValueError):
            files = None                                  # collect reports an unreadable base (EXT-BASE)
        for d in cuts:
            ref, why = _read_json(Path(d) / self.BASE_REF)
            if not isinstance(ref, dict):
                out.append(Problem("EXT-BASE", base_field, f"{Path(d).name}/{self.BASE_REF}: {why or 'not an object'}",
                                   self.BASE_FIX))
                continue
            rev = ref.get("revision")
            if not (isinstance(rev, str) and self.REVISION_RE.fullmatch(rev)):
                out.append(Problem("EXT-BASE", base_field, f"{Path(d).name}/{self.BASE_REF} revision {rev!r} is not a 40-hex "
                                   "commit (a branch name is refused, MM-18)", self.BASE_FIX))
            if files is not None and ref.get("files") != files:
                out.append(Problem("EXT-BASE", base_field, f"{Path(d).name}/{self.BASE_REF} files are not the SHA-256 of the "
                                   f"spec's base directory {base} (the run trained another base, MM-21)",
                                   self.BASE_FIX))
            elif isinstance(ref.get("files"), dict) and ref.get("files_sha256") != compact_sha256(ref["files"]):
                out.append(Problem("EXT-BASE", base_field, f"{Path(d).name}/{self.BASE_REF} files_sha256 is not the hash of "
                                   "its files", self.BASE_FIX))
            if (rb.get("repo"), rb.get("revision"), rb.get("files_sha256")) != (ref.get("repo"), rev,
                                                                               ref.get("files_sha256")):
                out.append(Problem("EXT-BASE", base_field, f"run_args.json base_ref is not {Path(d).name}/{self.BASE_REF} "
                                   "(repo, revision, files_sha256)", self.BASE_FIX))
            cfg, why = _read_json(Path(d) / "adapter_config.json")
            cfg = cfg if isinstance(cfg, dict) else {}
            if (cfg.get("base_model_name_or_path"), cfg.get("base_model_revision")) != (ref.get("repo"), rev):
                out.append(Problem("EXT-BASE", base_field, f"{Path(d).name}/adapter_config.json names base "
                                   f"{cfg.get('base_model_name_or_path')!r} @ {cfg.get('base_model_revision')!r}, not "
                                   f"{self.BASE_REF}'s {ref.get('repo')!r} @ {rev!r} (MM-21)", self.BASE_FIX))
        # EXT-122: the prompt shard the run read is the spec's, through a read-only mount.
        if data is not None:
            shard = _shard(data)
            path = args.get("train_file")
            if shard is None:
                out.append(Problem("EXT-DATA", data_field, f"the spec's training prompts {data} hold "
                                   f"{len(list(Path(data).glob('*.jsonl')))} shards, not the one shard the run read",
                                   "point the training dataset at the run's prompt directory (one train.jsonl)"))
            else:
                hit = container_file(path, mounts) if isinstance(path, str) else None
                if hit is None or Path(hit[1]).resolve() != (Path(data) / shard).resolve():
                    out.append(Problem("EXT-DATA", data_field, f"run_args.json train_file {path!r} is not, through a "
                                       f"read-only mount, the spec's shard {Path(data) / shard} (EXT-122)",
                                       self.DATA_FIX))
                elif args.get("train_file_sha256") != _sha_or_none(Path(data) / shard):
                    out.append(Problem("EXT-DATA", data_field, f"run_args.json train_file_sha256 "
                                       f"{args.get('train_file_sha256')!r} is not the SHA-256 of {Path(data) / shard} "
                                       "(EXT-122)", self.DATA_FIX))
        # EXT-125: seed, profile, script, sampler code, and the run finished its epochs.
        claimed, field = launch.get("training_seed"), "--launch-record.training_seed"
        seed = args.get("seed")
        if not (_int(seed) and seed == claimed):
            out.append(Problem("EXT-LAUNCH", field, f"the launch record claims training_seed {claimed!r}, but "
                               f"run_args.json `seed` is {seed!r} (MM-29)", "launch with the seed the record names"))
        if args.get("profile") != launch.get("profile"):
            out.append(Problem("EXT-PROFILE", "--launch-record.profile", f"run_args.json `profile` is "
                               f"{args.get('profile')!r}, the launch record's {launch.get('profile')!r}",
                               "collect the run launched with this record's profile"))
        rt, _ = _read_json(stage_out / self.RUNTIME_FILE)
        code = rt.get("sampler_code_sha256") if isinstance(rt, dict) else None
        for d in cuts:
            doc, _ = read_cut_marker(d)
            doc = doc or {}
            if doc.get("seed") != claimed:
                out.append(Problem("EXT-LAUNCH", field, f"{Path(d).name}/cut.json seed is {doc.get('seed')!r}, not the "
                                   f"launch record's training_seed {claimed!r} (MM-29)",
                                   "collect the cuts of the run launched with this record"))
            if doc.get("script_sha256") != launch.get("script_sha256"):
                out.append(Problem("EXT-LAUNCH", "--launch-record.script_sha256", f"{Path(d).name}/cut.json "
                                   f"script_sha256 is {doc.get('script_sha256')!r}, not the launch record's "
                                   f"{launch.get('script_sha256')!r} (EXT-83)",
                                   "collect the cuts of the run launched with this record's training script"))
            if doc.get("profile") != launch.get("profile"):
                out.append(Problem("EXT-PROFILE", "--launch-record.profile", f"{Path(d).name}/cut.json profile is "
                                   f"{doc.get('profile')!r}, the launch record's {launch.get('profile')!r}",
                                   "collect the run launched with this record's profile"))
            if not (isinstance(code, str) and doc.get("sampler_code_sha256") == code == args.get("sampler_code_sha256")):
                out.append(Problem("EXT-SAMPLER", "--stage-out", f"{Path(d).name}/cut.json sampler_code_sha256 "
                                   f"{doc.get('sampler_code_sha256')!r}, {self.RUNTIME_FILE} {code!r} and run_args.json "
                                   f"{args.get('sampler_code_sha256')!r} differ: the cut was not sampled by this run's "
                                   "sampler code", "collect the cuts of this run as it wrote them"))
            smp, _ = _read_json(Path(d) / "sampler.json")
            if smp != args.get("sampler"):
                out.append(Problem("EXT-SAMPLER", "--stage-out", f"{Path(d).name}/sampler.json is not run_args.json "
                                   "`sampler`", "collect the cuts of this run as it wrote them"))
        s = args.get("settings") if isinstance(args.get("settings"), dict) else {}
        m = self.CHECKPOINT_DIR_RE.match(Path(last_cut).name)
        last = int(m.group(1)) if m else None
        if not (_int(s.get("epochs")) and s["epochs"] == last):
            out.append(Problem("EXT-STAGE", "--stage-out", f"the last cut is epoch {last}, but run_args.json "
                               f"settings.epochs is {s.get('epochs')!r}: the run did not save its last epoch",
                               "collect a run that finished and saved its last epoch"))
        return out

    def sampler_problems(self, cuts: list[Path], sampler: Any) -> list[Problem]:
        """EXT-125: every cut's deployment sampler (sampler.json, a model component) is the registered sampler."""
        from ..ext import arm

        out = []
        for d in cuts:
            doc, why = _read_json(Path(d) / "sampler.json")
            if not arm.same_json(doc, sampler):
                out.append(Problem("EXT-SAMPLER", "registration.sampler", f"{Path(d).name}/sampler.json is not the "
                                   "registered sampler: the cut deploys another sampler than the gate evaluates",
                                   "register the deployment sampler the cuts carry (their sampler.json)"))
        return out
