"""FlowGrpoAdapter: shared by several recipes' adapters (moved verbatim from flow_grpo_sd35.FlowGrpoSd35Adapter)."""

from __future__ import annotations

import hashlib
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

from ..canonical import canonical_json, file_sha256, manifest_entries, manifest_hash, sha256_bytes, text_lines
from ..errors import Problem
from ..paths import resolve
from ..ext.sdk import (compact_sha256, derive_seed, is_int as _int, is_num as _num, plain_file as _plain_file,
                       read_cut_marker, read_json as _read_json, read_jsonl, read_run_args, safetensors_problem,
                       single_shard as _shard)
from .diffusers_lora import CHECKPOINT_DIR_RE, CUT_MARKER, DiffusersLoraAdapter
from ..ext.sdk import MAX_SEED, RUNTIME_FILE

# Layout of the flow_grpo launchers (train_flow_grpo.py and its Wan port; flow_grpo yifan123 879042cf, ported).
LORA_WEIGHTS = "pytorch_lora_weights.safetensors"
CUT_FILES = (LORA_WEIGHTS, "adapter_config.json", "sampler.json", "base_ref.json", "trainer_state.json")
COMPONENTS = (LORA_WEIGHTS, "adapter_config.json", "sampler.json", "base_ref.json")   # the launchers' DEPLOYABLE
BASE_EXCLUDES = (".cache",)
FRAMEWORK_PACKAGE = "diffusers"
REFERENCE_LINE_RE = re.compile(r"^# flow_grpo algorithm reference yifan123/flow_grpo [0-9a-f]{40} \(ported\)$")
FREEZE_FILE = "env-freeze.txt"
SAMPLES = "samples.jsonl"
PLAN = "epoch_plan.jsonl"
STEPS = "train_steps.jsonl"
FILTER_REASON = "zero_advantage"                              # 05 section 12.3; train_flow_grpo.py :502
# The launchers' cut rule (save_cut); an adapter adds its `library`, `sources` and `verified` (MM-20).
CUT_COMPLETENESS = {
    "rule": ("a checkpoint-<E> directory is a cut iff its cut.json, written last after an atomic rename from "
             ".checkpoint-<E>.partial, parses with epoch = E, non-negative integers global_step, samples_consumed and "
             "samples_filtered (an epoch whose samples are all filtered trains no step), a boolean ema, a deployable "
             "list of the LoRA weights, adapter_config.json, sampler.json and base_ref.json, and files listing exactly "
             "the other files of the directory (those four and trainer_state.json), each with its SHA-256; the LoRA "
             "safetensors file is whole; .partial directories are never cuts; the cut's last step included is its "
             "global_step"),
    "complete_dir_pattern": CHECKPOINT_DIR_RE.pattern,
}


def _str_list(x: Any) -> bool:
    return isinstance(x, list) and all(isinstance(i, str) for i in x)


def target_text(prompt: str) -> str | None:
    """train_flow_grpo.py `target_text` (flow_grpo ocr.py:32): the text between the first pair of double quotes."""
    parts = prompt.split('"')
    return parts[1] if len(parts) > 1 else None


class FlowGrpoAdapter(DiffusersLoraAdapter):
    """Family base (from flow_grpo_sd35.FlowGrpoSd35Adapter): the members several recipes' adapters share. A
    subclass sets the layout constants these read through `self` and implements the hooks they call."""

    # Layout constants, read through self so a family base class reads the subclass's values.
    SAMPLES = SAMPLES
    PLAN = PLAN
    RUNTIME_FILE = RUNTIME_FILE
    BASE_EXCLUDES = BASE_EXCLUDES

    COMPONENTS = COMPONENTS

    CUT_FILES = CUT_FILES

    CUT_MARKER = CUT_MARKER

    FILTER_REASON = FILTER_REASON

    FRAMEWORK_PACKAGE = FRAMEWORK_PACKAGE

    FREEZE_FILE = FREEZE_FILE

    LORA_WEIGHTS = LORA_WEIGHTS

    MAX_SEED = MAX_SEED

    REFERENCE_LINE_RE = REFERENCE_LINE_RE

    STEPS = STEPS

    framework_name = FRAMEWORK_PACKAGE

    # collect: the base manifest leaves these top-level names out (decision "Area 5 decisions").
    base_excludes = BASE_EXCLUDES

    consumption_fix = ("collect the run's own samples.jsonl, epoch_plan.jsonl and train_steps.jsonl as the run wrote "
                       "them (MM-24, MM-27, MM-28)")

    payload_source = "a training prompt's (the training shard or samples.jsonl; exact overlap, MM-11, EXT-168)"

    @classmethod
    def base_dir_problem(cls, d: Path) -> str | None:
        """None when `d` is the subclass's base directory: its `base_required` configs and whole safetensors weights in
        each of its `base_weight_dirs` (SD3.5 Medium: the transformer, the VAE and the three text encoders); the message
        names its `base_kind`."""
        for rel in cls.base_required:
            if not _plain_file(Path(d) / rel):
                return f"{rel} is missing (not {cls.base_kind})"
        for sub in cls.base_weight_dirs:
            weights = sorted(Path(d, sub).glob("*.safetensors"))
            if not weights:
                return f"{sub}/ has no safetensors weights"
            for w in weights:
                bad = safetensors_problem(w)
                if bad:
                    return f"{sub}/{bad}"
        return None

    def _cut_files_problem_unguarded(self, d: Path) -> str | None:
        doc, why = read_cut_marker(d)
        if doc is None:
            return why
        files = doc["files"]
        if sorted(files) != sorted(self.CUT_FILES):
            return f"{self.CUT_MARKER} lists {sorted(files)}, not {sorted(self.CUT_FILES)}"
        present = sorted(p.name for p in Path(d).iterdir())
        if present != sorted([*self.CUT_FILES, self.CUT_MARKER]):
            extra = sorted(set(present) - set(self.CUT_FILES) - {self.CUT_MARKER})
            return f"the directory holds {extra or present}, which {self.CUT_MARKER} does not list"
        for name in self.CUT_FILES:
            if not _plain_file(Path(d) / name):
                return f"{name} is not a plain file"
            if file_sha256(Path(d) / name) != files[name]:
                return f"{name} is not the SHA-256 {self.CUT_MARKER} records"
        if not (isinstance(doc.get("deployable"), list) and all(isinstance(x, str) for x in doc["deployable"])
                and sorted(doc["deployable"]) == sorted(self.COMPONENTS)):
            return f"{self.CUT_MARKER} deployable is {doc.get('deployable')!r}, not {list(self.COMPONENTS)}"
        if not (_int(doc.get("epoch")) and doc["epoch"] >= 1):
            return f"{self.CUT_MARKER} epoch is {doc.get('epoch')!r}, not a positive integer"
        for key in ("global_step", "samples_consumed", "samples_filtered"):
            if not (_int(doc.get(key)) and doc[key] >= 0):
                return f"{self.CUT_MARKER} {key} is {doc.get(key)!r}, not a non-negative integer"
        if not isinstance(doc.get("ema"), bool):
            return f"{self.CUT_MARKER} ema is {doc.get('ema')!r}, not a boolean"
        return safetensors_problem(Path(d) / self.LORA_WEIGHTS)

    def _base_entries(self, base: Path) -> list[dict]:
        """The base directory's manifest entries without its `.cache/`, walked once per adapter instance (27.4 GB on
        the real base)."""
        key = str(resolve(base))
        if key not in self._base_cache:
            self._base_cache[key] = manifest_entries(Path(base), exclude=self.BASE_EXCLUDES)
        return self._base_cache[key]

    def identity(self, cut: Path, base: Any = None, *, entries: list[dict] | None = None) -> dict[str, Any]:
        """EXT-162: as area 3 (MM-17, MM-18), the base's manifest without its `.cache/`. Entries passed for the base
        directory are filtered the same way."""
        if entries is not None and not self.is_cut(cut):
            entries = [e for e in entries if e["path"].split("/", 1)[0] not in self.BASE_EXCLUDES]
        return super().identity(cut, base, entries=entries)

    def _ledger_plan(self, stage_out: Path, data: Path, cuts: list) -> tuple[list[tuple[dict, int | None]], list[str]]:
        """([(samples.jsonl line, optimizer step or None)] for the samples of the epochs up to the last cut, in file
        order: one pair per step that applied a kept sample, one pair with None for a filtered sample) and the
        problems that refuse the logs (empty when they are consistent). Every field train_flow_grpo.py derives is
        re-derived (README item 6)."""
        args, why = read_run_args(stage_out)
        if args is None:
            return [], [why]
        if not cuts:
            return [], ["no complete cut"]
        s = args.get("settings") if isinstance(args.get("settings"), dict) else {}
        spe, group, inner_n = s.get("samples_per_epoch"), s.get("images_per_prompt"), s.get("num_inner_epochs")
        nb, gas, run_seed = s.get("num_batches_per_epoch"), s.get("gradient_accumulation_steps"), args.get("seed")
        if not all(_int(x) and x > 0 for x in (spe, group, inner_n, nb, gas)) or not _int(run_seed):
            return [], ["run_args.json has no positive integer settings.samples_per_epoch, images_per_prompt, "
                        "num_inner_epochs, num_batches_per_epoch, gradient_accumulation_steps and integer seed"]
        if spe % group or spe % nb or nb % gas:
            return [], [f"run_args.json settings: samples_per_epoch {spe} is not a multiple of images_per_prompt "
                        f"{group} and num_batches_per_epoch {nb}, or num_batches_per_epoch is not a multiple of "
                        f"gradient_accumulation_steps {gas} (train_flow_grpo.py resolve_settings)"]
        why = self._settings_problems(s)
        if why:
            return [], why
        per_inner = nb // gas                             # optimizer steps per inner epoch (train_flow_grpo.py :561)
        # Reproduce the zero-advantage rule for the runtime's tracker (PerPromptStatTracker
        # `grpo`, :445-460: advantage = (reward - group mean) / (std + 1e-4), std per group or global by
        # `global_std`, so it is 0 iff the reward equals its group's mean). Any other advantage setting refuses.
        if not isinstance(s.get("global_std"), bool):
            return [], [f"run_args.json settings.global_std is {s.get('global_std')!r}, not a boolean: the zero-advantage "
                        "filter cannot be re-derived for another advantage mode (train_flow_grpo.py PerPromptStatTracker)"]
        ocr_reward = s.get("reward") == "ocr"
        shard = _shard(data)
        if shard is None:
            return [], [f"the spec's training prompts {data} hold {len(list(Path(data).glob('*.jsonl')))} shards, not "
                        "the one shard the run read"]
        rows, why = read_jsonl(Path(data) / shard)
        if rows is None:
            return [], [f"the training shard: {why}"]
        prompt_of = {r.get("prompt_id"): r.get("prompt") for r in rows}
        marks = []
        for _, d in cuts:
            doc, why = read_cut_marker(d)
            if doc is None:
                return [], [f"{Path(d).name}: {why}"]
            marks.append((doc.get("epoch"), doc.get("global_step"), doc.get("samples_consumed"),
                          doc.get("samples_filtered"), Path(d).name))
        last_epoch = max(m[0] for m in marks)
        logs = {}
        for name in (self.SAMPLES, self.PLAN, self.STEPS):
            logs[name], why = read_jsonl(Path(stage_out) / name)
            if logs[name] is None:
                return [], [why]
        samples, plans, steps = logs[self.SAMPLES], logs[self.PLAN], logs[self.STEPS]
        out: list[str] = []

        def bad(msg: str) -> bool:
            out.append(msg)
            return len(out) >= 8

        # samples.jsonl lines (train_flow_grpo.py:441-479)
        by_epoch: dict[int, list[dict]] = {}
        seen_ids: set[str] = set()
        for n, x in enumerate(samples, start=1):
            e, i, sid, pid, seed, member = (x.get(k) for k in ("epoch", "sample_index", "sample_id", "prompt_id",
                                                                "seed", "member"))
            if not (_int(e) and e >= 1 and _int(i) and i >= 0 and isinstance(sid, str) and isinstance(pid, str)
                    and _int(member) and member >= 0 and isinstance(x.get("prompt"), str)
                    and isinstance(x.get("target"), str)):
                if bad(f"{self.SAMPLES} line {n} has no integer epoch, sample_index, member and string sample_id, "
                       "prompt_id, prompt, target"):
                    return [], out
                continue
            if not (_int(seed) and 0 <= seed < self.MAX_SEED):
                if bad(f"{self.SAMPLES} line {n}: seed {seed!r} is not a 63-bit integer noise seed"):
                    return [], out
                continue
            reward, acc = x.get("reward"), x.get("ocr_acc")
            if not (_num(reward) and _num(acc) and 0 <= acc <= 1):
                if bad(f"{self.SAMPLES} line {n}: reward {reward!r} is not a finite number or ocr_acc {acc!r} is not in "
                       "[0, 1]"):
                    return [], out
                continue
            # The fields train_flow_grpo.py derives (:447, :459-461, :476) are re-derived (MM-29).
            if seed != derive_seed("sample", run_seed, e, i):
                bad(f"{self.SAMPLES} line {n}: seed {seed} is not derive_seed('sample', {run_seed}, {e}, {i})")
            if member != i % group:
                bad(f"{self.SAMPLES} line {n}: member {member} is not sample_index {i} % images_per_prompt {group}")
            if x["target"] != target_text(x["prompt"]):
                bad(f"{self.SAMPLES} line {n}: target {x['target']!r} is not the prompt's first quoted text (flow_grpo "
                    "ocr.py:32)")
            for msg in self._reward_field_problems(x, n, s, ocr_reward):
                bad(msg)
            want_sid = compact_sha256([self.sample_id_domain, run_seed, x.get("attempt"), e, i, pid, seed, member])
            if sid != want_sid:
                bad(f"{self.SAMPLES} line {n}: sample_id is not the hash train_flow_grpo.py derives ({want_sid[:19]}...)")
            if len(out) >= 8:
                return [], out
            if x.get("attempt") != args.get("attempt"):
                if bad(f"{self.SAMPLES} line {n}: attempt {x.get('attempt')!r} is not run_args.json attempt "
                       f"{args.get('attempt')!r}"):
                    return [], out
            if pid not in prompt_of:
                if bad(f"{self.SAMPLES} line {n}: prompt_id {pid} is not a row of the training shard {shard}"):
                    return [], out
            elif x["prompt"] != prompt_of[pid]:
                if bad(f"{self.SAMPLES} line {n}: prompt {x['prompt']!r} is not the shard's prompt of {pid}"):
                    return [], out
            if sid in seen_ids:
                if bad(f"{self.SAMPLES} line {n}: sample_id repeats"):
                    return [], out
            seen_ids.add(sid)
            by_epoch.setdefault(e, []).append(x)
        if out:
            return [], out
        epochs = s.get("epochs")
        for e in range(1, last_epoch + 1):
            got = sorted(by_epoch.get(e, []), key=lambda x: x["sample_index"])
            if [x["sample_index"] for x in got] != list(range(spe)):
                if bad(f"{self.SAMPLES} epoch {e} logs {len(got)} samples (sample_index "
                       f"{[x['sample_index'] for x in got][:6]}...), not 0 .. {spe - 1} (samples_per_epoch)"):
                    return [], out
                continue
            for msg in self._epoch_prompt_problems(e, got, spe, group, s, rows):
                bad(msg)
            if len(out) >= 8:
                return [], out
        if out:
            return [], out
        index_of = {x["sample_id"]: (e, x["sample_index"]) for e, xs in by_epoch.items() for x in xs}
        sample_of = {x["sample_id"]: x for xs in by_epoch.values() for x in xs}

        # epoch_plan.jsonl lines (train_flow_grpo.py:480-503)
        plan_of: dict[int, dict] = {}
        for n, p in enumerate(plans, start=1):
            e = p.get("epoch")
            if not (_int(e) and e >= 1 and all(_str_list(p.get(k)) for k in ("kept", "filtered", "reincluded"))
                    and isinstance(p.get("steps"), list)):
                if bad(f"{self.PLAN} line {n} has no integer epoch and lists kept, filtered, reincluded, steps"):
                    return [], out
                continue
            if e in plan_of:
                if bad(f"{self.PLAN} line {n}: epoch {e} is planned twice"):
                    return [], out
                continue
            plan_of[e] = p
        if out:
            return [], out
        if [p["epoch"] for p in plans] != sorted(plan_of):
            return [], [f"{self.PLAN} epochs {[p['epoch'] for p in plans][:10]} are not in order"]
        kept_of: dict[int, list[str]] = {}
        filtered_of: dict[int, list[str]] = {}
        for e in range(1, last_epoch + 1):
            p = plan_of.get(e)
            if p is None:
                if bad(f"{self.PLAN} has no line for epoch {e} (each epoch's kept and filtered samples are planned before "
                       "its first optimizer step)"):
                    return [], out
                continue
            kept, filtered, reinc = p["kept"], p["filtered"], p["reincluded"]
            if p.get("filter_reason") != self.FILTER_REASON:
                bad(f"{self.PLAN} epoch {e}: filter_reason {p.get('filter_reason')!r} is not {self.FILTER_REASON!r} (the "
                    "runtime's only filter, 05 section 12.3)")
            for key, ids in (("kept", kept), ("filtered", filtered), ("reincluded", reinc)):
                if len(set(ids)) != len(ids):
                    bad(f"{self.PLAN} epoch {e}: a sample is {key} twice")
                strangers = [sid for sid in ids if index_of.get(sid, (None,))[0] != e]
                if strangers:
                    bad(f"{self.PLAN} epoch {e}: {key} names {strangers[0][:19]}..., which is not a sample of epoch {e} in "
                        f"{self.SAMPLES}")
            both = set(kept) & set(filtered)
            if both:
                bad(f"{self.PLAN} epoch {e}: {len(both)} sample(s) are both kept and filtered")
            missing = {x["sample_id"] for x in by_epoch.get(e, [])} - set(kept) - set(filtered)
            if missing:
                bad(f"{self.PLAN} epoch {e}: {len(missing)} sample(s) are neither kept nor filtered")
            for key, ids in (("kept", kept), ("filtered", filtered)):
                idx = [index_of[sid][1] for sid in ids if sid in index_of]
                if idx != sorted(idx):
                    bad(f"{self.PLAN} epoch {e}: {key} is not in sample order (train_flow_grpo.py :494, :498)")
            if len(kept) % nb:
                bad(f"{self.PLAN} epoch {e}: {len(kept)} kept samples are not a multiple of num_batches_per_epoch {nb} "
                    "(the padding rule, train_flow_grpo.py :486-497)")
            if not set(reinc) <= set(kept) or len(reinc) >= nb:
                bad(f"{self.PLAN} epoch {e}: reincluded ({len(reinc)}) is not a subset of kept smaller than "
                    f"num_batches_per_epoch {nb} (the padding re-includes at most {nb - 1} zero-advantage samples)")
            xs = sorted(by_epoch.get(e, []), key=lambda x: x["sample_index"])
            for msg in self._flat_group_problems(e, p, xs, group):
                bad(msg)
            # Only zero-advantage samples are filtered or re-included (train_flow_grpo.py :483-493): a
            # sample there has its group's mean reward, the mean computed exactly (numpy's float mean differs from it by
            # a few ulps at most, hence the tolerance).
            means = {}
            for g0 in range(0, len(xs), group):
                members = xs[g0:g0 + group]
                means[g0 // group] = sum((Fraction(m["reward"]) for m in members), Fraction(0)) / len(members)
            for key, ids in (("filtered", filtered), ("reincluded", reinc)):
                off = [sid for sid in ids if sid in sample_of
                       and abs(Fraction(sample_of[sid]["reward"]) - means[sample_of[sid]["sample_index"] // group])
                       > Fraction(1e-12) * max(1, abs(means[sample_of[sid]["sample_index"] // group]))]
                if off:
                    x = sample_of[off[0]]
                    bad(f"{self.PLAN} epoch {e}: {len(off)} {key} sample(s) have a reward other than their group mean (a "
                        f"nonzero advantage), e.g. sample_index {x['sample_index']} reward {x['reward']!r} against "
                        f"group mean {float(means[x['sample_index'] // group])!r}; only zero-advantage samples are "
                        "filtered or re-included (train_flow_grpo.py :483-493)")
            kept_of[e], filtered_of[e] = kept, filtered
            if len(out) >= 8:
                return [], out
        beyond_plan = sorted(e for e in plan_of if not (_int(epochs) and e <= epochs))
        if out:
            return [], out

        # train_steps.jsonl lines (train_flow_grpo.py:512-564)
        nums = [st.get("step") for st in steps]
        if not all(_int(k) for k in nums) or nums != list(range(1, len(steps) + 1)):     # bools compare equal to ints
            return [], [f"{self.STEPS} steps {nums[:10]}{' ...' if len(nums) > 10 else ''} are not consecutive from 1"]
        applied: dict[str, list[int]] = {}
        per_epoch: dict[int, list[dict]] = {}
        for st in steps:
            e, inner, ids = st.get("epoch"), st.get("inner"), st.get("sample_ids")
            if not (_int(e) and _int(inner) and 0 <= inner < inner_n and _str_list(ids)):
                return [], out + [f"{self.STEPS} step {st['step']} has no integer epoch, inner < num_inner_epochs and list of "
                                  "sample_ids"]
            per_epoch.setdefault(e, []).append(st)
            if e > last_epoch:
                continue
            for sid in ids:
                if sid not in index_of:
                    if bad(f"{self.STEPS} step {st['step']} (epoch {e}) applies {sid[:19]}..., which is not in {self.SAMPLES}"):
                        return [], out
                elif index_of[sid][0] != e:
                    if bad(f"{self.STEPS} step {st['step']} (epoch {e}) applies {sid[:19]}..., a sample of epoch "
                           f"{index_of[sid][0]}"):
                        return [], out
                elif sid in filtered_of.get(e, ()):
                    if bad(f"{self.STEPS} step {st['step']} (epoch {e}) applies {sid[:19]}..., a filtered sample "
                           f"({self.PLAN})"):
                        return [], out
                else:
                    applied.setdefault(sid, []).append(st["step"])
        for e in range(1, last_epoch + 1):
            sts = per_epoch.get(e, [])
            kept = kept_of.get(e, [])
            if not kept:
                if sts:
                    bad(f"{self.STEPS} epoch {e} has {len(sts)} optimizer steps, but {self.PLAN} keeps no samples of it (an "
                        "epoch with no kept samples has no kept samples to train on)")
                if plan_of[e]["steps"] != []:                  # train_flow_grpo.py :499, :503
                    bad(f"{self.PLAN} epoch {e} names steps {plan_of[e]['steps']!r}, not the epoch's steps [] (no kept "
                        "samples)")
                continue
            if len(sts) != per_inner * inner_n:
                if bad(f"{self.STEPS} epoch {e} has {len(sts)} optimizer steps, not num_batches_per_epoch / "
                       f"gradient_accumulation_steps x num_inner_epochs = {per_inner * inner_n}"):
                    return [], out
            size = len(kept) // per_inner
            for st in sts:
                if len(st["sample_ids"]) != size:
                    bad(f"{self.STEPS} step {st['step']} applies {len(st['sample_ids'])} sample ids, not {size} (the epoch's "
                        f"{len(kept)} kept samples over {per_inner} steps)")
            for inner in range(inner_n):
                ids = [sid for st in sts if st["inner"] == inner for sid in st["sample_ids"]]
                if sorted(ids) != sorted(kept):
                    if bad(f"{self.STEPS} epoch {e} inner {inner}: the steps apply {len(ids)} sample ids, not each of the "
                           f"epoch's {len(kept)} kept samples exactly once (a sample in no step or in two)"):
                        return [], out
            want = [min(st["step"] for st in sts), max(st["step"] for st in sts)] if sts else []
            if plan_of[e]["steps"] != want:
                bad(f"{self.PLAN} epoch {e} names steps {plan_of[e]['steps']!r}, not the epoch's steps {want}")
        # EXT-158 analogue: train_flow_grpo.py stops after settings.epochs.
        beyond = sorted({e for e in per_epoch if not (_int(epochs) and e <= epochs)}
                        | {e for e in by_epoch if not (_int(epochs) and e <= epochs)} | set(beyond_plan))
        if beyond:
            bad(f"{self.STEPS}, {self.PLAN} or {self.SAMPLES} log epochs {beyond[:6]} beyond run_args.json settings.epochs {epochs!r}")
        if out:
            return [], out
        order = [st.get("epoch") for st in steps]
        if order != sorted(order):
            return [], [f"{self.STEPS} epochs are not in order"]
        # MM-27: each cut's counts equal the logs up to its epoch (train_flow_grpo.py :577-578: samples_consumed adds
        # the epoch's kept count once, whatever num_inner_epochs is).
        for epoch, gs, consumed, filtered, name in marks:
            want_gs = max((st["step"] for e, sts in per_epoch.items() if e <= epoch for st in sts), default=0)
            if gs != want_gs:
                bad(f"{name} cut.json global_step is {gs!r}, but {self.STEPS} logs {want_gs} steps up to epoch {epoch}")
            want_c = sum(len(kept_of.get(e, [])) for e in range(1, epoch + 1))
            if consumed != want_c:
                bad(f"{name} cut.json samples_consumed is {consumed!r}, but {self.PLAN} keeps {want_c} samples up to epoch "
                    f"{epoch}")
            want_f = sum(len(filtered_of.get(e, [])) for e in range(1, epoch + 1))
            if filtered != want_f:
                bad(f"{name} cut.json samples_filtered is {filtered!r}, but {self.PLAN} filters {want_f} samples up to epoch "
                    f"{epoch}")
        if out:
            return [], out
        plan: list[tuple[dict, int | None]] = []
        for x in samples:
            if x["epoch"] > last_epoch:
                continue
            if x["sample_id"] in filtered_of.get(x["epoch"], ()):
                plan.append((x, None))
            else:
                plan += [(x, step) for step in applied.get(x["sample_id"], [])]
        return plan, []

    def _flat_group_problems(self, e: int, plan: dict, xs: list[dict], group: int) -> list[str]:
        """A prompt group whose OCR rewards are all equal has advantage 0 for every member (the group mean of G equal
        floats is that float for G = 8, numpy's pairwise sum, and for G = 2, 4), so its members are filtered or
        re-included by the padding, never plainly kept (train_flow_grpo.py :481-498). `xs` is the epoch's samples in
        sample order."""
        kept, reinc = plan["kept"], plan["reincluded"]
        out = []
        for g0 in range(0, len(xs), group):
            members = xs[g0:g0 + group]
            if len({m["reward"] for m in members}) == 1:
                plain = [m for m in members if m["sample_id"] in kept and m["sample_id"] not in reinc]
                if plain:
                    out.append(f"{self.PLAN} epoch {e}: prompt group {g0 // group} has equal rewards (zero advantage), but "
                               f"{len(plain)} of its samples are kept without being reincluded")
        return out

    def consumption(self, stage_out: Path, data: Path, cuts: list, *, run_id: str = "",
                    attempt_id: str = "0") -> dict[str, Any]:
        """Precision `row`, derivation `observed` (MM-24, MM-28): per sample of the epochs up to the last cut, one
        `consumed` record per optimizer step that applied it, or one `filtered` record (reason `zero_advantage`, step
        null) when the epoch plan filtered it. `data_ref.row` = sha256(prompt_id), shard = the spec's one prompt shard,
        `seed_inputs.seed` = the noise seed, `rewards` = [its OCR reward], `sampler_config_hash` = the RFC 8785 hash of
        run_args.json `sampler`, `sample_id` = H(run, attempt, epoch, step, prompt_id, seed, member) (NUL-joined text;
        a filtered record has `filtered` in the step's place)."""
        from ..ledger import WRITER_ID, check_precision

        plan, bad = self._ledger_plan(stage_out, data, cuts)
        if bad:                                          # collect refuses these before writing
            raise ValueError("; ".join(bad))
        check_precision("row", "observed")
        args, _ = read_run_args(stage_out)
        sampler = args.get("sampler")
        sampler_hash = sha256_bytes(canonical_json(sampler)) if isinstance(sampler, dict) else None
        shard = _shard(data)
        snapshot = manifest_hash(data)
        rows = []
        for x, step in plan:
            key = "\x00".join(str(v) for v in (run_id, attempt_id, x["epoch"], "filtered" if step is None else step,
                                               x["prompt_id"], x["seed"], x["member"]))
            rows.append({
                "run_id": run_id, "attempt_id": attempt_id, "step": step,
                "sample_id": "sha256:" + hashlib.sha256(key.encode("utf-8")).hexdigest(),
                "data_ref": {"snapshot": snapshot, "shard": shard,
                             "row": sha256_bytes(x["prompt_id"].encode("utf-8")), "mixture": None, "component": None},
                "parent_ref": None, "outcome": "filtered" if step is None else "consumed",
                "reason": self.FILTER_REASON if step is None else None, "rule_version": None, "gen_version": None,
                "reward_epoch": None, "rewards": [float(x["reward"])], "sampler_config_hash": sampler_hash,
                "seed_inputs": {"seed": x["seed"]}, "chunk": None, "precision": "row", "derivation": "observed",
                "writer_id": WRITER_ID,
            })
        last = max(step for step, _ in cuts)
        n_filtered = sum(1 for r in rows if r["outcome"] == "filtered")
        return {"precision": "row", "rows": rows, "derivation": "observed",
                "reason": (f"{self.SAMPLES} logs every generated sample, {self.PLAN} the epoch's kept and filtered samples before "
                           f"its first optimizer step, and {self.STEPS} the samples each optimizer step applied, before the "
                           f"step (MM-24, MM-27, MM-28): {len(rows) - n_filtered} consumed records of steps 1 .. {last} "
                           f"and {n_filtered} filtered records (outcome filtered, reason {self.FILTER_REASON}, no step), "
                           f"data_ref.row = sha256(prompt_id) in shard {shard}, derivation observed")}

    @staticmethod
    def item_overlap_problems(item_rows: list[dict], data: Path) -> list[str]:
        """Check for shared OCR targets using the trainer and scorer's target extraction.

        The target is the prompt's first quoted text (`target_text`, flow_grpo ocr.py:32). A
        row or item whose stated `target` differs, or whose prompt has no quoted text, refuses. No item's derived target
        equals a training row's after flow_grpo's normalization (spaces removed, lower case; ocr.py:46-49). Problems
        (each refuses) or empty; OSError propagates to collect."""
        shard = _shard(data)
        if shard is None:
            return []                                     # the data binding reports it (EXT-164)
        rows, why = read_jsonl(Path(data) / shard)
        if rows is None:
            return [f"the training shard: {why}"]
        out: list[str] = []

        def derived(row: dict, where: str) -> str | None:
            tgt = target_text(row["prompt"]) if isinstance(row.get("prompt"), str) else None
            if tgt is None:
                out.append(f"{where} has no prompt with a quoted target text (flow_grpo ocr.py:32)")
            elif "target" in row and row["target"] != tgt:
                out.append(f"{where} states target {row['target']!r}, but its prompt's first quoted text is {tgt!r} "
                           "(the text the trainer rewards and the evaluator scores; EXT-176)")
            return tgt

        def norm(text: str) -> str:
            return text.replace(" ", "").lower()

        train = set()
        for n, r in enumerate(rows, start=1):
            tgt = derived(r, f"training shard {shard} line {n} ({r.get('prompt_id')!r})")
            if tgt is not None and norm(tgt):
                train.add(norm(tgt))
        hits = []
        for r in item_rows:
            tgt = derived(r, f"item {r.get('item')!r}")
            if tgt is not None and norm(tgt) in train:
                hits.append(r["item"])
        out = out[:8] + ([f"{len(out) - 8} more target problems"] if len(out) > 8 else [])
        if hits:
            out.append(f"{len(hits)} item(s) have a target text equal to a training prompt's after normalization "
                       f"(spaces removed, lower case): {', '.join(map(str, hits[:5]))}{' ...' if len(hits) > 5 else ''} "
                       "(README item 7, EXT-176)")
        return out

    def framework_identity(self, stage_out: Path, launch: dict) -> dict[str, Any]:
        """`diffusers==<v>` in env-freeze.txt equals the launch record's framework; the freeze pins peft, torch,
        transformers and the reward engine (paddleocr, paddlepaddle) and names the algorithm reference commit (MM-33).
        Raises ValueError with the reason."""
        freeze = Path(stage_out) / self.FREEZE_FILE
        if not _plain_file(freeze):
            raise ValueError(f"{freeze} is missing")
        lines = [l.strip() for l in text_lines(freeze.read_text(encoding="utf-8"))]
        versions: dict[str, str] = {}
        for line in lines:
            if "==" in line and not line.startswith("#"):
                name, _, version = line.partition("==")
                versions[name.strip().lower().replace("_", "-")] = version.strip()
        missing = [p for p in self.pins if p not in versions]
        if missing:
            raise ValueError(f"{freeze} has no `<name>==<version>` line for {', '.join(missing)} (pins of {self.name})")
        if not any(self.REFERENCE_LINE_RE.fullmatch(l) for l in lines):
            raise ValueError(f"{freeze} has no `# flow_grpo algorithm reference yifan123/flow_grpo <40 hex> (ported)` "
                             "line (the algorithm reference the port names)")
        version = versions[self.FRAMEWORK_PACKAGE]
        fw = launch.get("framework") if isinstance(launch.get("framework"), dict) else {}
        if fw.get("name") != self.FRAMEWORK_PACKAGE or fw.get("version") != version:
            raise ValueError(f"{freeze} pins {self.FRAMEWORK_PACKAGE}=={version}, but the launch record claims "
                             f"framework {fw.get('name')!r} {fw.get('version')!r}")
        return {"name": self.FRAMEWORK_PACKAGE, "version": version, "freeze_sha256": file_sha256(freeze)}

    def checks(self, *, last_cut: Path, base: Path, launch: dict, base_field: str, stage_out: Path | None = None,
               data: Path | None = None, data_field: str = "data", **kw: Any) -> list[Problem]:
        """Area 3's EXT-121, EXT-122 and EXT-125 checks (EXT-163, EXT-164, EXT-169), plus: the shard's row count,
        run_args.json `script_sha256` and `sampler_sha256`, the sampler code equal to the evaluator's (pinned), EMA off
        (decision) and the OCR reward."""
        out = super().checks(last_cut=last_cut, base=base, launch=launch, base_field=base_field, stage_out=stage_out,
                             data=data, data_field=data_field, **kw)
        stage_out = Path(stage_out) if stage_out is not None else Path(last_cut).parent
        args, _ = read_run_args(stage_out)
        if args is None:
            return out
        s = args.get("settings") if isinstance(args.get("settings"), dict) else {}
        if data is not None and _shard(data) is not None:
            rows, _ = read_jsonl(Path(data) / _shard(data))
            if rows is not None and args.get("n_train_prompts") != len(rows):
                out.append(Problem("EXT-DATA", data_field, f"run_args.json n_train_prompts "
                                   f"{args.get('n_train_prompts')!r} is not the {len(rows)} rows of "
                                   f"{Path(data) / _shard(data)} (EXT-164)", self.DATA_FIX))
        if args.get("script_sha256") != launch.get("script_sha256"):
            out.append(Problem("EXT-LAUNCH", "--launch-record.script_sha256", f"run_args.json script_sha256 "
                               f"{args.get('script_sha256')!r} is not the launch record's "
                               f"{launch.get('script_sha256')!r} (EXT-83)",
                               "collect the run launched with this record's training script"))
        sampler = args.get("sampler")
        if not (isinstance(sampler, dict) and args.get("sampler_sha256") == compact_sha256(sampler)):
            out.append(Problem("EXT-SAMPLER", "--stage-out", f"run_args.json sampler_sha256 "
                               f"{args.get('sampler_sha256')!r} is not the hash of its sampler (compact sorted JSON)",
                               "collect the run as train_flow_grpo.py wrote it"))
        out += self._reward_setting_problems(s)
        if s.get("ema") is not False:
            out.append(Problem("EXT-STAGE", "--stage-out", f"run_args.json settings.ema is {s.get('ema')!r}: EMA is off "
                               "in this area (decision \"Area 5 decisions\": a cut holds the raw LoRA weights)",
                               "train without --ema"))
        rt, _ = _read_json(stage_out / self.RUNTIME_FILE)
        code = rt.get("sampler_code_sha256") if isinstance(rt, dict) else None
        cuts = [d for _, d in self.find_cuts(stage_out)[0]] or [Path(last_cut)]
        for d in cuts:
            doc, _ = read_cut_marker(d)
            doc = doc or {}
            if doc.get("ema") is not False:
                out.append(Problem("EXT-STAGE", "--stage-out", f"{Path(d).name}/cut.json ema is {doc.get('ema')!r}: "
                                   "EMA is off in this area (decision \"Area 5 decisions\")", "train without --ema"))
            cfg, _ = _read_json(Path(d) / "adapter_config.json")
            if not (isinstance(cfg, dict) and cfg.get("weights") == "raw"):
                out.append(Problem("EXT-STAGE", "--stage-out", f"{Path(d).name}/adapter_config.json weights is "
                                   f"{(cfg or {}).get('weights') if isinstance(cfg, dict) else None!r}, not 'raw': EMA "
                                   "is off in this area (decision \"Area 5 decisions\")", "train without --ema"))
            sc = doc.get("sampler_code_sha256")
            pinned = self.sampler_code_sha256
            if isinstance(code, str) and sc == code == args.get("sampler_code_sha256") and sc != pinned:
                out.append(Problem("EXT-SAMPLER", "--stage-out", f"{Path(d).name}/cut.json sampler_code_sha256 {sc} is "
                                   f"not the evaluator's sampler code {pinned} ({self.sampler_code_source})",
                                   "train and evaluate with the same SHARED SAMPLER block"))
        return out
