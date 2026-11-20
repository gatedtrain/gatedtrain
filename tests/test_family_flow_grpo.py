"""The Flow-GRPO family base (`gatedtrain.families.flow_grpo.FlowGrpoAdapter`) and the diffusers LoRA base it extends
(`gatedtrain.families.diffusers_lora`), on a synthetic run directory (EXT-160 to EXT-178, EXT-228; MM-17, MM-18,
MM-20, MM-24, MM-27, MM-28, MM-33).

Two recipes (SD3.5 OCR, Wan video) subclass the family and install gatedtrain from PyPI, so a change to the family on
main is tested by no recipe CI until it is released. `MiniFlowGrpo` is the smallest subclass: the hooks of the SD3.5
adapter (reward = OCR accuracy, one prompt per group) on a tiny base (`model_index.json`, `transformer/`). `build_run`
writes what train_flow_grpo.py writes, by the rules the family re-derives, independently of gatedtrain:

- 2 prompts x 2 images per epoch (`samples_per_epoch` 4), 2 virtual batches, 1 per optimizer step, 1 inner epoch;
- epoch 1: group 0 rewards (1, 0), group 1 (0.5, 0.5): the flat group has zero advantage and is filtered; epoch 2:
  both groups vary, all four samples are kept;
- `samples.jsonl` (noise seed derive_seed('sample', seed, e, i), member i % G, sample_id the compact hash of
  [domain, seed, attempt, e, i, prompt_id, noise, member]), `epoch_plan.jsonl`, `train_steps.jsonl`;
- cuts `checkpoint-1/` (global_step 2, consumed 2, filtered 2) and `checkpoint-2/` (4, 6, 2), cut.json last;
- run_args.json, env-runtime.json, base_ref.json and adapter_config.json carry the base and data bindings the family's
  run checks read (base_model, base_ref with verified, train_file and its SHA-256, profile, sampler code; base_ref.json
  files and files_sha256), so the simulated run passes `checks` with no problem.

`TestReincludedAndEmptyEpochs` runs the same simulator on 2 prompts x 4 images, where the padding re-includes a
zero-advantage sample and one epoch keeps nothing.

What the simulator leaves out of the real outputs (recipe gatedtrain-recipe-flow-grpo-sd35-ocr,
runtime/train_flow_grpo.py), and why that does not weaken these tests:

- cut.json `schema`, `schema_version`, `tags`, `created_at` (:624-628): no family check reads them (`read_cut_marker`
  needs only `files`; `cut_problem` and `checks` read epoch, counts, files, deployable, ema, seed, profile, script and
  sampler code hashes);
- trainer_state.json `epoch`, the counts and `note`: a listed file hashed in cut.json `files`, its fields unread;
- samples.jsonl `ocr_text`, `ocr_error`, `utc` (:829-834): the family leaves the reward fields to the area's
  `_reward_field_problems` hook (the recipe's SD3.5 adapter checks `ocr_error`/`ocr_text` there; this subclass
  checks reward = ocr_acc only), and `utc` is unread; epoch_plan.jsonl `utc` and run_args.json's descriptive fields (reward engine, device, memory) are
  unread too;
- sample_id: the recipe hashes `canonical(...)` (:149: compact sorted JSON, ensure_ascii False), the very encoding of
  `compact` here, under its own domain `flow-grpo-sd35` where this subclass declares `mini-flow-grpo`;
- the padding's choice: the runtime re-includes zero-advantage samples drawn by a seeded randperm (:846); the
  family checks only that they are kept, have advantage 0 and number fewer than num_batches_per_epoch, so the
  simulator takes the last ones;
- the step order: the runtime permutes the kept samples per inner epoch (:870-871); the family checks per inner
  epoch that each kept sample is applied once in steps of equal size, not the order.

Each refusal below is one fault the family must catch before a ledger is written.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import struct
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from _ext import sha_of
from _support import canonical_bytes, file_sha256, manifest_entries

from gatedtrain.errors import Problem
from gatedtrain.families import flow_grpo
from gatedtrain.families.flow_grpo import FlowGrpoAdapter, target_text

DOMAIN = "mini-flow-grpo"
SEED = 7
ATTEMPT = 1
SAMPLER = {"steps": 10, "guidance": 4.5, "resolution": 64}
SETTINGS = {"samples_per_epoch": 4, "images_per_prompt": 2, "num_inner_epochs": 1, "num_batches_per_epoch": 2,
            "gradient_accumulation_steps": 1, "global_std": False, "reward": "ocr", "epochs": 2, "ema": False}
PROMPTS = [{"prompt_id": f"p{i}", "prompt": f'a sign that reads "WORD {i}"'} for i in range(4)]
# epoch -> (prompt ids of its two groups, rewards per group)
EPOCHS = {1: (("p0", "p1"), ((1.0, 0.0), (0.5, 0.5))), 2: (("p2", "p3"), ((0.25, 0.75), (1.0, 0.5)))}
FREEZE = ("diffusers==0.36.0\npeft==0.17.1\ntorch==2.9.0\n"
          "# flow_grpo algorithm reference yifan123/flow_grpo " + "a" * 40 + " (ported)\n")


def derive_seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256(":".join(str(p) for p in parts).encode()).digest()[:8], "big") >> 1


def compact(obj) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                                 ensure_ascii=False).encode()).hexdigest()


def lora_bytes(tag: str) -> bytes:
    header = json.dumps({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
    return struct.pack("<Q", len(header)) + header + hashlib.sha256(tag.encode()).digest()[:8]


def write_jsonl(path: Path, docs: list[dict]) -> None:
    path.write_text("".join(json.dumps(d, sort_keys=True) + "\n" for d in docs), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


class MiniFlowGrpo(FlowGrpoAdapter):
    """The family's hooks as the SD3.5 adapter implements them, on a tiny base."""

    name = "mini_flow_grpo"
    evaluator_ids = ()
    sample_id_domain = DOMAIN
    base_required = ("model_index.json",)
    base_weight_dirs = ("transformer",)
    base_kind = "a mini base"
    pins = ("diffusers", "peft", "torch")
    sampler_code_sha256 = "sha256:" + "5" * 64
    sampler_code_source = "a test pin"
    BASE_FIX = DATA_FIX = "fix it"

    def _settings_problems(self, s: dict) -> list[str]:
        return []

    def _reward_field_problems(self, x: dict, n: int, s: dict, ocr_reward: bool) -> list[str]:
        return [f"{self.SAMPLES} line {n}: reward is not ocr_acc"] if ocr_reward and x["reward"] != x["ocr_acc"] else []

    def _epoch_prompt_problems(self, e, got, spe, group, s, rows) -> list[str]:
        return [f"{self.SAMPLES} epoch {e} group {g0 // group} has two prompts" for g0 in range(0, spe, group)
                if len({x["prompt_id"] for x in got[g0:g0 + group]}) != 1]

    def _reward_setting_problems(self, s: dict) -> list[Problem]:
        return [] if s.get("reward") == "ocr" else [Problem("EXT-STAGE", "--stage-out", "reward is not ocr", "fix")]


def build_base(d: Path) -> Path:
    (d / "transformer").mkdir(parents=True)
    (d / "model_index.json").write_text('{"_class_name": "Mini"}', encoding="utf-8")
    (d / "transformer" / "diffusion.safetensors").write_bytes(lora_bytes("base"))
    (d / ".cache").mkdir()
    (d / ".cache" / "download.lock").write_text("download metadata", encoding="utf-8")
    return d


def simulate(epochs: dict = EPOCHS, group: int = 2, nb: int = 2
             ) -> tuple[list[dict], list[dict], list[dict], dict[int, tuple[int, int, int]]]:
    """The logs of a run with `images_per_prompt` `group` and `num_batches_per_epoch` `nb` (one inner epoch, two
    optimizer steps per epoch with kept samples, as resolve_settings makes them)."""
    prompt_of = {r["prompt_id"]: r["prompt"] for r in PROMPTS}
    samples, plans, steps, state = [], [], [], {}
    gs = consumed = filtered = 0
    for e, (pids, rewards) in epochs.items():
        ids, zero = [], []
        for i in range(len(pids) * group):
            g, member = i // group, i % group
            pid, r, noise = pids[g], rewards[g][member], derive_seed("sample", SEED, e, i)
            sid = compact([DOMAIN, SEED, ATTEMPT, e, i, pid, noise, member])
            ids.append(sid)
            if Fraction(r) == sum(map(Fraction, rewards[g])) / group:     # advantage 0: the group mean (:838)
                zero.append(i)
            samples.append({"attempt": ATTEMPT, "epoch": e, "sample_index": i, "sample_id": sid, "prompt_id": pid,
                            "prompt": prompt_of[pid], "target": target_text(prompt_of[pid]), "seed": noise,
                            "member": member, "reward": r, "ocr_acc": r})
        # The padding (:840-848): a nonzero count that is not a multiple of nb re-includes zero-advantage samples.
        # The runtime picks them with a seeded randperm; the family checks only that they have advantage 0 and are
        # fewer than nb, so the last ones stand in here.
        true_count = len(ids) - len(zero)
        need = (nb - true_count % nb) % nb
        reinc = zero[len(zero) - need:] if need and len(zero) >= need else []
        kept = [ids[i] for i in range(len(ids)) if i not in zero or i in reinc]
        flat = [ids[i] for i in zero if i not in reinc]
        first = gs + 1
        if kept:
            per_step = len(kept) // 2
            for k in range(2):
                gs += 1
                steps.append({"step": gs, "epoch": e, "inner": 0,
                              "sample_ids": kept[k * per_step:(k + 1) * per_step]})
        plans.append({"epoch": e, "filter_reason": "zero_advantage", "kept": kept, "filtered": flat,
                      "reincluded": [ids[i] for i in reinc], "steps": [first, gs] if kept else []})
        consumed += len(kept)
        filtered += len(flat)
        state[e] = (gs, consumed, filtered)
    return samples, plans, steps, state


def base_files(base: Path) -> dict[str, str]:
    """base_ref.json `files`: {relative path: "sha256:<hex>"} of the base without `.cache/`."""
    return {e["path"]: "sha256:" + e["sha256"] for e in manifest_entries(base) if not e["path"].startswith(".cache/")}


def build_cut(out: Path, epoch: int, gs: int, consumed: int, filtered: int, files: dict[str, str]) -> Path:
    d = out / f"checkpoint-{epoch}"
    d.mkdir(parents=True)
    (d / flow_grpo.LORA_WEIGHTS).write_bytes(lora_bytes(f"lora {epoch}"))
    (d / "adapter_config.json").write_text(json.dumps({
        "r": 4, "weights": "raw", "base_model_name_or_path": "org/mini", "base_model_revision": "b" * 40}),
        encoding="utf-8")
    (d / "sampler.json").write_text(json.dumps(SAMPLER), encoding="utf-8")
    (d / "base_ref.json").write_text(json.dumps({"repo": "org/mini", "revision": "b" * 40, "files": files,
                                                 "files_sha256": compact(files)}), encoding="utf-8")
    (d / "trainer_state.json").write_text(json.dumps({"global_step": gs}), encoding="utf-8")
    files = {n: file_sha256(d / n) for n in flow_grpo.CUT_FILES}
    (d / "cut.json").write_text(json.dumps({
        "epoch": epoch, "global_step": gs, "samples_consumed": consumed, "samples_filtered": filtered, "files": files,
        "deployable": list(flow_grpo.COMPONENTS), "ema": False, "seed": SEED, "profile": "full",
        "script_sha256": "sha256:" + "6" * 64, "sampler_code_sha256": "sha256:" + "5" * 64}), encoding="utf-8")
    return d


class FlowCase(unittest.TestCase):
    EPOCHS, GROUP, SETTINGS = EPOCHS, 2, SETTINGS

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-family-flow-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.ad = MiniFlowGrpo()
        self.base = build_base(self.tmp / "base")
        self.data = self.tmp / "prompts"
        self.data.mkdir()
        write_jsonl(self.data / "train.jsonl", PROMPTS)
        self.out = self.tmp / "out"
        self.out.mkdir()
        samples, plans, steps, self.state = simulate(self.EPOCHS, self.GROUP, self.SETTINGS["num_batches_per_epoch"])
        write_jsonl(self.out / "samples.jsonl", samples)
        write_jsonl(self.out / "epoch_plan.jsonl", plans)
        write_jsonl(self.out / "train_steps.jsonl", steps)
        files = base_files(self.base)
        self.args = {"seed": SEED, "attempt": ATTEMPT, "settings": dict(self.SETTINGS), "sampler": SAMPLER,
                     "sampler_sha256": compact(SAMPLER), "n_train_prompts": len(PROMPTS), "profile": "full",
                     "script_sha256": "sha256:" + "6" * 64, "sampler_code_sha256": "sha256:" + "5" * 64,
                     "base_model": "/base", "train_file": "/data/train.jsonl",
                     "train_file_sha256": file_sha256(self.data / "train.jsonl"),
                     "base_ref": {"repo": "org/mini", "revision": "b" * 40, "files_sha256": compact(files),
                                  "verified": True}}
        self.write_args()
        (self.out / "env-runtime.json").write_text(json.dumps({"sampler_code_sha256": "sha256:" + "5" * 64}),
                                                   encoding="utf-8")
        for e, (gs, consumed, filtered) in self.state.items():
            build_cut(self.out, e, gs, consumed, filtered, files)
        (self.out / f".checkpoint-{len(self.EPOCHS) + 1}.partial").mkdir()       # never a cut
        self.launch = {"script_sha256": "sha256:" + "6" * 64, "training_seed": SEED, "profile": "full",
                       "mounts": [{"host": str(self.base), "container": "/base", "mode": "ro"},
                                  {"host": str(self.data), "container": "/data", "mode": "ro"}]}

    def write_args(self) -> None:
        (self.out / "run_args.json").write_text(json.dumps(self.args), encoding="utf-8")

    def cuts(self):
        return self.ad.find_cuts(self.out)[0]

    def edit(self, name: str, fn) -> None:
        docs = read_jsonl(self.out / name)
        write_jsonl(self.out / name, fn(docs) or docs)

    def problems(self) -> list[str]:
        return self.ad.consumption_problems(self.out, self.data, self.cuts())

    def assertRefused(self, text: str) -> None:
        msgs = self.problems()
        self.assertTrue(any(text in m for m in msgs), f"expected {text!r}; got {msgs}")


class TestCutsAndIdentity(FlowCase):
    """MM-20 (EXT-161): a cut is a checkpoint-<E> directory whose cut.json, written last, lists exactly its files with
    their SHA-256; MM-17, MM-18 (EXT-162): the base's hash leaves `.cache/` out and a cut's composes it."""

    def test_find_cuts(self):
        self.assertEqual(self.ad.find_cuts(self.out), ([(2, self.out / "checkpoint-1"), (4, self.out / "checkpoint-2")],
                                                       []))

    def test_incomplete_cuts(self):
        ck = self.out / "checkpoint-2"
        marker = json.loads((ck / "cut.json").read_text(encoding="utf-8"))
        faults = {
            "extra": lambda: (ck / "notes.txt").write_text("x", encoding="utf-8"),
            "sha": lambda: (ck / "sampler.json").write_text("{}", encoding="utf-8"),
            "files": lambda: self.marker(ck, dict(marker, files={k: v for k, v in marker["files"].items()
                                                                 if k != "sampler.json"})),
            "deployable": lambda: self.marker(ck, dict(marker, deployable=["sampler.json"])),
            "epoch0": lambda: self.marker(ck, dict(marker, epoch=0)),
            "count": lambda: self.marker(ck, dict(marker, samples_filtered=-1)),
            "ema": lambda: self.marker(ck, dict(marker, ema="no")),
            "marker": lambda: (ck / "cut.json").unlink(),
            "weights": lambda: self.rehash(ck, flow_grpo.LORA_WEIGHTS, lora_bytes("x")[:-1]),
            "epoch": lambda: self.marker(ck, dict(marker, epoch=5)),
        }
        for name, fault in faults.items():
            with self.subTest(fault=name):
                backup = self.tmp / "backup"
                shutil.copytree(ck, backup)
                fault()
                complete, incomplete = self.ad.find_cuts(self.out)
                self.assertEqual(incomplete, [ck], name)
                self.assertEqual([d for _, d in complete], [self.out / "checkpoint-1"])
                shutil.rmtree(ck)
                backup.rename(ck)

    def marker(self, ck: Path, doc: dict) -> None:
        (ck / "cut.json").write_text(json.dumps(doc), encoding="utf-8")

    def rehash(self, ck: Path, name: str, data: bytes) -> None:
        (ck / name).write_bytes(data)
        doc = json.loads((ck / "cut.json").read_text(encoding="utf-8"))
        doc["files"][name] = file_sha256(ck / name)
        self.marker(ck, doc)

    def test_base_directory(self):
        self.assertIsNone(self.ad.model_dir_problem(self.base))
        self.assertFalse(self.ad.needs_base(self.base))
        self.assertTrue(self.ad.needs_base(self.out / "checkpoint-1"))
        (self.base / "transformer" / "diffusion.safetensors").write_bytes(b"\0" * 3)
        self.assertIn("transformer/", self.ad.model_dir_problem(self.base))
        (self.base / "transformer" / "diffusion.safetensors").unlink()
        self.assertEqual(self.ad.model_dir_problem(self.base), "transformer/ has no safetensors weights")
        (self.base / "model_index.json").unlink()
        self.assertEqual(self.ad.model_dir_problem(self.base), "model_index.json is missing (not a mini base)")

    def test_identity(self):
        def h(entries):
            return "sha256:" + hashlib.sha256(canonical_bytes(entries)).hexdigest()
        base_entries = [e for e in manifest_entries(self.base) if not e["path"].startswith(".cache/")]
        got = self.ad.identity(self.base)
        self.assertEqual(got["model_hash"], h(base_entries))                      # `.cache/` left out
        self.assertNotIn(".cache/download.lock", got["components"])
        cut = self.out / "checkpoint-2"
        adapter = h([e for e in manifest_entries(cut) if e["path"] in flow_grpo.COMPONENTS])
        got = self.ad.identity(cut, self.base)
        self.assertEqual(got["model_hash"], sha_of({"base": h(base_entries), "adapter": adapter}))
        self.assertEqual(got["base_ref"], {"repo": "org/mini", "revision": "b" * 40, "manifest_hash": h(base_entries)})
        with self.assertRaises(ValueError):
            self.ad.identity(cut)


class TestConsumption(FlowCase):
    """MM-24, MM-28: one `consumed` record per optimizer step that applied a sample, one `filtered` record for a
    zero-advantage sample, precision `row`, derivation `observed`."""

    def test_rows(self):
        self.assertEqual(self.problems(), [])
        doc = self.ad.consumption(self.out, self.data, self.cuts(), run_id="r", attempt_id="0")
        self.assertEqual((doc["precision"], doc["derivation"]), ("row", "observed"))
        rows = doc["rows"]
        self.assertEqual([r["outcome"] for r in rows].count("consumed"), 6)
        self.assertEqual([r["outcome"] for r in rows].count("filtered"), 2)
        samples = read_jsonl(self.out / "samples.jsonl")
        flat = [x for x in samples if x["epoch"] == 1 and x["sample_index"] >= 2]
        filt = [r for r in rows if r["outcome"] == "filtered"]
        for x, r in zip(flat, filt):
            key = "\x00".join(map(str, ("r", "0", 1, "filtered", x["prompt_id"], x["seed"], x["member"])))
            self.assertEqual(r["sample_id"], "sha256:" + hashlib.sha256(key.encode()).hexdigest())
            self.assertEqual((r["step"], r["reason"], r["rewards"]), (None, "zero_advantage", [0.5]))
            self.assertEqual(r["data_ref"]["row"], "sha256:" + hashlib.sha256(x["prompt_id"].encode()).hexdigest())
            self.assertEqual(r["seed_inputs"], {"seed": x["seed"]})
        steps = sorted(r["step"] for r in rows if r["outcome"] == "consumed")
        self.assertEqual(steps, [1, 2, 3, 3, 4, 4])
        self.assertEqual({r["sampler_config_hash"] for r in rows}, {sha_of(SAMPLER)})
        self.assertEqual({r["data_ref"]["shard"] for r in rows}, {"train.jsonl"})

    def test_first_cut_only(self):
        """The plan stops at the last cut given: epoch 1's two consumed and two filtered records."""
        doc = self.ad.consumption(self.out, self.data, self.cuts()[:1])
        self.assertEqual(len(doc["rows"]), 4)

    def test_refused_raises(self):
        (self.out / "train_steps.jsonl").unlink()
        with self.assertRaises(ValueError):
            self.ad.consumption(self.out, self.data, self.cuts())

    def test_payloads(self):
        texts, why = self.ad.training_payloads(self.out, self.base, self.cuts(), data=self.data)
        self.assertIsNone(why)
        self.assertEqual(texts, {r["prompt"] for r in PROMPTS})


class TestConsumptionRefusedSamples(FlowCase):
    """Every field train_flow_grpo.py derives in samples.jsonl is re-derived (MM-29)."""

    def sample(self, n: int, **change) -> None:
        self.edit("samples.jsonl", lambda docs: docs.__setitem__(n, dict(docs[n], **change)))

    def test_seed(self):
        self.sample(0, seed=12345)
        self.assertRefused("is not derive_seed('sample', 7, 1, 0)")

    def test_seed_range(self):
        self.sample(0, seed=2 ** 63)
        self.assertRefused("is not a 63-bit integer noise seed")

    def test_member(self):
        self.sample(1, member=0)
        self.assertRefused("member 0 is not sample_index 1 % images_per_prompt 2")

    def test_target(self):
        self.sample(0, target="WORD 9")
        self.assertRefused("is not the prompt's first quoted text")

    def test_reward_hook(self):
        self.sample(0, reward=0.9)
        self.assertRefused("reward is not ocr_acc")

    def test_reward_range(self):
        self.sample(0, ocr_acc=1.5)
        self.assertRefused("is not in [0, 1]")

    def test_sample_id(self):
        self.sample(0, sample_id="sha256:" + "0" * 64)
        self.assertRefused("sample_id is not the hash train_flow_grpo.py derives")

    def test_attempt(self):
        self.args["attempt"] = 2
        self.write_args()
        self.assertRefused("attempt 1 is not run_args.json attempt 2")

    def test_prompt_not_in_shard(self):
        write_jsonl(self.data / "train.jsonl", PROMPTS[1:])
        self.assertRefused("prompt_id p0 is not a row of the training shard train.jsonl")

    def test_prompt_text(self):
        rows = copy.deepcopy(PROMPTS)
        rows[0]["prompt"] = 'another "WORD 0"'
        write_jsonl(self.data / "train.jsonl", rows)
        self.assertRefused("is not the shard's prompt of p0")

    def test_missing_sample(self):
        self.edit("samples.jsonl", lambda docs: docs[:3] + docs[4:])
        self.assertRefused("epoch 1 logs 3 samples")

    def test_group_hook(self):
        self.edit("epoch_plan.jsonl", lambda docs: None)
        rows = read_jsonl(self.out / "samples.jsonl")
        x = rows[1]
        x["prompt_id"], x["prompt"] = "p3", PROMPTS[3]["prompt"]
        x["target"] = target_text(x["prompt"])
        x["sample_id"] = compact([DOMAIN, SEED, ATTEMPT, 1, 1, "p3", x["seed"], 1])
        write_jsonl(self.out / "samples.jsonl", rows)
        self.assertRefused("epoch 1 group 0 has two prompts")

    def test_malformed_line(self):
        self.sample(2, epoch="1")
        self.assertRefused("samples.jsonl line 3 has no integer epoch")


class TestConsumptionRefusedPlanAndSteps(FlowCase):
    """epoch_plan.jsonl: only zero-advantage samples are filtered or re-included; train_steps.jsonl applies each kept
    sample once per inner epoch; every cut's counts equal the logs (MM-27)."""

    def plan(self, e: int, **change) -> None:
        self.edit("epoch_plan.jsonl", lambda docs: docs.__setitem__(e - 1, dict(docs[e - 1], **change)))

    def test_filtered_sample_with_advantage(self):
        p = read_jsonl(self.out / "epoch_plan.jsonl")[0]
        moved = p["kept"][0]
        self.plan(1, kept=p["kept"][1:], filtered=[moved] + p["filtered"])
        self.assertRefused("have a reward other than their group mean")

    def test_flat_group_kept(self):
        p = read_jsonl(self.out / "epoch_plan.jsonl")[0]
        self.plan(1, kept=p["kept"] + p["filtered"], filtered=[])
        self.assertRefused("has equal rewards (zero advantage), but 2 of its samples are kept")

    def test_filter_reason(self):
        self.plan(1, filter_reason="low_reward")
        self.assertRefused("filter_reason 'low_reward' is not 'zero_advantage'")

    def test_reincluded_not_kept(self):
        p = read_jsonl(self.out / "epoch_plan.jsonl")[0]
        self.plan(1, reincluded=p["filtered"][:1])
        self.assertRefused("reincluded (1) is not a subset of kept")

    def test_epoch_missing(self):
        self.edit("epoch_plan.jsonl", lambda docs: docs[1:])
        self.assertRefused("has no line for epoch 1")

    def test_epoch_twice(self):
        self.edit("epoch_plan.jsonl", lambda docs: docs + docs[:1])
        self.assertRefused("epoch 1 is planned twice")

    def test_plan_steps(self):
        self.plan(2, steps=[3, 5])
        self.assertRefused("names steps [3, 5], not the epoch's steps [3, 4]")

    def test_step_applies_a_filtered_sample(self):
        p = read_jsonl(self.out / "epoch_plan.jsonl")[0]
        self.edit("train_steps.jsonl", lambda docs: docs.__setitem__(0, dict(docs[0], sample_ids=p["filtered"][:1])))
        self.assertRefused("a filtered sample")

    def test_steps_not_consecutive(self):
        self.edit("train_steps.jsonl", lambda docs: docs.__setitem__(1, dict(docs[1], step=5)))
        self.assertRefused("are not consecutive from 1")

    def test_step_count(self):
        self.edit("train_steps.jsonl", lambda docs: docs[:3])
        self.assertRefused("epoch 2 has 1 optimizer steps, not")

    def test_sample_applied_twice(self):
        self.edit("train_steps.jsonl", lambda docs: docs.__setitem__(3, dict(docs[3], sample_ids=docs[2]["sample_ids"])))
        self.assertRefused("not each of the epoch's 4 kept samples exactly once")

    def test_cut_counts(self):
        ck = self.out / "checkpoint-1"
        doc = json.loads((ck / "cut.json").read_text(encoding="utf-8"))
        (ck / "cut.json").write_text(json.dumps(dict(doc, samples_consumed=4, samples_filtered=0)), encoding="utf-8")
        msgs = self.problems()
        self.assertTrue(any("checkpoint-1 cut.json samples_consumed is 4, but epoch_plan.jsonl keeps 2" in m
                            for m in msgs), msgs)
        self.assertTrue(any("samples_filtered is 0, but epoch_plan.jsonl filters 2" in m for m in msgs), msgs)

    def test_cut_global_step(self):
        ck = self.out / "checkpoint-1"
        doc = json.loads((ck / "cut.json").read_text(encoding="utf-8"))
        (ck / "cut.json").write_text(json.dumps(dict(doc, global_step=3)), encoding="utf-8")
        self.assertRefused("checkpoint-1 cut.json global_step is 3, but train_steps.jsonl logs 2 steps")

    def test_epochs_beyond_settings(self):
        self.args["settings"]["epochs"] = 1
        self.write_args()
        self.assertRefused("beyond run_args.json settings.epochs 1")


class TestReincludedAndEmptyEpochs(FlowCase):
    """The padding's positive path and an epoch that keeps nothing (train_flow_grpo.py :840-848, :854-858), on 2
    prompts x 4 images (num_batches_per_epoch 2, gradient_accumulation_steps 1):

    - epoch 1: group 0 rewards (1, 0.25, 0.25, 0.5) has 3 nonzero advantages, group 1 is flat; 3 is odd, so one
      zero-advantage sample (sample_index 7, of the flat group) is re-included: 4 kept, 4 filtered;
    - epoch 2: both groups flat: 0 kept, 8 filtered, plan steps [], no optimizer step; its cut repeats global_step 2;
    - epoch 3: group 0 (1, 0, 0.5, 0.5) keeps 2, the flat group 1 and the two mean-reward members are filtered."""

    EPOCHS = {1: (("p0", "p1"), ((1.0, 0.25, 0.25, 0.5), (0.5,) * 4)),
              2: (("p2", "p3"), ((0.0,) * 4, (1.0,) * 4)),
              3: (("p0", "p2"), ((1.0, 0.0, 0.5, 0.5), (0.75,) * 4))}
    GROUP = 4
    SETTINGS = dict(SETTINGS, samples_per_epoch=8, images_per_prompt=4, epochs=3)

    def plans(self) -> list[dict]:
        return read_jsonl(self.out / "epoch_plan.jsonl")

    def plan(self, e: int, **change) -> None:
        self.edit("epoch_plan.jsonl", lambda docs: docs.__setitem__(e - 1, dict(docs[e - 1], **change)))

    def test_accepted(self):
        p1, p2, p3 = self.plans()
        samples = read_jsonl(self.out / "samples.jsonl")
        self.assertEqual(p1["reincluded"], [samples[7]["sample_id"]])                 # the simulator's own shape
        self.assertEqual((len(p1["kept"]), len(p2["kept"]), p2["steps"], len(p3["kept"])), (4, 0, [], 2))
        self.assertEqual(self.state, {1: (2, 4, 4), 2: (2, 4, 12), 3: (4, 6, 18)})
        self.assertEqual(self.problems(), [])
        self.assertEqual(self.run_checks(), [])
        rows = self.ad.consumption(self.out, self.data, self.cuts())["rows"]
        consumed = sorted(r["step"] for r in rows if r["outcome"] == "consumed")
        self.assertEqual(consumed, [1, 1, 2, 2, 3, 4])
        self.assertEqual([r["outcome"] for r in rows].count("filtered"), 18)
        reinc = [r for r in rows if r["data_ref"]["row"] == "sha256:" + hashlib.sha256(b"p1").hexdigest()
                 and r["outcome"] == "consumed"]
        self.assertEqual(len(reinc), 1)                               # the re-included member of the flat group

    def run_checks(self) -> list[str]:
        return [p.message for p in self.ad.checks(last_cut=self.out / "checkpoint-3", base=self.base,
                                                  launch=self.launch, base_field="base", stage_out=self.out,
                                                  data=self.data)]

    def test_reincluded_with_advantage(self):
        """Only a zero-advantage sample is re-included: swap the re-included one for a sample with advantage."""
        p = self.plans()[0]
        samples = read_jsonl(self.out / "samples.jsonl")
        self.plan(1, reincluded=[samples[0]["sample_id"]])
        self.assertRefused("1 reincluded sample(s) have a reward other than their group mean")
        self.plan(1, reincluded=p["reincluded"])
        self.assertEqual(self.problems(), [])

    def test_reincluded_too_many(self):
        """The padding re-includes at most nb - 1 samples: three flat-group members kept as re-included."""
        sid = [x["sample_id"] for x in read_jsonl(self.out / "samples.jsonl")]
        self.plan(1, kept=[sid[i] for i in (0, 1, 2, 5, 6, 7)], filtered=[sid[3], sid[4]],
                  reincluded=[sid[i] for i in (5, 6, 7)])
        self.assertRefused("reincluded (3) is not a subset of kept smaller than num_batches_per_epoch 2")

    def test_reincluded_dropped(self):
        """A flat-group member kept but not listed as re-included is a plain kept zero-advantage sample."""
        self.plan(1, reincluded=[])
        self.assertRefused("prompt group 1 has equal rewards (zero advantage), but 1 of its samples are kept without "
                           "being reincluded")

    def test_empty_epoch_with_steps(self):
        """An epoch that keeps nothing has no optimizer step and plan steps []."""
        self.plan(2, steps=[3, 4])
        self.assertRefused("epoch 2 names steps [3, 4], not the epoch's steps [] (no kept samples)")

    def test_empty_epoch_trained(self):
        """Optimizer steps logged for an epoch that keeps nothing (the later steps renumbered to stay consecutive)."""
        self.edit("train_steps.jsonl", lambda docs: docs[:2] + [
            {"step": 3, "epoch": 2, "inner": 0, "sample_ids": []},
            {"step": 4, "epoch": 2, "inner": 0, "sample_ids": []}] + [dict(d, step=d["step"] + 2) for d in docs[2:]])
        self.plan(3, steps=[5, 6])
        self.assertRefused("train_steps.jsonl epoch 2 has 2 optimizer steps, but epoch_plan.jsonl keeps no samples of it")

    def test_empty_epoch_cut_counts(self):
        ck = self.out / "checkpoint-2"
        doc = json.loads((ck / "cut.json").read_text(encoding="utf-8"))
        (ck / "cut.json").write_text(json.dumps(dict(doc, global_step=3, samples_filtered=4)), encoding="utf-8")
        msgs = self.problems()
        self.assertTrue(any("checkpoint-2 cut.json global_step is 3, but train_steps.jsonl logs 2 steps up to epoch 2"
                            in m for m in msgs), msgs)
        self.assertTrue(any("checkpoint-2 cut.json samples_filtered is 4, but epoch_plan.jsonl filters 12" in m
                            for m in msgs), msgs)


class TestConsumptionRefusedSettings(FlowCase):

    def test_settings(self):
        for change, text in (({"samples_per_epoch": 5}, "is not a multiple of"),
                             ({"global_std": None}, "settings.global_std is None"),
                             ({"num_inner_epochs": 0}, "no positive integer settings")):
            with self.subTest(change=change):
                self.args["settings"] = dict(SETTINGS, **change)
                self.write_args()
                self.assertRefused(text)

    def test_files(self):
        (self.data / "extra.jsonl").write_text("{}\n", encoding="utf-8")
        self.assertRefused("hold 2 shards, not the one shard the run read")
        (self.data / "extra.jsonl").unlink()
        (self.out / "run_args.json").unlink()
        self.assertEqual(self.problems(), ["run_args.json is missing"])
        self.write_args()
        self.assertEqual(self.ad.consumption_problems(self.out, self.data, []), ["no complete cut"])
        (self.out / "epoch_plan.jsonl").write_text("[1]\n", encoding="utf-8")
        self.assertRefused("epoch_plan.jsonl line 1 is not a JSON object")


class TestOverlapAndPins(FlowCase):
    """EXT-176: target texts compared after flow_grpo's normalization; MM-33: the freeze pins and the algorithm
    reference line; the run checks the family adds (EXT-163, EXT-164, EXT-169)."""

    def test_item_overlap(self):
        items = [{"item": "a", "prompt": 'poster "word0"'}, {"item": "b", "prompt": 'poster "fresh"'}]
        msgs = self.ad.item_overlap_problems(items, self.data)
        self.assertEqual(len(msgs), 1)
        self.assertIn("1 item(s) have a target text equal to a training prompt's after normalization", msgs[0])
        self.assertIn(": a (README item 7", msgs[0])
        msgs = self.ad.item_overlap_problems([{"item": "c", "prompt": "no quotes"},
                                              {"item": "d", "prompt": 'x "Y"', "target": "Z"}], self.data)
        self.assertTrue(any("item 'c' has no prompt with a quoted target text" in m for m in msgs), msgs)
        self.assertTrue(any("item 'd' states target 'Z'" in m for m in msgs), msgs)
        self.assertEqual(self.ad.item_overlap_problems(items[1:], self.data), [])

    def test_framework_identity(self):
        launch = {"framework": {"name": "diffusers", "version": "0.36.0", "commit": None}}
        (self.out / "env-freeze.txt").write_text(FREEZE, encoding="utf-8")
        got = self.ad.framework_identity(self.out, launch)
        self.assertEqual(got, {"name": "diffusers", "version": "0.36.0",
                               "freeze_sha256": file_sha256(self.out / "env-freeze.txt")})
        for text, why in ((FREEZE.replace("peft==0.17.1\n", ""), "for peft"),
                          (FREEZE.replace(" (ported)", ""), "algorithm reference"),
                          (FREEZE.replace("0.36.0", "0.35.0"), "but the launch record claims")):
            with self.subTest(why=why):
                (self.out / "env-freeze.txt").write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError) as cm:
                    self.ad.framework_identity(self.out, launch)
                self.assertIn(why, str(cm.exception))
        (self.out / "env-freeze.txt").unlink()
        with self.assertRaises(ValueError):
            self.ad.framework_identity(self.out, launch)

    def run_check_messages(self, launch: dict) -> list[str]:
        out = self.ad.checks(last_cut=self.out / "checkpoint-2", base=self.base, launch=launch, base_field="base",
                             stage_out=self.out, data=self.data)
        return [p.message for p in out]

    def test_family_checks(self):
        launch = self.launch
        family = ("n_train_prompts", "run_args.json script_sha256", "sampler_sha256", "settings.ema",
                  "adapter_config.json weights", "reward is not ocr", "cut.json ema")
        self.assertEqual(self.run_check_messages(launch), [])           # the simulated run passes every run check
        self.args.update(n_train_prompts=3, script_sha256="sha256:" + "7" * 64, sampler_sha256=compact({}))
        self.args["settings"] = dict(SETTINGS, ema=True, reward="brightness")
        self.write_args()
        ck = self.out / "checkpoint-2"
        (ck / "adapter_config.json").write_text(json.dumps({"r": 4, "weights": "ema"}), encoding="utf-8")
        doc = json.loads((ck / "cut.json").read_text(encoding="utf-8"))
        doc["files"]["adapter_config.json"] = file_sha256(ck / "adapter_config.json")
        doc["ema"] = True
        (ck / "cut.json").write_text(json.dumps(doc), encoding="utf-8")
        self.assertEqual(len(self.cuts()), 2)                            # still a complete cut
        got = self.run_check_messages(launch)
        for word in family:
            self.assertTrue(any(word in m for m in got), (word, got))


if __name__ == "__main__":
    unittest.main()
