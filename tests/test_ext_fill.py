"""`gatedtrain ext-arm-fill` on the synthetic adapter (EXT-110, EXT-126, EXT-137, EXT-154, EXT-241; MM-2, MM-17,
MM-18, MM-19).

The recipes that use the command install gatedtrain from PyPI, so the framework suite pins it here. The expected
hashes come from the tests' own implementations (`_ext.model_hash`, `_ext.sampler_hash`, `_support.manifest_hash`),
not from gatedtrain. The LoRA, normalizer and environment-version paths run on a subclass of the synthetic adapter
that declares those hooks (`needs_base`, `normalizer_sha256`, `arm_env_version_sha256`), patched into the registry
for the test only.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from _ext import (EVALUATOR_ID, SAMPLER, build_arm, build_checkpoint, build_hf_dir, build_item_set, gsm8k_rows, jcs,
                  model_hash, sampler_hash, sha_of)
from _support import EXIT_OK, EXIT_REFUSED, describe, file_sha256, manifest_hash, parse_errors, run_cli

from gatedtrain.errors import Refused
from gatedtrain.ext import fill, registry
from gatedtrain.testing.synthetic import SyntheticAdapter

LORA_MARK = "adapter_config.json"          # the subclass's cut marker: a directory holding it is a LoRA cut


def compact(obj) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                                 ensure_ascii=False).encode("utf-8")).hexdigest()


class LoraSynthetic(SyntheticAdapter):
    """The synthetic adapter with the LoRA hooks of the diffusers adapters (MM-18): a cut holding `adapter_config.json`
    composes the base's hash; the normalizer is the cut's `normalizer.json`, the environment version the arm's
    `env_version.json` (EXT-241)."""

    def needs_base(self, d: Path) -> bool:
        return (Path(d) / LORA_MARK).exists()

    def model_dir_problem(self, d: Path):
        if self.needs_base(d):
            return None if (Path(d) / "model.safetensors").is_file() else "model.safetensors is missing"
        return super().model_dir_problem(d)

    def identity(self, cut: Path, base=None, *, entries=None):
        if not self.needs_base(cut):
            return super().identity(cut, base, entries=entries)
        if base is None:
            raise ValueError("a LoRA cut needs its base")
        return {"model_hash": sha_of({"base": super().identity(base)["model_hash"],
                                      "adapter": super().identity(cut)["cut_hash"]})}

    def normalizer_sha256(self, d: Path) -> str:
        return file_sha256(Path(d) / "normalizer.json")

    def arm_env_version_sha256(self, arm: Path) -> str:
        return file_sha256(Path(arm) / "env_version.json")


class FillCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-fill-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.cut = build_checkpoint(self.tmp / "out" / "checkpoint-10", 10, max_steps=10, save_steps=5)
        self.items = build_item_set(self.tmp / "items", gsm8k_rows("test")[:4])
        self.item_ids = [str(i) for i in range(4)]
        self.want = {"checkpoint.model_hash": model_hash(self.cut), "item_set_hash": manifest_hash(self.items),
                     "sampler_hash": sampler_hash()}

    def arm(self, name: str = "arm", *, raw: bool = False, **meta) -> Path:
        """An arm with null hashes unless `meta` sets them. `raw`: eval_meta.json as an evaluator writes it (plain
        JSON keeping the sampler's floats, `top_p` 1.0), not canonical JSON (which writes 1)."""
        base = {"checkpoint": {"model_hash": None, "base_model_hash": None}, "item_set_hash": None,
                "sampler_hash": None}
        base.update(meta)
        d = build_arm(self.tmp / name, self.item_ids, lambda i: 1.0, model="", item_set="", meta=base)
        if raw:
            doc = json.loads((d / "eval_meta.json").read_text(encoding="utf-8"))
            doc["sampler"] = dict(SAMPLER)
            (d / "eval_meta.json").write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")
        return d

    def fill(self, arm: Path, *, adapter: str = "synthetic", checkpoint: Path | None = None, base: Path | None = None):
        return fill.arm_fill(adapter=adapter, arm=arm, checkpoint=checkpoint or self.cut, items=self.items, base=base)

    def refused(self, arm: Path, field: str, **kw) -> list:
        before = (arm / "eval_meta.json").read_bytes() if (arm / "eval_meta.json").is_file() else None
        with self.assertRaises(Refused) as cm:
            self.fill(arm, **kw)
        probs = cm.exception.problems
        self.assertTrue(any(p.field == field for p in probs), [(p.rule, p.field, p.message) for p in probs])
        if before is not None:
            self.assertEqual((arm / "eval_meta.json").read_bytes(), before, "a refusal writes nothing")
        return probs


class TestFillAndVerify(FillCase):
    """EXT-110: null values are filled, set values verified, the file rewritten as canonical JSON only when filled."""

    def test_fills_null_values(self):
        arm = self.arm(created_at="2026-11-14T19:00:00Z")
        lines = (arm / "items.jsonl").read_bytes()
        out = self.fill(arm)
        self.assertEqual(out["filled"], ["checkpoint.model_hash", "item_set_hash", "sampler_hash"])
        self.assertEqual(out["converted"], [])
        self.assertEqual(out["values"], self.want)
        raw = (arm / "eval_meta.json").read_bytes()
        meta = json.loads(raw)
        self.assertEqual(raw, jcs(meta), "eval_meta.json is rewritten as canonical JSON (MM-2)")
        self.assertEqual(meta["checkpoint"], {"model_hash": self.want["checkpoint.model_hash"],
                                              "base_model_hash": None})
        self.assertEqual((meta["item_set_hash"], meta["sampler_hash"]),
                         (self.want["item_set_hash"], self.want["sampler_hash"]))
        self.assertEqual(meta["created_at"], "2026-11-14T19:00:00Z")          # other keys kept
        self.assertEqual((arm / "items.jsonl").read_bytes(), lines, "items.jsonl is never touched")

    def test_verified_values_leave_the_file(self):
        arm = self.arm(checkpoint={"model_hash": self.want["checkpoint.model_hash"], "base_model_hash": None},
                       item_set_hash=self.want["item_set_hash"], sampler_hash=self.want["sampler_hash"])
        pretty = json.dumps(json.loads((arm / "eval_meta.json").read_text(encoding="utf-8")), indent=2)
        (arm / "eval_meta.json").write_text(pretty, encoding="utf-8")
        out = self.fill(arm)
        self.assertEqual((out["filled"], out["converted"]), ([], []))
        self.assertEqual((arm / "eval_meta.json").read_text(encoding="utf-8"), pretty, "nothing to fill: unchanged")

    def test_mismatch_refused(self):
        other = "sha256:" + "0" * 64
        for key, meta in (("checkpoint.model_hash", {"checkpoint": {"model_hash": other, "base_model_hash": None}}),
                          ("item_set_hash", {"item_set_hash": other}), ("sampler_hash", {"sampler_hash": other})):
            with self.subTest(key=key):
                probs = self.refused(self.arm(f"arm-{key}", **meta), f"ARM.{key}")
                self.assertTrue(all(p.rule == "EXT-FILL" for p in probs))

    def test_compact_sampler_hash_converted(self):
        """EXT-126: the evaluator's compact-JSON hash of its own sampler is converted to the RFC 8785 hash (the
        sampler's 1.0 makes the two differ); the compact hash of another sampler is refused."""
        self.assertNotEqual(compact(SAMPLER), sampler_hash())
        arm = self.arm(sampler_hash=compact(SAMPLER), raw=True)
        out = self.fill(arm)
        self.assertEqual(out["converted"], ["sampler_hash"])
        self.assertNotIn("sampler_hash", out["filled"])
        self.assertEqual(json.loads((arm / "eval_meta.json").read_text(encoding="utf-8"))["sampler_hash"],
                         sampler_hash())
        self.refused(self.arm("arm-other", sampler_hash=compact(dict(SAMPLER, temperature=0.6)), raw=True),
                     "ARM.sampler_hash")


class TestFillRefusesTheArm(FillCase):
    """EXT-110: an arm that is not MM-2, another adapter's arm, a cut or item set that cannot be hashed."""

    def test_unreadable_or_not_mm2(self):
        arm = self.tmp / "empty"
        arm.mkdir()
        self.refused(arm, "ARM")
        (arm / "eval_meta.json").write_text("[]", encoding="utf-8")
        self.refused(arm, "ARM")
        for name, meta in (("missing", {"sampler": ...}), ("unknown", {"extra": 1}),
                           ("schema", {"schema": "ptc_mm.item_eval"})):
            with self.subTest(name=name):
                self.refused(self.arm(f"arm-{name}", **meta), "ARM")
        self.refused(self.arm("arm-ck", checkpoint={"model_hash": None}), "ARM.checkpoint")

    def test_another_adapter_or_evaluator(self):
        self.refused(self.arm("arm-a", adapter="trl_grpo"), "ARM.evaluator_id")
        self.refused(self.arm("arm-e", evaluator_id="other.items@0.1"), "ARM.evaluator_id")

    def test_unknown_adapter(self):
        probs = self.refused(self.arm(), "--adapter", adapter="no-such-adapter")
        self.assertEqual(probs[0].rule, "EXT-ADAPTER")

    def test_incomplete_cut(self):
        (self.cut / "config.json").unlink()
        self.refused(self.arm(), "--checkpoint")

    def test_items_missing(self):
        (self.items / "items.jsonl").unlink()
        self.refused(self.arm(), "--items")

    def test_unhashable_cut(self):
        """A cut that cannot be hashed (here: identity raises) is refused, never filled with a partial value."""
        with mock.patch.object(SyntheticAdapter, "identity", side_effect=OSError("unreadable")):
            self.refused(self.arm(), "--checkpoint")


class TestFillLora(FillCase):
    """EXT-126 (MM-18): a LoRA cut needs `--base`, a plain base directory, and both hashes are filled; EXT-154, MM-19,
    EXT-241: the normalizer and environment version are verified against the cut and the arm, never filled."""

    def setUp(self):
        super().setUp()
        self.base = build_hf_dir(self.tmp / "base", "base")
        self.lora = self.tmp / "lora-cut"
        self.lora.mkdir()
        (self.lora / LORA_MARK).write_text("{}", encoding="utf-8")
        (self.lora / "model.safetensors").write_bytes(b"adapter weights stand-in")
        (self.lora / "normalizer.json").write_text('{"mean": 0}', encoding="utf-8")
        patcher = mock.patch.object(registry, "adapter", lambda name: LoraSynthetic())
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_base_required(self):
        self.refused(self.arm(), "--base", checkpoint=self.lora)

    def test_base_must_be_a_base_directory(self):
        self.refused(self.arm(), "--base", checkpoint=self.lora, base=self.lora)          # a cut, not a base
        self.refused(self.arm(), "--base", checkpoint=self.lora, base=self.tmp / "items")

    def test_fills_composed_and_base_hashes(self):
        arm = self.arm()
        out = self.fill(arm, checkpoint=self.lora, base=self.base)
        base_hash = model_hash(self.base)
        cut_hash = manifest_hash(self.lora)
        self.assertEqual(out["values"]["checkpoint.base_model_hash"], base_hash)
        self.assertEqual(out["values"]["checkpoint.model_hash"], sha_of({"base": base_hash, "adapter": cut_hash}))
        self.assertIn("checkpoint.base_model_hash", out["filled"])
        meta = json.loads((arm / "eval_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["checkpoint"]["base_model_hash"], base_hash)
        # a set base hash of another base is refused
        self.refused(self.arm("arm-b", checkpoint={"model_hash": None, "base_model_hash": "sha256:" + "1" * 64}),
                     "ARM.checkpoint.base_model_hash", checkpoint=self.lora, base=self.base)

    def test_normalizer_and_env_version_verified(self):
        row = dict(registry.EVALUATOR_ROWS[EVALUATOR_ID], normalizer=True, env_version=True)
        with mock.patch.dict(registry.EVALUATOR_ROWS, {EVALUATOR_ID: row}):
            arm = self.arm()
            (arm / "env_version.json").write_text('{"env": "1"}', encoding="utf-8")
            probs = self.refused(arm, "ARM.normalizer_sha256", checkpoint=self.lora, base=self.base)
            self.assertIn("ARM.env_version_sha256", [p.field for p in probs])
            ok = self.arm("arm-ok", normalizer_sha256=file_sha256(self.lora / "normalizer.json"),
                          env_version_sha256=file_sha256(arm / "env_version.json"))
            (ok / "env_version.json").write_bytes((arm / "env_version.json").read_bytes())
            self.assertIn("checkpoint.model_hash", self.fill(ok, checkpoint=self.lora, base=self.base)["filled"])
            # a set value that is not the cut's is refused, each field on its own (MM-19, EXT-154)
            other = "sha256:" + "2" * 64
            for key in ("normalizer_sha256", "env_version_sha256"):
                vals = {"normalizer_sha256": file_sha256(self.lora / "normalizer.json"),
                        "env_version_sha256": file_sha256(arm / "env_version.json"), key: other}
                bad = self.arm(f"arm-{key}", **vals)
                (bad / "env_version.json").write_bytes((arm / "env_version.json").read_bytes())
                probs = self.refused(bad, f"ARM.{key}", checkpoint=self.lora, base=self.base)
                self.assertEqual([p.field for p in probs], [f"ARM.{key}"])
                self.assertIn(f"{key} is {other!r}, not the cut's", probs[0].message)
            (self.lora / "normalizer.json").unlink()                      # EXT-154: unreadable refuses
            self.refused(self.arm("arm-gone"), "--checkpoint", checkpoint=self.lora, base=self.base)


class TestFillCli(FillCase):
    """The `ext-arm-fill` command line: one `filled|converted|verified <key>: <value>` line per value; exit 3 and
    `error[EXT-FILL]` on a refusal."""

    def test_cli(self):
        arm = self.arm(sampler_hash=compact(SAMPLER), raw=True)
        args = ["ext-arm-fill", "--adapter", "synthetic", "--checkpoint", str(self.cut), "--items", str(self.items),
                str(arm)]
        proc = run_cli(args, cwd=self.tmp)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertEqual(proc.stdout.splitlines(), [
            f"filled checkpoint.model_hash: {self.want['checkpoint.model_hash']}",
            f"filled item_set_hash: {self.want['item_set_hash']}",
            f"converted sampler_hash: {self.want['sampler_hash']}"])
        proc = run_cli(args, cwd=self.tmp)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertTrue(all(l.startswith("verified ") for l in proc.stdout.splitlines()), proc.stdout)
        (self.cut / "tokenizer_config.json").write_text('{"retrained": true}', encoding="utf-8")
        proc = run_cli(args, cwd=self.tmp)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertIn(("EXT-FILL", "ARM.checkpoint.model_hash"),
                      [(e["rule"], e["field"]) for e in parse_errors(proc.stderr)])


if __name__ == "__main__":
    unittest.main()
