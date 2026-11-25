"""Stub marking, docs and Tier B command construction (fast).

- MVP-SPEC section 3: every stand-in is marked `skeleton-stub` in its code;
- MVP-SPEC section 4: Tier B tests cover command construction and parsing of
  recorded sample outputs only; the test suite never runs Tier B;
- MVP-SPEC section 6: docs.
"""

from __future__ import annotations

import unittest

from _support import DOCS_DIR, FIXTURES, PKG_DIR, STUB_MARK
from test_pairing import upstream_load_paired_records


class TestStubModulesMarked(unittest.TestCase):
    def test_package_layout(self):
        for rel in ("__init__.py", "__main__.py", "cli.py", "spec.py", "registration.py", "canonical.py",
                    "gates.py", "ledger.py", "impact.py", "report.py", "stubs/__init__.py",
                    "backends/__init__.py", "backends/nemo_rl.py", "backends/nemo_evaluator.py"):
            with self.subTest(module=rel):
                self.assertTrue((PKG_DIR / rel).is_file(), PKG_DIR / rel)

    def test_every_stub_module_carries_marker(self):
        self.assertTrue((PKG_DIR / "stubs").is_dir(), PKG_DIR / "stubs")
        stubs = sorted((PKG_DIR / "stubs").glob("*.py"))
        self.assertGreaterEqual(len(stubs), 3, "task, model, trainer and evaluator stand-ins")
        for path in stubs:
            with self.subTest(module=path.name):
                self.assertIn(STUB_MARK, path.read_text(encoding="utf-8"))

    def test_version(self):
        import gatedtrain

        self.assertEqual(gatedtrain.__version__, "0.6.0")


class TestDocs(unittest.TestCase):
    def read(self, name):
        path = DOCS_DIR / name
        self.assertTrue(path.is_file(), path)
        return path.read_text(encoding="utf-8")

    def test_quickstart(self):
        text = self.read("quickstart.md")
        for needle in ("uv sync", "gatedtrain warmup", "gatedtrain run", "gatedtrain impact", "gate report",
                       STUB_MARK, "unpromotable", "underpowered"):
            self.assertIn(needle, text)

    def test_tier_b_runbook(self):
        text = self.read("tier-b-runbook.md")
        for needle in ("--no-requeue", "4f39fd8", "ea35ba3", "R9-OI-1", "unpromotable", "GPU-hour"):
            self.assertIn(needle, text)

    def test_tier_b_runbook_shard_export(self):
        """Interim rule G-14: before the first Tier B
        run the training split is exported to local shard files under a manifest;
        without the export the stage is recorded at precision `stage` (05:348; the decision file cites 05:346)."""

        text = self.read("tier-b-runbook.md")
        low = text.lower()
        self.assertNotIn("not designed yet", low)
        lines = low.splitlines()
        self.assertTrue(any("export" in ln and "shard" in ln for ln in lines),
                        "a line describes exporting to local shard files")
        self.assertTrue(any("export" in ln and "split" in ln for ln in lines),
                        "the training split is what gets exported")
        self.assertIn("manifest", low)
        self.assertRegex(text, r"precision\W{0,3}`?stage`?", "falls back to precision `stage`")

    def test_tier_b_runbook_concatenation_rules(self):
        """Finding L-5: the concatenated train.jsonl must not sit inside the shard directory
        (it would become a shard itself), and the hash of the concatenation is recorded and
        checked against the shards concatenated in file-name order."""
        text = self.read("tier-b-runbook.md")
        low = text.lower()
        lines = low.splitlines()
        self.assertTrue(any("train.jsonl" in ln and "shard" in ln and ("not" in ln or "outside" in ln)
                            and ("inside" in ln or "outside" in ln or "within" in ln or " in the shard" in ln)
                            for ln in lines),
                        "a line says train.jsonl must not be inside the shard directory")
        self.assertTrue(any("concatenat" in ln and ("hash" in ln or "sha-256" in ln or "sha256" in ln)
                            and ("record" in ln or "check" in ln or "verif" in ln) for ln in lines),
                        "the hash of the concatenation is recorded and checked")
        self.assertTrue(any(("hash" in ln or "sha" in ln) and ("file-name order" in ln or "file name order" in ln)
                            for ln in lines),
                        "the check concatenates the shards in file-name order")

    def test_tier_b_runbook_root_owned_outputs(self):
        """HIGH-1: the NeMo RL image runs as root (docker/Dockerfile:846 at 4f39fd8), so the runbook
        pre-creates the host directories as the host user, removes base-model/.cache inside the
        container, and says root-owned files under out/*/rl are expected and only read on the host."""
        import re

        text = self.read("tier-b-runbook.md")
        low = text.lower()
        self.assertRegex(text, r"mkdir -p", "directories are pre-created as the host user")
        self.assertIn("tierb/base-model", text)
        self.assertIn("tierb/export", text)
        brace = re.search(r"out/\{smoke,long\}/\{rl,eval\}", text)
        explicit = all(f"out/{p}/{k}" in text for p in ("smoke", "long") for k in ("rl", "eval"))
        self.assertTrue(brace or explicit, "out/{smoke,long}/{rl,eval} are pre-created")
        self.assertRegex(text, r"rm -rf? /work/tierb/base-model/\.cache", "in-container .cache removal")
        self.assertNotRegex(text, r"rm\s+-r\w*\s+\S*HOST_WORK\S*/tierb/base-model/\.cache",
                            "no host-side removal of base-model/.cache")
        self.assertTrue(any("root" in ln and ("owned" in ln or "owner" in ln) and "read" in ln
                            for ln in low.splitlines()),
                        "states that root-owned outputs are expected and only read by host tools")

    def test_tier_b_runbook_tierb_base_row(self):
        """Final re-review LOW-2: the refusal-table row for TIERB-BASE covers both causes: symlinks in
        the baseline directory and a `policy.model_name` that is not the hashed base-model directory."""
        text = self.read("tier-b-runbook.md")
        rows = [ln for ln in text.splitlines() if ln.startswith("|") and "`TIERB-BASE`" in ln]
        self.assertTrue(rows, "a refusal-table row for TIERB-BASE")
        self.assertTrue(any("policy.model_name" in r and "symlink" in r.lower() for r in rows), rows)

    def test_findings_log(self):
        self.read("findings.md")


class TestTierBCommandConstruction(unittest.TestCase):
    STAGE = {
        "id": "rl", "uses": "rl.nemo_rl@0.1", "algorithm": "grpo",
        "model_family": "llm.small@0.1", "inputs": {"model": "base", "data": "train", "config": "cfg"},
        "outputs": {"model": "rl_ckpt"},
    }

    def test_sbatch_no_requeue(self):
        """Launcher: `sbatch --no-requeue` around the recipe's launch script."""
        from gatedtrain.backends.nemo_rl import build_command

        argv = build_command(self.STAGE, nemo_rl_dir="/opt/nemo-rl", launcher="sbatch")
        self.assertEqual(argv[0], "sbatch")
        self.assertIn("--no-requeue", argv)
        self.assertNotIn("--requeue", argv)
        self.assertTrue(all(isinstance(a, str) for a in argv))

    def test_local_command(self):
        from gatedtrain.backends.nemo_rl import build_command

        argv = build_command(self.STAGE, nemo_rl_dir="/opt/nemo-rl", launcher="local")
        self.assertNotIn("sbatch", argv)
        self.assertTrue(any("/opt/nemo-rl" in a for a in argv))

    def test_unknown_launcher(self):
        from gatedtrain.backends.nemo_rl import build_command

        with self.assertRaises(ValueError):
            build_command(self.STAGE, nemo_rl_dir="/opt/nemo-rl", launcher="nemo-run")

    def test_recorded_evaluator_output_parses_like_upstream(self):
        from gatedtrain.backends.nemo_evaluator import read_results

        for arm in ("baseline", "candidate"):
            with self.subTest(arm=arm):
                d = FIXTURES / "pairing" / arm
                self.assertEqual(read_results(d), upstream_load_paired_records()(d))


if __name__ == "__main__":
    unittest.main()
