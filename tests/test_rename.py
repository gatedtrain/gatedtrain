"""The renames `ptc-skel` 0.1.0 -> `ptc-hub` 0.2.0 -> `gatedtrain` 0.3.0: old persisted names are refused, never
translated.

- A registration that records an old `analysis_code_version` (either earlier name) is refused (REG-FIELD) with a fix that names the
  amendment (05 section 3.7: a new registration with `amends` and `created_at`). A version-only amendment (R17-1:
  the prior registration, next to it, differs only in `analysis_code_version`, `amends`, `created_at`, `post_hoc`)
  is not post hoc: it can PASS, and the report says `registration_amendment: version_only`. An amendment across the
  rename that changes anything else must declare `post_hoc: true` (REG-FIELD otherwise) and cannot PASS.
- A launch record, smoke record or arm metadata with an old schema id (`ptc_skel.*` or `ptc_hub.*`) is refused, and
  the message says the record was written before the rename.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from _ext import ExtScenario, smoke_record, write_canonical
from _support import (EXIT_GATE_NOT_PASS, EXIT_OK, EXIT_REFUSED, describe, file_sha256, load_report, parse_errors,
                      parse_run_dir, run_cli)

import yaml

from gatedtrain import ANALYSIS_CODE_VERSION, ANALYSIS_VERSION, PACKAGE, __version__
from gatedtrain.canonical import canonical_json, sha256_bytes
from gatedtrain.ext.registry import EVALUATOR_ROWS
from gatedtrain.ledger import WRITER_ID
from gatedtrain.registration import VERSION_ONLY, check_amendment
from gatedtrain.renamed import NAME_MAP, PRE_RENAME_VERSIONS, apply_name_map

OLD_VERSION = "ptc-hub 0.2.0"        # the version before the latest rename
OLDEST_VERSION = "ptc-skel 0.1.0"    # the version before the first rename


class TestNewNames(unittest.TestCase):

    def test_identifiers(self):
        self.assertEqual(ANALYSIS_CODE_VERSION, "gatedtrain 0.3.0")   # the analysis code version, not the package's
        self.assertEqual(WRITER_ID, f"gatedtrain.ledger@{__version__}")  # provenance: the package that wrote the ledger
        self.assertEqual(PRE_RENAME_VERSIONS, (OLDEST_VERSION, OLD_VERSION))

    def test_name_map_is_the_published_map(self):
        """Both renames, applied in order: a file of either earlier version maps to the current names."""
        self.assertEqual(apply_name_map(b"ptc-skel 0.1.0 ptc-skel.ledger@0.1.0 PTC_SKEL_DESIGN_DIR ptc_skel.item_eval"),
                         b"gatedtrain 0.3.0 gatedtrain.ledger@0.3.0 GATEDTRAIN_DESIGN_DIR gatedtrain.item_eval")
        self.assertEqual(apply_name_map(b"ptc-hub 0.2.0 ptc-hub.ledger@0.2.0 PTC_HUB_EXT_MIRROR_ROOT PTC_GPU_CAP_H "
                                        b"ptc_hub.item_eval ptc_recipe_x ptc-recipe-x ptc-hub"),
                         b"gatedtrain 0.3.0 gatedtrain.ledger@0.3.0 GATEDTRAIN_EXT_MIRROR_ROOT GATEDTRAIN_GPU_CAP_H "
                         b"gatedtrain.item_eval gatedtrain_recipe_x gatedtrain-recipe-x gatedtrain")
        self.assertEqual(apply_name_map(b"ptc-mm ptc_mm.cut"), b"ptc-mm ptc_mm.cut", "no other name is mapped")
        self.assertEqual(len(NAME_MAP), 13)


class TestAnalysisVersion(unittest.TestCase):
    """The analysis code version (05 section 3.7, design revision 24) names the gate statistics and registration
    semantics, not the package release: a package release that leaves them unchanged leaves it, and so every existing
    registration, unchanged."""

    def test_does_not_follow_the_package_version(self):
        self.assertEqual(__version__, "0.4.0")
        self.assertEqual(ANALYSIS_VERSION, "0.3.0")
        self.assertEqual(ANALYSIS_CODE_VERSION, f"{PACKAGE} {ANALYSIS_VERSION}")
        self.assertNotEqual(ANALYSIS_CODE_VERSION, f"{PACKAGE} {__version__}")

    def test_stub_evaluator_identity_follows_the_analysis_version(self):
        """A Tier A registration pins the stand-in evaluator's identity hash; a package release must not change it."""
        from gatedtrain.runner import stub_evaluator_hash

        with mock.patch("gatedtrain.runner.__version__", "9.9.9"):
            pinned = stub_evaluator_hash()
        self.assertEqual(stub_evaluator_hash(), pinned)
        with mock.patch("gatedtrain.runner.ANALYSIS_VERSION", "9.9.9"):
            self.assertNotEqual(stub_evaluator_hash(), pinned)


class RenameCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-rename-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.sc = ExtScenario(cls.tmp / "sc")
        cls.n = 0

    def runs(self) -> Path:
        type(self).n += 1
        return self.tmp / f"runs-{type(self).n}"

    def validate(self):
        return run_cli(["validate", str(self.sc.paths["spec"])], cwd=self.sc.root)


class TestOldAnalysisCodeVersion(RenameCase):
    """05 section 3.7 as of revision 17 (R17-1): an amendment that changes only `analysis_code_version`, from a
    pre-rename version to this code's, is a version-only amendment and is not post hoc; the prior registration,
    found next to the amended one by its hash, is compared field by field. Run for `ptc-hub 0.2.0` here and for
    `ptc-skel 0.1.0` in the subclass."""

    old_version = OLD_VERSION

    def tearDown(self):
        self.prior_path.unlink(missing_ok=True)
        self.sc.render()

    @property
    def prior_path(self) -> Path:
        return self.sc.root / "registration-prior.yaml"

    def render_old(self, *, keep: bool = True) -> str:
        """The registration as written before the rename; with `keep`, a copy stays next to the registration (the
        amended one replaces `registration.yaml`). Returns its hash."""
        self.sc.render(reg_mutate=lambda r: r.update(analysis_code_version=self.old_version))
        if keep:
            shutil.copyfile(self.sc.paths["registration"], self.prior_path)
        return file_sha256(self.sc.paths["registration"])

    def amend(self, prior: str, *, change=None, **fields) -> None:
        def mutate(r):
            r.update(analysis_code_version=ANALYSIS_CODE_VERSION, amends=prior, created_at="2026-11-17T09:00:00Z",
                     post_hoc=False)
            r.update(fields)
            if change is not None:
                change(r)
        self.sc.render(reg_mutate=mutate)

    def collected_report(self, expect: int = EXIT_OK) -> tuple[Path, dict]:
        proc = self.sc.collect(self.runs())
        self.assertEqual(proc.returncode, expect, describe(proc))
        run_dir = parse_run_dir(proc.stdout)
        return run_dir, load_report(run_dir)

    def assert_old_version_refused(self, proc):
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        hits = [e for e in parse_errors(proc.stderr)
                if e["rule"] == "REG-FIELD" and e["field"].endswith("analysis_code_version")]
        self.assertTrue(hits, describe(proc))
        self.assertIn(repr(self.old_version), hits[0]["message"])
        self.assertIn("rename to gatedtrain", hits[0]["message"])
        for part in ("amend the registration (05 section 3.7)", f"analysis_code_version: {ANALYSIS_CODE_VERSION}",
                     "amends: sha256:", "created_at", "`post_hoc: false`", "keep the prior file next to it",
                     "the commit that decided it"):
            self.assertIn(part, hits[0]["fix"])

    def assert_post_hoc_refused(self, proc):
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        hits = [e for e in parse_errors(proc.stderr)
                if (e["rule"], e["field"]) == ("REG-FIELD", "registration.post_hoc")]
        self.assertTrue(hits, describe(proc))
        self.assertIn("not a version-only amendment", hits[0]["message"])
        self.assertIn("margin_justification", hits[0]["message"])

    def test_old_version_refused_at_validate(self):
        self.render_old()
        self.assert_old_version_refused(self.validate())

    def test_old_version_refused_at_collect(self):
        self.render_old()
        runs = self.runs()
        self.assert_old_version_refused(self.sc.collect(runs))
        self.assertFalse(runs.exists() and any(runs.iterdir()), "a refusal creates no run directory")

    def test_amendment_still_naming_the_old_version_refused(self):
        prior = self.render_old()
        self.amend(prior, analysis_code_version=self.old_version)
        self.assert_old_version_refused(self.validate())

    def test_version_only_amendment_passes(self):
        prior = self.render_old()
        self.amend(prior)
        proc = self.validate()
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        run_dir, rep = self.collected_report()
        self.assertEqual(rep["registration_amends"], prior)
        self.assertEqual(rep["registration_amendment"], "version_only")
        self.assertIs(rep["post_hoc"], False)
        self.assertEqual(rep["analysis_code_version"], ANALYSIS_CODE_VERSION)
        self.assertEqual(rep["status"], "PASS")
        self.assertIn("registration_amendment: version_only",
                      (run_dir / "report" / "gate_report.txt").read_text(encoding="utf-8"))
        lock = json.loads((run_dir / "lockfile.json").read_text(encoding="utf-8"))
        self.assertEqual(lock["registration_sha256"], file_sha256(self.sc.paths["registration"]))
        self.assertNotEqual(lock["registration_sha256"], prior)

    def test_version_only_amendment_with_post_hoc_absent_passes(self):
        """`post_hoc` absent means false."""
        prior = self.render_old()
        self.amend(prior, change=lambda r: r.pop("post_hoc"))
        _, rep = self.collected_report()
        self.assertEqual((rep["registration_amendment"], rep["status"]), ("version_only", "PASS"))

    def test_other_change_with_post_hoc_false_refused(self):
        prior = self.render_old()
        self.amend(prior, margin_justification="rewritten after the results")
        self.assert_post_hoc_refused(self.validate())
        runs = self.runs()
        self.assert_post_hoc_refused(self.sc.collect(runs))
        self.assertFalse(runs.exists() and any(runs.iterdir()), "a refusal creates no run directory")

    def test_other_change_with_post_hoc_absent_refused(self):
        prior = self.render_old()
        self.amend(prior, margin_justification="rewritten after the results", change=lambda r: r.pop("post_hoc"))
        self.assert_post_hoc_refused(self.validate())

    def test_other_change_with_post_hoc_true_cannot_pass(self):
        prior = self.render_old()
        self.amend(prior, margin_justification="rewritten after the results", post_hoc=True)
        _, rep = self.collected_report(EXIT_GATE_NOT_PASS)
        self.assertIsNone(rep["registration_amendment"])
        self.assertEqual(rep["status"], "INCONCLUSIVE")
        self.assertIn("post_hoc", rep["status_reasons"])

    def test_post_hoc_true_still_cannot_pass(self):
        """A version-only amendment that declares `post_hoc: true` keeps the general rule: a PASS is reported
        INCONCLUSIVE."""
        self.amend(self.render_old(), post_hoc=True)
        _, rep = self.collected_report(EXIT_GATE_NOT_PASS)
        self.assertEqual(rep["registration_amendment"], "version_only")
        self.assertEqual(rep["status"], "INCONCLUSIVE")
        self.assertIn("post_hoc", rep["status_reasons"])

    def test_post_hoc_prior_cannot_be_cleared(self):
        """R19-1: a version-only amendment of a `post_hoc: true` registration keeps `post_hoc: true`; declaring it
        false (or leaving it out) is refused, so the rename cannot turn an INCONCLUSIVE gate into a PASS."""
        for change in (None, lambda r: r.pop("post_hoc")):
            with self.subTest(post_hoc="absent" if change else False):
                self.sc.render(reg_mutate=lambda r: r.update(analysis_code_version=self.old_version, post_hoc=True))
                shutil.copyfile(self.sc.paths["registration"], self.prior_path)
                self.amend(file_sha256(self.prior_path), change=change)
                proc = self.validate()
                self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
                hits = [e for e in parse_errors(proc.stderr)
                        if (e["rule"], e["field"]) == ("REG-FIELD", "registration.post_hoc")]
                self.assertTrue(hits, describe(proc))
                self.assertIn("prior registration is post_hoc: true", hits[0]["message"])

    def test_post_hoc_prior_kept_is_version_only(self):
        self.sc.render(reg_mutate=lambda r: r.update(analysis_code_version=self.old_version, post_hoc=True))
        shutil.copyfile(self.sc.paths["registration"], self.prior_path)
        self.amend(file_sha256(self.prior_path), post_hoc=True)
        _, rep = self.collected_report(EXIT_GATE_NOT_PASS)
        self.assertEqual(rep["registration_amendment"], "version_only")
        self.assertEqual(rep["status"], "INCONCLUSIVE")
        self.assertIn("post_hoc", rep["status_reasons"])

    def test_prior_not_found_is_not_version_only(self):
        """Without the prior file next to it nothing is compared: the amendment is not marked version-only and
        keeps its declared `post_hoc` (the general rule of 05 section 3.7)."""
        prior = self.render_old(keep=False)
        self.amend(prior)
        _, rep = self.collected_report()
        self.assertIsNone(rep["registration_amendment"])
        self.assertEqual(rep["registration_amends"], prior)

    def test_amendment_without_created_at_refused(self):
        prior = self.render_old()

        def mutate(r):
            r.update(analysis_code_version=ANALYSIS_CODE_VERSION, amends=prior)
            r.pop("created_at", None)
        self.sc.render(reg_mutate=mutate)
        proc = self.validate()
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertIn(("REG-FIELD", "registration.created_at"),
                      [(e["rule"], e["field"]) for e in parse_errors(proc.stderr)])


class TestOldestAnalysisCodeVersion(TestOldAnalysisCodeVersion):
    """The same rules for a registration written before the first rename (`ptc-skel 0.1.0`), amended directly to
    this version."""

    old_version = OLDEST_VERSION


class TestScorerRenameProof(unittest.TestCase):
    """05 section 3.7, R17-3: a version-only amendment may change `evaluators[*].manifest_hash` when its
    `rename_proof` shows the scorer changed only by the published name map (`renamed.NAME_MAP`): the pre-rename
    scorer file(s), next to the registration, hash to `code_sha256_before`, and mapped they give the row's
    `code_sha256` now. Checked in process on a temporary evaluator row; here the prior is a `ptc-hub 0.2.0`
    registration, in the subclass a `ptc-skel 0.1.0` one."""

    EVAL = "renamed.items@0.1"
    PRIOR_VERSION = OLD_VERSION
    OLD = b'meta = {"schema": "ptc_hub.item_eval"}  # read by ptc-hub 0.2.0 from PTC_HUB_EXT_MIRROR_ROOT\n'
    HELPER = b"def helper():\n    return 1\n"

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="gatedtrain-scorer-rename-"))
        self.addCleanup(shutil.rmtree, self.dir, True)

    def install(self, code: str) -> None:
        row = {"metric": "score", "direction": "higher_is_better", "score_range": (0, 1),
               "scorer": {"name": "s", "version": "1", "weights_sha256": None, "code_sha256": code}}
        patcher = mock.patch.dict(EVALUATOR_ROWS, {self.EVAL: row})
        patcher.start()
        self.addCleanup(patcher.stop)

    def identity(self, code: str) -> str:
        return sha256_bytes(canonical_json({"evaluator": self.EVAL, "metric": "score", "direction": "higher_is_better",
                                            "scorer": {"name": "s", "version": "1", "weights_sha256": None,
                                                       "code_sha256": code}}))

    @staticmethod
    def composite(blobs: dict[str, bytes]) -> str:
        doc = {n: sha256_bytes(d) for n, d in blobs.items()}
        return sha256_bytes(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8"))

    def amendment(self, *, old_code: str, new_code: str, files: list[str], composite: bool = False,
                  post_hoc: bool = False, evaluator_extra: dict | None = None, before: str | None = None):
        """Write the prior registration, return (kind, problems) of the amendment."""
        self.install(new_code)
        prior = {"registration_id": "r", "analysis_code_version": self.PRIOR_VERSION,
                 "evaluators": [{"id": self.EVAL, "manifest_hash": self.identity(old_code)}]}
        (self.dir / "registration.yaml").write_text(yaml.safe_dump(prior), encoding="utf-8")
        reg = {**prior, "analysis_code_version": ANALYSIS_CODE_VERSION,
               "amends": file_sha256(self.dir / "registration.yaml"), "created_at": "2026-11-17T09:00:00Z",
               "post_hoc": post_hoc,
               "evaluators": [{"id": self.EVAL, "manifest_hash": self.identity(new_code), **(evaluator_extra or {})}],
               "rename_proof": [{"evaluator": self.EVAL, "code_sha256_before": before or old_code,
                                 "files": files, "composite": composite}]}
        return check_amendment(reg, self.dir / "registration-gatedtrain.yaml")

    def write(self, name: str, data: bytes) -> str:
        path = self.dir / "scorer-before" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)
        return f"scorer-before/{name}"

    def assert_refused(self, result, needle: str):
        kind, problems = result
        self.assertIsNone(kind)
        self.assertEqual([(p.rule, p.field) for p in problems], [("REG-FIELD", "registration.post_hoc")])
        self.assertIn(needle, problems[0].message)

    def test_rename_only_scorer_is_version_only(self):
        f = self.write("score.py", self.OLD)
        new = apply_name_map(self.OLD)
        self.assertNotEqual(new, self.OLD)
        self.assertEqual(self.amendment(old_code=sha256_bytes(self.OLD), new_code=sha256_bytes(new), files=[f]),
                         (VERSION_ONLY, []))

    def test_rename_only_composite_is_version_only(self):
        files = [self.write("score.py", self.OLD), self.write("helper.py", self.HELPER)]
        old = self.composite({"score.py": self.OLD, "helper.py": self.HELPER})
        new = self.composite({"score.py": apply_name_map(self.OLD), "helper.py": self.HELPER})
        self.assertEqual(self.amendment(old_code=old, new_code=new, files=files, composite=True),
                         (VERSION_ONLY, []))

    def test_other_byte_change_refused(self):
        f = self.write("score.py", self.OLD)
        new = apply_name_map(self.OLD) + b"# this host\n"
        self.assert_refused(self.amendment(old_code=sha256_bytes(self.OLD), new_code=sha256_bytes(new), files=[f]),
                            "changed beyond the rename")

    def test_other_byte_change_with_post_hoc_true_accepted_not_version_only(self):
        f = self.write("score.py", self.OLD)
        new = apply_name_map(self.OLD) + b"# this host\n"
        self.assertEqual(self.amendment(old_code=sha256_bytes(self.OLD), new_code=sha256_bytes(new), files=[f],
                                        post_hoc=True), (None, []))

    def test_missing_old_file_refused(self):
        self.assert_refused(self.amendment(old_code=sha256_bytes(self.OLD),
                                           new_code=sha256_bytes(apply_name_map(self.OLD)),
                                           files=["scorer-before/score.py"]), "not found next to the registration")

    def test_old_file_not_the_registered_scorer_refused(self):
        f = self.write("score.py", self.OLD + b"\n")
        self.assert_refused(self.amendment(old_code=sha256_bytes(self.OLD),
                                           new_code=sha256_bytes(apply_name_map(self.OLD)), files=[f]),
                            "do not hash to")

    def test_other_evaluator_field_refused(self):
        f = self.write("score.py", self.OLD)
        self.assert_refused(self.amendment(old_code=sha256_bytes(self.OLD),
                                           new_code=sha256_bytes(apply_name_map(self.OLD)), files=[f],
                                           evaluator_extra={"scorer": "other"}), "changes more than manifest_hash")

    def test_before_hash_not_the_prior_identity_refused(self):
        f = self.write("score.py", self.OLD)
        self.assert_refused(self.amendment(old_code=sha256_bytes(self.OLD),
                                           new_code=sha256_bytes(apply_name_map(self.OLD)), files=[f],
                                           before="sha256:" + "1" * 64), "is not the prior manifest_hash")

class TestOldestScorerRenameProof(TestScorerRenameProof):
    """A scorer written before the first rename maps to this version's names through both renames."""

    PRIOR_VERSION = OLDEST_VERSION
    OLD = b'meta = {"schema": "ptc_skel.item_eval"}  # read by ptc-skel 0.1.0\n'


class TestOldSchemaIds(RenameCase):
    """Schema ids of both earlier names are refused."""

    OLD_PREFIXES = ("ptc_skel.", "ptc_hub.")

    def refused_before_rename(self, proc, rule: str, field_prefix: str):
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        hits = [e for e in parse_errors(proc.stderr) if e["rule"] == rule and e["field"].startswith(field_prefix)
                and "before the rename to gatedtrain" in e["message"]]
        self.assertTrue(hits, describe(proc))

    def test_old_launch_record_refused(self):
        for prefix in self.OLD_PREFIXES:
            with self.subTest(prefix=prefix):
                rec = self.sc.launch_record()
                rec["schema"] = prefix + "launch_record"
                path = write_canonical(self.tmp / f"launch-old-{prefix}json", rec)
                self.refused_before_rename(self.sc.collect(self.runs(), launch=path), "EXT-LAUNCH", "--launch-record")

    def test_old_smoke_record_refused(self):
        for prefix in self.OLD_PREFIXES:
            with self.subTest(prefix=prefix):
                rec = smoke_record(base_model=self.sc.base_model, item_set=self.sc.item_set_hash)
                rec["schema"] = prefix + "smoke_record"
                path = write_canonical(self.tmp / f"smoke-old-{prefix}json", rec)
                self.refused_before_rename(self.sc.collect(self.runs(), smoke=path), "EXT-SMOKE", "--smoke-record")

    def test_old_arm_metadata_refused(self):
        for prefix in self.OLD_PREFIXES:
            with self.subTest(prefix=prefix):
                arm = self.tmp / f"candidate-old-{prefix}"
                shutil.copytree(self.sc.candidate, arm)
                meta = json.loads((arm / "eval_meta.json").read_text(encoding="utf-8"))
                meta["schema"] = prefix + "item_eval"
                write_canonical(arm / "eval_meta.json", meta)
                proc = self.sc.collect(self.runs(), candidate=arm)
                self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
                self.assertIn("before the rename to gatedtrain", proc.stderr, describe(proc))
                self.assertIn(f"was written by {prefix[:-1].replace('_', '-')}", proc.stderr, describe(proc))


if __name__ == "__main__":
    unittest.main()
