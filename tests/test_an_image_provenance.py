"""IMG-7 .. IMG-9: image provenance in the signed report (the driver declares, the keeper checks against its own lock),
the report line, and the `verify` warning / `--require-maintainer-images`. Image IDs are synthetic or read from the lock."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gatedtrain import cli, imagelock, output
from gatedtrain.ainative import image_provenance as ip
from gatedtrain.ainative import keeper_trust, trust_stages
from gatedtrain.ainative.keeper import state, texts
from gatedtrain.ainative.project import create_project
from gatedtrain.errors import Problem, Refused

LOCK = imagelock.load_lock()
LOCK_ID = LOCK["images"]["nemo-rl/tierb"]["accepted"][0]["id"]
USER_ID = "sha256:" + "9" * 64
OTHER = "sha256:" + "8" * 64
COMMIT = "3" * 40
TOTALS = {"jobs": 1, "seconds": 3, "gpu_seconds": 0}


def declared(image_id, **over):
    d = {"id": image_id, "logical": None, "origin": None, "recipe": None, "in_maintainer_lock": False,
         "source": "unlisted"}
    d.update(over)
    return d


def user_cfg(tmp: Path, image_id=USER_ID, logical="nemo-rl/tierb"):
    entry = {"id": image_id, "origin": "user-built",
             "recipe": {"name": "mine", "source": None, "commit": COMMIT, "containerfile_sha256": None, "build": None},
             "base_id": None, "built_utc": "2026-11-25", "added_by": "test", "note": None}
    return imagelock.validate_config({"schema": imagelock.CONFIG_SCHEMA, "schema_version": 1,
                                      "images": {logical: {"accepted": [entry]}}}, LOCK)


class TestDriverSide(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(os.path.realpath(self._tmp.name))
        data = self.tmp / "data.txt"
        data.write_text("d")
        self.project = create_project(self.tmp / "proj", goal="g", data=data, pipeline="p",
                                      budget={"gpu_hours": 1.0, "llm_usd": 0})

    def job_records(self, *images):
        recs = {}
        for i, image in enumerate(images):
            recs[f"000{i}-x"] = {"job": {"runtime": {"kind": "container", "image": image}}}
        return recs

    def test_distinct_image_ids_in_job_records(self):
        recs = self.job_records(LOCK_ID, USER_ID, LOCK_ID, "not-an-id:tag")
        with mock.patch.object(ip, "_job_records", lambda project: list(recs.values())):
            self.assertEqual(ip.job_image_ids(self.project), sorted({LOCK_ID, USER_ID}))

    def test_provenance_rows(self):
        cfg = user_cfg(self.tmp)
        rows = ip.provenance_rows([LOCK_ID, USER_ID, OTHER], lock=LOCK, config=cfg)
        by = {r["id"]: r for r in rows}
        self.assertEqual((by[LOCK_ID]["source"], by[LOCK_ID]["in_maintainer_lock"]), ("lock", True))
        self.assertEqual((by[USER_ID]["source"], by[USER_ID]["in_maintainer_lock"], by[USER_ID]["origin"]),
                         ("owner_config", False, "user-built"))
        self.assertEqual((by[OTHER]["source"], by[OTHER]["logical"]), ("unlisted", None))

    def test_backend_compute_carries_images(self):
        jobs = [mock.Mock(seconds=3, gpu_seconds=0)]
        with mock.patch("gatedtrain.ainative.backend.LocalBackend.jobs", lambda self: jobs), \
                mock.patch.object(ip, "job_image_ids", lambda project: [LOCK_ID]):
            c = keeper_trust.backend_compute(self.project)
        self.assertEqual({k: c[k] for k in ("jobs", "seconds", "gpu_seconds")}, TOTALS)
        self.assertEqual([r["id"] for r in c["images"]], [LOCK_ID])
        self.assertEqual(c["images"][0]["source"], "lock")

    def test_a_job_directory_without_a_record_is_skipped_like_jobs_does(self):
        """A submit cut short leaves `jobs/<id>/` before its `job.json`: it is no job, not a reason to refuse."""
        (self.project.root / "jobs" / "0001-cut").mkdir(parents=True)
        self.assertEqual(ip.job_image_ids(self.project), [])
        self.assertEqual(ip.driver_images(self.project), [])

    def test_a_failure_in_the_images_keeps_the_job_totals(self):
        jobs = [mock.Mock(seconds=3, gpu_seconds=0)]
        with output.notice_scope():
            with mock.patch("gatedtrain.ainative.backend.LocalBackend.jobs", lambda self: jobs), \
                    mock.patch.object(ip, "driver_images", side_effect=Refused([Problem("X-1", "f", "m", "x")])):
                c = keeper_trust.backend_compute(self.project)
            self.assertEqual(c, TOTALS)                                  # totals only: the report then says the
            self.assertIn("trainer images cannot be read", "\n".join(output.drain_notices()))  # images are not stated

    def test_unreadable_owner_config_degrades_to_lock_only_and_says_so(self):
        for exc in (Refused([Problem("IMG-2", "f", "m", "x")]), RecursionError()):
            with output.notice_scope():
                with mock.patch.object(imagelock, "load_user_config", side_effect=exc):
                    rows = ip.driver_images(self.project, ids=[LOCK_ID, USER_ID])
                self.assertEqual([r["source"] for r in rows], ["lock", "unlisted"])
                self.assertIn("owner's image config could not be read", "\n".join(output.drain_notices()))

    def test_unreadable_job_records_are_a_notice_not_only_stderr(self):
        t = object.__new__(keeper_trust.KeeperTrust)
        t.project = self.project
        with output.notice_scope(), mock.patch.object(keeper_trust, "backend_compute", side_effect=OSError("x")), \
                mock.patch.object(keeper_trust.KeeperTrust, "_call", lambda self, op, args=None: {"report": {"ok": 1}}), \
                mock.patch.object(keeper_trust.KeeperTrust, "_keep", lambda self, rec, kind: rec), \
                mock.patch.object(keeper_trust.KeeperTrust, "sync", lambda self: None):
            t.report()
            self.assertIn("job records cannot be read", "\n".join(output.drain_notices()))


class TestKeeperChecks(unittest.TestCase):
    """IMG-8: the keeper re-derives `in_maintainer_lock` from its own shipped lock, whatever the driver says."""

    def compute(self, *images):
        return {**TOTALS, "images": list(images)}

    def test_report_without_images_is_valid(self):
        out = state._backend_compute(dict(TOTALS))
        self.assertEqual(out["jobs"], 1)
        self.assertNotIn("images", out)

    def test_driver_claim_is_ignored(self):
        lie = declared(USER_ID, in_maintainer_lock=True, source="lock", origin="maintainer-built", logical="nemo-rl/tierb")
        out = state._backend_compute(self.compute(lie))
        img = out["images"][0]
        self.assertFalse(img["in_maintainer_lock"])
        self.assertTrue(img["user_added"])
        self.assertEqual(img["checked_by_keeper"], {"in_maintainer_lock": False})
        self.assertEqual(img["declared_by"], "driver")

    def test_lock_id_is_not_user_added_even_if_driver_says_false(self):
        out = state._backend_compute(self.compute(declared(LOCK_ID, in_maintainer_lock=False)))
        img = out["images"][0]
        self.assertTrue(img["in_maintainer_lock"])
        self.assertFalse(img["user_added"])
        self.assertEqual(img["checked_by_keeper"], {"in_maintainer_lock": True})

    def test_listed_id_takes_logical_origin_and_recipe_from_the_keepers_lock(self):
        """IMG-8: the driver cannot re-label a lock image (a wrong logical name, origin or recipe is overwritten)."""
        lie = declared(LOCK_ID, logical="nemo-rl/tierb-uid1000", origin="user-built", source="owner_config",
                       recipe={"name": "forged", "source": None, "commit": COMMIT, "containerfile_sha256": None,
                               "build": None})
        img = state._backend_compute(self.compute(lie))["images"][0]
        entry = LOCK["images"]["nemo-rl/tierb"]["accepted"][0]
        self.assertEqual((img["logical"], img["origin"], img["source"]), ("nemo-rl/tierb", entry["origin"], "lock"))
        self.assertEqual(img["recipe"], entry["recipe"])
        self.assertEqual((img["user_added"], img["declared_by"]), (False, "keeper"))

    def test_unlisted_id_keeps_the_drivers_declaration_and_is_user_added(self):
        rec = {"name": "mine", "source": None, "commit": COMMIT, "containerfile_sha256": None, "build": None}
        img = state._backend_compute(self.compute(declared(USER_ID, logical="nemo-rl/tierb", origin="user-built",
                                                           source="owner_config", recipe=rec)))["images"][0]
        self.assertEqual((img["logical"], img["origin"], img["recipe"]), ("nemo-rl/tierb", "user-built", rec))
        self.assertEqual((img["user_added"], img["declared_by"]), (True, "driver"))

    def test_a_forged_lock_claim_for_an_unlisted_id_is_reduced_to_unlisted(self):
        """M2: the signed record never says `maintainer-built` / `source: lock` next to `user_added: true`."""
        lie = declared(USER_ID, logical="nemo-rl/tierb", origin="maintainer-built", source="lock",
                       recipe={"name": "forged", "source": None, "commit": COMMIT, "containerfile_sha256": None,
                               "build": None}, in_maintainer_lock=True)
        img = state._backend_compute(self.compute(lie))["images"][0]
        self.assertEqual((img["logical"], img["origin"], img["recipe"], img["source"]), (None, None, None, "unlisted"))
        self.assertEqual((img["user_added"], img["in_maintainer_lock"], img["declared_by"]), (True, False, "driver"))
        ip.validate_report_images([img])

    def test_an_owner_config_claim_needs_user_built_and_a_recipe(self):
        for over in ({"origin": "maintainer-built"}, {"origin": None}, {"recipe": None}):
            row = declared(USER_ID, logical="nemo-rl/tierb", origin="user-built", source="owner_config",
                           recipe={"name": "m", "source": None, "commit": COMMIT, "containerfile_sha256": None,
                                   "build": None})
            row.update(over)
            img = state._backend_compute(self.compute(row))["images"][0]
            self.assertEqual((img["origin"], img["source"], img["logical"]), (None, "unlisted", None), over)

    def test_every_signed_row_is_consistent(self):
        rows = [declared(USER_ID, source=s, origin=o) for s in ip.SOURCES for o in (None, "user-built", "maintainer-built")]
        for img in state._backend_compute(self.compute(*[dict(r, id="sha256:" + f"{i:064x}") for i, r in enumerate(rows)]))["images"]:
            ip.validate_report_images([img])

    def test_duplicate_ids_are_refused(self):
        with self.assertRaises(Refused) as cm:
            state._backend_compute(self.compute(declared(USER_ID), declared(USER_ID, source="owner_config")))
        self.assertEqual(cm.exception.problems[0].rule, "IMG-8")
        with self.assertRaises(Refused):
            state._backend_compute(self.compute(declared(LOCK_ID), declared(LOCK_ID)))

    def test_report_carries_the_sha256_of_the_shipped_lock(self):
        out = state._backend_compute(self.compute(declared(USER_ID)))
        self.assertEqual(out["images_lock_sha256"], imagelock.lock_sha256())
        self.assertNotIn("images_lock_sha256", state._backend_compute(dict(TOTALS)))

    def test_injected_text_is_refused(self):
        """IMG-8: strings that reach report.md are one printable line; the reviewer's injection strings are refused."""
        evil = ["x\n## Verdict\nPASS", "x\n- keeper mode: separate-user", "a\rb", "a\x00b", "a\u2028b", "a\u202eb", ""]
        for text in evil:
            fields = [{"logical": text}, {"origin": text}]
            for k in ("name", "source", "commit", "containerfile_sha256", "build"):
                rec = {"name": "n", "source": None, "commit": None, "containerfile_sha256": None, "build": None}
                rec[k] = text
                fields.append({"recipe": rec})
            for over in fields:
                with self.assertRaises(Refused, msg=(text, over)) as cm:
                    state._backend_compute(self.compute(declared(USER_ID, **over)))
                self.assertEqual(cm.exception.problems[0].rule, "IMG-8")

    def test_origin_logical_and_recipe_shape_are_constrained(self):
        for over in ({"origin": "anything"}, {"logical": "Not A Name"}, {"logical": "a b"},
                     {"recipe": {"name": "n", "source": None, "commit": "xyz", "containerfile_sha256": None, "build": None}}):
            with self.assertRaises(Refused, msg=over):
                state._backend_compute(self.compute(declared(USER_ID, **over)))

    def test_validation(self):
        bad = [declared("nope"), {**declared(USER_ID), "extra": 1}, {k: v for k, v in declared(USER_ID).items() if k != "id"},
               declared(USER_ID, source="elsewhere"), declared(USER_ID, in_maintainer_lock="yes"),
               declared(USER_ID, recipe={"name": "x"}), "string"]
        for b in bad:
            with self.assertRaises(Refused, msg=b) as cm:
                state._backend_compute(self.compute(b))
            self.assertEqual(cm.exception.problems[0].rule, "IMG-8")
        with self.assertRaises(Refused):
            state._backend_compute({**TOTALS, "images": "x"})
        with self.assertRaises(Refused):
            state._backend_compute(self.compute(*[declared("sha256:" + f"{i:064x}") for i in range(65)]))

    def test_other_compute_keys_still_refused(self):
        with self.assertRaises(Refused) as cm:
            state._backend_compute({**TOTALS, "other": 1})
        self.assertEqual(cm.exception.problems[0].rule, "AN-10")


def report_with(images):
    return {"schema_version": 2, "attempts": [], "ledger": [],
            "alpha_rule": {"rule": "r", "initial_wealth": 1, "payout": 1, "min_bid": 1, "planning_alpha": 1,
                           "guarantee": "g"},
            "keeper": {"fingerprint": "fp", "environment_sha256": "e"},
            "compute": {"keeper": {"containers": 0, "seconds": 0, "gpu_seconds": 0},
                        "backend": {"jobs": 1, "seconds": 3, "gpu_seconds": 0, "source": texts.COMPUTE_BACKEND_SOURCE,
                                    **({"images": images} if images is not None else {})}}}


class TestRenderAndVerify(unittest.TestCase):
    def images(self):
        mine = state._backend_compute({**TOTALS, "images": [
            declared(USER_ID, logical="nemo-rl/tierb", origin="user-built", source="owner_config",
                     recipe={"name": "mine", "source": None, "commit": COMMIT, "containerfile_sha256": None, "build": None}),
            declared(LOCK_ID, logical="nemo-rl/tierb", origin="maintainer-built", source="lock",
                     recipe={"name": "ref", "source": None, "commit": COMMIT, "containerfile_sha256": None, "build": None})]})
        return mine["images"]

    def test_render_lines(self):
        md = "\n".join(trust_stages._render_keeper(report_with(self.images())))
        self.assertIn(f"trainer image nemo-rl/tierb: user-built {USER_ID} built from recipe mine {COMMIT} "
                      "(USER-ADDED, not in the maintainer lock; the driver declares that jobs ran in it)", md)
        entry = LOCK["images"]["nemo-rl/tierb"]["accepted"][0]
        self.assertIn(f"trainer image nemo-rl/tierb: maintainer-built {LOCK_ID} built from recipe "
                      f"{entry['recipe']['name']} {entry['recipe']['commit'] or '-'} "
                      "(in the maintainer lock: the keeper recognises this ID; the driver declares that jobs ran in it)", md)
        self.assertIn("- not observed: job records are the driver's", md)             # the limit line, once
        self.assertEqual(md.count("not observed:"), 1)

    def test_render_unlisted_image(self):
        out = state._backend_compute({**TOTALS, "images": [declared(OTHER)]})["images"]
        md = "\n".join(trust_stages._render_keeper(report_with(out)))
        self.assertIn(f"trainer image unlisted: unknown {OTHER} built from recipe unknown", md)
        self.assertIn("USER-ADDED", md)

    def test_no_images_renders_as_before(self):
        md = "\n".join(trust_stages._render_keeper(report_with(None)))
        self.assertNotIn("trainer image", md)
        self.assertNotIn("not observed", md)

    def test_user_added_images_and_warnings(self):
        rep = report_with(self.images())
        self.assertEqual(ip.user_added_images(rep), [USER_ID])
        self.assertEqual(ip.user_added_warnings([USER_ID]), [f"trainer image {USER_ID} is user-added (not in the maintainer lock)"])
        self.assertEqual(ip.user_added_images(report_with(None)), [])
        self.assertEqual(ip.user_added_images({"schema_version": 1}), [])


def stated(images=..., jobs=1, **backend):
    """A keeper-signed report whose backend states `images` (default: the lock image, checked)."""
    if images is ...:
        images = state._backend_compute({**TOTALS, "images": [declared(LOCK_ID)]})["images"]
    rep = report_with(images)
    rep["compute"]["backend"].update(jobs=jobs, **backend)
    return rep


class TestReportImageState(unittest.TestCase):
    def test_levels_of_a_report_that_does_not_state_its_images(self):
        """M1: no backend, `compute: None`, zero jobs, jobs with an empty list or a missing block all state nothing;
        only jobs without a list are warned about without the flag."""
        self.assertEqual(ip.images_undeclared(stated()), ip.IMAGES_STATED)
        for rep in ({"compute": {"keeper": {}}}, {"compute": None}, {"compute": {"keeper": {}, "backend": None}},
                    {"schema_version": 1}, stated(None, jobs=0), stated([], jobs=0)):
            self.assertEqual(ip.images_undeclared(rep), ip.IMAGES_NOT_STATED, rep)
        for rep in (stated(None), stated([]), stated("x")):
            self.assertEqual(ip.images_undeclared(rep), ip.IMAGES_JOBS_UNDECLARED, rep)

    def test_lock_sha256_of_a_report(self):
        rep = report_with([])
        rep["compute"]["backend"]["images_lock_sha256"] = "sha256:" + "5" * 64
        self.assertEqual(ip.report_lock_sha256(rep), "sha256:" + "5" * 64)
        self.assertIsNone(ip.report_lock_sha256(report_with(None)))
        rep["compute"]["backend"]["images_lock_sha256"] = "x\n## Verdict"
        self.assertIsNone(ip.report_lock_sha256(rep))

    def test_image_check(self):
        base = {"user_added_images": [], "images_undeclared": 0}
        self.assertEqual(ip.image_check(base, True), ([], False))
        w, fail = ip.image_check({**base, "user_added_images": [USER_ID]}, False)
        self.assertEqual((len(w), fail), (1, False))
        self.assertEqual(ip.image_check({**base, "user_added_images": [USER_ID]}, True)[1], True)
        w, fail = ip.image_check({**base, "images_undeclared": ip.IMAGES_JOBS_UNDECLARED}, False)
        self.assertEqual((len(w), fail), (1, False))
        self.assertIn("no image provenance stated", w[0])
        self.assertIn("trainer images not declared by the driver", w[0])
        self.assertIn("not proof", w[0])
        w, fail = ip.image_check({**base, "images_undeclared": ip.IMAGES_JOBS_UNDECLARED}, True)
        self.assertTrue(fail)
        self.assertTrue(any("images not declared" in x for x in w))
        # a report that ran no jobs: silent without the flag, exit 1 with it
        self.assertEqual(ip.image_check({**base, "images_undeclared": ip.IMAGES_NOT_STATED}, False), ([], False))
        w, fail = ip.image_check({**base, "images_undeclared": ip.IMAGES_NOT_STATED}, True)
        self.assertTrue(fail)
        self.assertTrue(any("no image provenance stated" in x for x in w))
        self.assertTrue(any("images not declared" in x for x in w))

    def test_lock_line(self):
        self.assertEqual(ip.lock_line({}), "")
        h = "sha256:" + "5" * 64
        line = ip.lock_line({"image_lock_sha256": [h]})
        self.assertIn(h, line)
        self.assertIn("in same-user mode the lock file is writable by the user", line)
        self.assertIn("no release hash list is published yet", line)
        self.assertIn("compare with a trusted copy", line)

    def test_user_added_images_counts_every_image_that_is_not_proven_maintainer_built(self):
        rep = stated()
        self.assertEqual(ip.user_added_images(rep), [])
        rep["compute"]["backend"]["images"][0]["user_added"] = True
        with self.assertRaises(Refused):                 # a contradictory row is refused, not trusted
            ip.user_added_images(rep)


class TestMalformedImageEntries(unittest.TestCase):
    """L7: `verify` refuses (AN-14, exit 3) a report whose image entries are malformed, before any rendering."""

    def bad_rows(self):
        good = stated()["compute"]["backend"]["images"][0]
        rows = [{"id": 1}, "x", None, {**good, "extra": 1}, {k: v for k, v in good.items() if k != "id"},
                {**good, "recipe": "x"}, {**good, "in_maintainer_lock": "yes"}, {**good, "declared_by": "nobody"},
                {**good, "checked_by_keeper": {"in_maintainer_lock": False}}, {**good, "user_added": True},
                {**good, "id": "x\n## Verdict"}]
        return rows

    def test_render_lines_refuses_instead_of_crashing(self):
        for row in self.bad_rows():
            with self.assertRaises(Refused, msg=row) as cm:
                ip.render_lines([row])
            self.assertEqual(cm.exception.problems[0].rule, "AN-14")
        with self.assertRaises(Refused):
            ip.render_lines("not a list")

    def test_validate_accepts_what_the_keeper_signs(self):
        ip.validate_report_images(stated()["compute"]["backend"]["images"])

    def test_verify_exits_3_for_a_malformed_entry(self):
        rep = stated()
        rep["compute"]["backend"]["images"] = [{"id": 1}]
        with self.assertRaises(Refused):
            "\n".join(trust_stages._render_keeper(rep))


class TestVerifyWithoutProvenance(unittest.TestCase):
    def test_flag_fails_for_each_shape_and_passes_for_a_checked_lock_image(self):
        for rep in ({"compute": None}, stated(None), stated(None, jobs=0), stated([]), stated([], jobs=0)):
            out = {"user_added_images": [], "images_undeclared": ip.images_undeclared(rep)}
            self.assertTrue(ip.image_check(out, True)[1], rep)
            self.assertFalse(ip.image_check(out, False)[1])
            self.assertTrue(ip.image_check(out, True)[0])
        ok = {"user_added_images": [], "images_undeclared": ip.images_undeclared(stated())}
        self.assertEqual(ip.image_check(ok, True), ([], False))


class TestNotices(unittest.TestCase):
    """L6: a warning raised below the command (a stale keeper, unreadable job records) reaches `--json` warnings.
    Final review F9c: inside `output.notice_scope()` (which `cli.main` enters) a notice is queued for `emit`; outside
    one (a library caller) it goes to stderr at once, so it is never lost."""

    def test_emit_merges_pending_notices_into_the_json_warnings(self):
        with output.notice_scope():
            output.notice("first")
            output.notice("first")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                output.emit(output.Result("x", True, 0, {}, [], ["own"]), as_json=True, human="")
            self.assertEqual(json.loads(out.getvalue())["warnings"], ["own", "first"])
            self.assertEqual(output.drain_notices(), [])

    def test_text_mode_prints_them_to_stderr(self):
        err = io.StringIO()
        with output.notice_scope():
            output.notice("heads up")
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                output.emit(output.Result("x", True, 0), as_json=False, human="")
        self.assertEqual(err.getvalue().count("heads up"), 1)

    def test_outside_a_scope_a_notice_goes_to_stderr_at_once_and_is_not_queued(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            output.notice("library caller\x1b[2J")
        self.assertEqual(err.getvalue(), "library caller\\x1b[2J\n")        # escaped, one line
        self.assertEqual(output.drain_notices(), [])

    def test_inside_a_scope_nothing_is_printed_until_drained_or_the_scope_ends(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            with output.notice_scope():
                with output.notice_scope():                          # nested: only the outermost exit flushes
                    output.notice("a")
                output.notice("b")
                self.assertEqual(err.getvalue(), "")
                self.assertEqual(output.drain_notices(), ["a", "b"])
                output.notice("left over")
            self.assertEqual(err.getvalue(), "left over\n")          # never silently dropped
        self.assertEqual(output.drain_notices(), [])

    def test_main_runs_a_command_in_a_scope_so_json_carries_the_notice(self):
        def func(args):
            output.notice("from below")
            return output.emit(output.Result("x", True, 0), as_json=True, human="")
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), \
                mock.patch.object(cli, "cmd_validate", func):
            code = cli.main(["--json", "validate", "spec.yaml"])
        self.assertEqual((code, err.getvalue()), (0, ""))
        self.assertEqual(json.loads(out.getvalue())["warnings"], ["from below"])

    def test_the_report_docstring_names_the_notice_api(self):
        doc = keeper_trust.KeeperTrust.report.__doc__
        for name in ("output.notice", "output.drain_notices", "output.notice_scope"):
            self.assertIn(name, doc)


class TestOlderKeeper(unittest.TestCase):
    """A new driver against a keeper that predates `images`: it refuses `compute` with AN-10; the driver retries once
    without `images` and says so."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp = Path(os.path.realpath(self._tmp.name))
        (tmp / "d.txt").write_text("d")
        self.project = create_project(tmp / "proj", goal="g", data=tmp / "d.txt", pipeline="p",
                                      budget={"gpu_hours": 1.0, "llm_usd": 0})
        self.calls = []

    def trust(self, fake_call, compute):
        t = object.__new__(keeper_trust.KeeperTrust)
        t.project = self.project
        patches = [mock.patch.object(keeper_trust.KeeperTrust, "_call", lambda self, op, args=None: fake_call(op, args)),
                   mock.patch.object(keeper_trust.KeeperTrust, "_keep", lambda self, rec, kind: rec),
                   mock.patch.object(keeper_trust.KeeperTrust, "sync", lambda self: None),
                   mock.patch.object(keeper_trust, "backend_compute", lambda project: compute)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return t

    def older(self, op, args=None):
        self.calls.append(args)
        if args and set(args["compute"]) != {"jobs", "seconds", "gpu_seconds"}:
            raise Refused([Problem("AN-10", "compute", "compute is not {jobs, seconds, gpu_seconds}, each a whole "
                                   "number >= 0", "send the backend's job totals, or leave compute out")])
        return {"report": {"ok": 1}}

    def test_retries_once_without_images_and_warns(self):
        t = self.trust(self.older, {**TOTALS, "images": [declared(USER_ID)]})
        with output.notice_scope():
            self.assertEqual(t.report(), {"ok": 1})
            self.assertEqual(len(self.calls), 2)
            self.assertIn("images", self.calls[0]["compute"])
            self.assertEqual(self.calls[1]["compute"], TOTALS)
            self.assertIn("the keeper does not know trainer image provenance", "\n".join(output.drain_notices()))

    def test_only_the_old_keepers_exact_key_refusal_is_retried(self):
        """F9a: an AN-10 refusal on `compute` for any other reason is the keeper's real answer, not a keeper that
        predates `images`: it is re-raised after ONE call, with no notice claiming the keeper is old."""
        exact = "compute is not {jobs, seconds, gpu_seconds}, each a whole number >= 0"
        for message, compute in (("compute is too large", {**TOTALS, "images": [declared(USER_ID)]}),
                                 (exact, {**TOTALS, "seconds": -1, "images": [declared(USER_ID)]}),   # bad totals
                                 (exact, {**TOTALS, "seconds": True, "images": [declared(USER_ID)]})):
            self.calls.clear()

            def refuse(op, args=None, message=message):
                self.calls.append(args)
                raise Refused([Problem("AN-10", "compute", message, "fix")])
            t = self.trust(refuse, compute)
            with self.subTest(message=message, compute=compute), output.notice_scope():
                with self.assertRaises(Refused):
                    t.report()
                self.assertEqual(len(self.calls), 1)
                self.assertEqual(output.drain_notices(), [])

    def test_the_old_keeper_notice_is_given_only_when_the_retry_worked(self):
        def always(op, args=None):
            self.calls.append(args)
            raise Refused([Problem("AN-10", "compute", "compute is not {jobs, seconds, gpu_seconds}, each a whole "
                                   "number >= 0", "send the backend's job totals, or leave compute out")])
        t = self.trust(always, {**TOTALS, "images": [declared(USER_ID)]})
        with output.notice_scope():
            with self.assertRaises(Refused):
                t.report()
            self.assertEqual(len(self.calls), 2)
            self.assertEqual(output.drain_notices(), [])

    def test_a_current_keeper_gets_one_call_and_no_warning(self):
        def newer(op, args=None):
            self.calls.append(args)
            return {"report": {"ok": 1}}
        t = self.trust(newer, {**TOTALS, "images": [declared(USER_ID)]})
        with output.notice_scope():
            t.report()
            self.assertEqual(len(self.calls), 1)
            self.assertEqual(output.drain_notices(), [])

    def test_other_refusals_are_not_retried(self):
        def img8(op, args=None):
            self.calls.append(args)
            raise Refused([Problem("IMG-8", "compute.images", "bad", "fix")])
        t = self.trust(img8, {**TOTALS, "images": [declared(USER_ID)]})
        with self.assertRaises(Refused):
            t.report()
        self.assertEqual(len(self.calls), 1)

    def test_an10_refusal_without_images_is_not_retried(self):
        t = self.trust(self.older_always_refuses, dict(TOTALS))
        with self.assertRaises(Refused):
            t.report()
        self.assertEqual(len(self.calls), 1)

    def older_always_refuses(self, op, args=None):
        self.calls.append(args)
        raise Refused([Problem("AN-10", "compute", "bad totals", "fix")])


class TestVerifyCommand(unittest.TestCase):
    def out(self, user_added, **extra):
        from gatedtrain.ainative.keeper.texts import KEEPER_GUARANTEE
        return {"anchor": "fingerprint", "checked": 3, "fingerprint": "fp", "same_user": None, "root_docker": None,
                "report_guarantees": [KEEPER_GUARANTEE], "user_added_images": user_added, **extra}

    def run_verify(self, user_added, *flags, **extra):
        o, e = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e), \
                mock.patch("gatedtrain.ainative.verify.verify_project", lambda *a, **k: self.out(user_added, **extra)):
            code = cli.main(["--json", "verify", "x", "--fingerprint", "fp", *flags])
        return code, json.loads(o.getvalue())

    def test_undeclared_images_warn_and_require_flag_exits_1(self):
        base, _ = self.run_verify([])
        code, doc = self.run_verify([], images_undeclared=ip.IMAGES_JOBS_UNDECLARED)
        self.assertEqual(code, base)
        self.assertTrue(any("trainer images not declared by the driver" in w for w in doc["warnings"]))
        code, doc = self.run_verify([], "--require-maintainer-images", images_undeclared=ip.IMAGES_JOBS_UNDECLARED)
        self.assertEqual(code, 1)
        self.assertTrue(any("images not declared" in w for w in doc["warnings"]))

    def test_verify_prints_the_lock_hash(self):
        h = "sha256:" + "5" * 64
        o, e = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e), \
                mock.patch("gatedtrain.ainative.verify.verify_project",
                           lambda *a, **k: self.out([], image_lock_sha256=[h])):
            cli.main(["verify", "x", "--fingerprint", "fp"])
        self.assertIn(h, o.getvalue() + e.getvalue())

    def test_warning_does_not_change_the_exit_code(self):
        base, _ = self.run_verify([])
        code, doc = self.run_verify([USER_ID])
        self.assertEqual(code, base)
        self.assertTrue(any(USER_ID in w and "user-added" in w for w in doc["warnings"]))

    def test_require_maintainer_images_exits_1(self):
        base, _ = self.run_verify([])
        code, doc = self.run_verify([USER_ID], "--require-maintainer-images")
        self.assertEqual(code, 1)
        self.assertFalse(doc["ok"])
        self.assertTrue(any(USER_ID in w for w in doc["warnings"]))
        self.assertEqual(self.run_verify([], "--require-maintainer-images")[0], base)


if __name__ == "__main__":
    unittest.main()
