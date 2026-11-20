"""Host layout validation, launch rechecks and collection (EXT-63 to EXT-78).

The synthetic launcher logs the optimizer step and training row in samples.jsonl.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

from _ext import (ADAPTER, BASE_CONTAINER, CODE_CONTAINER, container_uid, container_user, ENV_HOST_WORK, ENV_INPUT_ROOTS, ENV_INTERPRETER, ENV_INTERPRETER_TREES,
                  ENV_MIRROR_ROOT, ENV_OUTPUT_ROOT, EVALUATOR_ID, EXPORT_CONTAINER, SCRIPT_CONTAINER,
                  EXT, INTERPRETER_CONTAINER, SEEDS, ExtScenario, baseline_score, build_arm, build_stage_out, build_train, gpu_step_script, guard_log, launch_label,
                  gsm8k_rows, write_canonical)
from _support import (EXIT_OK, EXIT_REFUSED, HOST_WORK, SKEL_ROOT, VENV_PYTHON, cli_env, describe, file_sha256, load_report,
                      parse_errors, parse_run_dir, run_cli)
from test_ext_from_nel import convert, nel_runs, scorer_candidate
import test_ext_hardening as hardening
from test_ext_hardening import CollectCase, LaunchCase, no_run_dir
from test_ext_launch import GOLDEN_INTERPRETER, GOLDEN_TREE, GPU_LOG, golden_request, launch, rooted
from test_ext_review2 import run_injected, unit_config, unit_request

EXIT_USAGE = 2
OLD = 1794301200            # 2026-11-10T09:00:00Z: before every fixture START (2026-11-11T09:00:00Z)


def problems(req: dict, *, kind: str = "request", config=None, gids=frozenset(), fresh=None) -> list[tuple[str, str]]:
    from gatedtrain.ext.launch import isolation_problems
    return isolation_problems(req, kind=kind, socket_gids=gids, config=config or unit_config(req),
                              fresh_output=fresh)


def fields(req: dict, **kw) -> list[str]:
    return [f for f, _ in problems(req, **kw)]


def with_output(path: str):
    """Request mutation: the output directory and its read-write mount (mounts[4] of the golden request)."""
    def mutate(r):
        r["output_dir"] = path
        r["mounts"][4]["host"] = path
    return mutate


class Launch3(LaunchCase):

    def refused_at(self, req: dict, rule: str, field: str, **kw):
        proc, out = launch(self.sub(), req, root=self.tmp / "host", **kw)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        got = [(e["rule"], e["field"]) for e in parse_errors(proc.stderr)]
        self.assertIn((rule, field), got, describe(proc))
        self.assertEqual(proc.stdout, "")
        self.assertFalse(out.exists())
        return parse_errors(proc.stderr)

    def accepted(self, req: dict, **kw):
        proc, out = launch(self.sub(), req, root=self.tmp / "host", **kw)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        return proc, out


# ---------------------------------------------------------------------------
# EXT-63 (H1): the one read-write mount is a new, empty directory under the output root
# ---------------------------------------------------------------------------

class TestOutputDirLaunch(Launch3):

    def test_refused(self):
        host = self.tmp / "host"
        rooted(golden_request(), host)
        work = host / "data" / "gatedtrain-work"
        (work / "bin").mkdir(parents=True, exist_ok=True)
        (work / "gpu-guard.log").write_text("", encoding="utf-8")
        link = host / "data" / "gatedtrain-mm" / "out" / "bin-link"
        link.symlink_to(work / "bin")
        busy = host / "data" / "gatedtrain-mm" / "out" / "busy"
        (busy / "checkpoint-10").mkdir(parents=True)
        cases = {
            "<host_work>/bin": "/data/gatedtrain-work/bin",
            "<host_work>/gpu-guard.log (a file)": "/data/gatedtrain-work/gpu-guard.log",
            "the skeleton's code": str(SKEL_ROOT / "src" / "gatedtrain"),
            "~/.ssh": str(Path.home() / ".ssh"),
            "/tmp": "/tmp",
            "the output root itself": "/data/gatedtrain-mm/out",
            "a link under the output root to <host_work>/bin": str(link),
            "not empty (an earlier run's cut)": str(busy),
            "missing": str(self.tmp / "absent-out"),
        }
        for name, path in cases.items():
            with self.subTest(case=name):
                req = golden_request()
                with_output(path)(req)
                self.refused_at(req, "EXT-LAUNCH", "--request.output_dir")

    def test_output_root_inside_host_work_refused(self):
        """An output root configured inside host_work does not make host_work writable."""
        host = self.tmp / "host"
        rooted(golden_request(), host)
        (host / "data" / "gatedtrain-work" / "out" / "run").mkdir(parents=True)
        req = golden_request()
        with_output("/data/gatedtrain-work/out/run")(req)
        self.refused_at(req, "EXT-LAUNCH", "--request.output_dir",
                        env={ENV_OUTPUT_ROOT: str(host / "data" / "gatedtrain-work" / "out")})

    def test_file_under_the_output_root_refused(self):
        """A file strictly under the output root, owned by this uid: refused as not a directory."""
        root = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-r3-file-"))
        self.addCleanup(shutil.rmtree, root, True)
        req = unit_request(root)
        f = root / "data" / "gatedtrain-mm" / "out" / "a1" / "file-out"
        f.write_text("", encoding="utf-8")
        with_output(str(f))(req)
        msgs = [m for k, m in problems(req) if k == "output_dir"]
        self.assertTrue(any("not a directory" in m for m in msgs), msgs)

    def test_recorded(self):
        _, out = self.accepted(golden_request())
        rec = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(rec["output_root"], str(self.tmp / "host" / "data" / "gatedtrain-mm" / "out"))

    def test_owner_is_the_container_uid(self):
        """Only root can chown here, so the rule is shown with a uid this test does not run as."""
        root = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-r3-"))
        self.addCleanup(shutil.rmtree, root, True)
        req = unit_request(root)
        uid = os.getuid() + 1
        req["user"] = f"{uid}:{uid}"
        cfg = unit_config(req, primary_gid=lambda u: u)
        self.assertIn("output_dir", fields(req, config=cfg))
        req["user"] = f"{os.getuid()}:{os.getuid()}"
        self.assertEqual(fields(req, config=unit_config(req, primary_gid=lambda u: u)), [])


class TestOutputDirCollect(CollectCase):

    def test_output_root_not_this_host(self):
        self.refused("EXT-LAUNCH", "--launch-record.output_root",
                     launch=self.relabelled(output_root=str(self.tmp / "elsewhere")))

    def test_stage_out_outside_the_output_root(self):
        """The fixture's output under another output root configured for ext-collect."""
        other = self.tmp / "other-root"
        other.mkdir(exist_ok=True)
        proc = run_cli(self.args(self.runs()), cwd=self.sc.root, env={**self.sc.env(), ENV_OUTPUT_ROOT: str(other)})
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertIn(("EXT-LAUNCH", "--launch-record.output_dir"),
                      [(e["rule"], e["field"]) for e in parse_errors(proc.stderr)])

    def test_collect_accepts_a_used_output(self):
        """ext-collect never requires the output to be empty: it holds the run (EXT-63 is fresh at launch only)."""
        self.assertEqual(self.passes()["status"], "PASS")

    def relabelled(self, **overrides) -> Path:
        rec = self.sc.launch_record(**overrides)
        guard_log(self.sc.host_work, rec["label"])
        self.addCleanup(guard_log, self.sc.host_work, self.sc.label)
        return write_canonical(self.path("launch").with_suffix(".json"), rec)

    def args(self, runs: Path) -> list[str]:
        sc = self.sc
        return ["ext-collect", str(sc.paths["spec"]), "--adapter", "synthetic", "--stage-out", str(sc.stage_out),
                "--launch-record", str(sc.launch), "--smoke-record", str(sc.smoke), "--eval-baseline",
                str(sc.baseline), "--eval-candidate", str(sc.candidate), "--runs-root", str(runs)]


# ---------------------------------------------------------------------------
# EXT-64 (H1): outputs older than the label's first START are an earlier run's
# ---------------------------------------------------------------------------

class TestStaleOutputs(CollectCase):

    def aged(self, path: Path) -> None:
        st = path.stat()
        os.utime(path, (OLD, OLD))
        self.addCleanup(os.utime, path, (st.st_atime, st.st_mtime))

    def test_old_cut_file_refused(self):
        self.aged(self.sc.final_cut / "model.safetensors")
        hits = self.refused("EXT-STAGE", "--stage-out")
        self.assertIn("checkpoint-10/model.safetensors", " ".join(h["message"] for h in hits))

    def test_old_earlier_cut_refused(self):
        self.aged(self.sc.stage_out / "checkpoint-5" / "trainer_state.json")
        self.refused("EXT-STAGE", "--stage-out")

    def test_old_run_args_refused(self):
        self.aged(self.sc.stage_out / "run_args.json")
        self.refused("EXT-STAGE", "--stage-out")

    def test_first_start_is_the_bound(self):
        """A file written between the first START (09:00, preempted attempt) and the second (09:30) passes."""
        t = OLD + 86400 + 600                             # 2026-11-11T09:10:00Z
        p = self.sc.final_cut / "config.json"
        st = p.stat()
        os.utime(p, (t, t))
        self.addCleanup(os.utime, p, (st.st_atime, st.st_mtime))
        self.assertEqual(self.passes()["status"], "PASS")

    def test_first_start_unit(self):
        from gatedtrain.ext.launch import first_start
        log = self.tmp / "fs.log"
        log.write_text("2026-11-11T08:00:00Z START other attempt=1\n2026-11-11T09:00:00Z START lab attempt=1\n"
                       "2026-11-11T09:30:00Z START lab attempt=2\n", encoding="utf-8")
        self.assertEqual(first_start(log, "lab"), OLD + 86400)
        with self.assertRaises(ValueError):
            first_start(log, "absent")


# ---------------------------------------------------------------------------
# EXT-65 (H2): the training file is the spec's export, mounted read-only
# ---------------------------------------------------------------------------

class TestTrainingFileCollect(CollectCase):

    def edit_run_args(self, mutate) -> None:
        p = self.sc.stage_out / "run_args.json"
        old = p.read_bytes()
        st = p.stat()
        doc = json.loads(old)
        mutate(doc)
        p.write_text(json.dumps(doc), encoding="utf-8")

        def restore():
            p.write_bytes(old)
            os.utime(p, (st.st_atime, st.st_mtime))
        self.addCleanup(restore)

    def test_probe_other_training_file_refused(self):
        """Review probe test_d: another train.jsonl (hash and path) was a PASS at precision `stage`."""
        self.edit_run_args(lambda a: a.update(train_file_sha256="sha256:" + "0" * 64,
                                              train_file="/inputs/other/train-with-test-items.jsonl"))
        hits = self.refused("EXT-DATA", "artifacts.train.uri")
        text = " ".join(h["message"] for h in hits)
        self.assertIn("train_file_sha256", text)
        self.assertIn("train_file", text)

    def test_hash_mismatch_refused(self):
        self.edit_run_args(lambda a: a.update(train_file_sha256="sha256:" + "0" * 64))
        self.refused("EXT-DATA", "artifacts.train.uri")

    def test_hash_missing_refused(self):
        self.edit_run_args(lambda a: a.pop("train_file_sha256"))
        hits = self.refused("EXT-DATA", "artifacts.train.uri")
        self.assertTrue(any("has no `train_file_sha256`" in h["message"] for h in hits), hits)

    def test_other_train_file_path_refused(self):
        """The right hash but another container path: not the export mount's train.jsonl."""
        self.edit_run_args(lambda a: a.update(train_file="/inputs/other/train.jsonl"))
        hits = self.refused("EXT-DATA", "artifacts.train.uri")
        self.assertTrue(any("`train_file`" in h["message"] for h in hits), hits)

    def test_run_args_missing_refused(self):
        p = self.sc.stage_out / "run_args.json"
        moved = self.tmp / "run_args.moved"
        shutil.move(p, moved)
        self.addCleanup(shutil.move, moved, p)
        self.refused("EXT-DATA", "artifacts.train.uri")
        self.refused("EXT-BASE", "artifacts.base.uri")

    def test_export_not_mounted_refused(self):
        rec = self.sc.launch_record()
        rec["mounts"] = [m for m in rec["mounts"] if m["container"] != EXPORT_CONTAINER]
        path = self.relabelled(rec)
        self.refused("EXT-DATA", "artifacts.train.uri", launch=path)

    def test_shards_mounted_instead_of_the_export_refused(self):
        """The shard directory itself is not the export the launcher reads `<export>/train.jsonl` from."""
        rec = self.sc.launch_record()
        for m in rec["mounts"]:
            if m["container"] == EXPORT_CONTAINER:
                m["host"] = str(self.sc.train)
        self.refused("EXT-DATA", "artifacts.train.uri", launch=self.relabelled(rec))

    def relabelled(self, rec: dict) -> Path:
        rec = dict(rec, label=rec["label"].rsplit("-", 1)[0])
        rec["label"] = launch_label(rec)
        guard_log(self.sc.host_work, rec["label"])
        self.addCleanup(guard_log, self.sc.host_work, self.sc.label)
        return write_canonical(self.path("launch").with_suffix(".json"), rec)


def scenario_request(sc: ExtScenario, out: Path, **overrides) -> dict:
    """An ext-launch request for the scenario's run into `out` (EXT-65, EXT-69)."""
    req = {"area": 1, "label": "ext-synthetic-full", "host_work": str(sc.host_work),
           "runtime": {"kind": "container", "image": "sha256:" + "ab" * 32},
           "framework": {"name": "synthetic-trainer", "version": "1.0.0", "commit": None}, "training_seed": 42, "gpus": True,
           "network": "none", "network_reason": None, "user": container_user(),
           "env": {"HF_HOME": "/tmp/hf", "HF_HUB_OFFLINE": "1"},
           "mounts": [{"host": str(sc.base), "container": BASE_CONTAINER, "mode": "ro"},
                      {"host": str(sc.export), "container": EXPORT_CONTAINER, "mode": "ro"},
                      {"host": str(sc.code), "container": CODE_CONTAINER, "mode": "ro"},       # EXT-83
                      {"host": str(out), "container": "/out", "mode": "rw"}],
           "tmpfs": ["/tmp"], "output_dir": str(out), "gpu_est_h": 2.5,
           "argv": [INTERPRETER_CONTAINER, SCRIPT_CONTAINER, "--model", BASE_CONTAINER, "--output-dir", "/out"]}
    req.update(overrides)
    return req


def ext_launch(sc: ExtScenario, tmp: Path, req: dict, *, extra: list[str] = (), env: dict | None = None):
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "request.yaml").write_text(yaml.safe_dump(req, sort_keys=False), encoding="utf-8")
    out = tmp / "launch.json"
    args = ["ext-launch", "--adapter", "synthetic", "--profile", "full", "--request", str(tmp / "request.yaml"),
            "--gpu-log", str(sc.host_work / "gpu-log.md"), "--smoke-record", str(sc.smoke), "--out-record", str(out),
            *extra]
    return run_cli(args, cwd=tmp, env={**sc.env(), **(env or {})}), out


def recheck(sc: ExtScenario, record: Path, *, extra: list[str] | None = None):
    """`--recheck` with the scenario's spec (EXT-84: required for an adapter that binds the spec) unless `extra` is given."""
    extra = ["--spec", str(sc.paths["spec"])] if extra is None else extra
    return run_cli(["ext-launch", "--adapter", "synthetic", "--recheck", str(record), *extra], cwd=sc.root,
                   env=sc.env())


class TestLaunchWithSpec(unittest.TestCase):
    """EXT-65 at ext-launch (`--spec`), and the whole path: ext-launch, `--recheck`, the run's outputs written
    after the START, ext-collect with the record ext-launch wrote."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-r3-spec-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.sc = ExtScenario(cls.tmp / "sc")
        cls.k = 0

    def new_out(self) -> Path:
        type(self).k += 1
        out = self.sc.output_root / "a1" / f"run-{type(self).k}"
        out.mkdir(parents=True)
        return out

    def spec(self) -> list[str]:
        return ["--spec", str(self.sc.paths["spec"])]

    def test_bound_mounts_accepted(self):
        proc, out = ext_launch(self.sc, self.tmp / "ok", scenario_request(self.sc, self.new_out()), extra=self.spec())
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))

    def test_unbound_refused(self):
        for container, rule, field in ((EXPORT_CONTAINER, "EXT-DATA", "artifacts.train.uri"),
                                       (BASE_CONTAINER, "EXT-BASE", "artifacts.base.uri")):
            with self.subTest(missing=container):
                req = scenario_request(self.sc, self.new_out())
                req["mounts"] = [m for m in req["mounts"] if m["container"] != container]
                proc, out = ext_launch(self.sc, self.tmp / f"no-{rule}", req, extra=self.spec())
                self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
                self.assertIn((rule, field), [(e["rule"], e["field"]) for e in parse_errors(proc.stderr)])
                self.assertFalse(out.exists())

    def test_launch_recheck_run_collect(self):
        sc, out = self.sc, self.new_out()
        proc, record = ext_launch(sc, self.tmp / "e2e", scenario_request(sc, out), extra=self.spec())
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        rec = json.loads(record.read_text(encoding="utf-8"))
        self.assertEqual((rec["output_root"], rec["gpu_step_sha256"]),
                         (str(sc.output_root), file_sha256(sc.host_work / "bin" / "gpu-step.sh")))
        self.assertIn(f"-v {os.path.realpath(out)}:/out:rw", proc.stdout)
        chk = recheck(sc, record, extra=self.spec())
        self.assertEqual(chk.returncode, EXIT_OK, describe(chk))
        self.assertIn("recheck: ok", chk.stdout)
        # the run: outputs written now (after the START below), the label's GPU log
        shutil.copytree(sc.stage_out, out, dirs_exist_ok=True, copy_function=shutil.copyfile)
        guard_log(sc.host_work, rec["label"])
        self.addCleanup(guard_log, sc.host_work, sc.label)
        col = sc.collect(self.tmp / "e2e-runs", stage_out=out, launch=record)
        self.assertEqual(col.returncode, EXIT_OK, describe(col))
        self.assertEqual(load_report(parse_run_dir(col.stdout))["status"], "PASS")
        # after the run, the output is no longer empty: a recheck now refuses (a relaunch needs a new directory)
        late = recheck(sc, record)
        self.assertEqual(late.returncode, EXIT_REFUSED, describe(late))
        self.assertIn(("EXT-LAUNCH", "--recheck.output_dir"), [(e["rule"], e["field"]) for e in parse_errors(late.stderr)])


# ---------------------------------------------------------------------------
# EXT-66 (M1): no credential store is mounted; the gid is the uid's primary group; inputs from the data tree
# ---------------------------------------------------------------------------

class TestCredentialMounts(unittest.TestCase):

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-r3-home-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.home = self.root / "home" / "alice"
        for d in (".ssh", ".docker", ".cache/huggingface", ".config", "work/data/x", "work/data/.venv"):
            (self.home / d).mkdir(parents=True)
        (self.root / "ssh-link").symlink_to(self.home / ".ssh")

    def check(self, host: str) -> list[str]:
        """Messages for `host` mounted read-only, with every input root allowed (`/`) and this test's home."""
        req = unit_request(self.root)
        req["mounts"].append({"host": host, "container": "/x", "mode": "ro"})
        cfg = dataclasses.replace(unit_config(req, home_dirs=(str(self.home), "/root")), input_roots=("/",))
        return [m for f, m in problems(req, config=cfg) if f == "mounts[5].host"]

    def test_refused(self):
        for name, host in {"/etc": "/etc", "/etc/ssl": "/etc/ssl", "/var/log": "/var/log", "/root": "/root",
                           "/home": "/home", "the home directory": str(self.home),
                           "the parent of a home": str(self.home.parent), "~/.ssh": str(self.home / ".ssh"),
                           "~/.docker": str(self.home / ".docker"),
                           "~/.cache/huggingface": str(self.home / ".cache" / "huggingface"),
                           "~/.config": str(self.home / ".config"), "a link to ~/.ssh": str(self.root / "ssh-link"),
                           "~/.ssh behind //": "/" + str(self.home / ".ssh")}.items():
            with self.subTest(case=name):
                msgs = self.check(host)
                self.assertTrue(any("credential store" in m for m in msgs), msgs)

    def test_accepted(self):
        """A data directory in a home, and a dot-directory deeper than the home's top level (a runtime venv)."""
        for host in (self.home / "work" / "data" / "x", self.home / "work" / "data" / ".venv"):
            with self.subTest(host=str(host)):
                self.assertEqual(self.check(str(host)), [])

    def test_host_home_dirs(self):
        from gatedtrain.ext.launch import host_home_dirs
        homes = host_home_dirs()
        self.assertIn("/root", homes)
        home = str(Path.home())
        if home.startswith("/home/") and Path(home).is_dir():
            self.assertIn(home, homes)

    def test_probe_host_paths_refused(self):
        """Review probe p_iso3 on the running host's home and system directories (read-only stat and realpath only)."""
        from gatedtrain.ext.launch import HostConfig
        out = self.root / "out" / "run"
        out.mkdir(parents=True)
        req = unit_request(self.root)
        req.update(host_work=HOST_WORK, output_dir=str(out),
                   mounts=[{"host": str(out), "container": "/out", "mode": "rw"}])
        cfg = HostConfig(output_root=str(self.root / "out"))
        home = str(Path.home())
        for host in (f"{home}/.ssh", f"{home}/.docker", f"{home}/.cache/huggingface", "/etc", "/var/log",
                     "/home", home, "/root"):
            with self.subTest(host=host):
                r = copy.deepcopy(req)
                r["mounts"].append({"host": host, "container": "/x", "mode": "ro"})
                self.assertIn("mounts[1].host", fields(r, config=cfg))


class TestInputRoots(unittest.TestCase):

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-r3-in-"))
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_outside_refused_inside_accepted(self):
        outside = self.root / "elsewhere"
        outside.mkdir()
        link = self.root / "data" / "link-out"
        req = unit_request(self.root)
        link.symlink_to(outside)
        for host, ok in ((str(outside), False), (str(link), False), (str(self.root / "data" / "gatedtrain-mm" / "python"), True)):
            with self.subTest(host=host):
                r = copy.deepcopy(req)
                r["mounts"].append({"host": host, "container": "/x", "mode": "ro"})
                msgs = [m for f, m in problems(r) if f == "mounts[5].host"]
                self.assertEqual(not msgs, ok, msgs)
                if not ok:
                    self.assertTrue(any("input root" in m for m in msgs), msgs)

    def test_environment_override(self):
        from gatedtrain.errors import Refused
        from gatedtrain.ext.launch import ENV_HOST_CONFIG, HostConfig, host_config
        missing = str(self.root / "no-such-host.toml")
        # no built-in host path: an unconfigured layout refuses, naming both keys
        with self.assertRaises(Refused) as ctx:
            host_config({ENV_HOST_CONFIG: missing})
        self.assertEqual(sorted(p.field for p in ctx.exception.problems), ["host_work", "output_root"])
        # review round 4 (EXT-79): no global input root; the adapter's registry row names them
        self.assertIsNone(HostConfig().input_roots)
        host = self.root / "host.toml"
        host.write_text('host_work = "/hw"\noutput_root = "/ho"\n', encoding="utf-8")
        cfg = host_config({ENV_HOST_CONFIG: str(host)})
        self.assertEqual((cfg.host_work, cfg.output_root, cfg.input_roots), ("/hw", "/ho", None))
        cfg = host_config({ENV_HOST_CONFIG: str(host), ENV_HOST_WORK: "/w", ENV_OUTPUT_ROOT: "/o",
                           ENV_INPUT_ROOTS: f"/a{os.pathsep}/b"})
        self.assertEqual((cfg.host_work, cfg.output_root, cfg.input_roots), ("/w", "/o", ("/a", "/b")))
        for text in ('host_work = "relative"\noutput_root = "/o"\n', 'host_work = "/w"\nextra = "/x"\n', "= broken"):
            host.write_text(text, encoding="utf-8")
            with self.assertRaises(Refused):
                host_config({ENV_HOST_CONFIG: str(host)})


class TestPrimaryGroup(Launch3):

    def test_supplementary_groups_refused(self):
        """Probe p_iso3: /etc with gid 42 (shadow), /var/log with gid 4 (adm): the gid alone is refused now."""
        primary = int(container_user().split(":")[1])
        for gid in [g for g in (4, 42, 1001, 1002) if g != primary]:
            with self.subTest(gid=gid):
                req = golden_request()
                req["user"] = f"{container_uid()}:{gid}"
                self.refused_at(req, "EXT-LAUNCH", "--request.user")

    def test_unit(self):
        root = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-r3-gid-"))
        self.addCleanup(shutil.rmtree, root, True)
        req = unit_request(root)
        req["user"] = "1000:1001"
        self.assertIn("user", fields(req, config=unit_config(req, primary_gid=lambda u: 1000)))
        self.assertNotIn("user", fields(req, config=unit_config(req, primary_gid=lambda u: 1001)))
        msgs = [m for f, m in problems(req, config=unit_config(req, primary_gid=lambda u: None)) if f == "user"]
        self.assertTrue(any("not a user of this host" in m for m in msgs), msgs)                     # no such user

    def test_passwd_lookup(self):
        import pwd
        from gatedtrain.ext.launch import passwd_gid
        me = pwd.getpwuid(os.getuid())
        self.assertEqual(passwd_gid(me.pw_uid), me.pw_gid)
        self.assertIsNone(passwd_gid(2**31 - 2))


# ---------------------------------------------------------------------------
# EXT-67 (M2): host_work, its wrapper and its GPU log are the configured ones
# ---------------------------------------------------------------------------

class TestHostWorkLaunch(Launch3):

    def test_other_host_work_refused(self):
        other = self.tmp / "host" / "data" / "other-work"
        gpu_step_script(other)
        req = golden_request()
        req["host_work"] = "/data/other-work"
        self.refused_at(req, "EXT-LAUNCH", "--request.host_work",
                        env={ENV_HOST_WORK: str(self.tmp / "host" / "data" / "gatedtrain-work")})

    def test_root_host_work_refused(self):
        req = golden_request()
        req["host_work"] = "/"
        self.refused_at(req, "EXT-LAUNCH", "--request.host_work",
                        env={ENV_HOST_WORK: str(self.tmp / "host" / "data" / "gatedtrain-work")})

    def test_gpu_log_elsewhere_refused(self):
        self.refused_at(golden_request(), "EXT-GPU", "--gpu-log", gpu_log=GPU_LOG)

    def test_wrapper_checked_at_launch(self):
        rooted(golden_request(), self.tmp / "host")
        gpu_step_script(self.tmp / "host" / "data" / "gatedtrain-work", work="/srv/gatedtrain/tierb")
        self.refused_at(golden_request(), "EXT-GPU", "--request.host_work")

    def test_zero_estimate_refused(self):
        req = golden_request()
        req["gpu_est_h"] = 0
        self.refused_at(req, "EXT-LAUNCH", "--request.gpu_est_h")

    def test_wrapper_hash_recorded(self):
        _, out = self.accepted(golden_request())
        rec = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(rec["gpu_step_sha256"],
                         file_sha256(self.tmp / "host" / "data" / "gatedtrain-work" / "bin" / "gpu-step.sh"))


class TestHostWorkCollect(CollectCase):

    def test_wrapper_changed_after_launch_refused(self):
        """Same WORK and LOG lines, other content: the recorded SHA-256 no longer matches."""
        p = gpu_step_script(self.sc.host_work)
        p.write_text(p.read_text(encoding="utf-8").replace("exit 99\n", "GATEDTRAIN_GPU_CAP_H=1000\nexit 99\n"),
                     encoding="utf-8")
        self.addCleanup(gpu_step_script, self.sc.host_work)
        self.refused("EXT-GPU", "--launch-record.gpu_step_sha256")

    def test_record_without_new_keys_refused(self):
        rec = self.sc.launch_record(hashed_label=False)
        for k in ("output_root", "gpu_step_sha256"):
            with self.subTest(key=k):
                r = {x: v for x, v in rec.items() if x != k}
                r["label"] = launch_label(r)
                path = write_canonical(self.path("launch").with_suffix(".json"), r)
                self.refused("EXT-LAUNCH", f"--launch-record.{k}", launch=path)


# ---------------------------------------------------------------------------
# EXT-68 (M3): run_args.json agrees with the launch record and the last cut
# ---------------------------------------------------------------------------

class TestRunArgsAgree(CollectCase):

    def edit(self, mutate) -> None:
        TestTrainingFileCollect.edit_run_args(self, mutate)

    def test_probe_smoke_dry_run_refused(self):
        """Review probe test_c: run_args profile smoke and dry_run true against a full launch record."""
        self.edit(lambda a: a.update(profile="smoke", dry_run=True))
        self.refused("EXT-PROFILE", "--launch-record.profile")
        self.refused("EXT-STAGE", "--stage-out")

    def test_dry_run_refused(self):
        for value in (True, None, "false"):
            with self.subTest(dry_run=value):
                self.edit(lambda a: a.update(dry_run=value))
                self.refused("EXT-STAGE", "--stage-out")
                self.doCleanups()

    def test_max_steps_not_the_last_cut_refused(self):
        for value in (20, 5, None, True):
            with self.subTest(max_steps=value):
                self.edit(lambda a: a.update(max_steps=value))
                hits = self.refused("EXT-STAGE", "--stage-out")
                self.assertTrue(any("max_steps" in h["message"] for h in hits))
                self.doCleanups()


# ---------------------------------------------------------------------------
# EXT-69 (M4): the command mounts resolved paths; --recheck re-validates right before it
# ---------------------------------------------------------------------------

class TestResolvedSources(Launch3):

    def test_link_source_printed_resolved(self):
        host = self.tmp / "host"
        rooted(golden_request(), host)
        link = host / "data" / "py-link"
        link.symlink_to(host / "data" / "gatedtrain-mm" / "python")
        req = golden_request()
        req["mounts"][0]["host"] = "/data/py-link"
        proc, _ = self.accepted(req)
        self.assertIn(f" -v {host / 'data' / 'gatedtrain-mm' / 'python'}:", proc.stdout)
        self.assertNotIn(str(link), proc.stdout)

    def test_build_command_unresolvable_source(self):
        from gatedtrain.ext.launch import build_command
        root = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-r3-bc-"))
        self.addCleanup(shutil.rmtree, root, True)
        req = unit_request(root)
        rec = dict(req, env_names=["HF_HUB_OFFLINE"], gpu_used_h=0.0)
        rec["mounts"][0]["host"] = str(root / "gone")
        with self.assertRaises(ValueError):
            build_command(rec, {"HF_HUB_OFFLINE": "1"}, est_h=1)


class TestRecheck(Launch3):

    def launched(self) -> tuple[Path, Path]:
        sub = self.sub()
        proc, out = launch(sub, golden_request(), root=self.tmp / "host")
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        return sub, out

    def recheck(self, record: Path, spec: Path | None = None):
        """EXT-84: `--spec` is the spec `test_ext_launch.launch` wrote next to the record (`<sub>/spec`)."""
        spec = spec if spec is not None else record.parent / "spec" / "spec.yaml"
        e = cli_env()
        e.update({ENV_HOST_WORK: str(self.tmp / "host" / "data" / "gatedtrain-work"),
                  ENV_OUTPUT_ROOT: str(self.tmp / "host" / "data" / "gatedtrain-mm" / "out"),
                  ENV_INPUT_ROOTS: str(self.tmp / "host" / "data"), ENV_MIRROR_ROOT: str(self.tmp / "host"),  # EXT-112
                  ENV_INTERPRETER: GOLDEN_INTERPRETER,                                                    # EXT-113
                  ENV_INTERPRETER_TREES: str(self.tmp / "host" / GOLDEN_TREE)})
        return subprocess.run([str(VENV_PYTHON), "-m", "gatedtrain.testing.synthetic", "ext-launch", "--adapter", "synthetic",
                               "--recheck", str(record), "--spec", str(spec)], cwd=str(self.tmp), env=e,
                              capture_output=True, text=True, timeout=60)

    def refused_recheck(self, record: Path, rule: str, field: str):
        proc = self.recheck(record)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertIn((rule, field), [(e["rule"], e["field"]) for e in parse_errors(proc.stderr)], describe(proc))

    def test_ok(self):
        _, rec = self.launched()
        proc = self.recheck(rec)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertTrue(proc.stdout.startswith("recheck: ok ext-synthetic-smoke-"), proc.stdout)

    def test_output_written_before_the_run(self):
        _, rec = self.launched()
        out = self.tmp / "host" / "data" / "gatedtrain-mm" / "out" / "a1" / "stage-out"
        (out / "stale.txt").write_text("x", encoding="utf-8")
        self.addCleanup((out / "stale.txt").unlink)
        self.refused_recheck(rec, "EXT-LAUNCH", "--recheck.output_dir")

    def test_input_swapped_for_a_link_to_a_credential_store(self):
        _, rec = self.launched()
        py = self.tmp / "host" / "data" / "gatedtrain-mm" / "python"
        saved = py.with_name("python.saved")
        py.rename(saved)
        py.symlink_to("/etc")
        self.addCleanup(saved.rename, py)
        self.addCleanup(py.unlink)
        self.refused_recheck(rec, "EXT-LAUNCH", "--recheck.mounts[0].host")

    def test_wrapper_changed(self):
        _, rec = self.launched()
        p = self.tmp / "host" / "data" / "gatedtrain-work" / "bin" / "gpu-step.sh"
        old = p.read_text(encoding="utf-8")
        p.write_text(old + "# edited\n", encoding="utf-8")
        self.addCleanup(p.write_text, old, encoding="utf-8")
        self.refused_recheck(rec, "EXT-GPU", "--recheck.gpu_step_sha256")

    def test_wrapper_log_moved(self):
        """gpu-step.sh now logs elsewhere: the EXT-46/EXT-57 script check runs at the recheck too."""
        _, rec = self.launched()
        work = self.tmp / "host" / "data" / "gatedtrain-work"
        p = gpu_step_script(work, work="/elsewhere")
        self.addCleanup(gpu_step_script, work)
        self.refused_recheck(rec, "EXT-GPU", "--recheck.host_work")
        self.assertTrue(p.is_file())

    def test_record_edited(self):
        _, rec = self.launched()
        doc = json.loads(rec.read_text(encoding="utf-8"))
        doc["gpu_est_h"] = 0.5
        edited = rec.parent / "edited.json"
        edited.write_text(json.dumps(doc), encoding="utf-8")
        self.refused_recheck(edited, "EXT-LAUNCH", "--recheck.label")

    def test_usage(self):
        proc = subprocess.run([str(VENV_PYTHON), "-m", "gatedtrain.testing.synthetic", "ext-launch", "--adapter", "synthetic"],
                              cwd=str(self.tmp), env=cli_env(), capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, EXIT_USAGE, describe(proc))
        self.assertIn("--recheck", proc.stderr)


# ---------------------------------------------------------------------------
# EXT-70 (M5): uid 0 with a non-root gid (mutant M18)
# ---------------------------------------------------------------------------

class TestRootUid(Launch3):

    def test_request(self):
        req = golden_request()
        req["user"] = "0:1000"
        errs = self.refused_at(req, "EXT-LAUNCH", "--request.user")
        self.assertTrue(any("uid 0" in e["message"] for e in errs), errs)

    def test_unit_record(self):
        root = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-r3-uid-"))
        self.addCleanup(shutil.rmtree, root, True)
        rec = unit_request(root)
        rec.update(user="0:1000", env_names=["HF_HUB_OFFLINE"])
        msgs = [m for f, m in problems(rec, kind="record", config=unit_config(rec, primary_gid=lambda u: 1000))
                if f == "user"]
        self.assertTrue(any("uid 0" in m for m in msgs), msgs)


class TestRootUidCollect(CollectCase):

    def test_record(self):
        rec = self.sc.launch_record(user="0:1000")
        guard_log(self.sc.host_work, rec["label"])
        self.addCleanup(guard_log, self.sc.host_work, self.sc.label)
        hits = self.refused("EXT-LAUNCH", "--launch-record.user",
                            launch=write_canonical(self.path("launch").with_suffix(".json"), rec))
        self.assertTrue(any("uid 0" in h["message"] for h in hits), hits)


# ---------------------------------------------------------------------------
# EXT-71, EXT-72 (L1, L2): credential and reserved environment names
# ---------------------------------------------------------------------------

class TestEnvNames(unittest.TestCase):

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-r3-env-"))
        self.addCleanup(shutil.rmtree, self.root, True)

    def check(self, refused: tuple, allowed: tuple, *, word: str | None = None):
        """`refused` names are refused, `allowed` accepted; with `word`, the test reads the refusal naming it, since
        review round 5 (EXT-92) refuses every name outside the adapter's allow-list as well."""
        for name in refused + allowed:
            with self.subTest(name=name):
                req = unit_request(self.root)
                req["env"][name] = "x"
                rec = dict(req, env_names=sorted(req["env"]))
                if word is None:
                    self.assertEqual(f"env.{name}" in fields(req), name in refused)
                    self.assertEqual("env_names" in fields(rec, kind="record"), name in refused)
                    continue
                msgs = [m for f, m in problems(req) if f == f"env.{name}"]
                self.assertEqual(any(word in m for m in msgs), name in refused, msgs)
                msgs = [m for f, m in problems(rec, kind="record") if f == "env_names" and m.startswith(name + " ")]
                self.assertEqual(any(word in m for m in msgs), name in refused, msgs)

    def test_credentials(self):
        self.check(("HFTOKEN", "OPENAI_APIKEY", "GH_PAT", "HF_AUTH", "MY_PASSPHRASE", "GITHUB_PATS", "HF_APIKEYS",
                    "WANDBKEY", "hftoken", "HF_TOKEN", "OPENAI_API_KEY"),
                   ("TOKENIZERS_PARALLELISM", "HF_HOME", "KEYBOARD", "MONKEY_PATCH", "PYTHONPATH", "AUTHOR_NAME",
                    "PATH_PREFIX", "SECRETARY_NAME"), word="credential")

    def test_reserved(self):
        """Round 5 (EXT-92): the EXT-72 deny-list became the adapter's allow-list; every name below but
        TORCH_CUDA_ARCH_LIST (in the synthetic adapter's list) is refused."""
        self.check(("NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "DOCKER_HOST", "DOCKER_CONFIG",
                    "LD_PRELOAD", "LD_LIBRARY_PATH", "ld_preload", "CUDA_VISIBLE_DEVICES", "LD_BIND_NOW_X", "NVIDIAX"),
                   ("TORCH_CUDA_ARCH_LIST",))


# ---------------------------------------------------------------------------
# EXT-73 (L3): engine data directories from the daemon configurations, and their ancestors
# ---------------------------------------------------------------------------

class TestEngineData(unittest.TestCase):

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-r3-eng-"))
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_data_root_read(self):
        from gatedtrain.ext.launch import ENGINE_DATA_DIRS, engine_data_dirs
        a, b, c = self.root / "a.json", self.root / "b.json", self.root / "c.json"
        a.write_text(json.dumps({"data-root": "/srv/engine/data", "hosts": ["unix:///x"]}), encoding="utf-8")
        b.write_text("{not json", encoding="utf-8")
        c.write_text(json.dumps({"data-root": "relative"}), encoding="utf-8")
        got = engine_data_dirs((str(a), str(b), str(c), str(self.root / "missing.json")))
        self.assertEqual(got, ENGINE_DATA_DIRS + ("/srv/engine/data",))

    def test_host_daemon(self):
        """The second daemon's configuration of this host, built in or listed in the operator's host file
        (`extra_daemon_configs` in the default host file, not the suite's fixture), names an engine data
        directory."""
        from gatedtrain.ext.launch import DAEMON_CONFIGS, engine_data_dirs, host_file
        extra = tuple(host_file({}).get("extra_daemon_configs", ()))      # {}: the default host file path
        candidates = [c for c in DAEMON_CONFIGS + extra if c != "/etc/docker/daemon.json"]
        roots = []
        for config in candidates:
            try:
                roots.append(json.loads(Path(config).read_text(encoding="utf-8")).get("data-root"))
            except (OSError, ValueError):
                continue
        if not roots:
            self.skipTest(f"no second-daemon configuration is readable here ({', '.join(candidates)}); list one "
                          "in the host file's extra_daemon_configs")
        for root in roots:
            self.assertIn(root, engine_data_dirs(DAEMON_CONFIGS + extra))

    def test_refused(self):
        data = self.root / "engine" / "data"
        (data / "volumes").mkdir(parents=True)
        req = unit_request(self.root)
        cfg = dataclasses.replace(unit_config(req, engine_dirs=("/var/lib/docker", str(data))), input_roots=("/",))
        for host in (str(data), str(data / "volumes"), str(self.root / "engine"), "/var/lib"):
            with self.subTest(host=host):
                r = copy.deepcopy(req)
                r["mounts"].append({"host": host, "container": "/x", "mode": "ro"})
                self.assertIn("mounts[5].host", fields(r, config=cfg))

    def test_probe_host_defaults(self):
        """Probe p_iso3: /var/lib and the configured daemons' data-root directories and their parents, as the running
        host has them (`launch.DAEMON_CONFIGS`)."""
        from gatedtrain.ext.launch import ENGINE_DATA_DIRS, HostConfig, engine_data_dirs
        out = self.root / "out" / "run"
        out.mkdir(parents=True)
        req = unit_request(self.root)
        req.update(host_work=HOST_WORK, output_dir=str(out),
                   mounts=[{"host": str(out), "container": "/out", "mode": "rw"}])
        roots = [d for d in engine_data_dirs() if d not in ENGINE_DATA_DIRS]
        for host in ("/var/lib", *roots, *(str(Path(d).parent) for d in roots)):
            if not Path(host).exists():
                continue
            with self.subTest(host=host):
                r = copy.deepcopy(req)
                r["mounts"].append({"host": host, "container": "/x", "mode": "ro"})
                self.assertIn("mounts[1].host", fields(r, config=HostConfig(output_root=str(self.root / "out"))))


# ---------------------------------------------------------------------------
# EXT-74 (L4): 32-bit ids, distinct container paths
# ---------------------------------------------------------------------------

class TestIdsAndContainerPaths(unittest.TestCase):

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-r3-ids-"))
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_ids(self):
        for user in ("4294967296:1000", "1000:4294968280", "2147483648:1000", "1000:2147483648"):
            with self.subTest(user=user):
                req = unit_request(self.root)
                req["user"] = user
                self.assertIn("user", fields(req, config=unit_config(req, primary_gid=lambda u: 1000)))

    def test_duplicate_container_paths(self):
        cases = {"two mounts at /out": ("/out", "mounts[5].container"),
                 "/out/ is /out": ("/out/", "mounts[5].container"),
                 "a mount over the tmpfs": ("/tmp", "mounts[5].container")}
        for name, (container, field) in cases.items():
            with self.subTest(case=name):
                req = unit_request(self.root)
                req["mounts"].append({"host": req["mounts"][0]["host"], "container": container, "mode": "ro"})
                self.assertIn(field, fields(req))
        self.assertEqual(fields(unit_request(self.root)), [])


# ---------------------------------------------------------------------------
# EXT-75 (L5): an OSError while writing is a refusal; other exceptions are bugs and propagate
# ---------------------------------------------------------------------------

class TestWriteFailures(CollectCase):

    def unwritable(self, name: str) -> Path:
        d = self.tmp / name
        d.mkdir()
        os.chmod(d, 0o500)
        self.addCleanup(os.chmod, d, 0o700)
        return d

    def write_refused(self, proc, field: str):
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertIn(("EXT-WRITE", field), [(e["rule"], e["field"]) for e in parse_errors(proc.stderr)])
        self.assertNotIn("Traceback", proc.stderr)

    @unittest.skipIf(os.geteuid() == 0, "root writes into a 0500 directory")
    def test_probe_runs_root_unwritable(self):
        """Review probe test_b: a 0500 runs root gave a traceback and exit 1."""
        rr = self.unwritable("ro-runs")
        self.write_refused(self.sc.collect(rr), "--runs-root")
        self.assertEqual(list(rr.iterdir()), [])

    @unittest.skipIf(os.geteuid() == 0, "root writes into a 0500 directory")
    def test_merge_out_parent_unwritable(self):
        srcs = [build_arm(self.tmp / "m-src" / f"s{s}", self.sc.items, baseline_score, model=self.sc.base_model,
                          item_set=self.sc.item_set_hash, seeds=[s]) for s in SEEDS]
        parent = self.unwritable("m-ro")
        proc = run_cli(["ext-merge-eval", "--registration", str(self.sc.paths["registration"]), "--out",
                        str(parent / "sub" / "merged"), *map(str, srcs)], cwd=self.sc.root)
        self.write_refused(proc, "--out")

    @unittest.skipIf(os.geteuid() == 0, "root writes into a 0500 directory")
    def test_from_nel_out_parent_unwritable(self):
        runs = nel_runs(self.tmp / "n-src", self.sc.rows, score_fn=scorer_candidate, model="cand")
        parent = self.unwritable("n-ro")
        proc = convert(self.sc, parent / "arm", runs, checkpoint=self.sc.final_cut)
        self.write_refused(proc, "--out")

    @unittest.skipIf(os.geteuid() == 0, "root writes into a 0500 directory")
    def test_launch_out_record_unwritable(self):
        parent = self.unwritable("l-ro")
        sub = self.tmp / "l-case"
        sub.mkdir()
        proc, out = launch(sub, golden_request(), out=parent / "launch.json")
        self.write_refused(proc, "--out-record")
        self.assertFalse(out.exists())

    def test_programming_error_propagates(self):
        """A non-OSError during the writes is a bug: traceback, never EXT-WRITE; the partial run is removed."""
        inject = ("import sys, importlib\n"
                  "mod = importlib.import_module(sys.argv[1]); name = sys.argv[2]\n"
                  "def boom(*a, **k):\n"
                  "    raise RuntimeError('injected bug')\n"
                  "setattr(mod, name, boom)\n"
                  "from gatedtrain.testing.synthetic import install; install()\n"
                  "from gatedtrain.cli import main\n"
                  "sys.exit(main(sys.argv[3:]))\n")
        sc, runs = self.sc, self.runs()
        args = TestOutputDirCollect.args(self, runs)
        proc = run_injected("gatedtrain.ext.collect", "write_lineage", args, sc.root, env=sc.env(), inject=inject)
        self.assertEqual(proc.returncode, 1, describe(proc))
        self.assertIn("RuntimeError: injected bug", proc.stderr)
        self.assertNotIn("EXT-WRITE", proc.stderr)
        self.assertTrue(no_run_dir(runs))


# ---------------------------------------------------------------------------
# EXT-76 (L6): merged records in a golden order (mutant M15)
# ---------------------------------------------------------------------------

class TestMergeOrder(unittest.TestCase):

    def test_golden_order(self):
        """Registered seeds [11, 12]; the first source (seed 12) lists items b, a, c; the second (seed 11) c, a, b.
        Expected, written by hand: the first source's item order, then the repeat of the registered seed order."""
        from gatedtrain.ext.arm import merge_records

        def recs(seed, items):
            return [{"item": i, "repeat": 0, "seed": seed, "score": 1.0, "error": None} for i in items]
        merged = merge_records([(recs(12, "bac"), [12]), (recs(11, "cab"), [11])], [11, 12])
        self.assertEqual([(r["item"], r["repeat"], r["seed"]) for r in merged],
                         [("b", 0, 11), ("b", 1, 12), ("a", 0, 11), ("a", 1, 12), ("c", 0, 11), ("c", 1, 12)])


# ---------------------------------------------------------------------------
# EXT-77 (L7): every path lists its chain of edges
# ---------------------------------------------------------------------------

class TestImpactChain(unittest.TestCase):
    S, SNAP = hardening.TestImpactMerge.S, hardening.TestImpactMerge.SNAP
    setUp = hardening.TestImpactMerge.setUp
    mk = hardening.TestImpactMerge.mk
    query = hardening.TestImpactMerge.query

    def test_three_hop_chain(self):
        """X consumed S (`stage`); Y starts from X's model, Z from Y's; neither Y nor Z consumed S itself (their
        ledgers are empty at `iteration`). Z's only path lists X's data edge and both model edges."""
        self.mk("X", ckpts=[10], used_model="m-base", prec="stage")
        self.mk("Y", ckpts=[10], used_model="m-X-10", prec="iteration")
        self.mk("Z", ckpts=[7], used_model="m-Y-10", prec="iteration")
        got = self.query()
        self.assertEqual(set(got), {("X", 10), ("Y", 10), ("Z", 7)})
        z = got[("Z", 7)]
        self.assertEqual((z["precision"], z["via_model_hash"]), ("stage", "m-Y-10"))
        self.assertEqual(len(z["paths"]), 1)
        self.assertEqual(z["paths"][0]["chain"], [
            {"edge": "data", "run_id": "X", "stage_id": "rl", "precision": "stage", "derivation": None,
             "first_step": None},
            {"edge": "model", "model_hash": "m-X-10", "checkpoint_hash": "c-X-10", "run_id": "Y", "stage_id": "rl",
             "precision": None},
            {"edge": "model", "model_hash": "m-Y-10", "checkpoint_hash": "c-Y-10", "run_id": "Z", "stage_id": "rl",
             "precision": None}])

    def test_direct_path_chain(self):
        self.mk("X", ckpts=[10], used_model="m-base", prec="iteration", ledger_steps=[5])
        x = self.query()[("X", 10)]
        self.assertEqual(x["paths"][0]["chain"], [{"edge": "data", "run_id": "X", "stage_id": "rl",
                                                   "precision": "iteration", "derivation": "observed",
                                                   "first_step": 5}])

    def test_every_path_has_a_chain(self):
        self.mk("X", ckpts=[10], used_model="m-base", prec="stage")
        self.mk("Y", ckpts=[10, 50], used_model="m-X-10", prec="iteration", ledger_steps=[50])
        self.mk("Z", ckpts=[7], used_model="m-Y-50", prec="iteration", ledger_steps=[3])
        for key, r in self.query().items():
            for p in r["paths"]:
                with self.subTest(cut=key, via=p["via_model_hash"]):
                    chain = p["chain"]
                    self.assertEqual(chain[0]["edge"], "data")
                    self.assertTrue(all(e["edge"] == "model" and e["precision"] is None for e in chain[1:]))
                    if p["via_model_hash"] is None:
                        self.assertEqual(len(chain), 1)
                    else:
                        self.assertGreaterEqual(len(chain), 2)
                        self.assertEqual(chain[-1]["model_hash"], p["via_model_hash"])
                        self.assertEqual((chain[-1]["run_id"], chain[-1]["stage_id"]), (r["run_id"], r["stage_id"]))


# ---------------------------------------------------------------------------
# EXT-78 (L8): samples.jsonl must log the reconstructed rows
# ---------------------------------------------------------------------------

class SyntheticCase(unittest.TestCase):
    """Training shards of given row counts and a stage output whose run_args.json records them (the synthetic
    adapter's run layout)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-syn-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def stage(self, rows_per_shard: list[int], *, steps=(5, 10), mutate=None) -> tuple[Path, Path]:
        rows = gsm8k_rows("train")
        train = self.tmp / "train"
        train.mkdir()
        start = 0
        for i, n in enumerate(rows_per_shard):
            build_train(self.tmp / "part", rows[start:start + n], rows_per_shard=n)
            (self.tmp / "part" / "shard-00000.jsonl").rename(train / f"shard-{i:05d}.jsonl")
            start += n
        out = build_stage_out(self.tmp / "out", steps=steps, max_steps=max(steps), train=train, args_mutate=mutate)
        return out, train

    def consumption(self, out: Path, train: Path, cut_steps=None):
        from gatedtrain.ext import registry
        ad = registry.adapter(ADAPTER)
        complete, _ = ad.find_cuts(out)
        if cut_steps is not None:
            complete = [(s, p) for s, p in complete if s in cut_steps]
        return ad.consumption(out, train, complete, run_id="run-1")


class TestSamplesCrossCheck(SyntheticCase):

    def lines(self, out: Path) -> list[dict]:
        return [json.loads(l) for l in (out / "samples.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    def write(self, out: Path, lines: list[dict]) -> None:
        (out / "samples.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines), encoding="utf-8")

    def test_agreeing_log_keeps_iteration(self):
        out, train = self.stage([10, 10, 10, 10])
        cons = self.consumption(out, train)
        self.assertEqual(cons["precision"], "iteration")
        self.assertIn("samples.jsonl logs the same rows", cons["reason"])

    def test_wrapped_epoch_agrees(self):
        out, train = self.stage([10, 10, 5])
        self.assertEqual(self.consumption(out, train)["precision"], "iteration")

    def test_disagreeing_log_degrades(self):
        cases = {
            "a row moved": lambda ls: [dict(l, row=39) if (l["step"], l["position"]) == (3, 0) else l for l in ls],
            "a step missing": lambda ls: [l for l in ls if l["step"] != 4],
            "an extra row": lambda ls: ls + [dict(ls[0], row=30)],
            "steps shifted by one (generated_at_global_step as step)":
                lambda ls: [dict(l, step=l["generated_at_global_step"]) for l in ls],
            "not JSON": None,
            "a boolean row": lambda ls: [dict(ls[0], row=True)] + ls[1:],
            "a negative step": lambda ls: ls + [dict(ls[0], step=-1)],
        }
        for name, edit in cases.items():
            with self.subTest(case=name):
                sub = self.tmp / name.replace(" ", "_").replace("(", "").replace(")", "")
                sub.mkdir()
                self.tmp, base = sub, self.tmp
                try:
                    out, train = self.stage([10, 10, 10, 10])
                finally:
                    self.tmp = base
                if edit is None:
                    (out / "samples.jsonl").write_text("{\n", encoding="utf-8")
                else:
                    self.write(out, edit(self.lines(out)))
                cons = self.consumption(out, train)
                self.assertEqual((cons["precision"], list(cons["rows"])), ("stage", []))
                self.assertIn("samples.jsonl", cons["reason"])
                if name in ("a boolean row", "a negative step"):
                    self.assertIn("no integer `step` and `row`", cons["reason"])

    def test_steps_after_the_last_cut_ignored(self):
        """Lines of steps after the last complete cut (orphaned) are not compared."""
        out, train = self.stage([10, 10, 10, 10])
        lines = [l for l in self.lines(out) if l["step"] <= 5] + [dict(self.lines(out)[0], step=8, row=39)]
        self.write(out, lines)
        self.assertEqual(self.consumption(out, train, cut_steps={5})["precision"], "iteration")

    def test_absent_log_keeps_iteration(self):
        out, train = self.stage([10, 10, 10, 10])
        (out / "samples.jsonl").unlink()
        self.assertEqual(self.consumption(out, train)["precision"], "iteration")


class TestSamplesCrossCheckCollect(CollectCase):

    def test_disagreeing_log_reported_stage(self):
        p = self.sc.stage_out / "samples.jsonl"
        old = p.read_bytes()
        lines = [json.loads(l) for l in old.decode("utf-8").splitlines() if l.strip()]
        p.write_text("".join(json.dumps(dict(l, row=(l["row"] + 1) % 40)) + "\n" for l in lines), encoding="utf-8")
        self.addCleanup(p.write_bytes, old)
        rep = self.passes()
        self.assertEqual(rep["consumption_precision"], "stage")


class TestHostFileDaemonLists(unittest.TestCase):
    """The host file's optional `extra_daemon_configs` and `extra_daemon_sockets` (lists of absolute paths) add
    daemon configurations whose `data-root` is refused and sockets that are never mounted; they enter
    `host_config_sha256` only when non-empty, so a host file without them hashes as before."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-r3-hostfile-"))
        self.addCleanup(shutil.rmtree, self.root, True)

    def config(self, extra: str = ""):
        from gatedtrain.ext.launch import ENV_HOST_CONFIG, host_config
        path = self.root / "host.toml"
        path.write_text(f'host_work = "{self.root}/work"\noutput_root = "{self.root}/out"\n{extra}', encoding="utf-8")
        return host_config({ENV_HOST_CONFIG: str(path)})

    def refused(self, extra: str, key: str):
        from gatedtrain.errors import Refused
        with self.assertRaises(Refused) as ctx:
            self.config(extra)
        self.assertEqual([(p.rule, p.field.rsplit(".", 1)[-1]) for p in ctx.exception.problems], [("EXT-HOST", key)])
        self.assertIn("list of absolute paths", ctx.exception.problems[0].message)

    def test_absent(self):
        from gatedtrain.ext.launch import host_config_sha256 as launch_hash
        from gatedtrain.testing.ext import host_config_sha256
        cfg = self.config()
        self.assertEqual((cfg.extra_daemon_configs, cfg.extra_daemon_sockets), ((), ()))
        self.assertEqual(launch_hash(cfg), host_config_sha256(self.root / "work", self.root / "out", []))

    def test_valid_lists(self):
        from gatedtrain.ext.launch import _socket_problem, engine_data_dirs, host_config_sha256 as launch_hash
        from gatedtrain.testing.ext import host_config_sha256
        daemon = self.root / "etc" / "daemon.json"
        daemon.parent.mkdir()
        daemon.write_text(json.dumps({"data-root": "/srv/second-engine"}), encoding="utf-8")
        sock = "/srv/second-daemon/engine.socket"
        cfg = self.config(f'extra_daemon_configs = ["{daemon}"]\nextra_daemon_sockets = ["{sock}"]\n')
        self.assertEqual(cfg.extra_daemon_configs, (str(daemon),))
        self.assertEqual(cfg.extra_daemon_sockets, (sock,))
        self.assertIn("/srv/second-engine", engine_data_dirs(cfg.extra_daemon_configs))
        self.assertIsNone(_socket_problem(sock, sock))
        self.assertIn("Docker socket", _socket_problem(sock, sock, cfg.extra_daemon_sockets))
        self.assertIn("Docker socket", _socket_problem("/srv/second-daemon", "/srv/second-daemon",
                                                       cfg.extra_daemon_sockets))
        hashed = launch_hash(cfg)
        self.assertEqual(hashed, host_config_sha256(self.root / "work", self.root / "out", [],
                                                    extra_daemon_configs=[daemon], extra_daemon_sockets=[sock]))
        self.assertNotEqual(hashed, launch_hash(self.config()))

    def test_engine_data_refused_in_a_request(self):
        daemon = self.root / "daemon.json"
        data = self.root / "engine-data"
        (data / "volumes").mkdir(parents=True)
        daemon.write_text(json.dumps({"data-root": str(data)}), encoding="utf-8")
        cfg = self.config(f'extra_daemon_configs = ["{daemon}"]\n')
        req = unit_request(self.root)
        base = dataclasses.replace(unit_config(req), input_roots=("/",))
        r = copy.deepcopy(req)
        r["mounts"].append({"host": str(data / "volumes"), "container": "/x", "mode": "ro"})
        self.assertNotIn("mounts[5].host", fields(r, config=base))
        listed = dataclasses.replace(base, extra_daemon_configs=cfg.extra_daemon_configs)
        self.assertIn("mounts[5].host", fields(r, config=listed))

    def test_invalid_types_refused(self):
        for key in ("extra_daemon_configs", "extra_daemon_sockets"):
            for value in ('"/etc/one.json"', '["relative.json"]', "[1]", "{a = 1}"):
                with self.subTest(key=key, value=value):
                    self.refused(f"{key} = {value}\n", key)


if __name__ == "__main__":
    unittest.main()
