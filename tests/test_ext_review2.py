"""Launch isolation, arm provenance and error handling (EXT-51 to EXT-62).

The multi-path impact case (EXT-58) is in test_ext_hardening.TestImpactMerge.
"""

from __future__ import annotations

import copy
import glob
import json
import os
import shutil
import socket
import subprocess
import tempfile
import unittest
from pathlib import Path

from _ext import (EVALUATOR_ID, LABEL, container_uid, SEEDS, baseline_score, build_arm, gpu_step_script, guard_log, launch_label,
                  merge, write_canonical)
from _support import TIERB_ROOT, TIERB_ROOT_ENV, EXIT_OK, EXIT_REFUSED, VENV_PYTHON, cli_env, describe, parse_errors
from test_ext_from_nel import convert, nel_runs, scorer_candidate
from test_ext_hardening import CollectCase, LaunchCase, no_run_dir
from test_ext_launch import golden_request, launch, rooted

DOCKER_SOCKETS = ("/run/docker.sock", "/var/run/docker.sock", "/run/docker-*.sock")


def host_socket_gids() -> set[int]:
    gids = set()
    for pattern in DOCKER_SOCKETS:
        for p in glob.glob(pattern):
            try:
                gids.add(os.stat(p).st_gid)
            except OSError:
                pass
    return gids


def unit_request(root: Path) -> dict:
    """The golden request under `root` (directories created), as a plain container (no lock file to read)."""
    req = rooted(golden_request(), root)
    req["runtime"] = {"kind": "container", "image": "sha256:" + "ab" * 32}
    return req


def unit_config(req: dict, **kw):
    """The host layout of a `unit_request` (EXT-63, EXT-66): host work its host_work, output root
    `<data>/gatedtrain-mm/out`, input root `<data>`, `<data>` the parent of host_work; the synthetic adapter's environment
    allow-list (EXT-92), no scan skip list and no interpreter tree (EXT-93); its container paths with `<data>`'s parent
    as the identity mounts' `/` (EXT-112) and its path-valued environment names (EXT-114)."""
    from gatedtrain.ext.launch import HostConfig
    from gatedtrain.ext import registry
    data = Path(req["host_work"]).parent
    kw.setdefault("env_names", registry.env_names("synthetic"))           # EXT-92: the adapter's allow-list
    kw.setdefault("container_paths", registry.container_paths("synthetic"))   # EXT-112: the adapter's container paths
    kw.setdefault("mirror_root", str(data.parent))                      # EXT-112: `<root>` stands for `/`
    kw.setdefault("env_paths", registry.env_paths("synthetic"))           # EXT-114: path-valued names
    return HostConfig(host_work=req["host_work"], output_root=str(data / "gatedtrain-mm" / "out"),
                      input_roots=(str(data),), **kw)


def problems(req: dict, *, gids=frozenset(), kind: str = "request", config=None) -> list[tuple[str, str]]:
    from gatedtrain.ext.launch import isolation_problems
    return isolation_problems(req, kind=kind, socket_gids=gids, config=config or unit_config(req))


# ---------------------------------------------------------------------------
# EXT-51 (H-1): no host path under /proc, /sys, /dev; every mount source resolves; no docker-group gid
# ---------------------------------------------------------------------------

class TestHostPathsLaunch(LaunchCase):

    def cases(self) -> dict[str, str]:
        link = self.tmp / "hostroot-link"
        link.symlink_to("/proc/1/root")
        sys_link = self.tmp / "sys-link"                 # readable: resolves (strict) to /sys
        sys_link.symlink_to("/sys")
        shm_link = self.tmp / "shm-link"
        shm_link.symlink_to("/dev/shm")
        return {"/proc/1/root": "/proc/1/root", "/proc/1/cwd": "/proc/1/cwd", "/proc/1/root/run": "/proc/1/root/run",
                "/proc/self/root": "/proc/self/root", "/proc": "/proc", "/sys": "/sys",
                "/sys/fs/cgroup": "/sys/fs/cgroup",
                "/proc/self/cwd (resolves outside /proc for this user)": "/proc/self/cwd",
                "//proc/self/cwd": "//proc/self/cwd",
                "/dev": "/dev", "/dev/shm": "/dev/shm", "//proc/1/root": "//proc/1/root",
                "symlink to /proc/1/root": str(link), "symlink to /sys": str(sys_link),
                "symlink to /dev/shm": str(shm_link), "missing path": str(self.tmp / "no-such-dir"),
                "dangling symlink": self.dangling()}

    def dangling(self) -> str:
        link = self.tmp / "dangling"
        link.symlink_to(self.tmp / "gone")
        return str(link)

    def test_launch_refuses(self):
        """The request root lies outside the working directory, so `/proc/self/cwd` (the working directory for
        this user) is refused for being under /proc, not for being a parent of the output."""
        for name, host in self.cases().items():
            with self.subTest(case=name):
                req = golden_request()
                req["mounts"].insert(0, {"host": host, "container": "/hostroot", "mode": "ro"})
                proc, out = launch(self.sub(), req, root=self.tmp / "host")
                self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
                fields = [e["field"] for e in parse_errors(proc.stderr) if e["rule"] == "EXT-LAUNCH"]
                self.assertIn("--request.mounts[0].host", fields)
                self.assertFalse(out.exists())

    def test_gid_zero_refused(self):
        req = golden_request()
        req["user"] = f"{container_uid()}:0"
        self.launch_refused(req, "user")

    @unittest.skipUnless(host_socket_gids(), "no Docker daemon socket on this host")
    def test_docker_socket_gid_refused(self):
        for gid in sorted(host_socket_gids()):
            with self.subTest(gid=gid):
                req = golden_request()
                req["user"] = f"{container_uid()}:{gid}"
                self.launch_refused(req, "user")


class TestHostPathsCollect(CollectCase):

    def with_mount(self, host: str, **kw) -> Path:
        rec = self.sc.launch_record(hashed_label=False, **kw)
        rec["mounts"] = rec["mounts"] + [{"host": host, "container": "/hostroot", "mode": "ro"}]
        rec["label"] = launch_label(rec)
        guard_log(self.sc.host_work, rec["label"])
        self.addCleanup(guard_log, self.sc.host_work, self.sc.label)
        return write_canonical(self.path("launch").with_suffix(".json"), rec)

    def test_proc_paths_refused(self):
        for host in ("/proc/1/root", "/proc/1/cwd", "/sys", "/dev", str(self.tmp / "missing")):
            with self.subTest(host=host):
                self.refused("EXT-LAUNCH", "--launch-record.mounts[4].host", launch=self.with_mount(host))

    @unittest.skipUnless(host_socket_gids(), "no Docker daemon socket on this host")
    def test_probe_p4_proc_root_and_docker_gid(self):
        """Review probe p4: /proc/1/root read-only and the docker group's gid (984 here): both refused."""
        gid = sorted(host_socket_gids())[0]
        path = self.with_mount("/proc/1/root", user=f"{container_uid()}:{gid}")
        self.refused("EXT-LAUNCH", "--launch-record.mounts[4].host", launch=path)
        self.refused("EXT-LAUNCH", "--launch-record.user", launch=path)


class TestSocketGidLookup(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="s", dir="/tmp"))      # AF_UNIX paths are short
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-r2-"))
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_injected_gid_refused(self):
        req = unit_request(self.root)
        self.assertEqual(problems(req), [])
        req["user"] = f"{container_uid()}:4242"
        cfg = unit_config(req, primary_gid=lambda uid: 4242)          # EXT-66: 4242 is the uid's primary group
        self.assertIn("user", [f for f, _ in problems(req, gids={4242}, config=cfg)])
        self.assertEqual(problems(req, gids={984}, config=cfg), [])

    def test_record_gid_refused(self):
        rec = unit_request(self.root)
        rec["user"] = f"{container_uid()}:4242"
        cfg = unit_config(rec, primary_gid=lambda uid: 4242)
        self.assertIn("user", [f for f, _ in problems(rec, gids={4242}, kind="record", config=cfg)])

    def test_lookup_reads_existing_sockets(self):
        from gatedtrain.ext.launch import docker_socket_gids
        sock = self.tmp / "d-gatedtrain.sock"
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(str(sock))
        self.addCleanup(s.close)
        got = docker_socket_gids(paths=(str(self.tmp / "missing.sock"), str(self.tmp / "d-*.sock")))
        self.assertEqual(got, {os.stat(sock).st_gid})

    def test_default_lookup_is_the_host_sockets(self):
        from gatedtrain.ext.launch import docker_socket_gids
        self.assertEqual(docker_socket_gids(), host_socket_gids())


# ---------------------------------------------------------------------------
# EXT-52 (M-a): a merged arm's items.jsonl is re-derived from its sources
# ---------------------------------------------------------------------------

class TestRederivedArm(CollectCase):

    def merged(self, name: str) -> Path:
        srcs = [build_arm(self.tmp / name / f"s{s}", self.sc.items, baseline_score, model=self.sc.base_model,
                          item_set=self.sc.item_set_hash, seeds=[s]) for s in SEEDS]
        out = self.tmp / name / "merged"
        proc = merge(self.sc.paths["registration"], out, srcs, cwd=self.sc.root)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        return out

    @staticmethod
    def rewrite(arm: Path, fn) -> None:
        recs = [json.loads(l) for l in (arm / "items.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        recs = fn(recs)
        (arm / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")

    def test_probe_p1_scores_edited(self):
        """Review probe p1: every score of the merged arm set to 1.0; sources and merge.json unchanged."""
        out = self.merged("p1")
        self.rewrite(out, lambda recs: [dict(r, score=1.0) if r["score"] is not None else r for r in recs])
        self.refused("EXT-ARM", "--eval-baseline", baseline=out)

    def test_lines_reordered(self):
        out = self.merged("order")
        self.rewrite(out, lambda recs: list(reversed(recs)))
        self.refused("EXT-ARM", "--eval-baseline", baseline=out)

    def test_untouched_merged_arm_passes(self):
        self.assertEqual(self.passes(baseline=self.merged("ok"))["status"], "PASS")

    def test_from_nel_arm_scores_edited(self):
        runs = nel_runs(self.tmp / "nel-edit", self.sc.rows, score_fn=scorer_candidate, model="cand")
        arm = self.tmp / "nel-edit-arm"
        proc = convert(self.sc, arm, runs, checkpoint=self.sc.final_cut)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertEqual(self.passes(candidate=arm)["status"], "PASS")
        self.rewrite(arm, lambda recs: [dict(r, score=1.0) if r["score"] is not None else r for r in recs])
        self.refused("EXT-ARM", "--eval-candidate", candidate=arm)


# ---------------------------------------------------------------------------
# EXT-53 (M-b): the label's last terminal event is `END rc=0`
# ---------------------------------------------------------------------------

class TestLastTerminalEvent(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-r2-gpu-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def seconds(self, lines: list[str]):
        from gatedtrain.ext.launch import gpu_seconds
        p = self.tmp / "gpu-guard.log"
        p.write_text("".join(l + "\n" for l in lines), encoding="utf-8")
        return gpu_seconds(p, LABEL)

    def test_success_then_preempted_refused(self):
        with self.assertRaises(ValueError) as cm:
            self.seconds([f"2026-11-11T09:00:00Z START {LABEL} attempt=1",
                          f"2026-11-11T09:40:00Z END {LABEL} attempt=1 rc=0",
                          f"2026-11-11T10:00:00Z START {LABEL} attempt=1",
                          f"2026-11-11T10:05:00Z PREEMPTED {LABEL} attempt=1 rc=137"])
        self.assertIn("PREEMPTED", str(cm.exception))

    def test_preempted_then_success_accepted(self):
        self.assertEqual(self.seconds([f"2026-11-11T09:00:00Z START {LABEL} attempt=1",
                                       f"2026-11-11T09:05:00Z PREEMPTED {LABEL} attempt=1 rc=137",
                                       f"2026-11-11T09:10:00Z START {LABEL} attempt=2",
                                       f"2026-11-11T09:20:00Z END {LABEL} attempt=2 rc=0"]), 900)


class TestLastTerminalEventCollect(CollectCase):

    def test_probe_p3_success_then_preempted_relaunch(self):
        lab = self.sc.label
        guard_log(self.sc.host_work, lab, lines=[
            f"2026-11-11T09:00:00Z START {lab} attempt=1", f"2026-11-11T09:40:00Z END {lab} attempt=1 rc=0",
            f"2026-11-11T10:00:00Z START {lab} attempt=1", f"2026-11-11T10:05:00Z PREEMPTED {lab} attempt=1 rc=137"])
        self.addCleanup(guard_log, self.sc.host_work, lab)
        self.refused("EXT-GPU", "--launch-record.label")


# ---------------------------------------------------------------------------
# EXT-54 (L-c): whole-string matches and ASCII digits
# ---------------------------------------------------------------------------

class TestStrictGrammar(unittest.TestCase):

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-r2-"))
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_request_values_refused(self):
        cases = {
            "image with a trailing newline": (lambda r: r["runtime"].update(image="sha256:" + "ab" * 32 + "\n"),
                                              "runtime.image"),
            "label with a trailing newline": (lambda r: r.update(label="probe\n"), "label"),
            "user with a trailing newline": (lambda r: r.update(user="1000:1000\n"), "user"),
            "user with Arabic-Indic digits": (lambda r: r.update(user="١٠٠٠:١٠٠٠"),
                                              "user"),
            "uid 0 in Arabic-Indic digits": (lambda r: r.update(user="٠:1000"), "user"),
            "network with a trailing newline": (lambda r: r.update(network="testnet\n", network_reason="x"), "network"),
            "env name with a trailing newline": (lambda r: r["env"].update({"FOO\n": "1"}), "env.FOO\n"),
        }
        for name, (mutate, field) in cases.items():
            with self.subTest(case=name):
                req = unit_request(self.root)
                mutate(req)
                self.assertIn(field, [f for f, _ in problems(req)])

    def test_record_values_refused(self):
        from gatedtrain.ext.launch import label_problem
        rec = unit_request(self.root)
        rec.update(env_names=["HF_HUB_OFFLINE", "FOO\n"])
        self.assertIn("env_names", [f for f, _ in problems(rec, kind="record")])
        rec = dict(unit_request(self.root), label="probe")
        rec["label"] = launch_label(rec) + "\n"
        self.assertIsNotNone(label_problem(rec))

    def test_rc_with_unicode_digits_refused(self):
        from gatedtrain.ext.launch import gpu_seconds
        tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-r2-rc-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        p = tmp / "gpu-guard.log"
        p.write_text(f"2026-11-11T09:00:00Z START {LABEL} attempt=1\n"
                     f"2026-11-11T09:30:00Z END {LABEL} attempt=1 rc=٠\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            gpu_seconds(p, LABEL)


# ---------------------------------------------------------------------------
# EXT-55 (maintainers' decision): credential words match whole name components
# ---------------------------------------------------------------------------

class TestCredentialComponents(unittest.TestCase):

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-r2-"))
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_names(self):
        refused = ("HF_TOKEN", "TOKEN", "GH_TOKENS", "API_KEY", "OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY",
                   "AWS_ACCESS_KEY_ID", "DB_PASSWORD", "MY_SECRET", "GOOGLE_CREDENTIALS", "hf_token", "WANDB_KEY")
        allowed = ("TOKENIZERS_PARALLELISM", "HF_HOME", "KEYBOARD", "MONKEY_PATCH", "SECRETARY_NAME")
        for name in refused + allowed:
            with self.subTest(name=name):
                req = unit_request(self.root)
                req["env"][name] = "x"
                # Round 5 (EXT-92): names outside the adapter's allow-list are refused too; this test pins the
                # credential-word matcher, so it reads the credential refusal.
                msgs = [m for f, m in problems(req) if f == f"env.{name}"]
                self.assertEqual(any("credential" in m for m in msgs), name in refused, msgs)

    def test_golden_request_keeps_tokenizers_parallelism(self):
        self.assertEqual(golden_request()["env"]["TOKENIZERS_PARALLELISM"], "false")


# ---------------------------------------------------------------------------
# EXT-56 (L-d): `//` prefixes and symbolic links are judged on the resolved path (MM-34 mount rules)
# ---------------------------------------------------------------------------

class TestMountNormalisation(LaunchCase):

    def test_launch_refuses(self):
        host = self.tmp / "host"
        rooted(golden_request(), host)                     # create the golden directories
        work_link = self.tmp / "work-link"
        work_link.symlink_to(host / "data" / "gatedtrain-work")
        ext_link = self.tmp / "ext-link"
        ext_link.symlink_to(host / "data" / "gatedtrain-mm" / "out" / "a1")          # EXT-63: the output's parent
        cases = {
            "host work behind //": "/" + str(host / "data" / "gatedtrain-work"),
            "parent of the output behind //": "/" + str(host / "data" / "gatedtrain-mm" / "out" / "a1"),
            "symlink to host work": str(work_link),
            "symlink to a parent of the output": str(ext_link),
            "//run": "//run",
            "//var/run": "//var/run",
        }
        for name, path in cases.items():
            with self.subTest(case=name):
                req = golden_request()
                req["mounts"].append({"host": path, "container": "/x", "mode": "ro"})
                proc, rec = launch(self.sub(), req, root=host)
                self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
                fields = [e["field"] for e in parse_errors(proc.stderr) if e["rule"] == "EXT-LAUNCH"]
                self.assertIn("--request.mounts[5].host", fields)


    def test_symlink_to_host_work_outside_the_output_tree(self):
        """host_work that is not an ancestor of the output: a link to it is refused as host work (resolved)."""
        host = self.tmp / "host2"
        work = self.tmp / "work2"
        work.mkdir()
        link = self.tmp / "work2-link"
        link.symlink_to(work)
        req = golden_request()
        req["host_work"] = str(work)
        req["mounts"].append({"host": str(link), "container": "/x", "mode": "ro"})
        proc, _ = launch(self.sub(), req, root=host)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        hits = [e for e in parse_errors(proc.stderr) if e["field"] == "--request.mounts[5].host"]
        self.assertTrue(any("host work" in e["message"] for e in hits), hits)


# ---------------------------------------------------------------------------
# EXT-62 (maintainers' decision after review round 2): no ancestor of host_work, no Docker data directory
# ---------------------------------------------------------------------------

REAL_HOST_WORK = str(TIERB_ROOT)   # GATEDTRAIN_TEST_TIERB_ROOT; unset: a path that never exists, so the real tests skip


class TestAncestorAndDockerData(LaunchCase):

    def refused_mount(self, host: str, *, host_work: Path | None = None):
        req = golden_request()
        if host_work is not None:
            req["host_work"] = str(host_work)
        req["mounts"].append({"host": host, "container": "/x", "mode": "ro"})
        proc, out = launch(self.sub(), req, root=self.tmp / "host")
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        fields = [e["field"] for e in parse_errors(proc.stderr) if e["rule"] == "EXT-LAUNCH"]
        self.assertIn("--request.mounts[5].host", fields)
        self.assertFalse(out.exists())

    def test_ancestor_of_host_work_refused(self):
        """host_work outside the output tree, so only the ancestor rule applies (EXT-62)."""
        work = self.tmp / "w" / "work"
        work.mkdir(parents=True)
        link = self.tmp / "w-link"
        link.symlink_to(self.tmp / "w")
        for name, host in {"parent": str(self.tmp / "w"), "parent behind //": "/" + str(self.tmp / "w"),
                           "link to the parent": str(link)}.items():
            with self.subTest(case=name):
                self.refused_mount(host, host_work=work)

    def test_docker_data_refused(self):
        for host in ("/var/lib/docker", "/var/lib/docker/volumes", "/var/lib/containerd", "//var/lib/docker"):
            with self.subTest(host=host):
                self.refused_mount(host)

    def test_golden_request_accepted(self):
        proc, out = launch(self.sub(), golden_request(), root=self.tmp / "host")
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertTrue(out.is_file())

class TestAncestorAndDockerDataCollect(CollectCase):

    def test_collect_refuses(self):
        for host in (str(self.sc.host_work.parent), "/var/lib/docker"):
            with self.subTest(host=host):
                rec = self.sc.launch_record(hashed_label=False)
                rec["mounts"] = rec["mounts"] + [{"host": host, "container": "/x", "mode": "ro"}]
                rec["label"] = launch_label(rec)
                guard_log(self.sc.host_work, rec["label"])
                self.addCleanup(guard_log, self.sc.host_work, self.sc.label)
                path = write_canonical(self.path("launch").with_suffix(".json"), rec)
                self.refused("EXT-LAUNCH", "--launch-record.mounts[4].host", launch=path)


# ---------------------------------------------------------------------------
# EXT-57 (L-b): gpu-step.sh assigns WORK and LOG once each
# ---------------------------------------------------------------------------

class TestGpuStepAssignments(CollectCase):

    def script(self, extra: str) -> Path:
        """Edit gpu-step.sh and return a launch record of that script (EXT-67 pins its SHA-256)."""
        p = gpu_step_script(self.sc.host_work)
        p.write_text(p.read_text(encoding="utf-8").replace("exit 99\n", extra + "exit 99\n"), encoding="utf-8")
        self.addCleanup(gpu_step_script, self.sc.host_work)
        rec = self.sc.launch_record()
        guard_log(self.sc.host_work, rec["label"])
        self.addCleanup(guard_log, self.sc.host_work, self.sc.label)
        return write_canonical(self.path("launch").with_suffix(".json"), rec)

    def test_second_assignment_refused(self):
        for extra in ('LOG="/tmp/elsewhere.log"\n', "export WORK=/tmp/elsewhere\n", "  LOG=/tmp/x.log\n",
                      ": ${LOG:=/tmp/x.log}\n", "true; WORK=/tmp/elsewhere\n"):
            with self.subTest(extra=extra):
                self.refused("EXT-GPU", "--launch-record.host_work", launch=self.script(extra))

    def test_uses_are_not_assignments(self):
        path = self.script('echo "$WORK $LOG" >> "$LOG"\nFLAG="$WORK/.preempted"\n')
        self.assertEqual(self.passes(launch=path)["status"], "PASS")

    def test_host_script_shape_accepted(self):
        """The real host's script (read-only) has one WORK= and one LOG= assignment."""
        from gatedtrain.ext.launch import gpu_step_assignments
        real = Path(REAL_HOST_WORK, "bin", "gpu-step.sh")
        if not real.is_file():
            self.skipTest(f"the real host's gpu-step.sh is absent (set {TIERB_ROOT_ENV})")
        got = gpu_step_assignments(real.read_text(encoding="utf-8"))
        self.assertEqual((len(got["WORK"]), len(got["LOG"])), (1, 1))


# ---------------------------------------------------------------------------
# EXT-61 (L-a): the hashed label claims only protection against accidental edits
# ---------------------------------------------------------------------------

class TestLabelWording(unittest.TestCase):

    def test_message(self):
        from gatedtrain.ext.launch import label_problem
        rec = {"label": "probe", "gpu_est_h": 1.0}
        rec["label"] = launch_label(rec)
        rec["gpu_est_h"] = 2.0
        msg = label_problem(rec)
        self.assertIn("accidental", msg)
        self.assertNotIn("was edited after ext-launch", msg)


# ---------------------------------------------------------------------------
# EXT-59 (L-f): a failed write after the checks is a refusal (exit 3), never the gate's exit 1
# ---------------------------------------------------------------------------

INJECT = ("import sys, importlib\n"
          "mod = importlib.import_module(sys.argv[1]); name = sys.argv[2]\n"
          "def boom(*a, **k):\n"
          "    raise OSError(28, 'No space left on device (injected)')\n"
          "setattr(mod, name, boom)\n"
          "from gatedtrain.testing.synthetic import install; install()\n"
          "from gatedtrain.cli import main\n"
          "sys.exit(main(sys.argv[3:]))\n")


def run_injected(module: str, name: str, args: list[str], cwd: Path, env: dict | None = None,
                 inject: str = INJECT) -> subprocess.CompletedProcess:
    return subprocess.run([str(VENV_PYTHON), "-c", inject, module, name, *args], cwd=str(cwd),
                          env={**cli_env(), **(env or {})}, capture_output=True, text=True, timeout=300)


class TestWriteFailure(CollectCase):

    def check(self, proc, rule: str = "EXT-WRITE"):
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertIn(rule, [e["rule"] for e in parse_errors(proc.stderr)], proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)

    def test_collect(self):
        sc, runs = self.sc, self.runs()
        args = ["ext-collect", str(sc.paths["spec"]), "--adapter", "synthetic", "--stage-out", str(sc.stage_out),
                "--launch-record", str(sc.launch), "--smoke-record", str(sc.smoke), "--eval-baseline",
                str(sc.baseline), "--eval-candidate", str(sc.candidate), "--runs-root", str(runs)]
        proc = run_injected("gatedtrain.ext.collect", "write_lineage", args, sc.root, env=sc.env())
        self.check(proc)
        self.assertTrue(no_run_dir(runs), "the partial run directory is removed")

    def test_merge(self):
        srcs = [build_arm(self.tmp / "wf" / f"s{s}", self.sc.items, baseline_score, model=self.sc.base_model,
                          item_set=self.sc.item_set_hash, seeds=[s]) for s in SEEDS]
        out = self.tmp / "wf" / "merged"
        proc = run_injected("gatedtrain.ext.merge", "write_canonical",
                            ["ext-merge-eval", "--registration", str(self.sc.paths["registration"]), "--out", str(out),
                             *map(str, srcs)], self.sc.root)
        self.check(proc)
        self.assertEqual(sorted(p.name for p in (self.tmp / "wf").iterdir()), ["s11", "s12"])

    def test_from_nel(self):
        runs = nel_runs(self.tmp / "wf-nel", self.sc.rows, score_fn=scorer_candidate, model="cand")
        out = self.tmp / "wf-nel-out" / "arm"
        out.parent.mkdir()
        proc = run_injected("gatedtrain.ext.from_nel", "write_canonical",
                            ["ext-from-nel", "--registration", str(self.sc.paths["registration"]), "--adapter",
                             "synthetic", "--evaluator", EVALUATOR_ID, "--items", str(self.sc.items_dir),
                             "--checkpoint", str(self.sc.final_cut), "--out", str(out), *map(str, runs)], self.sc.root)
        self.check(proc)
        self.assertEqual(list(out.parent.iterdir()), [])


# ---------------------------------------------------------------------------
# EXT-60 (mutant X2): build_command refuses a value starting with `-` before the image
# ---------------------------------------------------------------------------

class TestBuildCommandValues(unittest.TestCase):

    def record(self) -> dict:
        root = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-r2-"))
        self.addCleanup(shutil.rmtree, root, True)
        req = unit_request(root)
        return dict(req, env_names=sorted(req["env"]), gpu_used_h=0.0,
                    runtime={"kind": "container", "image": "sha256:" + "ab" * 32, "venv_lock_sha256": None})

    def test_dash_values(self):
        from gatedtrain.ext.launch import build_command
        rec = self.record()
        env = {"HF_HUB_OFFLINE": "1"}
        self.assertIn("sha256:", build_command(rec, env, est_h=1))
        cases = {
            "label": lambda r, e: r.update(label="--privileged"),
            "network": lambda r, e: r.update(network="--privileged"),
            "user": lambda r, e: r.update(user="-u0"),
            "image": lambda r, e: r["runtime"].update(image="--privileged"),
            "mount host": lambda r, e: r["mounts"][0].update(host="-v/:/h"),
            "tmpfs": lambda r, e: r.update(tmpfs=["--privileged"]),
            "empty network": lambda r, e: r.update(network=""),
            "env name": lambda r, e: e.update({"-e": "x"}),
        }
        for name, mutate in cases.items():
            with self.subTest(case=name):
                r, e = copy.deepcopy(rec), dict(env)
                mutate(r, e)
                with self.assertRaises(ValueError):
                    build_command(r, e, est_h=1)


if __name__ == "__main__":
    unittest.main()
