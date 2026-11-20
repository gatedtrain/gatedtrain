"""Launcher contract, GPU accounting and smoke records (tests/README.md section 11.18: EXT-22 to EXT-30).

Sources: MULTIMODAL-SPEC.md MM-31 to MM-36, MM-38, MM-39. The GPU log format is gpu-step.sh's
(`<HOST_WORK>/bin/gpu-step.sh`, the operator's wrapper, read 2026-11-11): one line per event,
`<UTC %FT%TZ> START <label> attempt=<n>`, `... END <label> attempt=<n> rc=<rc>`,
`... PREEMPTED <label> attempt=<n> rc=<rc>`, plus WAIT, REFUSED (no GPU time) and gpu-guard.sh's
`GUARD ...`, `PREEMPT ...` and other guard lines (ignored). The attempt counter restarts at 1 on every
gpu-step.sh invocation, so one label can carry several `attempt=1` pairs (seen in the Tier B log).
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from _ext import (ENV_HOST_WORK, ENV_INPUT_ROOTS, ENV_INTERPRETER, ENV_INTERPRETER_TREES, ENV_MIRROR_ROOT, ENV_OUTPUT_ROOT,
                  EXPORT_CONTAINER, EXT, INTERPRETER_TEXT, LABEL, SCRIPT_TEXT,
                  container_user, gpu_step_script, jcs, launch_label, smoke_record, write_bound_spec)
from _support import EXIT_OK, EXIT_REFUSED, SKEL_ROOT, VENV_PYTHON, cli_env, describe, file_sha256, parse_errors

import subprocess

GOLDEN_REQUEST = EXT / "launch-request.golden.yaml"
GOLDEN_COMMAND = EXT / "launch-command.golden.txt"
GPU_LOG = EXT / "gpu-log.fixture.md"
VENV_LOCK = EXT / "venv.lock.fixture"     # the golden request's `venv_lock: uv.lock`, resolved in the cwd


GOLDEN_PREFIX = "/data/"        # the golden request's host paths; rooted under a test directory by `rooted`
# EXT-113 (review round 6, L1): the golden request's pinned interpreter (argv[0] as written) and the tree its stub lies
# in, relative to the test root.
GOLDEN_INTERPRETER = "/data/gatedtrain-mm/synthetic-runtime/.venv/bin/python"
GOLDEN_TREE = "data/gatedtrain-mm/synthetic-runtime/.venv"


def rooted(request: dict, root: Path) -> dict:
    """EXT-51 (review 2, H-1): ext-launch resolves every mount source on this host (`realpath`, strict), so the
    golden request's `/data/...` host paths (mount sources, host_work, output_dir) are moved under `root` and the
    directories they name are created. Strings that do not start with `/data/` (socket paths, `/proc/...`) or that
    hold `..` (a lexical escape such as `/data/../run`) are left as they are. Container paths and argv stay as
    written (review round 5, EXT-94: moved under the test root, they would lie inside the `/tmp` tmpfs). Review
    round 4 (EXT-83): an argv element `/data/....py` whose host directory `<root>/data/...` exists is created there
    as a script stub, so the request names a training script in a read-only mount; review round 5 (EXT-91): an
    argv[0] `/data/.../python` whose venv directory exists is created as an interpreter stub (a regular file; the
    host's is a link into the Python mount, `test_ext_review5`)."""
    def move(x):
        if isinstance(x, str) and x.startswith(GOLDEN_PREFIX) and ".." not in x.split("/"):
            return f"{root}{x}"
        return x
    req = copy.deepcopy(request)
    for m in req.get("mounts") or [] if isinstance(req.get("mounts"), list) else []:
        if isinstance(m, dict) and "host" in m:
            m["host"] = move(m["host"])
    for k in ("host_work", "output_dir"):
        if k in req:
            req[k] = move(req[k])
    dirs = [m.get("host") for m in req.get("mounts") or [] if isinstance(m, dict)]
    dirs += [req.get("host_work"), req.get("output_dir")]
    for d in dirs:
        if isinstance(d, str) and d.startswith(f"{root}/"):
            try:
                Path(d).mkdir(parents=True, exist_ok=True)
            except (OSError, ValueError):
                pass                                    # a path the rules refuse anyway (`:`, control characters)
    argv = req.get("argv") if isinstance(req.get("argv"), list) else []
    stubs = [(a, INTERPRETER_TEXT, 3) for a in argv[:1] if isinstance(a, str) and a.endswith("/python")]
    stubs += [(a, SCRIPT_TEXT, 1) for a in argv[1:] if isinstance(a, str) and a.endswith(".py")]
    for a, text, up in stubs:
        host = move(a)
        if host == a:
            continue
        try:
            if not Path(host).exists() and Path(host).parents[up - 1].is_dir():
                Path(host).parent.mkdir(parents=True, exist_ok=True)
                Path(host).write_text(text, encoding="utf-8")
        except (OSError, ValueError):
            pass
    return req


def unrooted(text: str, root: Path, label: str) -> str:
    """The printed command with the test root removed and the hashed label as `<label>` (EXT-46, EXT-51)."""
    return text.replace(str(root), "").replace(label, "<label>")


def host_layout(request: dict, root: Path) -> dict[str, str]:
    """EXT-63, EXT-66, EXT-67: the host layout of a rooted request: host work is the request's (absolute) host_work,
    else `<root>/data/gatedtrain-work`, with a gpu-step.sh stand-in whose WORK is that directory (unless the test wrote
    one) and the fixture gpu-log.md; output root `<root>/data/gatedtrain-mm/out`; input root `<root>/data`. Review round 6:
    `<root>` stands for the container's `/` in an identity mount (EXT-112), so `/data/gatedtrain-mm/python` mounted from
    `<root>/data/gatedtrain-mm/python` is the README's identity mount; the pinned interpreter is the golden argv[0] and its
    tree the golden venv (EXT-113)."""
    hw = request.get("host_work")
    hw = Path(hw) if isinstance(hw, str) and hw.startswith("/") and ":" not in hw else root / "data" / "gatedtrain-work"
    try:
        if not str(hw).startswith(tempfile.gettempdir() + "/"):
            raise OSError("set up test directories only")     # e.g. host_work `/`: refused, never written
        if not (hw / "bin" / "gpu-step.sh").exists():
            gpu_step_script(hw)
        if not (hw / "gpu-log.md").exists():
            shutil.copyfile(GPU_LOG, hw / "gpu-log.md")
    except OSError:
        pass                                            # a host_work the rules refuse anyway
    (root / "data" / "gatedtrain-mm" / "out").mkdir(parents=True, exist_ok=True)
    return {ENV_HOST_WORK: str(hw), ENV_OUTPUT_ROOT: str(root / "data" / "gatedtrain-mm" / "out"),
            ENV_INPUT_ROOTS: str(root / "data"), ENV_MIRROR_ROOT: str(root),                     # EXT-112
            ENV_INTERPRETER: GOLDEN_INTERPRETER, ENV_INTERPRETER_TREES: str(root / GOLDEN_TREE)}   # EXT-113


def bound_spec(tmp: Path, request: dict, root: Path) -> Path:
    """EXT-84: `<tmp>/spec/spec.yaml` binding the rooted request's base (`/inputs/base-model`) and export
    (`/inputs/export`) mounts; the golden paths when the request has no such mount."""
    hosts = {m.get("container"): m.get("host") for m in request.get("mounts") or [] if isinstance(m, dict)}
    a1 = root / "data" / "gatedtrain-work" / "ext" / "a1"
    base = hosts.get("/inputs/base-model") if isinstance(hosts.get("/inputs/base-model"), str) else a1 / "base-model"
    export = hosts.get(EXPORT_CONTAINER) if isinstance(hosts.get(EXPORT_CONTAINER), str) else a1 / "export"
    return write_bound_spec(tmp / "spec", base=Path(base), export=Path(export))


def launch(tmp: Path, request: dict | Path, *, profile: str = "smoke", out: Path | None = None,
           smoke: Path | None = None, env: dict | None = None, extra: list[str] = (), root: Path | None = None,
           gpu_log: Path | None = None, spec: Path | bool = True):
    """Run ext-launch on the request rooted under `root` (default `<tmp>/host`) with that root's host layout
    (`host_layout`; `env` overrides it) and `--gpu-log <host_work>/gpu-log.md` (or `gpu_log`). EXT-84: `--spec`
    is `bound_spec` (`spec=True`), the given path, or absent (`spec=False`)."""
    if isinstance(request, Path):
        request = load_request(request)
    root = root if root is not None else tmp / "host"
    request = rooted(request, root)
    path = tmp / "request.yaml"
    path.write_text(yaml.safe_dump(request, sort_keys=False), encoding="utf-8")
    out = out or tmp / "launch.json"
    if not (tmp / "uv.lock").exists():
        shutil.copyfile(VENV_LOCK, tmp / "uv.lock")
    layout = host_layout(request, root)
    log = gpu_log if gpu_log is not None else Path(layout[ENV_HOST_WORK]) / "gpu-log.md"
    args = [str(VENV_PYTHON), "-m", "gatedtrain.testing.synthetic", "ext-launch", "--adapter", "synthetic", "--profile", profile,
            "--request", str(path), "--gpu-log", str(log), "--out-record", str(out), *extra]
    if smoke is not None:
        args += ["--smoke-record", str(smoke)]
    if spec is True:
        args += ["--spec", str(bound_spec(tmp, request, root))]
    elif spec:
        args += ["--spec", str(spec)]
    e = cli_env()
    e.update(layout)
    e.update(env or {})
    return subprocess.run(args, cwd=str(tmp), env=e, capture_output=True, text=True, timeout=60), out


def load_request(path: Path) -> dict:
    """A request file with its `<user>` placeholder filled (`_ext.container_user`, EXT-63)."""
    return yaml.safe_load(path.read_text(encoding="utf-8").replace("<user>", container_user()))


def golden_request() -> dict:
    return load_request(GOLDEN_REQUEST)


class LaunchCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-launch-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def refused(self, proc, out: Path, rule: str, field_suffix: str | None = None):
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        errs = parse_errors(proc.stderr)
        hits = [e for e in errs if e["rule"] == rule and (field_suffix is None or e["field"].endswith(field_suffix))]
        self.assertTrue(hits, f"expected error[{rule}] ...{field_suffix}; got {errs}\n{describe(proc)}")
        self.assertFalse(out.exists(), "a refused launch writes no record")


# ---------------------------------------------------------------------------
# EXT-22, EXT-23, EXT-24 (MM-31, MM-35): record and printed command
# ---------------------------------------------------------------------------

class TestGoldenCommand(LaunchCase):

    def test_command_matches_golden(self):
        """The golden command with the test root removed and the hashed label written `<label>` (EXT-51)."""
        proc, out = launch(self.tmp, GOLDEN_REQUEST)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        label = json.loads(out.read_text(encoding="utf-8"))["label"]
        want = GOLDEN_COMMAND.read_text(encoding="utf-8").replace(" --user <user> ", f" --user {container_user()} ")
        self.assertEqual(unrooted(proc.stdout, self.tmp / "host", label), want)

    def test_record_contents(self):
        proc, out = launch(self.tmp, GOLDEN_REQUEST)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        raw = out.read_bytes()
        rec = json.loads(raw)
        self.assertEqual(raw, jcs(rec), "launch.json is canonical JSON")
        req = rooted(golden_request(), self.tmp / "host")
        self.assertEqual(rec["schema"], "gatedtrain.launch_record")
        self.assertEqual(rec["schema_version"], "2.0")                        # EXT-95 (review round 5)
        self.assertEqual(set(rec["tags"]), {"prototype", "unpromotable"})
        self.assertEqual((rec["adapter"], rec["profile"], rec["area"]), ("synthetic", "smoke", 1))
        # EXT-46: the request label plus 12 hex digits of the record's hash
        self.assertEqual(rec["label"], launch_label(dict(rec, label="ext-synthetic-smoke")))
        for key in ("host_work", "framework", "training_seed", "argv", "mounts", "tmpfs", "network",
                    "network_reason", "user", "gpus", "gpu_est_h", "output_dir"):
            self.assertEqual(rec[key], req[key], key)
        self.assertEqual(rec["runtime"], {"kind": "venv_in_container", "image": req["runtime"]["image"],
                                          "venv_lock_sha256": file_sha256(VENV_LOCK)})
        self.assertEqual(rec["env_names"], sorted(req["env"]))
        self.assertIn("HF_HUB_OFFLINE", rec["env_names"])
        self.assertNotIn("env", rec)                  # names only (MM-35)
        self.assertEqual(rec["gpu_used_h"], 3.12)

    def test_never_runs_the_command(self):
        """MM-31: ext-launch writes the record and prints the command; it never runs docker or gpu-step.sh."""
        host_work = self.tmp / "work"
        (host_work / "bin").mkdir(parents=True)
        fake_bin = self.tmp / "fakebin"
        fake_bin.mkdir()
        marker = self.tmp / "ran"
        for path in (host_work / "bin" / "gpu-step.sh", fake_bin / "docker"):
            # the WORK= and LOG= lines ext-launch reads (EXT-67), then a marker if anything ran it
            path.write_text(f"#!/bin/sh\nWORK={host_work}\nLOG=\"$WORK/gpu-guard.log\"\ntouch {marker}\n",
                            encoding="utf-8")
            path.chmod(0o755)
        req = golden_request()
        old = req["host_work"]
        req = json.loads(json.dumps(req).replace(old, str(host_work)))
        for m in req["mounts"]:                          # mount sources exist on the launch host (EXT-51)
            if m["host"].startswith(str(host_work)):
                Path(m["host"]).mkdir(parents=True, exist_ok=True)
        proc, out = launch(self.tmp, req, env={"PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
                                               ENV_INPUT_ROOTS: f"{self.tmp / 'host' / 'data'}:{host_work}"})
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertTrue(out.is_file())
        self.assertIn(f"{host_work}/bin/gpu-step.sh ext-synthetic-smoke-", proc.stdout)      # EXT-46 label
        self.assertFalse(marker.exists(), "ext-launch ran a command")

    def test_gpu_runtime_shm_and_exec_scratch(self):
        """EXT-24 (synthetic-runtime README "Command shape"; docs/tier-b-runbook.md section 8): a GPU step adds
        `--runtime=nvidia` before `--gpus all` (the gatedtrain daemon's CDI mode refuses `--gpus` alone), every step
        `--shm-size=8g` after the security options, and the tmpfs scratch is mounted `rw,exec,nosuid,nodev,size=8g`
        (MM-34: Triton and torch.compile load compiled objects from /tmp; Docker's tmpfs default is noexec). Without
        GPUs neither `--runtime=nvidia` nor `--gpus` is printed."""
        proc, out = launch(self.tmp, GOLDEN_REQUEST)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertIn(" --rm --runtime=nvidia --gpus all --network none ", proc.stdout)
        self.assertIn(" --security-opt no-new-privileges --shm-size=8g -e ", proc.stdout)
        self.assertIn(" --tmpfs /tmp:rw,exec,nosuid,nodev,size=8g ", proc.stdout)
        rec = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(rec["tmpfs"], ["/tmp"])                      # the record keeps paths only (EXT-39)
        req = golden_request()
        req["gpus"] = False
        cpu = self.tmp / "cpu"
        cpu.mkdir()
        proc, _ = launch(cpu, req)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertNotIn("--runtime=nvidia", proc.stdout)
        self.assertNotIn("--gpus", proc.stdout)
        self.assertIn(" --shm-size=8g ", proc.stdout)
        self.assertIn(" --tmpfs /tmp:rw,exec,nosuid,nodev,size=8g ", proc.stdout)

    def test_existing_record_refused(self):
        out = self.tmp / "launch.json"
        out.write_text("{}", encoding="utf-8")
        proc, _ = launch(self.tmp, GOLDEN_REQUEST, out=out)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertIn(("EXT-LAUNCH", "--out-record"), [(e["rule"], e["field"]) for e in parse_errors(proc.stderr)])
        self.assertEqual(out.read_text(encoding="utf-8"), "{}")


# ---------------------------------------------------------------------------
# EXT-25, EXT-26 (MM-32, MM-34): rules on the request (ext-collect re-checks the record: test_ext_collect)
# ---------------------------------------------------------------------------

def _mount(req, i, **kw):
    req["mounts"][i].update(kw)


CASES = [
    # (name, mutate, field suffix)
    ("input mount read-write", lambda r: _mount(r, 0, mode="rw"), "mounts[0].mode"),
    ("mode neither ro nor rw", lambda r: _mount(r, 1, mode="z"), "mounts[1].mode"),
    ("output not mounted read-write", lambda r: _mount(r, 4, mode="ro"), "output_dir"),
    ("output mounted elsewhere", lambda r: r.update(output_dir="/data/gatedtrain-mm/out/a1/other"), "output_dir"),
    ("host work mounted", lambda r: r["mounts"].append({"host": "/data/gatedtrain-work", "container": "/work", "mode": "ro"}),
     "mounts[5].host"),
    ("parent of output mounted", lambda r: r["mounts"].append({"host": "/data/gatedtrain-mm/out/a1", "container": "/ext",
                                                                "mode": "ro"}), "mounts[5].host"),
    ("docker socket", lambda r: r["mounts"].append({"host": "/var/run/docker.sock", "container": "/var/run/docker.sock",
                                                     "mode": "ro"}), "mounts[5].host"),
    ("host network without reason", lambda r: r.update(network="host"), "network"),            # EXT-81
    ("named network without reason", lambda r: r.update(network="testnet"), "network_reason"),
    ("host network without daemon reason", lambda r: r.update(network="host", network_reason="vLLM on localhost"),
     "network"),                                                                                # EXT-81
    ("root user", lambda r: r.update(user="0:0"), "user"),
    ("user without gid", lambda r: r.update(user="1000"), "user"),
    ("no user", lambda r: r.pop("user"), "user"),
    ("hub online", lambda r: r["env"].update(HF_HUB_OFFLINE="0"), "env.HF_HUB_OFFLINE"),
    ("hub offline unset", lambda r: r["env"].pop("HF_HUB_OFFLINE"), "env.HF_HUB_OFFLINE"),
    ("credential in env", lambda r: r["env"].update(HF_TOKEN="hf_fixture_not_a_token"), "env.HF_TOKEN"),
    ("api key in env", lambda r: r["env"].update(OPENAI_API_KEY="x"), "env.OPENAI_API_KEY"),
    ("mutable image tag", lambda r: r["runtime"].update(image="trl:0.19.1"), "runtime.image"),
    ("short image id", lambda r: r["runtime"].update(image="sha256:abab"), "runtime.image"),
    ("unknown runtime kind", lambda r: r["runtime"].update(kind="podman"), "runtime.kind"),
    ("venv without lock", lambda r: r.update(runtime={"kind": "venv_in_container", "image": "sha256:" + "ab" * 32}),
     "runtime.venv_lock"),
    ("relative mount", lambda r: _mount(r, 1, host="data/train"), "mounts[1].host"),
]


class TestLaunchRules(LaunchCase):

    def test_golden_request_accepted(self):
        proc, out = launch(self.tmp, golden_request())
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))

    def test_each_rule_refuses(self):
        for name, mutate, suffix in CASES:
            with self.subTest(case=name):
                req = copy.deepcopy(golden_request())
                mutate(req)
                sub = self.tmp / name.replace(" ", "_")
                sub.mkdir()
                proc, out = launch(sub, req)
                self.refused(proc, out, "EXT-LAUNCH", suffix)

    def test_credential_value_never_printed(self):
        req = golden_request()
        req["env"]["HF_TOKEN"] = "hf_fixture_secret_value_123"
        proc, out = launch(self.tmp, req)
        self.assertNotIn("hf_fixture_secret_value_123", proc.stdout + proc.stderr)

    def test_digest_reference_accepted(self):
        req = golden_request()
        req["runtime"]["image"] = "ghcr.io/example/trl@sha256:" + "cd" * 32
        proc, out = launch(self.tmp, req)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))

    def test_network_host_with_both_reasons_refused(self):
        """Review round 4 (EXT-81): `host` is refused whatever the reasons, with or without --allow-network."""
        req = golden_request()
        req.update(network="host", network_reason="vLLM server on localhost for GRPO generation",
                   network_host_reason="the gatedtrain daemon has no bridge network (a host deviation)")
        for extra in ([], ["--allow-network"]):
            with self.subTest(extra=extra):
                sub = self.tmp / f"host{len(extra)}"
                sub.mkdir()
                proc, out = launch(sub, req, extra=extra)
                self.refused(proc, out, "EXT-LAUNCH", "--request.network")

    def test_venv_in_container_records_lock_hash(self):
        lock = self.tmp / "uv.lock"
        lock.write_text("version = 1\n", encoding="utf-8")
        req = golden_request()
        req["runtime"] = {"kind": "venv_in_container", "image": "sha256:" + "ef" * 32, "venv_lock": str(lock)}
        req["mounts"].insert(0, {"host": "/data/venvs/trl", "container": "/inputs/venv", "mode": "ro"})   # EXT-112
        proc, out = launch(self.tmp, req)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        rec = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(rec["runtime"]["kind"], "venv_in_container")
        self.assertEqual(rec["runtime"]["venv_lock_sha256"], file_sha256(lock))

    def test_venv_mounted_read_write_refused(self):
        lock = self.tmp / "uv.lock"
        lock.write_text("version = 1\n", encoding="utf-8")
        req = golden_request()
        req["runtime"] = {"kind": "venv_in_container", "image": "sha256:" + "ef" * 32, "venv_lock": str(lock)}
        req["mounts"].insert(0, {"host": "/data/venvs/trl", "container": "/venv", "mode": "rw"})
        proc, out = launch(self.tmp, req)
        self.refused(proc, out, "EXT-LAUNCH", "mounts[0].mode")


# ---------------------------------------------------------------------------
# EXT-27, EXT-30 (MM-31, MM-39): the full profile needs a PASS smoke record
# ---------------------------------------------------------------------------

class TestFullProfile(LaunchCase):

    def smoke(self, **kw) -> Path:
        p = self.tmp / "smoke.json"
        p.write_bytes(jcs(smoke_record(base_model="sha256:" + "1" * 64, item_set="sha256:" + "2" * 64, **kw)))
        return p

    def test_full_without_smoke_refused(self):
        proc, out = launch(self.tmp, golden_request(), profile="full")
        self.refused(proc, out, "EXT-SMOKE")

    def test_full_with_failing_smoke_refused(self):
        proc, out = launch(self.tmp, golden_request(), profile="full", smoke=self.smoke(peak_gpu_mem_gb=23.0))
        self.refused(proc, out, "EXT-SMOKE")

    def test_full_uses_smoke_projection(self):
        proc, out = launch(self.tmp, golden_request(), profile="full", smoke=self.smoke(projected_gpu_h=2.5))
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertIn(" GATEDTRAIN_GPU_STEP_EST_H=2.5 ", proc.stdout)
        self.assertTrue(proc.stdout.startswith("GATEDTRAIN_GPU_CAP_H=1000 GATEDTRAIN_GPU_USED_H=3.12 "), proc.stdout)
        self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["profile"], "full")


# ---------------------------------------------------------------------------
# EXT-28 (MM-36): GPU seconds from gpu-guard.log
# ---------------------------------------------------------------------------

class TestGpuSeconds(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-gpu-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def seconds(self, text: str, label: str = LABEL):
        from gatedtrain.ext.launch import gpu_seconds
        p = self.tmp / "gpu-guard.log"
        p.write_text(text, encoding="utf-8")
        return gpu_seconds(p, label)

    def test_fixture_log(self):
        """Preempted attempt 302 s + failed attempt 1800 s + a second invocation's attempt 1800 s = 3902 s;
        the `-eval` label, PREEMPT, WAIT, REFUSED and GUARD lines add nothing."""
        self.assertEqual(self.seconds((EXT / "gpu-guard.fixture.log").read_text(encoding="utf-8")), 3902)

    def test_other_label_not_a_prefix_match(self):
        self.assertEqual(self.seconds((EXT / "gpu-guard.fixture.log").read_text(encoding="utf-8"),
                                      "ext-synthetic-full-eval"), 300)

    def test_no_end_refused(self):
        text = ("2026-11-11T09:00:00Z START ext-synthetic-full attempt=1\n"
                "2026-11-11T09:05:00Z PREEMPTED ext-synthetic-full attempt=1 rc=137\n")
        with self.assertRaises(ValueError):
            self.seconds(text)

    def test_unterminated_attempt_refused(self):
        text = ("2026-11-11T09:00:00Z START ext-synthetic-full attempt=1\n"
                "2026-11-11T09:30:00Z END ext-synthetic-full attempt=1 rc=0\n"
                "2026-11-11T09:40:00Z START ext-synthetic-full attempt=1\n")
        with self.assertRaises(ValueError):
            self.seconds(text)

    def test_label_absent_refused(self):
        with self.assertRaises(ValueError):
            self.seconds("2026-11-11T09:00:00Z GUARD started pid=1\n")


# ---------------------------------------------------------------------------
# EXT-29 (MM-38): the smoke verdict
# ---------------------------------------------------------------------------

class TestSmokeVerdict(unittest.TestCase):

    def verdict(self, *, score_range=(0, 1), binary=True, **kw):
        from gatedtrain.ext.smoke import smoke_verdict
        rec = smoke_record(base_model="sha256:" + "1" * 64, item_set="sha256:" + "2" * 64, **kw)
        return smoke_verdict(rec, score_range=score_range, binary=binary)

    def check(self, expected_reasons: set[str], **kw):
        v = self.verdict(**kw)
        self.assertEqual(v["verdict"], "FAIL" if expected_reasons else "PASS", v)
        self.assertEqual(set(v["reasons"]), expected_reasons, v)

    def test_pass(self):
        self.check(set())

    def test_gpu_memory(self):
        self.check({"gpu_memory"}, peak_gpu_mem_gb=22.6)
        self.check(set(), peak_gpu_mem_gb=22.5)

    def test_budget(self):
        self.check({"budget"}, projected_gpu_h=2.0, gpu_log_total_h=998.5)
        self.check(set(), projected_gpu_h=1.5, gpu_log_total_h=998.5)
        self.check(set(), projected_gpu_h=12.0, gpu_log_total_h=16.7)          # above 24 h, below the budget ceiling

    def test_evaluator_errors(self):
        self.check({"evaluator_errors"}, error_count=1)

    def test_all_zero_baseline_degenerate(self):
        v = self.verdict(baseline_scores=[0.0] * 20)
        self.assertEqual(v["verdict"], "FAIL")
        self.assertIn("degenerate_mean", v["reasons"])

    def test_binary_degenerate_only_at_a_bound(self):
        """Design decision (EXT-G3): a binary metric is degenerate only when its mean is 0 or 1."""
        v = self.verdict(baseline_scores=[1.0] * 20)
        self.assertEqual(v["verdict"], "FAIL")
        self.assertIn("degenerate_mean", v["reasons"])
        self.check(set(), baseline_scores=[1.0] + [0.0] * 39)                  # mean 0.025: not degenerate

    def test_continuous_mean_near_floor(self):
        near = [0.001 * (i + 1) for i in range(20)]                            # mean 0.0105 < 0.05
        self.check({"degenerate_mean"}, baseline_scores=near, binary=False)
        self.check(set(), baseline_scores=[0.03 + 0.001 * i for i in range(20)] + [0.3], binary=False)  # 0.052

    def test_binary_metric_skips_mode_rule(self):
        self.check(set(), baseline_scores=[1.0] * 15 + [0.0] * 5)               # 75 % share 1.0

    def test_continuous_mode_rule(self):
        scores = [0.5] * 10 + [0.1 * i for i in range(10)]                     # 50 % share 0.5
        self.check({"degenerate_mode"}, baseline_scores=scores, binary=False)

    def test_continuous_constant(self):
        self.check({"degenerate_mode", "degenerate_sd"}, baseline_scores=[0.5] * 20, binary=False)

    def test_unbounded_range_skips_mean_rule(self):
        scores = [0.001 * i for i in range(20)]
        self.check(set(), baseline_scores=scores, score_range=(0, None), binary=False)


if __name__ == "__main__":
    unittest.main()
