"""Launch and collection checks for input mounts, network opt-in and script binding (EXT-79 to EXT-89)."""

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

from _ext import (ENV_HOST_WORK, ENV_INPUT_ROOTS, ENV_INTERPRETER, ENV_INTERPRETER_TREES, ENV_MIRROR_ROOT, ENV_OUTPUT_ROOT,
                  SCRIPT_CONTAINER, gpu_step_script, guard_log,
                  host_config_sha256, launch_label, write_canonical)
from _support import (EXIT_OK, EXIT_REFUSED, HOST_ENV, VENV_PYTHON, SKEL_ROOT, cli_env, describe, file_sha256,
                      parse_errors, run_cli)
from test_ext_hardening import CollectCase
from test_ext_launch import GOLDEN_INTERPRETER, GOLDEN_TREE, golden_request, launch, rooted
from test_ext_review2 import unit_config, unit_request
import test_ext_review3 as review3
from test_ext_review3 import Launch3, problems

EXIT_USAGE = 2


def registry_roots() -> tuple[str, ...]:
    """The synthetic adapter's registered input roots."""
    from gatedtrain.ext import registry
    return tuple(registry.input_roots("synthetic"))


def fields(req: dict, **kw) -> list[str]:
    return [f for f, _ in problems(req, **kw)]


def messages(req: dict, field: str, **kw) -> list[str]:
    return [m for f, m in problems(req, **kw) if f == field]


class TmpCase(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-r4-"))
        self.addCleanup(shutil.rmtree, self.root, True)


# ---------------------------------------------------------------------------
# EXT-79 (H1 a): read-only inputs come from the adapter's input roots, not the whole data disk
# ---------------------------------------------------------------------------

class TestAdapterInputRoots(TmpCase):

    def test_registry_roots(self):
        from gatedtrain.ext import registry
        from gatedtrain.ext.launch import HostConfig, adapter_config, host_config
        roots = registry_roots()
        self.assertTrue(roots)
        self.assertIsNone(HostConfig().input_roots)
        self.assertIsNone(host_config(HOST_ENV).input_roots)
        self.assertEqual(adapter_config(HostConfig(), "synthetic").input_roots, roots)
        explicit = HostConfig(input_roots=("/a",))
        self.assertEqual(adapter_config(explicit, "synthetic").input_roots, ("/a",))          # the override wins
        self.assertEqual(host_config({**HOST_ENV, ENV_INPUT_ROOTS: f"/a{os.pathsep}/b"}).input_roots, ("/a", "/b"))

    def test_no_roots_refuses_every_input(self):
        """A layout without input roots (none configured, no adapter) accepts no read-only mount."""
        req = unit_request(self.root)
        cfg = dataclasses.replace(unit_config(req), input_roots=None)
        self.assertTrue(all(f"mounts[{i}].host" in fields(req, config=cfg) for i in range(4)))

class TestAdapterRootsApplied(CollectCase):
    """EXT-79: ext-launch, `--recheck` and ext-collect use the adapter's roots when no override names any (the
    scenario's inputs lie outside them, so every read-only mount is refused naming the adapter's roots)."""

    def config(self):
        from gatedtrain.ext.launch import HostConfig
        return HostConfig(host_work=str(self.sc.host_work), output_root=str(self.sc.output_root))

    def assert_adapter_roots(self, msgs: list[str]):
        self.assertTrue(msgs and all(registry_roots()[0] in m for m in msgs), msgs)

    def test_record_problems(self):
        from gatedtrain.ext.launch import record_problems
        got = record_problems(self.sc.launch_record(), adapter="synthetic", stage_out=None, config=self.config())
        self.assert_adapter_roots([p.message for p in got if p.field == "--launch-record.mounts[0].host"
                                   and "input root" in p.message])

    def test_ext_launch(self):
        from gatedtrain.errors import Refused
        from gatedtrain.ext.launch import ext_launch
        from test_ext_review3 import scenario_request
        import yaml
        out = self.sc.output_root / "a1" / "roots-run"
        out.mkdir(parents=True, exist_ok=True)
        req = self.path("request").with_suffix(".yaml")
        req.write_text(yaml.safe_dump(scenario_request(self.sc, out)), encoding="utf-8")
        with self.assertRaises(Refused) as ctx:
            ext_launch(adapter="synthetic", profile="smoke", request=req, gpu_log=self.sc.host_work / "gpu-log.md",
                       out_record=self.path("rec").with_suffix(".json"), spec=self.sc.paths["spec"],
                       config=self.config())
        self.assert_adapter_roots([p.message for p in ctx.exception.problems
                                   if p.field == "--request.mounts[0].host" and "input root" in p.message])

    def test_recheck(self):
        from gatedtrain.errors import Refused
        from gatedtrain.ext.launch import recheck
        with self.assertRaises(Refused) as ctx:
            recheck(adapter="synthetic", record_path=self.sc.launch, spec=self.sc.paths["spec"], config=self.config())
        self.assert_adapter_roots([p.message for p in ctx.exception.problems
                                   if p.field == "--recheck.mounts[0].host" and "input root" in p.message])


# ---------------------------------------------------------------------------
# EXT-80 (H1 b): a file-name scan of every read-only source (contents never read)
# ---------------------------------------------------------------------------

class TestCredentialFileNames(TmpCase):

    n = 0

    def check(self, rel: str, *, directory: bool = False) -> list[str]:
        """Messages for a read-only source (a new directory under the input root) holding `rel`."""
        type(self).n += 1
        src = self.root / "data" / "inputs" / f"x{type(self).n}"
        p = src / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if directory:
            p.mkdir()
        else:
            p.write_text("fixture\n", encoding="utf-8")
        req = unit_request(self.root)
        req["mounts"].append({"host": str(src), "container": "/x", "mode": "ro"})
        return messages(req, "mounts[5].host")

    def test_refused(self):
        for rel in (".env", ".env.local", "deep/er/.env.production", "aws_credentials.json", "Credentials.JSON",
                    "my-secret.txt", "SECRETS.yaml", "server.pem", "tls.key", "id_rsa", "id_rsa.pub", "id_ed25519",
                    "id_ed25519.pub", "token", "TOKEN", ".netrc", ".git-credentials", "app/secrets.py"):
            with self.subTest(name=rel):
                msgs = self.check(rel)
                self.assertTrue(any("credential" in m and rel.rsplit("/", 1)[-1] in m for m in msgs), msgs)

    def test_directory_name_refused(self):
        msgs = self.check("conf/secrets", directory=True)
        self.assertTrue(any("credential" in m for m in msgs), msgs)

    def test_source_name_refused(self):
        """The mounted directory's own name counts (a source named `secrets`)."""
        src = self.root / "data" / "inputs" / "secrets"
        src.mkdir(parents=True)
        req = unit_request(self.root)
        req["mounts"].append({"host": str(src), "container": "/x", "mode": "ro"})
        msgs = messages(req, "mounts[5].host")
        self.assertTrue(any("credential-like name" in m for m in msgs), msgs)

    def test_allowed(self):
        for rel in ("tokenizer.json", "tokenizer_config.json", "token.py", ".envrc", "keyboard.py",
                    "model.safetensors", "environment.yml", "monkey.txt", "keys.txt"):
            with self.subTest(name=rel):
                self.assertEqual(self.check(rel), [])

    def test_venv_and_site_packages_scanned(self):
        """Round 5 (EXT-93, review L2): a `.venv` or `site-packages` directory is no longer skipped by its name."""
        for rel in (".venv/lib/.env", "lib/python3.12/site-packages/pkg/secret.py", "a/.venv/token"):
            with self.subTest(name=rel):
                self.assertTrue(any("credential" in m for m in self.check(rel)))

    def test_stdlib_secrets_module_allowed(self):
        """The standard library's `secrets` module (in a real interpreter tree) is not a credential file; round
        5 (EXT-93): only in a declared interpreter tree."""
        from gatedtrain.ext.launch import credential_name_problem
        for rel in ("cpython-3.12.16/lib/python3.12/secrets.py",
                    "cpython-3.12.16/lib/python3.12/__pycache__/secrets.cpython-312.pyc"):
            with self.subTest(name=rel):
                self.assertTrue(any("credential" in m for m in self.check(rel)))
                src = self.root / "data" / "inputs" / f"x{type(self).n}"
                self.assertIsNone(credential_name_problem(str(src), interpreter_trees=(str(src / "cpython-3.12.16"),)))

    def test_unit(self):
        from gatedtrain.ext.launch import SCAN_LIMIT, credential_name_problem
        self.assertEqual(SCAN_LIMIT, 200_000)
        d = self.root / "many"
        d.mkdir()
        for i in range(10):
            (d / f"f{i}").write_text("", encoding="utf-8")
        self.assertIsNone(credential_name_problem(str(d)))
        why = credential_name_problem(str(d), limit=5)
        self.assertIn("more than 5", why)
        (d / "id_rsa").write_text("", encoding="utf-8")
        self.assertIn("id_rsa", credential_name_problem(str(d)))

    def test_unlistable_directory_refused(self):
        if os.geteuid() == 0:
            self.skipTest("root lists every directory")
        src = self.root / "data" / "inputs" / "x"
        (src / "locked").mkdir(parents=True)
        (src / "locked").chmod(0)
        self.addCleanup((src / "locked").chmod, 0o700)
        req = unit_request(self.root)
        req["mounts"].append({"host": str(src), "container": "/x", "mode": "ro"})
        msgs = messages(req, "mounts[5].host")
        self.assertTrue(any("cannot list" in m for m in msgs), msgs)


class TestCredentialFileNamesLaunch(Launch3):

    def test_value_never_read_or_printed(self):
        """The `.env` is unreadable (mode 000) and holds a marker; the refusal names the file, never its content."""
        host = self.tmp / "host"
        req = golden_request()
        rooted(req, host)
        env = host / "data" / "gatedtrain-mm" / "synthetic-runtime" / ".env"
        env.write_text("HF_TOKEN=hf_marker_value_never_printed\n", encoding="utf-8")
        env.chmod(0)
        self.addCleanup(env.chmod, 0o600)
        errs = self.refused_at(req, "EXT-LAUNCH", "--request.mounts[1].host")
        self.assertTrue(any(".env" in e["message"] and "credential" in e["message"] for e in errs), errs)
        proc, _ = launch(self.sub(), req, root=host)
        self.assertNotIn("hf_marker_value_never_printed", proc.stdout + proc.stderr)


class TestCredentialFileNamesCollect(CollectCase):

    def test_export_with_a_dotenv_refused(self):
        p = self.sc.export / ".env"
        p.write_text("X=1\n", encoding="utf-8")
        self.addCleanup(p.unlink)
        hits = self.refused("EXT-LAUNCH", "--launch-record.mounts[1].host")
        self.assertTrue(any("credential" in h["message"] for h in hits), hits)


# ---------------------------------------------------------------------------
# EXT-81 (M1): never `host`; another network only with --allow-network
# ---------------------------------------------------------------------------

class TestNetwork(Launch3):

    def unit(self, **kw) -> list[str]:
        root = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-r4-net-"))
        self.addCleanup(shutil.rmtree, root, True)
        req = unit_request(root)
        allow = kw.pop("allow", False)
        req.update(kw)
        from gatedtrain.ext.launch import isolation_problems
        return [f for f, _ in isolation_problems(req, kind="request", socket_gids=frozenset(),
                                                  config=unit_config(req), allow_network=allow)]

    def test_unit(self):
        host = dict(network="host", network_reason="vLLM on localhost", network_host_reason="no bridge on the daemon")
        bridge = dict(network="bridge", network_reason="a model server on another container")
        self.assertIn("network", self.unit(**host))
        self.assertIn("network", self.unit(allow=True, **host))
        self.assertIn("network", self.unit(**bridge))
        self.assertNotIn("network", self.unit(allow=True, **bridge))
        self.assertIn("network_reason", self.unit(allow=True, network="bridge", network_reason=" "))
        self.assertNotIn("network", self.unit())

    def test_probe_host_refused(self):
        """Probe p4: `host` with one-character reasons was accepted."""
        req = golden_request()
        req.update(network="host", network_reason="x", network_host_reason="y")
        self.refused_at(req, "EXT-LAUNCH", "--request.network")
        self.refused_at(req, "EXT-LAUNCH", "--request.network", extra=["--allow-network"])

    def test_bridge_needs_the_flag(self):
        req = golden_request()
        req.update(network="bridge", network_reason="a localhost model server for the evaluator")
        errs = self.refused_at(req, "EXT-LAUNCH", "--request.network")
        self.assertTrue(any("--allow-network" in e["fix"] or "--allow-network" in e["message"] for e in errs), errs)
        proc, out = self.accepted(req, extra=["--allow-network"])
        self.assertIn(" --network bridge ", proc.stdout)
        rec = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual((rec["allow_network"], rec["network"], rec["network_reason"]),
                         (True, "bridge", req["network_reason"]))

    def test_none_records_false(self):
        _, out = self.accepted(golden_request())
        self.assertIs(json.loads(out.read_text(encoding="utf-8"))["allow_network"], False)
        _, out = self.accepted(golden_request(), extra=["--allow-network"])    # the flag alone changes nothing
        self.assertIs(json.loads(out.read_text(encoding="utf-8"))["allow_network"], False)


class TestNetworkCollect(CollectCase):

    def relabelled(self, **overrides) -> Path:
        rec = self.sc.launch_record(**overrides)
        guard_log(self.sc.host_work, rec["label"])
        self.addCleanup(guard_log, self.sc.host_work, self.sc.label)
        return write_canonical(self.path("launch").with_suffix(".json"), rec)

    def test_record(self):
        for name, kw in {"bridge without allow_network": dict(network="bridge", network_reason="x"),
                         "host with every reason": dict(network="host", network_reason="x", network_host_reason="y",
                                                        allow_network=True),
                         "allow_network not a boolean": dict(network="bridge", network_reason="x",
                                                             allow_network="yes")}.items():
            with self.subTest(case=name):
                path = self.relabelled(**kw)
                self.refused("EXT-LAUNCH", "--launch-record.network", launch=path)
                if kw.get("allow_network") == "yes":
                    self.refused("EXT-LAUNCH", "--launch-record.allow_network", launch=path)
                self.doCleanups()


# ---------------------------------------------------------------------------
# EXT-82 (M2): --recheck compares gpu_used_h with the running total now
# ---------------------------------------------------------------------------

class TestRecheckGpuUsed(Launch3):

    def test_running_total_moved(self):
        sub = self.sub()
        proc, rec = launch(sub, golden_request(), root=self.tmp / "host")
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        log = self.tmp / "host" / "data" / "gatedtrain-work" / "gpu-log.md"
        text = log.read_text(encoding="utf-8")
        log.write_text(text.replace("| 0.20 | 3.12 | PASS |", "| 0.20 | 3.12 | PASS |\n| 11 other step | "
                                    "2026-11-11T11:00:00Z | 2026-11-11T12:00:00Z | 1.00 | 4.12 | PASS |"),
                       encoding="utf-8")
        self.addCleanup(log.write_text, text, encoding="utf-8")
        chk = self.recheck(rec)
        self.assertEqual(chk.returncode, EXIT_REFUSED, describe(chk))
        hits = [e for e in parse_errors(chk.stderr) if (e["rule"], e["field"]) == ("EXT-GPU", "--recheck.gpu_used_h")]
        self.assertTrue(hits and "4.12" in hits[0]["message"], describe(chk))
        log.write_text(text, encoding="utf-8")
        ok = self.recheck(rec)
        self.assertEqual(ok.returncode, EXIT_OK, describe(ok))

    def recheck(self, record: Path):
        e = cli_env()
        e.update({ENV_HOST_WORK: str(self.tmp / "host" / "data" / "gatedtrain-work"),
                  ENV_OUTPUT_ROOT: str(self.tmp / "host" / "data" / "gatedtrain-mm" / "out"),
                  ENV_INPUT_ROOTS: str(self.tmp / "host" / "data"), ENV_MIRROR_ROOT: str(self.tmp / "host"),  # EXT-112
                  ENV_INTERPRETER: GOLDEN_INTERPRETER,                                                    # EXT-113
                  ENV_INTERPRETER_TREES: str(self.tmp / "host" / GOLDEN_TREE)})
        return subprocess.run([str(VENV_PYTHON), "-m", "gatedtrain.testing.synthetic", "ext-launch", "--adapter", "synthetic", "--recheck",
                               str(record), "--spec", str(record.parent / "spec" / "spec.yaml")], cwd=str(self.tmp),
                              env=e, capture_output=True, text=True, timeout=60)


# ---------------------------------------------------------------------------
# EXT-83 (M3): the training script is bound to the record
# ---------------------------------------------------------------------------

class TestScriptUnit(TmpCase):

    def test_argv_script(self):
        """Round 5 (EXT-91): `bind_argv` replaces `argv_script`; argv[0] is the interpreter in a read-only mount and
        argv[1] the script, so the round-4 cases that skipped options or directories before the script refuse."""
        from gatedtrain.ext.launch import bind_argv
        code = self.root / "code"
        (code / "pkg").mkdir(parents=True)
        (code / "run.py").write_text("x", encoding="utf-8")
        (code / "bin").mkdir()
        (code / "bin" / "python").write_text("", encoding="utf-8")
        (self.root / "secret-elsewhere.py").write_text("x", encoding="utf-8")
        (code / "escape.py").symlink_to(self.root / "secret-elsewhere.py")
        out = self.root / "out"
        out.mkdir()
        (out / "run.py").write_text("x", encoding="utf-8")
        mounts = [{"host": str(code), "container": "/code", "mode": "ro"},
                  {"host": str(out), "container": "/out", "mode": "rw"}]
        cases = {
            "interpreter in the mount": (["/code/bin/python", "/code/run.py"], "/code/run.py"),
            "options before the script": (["/code/bin/python", "-u", "/code/run.py"], None),
            "a directory before the script": (["/code/bin/python", "/code/pkg", "/code/run.py"], None),
            "a relative interpreter": (["python", "/code/run.py"], None),
            "a file in the rw output is not a script": (["/code/bin/python", "/out/run.py"], None),
            "a link out of the mount is not a script": (["/code/bin/python", "/code/escape.py"], None),
            "no file": (["/code/bin/python", "-c", "print(1)"], None),
            "a missing file": (["/code/bin/python", "/code/absent.py"], None),
        }
        for name, (argv, want) in cases.items():
            with self.subTest(case=name):
                got, _ = bind_argv(argv, mounts)
                self.assertEqual(got.script_path if got else None, want)
                if got:
                    self.assertEqual(got.script_host, os.path.realpath(code / "run.py"))


class TestScriptLaunch(Launch3):

    def test_recorded(self):
        _, out = self.accepted(golden_request())
        rec = json.loads(out.read_text(encoding="utf-8"))
        script = self.tmp / "host" / "data" / "gatedtrain-mm" / "synthetic-runtime" / "train.py"
        self.assertEqual((rec["script_path"], rec["script_sha256"]),       # round 5: argv stays a container path
                         ("/data/gatedtrain-mm/synthetic-runtime/train.py", file_sha256(script)))

    def test_no_script_refused(self):
        """Probe p4: argv running another program (`python -c`) was accepted."""
        req = golden_request()
        req["argv"] = ["/data/gatedtrain-mm/synthetic-runtime/.venv/bin/python", "-c", "print(1)"]
        self.refused_at(req, "EXT-LAUNCH", "--request.argv")

    def test_recheck_script_changed(self):
        sub = self.sub()
        proc, rec = launch(sub, golden_request(), root=self.tmp / "host")
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        script = self.tmp / "host" / "data" / "gatedtrain-mm" / "synthetic-runtime" / "train.py"
        old = script.read_text(encoding="utf-8")
        script.write_text(old + "import os\n", encoding="utf-8")
        self.addCleanup(script.write_text, old, encoding="utf-8")
        chk = TestRecheckGpuUsed.recheck(self, rec)
        self.assertEqual(chk.returncode, EXIT_REFUSED, describe(chk))
        self.assertIn(("EXT-LAUNCH", "--recheck.script_sha256"),
                      [(e["rule"], e["field"]) for e in parse_errors(chk.stderr)])

class TestScriptCollect(CollectCase):

    def relabelled(self, rec: dict) -> Path:
        rec = dict(rec, label=rec["label"].rsplit("-", 1)[0])
        rec["label"] = launch_label(rec)
        guard_log(self.sc.host_work, rec["label"])
        self.addCleanup(guard_log, self.sc.host_work, self.sc.label)
        return write_canonical(self.path("launch").with_suffix(".json"), rec)

    def edit_runtime(self, mutate) -> None:
        p = self.sc.stage_out / "env-runtime.json"
        old = p.read_bytes()
        doc = json.loads(old)
        mutate(doc)
        p.write_text(json.dumps(doc), encoding="utf-8")
        self.addCleanup(p.write_bytes, old)

    def test_script_changed_after_launch(self):
        old = self.sc.script.read_text(encoding="utf-8")
        self.sc.script.write_text(old + "# edited\n", encoding="utf-8")
        self.addCleanup(self.sc.script.write_text, old, encoding="utf-8")
        self.refused("EXT-LAUNCH", "--launch-record.script_sha256")

    def test_runtime_names_another_script(self):
        self.edit_runtime(lambda d: d.update(script_sha256="sha256:" + "0" * 64))
        hits = self.refused("EXT-LAUNCH", "--launch-record.script_sha256")
        self.assertTrue(any("env-runtime.json" in h["message"] for h in hits), hits)

    def test_runtime_without_the_key(self):
        self.edit_runtime(lambda d: d.pop("script_sha256"))
        self.refused("EXT-LAUNCH", "--launch-record.script_sha256")

    def test_runtime_missing_or_a_link(self):
        """A missing env-runtime.json, or a symbolic link to one with the right hash, is no proof of the run."""
        p = self.sc.stage_out / "env-runtime.json"
        saved = self.tmp / "env-runtime.saved"
        shutil.copyfile(p, saved)
        p.unlink()
        self.addCleanup(shutil.copyfile, saved, p)
        self.addCleanup(lambda: p.unlink() if p.is_symlink() else None)
        hits = self.refused("EXT-LAUNCH", "--launch-record.script_sha256")
        self.assertTrue(any("missing" in h["message"] for h in hits), hits)
        p.symlink_to(saved)
        hits = self.refused("EXT-LAUNCH", "--launch-record.script_sha256")
        self.assertTrue(any("missing" in h["message"] for h in hits), hits)

    def test_record_script_path_not_the_argv_script(self):
        rec = self.sc.launch_record()
        rec["script_path"] = "/code/other.py"
        self.refused("EXT-LAUNCH", "--launch-record.script_path", launch=self.relabelled(rec))

    def test_record_argv_without_a_script(self):
        rec = self.sc.launch_record()
        rec["argv"] = [rec["argv"][0], "-c", "print(1)"]
        self.refused("EXT-LAUNCH", "--launch-record.argv", launch=self.relabelled(rec))     # round 5: EXT-91

    def test_passes(self):
        self.assertEqual(self.passes()["status"], "PASS")


# ---------------------------------------------------------------------------
# EXT-84 (L1): --spec is required when the adapter binds the spec's mounts
# ---------------------------------------------------------------------------

class TestSpecRequired(Launch3):

    def test_launch(self):
        from gatedtrain.ext import registry
        self.assertTrue(registry.adapter("synthetic").binds_spec)
        errs = self.refused_at(golden_request(), "EXT-SPEC", "--spec", spec=False)
        self.assertTrue(errs)

    def test_recheck(self):
        sub = self.sub()
        proc, rec = launch(sub, golden_request(), root=self.tmp / "host")
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        e = cli_env()
        e.update({ENV_HOST_WORK: str(self.tmp / "host" / "data" / "gatedtrain-work"),
                  ENV_OUTPUT_ROOT: str(self.tmp / "host" / "data" / "gatedtrain-mm" / "out"),
                  ENV_INPUT_ROOTS: str(self.tmp / "host" / "data"), ENV_MIRROR_ROOT: str(self.tmp / "host"),  # EXT-112
                  ENV_INTERPRETER: GOLDEN_INTERPRETER,                                                    # EXT-113
                  ENV_INTERPRETER_TREES: str(self.tmp / "host" / GOLDEN_TREE)})
        chk = subprocess.run([str(VENV_PYTHON), "-m", "gatedtrain.testing.synthetic", "ext-launch", "--adapter", "synthetic", "--recheck",
                              str(rec)], cwd=str(self.tmp), env=e, capture_output=True, text=True, timeout=60)
        self.assertEqual(chk.returncode, EXIT_REFUSED, describe(chk))
        self.assertIn(("EXT-SPEC", "--spec"), [(x["rule"], x["field"]) for x in parse_errors(chk.stderr)])


# ---------------------------------------------------------------------------
# EXT-85 (L2): the effective host layout is hashed into the record; overrides warn
# ---------------------------------------------------------------------------

class TestHostConfigHash(Launch3):

    def test_recorded_and_warned(self):
        proc, out = self.accepted(golden_request())
        rec = json.loads(out.read_text(encoding="utf-8"))
        data = self.tmp / "host" / "data"
        self.assertEqual(rec["host_config_sha256"],
                         host_config_sha256(data / "gatedtrain-work", data / "gatedtrain-mm" / "out", [data],      # EXT-117
                                            mirror_root=self.tmp / "host", interpreter=GOLDEN_INTERPRETER,
                                            interpreter_trees=[self.tmp / "host" / GOLDEN_TREE]))
        for var in (ENV_HOST_WORK, ENV_OUTPUT_ROOT, ENV_INPUT_ROOTS):
            self.assertIn(f"warning: {var} overrides", proc.stderr)
        self.assertEqual(parse_errors(proc.stderr), [])

    def test_unit(self):
        from gatedtrain.ext.launch import HostConfig, host_config_sha256 as cfg_sha, override_warnings
        self.assertEqual(override_warnings({}), [])
        self.assertEqual(len(override_warnings({ENV_HOST_WORK: "/w", "GATEDTRAIN_EXT_OTHER": "x", "PATH": "/bin"})), 2)
        self.assertEqual(cfg_sha(HostConfig(host_work="/w", output_root="/o", input_roots=("/a", "/b"))),
                         host_config_sha256("/w", "/o", ["/a", "/b"]))
        self.assertNotEqual(cfg_sha(HostConfig(host_work="/w", output_root="/o", input_roots=("/a",))),
                            cfg_sha(HostConfig(host_work="/w", output_root="/o", input_roots=("/a", "/b"))))


class TestHostConfigHashCollect(CollectCase):

    def test_other_input_roots_refused(self):
        sc = self.sc
        args = ["ext-collect", str(sc.paths["spec"]), "--adapter", "synthetic", "--stage-out", str(sc.stage_out),
                "--launch-record", str(sc.launch), "--smoke-record", str(sc.smoke), "--eval-baseline",
                str(sc.baseline), "--eval-candidate", str(sc.candidate), "--runs-root", str(self.runs())]
        proc = run_cli(args, cwd=sc.root, env={**sc.env(), ENV_INPUT_ROOTS: f"{sc.root}{os.pathsep}/home"})
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertIn(("EXT-LAUNCH", "--launch-record.host_config_sha256"),
                      [(e["rule"], e["field"]) for e in parse_errors(proc.stderr)])
        self.assertIn(f"warning: {ENV_INPUT_ROOTS} overrides", proc.stderr)

    def test_record_without_the_hash_refused(self):
        rec = self.sc.launch_record(hashed_label=False)
        rec.pop("host_config_sha256")
        rec["label"] = launch_label(rec)
        self.refused("EXT-LAUNCH", "--launch-record.host_config_sha256",
                     launch=write_canonical(self.path("launch").with_suffix(".json"), rec))


# ---------------------------------------------------------------------------
# EXT-86 (L3, mutant R1): the output root itself is never the output
# ---------------------------------------------------------------------------

class TestOutputIsTheRoot(Launch3):

    def test_launch_empty_root_refused(self):
        host = self.tmp / "host"
        req = golden_request()
        req["output_dir"] = "/data/gatedtrain-mm/empty-root"
        req["mounts"][4]["host"] = "/data/gatedtrain-mm/empty-root"
        rooted(req, host)
        errs = self.refused_at(req, "EXT-LAUNCH", "--request.output_dir",
                               env={ENV_OUTPUT_ROOT: str(host / "data" / "gatedtrain-mm" / "empty-root")})
        self.assertTrue(any("not strictly under" in e["message"] for e in errs), errs)

    def test_unit_record(self):
        root = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-r4-root-"))
        self.addCleanup(shutil.rmtree, root, True)
        req = unit_request(root)
        out_root = Path(req["output_dir"])
        cfg = dataclasses.replace(unit_config(req), output_root=str(out_root))
        rec = dict(req, env_names=["HF_HUB_OFFLINE"])
        msgs = [m for f, m in problems(rec, kind="record", config=cfg) if f == "output_dir"]
        self.assertTrue(any("not strictly under" in m for m in msgs), msgs)


class TestOutputIsTheRootCollect(CollectCase):

    def test_refused(self):
        sc = self.sc
        rec = sc.launch_record(output_root=sc.stage_out)
        guard_log(sc.host_work, rec["label"])
        self.addCleanup(guard_log, sc.host_work, sc.label)
        path = write_canonical(self.path("launch").with_suffix(".json"), rec)
        args = ["ext-collect", str(sc.paths["spec"]), "--adapter", "synthetic", "--stage-out", str(sc.stage_out),
                "--launch-record", str(path), "--smoke-record", str(sc.smoke), "--eval-baseline",
                str(sc.baseline), "--eval-candidate", str(sc.candidate), "--runs-root", str(self.runs())]
        proc = run_cli(args, cwd=sc.root, env={**sc.env(), ENV_OUTPUT_ROOT: str(sc.stage_out)})
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        hits = [e for e in parse_errors(proc.stderr) if (e["rule"], e["field"]) == ("EXT-LAUNCH",
                                                                                  "--launch-record.output_dir")]
        self.assertTrue(any("not strictly under" in h["message"] for h in hits), describe(proc))


# ---------------------------------------------------------------------------
# EXT-87 (L4, mutant R5): each read output alone is checked for staleness
# ---------------------------------------------------------------------------

class TestStaleReadOutputs(CollectCase):

    def test_each_alone(self):
        for name in ("env-freeze.txt", "samples.jsonl", "env-runtime.json"):
            with self.subTest(file=name):
                review3.TestStaleOutputs.aged(self, self.sc.stage_out / name)
                hits = self.refused("EXT-STAGE", "--stage-out")
                self.assertTrue(any(name in h["message"] for h in hits), hits)
                self.doCleanups()


# ---------------------------------------------------------------------------
# EXT-88 (L5): tmpfs `/`, empty argv strings, env values with newline or NUL, mounts nested in a rw mount
# ---------------------------------------------------------------------------

class TestShapes(TmpCase):

    def test_request(self):
        cases = {
            "tmpfs /": (lambda r: r.update(tmpfs=["/tmp", "/"]), "tmpfs"),
            "tmpfs //": (lambda r: r.update(tmpfs=["//"]), "tmpfs"),
            "empty argv string": (lambda r: r["argv"].append(""), "argv"),
            "env value with a newline": (lambda r: r["env"].update(HOME="a\nb"), "env.HOME"),     # round 5: a
            "env value with NUL": (lambda r: r["env"].update(HOME="a\x00b"), "env.HOME"),        # listed name
            "env value with CR": (lambda r: r["env"].update(HOME="a\rb"), "env.HOME"),           # (EXT-92)
            "ro mount nested under the rw output": (lambda r: r["mounts"].append(
                {"host": r["mounts"][2]["host"], "container": "/out/checkpoint-10", "mode": "ro"}), "mounts[5].container"),
            "tmpfs nested under the rw output": (lambda r: r.update(tmpfs=["/tmp", "/out/cache"]), "tmpfs"),
        }
        for name, (mutate, field) in cases.items():
            with self.subTest(case=name):
                req = unit_request(self.root)
                self.assertEqual(fields(req), [])
                mutate(req)
                self.assertIn(field, fields(req))

    def test_accepted(self):
        """Round 5: FOO is outside the allow-list (EXT-92) and a mount inside another is refused (EXT-94), so a listed
        name carries the value and the extra mount is a sibling."""
        req = unit_request(self.root)
        req["env"]["MASTER_ADDR"] = "a b\tc"            # round 6: HOME is path-valued (EXT-114); one tmpfs, /tmp (EXT-112)
        req["mounts"].append({"host": req["mounts"][2]["host"], "container": "/inputs/base-model-sub", "mode": "ro"})
        self.assertEqual(fields(req), [])

    def test_record(self):
        req = unit_request(self.root)
        rec = dict(req, env_names=["HF_HUB_OFFLINE"], tmpfs=["/"])
        self.assertIn("tmpfs", fields(rec, kind="record"))
        rec = dict(req, env_names=["HF_HUB_OFFLINE"], argv=["python", ""])
        self.assertIn("argv", fields(rec, kind="record"))
        rec = copy.deepcopy(dict(req, env_names=["HF_HUB_OFFLINE"]))
        rec["mounts"].append({"host": rec["mounts"][2]["host"], "container": "/out/x", "mode": "ro"})
        self.assertIn("mounts[5].container", fields(rec, kind="record"))

    def test_value_never_printed(self):
        req = unit_request(self.root)
        req["env"]["HOME"] = "marker_value\nsecond"
        msgs = " ".join(m for _, m in problems(req))
        self.assertNotIn("marker_value", msgs)


# ---------------------------------------------------------------------------
# EXT-89 (L6): `python -m gatedtrain.cli` runs the CLI
# ---------------------------------------------------------------------------

class TestModuleMain(unittest.TestCase):

    def run_module(self, *args: str):
        return subprocess.run([str(VENV_PYTHON), "-m", "gatedtrain.cli", *args], cwd=str(SKEL_ROOT), env=cli_env(),
                              capture_output=True, text=True, timeout=60)

    def test_runs(self):
        proc = self.run_module("--version")
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertIn("gatedtrain", proc.stdout)
        proc = self.run_module("ext-launch", "--adapter", "synthetic")
        self.assertEqual(proc.returncode, EXIT_USAGE, describe(proc))
        self.assertIn("--recheck", proc.stderr)

    def test_package_entry_unchanged(self):
        proc = subprocess.run([str(VENV_PYTHON), "-m", "gatedtrain", "--version"], cwd=str(SKEL_ROOT), env=cli_env(),
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertIn("gatedtrain", proc.stdout)


if __name__ == "__main__":
    unittest.main()
