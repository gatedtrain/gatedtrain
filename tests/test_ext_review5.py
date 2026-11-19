"""Interpreter binding, the environment allow-list and launch record compatibility (EXT-91 to EXT-97)."""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import re
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from _ext import guard_log, launch_label, write_canonical
from _support import EXIT_OK, EXIT_REFUSED, describe, file_sha256, parse_errors
from test_ext_hardening import CollectCase
from test_ext_launch import golden_request, launch, rooted
from test_ext_review2 import unit_config, unit_request
from test_ext_review3 import Launch3, problems
import test_ext_review4 as review4

ENV_NAMES = ("DO_NOT_TRACK", "FLASHINFER_WORKSPACE_BASE", "HF_HOME", "HF_HUB_OFFLINE", "HOME", "MASTER_ADDR",
                    "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED", "TOKENIZERS_PARALLELISM", "TORCHINDUCTOR_CACHE_DIR",
                    "TORCH_CUDA_ARCH_LIST", "TRANSFORMERS_OFFLINE", "TRITON_CACHE_DIR", "VLLM_CACHE_ROOT",
                    "VLLM_CONFIG_ROOT", "VLLM_NO_USAGE_STATS", "VLLM_USE_FLASHINFER_SAMPLER", "XDG_CACHE_HOME")


def fields(req: dict, **kw) -> list[str]:
    return [f for f, _ in problems(req, **kw)]


def messages(req: dict, field: str, **kw) -> list[str]:
    return [m for f, m in problems(req, **kw) if f == field]


class TmpCase(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-r5-"))
        self.addCleanup(shutil.rmtree, self.root, True)


# ---------------------------------------------------------------------------
# EXT-91 (M1): the interpreter and the script are argv[0] and argv[1], both files in read-only mounts
# ---------------------------------------------------------------------------

class VenvLayout:
    """A `venv_in_container` layout under `root`, mounted at the host paths: `<root>/py/cpython/bin/
    python3.12` (a file), `<root>/a1/.venv/bin/python` (an absolute link to it), `<root>/a1/train.py`,
    `<root>/a1/uv.lock`, and the read-write output `<root>/out` at `/out` holding `python` and `run.py`."""

    def __init__(self, root: Path):
        self.py = root / "py"
        self.a1 = root / "a1"
        self.out = root / "out"
        self.real_python = self.py / "cpython" / "bin" / "python3.12"
        self.real_python.parent.mkdir(parents=True)
        self.real_python.write_text("# interpreter stand-in\n", encoding="utf-8")
        (self.a1 / ".venv" / "bin").mkdir(parents=True)
        self.python = self.a1 / ".venv" / "bin" / "python"
        self.python.symlink_to(self.real_python)
        self.script = self.a1 / "train.py"
        self.script.write_text("# script stand-in\n", encoding="utf-8")
        (self.a1 / "uv.lock").write_text("lock\n", encoding="utf-8")
        (self.a1 / "run.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        self.out.mkdir()
        (self.out / "python").write_text("", encoding="utf-8")
        (self.out / "run.py").write_text("", encoding="utf-8")
        self.mounts = [{"host": str(self.py), "container": str(self.py), "mode": "ro"},
                       {"host": str(self.a1), "container": str(self.a1), "mode": "ro"},
                       {"host": str(self.out), "container": "/out", "mode": "rw"}]


class TestArgvBindingUnit(TmpCase):

    def setUp(self):
        super().setUp()
        self.lay = VenvLayout(self.root)

    def bind(self, argv, mounts=None):
        from gatedtrain.ext.launch import bind_argv
        return bind_argv(argv, self.lay.mounts if mounts is None else mounts)

    def test_venv_shape_bound(self):
        """The interpreter link resolves through the read-only mounts, as in the container; the record keeps the
        resolved interpreter (the file it hashes) and the script as written."""
        b, why = self.bind([str(self.lay.python), str(self.lay.script), "--profile", "full", "--out", "/out"])
        self.assertIsNone(why)
        self.assertEqual((b.interpreter_path, b.interpreter_host), (str(self.lay.real_python), str(self.lay.real_python)))
        self.assertEqual((b.script_path, b.script_host), (str(self.lay.script), str(self.lay.script)))

    def test_argv_with_unbound_interpreter_or_script_refused(self):
        py, script = str(self.lay.python), str(self.lay.script)
        cases = {
            "python -c code script": [py, "-c", "import runpy; runpy.run_path('x')", script],
            "python -m evil script": [py, "-m", "evil", script],
            "/bin/sh -c ... script": ["/bin/sh", "-c", "true", script],
            "interpreter from the rw output": ["/out/python", script],
            "python -c pass uv.lock": [py, "-c", "pass", str(self.lay.a1 / "uv.lock")],
            "python - script": [py, "-", script],
            "python -u script (an option before the script)": [py, "-u", script],
            "python -I script": [py, "-I", script],
            "a non-.py file as the script": [py, str(self.lay.a1 / "uv.lock")],
            "a script in the rw output": [py, "/out/run.py"],
            "a missing script": [py, str(self.lay.a1 / "absent.py")],
            "a relative interpreter": ["python", script],
            "an interpreter that is not python": [str(self.lay.a1 / "run.sh"), script],
            "no script": [py],
            "not a list": "python train.py",
            "the interpreter as the script": [py, py],
        }
        for name, argv in cases.items():
            with self.subTest(case=name):
                b, why = self.bind(argv)
                self.assertIsNone(b)
                self.assertTrue(why and "EXT-91" in why, why)
                if isinstance(argv, list) and len(argv) > 1 and argv[1].startswith("-") and argv[0] == py:
                    self.assertIn("is an option", why)          # named as such, not as a missing file

    def test_container_resolution(self):
        """Links resolve as the container sees them: an absolute target is a container path, so a link valid on
        the host but pointing outside every read-only mount inside the container is refused."""
        moved = [{"host": str(self.lay.py), "container": "/py", "mode": "ro"},
                 {"host": str(self.lay.a1), "container": "/code", "mode": "ro"},
                 {"host": str(self.lay.out), "container": "/out", "mode": "rw"}]
        b, why = self.bind(["/code/.venv/bin/python", "/code/train.py"], moved)
        self.assertIsNone(b)
        self.assertIn("argv[0]", why)
        rel = self.lay.a1 / ".venv" / "bin" / "python-rel"            # a relative link stays inside the mount
        os.symlink("../../train.py", self.lay.a1 / ".venv" / "bin" / "script-rel.py")
        rel.symlink_to("python")
        b, why = self.bind(["/code/.venv/bin/python-rel", "/code/train.py"], moved)
        self.assertIsNone(b)                                          # still the host-absolute link behind it
        (self.lay.a1 / "bin").mkdir()
        os.symlink("/py/cpython/bin/python3.12", self.lay.a1 / "bin" / "python3")
        b, why = self.bind(["/code/bin/python3", "/code/.venv/bin/script-rel.py"], moved)
        self.assertIsNone(why)
        self.assertEqual((b.interpreter_path, b.interpreter_host), ("/py/cpython/bin/python3.12",
                                                                     str(self.lay.real_python)))
        self.assertEqual((b.script_path, b.script_host), ("/code/.venv/bin/script-rel.py", str(self.lay.script)))

    def test_links_out_of_the_mounts_refused(self):
        (self.root / "elsewhere.py").write_text("x", encoding="utf-8")
        os.symlink(self.root / "elsewhere.py", self.lay.a1 / "escape.py")
        os.symlink("/out/run.py", self.lay.a1 / "into-out.py")
        os.symlink("/etc/hostname", self.lay.a1 / "etc.py")
        os.symlink("loop.py", self.lay.a1 / "loop.py")
        for name in ("escape.py", "into-out.py", "etc.py", "loop.py"):
            with self.subTest(link=name):
                b, why = self.bind([str(self.lay.python), str(self.lay.a1 / name)])
                self.assertIsNone(b)
                self.assertIn("argv[1]", why)

    def test_nested_mount_never_binds(self):
        """Q4 (round 5): with nested mounts refused (EXT-94) no file resolves through two mounts; a read-only file
        mounted on the script's container path makes the binding fail rather than pick either mount."""
        other = self.root / "other"
        other.mkdir()
        (other / "train.py").write_text("# other\n", encoding="utf-8")
        mounts = self.lay.mounts + [{"host": str(other / "train.py"), "container": str(self.lay.script),
                                     "mode": "ro"}]
        b, why = self.bind([str(self.lay.python), str(self.lay.script)], mounts)
        self.assertIsNone(b)
        self.assertIn("argv[1]", why)


class TestArgvBindingLaunch(Launch3):

    def test_recorded(self):
        _, out = self.accepted(golden_request())
        rec = json.loads(out.read_text(encoding="utf-8"))
        a1 = self.tmp / "host" / "data" / "gatedtrain-mm" / "synthetic-runtime"
        self.assertEqual(rec["interpreter_path"], "/data/gatedtrain-mm/synthetic-runtime/.venv/bin/python")
        self.assertEqual(rec["interpreter_sha256"], file_sha256(a1 / ".venv" / "bin" / "python"))
        self.assertEqual(rec["script_path"], "/data/gatedtrain-mm/synthetic-runtime/train.py")

    def test_refused(self):
        py, script = "/data/gatedtrain-mm/synthetic-runtime/.venv/bin/python", "/data/gatedtrain-mm/synthetic-runtime/train.py"
        for name, argv in {"python -c": [py, "-c", "print(1)", script], "python -m": [py, "-m", "evil", script],
                           "/bin/sh": ["/bin/sh", "-c", "true", script], "rw interpreter": ["/out/python", script],
                           "uv.lock": [py, "-c", "pass", "/data/gatedtrain-mm/synthetic-runtime/uv.lock"]}.items():
            with self.subTest(case=name):
                req = golden_request()
                req["argv"] = argv
                self.refused_at(req, "EXT-LAUNCH", "--request.argv")

    def test_recheck_interpreter_changed(self):
        sub = self.sub()
        proc, rec = launch(sub, golden_request(), root=self.tmp / "host")
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        py = self.tmp / "host" / "data" / "gatedtrain-mm" / "synthetic-runtime" / ".venv" / "bin" / "python"
        old = py.read_bytes()
        py.write_bytes(old + b"# patched\n")
        self.addCleanup(py.write_bytes, old)
        chk = review4.TestRecheckGpuUsed.recheck(self, rec)
        self.assertEqual(chk.returncode, EXIT_REFUSED, describe(chk))
        self.assertIn(("EXT-LAUNCH", "--recheck.interpreter_sha256"),
                      [(e["rule"], e["field"]) for e in parse_errors(chk.stderr)])


class Relabel:
    def relabelled(self, rec: dict) -> Path:
        rec = dict(rec, label=rec["label"].rsplit("-", 1)[0])
        rec["label"] = launch_label(rec)
        guard_log(self.sc.host_work, rec["label"])
        self.addCleanup(guard_log, self.sc.host_work, self.sc.label)
        return write_canonical(self.path("launch").with_suffix(".json"), rec)


class TestArgvBindingCollect(Relabel, CollectCase):

    def test_interpreter_changed_after_launch(self):
        old = self.sc.interpreter.read_bytes()
        self.sc.interpreter.write_bytes(old + b"# patched\n")
        self.addCleanup(self.sc.interpreter.write_bytes, old)
        self.refused("EXT-LAUNCH", "--launch-record.interpreter_sha256")

    def test_record_fields(self):
        for name, (mutate, field) in {
            "argv -c before the script": (lambda r: r.update(argv=[r["argv"][0], "-c", "x", *r["argv"][1:]]),
                                          "--launch-record.argv"),
            "interpreter_path not the resolved argv[0]": (lambda r: r.update(interpreter_path="/code/python"),
                                                          "--launch-record.interpreter_path"),
            "interpreter_sha256 another file's": (lambda r: r.update(interpreter_sha256="sha256:" + "0" * 64),
                                                  "--launch-record.interpreter_sha256"),
        }.items():
            with self.subTest(case=name):
                rec = self.sc.launch_record()
                mutate(rec)
                self.refused("EXT-LAUNCH", field, launch=self.relabelled(rec))
                self.doCleanups()


# ---------------------------------------------------------------------------
# EXT-92 (L1): environment names come from the adapter's allow-list
# ---------------------------------------------------------------------------

class TestEnvAllowList(TmpCase):

    UNLISTED_ENV_NAMES = ("PYTHONPATH", "PYTHONHOME", "PYTHONWARNINGS", "PYTHONBREAKPOINT", "PYTHONSTARTUP",
                          "BASH_ENV", "ENV", "VLLM_PLUGINS", "LD_AUDIT", "LD_PRELOAD", "GCONV_PATH", "PATH",
                          "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "DOCKER_HOST", "hf_home",
                          "PYTORCH_CUDA_ALLOC_CONF")

    def test_every_listed_name_accepted(self):
        req = unit_request(self.root)
        self.assertEqual(sorted(req["env"]), list(ENV_NAMES))
        self.assertEqual(fields(req), [])
        rec = dict(req, env_names=sorted(req["env"]))
        self.assertNotIn("env_names", fields(rec, kind="record"))

    def test_others_refused(self):
        for name in self.UNLISTED_ENV_NAMES:
            with self.subTest(name=name):
                req = unit_request(self.root)
                req["env"][name] = "/out/marker_value"
                msgs = messages(req, f"env.{name}")
                self.assertTrue(msgs and any("allow-list" in m for m in msgs), msgs)
                self.assertNotIn("marker_value", " ".join(m for _, m in problems(req)))
                rec = dict(req, env_names=sorted(req["env"]))
                self.assertTrue(any("allow-list" in m for m in messages(rec, "env_names", kind="record")))

    def test_credential_message_kept(self):
        req = unit_request(self.root)
        req["env"]["HF_TOKEN"] = "x"
        self.assertTrue(any("credential" in m for m in messages(req, "env.HF_TOKEN")))

    def test_without_an_adapter_nothing_allowed(self):
        req = unit_request(self.root)
        cfg = dataclasses.replace(unit_config(req), env_names=None)
        self.assertIn("env.HF_HOME", fields(req, config=cfg))

    def test_adapter_config_fills_it_under_an_override(self):
        """The input-roots override does not drop the adapter's allow-list, skip list or interpreter trees."""
        from gatedtrain.ext import registry
        from gatedtrain.ext.launch import HostConfig, adapter_config
        cfg = adapter_config(HostConfig(input_roots=("/a",)), "synthetic")
        self.assertEqual(cfg.input_roots, ("/a",))
        self.assertEqual(cfg.env_names, registry.env_names("synthetic"))
        self.assertEqual(cfg.scan_skip, registry.scan_skip("synthetic"))
        self.assertEqual(cfg.interpreter_trees, registry.interpreter_trees("synthetic"))


class TestEnvAllowListLaunch(Launch3):

    def test_refused(self):
        req = golden_request()
        req["env"]["PYTHONPATH"] = "/out"
        self.refused_at(req, "EXT-LAUNCH", "--request.env.PYTHONPATH")


# ---------------------------------------------------------------------------
# EXT-93 (L2, Q1): credential file names, NFKC, the adapter's skip list, the stdlib exemption in the interpreter tree
# ---------------------------------------------------------------------------

class TestCredentialNames5(TmpCase):

    n = 0

    def source(self, rel: str, *, directory: bool = False) -> Path:
        type(self).n += 1
        src = self.root / "data" / "inputs" / f"x{type(self).n}"
        p = src / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if directory:
            p.mkdir()
        else:
            p.write_text("fixture\n", encoding="utf-8")
        return src

    def check(self, rel: str, **kw) -> str | None:
        from gatedtrain.ext.launch import credential_name_problem
        return credential_name_problem(str(self.source(rel)), **kw)

    def test_new_names_refused(self):
        for rel in ("id_ecdsa", "id_ecdsa.pub", "id_dsa", "id_dsa.pub", ".npmrc", ".pypirc", "home/.docker/config.json",
                    ".kube/config", "cert.p12", "CERT.PFX", ".s3cfg", ".boto", "gh/hosts.yml", ".config/gh/hosts.yml",
                    "stored_tokens", "ＩＤ_rsa", "ｉｄ＿ｒｓａ", ".ＥＮＶ",
                    "．docker/config.json"):
            with self.subTest(name=rel):
                why = self.check(rel)
                self.assertTrue(why and "credential" in why, why)

    def test_context_names_allowed_elsewhere(self):
        for rel in ("config.json", "docker/config.json", "config", "kube/config", "hosts.yml", "github/hosts.yml",
                    "tokens.txt", "p12.txt"):
            with self.subTest(name=rel):
                self.assertIsNone(self.check(rel))

    def test_any_venv_or_site_packages_scanned(self):
        """Round 5 L2: a directory named `.venv` or `site-packages` is no longer skipped by name."""
        for rel in ("export/.venv/.env", "lib/python3.12/site-packages/pkg/secret.py", "a/.venv/token"):
            with self.subTest(name=rel):
                self.assertIsNotNone(self.check(rel))

    def test_configured_skip_only(self):
        src = self.source("v/.venv/lib/x/.env")
        (src / "w" / ".venv").mkdir(parents=True)
        from gatedtrain.ext.launch import credential_name_problem
        self.assertIsNone(credential_name_problem(str(src), skip=(str(src / "v" / ".venv"),)))
        (src / "w" / ".venv" / "token").write_text("", encoding="utf-8")
        why = credential_name_problem(str(src), skip=(str(src / "v" / ".venv"),))
        self.assertIn("token", why)

    def test_stdlib_secrets_in_the_interpreter_tree_only(self):
        """Q1 (round 5): the exemption needs `<tree>/lib/python3.N/` exactly, `<tree>` a declared interpreter tree."""
        from gatedtrain.ext.launch import credential_name_problem
        type(self).n += 1
        src = self.root / "data" / "inputs" / f"x{type(self).n}"
        tree = src / "cpython-3.12.16"
        ok = ("lib/python3.12/secrets.py", "lib/python3.12/__pycache__/secrets.cpython-312.pyc",
              "lib/python3.12/__pycache__/secrets.cpython-312.opt-1.pyc")
        for rel in ok:
            (tree / rel).parent.mkdir(parents=True, exist_ok=True)
            (tree / rel).write_text("", encoding="utf-8")
        self.assertIsNone(credential_name_problem(str(src), interpreter_trees=(str(tree),)))
        self.assertIsNotNone(credential_name_problem(str(src)))                   # no declared tree, no exemption
        self.assertIsNotNone(credential_name_problem(str(src), interpreter_trees=(str(src / "other"),)))
        for bad in ("foo/python3.12/secrets.py", "x/lib/python3.12/secrets.py", "lib/python3.12/sub/secrets.py",
                    "lib/pythonX/secrets.py", "lib/python3.12/__pycache__/secrets.pyc",
                    "lib/python3.12/Secrets.py"):
            with self.subTest(name=bad):
                p = tree / bad
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("", encoding="utf-8")
                why = credential_name_problem(str(src), interpreter_trees=(str(tree),))
                self.assertTrue(why and Path(bad).name in why, why)
                p.unlink()

    def test_through_isolation_problems(self):
        """The configured skip list and trees reach the mount scan (HostConfig)."""
        req = unit_request(self.root)
        a1 = Path(req["mounts"][1]["host"])
        (a1 / ".venv" / "lib").mkdir(parents=True)
        (a1 / ".venv" / "lib" / "credentials.py").write_text("", encoding="utf-8")
        self.assertIn("mounts[1].host", fields(req))
        cfg = dataclasses.replace(unit_config(req), scan_skip=(str(a1 / ".venv"),))
        self.assertNotIn("mounts[1].host", fields(req, config=cfg))

# ---------------------------------------------------------------------------
# EXT-94 (L3, Q4): no mount or tmpfs inside or at another's container path, in either direction
# ---------------------------------------------------------------------------

class TestNoNesting(TmpCase):

    def test_refused(self):
        def add(container, mode="ro", index=2):
            def mutate(r):
                r["mounts"].append({"host": r["mounts"][index]["host"], "container": container, "mode": mode})
            return mutate

        def rw_into(container):
            def mutate(r):
                r["mounts"][4]["container"] = container
            return mutate
        a1 = "{a1}"
        cases = {
            "rw output inside a ro mount": (rw_into(f"{a1}/.venv/lib/python3.12/site-packages"), "mounts[4].container"),
            "ro inside ro": (add("/inputs/base-model/sub"), "mounts[5].container"),
            "ro onto a ro mount's venv": (add(f"{a1}/.venv", index=3), "mounts[5].container"),
            "ro parent of a ro mount": (add("/inputs"), "mounts[2].container"),
            "ro file onto the script (Q4)": (add(f"{a1}/train.py"), "mounts[5].container"),
            "tmpfs inside a ro mount": (lambda r: r["tmpfs"].append("/inputs/export/tmp"), "tmpfs"),
            "ro inside the tmpfs": (add("/tmp/inputs"), "mounts[5].container"),
            "ro inside the rw output": (add("/out/checkpoint-10"), "mounts[5].container"),
            "tmpfs inside the rw output": (lambda r: r["tmpfs"].append("/out/cache"), "tmpfs"),
            "ro equal to a tmpfs": (add("/tmp"), "mounts[5].container"),
        }
        for name, (mutate, field) in cases.items():
            with self.subTest(case=name):
                req = unit_request(self.root)
                self.assertEqual(fields(req), [])
                a1_container = req["mounts"][1]["container"]
                mutate(req)
                for m in req["mounts"]:
                    m["container"] = m["container"].replace("{a1}", a1_container)
                self.assertIn(field, fields(req))
                rec = copy.deepcopy(dict(req, env_names=sorted(req.pop("env"))))
                self.assertIn(field, fields(rec, kind="record"))

    def test_siblings_accepted(self):
        req = unit_request(self.root)
        req["mounts"].append({"host": req["mounts"][2]["host"], "container": "/inputs/base-model-2", "mode": "ro"})
        self.assertEqual(fields(req), [])


# ---------------------------------------------------------------------------
# EXT-95 (M2): the launch record schema version; older records refuse, never upgraded
# ---------------------------------------------------------------------------

class TestSchemaVersion(TmpCase):

    def test_current(self):
        from gatedtrain.ext.launch import SCHEMA_VERSION
        self.assertEqual(SCHEMA_VERSION, "2.0")

    def test_old_record_one_clear_problem(self):
        from gatedtrain.ext.launch import record_problems
        rec = {"schema": "gatedtrain.launch_record", "schema_version": "1.0", "label": "x"}
        got = record_problems(rec, adapter="synthetic", stage_out=None)
        self.assertEqual([p.field for p in got], ["--launch-record.schema_version"])
        self.assertIn("'1.0'", got[0].message)
        self.assertIn("predates", got[0].message)
        self.assertIn("2.0", got[0].message)

class TestSchemaVersionCollect(Relabel, CollectCase):

    def test_collect(self):
        hits = self.refused("EXT-LAUNCH", "--launch-record.schema_version",
                            launch=self.relabelled(self.sc.launch_record(schema_version="1.0")))
        self.assertIn("predates", hits[0]["message"])


class TestSchemaVersionRecheck(Launch3):

    def test_recheck(self):
        sub = self.sub()
        proc, rec = launch(sub, golden_request(), root=self.tmp / "host")
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        doc = json.loads(rec.read_text(encoding="utf-8"))
        doc["schema_version"] = "1.0"
        doc["gpu_used_h"] = 9.99                        # would be a second problem if anything else were judged
        doc["label"] = launch_label(dict(doc, label=doc["label"].rsplit("-", 1)[0]))
        rec.write_text(json.dumps(doc), encoding="utf-8")
        chk = review4.TestRecheckGpuUsed.recheck(self, rec)
        self.assertEqual(chk.returncode, EXIT_REFUSED, describe(chk))
        errs = parse_errors(chk.stderr)
        self.assertEqual([(e["rule"], e["field"]) for e in errs], [("EXT-LAUNCH", "--recheck.schema_version")])
        self.assertIn("predates", errs[0]["message"])


# ---------------------------------------------------------------------------
# EXT-96 (Q5): --recheck compares gpu_used_h exactly
# ---------------------------------------------------------------------------

class TestRecheckExact(Launch3):

    def test_hundredth_refused(self):
        sub = self.sub()
        proc, rec = launch(sub, golden_request(), root=self.tmp / "host")
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        doc = json.loads(rec.read_text(encoding="utf-8"))
        self.assertEqual(doc["gpu_used_h"], 3.12)
        for used in (3.13, 3.11):
            with self.subTest(gpu_used_h=used):
                d = dict(doc, gpu_used_h=used)
                d["label"] = launch_label(dict(d, label=d["label"].rsplit("-", 1)[0]))
                path = sub / f"rec-{used}.json"
                write_canonical(path, d)
                chk = review4.TestRecheckGpuUsed.recheck(self, path)
                self.assertEqual(chk.returncode, EXIT_REFUSED, describe(chk))
                self.assertIn(("EXT-GPU", "--recheck.gpu_used_h"),
                              [(e["rule"], e["field"]) for e in parse_errors(chk.stderr)])
        ok = review4.TestRecheckGpuUsed.recheck(self, rec)
        self.assertEqual(ok.returncode, EXIT_OK, describe(ok))


# ---------------------------------------------------------------------------
# EXT-97: network_host_reason is retired (host network is always refused)
# ---------------------------------------------------------------------------

class TestNetworkHostReasonRetired(Launch3):

    def test_keys(self):
        from gatedtrain.ext.launch import RECORD_KEYS, REQUEST_KEYS, RETIRED_REQUEST_KEYS
        self.assertNotIn("network_host_reason", RECORD_KEYS)
        self.assertNotIn("network_host_reason", REQUEST_KEYS)
        self.assertEqual(RETIRED_REQUEST_KEYS, ("network_host_reason",))
        self.assertIn("interpreter_path", RECORD_KEYS)
        self.assertIn("interpreter_sha256", RECORD_KEYS)

    def test_null_accepted_not_recorded(self):
        req = golden_request()
        self.assertNotIn("network_host_reason", req)
        req["network_host_reason"] = None
        _, out = self.accepted(req)
        self.assertNotIn("network_host_reason", json.loads(out.read_text(encoding="utf-8")))

    def test_value_refused(self):
        req = golden_request()
        req["network_host_reason"] = "no bridge"
        self.refused_at(req, "EXT-LAUNCH", "--request.network_host_reason")


class TestNetworkHostReasonCollect(Relabel, CollectCase):

    def test_record_key_unknown(self):
        rec = self.sc.launch_record()
        rec["network_host_reason"] = None
        hits = self.refused("EXT-LAUNCH", "--launch-record.network_host_reason", launch=self.relabelled(rec))
        self.assertIn("unknown", hits[0]["message"])


if __name__ == "__main__":
    unittest.main()
