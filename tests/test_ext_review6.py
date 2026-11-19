"""Container mount paths, interpreter pins and environment paths (EXT-112 to EXT-117).

The fixture mounts the runtime and uv-managed Python tree at their host paths
so links inside the virtual environment resolve in the container.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from _ext import ExtScenario, launch_label
from _support import EXIT_OK, EXIT_REFUSED, HOST_ENV, describe, parse_errors
from test_ext_launch import GOLDEN_INTERPRETER, golden_request
from test_ext_review2 import unit_config, unit_request
from test_ext_review3 import Launch3, problems
from test_ext_review5 import VenvLayout

# Sensitive paths for the loader, image entrypoint and Python library.
SYSTEM_PATHS = ("/etc/ld.so.preload", "/opt/nvidia/nvidia_entrypoint.sh", "/usr/lib/python3")
# EXT-114: the synthetic adapter's path-valued variables (homes and caches).
PATH_ENV = ("FLASHINFER_WORKSPACE_BASE", "HF_HOME", "HOME", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
                  "VLLM_CACHE_ROOT", "VLLM_CONFIG_ROOT", "XDG_CACHE_HOME")


def fields(req: dict, **kw) -> list[str]:
    return [f for f, _ in problems(req, **kw)]


def as_record(req: dict) -> dict:
    """The request's isolation-relevant fields as a record carries them (env names only)."""
    rec = {k: v for k, v in req.items() if k != "env"}
    rec["env_names"] = sorted(req["env"])
    rec["allow_network"] = False
    rec["label"] = launch_label(rec)                          # EXT-46: hashed over the record
    return rec


class TmpCase(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-r6-"))
        self.addCleanup(shutil.rmtree, self.root, True)


# ---------------------------------------------------------------------------
# EXT-112 (Medium-1): container paths come from the adapter's registry row; system directories are never mounted over
# ---------------------------------------------------------------------------

class TestContainerPathsUnit(TmpCase):

    def request(self) -> dict:
        req = unit_request(self.root)
        (Path(req["mounts"][1]["host"]) / "README.md").write_text("# readme\n", encoding="utf-8")
        return req

    def add(self, req: dict, container: str, *, host: str | None = None, mode: str = "ro") -> str:
        req["mounts"].append({"host": host or f"{req['mounts'][1]['host']}/README.md", "container": container,
                              "mode": mode})
        return f"mounts[{len(req['mounts']) - 1}].container"

    def test_golden_accepted(self):
        """The README shape: the Python and runtime directories at their own paths (`<root>` stands for `/`), the
        base model and export under /inputs/, the output at /out, one tmpfs at /tmp."""
        req = self.request()
        self.assertEqual(fields(req), [])
        self.assertEqual(fields(as_record(req), kind="record"), [])

    def test_system_mounts_refused(self):
        for container in SYSTEM_PATHS:
            with self.subTest(container=container):
                req = self.request()
                field = self.add(req, container)
                self.assertIn(field, fields(req))
                self.assertIn(field, fields(as_record(req), kind="record"))

    def test_system_directories_refused_whatever_the_registry(self):
        """`/` and anything at or under the system directories is refused even when the registry would allow it."""
        permissive = {"ro_identity": True, "ro_under": ("/",), "rw": ("/out",), "tmpfs": ("/tmp",)}
        for container in ("/", "/etc", "/etc/ld.so.preload", "/usr/lib/python3", "/bin/sh", "/sbin/init", "/lib",
                          "/lib/x86_64-linux-gnu/libc.so.6", "/lib64/ld-linux-x86-64.so.2", "/libx32/a", "/opt/nvidia",
                          "/proc/self", "/sys/kernel", "/dev/shm", "/run/x", "/var/run/docker.sock", "/root/.bashrc",
                          "/home/u/.bashrc", "//etc/passwd", "/inputs/../etc/passwd", "/srv/../usr/bin/python3"):
            with self.subTest(container=container):
                req = self.request()
                field = self.add(req, container)
                cfg = unit_config(req, container_paths=permissive)
                self.assertIn(field, fields(req, config=cfg))
                self.assertIn(field, fields(as_record(req), kind="record", config=cfg))
        req = self.request()
        field = self.add(req, "/srv/inputs-x")
        self.assertNotIn(field, fields(req, config=unit_config(req, container_paths=permissive)))

    def test_system_directories_refused_for_tmpfs_and_rw(self):
        """System directory restrictions apply even when the registry permits the requested path."""
        req = self.request()
        req["tmpfs"] = ["/tmp", "/var/tmp"]
        cfg = unit_config(req, container_paths={"ro_identity": True, "ro_under": ("/inputs",), "rw": ("/out",),
                                                "tmpfs": ("/tmp", "/var/tmp")})
        self.assertTrue(any("/var" in m for f, m in problems(req, config=cfg) if f == "tmpfs"))
        req = self.request()
        out_root = self.root / "var"
        out = out_root / "run-1"
        out.mkdir(parents=True)
        req["output_dir"] = str(out)
        req["mounts"][4].update(host=str(out), container="/var/run-1")
        cfg = dataclasses.replace(
            unit_config(req, container_paths={"ro_identity": True, "ro_under": ("/inputs",), "rw": ("/var/run-1",),
                                              "tmpfs": ("/tmp",)}),
            output_root=str(out_root), input_roots=(str(self.root / "data"), str(out_root)))
        msgs = [m for f, m in problems(req, config=cfg) if f == "mounts[4].container"]
        self.assertTrue(any("/var" in m for m in msgs), problems(req, config=cfg))

    def test_adapter_config_fills_the_rows(self):
        """Use registry defaults when the host settings are absent."""
        from gatedtrain.ext import registry
        from gatedtrain.ext.launch import HostConfig, adapter_config, host_config
        self.assertEqual((host_config(HOST_ENV).mirror_root, host_config(HOST_ENV).interpreter), (None, None))
        for adapter in ("synthetic",):
            with self.subTest(adapter=adapter):
                cfg = adapter_config(HostConfig(input_roots=("/a",)), adapter)
                self.assertEqual((cfg.interpreter, dict(cfg.container_paths), cfg.env_paths),
                                 (registry.interpreter(adapter), registry.container_paths(adapter),
                                  registry.env_paths(adapter)))
        cfg = adapter_config(host_config({**HOST_ENV, "GATEDTRAIN_EXT_INTERPRETER": "/x/bin/python",
                                          "GATEDTRAIN_EXT_INTERPRETER_TREES": "/x", "GATEDTRAIN_EXT_MIRROR_ROOT": "/m"}),
                             "synthetic")
        self.assertEqual((cfg.interpreter, cfg.interpreter_trees, cfg.mirror_root), ("/x/bin/python", ("/x",), "/m"))

    def test_adapter_rules(self):
        """Read-only: at the host path inside an input root, or strictly under /inputs/; read-write: /out only;
        tmpfs: /tmp only."""
        for name, mutate, field, refused in (
            ("ro elsewhere", lambda r: self.add(r, "/code"), "mounts[5].container", True),
            ("ro at /inputs itself", lambda r: self.add(r, "/inputs"), "mounts[5].container", True),
            ("ro inside /out", lambda r: self.add(r, "/out/x"), "mounts[5].container", True),
            ("ro inside /tmp", lambda r: self.add(r, "/tmp/x"), "mounts[5].container", True),
            ("ro under /inputs/", lambda r: self.add(r, "/inputs/readme"), "mounts[5].container", False),
            ("identity mount moved", lambda r: r["mounts"][0].update(container="/data/gatedtrain-mm/python2"),
             "mounts[0].container", True),
            ("identity of another mount's host", lambda r: r["mounts"][0].update(container="/data/gatedtrain-mm/synthetic-runtime"),
             "mounts[0].container", True),
            ("rw elsewhere", lambda r: r["mounts"][4].update(container="/output"), "mounts[4].container", True),
            ("rw at /tmp", lambda r: r["mounts"][4].update(container="/tmp/out"), "mounts[4].container", True),
            ("tmpfs elsewhere", lambda r: r.update(tmpfs=["/scratch"]), "tmpfs", True),
            ("tmpfs at /var/tmp", lambda r: r.update(tmpfs=["/var/tmp"]), "tmpfs", True),
            ("tmpfs /tmp/ (normal form /tmp)", lambda r: r.update(tmpfs=["/tmp/"]), "tmpfs", False),
            ("no tmpfs", lambda r: r.update(tmpfs=[]), "tmpfs", False),
        ):
            with self.subTest(case=name):
                req = self.request()
                mutate(req)
                (self.assertIn if refused else self.assertNotIn)(field, fields(req))
                (self.assertIn if refused else self.assertNotIn)(field, fields(as_record(req), kind="record"))

    def test_identity_needs_an_input_root(self):
        """An identity mount whose host path lies in no input root is refused at the container path too."""
        req = self.request()
        cfg = dataclasses.replace(unit_config(req), input_roots=(req["mounts"][1]["host"],
                                                                 str(Path(req["host_work"]) / "ext")))
        got = fields(req, config=cfg)
        self.assertIn("mounts[0].container", got)
        self.assertNotIn("mounts[1].container", got)

    def test_production_identity(self):
        """Without a mirror root (the host default) an identity mount is container path == host path."""
        req = self.request()
        req["tmpfs"] = []                                     # the test root lies under /tmp
        for m in req["mounts"][:2]:
            m["container"] = m["host"]
        cfg = dataclasses.replace(unit_config(req), mirror_root=None)
        got = fields(req, config=cfg)
        self.assertNotIn("mounts[0].container", got)
        self.assertNotIn("mounts[1].container", got)
        req["mounts"][0]["container"] = req["mounts"][0]["host"] + "-x"
        self.assertIn("mounts[0].container", fields(req, config=cfg))

    def test_home_only_as_an_input_roots_identity(self):
        """`/home/...` is allowed only as the identity path of an input root (a runtime under
        `/home` is mounted at its own path); `<root>` stands for `/` here."""
        req = self.request()
        inp = self.root / "home" / "u" / "data" / "inp"
        inp.mkdir(parents=True)
        req["mounts"].append({"host": str(inp), "container": "/home/u/data/inp", "mode": "ro"})
        cfg = dataclasses.replace(unit_config(req, home_dirs=("/root",)), input_roots=(str(self.root / "data"), str(inp)))
        self.assertNotIn("mounts[5].container", fields(req, config=cfg))
        req["mounts"][5]["container"] = "/home/u/data/other"
        self.assertIn("mounts[5].container", fields(req, config=cfg))

    def test_without_an_adapter_nothing_allowed(self):
        req = self.request()
        cfg = dataclasses.replace(unit_config(req), container_paths=None)
        got = fields(req, config=cfg)
        for i in range(5):
            self.assertIn(f"mounts[{i}].container", got)

    def test_registry_rows(self):
        from gatedtrain.ext import registry
        want = {"ro_identity": True, "ro_under": ("/inputs",), "rw": ("/out",), "tmpfs": ("/tmp",)}
        for adapter in ("synthetic",):
            with self.subTest(adapter=adapter):
                self.assertEqual(dict(registry.container_paths(adapter)), want)


class TestContainerPathsLaunch(Launch3):

    def test_system_mounts_refused(self):
        for container in SYSTEM_PATHS:
            with self.subTest(container=container):
                req = golden_request()
                req["mounts"].append({"host": "/data/gatedtrain-mm/synthetic-runtime/README.md", "container": container,
                                      "mode": "ro"})
                self.refused_at(req, "EXT-LAUNCH", "--request.mounts[5].container")

    def test_golden_accepted(self):
        self.accepted(golden_request())



class TestContainerPathsTrlRecord(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-r6-syn-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.sc = ExtScenario(cls.tmp / "sc")

    def check(self, rec: dict) -> list[str]:
        from gatedtrain.ext.launch import host_config, record_problems
        return [p.field for p in record_problems(rec, adapter="synthetic", stage_out=self.sc.stage_out,
                                                 framework="synthetic-trainer", config=host_config(self.sc.env()))]

    def test_record_mode(self):
        self.assertEqual(self.check(self.sc.launch_record()), [])
        for container in SYSTEM_PATHS:
            with self.subTest(container=container):
                mounts = self.sc.launch_record()["mounts"]
                rec = self.sc.launch_record(mounts=mounts + [{"host": str(self.sc.script), "container": container,
                                                              "mode": "ro"}])
                self.assertIn(f"--launch-record.mounts[{len(mounts)}].container", self.check(rec))

    def test_interpreter_pinned_in_record_mode(self):
        """EXT-113: the record's argv[0] is the pinned interpreter, at ext-collect and --recheck too."""
        other = self.sc.code / "bin" / "python"
        other.write_text("# another interpreter stand-in\n", encoding="utf-8")
        self.addCleanup(other.unlink)
        from _ext import file_sha256
        rec = self.sc.launch_record()
        argv = ["/inputs/code/bin/python", *rec["argv"][1:]]
        rec = self.sc.launch_record(argv=argv, interpreter_path="/inputs/code/bin/python",
                                    interpreter_sha256=file_sha256(other))
        self.assertIn("--launch-record.argv", self.check(rec))


# ---------------------------------------------------------------------------
# EXT-113 (Low-1): argv[0] is the registry's interpreter path; its resolved file lies in an interpreter tree
# ---------------------------------------------------------------------------

class TestInterpreterPin(TmpCase):

    def setUp(self):
        super().setUp()
        self.lay = VenvLayout(self.root)
        self.tree = str(self.root / "py" / "cpython")

    def bind(self, argv, *, interpreter=None, trees=None, mounts=None):
        from gatedtrain.ext.launch import bind_argv
        return bind_argv(argv, self.lay.mounts if mounts is None else mounts,
                         interpreter=str(self.lay.python) if interpreter is None else interpreter,
                         trees=(self.tree,) if trees is None else trees)

    def test_pinned_accepted(self):
        b, why = self.bind([str(self.lay.python), str(self.lay.script)])
        self.assertIsNone(why)
        self.assertEqual(b.interpreter_host, str(self.lay.real_python))

    def test_other_bin_dir_to_the_same_python_refused(self):
        """A different venv can link to the same Python binary and load different site-packages."""
        other = self.lay.a1 / "other" / "bin"
        other.mkdir(parents=True)
        (other / "python").symlink_to(self.lay.real_python)
        (self.lay.a1 / "other" / "pyvenv.cfg").write_text("home = x\n", encoding="utf-8")
        b, why = self.bind([str(other / "python"), str(self.lay.script)])
        self.assertIsNone(b)
        self.assertIn("EXT-113", why)
        self.assertIn(str(self.lay.python), why)

    def test_sh_copy_named_python_refused(self):
        fake = self.lay.a1 / "sh-bin" / "python3.11"
        fake.parent.mkdir()
        shutil.copyfile("/bin/sh", fake)
        b, why = self.bind([str(fake), str(self.lay.script)])               # not the pinned path
        self.assertIsNone(b)
        self.assertIn("EXT-113", why)
        self.lay.python.unlink()                                            # the pinned path leads to the copy
        self.lay.python.symlink_to(fake)
        b, why = self.bind([str(self.lay.python), str(self.lay.script)])
        self.assertIsNone(b)
        self.assertIn("EXT-113", why)
        self.assertIn("interpreter tree", why)

    def test_tree_required(self):
        b, why = self.bind([str(self.lay.python), str(self.lay.script)], trees=(str(self.lay.a1),))
        self.assertIsNone(b)
        self.assertIn("interpreter tree", why)
        b, why = self.bind([str(self.lay.python), str(self.lay.script)], trees=())
        self.assertIsNone(b)

    def test_pin_is_the_string_as_written(self):
        """No normal form: `//`, `/./` or a trailing component change the venv CPython finds."""
        py = str(self.lay.python)
        for argv0 in ("/" + py, py.replace("/.venv/", "/./.venv/"), py.replace("/bin/python", "/bin/../bin/python")):
            with self.subTest(argv0=argv0):
                b, why = self.bind([argv0, str(self.lay.script)])
                self.assertIsNone(b)
                self.assertIn("EXT-113", why)

    def test_registry_rows(self):
        """The row's interpreter lies under one of its input roots, so ext-launch can bind it to a mount."""
        from gatedtrain.ext import registry
        from gatedtrain.ext.sdk import within
        self.assertTrue(any(within(registry.interpreter("synthetic"), r) for r in registry.input_roots("synthetic")))

    def test_host_interpreters_lie_in_their_trees(self):
        from gatedtrain.ext import registry
        for adapter in ("synthetic",):
            path = registry.interpreter(adapter)
            if not os.path.exists(path):
                continue
            with self.subTest(adapter=adapter):
                real = os.path.realpath(path)
                self.assertTrue(any(real.startswith(t.rstrip("/") + "/") for t in registry.interpreter_trees(adapter)))


class TestInterpreterPinLaunch(Launch3):

    def test_other_interpreter_refused(self):
        """A stub `python` in another bin directory of the runtime mount (a regular file named python) was bound."""
        req = golden_request()
        req["argv"] = ["/data/gatedtrain-mm/synthetic-runtime/other/bin/python", *req["argv"][1:]]
        errs = self.refused_at(req, "EXT-LAUNCH", "--request.argv")
        self.assertTrue(any(GOLDEN_INTERPRETER in e["message"] for e in errs), errs)



# ---------------------------------------------------------------------------
# EXT-114 (Low-2): path-valued environment values lie under /tmp or /out
# ---------------------------------------------------------------------------

class TestEnvPaths(TmpCase):

    def test_registry_rows(self):
        from gatedtrain.ext import registry
        self.assertEqual(registry.env_paths("synthetic"), PATH_ENV)
        for adapter in ("synthetic",):
            self.assertLessEqual(set(registry.env_paths(adapter)), set(registry.env_names(adapter)))

    def test_values(self):
        marker = "marker_value"
        for value, ok in (("/tmp", True), ("/tmp/cache", True), ("/out", True), ("/out/c", True),
                          ("/tmp/cache/", True), (f"/inputs/base-model/{marker}", False),
                          (f"/data/gatedtrain-mm/python/{marker}", False), (f"relative/{marker}", False), ("", False),
                          (f"/tmp/../etc/{marker}", False), (f"/tmpx/{marker}", False), (f"/outx/{marker}", False),
                          (f"~/{marker}", False), (f"//etc/{marker}", False)):
            for name in PATH_ENV:
                with self.subTest(name=name, value=value):
                    req = unit_request(self.root)
                    req["env"][name] = value
                    got = problems(req)
                    (self.assertNotIn if ok else self.assertIn)(f"env.{name}", [f for f, _ in got])
                    self.assertNotIn(marker, " ".join(m for _, m in got))

    def test_other_names_unaffected(self):
        req = unit_request(self.root)
        req["env"]["TORCH_CUDA_ARCH_LIST"] = "8.9"
        req["env"]["MASTER_ADDR"] = "127.0.0.1"
        self.assertEqual(fields(req), [])

class TestEnvPathsLaunch(Launch3):

    def test_refused(self):
        req = golden_request()
        req["env"]["HF_HOME"] = "/inputs/base-model"
        self.refused_at(req, "EXT-LAUNCH", "--request.env.HF_HOME")



# ---------------------------------------------------------------------------
# EXT-115 (Low-3): each tmpfs path once
# ---------------------------------------------------------------------------

class TestDuplicateTmpfs(TmpCase):

    def test_refused(self):
        for tmpfs in (["/tmp", "/tmp"], ["/tmp", "/tmp/"], ["/tmp/", "//tmp"], ["/tmp", "/tmp/."]):
            with self.subTest(tmpfs=tmpfs):
                req = unit_request(self.root)
                req["tmpfs"] = tmpfs
                msgs = [m for f, m in problems(req) if f == "tmpfs"]
                self.assertTrue(any("once" in m for m in msgs), msgs)
                self.assertIn("tmpfs", fields(as_record(req), kind="record"))


class TestDuplicateTmpfsLaunch(Launch3):

    def test_refused(self):
        req = golden_request()
        req["tmpfs"] = ["/tmp", "/tmp/"]
        self.refused_at(req, "EXT-LAUNCH", "--request.tmpfs")


# ---------------------------------------------------------------------------
# EXT-116: symlink limits, script file types and stdlib scan exemptions
# ---------------------------------------------------------------------------

class TestBindingAndScanBoundaries(TmpCase):

    def setUp(self):
        super().setUp()
        self.lay = VenvLayout(self.root)

    def bind(self, argv):
        from gatedtrain.ext.launch import bind_argv
        return bind_argv(argv, self.lay.mounts)

    def chain(self, n: int, tag: str) -> Path:
        """`<a1>/<tag>-0.py -> <tag>-1.py -> ... -> <tag>-<n-1>.py -> train.py`: n links (relative)."""
        target = "train.py"
        for i in reversed(range(n)):
            link = self.lay.a1 / f"{tag}-{i}.py"
            link.symlink_to(target)
            target = link.name
        return self.lay.a1 / f"{tag}-0.py"

    def test_n2_link_hops(self):
        """N2: exactly LINK_HOPS (40) links resolve; 41 refuse (the kernel's limit is 40)."""
        from gatedtrain.ext.launch import LINK_HOPS
        self.assertEqual(LINK_HOPS, 40)
        b, why = self.bind([str(self.lay.python), str(self.chain(40, "ok"))])
        self.assertIsNone(why)
        self.assertEqual(b.script_host, str(self.lay.script))
        b, why = self.bind([str(self.lay.python), str(self.chain(41, "long"))])
        self.assertIsNone(b)
        self.assertIn("argv[1]", why)

    def test_n4_py_link_to_sh(self):
        """N4: argv[1] ends in `.py` but resolves to a shell script: refused as not a `.py` script."""
        (self.lay.a1 / "evil.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        (self.lay.a1 / "s.py").symlink_to("evil.sh")
        b, why = self.bind([str(self.lay.python), str(self.lay.a1 / "s.py")])
        self.assertIsNone(b)
        self.assertIn("is not a `.py` script", why)
        self.assertIn("evil.sh", why)

    def test_n9_pyc_outside_pycache(self):
        """N9: only `<tree>/lib/python3.<N>/__pycache__/secrets.cpython-*.pyc` is the stdlib's byte code."""
        from gatedtrain.ext.launch import credential_name_problem
        tree = self.root / "cpython"
        cache = tree / "lib" / "python3.12" / "__pycache__"
        cache.mkdir(parents=True)
        (cache / "secrets.cpython-312.pyc").write_bytes(b"")
        self.assertIsNone(credential_name_problem(str(tree), interpreter_trees=(str(tree),)))
        other = tree / "lib" / "python3.12" / "x"
        other.mkdir()
        (other / "secrets.cpython-312.pyc").write_bytes(b"")
        why = credential_name_problem(str(tree), interpreter_trees=(str(tree),))
        self.assertIsNotNone(why)
        self.assertIn("x/secrets.cpython-312.pyc", why)


# ---------------------------------------------------------------------------
# EXT-117: interpreter and mount overrides affect host_config_sha256
# ---------------------------------------------------------------------------

class TestOverrideHash(unittest.TestCase):
    """A record written under `GATEDTRAIN_EXT_MIRROR_ROOT`, `GATEDTRAIN_EXT_INTERPRETER` or `GATEDTRAIN_EXT_INTERPRETER_TREES`
    refuses at ext-collect and `--recheck` under another value (EXT-85), also where the other rules would pass."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-r6-hash-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.sc = ExtScenario(cls.tmp / "sc")

    def fields(self, env: dict) -> list[str]:
        from gatedtrain.ext.launch import host_config, record_problems
        return [p.field for p in record_problems(self.sc.launch_record(), adapter="synthetic",
                                                 stage_out=self.sc.stage_out, framework="synthetic-trainer",
                                                 config=host_config(env))]

    def test_unit(self):
        from gatedtrain.ext.launch import HostConfig, host_config_sha256
        base = HostConfig(host_work="/w", output_root="/o", input_roots=("/a",))
        for change in ({"mirror_root": "/m"}, {"interpreter": "/a/bin/python"}, {"interpreter_trees": ("/a",)}):
            with self.subTest(change=change):
                self.assertNotEqual(host_config_sha256(base), host_config_sha256(dataclasses.replace(base, **change)))

    def test_same_layout_passes(self):
        self.assertEqual(self.fields(self.sc.env()), [])

    def test_other_trees_refused(self):
        """Another tree that still holds the interpreter: only the hash tells the layouts apart."""
        env = dict(self.sc.env(), GATEDTRAIN_EXT_INTERPRETER_TREES=str(self.sc.code))
        self.assertEqual(self.fields(env), ["--launch-record.host_config_sha256"])

    def test_without_the_overrides_refused(self):
        for name in ("GATEDTRAIN_EXT_INTERPRETER", "GATEDTRAIN_EXT_INTERPRETER_TREES"):
            with self.subTest(dropped=name):
                env = {k: v for k, v in self.sc.env().items() if k != name}
                self.assertIn("--launch-record.host_config_sha256", self.fields(env))
        env = dict(self.sc.env(), GATEDTRAIN_EXT_MIRROR_ROOT="/")
        self.assertIn("--launch-record.host_config_sha256", self.fields(env))


class TestOverrideHashLaunch(Launch3):

    def test_recheck_without_mirror_refused(self):
        """ext-launch under the golden layout (mirror root set); `--recheck` with the mirror root dropped."""
        from _support import VENV_PYTHON, cli_env
        from test_ext_launch import host_layout, launch, rooted
        import subprocess
        sub = self.sub()
        proc, rec = launch(sub, golden_request(), root=self.tmp / "host")
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        env = cli_env()
        env.update({k: v for k, v in host_layout(rooted(golden_request(), self.tmp / "host"), self.tmp / "host").items()
                    if k != "GATEDTRAIN_EXT_MIRROR_ROOT"})
        for e in (dict(env, GATEDTRAIN_EXT_MIRROR_ROOT=str(self.tmp / "host")), env):
            chk = subprocess.run([str(VENV_PYTHON), "-m", "gatedtrain.testing.synthetic", "ext-launch", "--adapter", "synthetic", "--recheck",
                                  str(rec), "--spec", str(rec.parent / "spec" / "spec.yaml")], cwd=str(self.tmp),
                                 env=e, capture_output=True, text=True, timeout=60)
            got = [(x["rule"], x["field"]) for x in parse_errors(chk.stderr)]
            if "GATEDTRAIN_EXT_MIRROR_ROOT" in e:
                self.assertEqual(chk.returncode, EXIT_OK, describe(chk))
            else:
                self.assertEqual(chk.returncode, EXIT_REFUSED, describe(chk))
                self.assertIn(("EXT-LAUNCH", "--recheck.host_config_sha256"), got)


if __name__ == "__main__":
    unittest.main()
