"""AN-34: the setup layout file (tests/README.md section 15).

One TOML file replaces the path flags of `setup host` and `setup keeper`. It is printed input: precedence is flag, then
layout, then the built-in default; the printed commands record the file's path and hash and where each non-default
value came from. The registry `layout.LAYOUT_KEYS` is compared with both commands' argparse flags in both directions,
so a new flag cannot be forgotten. Nothing is written or run.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import os
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from test_an_keeper_cli import no_processes, run, run_text

from gatedtrain import cli
from gatedtrain.ainative import hostsetup, layout
from gatedtrain.ainative.keeper import setup, setup_user
from gatedtrain.errors import Refused

HEX = "0123456789abcdef" * 4
REF = f"ghcr.io/example/cuda-base:12.8@sha256:{HEX}"
VERSION = f"schema_version = {layout.SCHEMA_VERSION}\n"


def subcommand_flags() -> dict[str, set[str]]:
    """The long options of `setup host` and `setup keeper`, read from the real parser."""
    def subparsers(parser):
        return next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction)).choices

    setup_cmds = subparsers(subparsers(cli.build_parser())["setup"])
    return {name: {o for a in setup_cmds[name]._actions for o in a.option_strings
                   if o.startswith("--") and o not in ("--help",)} for name in ("host", "keeper")}


def registry_flags(command: str, *, supported: bool) -> set[str]:
    return {k.flags[s] for k in layout.LAYOUT_KEYS for s in k.flags
            if layout._scope_command(s) == command and k.flags[s] and k.supported == supported}


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.work = self.tmp / "cwd"
        self.work.mkdir()
        cwd = os.getcwd()
        os.chdir(self.work)
        self.addCleanup(os.chdir, cwd)

    def write(self, body: str, name: str = "layout.toml") -> str:
        path = self.tmp / name
        path.write_text(body)
        return str(path)

    def refused(self, body: str, scope: str = layout.SEPARATE) -> list[str]:
        with self.assertRaises(Refused) as ctx:
            layout.parse((VERSION + body).encode(), "L", scope)
        return [p.line() for p in ctx.exception.problems]


class RegistryTest(unittest.TestCase):                                  # AN-34
    def test_every_flag_has_a_layout_key_or_a_stated_reason_and_vice_versa(self):
        flags = subcommand_flags()
        for command, present in flags.items():
            exempt = {f for (c, f) in layout.NOT_LAYOUT_FLAGS if c == command}
            keyed = registry_flags(command, supported=True)
            self.assertEqual(present - exempt, keyed - {None}, f"setup {command}: flags and registry differ")
            self.assertFalse(keyed & exempt, f"setup {command}: a flag is both keyed and exempt")
            self.assertFalse(exempt - present, f"setup {command}: an exempt flag that does not exist")
            # a reserved key's flag does not exist yet; when it does, the row must be flipped to supported
            self.assertFalse(registry_flags(command, supported=False) & present,
                             f"setup {command} has the option of a reserved layout key: flip its row")

    def test_the_rows_are_complete_and_unique(self):
        seen = set()
        for key in layout.LAYOUT_KEYS:
            self.assertNotIn((key.table, key.name), seen)
            seen.add((key.table, key.name))
            self.assertIn(key.table, layout.TABLES)
            self.assertTrue(key.doc and key.kind in ("str", "list", "number", "int"), key)
            self.assertTrue(set(key.flags) <= set(layout.SCOPES), key)
            self.assertEqual(key.supported, key.check is not None, key)
            if key.supported and not key.default and not key.computed:
                self.assertTrue(key.placeholder, f"{key.dotted} has no default and no placeholder")
        self.assertEqual({k.table for k in layout.LAYOUT_KEYS}, set(layout.TABLES))

    def test_defaults_come_from_the_existing_constants(self):
        eff = layout.effective_layout(layout.HOST)
        self.assertEqual(eff["host.handoff"], (hostsetup.DEFAULT_HANDOFF, "default"))
        self.assertEqual(eff["keeper.user"], (setup.DEFAULT_USER, "default"))
        self.assertEqual(eff["host.eval_python_root"], (hostsetup.DEFAULT_EVAL_PYTHON_ROOT, "default"))
        self.assertEqual(layout.effective_layout(layout.SEPARATE)["keeper.venv"], (setup.DEFAULT_VENV, "default"))

    def test_constants_have_a_reason(self):
        self.assertEqual(layout.SCHEMA_VERSION, 1)
        self.assertEqual(layout.MAX_LAYOUT_BYTES, 64 * 1024)
        before = inspect.getsource(layout).split("MAX_LAYOUT_BYTES = ")[0]
        self.assertIn("64 KiB", before[before.rindex("# A layout"):])              # the reason sits above the constant

    def test_no_machine_facts_in_the_code(self):
        from gatedtrain import cli_layout

        for module in (layout, cli_layout):
            tree = ast.parse(inspect.getsource(module))
            docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                          if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
                          and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                    self.assertFalse(node.value.startswith(("/", "~")) or "/home" in node.value
                                     or "/etc" in node.value, f"{module.__name__}: {node.value!r}")


class SchemaTest(Base):                                                 # AN-34
    def test_schema_version_is_required_and_exactly_one(self):
        for body in ("", "schema_version = 2\n", "schema_version = 0\n", "schema_version = true\n",
                     'schema_version = "1"\n', "schema_version = 1.0\n"):
            with self.assertRaises(Refused) as ctx:
                layout.parse(body.encode(), "L", layout.HOST)
            self.assertIn("schema_version", ctx.exception.problems[0].message, body)
        self.assertEqual(layout.parse(VERSION.encode(), "L", layout.HOST), {})

    def test_unknown_table_and_key_are_named(self):
        lines = self.refused("[nope]\nx = 1\n[keeper]\nsate = '/x'\n")
        self.assertTrue(any("nope" in x and "unknown table" in x for x in lines), lines)
        self.assertTrue(any("sate" in x and "unknown key" in x for x in lines), lines)

    def test_a_top_level_key_is_refused(self):
        self.assertTrue(any("stray" in x for x in self.refused("stray = 1\n")))

    def test_wrong_types_are_refused(self):
        for body in ("[keeper]\nstate = 5\n", "[keeper]\nuser = true\n", "[inference]\neval_trees = '/x'\n",
                     "[inference]\neval_trees = [1]\n"):
            lines = self.refused(body)
            self.assertTrue(any("must be" in x for x in lines), (body, lines))

    def test_invalid_toml_and_duplicate_keys_are_refused(self):
        for body in ("[keeper\n", "[keeper]\nstate = '/a'\nstate = '/b'\n"):
            with self.assertRaises(Refused) as ctx:
                layout.parse((VERSION + body).encode(), "L", layout.SEPARATE)
            self.assertIn("not valid TOML", ctx.exception.problems[0].message)
        with self.assertRaises(Refused):
            layout.parse(b"\xff\xfe", "L", layout.HOST)

    def test_values_go_through_the_validators_the_flags_use(self):
        cases = {
            "[keeper]\nstate = 'relative/dir'\n": "absolute",
            "[keeper]\nstate = '/a/../b'\n": "absolute",
            "[keeper]\nstate = '/a b'\n": "systemd",
            "[keeper]\nuser = 'Not A User'\n": "user name",
            "[keeper]\nsocket = '/" + "s" * 120 + "'\n": "longer than",
            "[inference]\neval_trees = ['/a:b']\n": "docker mount",
        }
        for body, word in cases.items():
            lines = self.refused(body)
            self.assertTrue(any(word in x and "AN-34" in x for x in lines), (body, lines))
        self.assertTrue(any("pinned by digest" in x for x in
                            self.refused("[inference]\nimage = 'ghcr.io/example/cuda-base:12.8'\n", layout.HOST)))
        self.assertTrue(any("absolute" in x for x in
                            self.refused("[inference]\neval_requirements = 'req.txt'\n", layout.HOST)))
        for body in ("[host]\nhandoff = '/a b'\n", "[host]\nhandoff = '/'\n"):          # the host's stricter path rule
            self.assertTrue(self.refused(body, layout.HOST), body)

    def test_every_problem_is_reported_at_once(self):
        lines = self.refused("[keeper]\nstate = 'x'\nuser = 'X'\n")
        self.assertEqual(len(lines), 2, lines)

    def test_a_full_valid_layout_parses(self):
        values = layout.parse((VERSION + f"[keeper]\nstate = '/a/b'\n[inference]\neval_trees = ['/t1', '/t2']\n"
                               ).encode(), "L", layout.SEPARATE)
        self.assertEqual(values, {("keeper", "state"): "/a/b", ("inference", "eval_trees"): ["/t1", "/t2"]})


class ReadTest(Base):                                                   # AN-34
    def test_a_missing_symlinked_or_non_regular_file_is_refused(self):
        real = self.write(VERSION)
        link = self.tmp / "link.toml"
        link.symlink_to(real)
        fifo = self.tmp / "fifo.toml"
        os.mkfifo(fifo)
        for path in (str(self.tmp / "absent.toml"), str(link), str(self.tmp), str(fifo)):
            with self.assertRaises(Refused) as ctx:
                layout.load(path, layout.HOST)
            self.assertEqual(ctx.exception.problems[0].rule, "AN-34", path)
        self.assertEqual(layout.load(real, layout.HOST).values, {})

    def test_the_size_cap(self):
        pad = "#" * (layout.MAX_LAYOUT_BYTES - len(VERSION) - 1) + "\n"
        self.assertEqual(layout.load(self.write(VERSION + pad), layout.HOST).values, {})        # exactly the cap
        with self.assertRaises(Refused) as ctx:
            layout.load(self.write(VERSION + pad + "#"), layout.HOST)
        self.assertIn("larger than", ctx.exception.problems[0].message)

    def test_deeply_nested_toml_under_the_cap_is_a_refusal_not_a_crash(self):
        """tomllib recurses per nesting level: a document well under the size cap can still exceed the interpreter's
        depth. It must be an AN-34 refusal for the three shapes that nest, through `parse` and through `load`."""
        bodies = {"array": "a = " + "[" * 30000 + "]" * 30000 + "\n",
                  "inline tables": "a = " + "{a=" * 5000 + "1" + "}" * 5000 + "\n",
                  "dotted table headers": "[" + ".".join(["a"] * 20000) + "]\n"}
        for name, body in bodies.items():
            with self.subTest(shape=name):
                data = (VERSION + body).encode()
                self.assertLessEqual(len(data), layout.MAX_LAYOUT_BYTES, name)
                for scope in (layout.HOST, layout.SEPARATE, layout.SAME):
                    with self.assertRaises(Refused) as ctx:
                        layout.parse(data, "L", scope)
                    self.assertEqual(ctx.exception.problems[0].rule, "AN-34")
                    self.assertIn("nested too deeply", ctx.exception.problems[0].message)
                with self.assertRaises(Refused):
                    layout.load(self.write(VERSION + body), layout.HOST)

    def test_the_hash_is_of_the_bytes_read(self):
        path = self.write(VERSION + "# note\n")
        self.assertEqual(layout.load(path, layout.HOST).sha256, hashlib.sha256(Path(path).read_bytes()).hexdigest())


class ModeTest(Base):                                                   # AN-34
    def test_tables_of_the_other_mode_or_command_are_refused_naming_the_scope(self):
        cases = [(layout.SAME, "[keeper]\nstate = '/a'\n", "[keeper]", "--mode same-user"),
                 (layout.SEPARATE, "[same_user]\nstate = '/a'\n", "[same_user]", "--mode separate-user"),
                 (layout.SEPARATE, "[host]\nhandoff = '/a'\n", "[host]", "separate-user"),
                 (layout.HOST, "[same_user]\nstate = '/a'\n", "[same_user]", "setup host"),
                 (layout.HOST, "[gpu]\n", "[gpu]", "setup host")]
        for scope, body, table, word in cases:
            lines = self.refused(body, scope)
            self.assertTrue(any(table in x and word in x and "does not apply" in x for x in lines), (body, lines))

    def test_a_key_of_a_shared_table_that_the_scope_lacks_is_refused(self):
        self.assertTrue(any("eval_python" in x and "does not apply" in x for x in
                            self.refused("[inference]\neval_python = '/a'\n", layout.SAME)))
        self.assertTrue(any("eval_trees" in x and "does not apply" in x for x in
                            self.refused("[inference]\neval_trees = ['/a']\n", layout.HOST)))

    def test_the_gpu_keys_are_reserved_and_refused(self):
        for scope in (layout.SEPARATE, layout.SAME):
            lines = self.refused("[gpu]\nwait_s = 60\nbusy_mib = 1024\n[inference]\nkeeper_eval_python = '/p'\n",
                                 scope)
            self.assertEqual(len(lines), 3, lines)
            for line in lines:
                self.assertIn("not supported by this version (the keeper has no such option yet)", line)


class PrintedDefaultsTest(Base):                                        # AN-34
    def test_the_printed_layout_round_trips_and_documents_every_key(self):
        env = {"HOME": str(self.tmp / "home")}
        for scope in layout.SCOPES:
            text = layout.render(scope, env=env, uid=4321)
            doc = tomllib.loads(text)
            eff = layout.effective_layout(scope, env=env, uid=4321)
            self.assertEqual(doc.pop("schema_version"), layout.SCHEMA_VERSION)
            printed = {f"{t}.{k}": v for t, body in doc.items() for k, v in body.items()}
            self.assertEqual(printed, {d: v for d, (v, _s) in eff.items() if v is not None}, scope)
            values = layout.parse(text.encode(), "L", scope)                 # the printed file is itself valid
            self.assertEqual({f"{t}.{k}" for t, k in values}, set(printed))
            for key in layout.LAYOUT_KEYS:
                if scope in key.flags:
                    self.assertIn(f"# {key.doc}", text)
                    self.assertRegex(text, rf"(?m)^#? ?{key.name} = ")
            self.assertEqual(text.count("\n[") + text.startswith("["), len({k.table for k in layout.LAYOUT_KEYS
                                                                           if scope in k.flags}))

    def test_the_same_user_defaults_follow_the_xdg_environment(self):
        text = layout.render(layout.SAME, env={"HOME": "/h", "XDG_DATA_HOME": str(self.tmp / "data"),
                                               "XDG_RUNTIME_DIR": str(self.tmp / "run")}, uid=1)
        doc = tomllib.loads(text)["same_user"]
        self.assertEqual(doc["state"], f"{self.tmp}/data/{setup_user.APP_DIR}")
        self.assertEqual(doc["socket"], f"{self.tmp}/run/{setup_user.APP_DIR}/{setup_user.SOCKET_NAME}")

    def test_the_command_prints_it_and_writes_nothing(self):
        with no_processes():
            code, text, _ = run_text(["setup", "layout", "--mode", "separate-user"])
            host_code, host_doc = run(["setup", "layout", "--for", "host"])
            default_code, default_text, _ = run_text(["setup", "layout"])
        self.assertEqual((code, host_code, default_code), (0, 0, 0))
        self.assertIn("[keeper]", text)
        self.assertIn("[same_user]", default_text)                           # follows `setup keeper`'s default mode
        self.assertEqual(set(host_doc["data"]), {"scope", "toml"})
        self.assertIn("[host]", host_doc["data"]["toml"])
        self.assertEqual(list(self.work.iterdir()), [])
        code, doc = run(["setup", "layout", "--for", "host", "--mode", "same-user"])
        self.assertNotEqual(code, 0)
        self.assertEqual(doc["problems"][0]["rule"], "AN-34")


class CommandTest(Base):                                                # AN-34
    def host(self, *extra: str) -> tuple[int, dict]:
        with no_processes():
            code, doc = run(["setup", "host", "--owner", "alice", "--owner-uid", "1234", *extra])
        self.assertEqual(list(self.work.iterdir()), [], "setup host created a file")
        return code, doc

    def test_a_layout_sets_the_host_paths_and_the_script_records_it(self):
        path = self.write(VERSION + f"[host]\nhandoff = '{self.tmp}/out'\neval_venv = '{self.tmp}/ev'\n"
                          f"[keeper]\nstate = '{self.tmp}/st'\n[inference]\nimage = '{REF}'\n")
        code, doc = self.host("--layout", path)
        self.assertEqual(code, 0, doc)
        script = doc["data"]["script"]
        lines = script.splitlines()
        sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.assertEqual(lines[0], "#!/bin/bash")
        self.assertEqual(lines[1], f"# layout: {path} sha256:{sha}")
        self.assertEqual(lines[2], "# source of each value")
        self.assertIn(f"#   host.handoff = {self.tmp}/out (layout)", lines[3:7])
        self.assertIn(f"#   inference.image = {REF} (layout)", lines)
        self.assertNotIn("host.keeper_home", script.split("set -euo")[0])          # a default is not listed
        self.assertIn(f"{self.tmp}/out", script)
        self.assertEqual(doc["data"]["layout"]["path"], path)
        self.assertEqual(doc["data"]["layout"]["sha256"], sha)
        keys = doc["data"]["layout"]["keys"]
        self.assertEqual((keys["host.handoff"], keys["host.keeper_home"], keys["keeper.state"]),
                         ("layout", "default", "layout"))
        self.assertEqual(set(doc["data"]), {"script", "command", "layout"})

    def test_a_flag_beats_the_layout_and_the_header_says_so(self):
        path = self.write(VERSION + f"[host]\nhandoff = '{self.tmp}/from-layout'\nkeeper_home = '{self.tmp}/kh'\n")
        code, doc = self.host("--layout", path, "--handoff", f"{self.tmp}/from-flag")
        self.assertEqual(code, 0, doc)
        script = doc["data"]["script"]
        self.assertIn(f"#   host.handoff = {self.tmp}/from-flag (flag)", script)
        self.assertIn(f"#   host.keeper_home = {self.tmp}/kh (layout)", script)
        self.assertNotIn("from-layout", script)
        self.assertEqual(doc["data"]["layout"]["keys"]["host.handoff"], "flag")

    def test_the_default_is_used_without_a_layout_and_the_output_is_unchanged(self):
        code, plain = self.host()
        self.assertEqual(code, 0)
        self.assertEqual(set(plain["data"]), {"script", "command"})
        path = self.write(VERSION)
        _, with_empty = self.host("--layout", path)
        body = with_empty["data"]["script"].splitlines()
        self.assertEqual(body[2], "# source of each value: every key is its default")
        self.assertEqual(body[:1] + body[3:], plain["data"]["script"].splitlines())

    def test_eval_python_root_is_a_layout_key_the_script_follows(self):
        root = f"{self.tmp}/py"
        req = self.tmp / "req.txt"
        req.write_bytes(b"numpy==2.2.6 --hash=sha256:" + b"3" * 64 + b"\n")
        path = self.write(VERSION + f"[host]\neval_python_root = '{root}'\n[inference]\n"
                          f"eval_python = '{root}/cpy/bin/python3.12'\neval_requirements = '{req}'\n")
        code, doc = self.host("--layout", path)
        self.assertEqual(code, 0, doc)
        self.assertIn(f"PYROOT={root}", doc["data"]["script"])
        self.assertNotIn(hostsetup.DEFAULT_EVAL_PYTHON_ROOT, doc["data"]["script"])
        bad = self.write(VERSION + f"[host]\neval_python_root = '{self.tmp}/ev/inner'\neval_venv = '{self.tmp}/ev'\n",
                         "bad.toml")
        code, doc = self.host("--layout", bad)
        self.assertEqual((code, doc["problems"][0]["field"]), (doc["exit"], "eval_python_root"))
        self.assertNotEqual(code, 0)

    def test_separate_user_keeper_follows_the_layout(self):
        path = self.write(VERSION + f"[keeper]\nuser = 'kuser'\nstate = '{self.tmp}/st'\nvenv = '{self.tmp}/v'\n"
                          "socket = '/srv/gk/run/k.sock'\n")        # printed only, and short: the socket path has a limit
        with no_processes():
            code, doc = run(["setup", "keeper", "--mode", "separate-user", "--packages", "p==1", "--layout", path,
                             "--venv", f"{self.tmp}/flagv"])
        self.assertEqual(code, 0, doc)
        data = doc["data"]
        self.assertIn(f"ExecStart={self.tmp}/flagv/bin/gatedtrain keeper serve --state {self.tmp}/st", data["unit"])
        self.assertIn("User=kuser\n", data["unit"])
        self.assertEqual(data["commands"][0].split(" sha256:")[0], f"# layout: {path}")
        self.assertEqual(data["layout"]["keys"]["keeper.venv"], "flag")
        self.assertEqual(data["layout"]["keys"]["keeper.user"], "layout")
        self.assertNotIn("layout:", data["unit"])                                  # the unit text is unchanged

    def test_same_user_keeper_follows_the_layout_with_changed_home_and_xdg(self):
        path = self.write(VERSION + f"[same_user]\nstate = '{self.tmp}/st'\n[inference]\neval_trees = ['/e']\n")
        env = {"HOME": str(self.tmp / "other-home"), "XDG_DATA_HOME": str(self.tmp / "xdg"),
               "XDG_RUNTIME_DIR": "/rt/xrun", "XDG_CONFIG_HOME": str(self.tmp / "xcfg")}
        with mock.patch.dict(os.environ, env, clear=False), no_processes():
            code, doc = run(["setup", "keeper", "--layout", path])
            self.assertNotEqual(code, 0)                                           # eval_trees need --inference-image
            self.assertIn("--eval-tree needs --inference-image", doc["problems"][0]["message"])
            path = self.write(VERSION + f"[same_user]\nstate = '{self.tmp}/st'\n", "l2.toml")
            code, doc = run(["setup", "keeper", "--layout", path])
        self.assertEqual(code, 0, doc)
        self.assertIn(f"--state {self.tmp}/st --socket /rt/xrun/", doc["data"]["unit"])
        self.assertEqual(doc["data"]["layout"]["keys"], {"same_user.state": "layout", "same_user.socket": "default",
                                                         "same_user.executable": "default",
                                                         "inference.eval_trees": "default"})
        self.assertTrue(doc["data"]["commands"][0].startswith("# layout: "))

    def test_a_mode_inapplicable_table_is_refused_through_the_command(self):
        path = self.write(VERSION + f"[keeper]\nstate = '{self.tmp}/st'\n")
        with no_processes():
            code, doc = run(["setup", "keeper", "--layout", path])
        self.assertNotEqual(code, 0)
        self.assertEqual(doc["problems"][0]["rule"], "AN-34")
        self.assertIn("same-user", doc["problems"][0]["message"])

    def test_a_bad_layout_is_refused_with_the_rule_and_no_output(self):
        path = self.write(VERSION + "[host]\nhandoff = 'relative'\n")
        code, doc = self.host("--layout", path)
        self.assertNotEqual(code, 0)
        self.assertEqual(doc["data"], {})
        self.assertEqual({p["rule"] for p in doc["problems"]}, {"AN-34"})
        code, doc = self.host("--layout", str(self.tmp / "absent.toml"))
        self.assertEqual(doc["problems"][0]["rule"], "AN-34")

    def test_no_environment_variable_or_default_location_names_a_layout(self):
        path = self.write(VERSION + f"[host]\nhandoff = '{self.tmp}/envout'\n")
        env = {"HOME": str(self.tmp), "XDG_CONFIG_HOME": str(self.tmp), "GATEDTRAIN_LAYOUT": path}
        (self.tmp / "gatedtrain").mkdir()
        (self.tmp / "gatedtrain" / "layout.toml").write_text(Path(path).read_text())
        with mock.patch.dict(os.environ, env):
            code, doc = self.host()
        self.assertEqual(code, 0)
        self.assertNotIn("layout", doc["data"])
        self.assertNotIn("envout", doc["data"]["script"])
        for name in dir(layout):
            self.assertNotIn("environ", name)
        self.assertNotIn("getenv", inspect.getsource(layout.resolve) + inspect.getsource(layout.load))


if __name__ == "__main__":
    unittest.main()
