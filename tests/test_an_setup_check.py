"""AN-32, AN-33: `gatedtrain setup check` names the outside prerequisites of the sudo-free setup.

The check reads the host only through a `HostProbe`; these tests use a fake one (a healthy host the tests then break
one prerequisite at a time), so they run on any machine and the fake records every call: no write, no sudo.
"""

from __future__ import annotations

import json
import stat
import unittest
from unittest import mock

from test_an_keeper_cli import no_processes, run, run_text

from gatedtrain import cli_setup_check
from gatedtrain.ainative import setup_check as sc

SECURITY_OPTIONS_ROOTLESS = '["name=seccomp,profile=builtin","name=rootless","name=cgroupns"]'
SECURITY_OPTIONS_ROOTFUL = '["name=seccomp,profile=builtin","name=cgroupns"]'
UID = 4321
USER = "alice"
RT = f"/run/user/{UID}"


class FakeProbe(sc.HostProbe):
    """A healthy rootless host; `files` maps path -> text, `stats` path -> (uid, mode)."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.platform = "linux"
        self.environ = {"XDG_RUNTIME_DIR": RT, "XDG_CONFIG_HOME": "/cfg"}
        self.files = {
            sc.SUBID_FILES["subuid"]: f"{USER}:100000:{sc.SUBID_MIN_COUNT}\n",
            sc.SUBID_FILES["subgid"]: f"{USER}:100000:{sc.SUBID_MIN_COUNT}\n",
            sc.USERNS_MAX_FILE: "63000\n",
            sc.USERNS_APPARMOR_FILE: "0\n",
            sc.PTRACE_SCOPE_FILE: "1\n",
            sc.LINGER_DIR + "/" + USER: "",
            "/cdi/nvidia.yaml": f"cdiVersion: 0.5.0\nkind: {sc.CDI_GPU_KIND}\n",
        }
        self.stats = {f"{RT}/docker.sock": (UID, stat.S_IFSOCK | 0o660)}
        self.dirs = {"/cdi": ["nvidia.yaml"], "/proc/driver/nvidia": []}
        self.binaries = {"docker": "/bin/docker", "newuidmap": "/bin/newuidmap", "newgidmap": "/bin/newgidmap",
                         "slirp4netns": "/bin/slirp4netns", "nvidia-smi": "/bin/nvidia-smi"}
        for b in ("newuidmap", "newgidmap"):
            self.stats[self.binaries[b]] = (0, stat.S_IFREG | stat.S_ISUID | 0o755)
        self.runs = {("docker", "info"): (0, SECURITY_OPTIONS_ROOTLESS),
                     ("getcap",): (0, ""),
                     ("systemctl", "--user", "is-system-running"): (0, "running")}
        self.memlock = sc.UNLIMITED

    def _log(self, *a):
        self.calls.append(a)

    def system(self): self._log("system"); return self.platform
    def uid(self): self._log("uid"); return UID
    def username(self): self._log("username"); return USER
    def env(self, name): self._log("env", name); return self.environ.get(name)
    def read_text(self, path): self._log("read_text", path); return self.files.get(path)
    def listdir(self, path): self._log("listdir", path); return self.dirs.get(path)
    def stat(self, path): self._log("stat", path); return self.stats.get(path)
    def which(self, name): self._log("which", name); return self.binaries.get(name)
    def memlock_hard(self): self._log("memlock"); return self.memlock

    def run(self, argv, timeout, env=None):
        self._log("run", tuple(argv))
        for n in range(len(argv), 0, -1):
            if (key := tuple(argv[:n])) in self.runs:
                return self.runs[key]
        return None

    def cdi_dirs(self):                    # the directories dockerd reads by default (a fake: not the real table)
        return ["/cdi"]


def check(probe, gpu=None):
    return {c["id"]: c for c in sc.run_checks(probe, gpu=gpu)}


class HealthyHostTest(unittest.TestCase):                           # AN-32
    def test_everything_ok_and_exit_zero(self):
        probe = FakeProbe()
        checks = sc.run_checks(probe, gpu=True)
        self.assertEqual([c["id"] for c in checks], [spec.id for spec in sc.CHECKS])
        bad = [c for c in checks if c["status"] != "ok"]
        self.assertEqual(bad, [])
        self.assertEqual(sc.exit_code(checks), 0)

    def test_each_check_has_the_documented_shape(self):
        for c in sc.run_checks(FakeProbe(), gpu=True):
            self.assertEqual(set(c), {"id", "status", "required", "owner", "summary", "fix"}, c)
            self.assertIn(c["status"], ("ok", "missing", "unknown", "not_applicable"))
            self.assertIn(c["owner"], ("you", "administrator"))
            self.assertIsInstance(c["required"], bool)
            self.assertTrue(c["summary"] and c["fix"])

    def test_the_probe_only_reads(self):
        probe = FakeProbe()
        sc.run_checks(probe, gpu=True)
        verbs = {c[0] for c in probe.calls}
        self.assertLessEqual(verbs, {"system", "uid", "username", "env", "read_text", "listdir", "stat", "which",
                                     "memlock", "run"})
        for call in probe.calls:
            if call[0] == "run":
                self.assertNotIn("sudo", call[1])
                self.assertIn(call[1][0], sc.RUNNABLE, call)       # only the named read-only commands


class ConstantsTest(unittest.TestCase):                             # AN-32: one table, each constant explained
    def test_the_table_values(self):
        self.assertEqual(sc.SUBID_MIN_COUNT, 65536)                 # rootless Docker maps one full 16-bit id range
        self.assertEqual(sc.SUBID_FILES, {"subuid": "/etc/subuid", "subgid": "/etc/subgid"})
        self.assertEqual(sc.CDI_SYSTEM_DIRS, ("/etc/cdi", "/var/run/cdi"))
        self.assertEqual(sc.CDI_GPU_KIND, "nvidia.com/gpu")
        self.assertEqual(sc.UID_HELPERS, ("newuidmap", "newgidmap"))
        self.assertEqual(sc.NET_HELPERS, ("slirp4netns", "pasta"))
        self.assertEqual(sc.LINGER_DIR, "/var/lib/systemd/linger")

    def test_every_constant_has_a_reason(self):
        for name in sc.CONSTANT_REASONS:
            self.assertTrue(hasattr(sc, name), name)
            self.assertGreater(len(sc.CONSTANT_REASONS[name]), 20, name)
        for name in ("SUBID_MIN_COUNT", "SUBID_FILES", "CDI_SYSTEM_DIRS", "CDI_GPU_KIND", "UID_HELPERS",
                     "NET_HELPERS", "LINGER_DIR", "USERNS_MAX_FILE", "USERNS_APPARMOR_FILE", "PTRACE_SCOPE_FILE"):
            self.assertIn(name, sc.CONSTANT_REASONS)

    def test_user_paths_derive_from_the_probe(self):
        probe = FakeProbe()
        self.assertEqual(sc.SystemProbe.cdi_user_dirs(probe_with_env({"XDG_CONFIG_HOME": "/c"})), ["/c/cdi"])
        self.assertEqual(sc.docker_socket_candidates(probe_with_env({"XDG_RUNTIME_DIR": "/r"}), 7),
                         ["/r/docker.sock"])
        self.assertEqual(sc.docker_socket_candidates(probe_with_env({}), 7), ["/run/user/7/docker.sock"])
        self.assertEqual(sc.docker_socket_candidates(probe_with_env({"DOCKER_HOST": "unix:///x/d.sock"}), 7),
                         ["/x/d.sock"])
        del probe


def probe_with_env(env: dict):
    p = FakeProbe()
    p.environ = env
    p.cdi_dirs = lambda: sc.SystemProbe.cdi_dirs(p)
    p.cdi_user_dirs = lambda: sc.SystemProbe.cdi_user_dirs(p)
    return p


class MissingPrerequisiteTest(unittest.TestCase):                   # AN-32
    def assertMissing(self, probe, cid, *, owner, required=True, status="missing", gpu=False):
        c = check(probe, gpu=gpu)[cid]
        self.assertEqual((c["status"], c["owner"], c["required"]), (status, owner, required), c)
        return c

    def test_not_linux(self):
        p = FakeProbe()
        p.platform = "darwin"
        self.assertMissing(p, "linux", owner="you")

    def test_no_docker_cli(self):
        p = FakeProbe()
        del p.binaries["docker"]
        c = self.assertMissing(p, "rootless_docker", owner="administrator")
        self.assertIn("docker", c["summary"])

    def test_docker_without_a_rootless_socket(self):
        p = FakeProbe()
        p.stats.clear()
        c = self.assertMissing(p, "rootless_docker", owner="you")
        self.assertIn("dockerd-rootless-setuptool.sh install", c["fix"])

    def test_socket_owned_by_someone_else(self):
        p = FakeProbe()
        p.stats[f"{RT}/docker.sock"] = (0, stat.S_IFSOCK | 0o660)
        self.assertMissing(p, "rootless_docker", owner="you")

    def test_a_socket_that_does_not_answer_is_unknown(self):
        p = FakeProbe()
        p.runs.clear()
        self.assertMissing(p, "rootless_docker", owner="you", status="unknown")

    def test_a_rootful_daemon_is_not_rootless(self):
        p = FakeProbe()
        p.runs[("docker", "info")] = (0, SECURITY_OPTIONS_ROOTFUL)
        c = self.assertMissing(p, "rootless_docker", owner="you")
        self.assertIn("not rootless", c["summary"])

    def test_the_rootless_option_is_matched_exactly_not_as_a_substring(self):
        for out in ('["name=notrootless","name=rootlessish","name=seccomp,profile=name=rootless"]', '["rootless"]',
                    '"name=rootless"', '{"name=rootless": 1}'):
            p = FakeProbe()
            p.runs[("docker", "info")] = (0, out)
            self.assertIn(check(p)["rootless_docker"]["status"], ("missing", "unknown"), out)
        p = FakeProbe()
        p.runs[("docker", "info")] = (0, "not json at all name=rootless")
        self.assertMissing(p, "rootless_docker", owner="you", status="unknown")

    def test_docker_info_is_asked_for_the_json_security_options(self):
        p = FakeProbe()
        check(p)
        self.assertIn(("run", ("docker", "info", "--format", "{{json .SecurityOptions}}")), p.calls)

    def test_a_docker_host_that_is_not_a_unix_socket_is_not_the_rootless_daemon(self):
        for host in ("tcp://10.0.0.5:2375", "ssh://builder@host", "npipe:////./pipe/docker_engine"):
            p = FakeProbe()
            p.environ["DOCKER_HOST"] = host
            c = self.assertMissing(p, "rootless_docker", owner="you")
            self.assertIn("DOCKER_HOST", c["summary"])
            self.assertIn(host.split(":")[0], c["summary"])
            self.assertNotIn("builder", c["summary"])                          # never echo credentials-looking parts

    def test_a_docker_context_is_reported_as_a_note(self):
        p = FakeProbe()
        p.environ["DOCKER_CONTEXT"] = "remote-box"
        c = check(p)["rootless_docker"]
        self.assertEqual(c["status"], "ok")
        self.assertIn("DOCKER_CONTEXT", c["summary"] + c["fix"])
        self.assertIn("remote-box", c["summary"] + c["fix"])
        self.assertNotIn("DOCKER_CONTEXT", check(FakeProbe())["rootless_docker"]["summary"])

    def test_docker_info_failure_is_unknown(self):
        p = FakeProbe()
        p.runs[("docker", "info")] = (1, "")
        self.assertMissing(p, "rootless_docker", owner="you", status="unknown")

    def test_uidmap_missing_or_not_setuid(self):
        p = FakeProbe()
        del p.binaries["newgidmap"]
        self.assertMissing(p, "uidmap", owner="administrator")
        p = FakeProbe()
        p.stats["/bin/newuidmap"] = (0, stat.S_IFREG | 0o755)
        p.binaries["getcap"] = "/bin/getcap"                                    # file capabilities can be checked: none
        c = self.assertMissing(p, "uidmap", owner="administrator")
        self.assertIn("setuid", c["summary"])
        self.assertIn("/bin/newuidmap", c["summary"])                            # says which file was inspected

    def test_uidmap_without_setuid_and_without_getcap_is_unknown_not_missing(self):
        p = FakeProbe()
        p.stats["/bin/newuidmap"] = (0, stat.S_IFREG | 0o755)
        c = self.assertMissing(p, "uidmap", owner="administrator", status="unknown")
        self.assertIn("getcap", c["summary"])

    def test_uidmap_accepts_file_capabilities(self):
        p = FakeProbe()
        p.binaries["getcap"] = "/bin/getcap"
        for name, cap in (("newuidmap", "cap_setuid"), ("newgidmap", "cap_setgid")):
            p.stats[p.binaries[name]] = (0, stat.S_IFREG | 0o755)
        p.runs[("getcap", "/bin/newuidmap")] = (0, "/bin/newuidmap cap_setuid=ep\n")
        p.runs[("getcap", "/bin/newgidmap")] = (0, "/bin/newgidmap cap_setgid=ep\n")
        c = check(p)["uidmap"]
        self.assertEqual(c["status"], "ok", c)
        self.assertIn("capabilit", c["summary"])
        p.runs[("getcap", "/bin/newgidmap")] = (0, "/bin/newgidmap cap_setuid=ep\n")      # the wrong capability
        self.assertMissing(p, "uidmap", owner="administrator")

    def test_uidmap_helpers_must_be_owned_by_root(self):
        p = FakeProbe()
        p.stats["/bin/newuidmap"] = (UID, stat.S_IFREG | stat.S_ISUID | 0o755)
        c = self.assertMissing(p, "uidmap", owner="administrator")
        self.assertIn("root", c["summary"])

    def test_uidmap_inspects_the_first_path_hit_only_and_names_it(self):
        p = FakeProbe()
        c = check(p)["uidmap"]
        self.assertEqual(c["status"], "ok")
        self.assertIn("/bin/newuidmap", c["summary"])
        self.assertIn("/bin/newgidmap", c["summary"])
        self.assertEqual([x for x in p.calls if x[0] == "which" and x[1] == "newuidmap"], [("which", "newuidmap")])

    def test_subuid_and_subgid(self):
        p = FakeProbe()
        p.files[sc.SUBID_FILES["subuid"]] = "bob:100000:65536\n"
        self.assertMissing(p, "subuid", owner="administrator")
        p = FakeProbe()
        p.files[sc.SUBID_FILES["subgid"]] = f"{USER}:100000:1000\n"
        c = self.assertMissing(p, "subuid", owner="administrator")
        self.assertIn("subgid", c["summary"])
        p = FakeProbe()
        p.files[sc.SUBID_FILES["subuid"]] = f"{UID}:100000:65536\n"            # an entry by uid counts
        self.assertEqual(check(p)["subuid"]["status"], "ok")
        p = FakeProbe()
        del p.files[sc.SUBID_FILES["subuid"]]
        self.assertMissing(p, "subuid", owner="administrator")

    def test_subuid_ranges_add_up(self):
        p = FakeProbe()
        half = sc.SUBID_MIN_COUNT // 2
        p.files[sc.SUBID_FILES["subuid"]] = f"{USER}:100000:{half}\n{USER}:200000:{half}\n"
        self.assertEqual(check(p)["subuid"]["status"], "ok")

    def test_userns_disabled_and_apparmor_note(self):
        p = FakeProbe()
        p.files[sc.USERNS_MAX_FILE] = "0\n"
        self.assertMissing(p, "userns", owner="administrator")
        p = FakeProbe()
        del p.files[sc.USERNS_MAX_FILE]
        self.assertMissing(p, "userns", owner="administrator", status="unknown")

    def test_apparmor_restriction_is_its_own_informational_unknown(self):
        """The restriction does not say whether a profile allows rootlesskit, so it is `unknown`, and it is not a
        required check (a healthy Ubuntu 24.04 host with the profile must not exit 1)."""
        p = FakeProbe()
        p.files[sc.USERNS_APPARMOR_FILE] = "1\n"
        checks = check(p)
        c = checks["userns_apparmor"]
        self.assertEqual((c["status"], c["required"], c["owner"]), ("unknown", False, "administrator"))
        self.assertIn("apparmor", c["summary"].lower())
        self.assertEqual(checks["userns"]["status"], "ok")
        self.assertEqual(sc.exit_code(list(checks.values())), 0)
        for text in ("0\n", None):
            p = FakeProbe()
            if text is None:
                del p.files[sc.USERNS_APPARMOR_FILE]
            else:
                p.files[sc.USERNS_APPARMOR_FILE] = text
            self.assertEqual(check(p)["userns_apparmor"]["status"], "ok")

    def test_the_subuid_fix_hardcodes_no_range(self):
        p = FakeProbe()
        del p.files[sc.SUBID_FILES["subuid"]]
        c = check(p)["subuid"]
        self.assertNotIn("100000", c["fix"])
        self.assertIn("not used in /etc/subuid", c["fix"])
        self.assertIn("usermod --add-subuids", c["fix"])

    def test_unreadable_subid_files_are_unknown_not_missing(self):
        p = FakeProbe()
        del p.files[sc.SUBID_FILES["subuid"]]
        p.stats[sc.SUBID_FILES["subuid"]] = (0, stat.S_IFREG | 0o600)        # it exists but cannot be read
        self.assertMissing(p, "subuid", owner="administrator", status="unknown")

    def test_a_non_file_subid_source_makes_a_short_file_unknown(self):
        for line in ("subid: sss", "subid:      files sss", "subid: ldap"):
            p = FakeProbe()
            p.files[sc.SUBID_FILES["subuid"]] = "bob:100000:65536\n"
            p.files[sc.NSSWITCH_FILE] = f"passwd: files\n{line}\n"
            c = check(p)["subuid"]
            self.assertEqual(c["status"], "unknown", line)
            self.assertIn("nsswitch", c["summary"])
        p = FakeProbe()
        p.files[sc.SUBID_FILES["subuid"]] = "bob:100000:65536\n"
        p.files[sc.NSSWITCH_FILE] = "subid: files\n"
        self.assertEqual(check(p)["subuid"]["status"], "missing")
        p.files[sc.SUBID_FILES["subuid"]] = f"{USER}:100000:65536\n"
        p.files[sc.SUBID_FILES["subgid"]] = f"{USER}:100000:65536\n"
        p.files[sc.NSSWITCH_FILE] = "subid: sss\n"
        self.assertEqual(check(p)["subuid"]["status"], "ok")                 # enough in the files: nothing to doubt

    def test_an_unreadable_cdi_directory_or_spec_is_unknown_not_missing(self):
        p = FakeProbe()
        del p.dirs["/cdi"]
        p.stats["/cdi"] = (0, stat.S_IFDIR | 0o700)                          # present, listing denied
        c = check(p, gpu=True)["nvidia_cdi"]
        self.assertEqual(c["status"], "unknown")
        self.assertIn("/cdi", c["summary"])
        p = FakeProbe()
        del p.files["/cdi/nvidia.yaml"]                                      # listed, content denied
        self.assertEqual(check(p, gpu=True)["nvidia_cdi"]["status"], "unknown")
        p = FakeProbe()
        del p.dirs["/cdi"]                                                   # absent everywhere: really missing
        self.assertEqual(check(p, gpu=True)["nvidia_cdi"]["status"], "missing")

    def test_docker_config_current_context_is_read_through_the_probe_and_noted(self):
        for home_cfg, env in (("/home/u/.docker/config.json", {"HOME": "/home/u"}),
                              ("/dc/config.json", {"HOME": "/home/u", "DOCKER_CONFIG": "/dc"})):
            p = FakeProbe()
            p.environ.update(env)
            p.files[home_cfg] = json.dumps({"currentContext": "remote-box"})
            c = check(p)["rootless_docker"]
            self.assertEqual(c["status"], "ok")
            self.assertIn("remote-box", c["summary"] + c["fix"])
            self.assertIn("config.json", c["summary"] + c["fix"])
            self.assertIn(("read_text", home_cfg), p.calls)
        for text in (json.dumps({"currentContext": "default"}), json.dumps({}), "not json", json.dumps([1])):
            p = FakeProbe()
            p.environ["HOME"] = "/home/u"
            p.files["/home/u/.docker/config.json"] = text
            self.assertNotIn("config.json", check(p)["rootless_docker"]["summary"])

    def test_deeply_nested_json_is_unknown_input_not_a_crash(self):
        """Every JSON read of this module (Docker config, `docker info`, a CDI spec) treats a document nested beyond the
        parser's depth as unreadable, like malformed JSON."""
        deep = "[" * 100000 + "]" * 100000
        p = FakeProbe()
        p.environ["HOME"] = "/home/u"
        p.files["/home/u/.docker/config.json"] = deep
        self.assertIsNone(sc.configured_context(p))
        self.assertEqual(check(p)["rootless_docker"]["status"], "ok")
        self.assertIsNone(sc.cdi_kind("{" + deep))
        self.assertIsNone(sc.cdi_kind('{"kind": ' + deep + "}"))
        p = FakeProbe()
        p.runs[("docker", "info")] = (0, deep)
        c = check(p)["rootless_docker"]
        self.assertEqual(c["status"], "unknown", c)

    def test_printed_environment_values_cannot_inject_terminal_text(self):
        p = FakeProbe()
        p.environ["DOCKER_HOST"] = "unix:///x/\x1b[31m\nforged.sock"
        p.stats["/x/\x1b[31m\nforged.sock"] = (4321, stat.S_IFSOCK | 0o660)
        del p.stats[f"{RT}/docker.sock"]
        p.runs[("docker", "info")] = (1, "")
        c = check(p)["rootless_docker"]
        for part in (c["summary"], c["fix"]):
            self.assertNotIn("\x1b", part)
            self.assertNotIn("\n", part)
        p = FakeProbe()
        p.environ["DOCKER_HOST"] = "unix:///x/d.sock\x1b[2J"
        c = check(p)["rootless_docker"]
        self.assertNotIn("\x1b", c["summary"] + c["fix"])
        p = FakeProbe()
        p.environ["DOCKER_CONTEXT"] = "a\x1b[31m"
        self.assertNotIn("\x1b", check(p)["rootless_docker"]["summary"])

    def test_named_limits_have_reasons_and_are_applied_at_their_edge(self):
        for name in ("READ_LIMIT_BYTES", "DISPLAY_MAX_CHARS", "SCHEME_MAX_LEN", "NSSWITCH_FILE", "DOCKER_CONFIG_FILE"):
            self.assertIn(name, sc.CONSTANT_REASONS)
        p = FakeProbe()
        p.environ["DOCKER_CONTEXT"] = "c" * sc.DISPLAY_MAX_CHARS
        self.assertIn("c" * sc.DISPLAY_MAX_CHARS, check(p)["rootless_docker"]["summary"])
        p.environ["DOCKER_CONTEXT"] = "c" * (sc.DISPLAY_MAX_CHARS + 1)
        self.assertIn("(unprintable or long)", check(p)["rootless_docker"]["summary"])
        for n, shown in ((sc.SCHEME_MAX_LEN, True), (sc.SCHEME_MAX_LEN + 1, False)):
            p = FakeProbe()
            p.environ["DOCKER_HOST"] = "a" * n + "://h"
            self.assertEqual("a" * n in check(p)["rootless_docker"]["summary"], shown, n)

    def test_the_real_probe_reads_at_most_the_named_limit(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w") as fh:
            fh.write("x" * (sc.READ_LIMIT_BYTES + 10))
            fh.flush()
            self.assertEqual(len(sc.SystemProbe().read_text(fh.name)), sc.READ_LIMIT_BYTES)

    def test_rootless_net_helper(self):
        p = FakeProbe()
        del p.binaries["slirp4netns"]
        self.assertMissing(p, "rootless_net", owner="administrator")
        p.binaries["pasta"] = "/bin/pasta"
        self.assertEqual(check(p)["rootless_net"]["status"], "ok")

    def test_gpu_checks(self):
        p = FakeProbe()
        del p.binaries["nvidia-smi"]
        self.assertMissing(p, "nvidia_driver", owner="administrator", gpu=True)
        p = FakeProbe()
        p.dirs["/cdi"] = []
        self.assertMissing(p, "nvidia_cdi", owner="administrator", gpu=True)
        p = FakeProbe()
        p.files["/cdi/nvidia.yaml"] = "kind: other.com/thing\n"
        self.assertMissing(p, "nvidia_cdi", owner="administrator", gpu=True)

    def test_cdi_spec_is_parsed_not_substring_matched(self):
        decoys = {
            "a.yaml": f"cdiVersion: 0.5.0\nkind: other.com/x\n# {sc.CDI_GPU_KIND}\n",              # in a comment
            "b.yaml": f"cdiVersion: 0.5.0\nkind: other.com/x\ndevices:\n- name: a\n  note: kind: {sc.CDI_GPU_KIND}\n",
            "c.json": json.dumps({"cdiVersion": "0.5.0", "kind": "other.com/x", "devices": [{"name": sc.CDI_GPU_KIND}]}),
            "d.yaml": f"cdiVersion: 0.5.0\n  kind: {sc.CDI_GPU_KIND}\n",                               # not top level
            "e.yaml": f"kind: {sc.CDI_GPU_KIND}-extra\n",
        }
        for name, text in decoys.items():
            p = FakeProbe()
            p.dirs["/cdi"] = [name]
            p.files = {k: v for k, v in p.files.items() if not k.startswith("/cdi")}
            p.files[f"/cdi/{name}"] = text
            self.assertMissing(p, "nvidia_cdi", owner="administrator", gpu=True)

    def test_cdi_spec_forms_that_count(self):
        good = {"a.json": json.dumps({"cdiVersion": "0.5.0", "kind": sc.CDI_GPU_KIND, "devices": []}),
                "b.yaml": f"cdiVersion: 0.5.0\nkind: '{sc.CDI_GPU_KIND}'  # gpus\n",
                "c.yml": f'kind: "{sc.CDI_GPU_KIND}"\n', "d.yaml": f"---\nkind: {sc.CDI_GPU_KIND}\n"}
        for name, text in good.items():
            p = FakeProbe()
            p.dirs["/cdi"] = [name]
            p.files[f"/cdi/{name}"] = text
            self.assertEqual(check(p, gpu=True)["nvidia_cdi"]["status"], "ok", name)

    def test_a_spec_only_in_the_user_directory_is_unknown_with_a_note(self):
        p = FakeProbe()
        p.dirs["/cdi"] = []
        p.dirs["/cfg/cdi"] = ["nvidia.yaml"]
        p.files["/cfg/cdi/nvidia.yaml"] = f"kind: {sc.CDI_GPU_KIND}\n"
        c = check(p, gpu=True)["nvidia_cdi"]
        self.assertEqual(c["status"], "unknown")
        self.assertIn("dockerd does not read this directory unless configured", c["summary"])

    def test_the_daemon_directories_come_from_the_named_table_only(self):
        probe = probe_with_env({"XDG_CONFIG_HOME": "/c"})
        self.assertEqual(sc.SystemProbe.cdi_dirs(probe), list(sc.CDI_SYSTEM_DIRS))
        self.assertEqual(sc.SystemProbe.cdi_user_dirs(probe), ["/c/cdi"])

    def test_cdi_ok_carries_the_feature_note(self):
        c = check(FakeProbe(), gpu=True)["nvidia_cdi"]
        self.assertEqual(c["status"], "ok")
        self.assertIn("cdi", c["fix"].lower())
        self.assertIn("daemon", c["fix"].lower())

    def test_gpu_checks_are_not_applicable_without_gpu(self):
        p = FakeProbe()
        del p.binaries["nvidia-smi"]
        p.dirs["/cdi"] = []
        for cid in ("nvidia_driver", "nvidia_cdi"):
            self.assertMissing(p, cid, owner="administrator", required=False, status="not_applicable", gpu=False)
        self.assertEqual(sc.exit_code(sc.run_checks(p, gpu=False)), 0)

    def test_gpu_default_follows_the_nvidia_proc_directory(self):
        p = FakeProbe()
        self.assertTrue(sc.gpu_default(p))
        del p.dirs[sc.NVIDIA_PROC_DIR]
        self.assertFalse(sc.gpu_default(p))
        p = FakeProbe()
        del p.dirs[sc.NVIDIA_PROC_DIR]
        self.assertEqual(check(p, gpu=None)["nvidia_driver"]["status"], "not_applicable")

    def test_informational_checks_never_fail_the_exit_code(self):
        p = FakeProbe()
        del p.files[sc.LINGER_DIR + "/" + USER]
        p.memlock = 65536
        p.files[sc.PTRACE_SCOPE_FILE] = "0\n"
        p.environ.pop("XDG_RUNTIME_DIR")
        checks = check(p)
        for cid in ("systemd_user", "linger", "memlock", "ptrace_scope"):
            self.assertEqual(checks[cid]["status"], "missing", cid)
            self.assertFalse(checks[cid]["required"], cid)
        self.assertIn("VLM pack runs without it", checks["memlock"]["fix"])
        self.assertEqual(sc.exit_code(list(checks.values())), 1 if checks["rootless_docker"]["status"] != "ok" else 0)

    def test_systemd_user_needs_a_runtime_dir_and_a_reachable_manager(self):
        p = FakeProbe()
        p.runs.pop(("systemctl", "--user", "is-system-running"))
        self.assertEqual(check(p)["systemd_user"]["status"], "unknown")
        p = FakeProbe()
        p.runs[("systemctl", "--user", "is-system-running")] = (1, "degraded")
        self.assertEqual(check(p)["systemd_user"]["status"], "ok")

    def test_administrator_fixes_say_gatedtrain_needs_no_sudo(self):
        p = FakeProbe()
        p.binaries.clear()
        for c in sc.run_checks(p, gpu=True):
            if c["owner"] == "administrator" and c["status"] == "missing":
                self.assertIn("not a gatedtrain step", c["fix"], c)
                self.assertIn("needs no sudo", c["fix"], c)

    def test_a_crashing_check_is_unknown_not_a_crash(self):
        p = FakeProbe()
        p.which = mock.Mock(side_effect=OSError("boom"))
        checks = check(p)
        self.assertEqual(checks["rootless_docker"]["status"], "unknown")
        self.assertEqual(len(checks), len(sc.CHECKS))


class ExitCodeTest(unittest.TestCase):                              # AN-33
    def test_required_missing_or_unknown_is_one(self):
        for cid, mutate in (("uidmap", lambda p: p.binaries.pop("newuidmap")),
                            ("rootless_docker", lambda p: p.runs.clear())):
            p = FakeProbe()
            mutate(p)
            checks = sc.run_checks(p, gpu=False)
            self.assertEqual(sc.exit_code(checks), 1, cid)
            self.assertIn(cid, sc.failing_ids(checks))


class SystemProbeTest(unittest.TestCase):                           # AN-32: the real probe on this host
    def test_reads_do_not_raise(self):
        probe = sc.SystemProbe()
        self.assertIsInstance(probe.system(), str)
        self.assertIsInstance(probe.uid(), int)
        self.assertIsNone(probe.read_text("/nonexistent/x"))
        self.assertIsNone(probe.listdir("/nonexistent"))
        self.assertIsNone(probe.stat("/nonexistent/x"))
        self.assertIsNone(probe.which("definitely-not-a-command-xyz"))
        self.assertIsNone(probe.run(["definitely-not-a-command-xyz"], timeout=1))
        checks = sc.run_checks(probe, gpu=False)
        self.assertEqual(len(checks), len(sc.CHECKS))


class CliTest(unittest.TestCase):                                   # AN-33
    def run_cli(self, argv, probe=None):
        probe = probe or FakeProbe()
        with mock.patch.object(cli_setup_check, "make_probe", lambda: probe), no_processes():
            return run(["setup", "check", *argv]), probe

    def test_json_envelope_and_exit_zero(self):
        (code, doc), probe = self.run_cli(["--gpu"])
        self.assertEqual((code, doc["command"], doc["ok"], doc["problems"]), (0, "setup check", True, []), doc)
        self.assertEqual(set(doc["data"]), {"checks", "gpu", "failing", "needs_administrator"})
        self.assertEqual(doc["data"]["failing"], [])
        self.assertTrue(doc["data"]["gpu"])
        self.assertEqual([c["id"] for c in doc["data"]["checks"]], [s.id for s in sc.CHECKS])

    def test_exit_one_names_the_failing_checks(self):
        p = FakeProbe()
        del p.binaries["newuidmap"]
        p.files[sc.SUBID_FILES["subuid"]] = ""
        (code, doc), _ = self.run_cli(["--no-gpu"], p)
        self.assertEqual((code, doc["ok"]), (1, False), doc)
        self.assertEqual(doc["data"]["failing"], ["uidmap", "subuid"])
        self.assertEqual(doc["data"]["needs_administrator"], ["uidmap", "subuid"])

    def test_no_gpu_flag_wins_over_the_default(self):
        (code, doc), _ = self.run_cli(["--no-gpu"])
        self.assertFalse(doc["data"]["gpu"])
        self.assertEqual({c["id"]: c["status"] for c in doc["data"]["checks"]}["nvidia_cdi"], "not_applicable")

    def test_human_table_and_closing_line(self):
        p = FakeProbe()
        del p.binaries["newuidmap"]
        with mock.patch.object(cli_setup_check, "make_probe", lambda: p), no_processes():
            code, text, err = run_text(["setup", "check", "--no-gpu"])
        self.assertEqual(code, 1)
        for spec in sc.CHECKS:
            self.assertIn(spec.id, text)
        self.assertIn("uidmap", text)
        self.assertTrue(text.rstrip().endswith(sc.CLOSING_LINE), text[-300:])
        self.assertEqual(sc.CLOSING_LINE, 'gatedtrain itself needs no sudo for the same-user setup; items marked '
                         '"administrator" are Docker/OS/NVIDIA prerequisites that need root on a machine that lacks '
                         'them.')

    def test_the_command_writes_nothing_and_runs_nothing_privileged(self):
        (_, _), probe = self.run_cli([])
        for call in probe.calls:
            self.assertNotIn(call[0], ("write", "write_text", "mkdir", "chmod", "sudo"))

    def test_bad_flag_is_a_usage_error(self):
        with self.assertRaises(SystemExit) as ctx, no_processes():
            run_text(["setup", "check", "--bogus"])
        self.assertEqual(ctx.exception.code, 2)

    def test_the_json_line_is_json(self):
        p = FakeProbe()
        with mock.patch.object(cli_setup_check, "make_probe", lambda: p), no_processes():
            code, text, err = run_text(["--json", "setup", "check", "--no-gpu"])
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(json.loads(text)["command"], "setup check")


if __name__ == "__main__":
    unittest.main()
