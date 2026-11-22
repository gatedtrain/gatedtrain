"""AN-21, AN-14: rootful-access detection by the owner's account and the downgraded keeper guarantee.

The detection (`gatedtrain.ainative.keeper.rootful.owner_root_docker`) runs on fake host trees under a temporary
directory (`HostView(root=tmp, root_uid=os.getuid())`, real Unix sockets bound in the tree), never on the real host:
the CI runner is in the `docker` group. Keeper subprocesses read the tree `GATEDTRAIN_TEST_HOST_VIEW` names
(`tests/_keeper.py` writes a clean one by default). The sticky per-set downgrade runs on `KeeperState` in this process
with an owner uid other than the test user's, so the separate-user texts are the ones stated.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from _keeper import HOST_VIEW_ENV, KeeperProcess, skip_unless_linux, write_host_tree
import test_an_keeper_attempts as att
from test_an_keeper_attempts import BASE, TOY, arm, registration
from test_an_keeper_cli import CliProject, run
from test_an_keeper_intake import confirm_args, project_id

from gatedtrain.ainative import signing
from gatedtrain.ainative.keeper import rootful, server, texts
from gatedtrain.ainative.keeper.protocol import KeeperClient
from gatedtrain.ainative.keeper.rootful import HostView, owner_root_docker
from gatedtrain.ainative.keeper.state import KeeperState
from gatedtrain.ainative.keeper_trust import RECEIPTS
from gatedtrain.ainative.trust_stages import render_markdown
from gatedtrain.ainative.verify import ROOT_DOCKER_KEEPER, ROOT_DOCKER_REPORT, UNCHECKED_GUARANTEE_REPORT
from gatedtrain.canonical import sha256_bytes

UID = os.getuid()
GID = os.getgid()
OTHER_GID = 4242 if GID != 4242 else 4243          # a primary gid no fake group makes risky


def status(uid: int, groups: list[int], name: str = "bash") -> str:
    return (f"Name:\t{name}\nUmask:\t0022\nState:\tS (sleeping)\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n"
            f"Gid:\t{OTHER_GID}\t{OTHER_GID}\t{OTHER_GID}\t{OTHER_GID}\nGroups:\t{' '.join(map(str, groups))}\n")


def tcp_line(port: int, state: str, *, v6: bool = False) -> str:
    addr = ("0" * 32) if v6 else "00000000"
    return (f"   0: {addr}:{port:04X} {addr}:0000 {state} 00000000:00000000 00:00000000 00000000     0        0 "
            "12345 1 0000000000000000 100 0 0 10 0\n")


class Tree(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="gr-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.socks: list[socket.socket] = []
        self.addCleanup(lambda: [s.close() for s in self.socks])

    def view(self, **kw) -> HostView:
        return HostView(root=self.root, root_uid=kw.pop("root_uid", UID), **kw)

    def tree(self, **kw) -> Path:
        kw.setdefault("primary_gid", OTHER_GID)
        return write_host_tree(self.root, **kw)

    def bind(self, rel: str = "run/docker.sock", mode: int = 0o660) -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(str(path))
        self.socks.append(s)
        os.chmod(path, mode)
        return path

    def reasons(self, **kw) -> list[str]:
        return owner_root_docker(UID, self.view(**kw))


@skip_unless_linux
class DetectionTest(Tree):
    def test_constants(self):
        self.assertEqual(rootful.RULE, "AN-21")
        self.assertEqual(rootful.DOCKER_TCP_PORTS, (2375, 2376))
        self.assertEqual(rootful.ROOTFUL_SOCKET_GLOBS[-2:], ("/run/containerd/containerd.sock",
                                                             "/var/run/containerd/containerd.sock"))
        self.assertIn("/var/run/docker.sock", rootful.ROOTFUL_SOCKET_GLOBS)

    def test_clean_host(self):
        self.tree()
        self.assertEqual(self.reasons(), [])
        self.bind()                                     # a rootful socket alone, group not the owner's: still clean
        self.assertEqual(self.reasons(), [])

    def test_docker_group_member(self):
        for name in ("docker", "lxd", "incus-admin", "disk"):
            with self.subTest(group=name):
                self.tree(group=f"root:x:0:\n{name}:x:998:alice,owner\nusers:x:100:owner\n")
                reasons = self.reasons()
                self.assertEqual(len(reasons), 1, reasons)
                self.assertIn(f"uid {UID} is in gid 998 (group {name}) by /etc/group", reasons[0])
        self.tree(group="root:x:0:\ndocker:x:998:alice\n")              # another user's membership
        self.assertEqual(self.reasons(), [])
        self.tree(group="root:x:0:\ndocker:x:998:\n", primary_gid=998)  # the primary gid in /etc/passwd
        self.assertIn("by its primary gid", self.reasons()[0])

    def test_primary_gid_owns_socket(self):
        self.tree(primary_gid=GID)
        self.bind("var/run/docker.sock")
        reasons = self.reasons()
        self.assertEqual(reasons, [f"uid {UID} is in gid {GID} (the group of /var/run/docker.sock) by its primary "
                                   "gid (/etc/passwd)"])
        self.assertEqual(self.reasons(root_uid=UID + 1), [])             # not owned by root: not rootful
        self.tree(primary_gid=OTHER_GID, group=f"root:x:0:\nsock:x:{GID}:owner\n")
        self.assertIn("by /etc/group", self.reasons()[0])

    def test_containerd_socket(self):
        self.tree(primary_gid=GID)
        self.bind("run/containerd/containerd.sock")
        self.assertIn("/run/containerd/containerd.sock", self.reasons()[0])

    def test_old_session_gid(self):                     # Review Focus 4: left the group, an old session keeps it
        self.tree(statuses={1234: status(UID, [100, GID]), 1300: status(UID + 1, [GID], name="other"),
                            1400: status(UID, [100], name="fresh")})
        self.bind()
        reasons = self.reasons()
        self.assertEqual(reasons, [f"pid 1234 (bash) of uid {UID} carries gid {GID} (the group of /run/docker.sock)"])
        self.tree(group="root:x:0:\ndocker:x:998:\n", statuses={77: status(UID, [998], name="tmux")})
        self.assertIn(f"pid 77 (tmux) of uid {UID} carries gid 998 (group docker)", self.reasons())

    def test_world_writable_socket(self):
        self.tree()
        self.bind(mode=0o666)
        self.assertEqual(self.reasons(), ["/run/docker.sock is a rootful daemon socket writable by every user"])
        self.bind("run/docker-x.sock", mode=0o666)
        self.assertEqual(len(self.reasons(root_uid=UID + 1)), 0)        # not owned by root: ignored

    def test_tcp_listener(self):
        self.tree(tcp=tcp_line(0x0947, "0A"))
        self.assertEqual(self.reasons(), ["a TCP socket listens on port 2375, a Docker TCP endpoint port "
                                          "(/proc/net/tcp)"])
        self.tree(tcp=tcp_line(2375, "01") + tcp_line(2377, "0A"))       # established, or another port
        self.assertEqual(self.reasons(), [])
        self.tree(tcp6=tcp_line(2376, "0A", v6=True))
        self.assertEqual(self.reasons(), ["a TCP socket listens on port 2376, a Docker TCP endpoint port "
                                          "(/proc/net/tcp6)"])
        self.tree()
        (self.root / "proc" / "net" / "tcp6").unlink()                    # no IPv6: fine
        self.assertEqual(self.reasons(), [])

    def test_hidepid_fails_closed(self):                # Review Focus 4
        for opt in ("hidepid=2", "hidepid=invisible", "hidepid=1"):
            with self.subTest(opt=opt):
                self.tree(mountinfo=f"22 1 0:21 / /proc rw,nosuid,relatime shared:12 - proc proc rw,{opt}\n")
                reasons = self.reasons()
                self.assertEqual(len(reasons), 1, reasons)
                self.assertIn("hidepid", reasons[0])
                self.assertIn("fail closed", reasons[0])
        self.tree(mountinfo="22 1 0:21 / /proc rw,nosuid,relatime shared:12 - proc proc rw,hidepid=0\n")
        self.assertEqual(self.reasons(), [])

    def test_unreadable_group_fails_closed(self):
        self.tree()
        (self.root / "etc" / "group").unlink()
        self.assertEqual(self.reasons(), ["cannot read /etc/group (FileNotFoundError): counted as rootful access "
                                          "(fail closed)"])
        self.tree()
        (self.root / "proc" / "self" / "mountinfo").unlink()
        self.assertIn("cannot read /proc/self/mountinfo", self.reasons()[0])
        self.tree()
        shutil.rmtree(self.root / "proc")
        reasons = self.reasons()
        self.assertTrue(any(r.startswith("cannot read /proc ") for r in reasons), reasons)

    def test_unreadable_status_fails_closed(self):     # preflight finding 12: only a vanished process is skipped
        self.tree()
        (self.root / "proc" / "55").mkdir()                               # no status: the process ended
        self.assertEqual(self.reasons(), [])
        (self.root / "proc" / "56" / "status").mkdir(parents=True)        # unreadable for another reason
        self.assertEqual(self.reasons(), ["cannot read /proc/56/status (IsADirectoryError): counted as rootful "
                                          "access (fail closed)"])

    def test_reasons_are_capped(self):
        self.tree(group="root:x:0:\ndocker:x:998:\n",
                  statuses={pid: status(UID, [998]) for pid in range(100, 100 + rootful.MAX_REASONS + 5)})
        reasons = self.reasons()
        self.assertEqual(len(reasons), rootful.MAX_REASONS + 1)
        self.assertEqual(reasons[-1], "and 5 more reasons")

    def test_name_service_groups(self):                # preflight finding 12: sssd or LDAP groups count
        self.tree(group="root:x:0:\n")
        with mock.patch.object(rootful.grp, "getgrnam", side_effect=lambda n: mock.Mock(gr_gid=990)
                               if n == "docker" else (_ for _ in ()).throw(KeyError(n))), \
                mock.patch.object(rootful.os, "getgrouplist", return_value=[OTHER_GID, 990]) as gl:
            reasons = self.reasons(nss=True)
            self.assertEqual(reasons, [f"uid {UID} is in gid 990 (group docker) by the name service (getgrouplist)"])
            gl.assert_called_with("owner", OTHER_GID)
            self.assertEqual(self.reasons(), [])                          # a fake tree never asks the name service

    def test_reasons_hold_no_names_but_ids_and_paths(self):
        self.tree(group="root:x:0:\ndocker:x:998:owner\n", statuses={9: status(UID, [998])})
        for r in self.reasons():
            self.assertNotIn("owner", r)
            self.assertNotIn(str(self.root), r)


class GuaranteeTest(unittest.TestCase):
    def test_guarantee_precedence(self):
        g = texts.sealed_guarantee
        self.assertEqual(g(5, 5), texts.KEEPER_SAME_USER_GUARANTEE)
        self.assertEqual(g(5, 5, root_docker=True), texts.KEEPER_SAME_USER_GUARANTEE)
        self.assertEqual(g(5, 6), texts.KEEPER_GUARANTEE)
        self.assertEqual(g(5, 6, root_docker=True), texts.KEEPER_ROOT_DOCKER_GUARANTEE)
        self.assertTrue(texts.KEEPER_ROOT_DOCKER_GUARANTEE.startswith("the keeper runs as a separate OS user, but "))
        self.assertIn("take in new data", texts.KEEPER_ROOT_DOCKER_GUARANTEE)
        self.assertIn("a root user or the keeper's own account can break all of it.", texts.KEEPER_GUARANTEE_V1)


@skip_unless_linux
class HelloTest(Tree):
    def docker_tree(self) -> Path:
        return self.tree(group="root:x:0:\ndocker:x:998:owner\n")

    def test_hello_fields(self):
        with KeeperProcess(env={HOST_VIEW_ENV: str(self.docker_tree())}) as kp:
            hello = kp.client.call("hello", None, {})
            code, doc = run(["keeper", "fingerprint", "--socket", str(kp.socket)])
        self.assertIs(hello["owner_has_root_docker"], True)
        self.assertEqual(hello["root_docker_reasons"], [f"uid {UID} is in gid 998 (group docker) by /etc/group"])
        self.assertEqual(hello["sealed_guarantee"], texts.KEEPER_SAME_USER_GUARANTEE)   # same user wins
        self.assertEqual(code, 0, doc)                                                # preflight finding 6
        self.assertIs(doc["data"]["owner_has_root_docker"], True)
        self.assertEqual(doc["data"]["root_docker_reasons"], hello["root_docker_reasons"])
        with KeeperProcess() as kp:                    # the default: a clean fake tree, whatever the host's groups
            hello = kp.client.call("hello", None, {})
        self.assertEqual((hello["owner_has_root_docker"], hello["root_docker_reasons"]), (False, []))

    def test_hello_separate_user(self):                # preflight finding 6: a keeper under its own uid
        keeper = server.Keeper(self.root, UID, host_view=self.view())
        keeper.keeper_uid = UID + 1
        self.docker_tree()
        hello = keeper.handle(b'{"op":"hello","project":null,"args":{}}', UID)["result"]
        self.assertEqual((hello["owner_has_root_docker"], hello["sealed_guarantee"]),
                         (True, texts.KEEPER_ROOT_DOCKER_GUARANTEE))
        self.tree()
        hello = keeper.handle(b'{"op":"hello","project":null,"args":{}}', UID)["result"]
        self.assertEqual((hello["owner_has_root_docker"], hello["sealed_guarantee"]), (False, texts.KEEPER_GUARANTEE))


@skip_unless_linux
class StickyTest(unittest.TestCase):
    """`KeeperState` in this process for an owner uid other than the test user's (preflight finding 2)."""

    OWNER = UID + 1

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gk-state-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.dir = self.tmp / "state"
        self.dir.mkdir(mode=0o700)
        att.trust_stages._scenario_entry_points = lambda: [att._ScenarioEP()]
        att.trust_stages._reset_scenario_snapshot()
        self.addCleanup(att.trust_stages._reset_scenario_snapshot)
        self.addCleanup(setattr, att.trust_stages, "_scenario_entry_points", att._ORIGINAL_EPS)
        self.positive = True
        self.calls = 0
        self.state = self.open()

    def candidate(self, r: dict, k: int) -> dict:
        full = att.pack.fit(r["train_rows"], 0.0)
        tok = sorted(full)[k]
        return arm(att.model({tok: full[tok]}, f"one-token-{k}"))

    def open(self) -> KeeperState:

        def detect() -> list[str]:
            self.calls += 1
            return ["uid 1 is in gid 998 (group docker) by /etc/group"] if self.positive else []
        st = KeeperState(self.dir, owner_uid=self.OWNER, root_docker=detect)
        self.addCleanup(st.close)
        return st

    def handle(self, keeper, op: str, project: str | None) -> dict:
        resp = keeper.handle(json.dumps({"op": op, "project": project, "args": {}}).encode(), UID)
        self.assertIs(resp["ok"], True, resp)
        return resp["result"]

    def test_report_sticky(self):                      # Review Focus 4
        hv = Path(tempfile.mkdtemp(prefix="gr-"))
        self.addCleanup(shutil.rmtree, hv, True)
        keeper = server.Keeper(self.dir, self.OWNER, key=self.state.key, state=self.state,
                               host_view=HostView(root=write_host_tree(hv, owner_uid=self.OWNER), root_uid=UID))
        pid = project_id("p1")
        self.state.open_project(pid)
        r = self.state.intake(pid, TOY, sha256_bytes(TOY))
        self.assertEqual(r["sealed_summary"]["guarantee"], texts.KEEPER_ROOT_DOCKER_GUARANTEE)
        seen = self.state.root_docker_seen(r["sealed_id"])
        self.assertEqual(seen["reasons"], ["uid 1 is in gid 998 (group docker) by /etc/group"])
        self.positive = False                           # the owner left the group, every session ended
        args = confirm_args()
        sentence = args.pop("sentence")
        self.state.confirm_metric(pid, args, sentence, self.OWNER)
        reg = self.state.register(pid, registration(r["sealed_id"]))["hash"]
        a = self.state.evaluate(pid, reg, arm(BASE), self.candidate(r, 0))
        self.state.gate(pid, a["attempt_id"])
        self.assertEqual(self.state.sealed_summary(pid)["guarantee"], texts.KEEPER_ROOT_DOCKER_GUARANTEE)
        hello = self.handle(keeper, "hello", pid)
        self.assertEqual((hello["sealed_guarantee"], hello["owner_has_root_docker"]),
                         (texts.KEEPER_ROOT_DOCKER_GUARANTEE, True))
        self.assertTrue(hello["root_docker_reasons"][0].startswith(f"recorded on this sealed set at "
                                                                   f"{seen['first_utc']}: "), hello)
        hello = self.handle(keeper, "hello", None)      # the current host state
        self.assertEqual((hello["sealed_guarantee"], hello["owner_has_root_docker"], hello["root_docker_reasons"]),
                         (texts.KEEPER_GUARANTEE, False, []))
        report = signing.verify(self.state.report(pid)["report"], self.state.key.public_key(), kind="report")
        self.assertEqual(report["sealed_guarantee"], texts.KEEPER_ROOT_DOCKER_GUARANTEE)
        self.assertEqual(report["known_limits"][0], {"id": "sealed_guarantee",
                                                     "text": texts.KEEPER_ROOT_DOCKER_GUARANTEE})
        self.assertEqual(self.state.root_docker_seen(r["sealed_id"]), seen)      # recorded once
        self.assertGreaterEqual(self.calls, 5)          # intake, sealed summary, evaluate, gate, report

    def test_clean_set_states_the_separate_user_text(self):
        self.positive = False
        pid, r, reg = self.ready_as_owner()
        self.assertEqual(r["sealed_summary"]["guarantee"], texts.KEEPER_GUARANTEE)
        self.positive = True                            # seen at evaluate: from then on downgraded
        a = self.state.evaluate(pid, reg, arm(BASE), self.candidate(r, 0))
        self.state.gate(pid, a["attempt_id"])
        self.assertEqual(self.state.sealed_summary(pid)["guarantee"], texts.KEEPER_ROOT_DOCKER_GUARANTEE)

    def test_failing_detector_fails_closed(self):
        self.state.root_docker = lambda: 1 / 0
        self.assertEqual(self.state.detect_root_docker(), ["the rootful-access detection failed "
                                                           "(ZeroDivisionError): counted as rootful access"])

    def ready_as_owner(self):
        pid = project_id("p2")
        self.state.open_project(pid)
        r = self.state.intake(pid, TOY, sha256_bytes(TOY))
        args = confirm_args()
        sentence = args.pop("sentence")
        self.state.confirm_metric(pid, args, sentence, self.OWNER)
        return pid, r, self.state.register(pid, registration(r["sealed_id"]))["hash"]


@skip_unless_linux
class VerifyRootDockerTest(CliProject, unittest.TestCase):
    def setUp(self):
        self.start()

    def public(self):
        return signing.load_private_key(self.kp.state / "keeper.key").public_key()

    def patched_hello(self, **fields):
        real = KeeperClient.call

        def call(client, op, project, args):
            out = real(client, op, project, args)
            if op != "hello":
                return out
            out = {**out, "same_user": False, **fields}
            if fields.get("owner_has_root_docker") is None:
                out.pop("owner_has_root_docker")
            return out
        return mock.patch.object(KeeperClient, "call", call)

    def test_verify_exit_1_on_root_docker(self):
        proj = self.create()
        self.to_report(proj)
        root, sock = str(proj.root), str(self.kp.socket)
        with self.patched_hello(owner_has_root_docker=True, root_docker_reasons=["uid 1 is in gid 998"]):
            code, doc = run(["verify", root, "--keeper", sock])
        self.assertEqual((code, doc["ok"], doc["data"]["root_docker"], doc["warnings"]),
                         (1, False, True, [ROOT_DOCKER_KEEPER]), doc)
        with self.patched_hello(owner_has_root_docker=None):              # a hello without the field
            code, doc = run(["verify", root, "--keeper", sock])
        self.assertEqual((code, doc["warnings"]), (1, [ROOT_DOCKER_KEEPER]), doc)
        with self.patched_hello(owner_has_root_docker=False):
            code, doc = run(["verify", root, "--keeper", sock])
        self.assertEqual((code, doc["data"]["root_docker"], doc["warnings"]), (0, False, []), doc)

    def resign_report(self, proj, guarantee: str) -> None:
        key = signing.load_private_key(self.kp.state / "keeper.key")
        rel = proj.root / "stages" / "report"
        doc = json.loads((rel / "report.json").read_text())
        doc["sealed_guarantee"] = guarantee
        doc["known_limits"][0]["text"] = guarantee
        env = signing.sign(key, "report", doc)
        lines = [json.loads(x) for x in (proj.root / RECEIPTS).read_bytes().splitlines() if x]
        lines = [env if e["kind"] == "report" else e for e in lines]
        (proj.root / RECEIPTS).write_bytes(b"".join(json.dumps(e).encode() + b"\n" for e in lines))
        (rel / "report.json").write_text(json.dumps(doc))
        (rel / "report.md").write_text(render_markdown(doc), encoding="utf-8")

    def test_verify_exit_1_on_downgraded_report(self):
        proj = self.create()
        self.to_report(proj)
        fp = signing.fingerprint(self.public())
        code, doc = run(["verify", str(proj.root), "--fingerprint", fp])
        self.assertEqual(code, 0, doc)
        self.resign_report(proj, texts.KEEPER_ROOT_DOCKER_GUARANTEE)
        code, doc = run(["verify", str(proj.root), "--fingerprint", fp])
        self.assertEqual((code, doc["ok"], doc["problems"]), (1, False, []), doc)
        self.assertIn(ROOT_DOCKER_REPORT, doc["warnings"])
        code, doc = run(["verify", str(proj.root)])                       # no anchor: both warnings
        self.assertIn(ROOT_DOCKER_REPORT, doc["warnings"])
        self.resign_report(proj, texts.KEEPER_GUARANTEE_V1)               # preflight finding 18
        code, doc = run(["verify", str(proj.root), "--fingerprint", fp])
        self.assertEqual((code, doc["problems"]), (1, []), doc)
        self.assertIn(UNCHECKED_GUARANTEE_REPORT, doc["warnings"])


if __name__ == "__main__":
    unittest.main()
