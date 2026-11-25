"""AN-27, AN-28, AN-29: the keeper mode is an observed fact (tests/README.md section 13).

`modes.observe` takes values and does no I/O; the policy values, mode names, texts and limits are trust constants that
are pinned here once. The client side (`attested_listener_uid`, `effective_client_mode`) is tested on real sockets and
on injected answers. Nothing here names a user, a uid or a path of the host: uids are derived from the running process.
"""

from __future__ import annotations

import os
import socket
import stat
import tempfile
import threading
import types
import unittest
from pathlib import Path

from _keeper import skip_unless_linux

from gatedtrain.ainative.keeper import modes, protocol, texts
from gatedtrain.errors import Refused

KEEPER = 5000          # arbitrary distinct numbers: observe takes values, it reads no host


def st(uid: int = KEEPER, perm: int = 0o700) -> types.SimpleNamespace:
    return types.SimpleNamespace(st_uid=uid, st_mode=stat.S_IFDIR | perm)


class ConstantsTest(unittest.TestCase):
    """The trust constants have one name and one value."""

    def test_names_and_order(self):
        self.assertEqual((modes.SEPARATE_USER, modes.SAME_USER), ("separate_user", "same_user"))
        self.assertEqual(modes.MODES, ("same_user", "separate_user"))          # weakest first
        self.assertEqual(modes.REQUIREMENTS, ("any", "separate_user"))
        self.assertEqual(modes.OTHERS_AND_GROUP_BITS, 0o077)
        self.assertEqual((modes.RULE_OBSERVE, modes.RULE_POLICY, modes.RULE_ATTESTATION), ("AN-27", "AN-28", "AN-29"))

    def test_report_versions_and_accept_values(self):
        from gatedtrain.ainative import keeper_trust, trust_stages, verify_mode
        from gatedtrain.ainative.keeper import state

        self.assertEqual(trust_stages.KEEPER_REPORT_VERSIONS, (2, 3))      # 3: the mode; 2 still verifies
        self.assertEqual(keeper_trust.KEEPER_RECORD_VERSIONS, (2, 3))
        self.assertEqual(state.REPORT_SCHEMA_VERSION, 3)
        self.assertEqual(state.SCHEMA_MODE_SEEN, "gatedtrain.keeper.mode_seen")
        self.assertEqual(verify_mode.ACCEPT_MODES, ("same-user",))         # the only weaker mode a reader may accept

    def test_rank_and_weakest(self):
        self.assertLess(modes.rank(modes.SAME_USER), modes.rank(modes.SEPARATE_USER))
        self.assertEqual(modes.weakest("separate_user", "same_user", "separate_user"), "same_user")
        self.assertEqual(modes.weakest("separate_user"), "separate_user")
        for bad in ((), ("nope",), ("same_user", "")):
            with self.subTest(bad), self.assertRaises(ValueError):
                modes.weakest(*bad)
        with self.assertRaises(ValueError):
            modes.rank("separate-user")                                   # prose spelling is not a mode value

    def test_satisfies(self):
        for mode in modes.MODES:
            self.assertTrue(modes.satisfies(mode, "any"))
        self.assertTrue(modes.satisfies("separate_user", "separate_user"))
        self.assertFalse(modes.satisfies("same_user", "separate_user"))
        with self.assertRaises(ValueError):
            modes.satisfies("same_user", "strict")

    def test_there_is_no_back_door(self):
        """The mode is observed only: the module reads no environment, argument or file (it imports neither `os` nor
        `sys` nor `pathlib`, and `observe` takes values)."""
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(modes))
        imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        self.assertFalse(imported & {"os", "sys", "pathlib", "json", "subprocess"}, imported)
        self.assertEqual(list(inspect.signature(modes.observe).parameters),
                         ["keeper_euid", "owner_uid", "state_dir_stat", "peer_uid"])

    def test_texts(self):
        self.assertEqual(texts.KEEPER_OWNER_RUN_GUARANTEE, (
            "the keeper ran under the same OS user as the owner (same-user mode). It followed the procedure: the "
            "registration was signed before training, every sealed attempt spent its alpha before it was scored, and "
            "a batch generator's model ran in a container without network access that was given the inputs only, "
            "never the labels (a per-item generator runs inside the keeper process, without that confinement). It "
            "does not protect the sealed items, the labels, the alpha ledger or its signing key from anything "
            "running as that user, such as an agent with a shell, so these records show that the procedure was "
            "followed, not that the sealed data stayed unseen. Run the keeper under a separate OS user (`gatedtrain "
            "setup keeper --mode separate-user`, one sudo) for that guarantee."))
        self.assertIn("nothing is sealed", texts.KEEPER_SAME_USER_GUARANTEE)         # the legacy text is untouched
        self.assertEqual(set(texts.MODE_LINE), set(modes.MODES))
        self.assertTrue(all(line.startswith("keeper mode: ") for line in texts.MODE_LINE.values()))
        self.assertIn("NOT sealed", texts.MODE_LINE["same_user"])
        self.assertNotIn("NOT", texts.MODE_LINE["separate_user"])
        self.assertIn("same-user", texts.HEADLINE_SUFFIX_SAME_USER)

    def test_sealed_guarantee_follows_the_mode(self):
        g = texts.sealed_guarantee
        self.assertEqual(g(1, 2), texts.KEEPER_GUARANTEE)                       # no mode: the uids decide, as before
        self.assertEqual(g(1, 1), texts.KEEPER_OWNER_RUN_GUARANTEE)
        self.assertEqual(g(1, 2, mode="same_user"), texts.KEEPER_OWNER_RUN_GUARANTEE)    # the mode decides
        self.assertEqual(g(1, 1, mode="separate_user"), texts.KEEPER_GUARANTEE)
        self.assertEqual(g(1, 2, root_docker=True, mode="separate_user"), texts.KEEPER_ROOT_DOCKER_GUARANTEE)
        self.assertEqual(g(1, 2, root_docker=True, mode="same_user"), texts.KEEPER_OWNER_RUN_GUARANTEE)

    def test_mode_of_text(self):
        for text, mode in ((texts.KEEPER_SAME_USER_GUARANTEE, "same_user"),
                           (texts.KEEPER_OWNER_RUN_GUARANTEE, "same_user"),
                           (texts.KEEPER_GUARANTEE, "separate_user"),
                           (texts.KEEPER_ROOT_DOCKER_GUARANTEE, "separate_user"),
                           (texts.KEEPER_GUARANTEE_V1, "separate_user"), ("another text", None), (None, None)):
            self.assertEqual(modes.mode_of_text(text), mode)


class ObserveTest(unittest.TestCase):
    def test_separate_user_only_when_every_condition_holds(self):
        out = modes.observe(KEEPER, KEEPER + 1, st())
        self.assertEqual(out["mode"], "separate_user")
        self.assertEqual(out["facts"], {"keeper_euid": KEEPER, "owner_uid": KEEPER + 1, "peer_uid": None,
                                        "state_dir_uid": KEEPER, "state_dir_mode": "0700", "reasons": []})

    def test_each_failed_condition_gives_same_user(self):
        cases = {
            "owner is the keeper's uid": (KEEPER, KEEPER, st(), None),
            "state dir of another uid": (KEEPER, KEEPER + 1, st(uid=KEEPER + 2), None),
            "state dir open to group": (KEEPER, KEEPER + 1, st(perm=0o770), None),
            "state dir open to others": (KEEPER, KEEPER + 1, st(perm=0o705), None),
            "state dir readable by group": (KEEPER, KEEPER + 1, st(perm=0o740), None),
            "peer is the keeper's uid": (KEEPER, KEEPER + 1, st(), KEEPER),
        }
        for name, (k, o, s, peer) in cases.items():
            with self.subTest(name):
                out = modes.observe(k, o, s, peer)
                self.assertEqual(out["mode"], "same_user")
                self.assertEqual(len(out["facts"]["reasons"]), 1, out["facts"]["reasons"])

    def test_rogue_owner_uid_is_same_user_for_its_peer(self):
        """A keeper the agent started with `--owner-uid <any other number>` is reached as the keeper's own uid."""
        agent = KEEPER
        out = modes.observe(keeper_euid=agent, owner_uid=agent + 12345, state_dir_stat=st(uid=agent), peer_uid=agent)
        self.assertEqual(out["mode"], "same_user")
        self.assertIn("peer", out["facts"]["reasons"][0])

    def test_a_peer_of_another_uid_keeps_separate_user(self):
        self.assertEqual(modes.observe(KEEPER, KEEPER + 1, st(), KEEPER + 1)["mode"], "separate_user")

    def test_all_reasons_are_listed(self):
        out = modes.observe(KEEPER, KEEPER, st(uid=KEEPER + 1, perm=0o777), KEEPER)
        self.assertEqual(len(out["facts"]["reasons"]), 4)
        self.assertEqual(out["facts"]["state_dir_mode"], "0777")

    def test_the_facts_are_plain_values(self):
        import json

        json.dumps(modes.observe(KEEPER, KEEPER + 1, st(), KEEPER + 1))


class ClientModeTest(unittest.TestCase):
    OWN = 7000

    def test_separate_only_when_claim_and_kernel_agree(self):
        mode, note = modes.effective_client_mode("separate_user", self.OWN + 1, self.OWN)
        self.assertEqual(mode, "separate_user")
        self.assertIn("kernel", note)

    def test_contradiction_is_refused(self):
        with self.assertRaises(Refused) as cm:
            modes.effective_client_mode("separate_user", self.OWN, self.OWN)
        p = cm.exception.problems[0]
        self.assertEqual(p.rule, "AN-29")
        self.assertIn("process of your own uid", p.fix)

    def test_unavailable_attestation_never_upgrades(self):
        mode, note = modes.effective_client_mode("separate_user", None, self.OWN)
        self.assertEqual(mode, "same_user")
        self.assertIn("kernel attestation unavailable", note)

    def test_a_same_user_claim_is_never_upgraded(self):
        for uid in (None, self.OWN, self.OWN + 1):
            self.assertEqual(modes.effective_client_mode("same_user", uid, self.OWN)[0], "same_user")

    def test_a_claim_that_is_not_a_mode_counts_as_same_user(self):
        self.assertEqual(modes.effective_client_mode("sealed", self.OWN + 1, self.OWN)[0], "same_user")

    def test_claimed_mode_of_a_hello(self):
        self.assertEqual(modes.claimed_mode({"keeper_mode": "separate_user"}), "separate_user")
        self.assertEqual(modes.claimed_mode({"keeper_mode": "same_user", "same_user": False}), "same_user")
        self.assertEqual(modes.claimed_mode({"same_user": False}), "separate_user")      # a keeper before AN-27
        self.assertEqual(modes.claimed_mode({"same_user": True}), "same_user")
        self.assertEqual(modes.claimed_mode({}), "same_user")
        self.assertEqual(modes.claimed_mode({"keeper_mode": "banana"}), "same_user")
        self.assertEqual(modes.claimed_mode(None), "same_user")


@skip_unless_linux
class AttestedListenerTest(unittest.TestCase):
    def test_listener_in_this_process_has_our_uid(self):
        with tempfile.TemporaryDirectory(prefix="gk-") as tmp:
            path = Path(tmp) / "s"
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as srv:
                srv.bind(str(path))
                srv.listen(1)
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as cli:
                    cli.connect(str(path))
                    self.assertEqual(protocol.attested_listener_uid(cli), os.geteuid())

    def test_a_socket_that_is_not_connected_gives_none_or_ours(self):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            self.assertIn(protocol.attested_listener_uid(s), (None, os.geteuid()))

    def test_an_error_gives_none(self):
        class Broken:
            def getsockopt(self, *a):
                raise OSError("boom")
        self.assertIsNone(protocol.attested_listener_uid(Broken()))

    def test_uid_minus_one_gives_none(self):
        import struct

        class Anonymous:
            def getsockopt(self, *a):
                return struct.pack("3i", 0, -1, -1)
        self.assertIsNone(protocol.attested_listener_uid(Anonymous()))

    def test_no_so_peercred_gives_none(self):
        from unittest import mock

        with mock.patch.object(protocol.socket, "SO_PEERCRED", None, create=True):
            self.assertIsNone(protocol.attested_listener_uid(object()))

    def test_a_listener_of_another_uid(self):
        """Any unix socket on this host that another uid listens on; skipped when there is none (no host path is
        assumed: the sockets are read from the kernel's table)."""
        try:
            lines = Path("/proc/net/unix").read_text().splitlines()[1:]
        except OSError:
            self.skipTest("the kernel's unix socket table is not readable")
        paths = [ln.split()[-1] for ln in lines if len(ln.split()) >= 8 and ln.split()[-1].startswith("/")]
        for path in dict.fromkeys(paths):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as cli:
                cli.settimeout(1)
                try:
                    cli.connect(path)
                except OSError:
                    continue
                uid = protocol.attested_listener_uid(cli)
                if uid is not None and uid != os.geteuid():
                    return
        self.skipTest("no unix socket served by another uid is reachable on this host")


class ClientRecordsLastAttestationTest(unittest.TestCase):
    def test_call_stores_the_attested_uid(self):
        from gatedtrain.ainative.keeper.protocol import KeeperClient

        with tempfile.TemporaryDirectory(prefix="gk-") as tmp:
            path = Path(tmp) / "s"
            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            srv.bind(str(path))
            srv.listen(1)

            def answer():
                conn, _ = srv.accept()
                with conn:
                    conn.recv(65536)
                    conn.sendall(b'{"ok":true,"result":{},"problems":[]}\n')
            t = threading.Thread(target=answer, daemon=True)
            t.start()
            client = KeeperClient(path, timeout=5)
            self.assertIsNone(client.last_attested_uid)
            client.call("hello", None, {})
            t.join(5)
            srv.close()
            self.assertEqual(client.last_attested_uid, os.geteuid())


if __name__ == "__main__":
    unittest.main()
