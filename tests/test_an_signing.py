"""AN-14: Ed25519 signing of trust records (tests/README.md section 13).

Signed bytes are the RFC 8785 canonical JSON of `{"kind", "key_fingerprint", "record"}`; `verify` refuses every
tamper with rule AN-14; key files are created exclusively with mode 0600 and loaded only when no group or other bit
is set.
"""

from __future__ import annotations

import base64
import copy
import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gatedtrain.ainative import signing
from gatedtrain.canonical import canonical_json, sha256_bytes
from gatedtrain.errors import Refused


def _flip_signature_byte(env: dict) -> dict:
    raw = bytearray(base64.b64decode(env["signature"]))
    raw[0] ^= 0x01
    env["signature"] = base64.b64encode(bytes(raw)).decode("ascii")
    return env


def _set(key: str, value: object):
    def mutate(env: dict) -> dict:
        env[key] = value
        return env
    return mutate


def _record_value(env: dict) -> dict:
    env["record"]["a"] = 2
    return env


def _record_extra(env: dict) -> dict:
    env["record"]["extra"] = True
    return env


def _extra_key(env: dict) -> dict:
    env["extra"] = 1
    return env


def _drop_signature(env: dict) -> dict:
    del env["signature"]
    return env


def _non_canonical_base64(env: dict) -> dict:
    # The last data character of a 64-byte signature carries 4 unused bits; b64decode ignores them.
    sig = env["signature"]
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    env["signature"] = sig[:-3] + alphabet[alphabet.index(sig[-3]) ^ 1] + "=="
    assert base64.b64decode(env["signature"], validate=True) == base64.b64decode(sig)
    return env


TAMPERS = {
    "signature non-canonical base64": _non_canonical_base64,
    "record value": _record_value,
    "record extra field": _record_extra,
    "kind": _set("kind", "report"),
    "unknown kind": _set("kind", "nonsense"),
    "fingerprint": _set("key_fingerprint", "sha256:" + "0" * 64),
    "signature byte": _flip_signature_byte,
    "signature not base64": _set("signature", "not base64 !!"),
    "signature not text": _set("signature", 5),
    "extra envelope key": _extra_key,
    "missing signature": _drop_signature,
    "schema": _set("schema", "gatedtrain.ainative.other"),
    "schema version": _set("schema_version", 2),
    "record not a dict": _set("record", [1]),
}


class TestSigning(unittest.TestCase):
    def setUp(self) -> None:
        self.key = Ed25519PrivateKey.generate()
        self.pub = self.key.public_key()
        self.env = signing.sign(self.key, "claim", {"a": 1})

    def test_envelope_shape(self) -> None:
        self.assertEqual(
            self.env,
            {"schema": "gatedtrain.ainative.signed", "schema_version": 1, "kind": "claim",
             "key_fingerprint": signing.fingerprint(self.pub), "record": {"a": 1},
             "signature": self.env["signature"]},
        )
        self.assertEqual(len(base64.b64decode(self.env["signature"], validate=True)), 64)
        self.assertEqual(signing.SIGNED_SCHEMA, "gatedtrain.ainative.signed")
        self.assertEqual(
            signing.KINDS,
            ("registration", "metric_confirmation", "claim", "ledger_entry", "gate_record", "gate_summary",
             "report", "environment"))

    def test_round_trip(self) -> None:
        self.assertEqual(signing.verify(self.env, self.pub, kind="claim"), {"a": 1})
        self.assertEqual(signing.verify(self.env, self.pub), {"a": 1})

    def test_signed_bytes_are_kind_fingerprint_record(self) -> None:
        body = canonical_json({"kind": "claim", "key_fingerprint": signing.fingerprint(self.pub),
                               "record": {"a": 1}})
        self.pub.verify(base64.b64decode(self.env["signature"]), body)

    def test_every_tamper_is_refused(self) -> None:
        for name, mutate in TAMPERS.items():
            with self.subTest(tamper=name):
                with self.assertRaises(Refused) as cm:
                    signing.verify(mutate(copy.deepcopy(self.env)), self.pub)
                self.assertEqual(cm.exception.problems[0].rule, "AN-14")

    def test_kind_mismatch_refused(self) -> None:
        with self.assertRaises(Refused) as cm:
            signing.verify(self.env, self.pub, kind="report")
        self.assertEqual(cm.exception.problems[0].rule, "AN-14")

    def test_envelope_not_a_dict_refused(self) -> None:
        for bad in (None, [], "x", 3):
            with self.subTest(bad=bad):
                with self.assertRaises(Refused) as cm:
                    signing.verify(bad, self.pub)
                self.assertEqual(cm.exception.problems[0].rule, "AN-14")

    def test_other_key_refused(self) -> None:
        other = Ed25519PrivateKey.generate().public_key()
        with self.assertRaises(Refused) as cm:
            signing.verify(self.env, other)
        self.assertEqual(cm.exception.problems[0].rule, "AN-14")

    def test_bad_signature_with_matching_fingerprint_refused(self) -> None:
        other = Ed25519PrivateKey.generate()
        forged = signing.sign(other, "claim", {"a": 1})
        forged["key_fingerprint"] = signing.fingerprint(self.pub)
        with self.assertRaises(Refused) as cm:
            signing.verify(forged, self.pub)
        self.assertEqual(cm.exception.problems[0].rule, "AN-14")

    def test_unknown_kind_not_signable(self) -> None:
        with self.assertRaises(Refused) as cm:
            signing.sign(self.key, "nonsense", {"a": 1})
        self.assertEqual(cm.exception.problems[0].rule, "AN-14")

    def test_non_canonical_record_refused(self) -> None:
        with self.assertRaises(Refused) as cm:
            signing.sign(self.key, "claim", {"a": float("nan")})
        self.assertEqual(cm.exception.problems[0].rule, "AN-14")
        # A record that is not canonical JSON on the verify side (here a NaN smuggled into a signed envelope).
        env = copy.deepcopy(self.env)
        env["record"] = {"a": float("nan")}
        with self.assertRaises(Refused) as cm:
            signing.verify(env, self.pub)
        self.assertEqual(cm.exception.problems[0].rule, "AN-14")

    def test_signed_bytes_are_rfc8785(self) -> None:
        # Key order and float form do not change the signature (Ed25519 is deterministic).
        self.assertEqual(signing.sign(self.key, "claim", {"b": 1.0, "a": 2})["signature"],
                         signing.sign(self.key, "claim", {"a": 2, "b": 1})["signature"])

    def test_fingerprint_shape(self) -> None:
        self.assertRegex(signing.fingerprint(self.pub), r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(signing.fingerprint(self.key), signing.fingerprint(self.pub))

    def test_public_pem_round_trip(self) -> None:
        pem = signing.public_pem(self.key)
        self.assertEqual(pem, signing.public_pem(self.pub))
        self.assertTrue(pem.startswith(b"-----BEGIN PUBLIC KEY-----"))
        loaded = signing.load_public_pem(pem)
        self.assertEqual(signing.fingerprint(loaded), signing.fingerprint(self.pub))
        self.assertEqual(signing.verify(self.env, loaded), {"a": 1})

    def test_load_public_pem_refuses_garbage(self) -> None:
        with self.assertRaises(Refused) as cm:
            signing.load_public_pem(b"not a pem")
        self.assertEqual(cm.exception.problems[0].rule, "AN-14")

    def test_envelope_sha256(self) -> None:
        self.assertEqual(signing.envelope_sha256(self.env), sha256_bytes(canonical_json(self.env)))
        other = signing.sign(self.key, "claim", {"a": 2})
        self.assertNotEqual(signing.envelope_sha256(self.env), signing.envelope_sha256(other))


class TestKeyFiles(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="an-signing-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.path = self.tmp / "keeper.key"

    def test_key_file_mode_and_exclusive(self) -> None:
        signing.generate_key(self.path)
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        self.assertTrue(self.path.read_bytes().startswith(b"-----BEGIN PRIVATE KEY-----"))
        key = signing.load_private_key(self.path)
        self.assertEqual(signing.verify(signing.sign(key, "report", {"x": 1}), key.public_key()), {"x": 1})
        before = self.path.read_bytes()
        with self.assertRaises(Refused) as cm:
            signing.generate_key(self.path)
        self.assertEqual(cm.exception.problems[0].rule, "AN-14")
        self.assertEqual(self.path.read_bytes(), before)
        for mode in (0o640, 0o604, 0o644, 0o660, 0o666):
            with self.subTest(mode=oct(mode)):
                os.chmod(self.path, mode)
                with self.assertRaises(Refused) as cm:
                    signing.load_private_key(self.path)
                self.assertEqual(cm.exception.problems[0].rule, "AN-14")
        os.chmod(self.path, 0o400)
        signing.load_private_key(self.path)

    def test_generate_refuses_a_symbolic_link(self) -> None:
        target = self.tmp / "elsewhere"
        self.path.symlink_to(target)
        with self.assertRaises(Refused) as cm:
            signing.generate_key(self.path)
        self.assertEqual(cm.exception.problems[0].rule, "AN-14")
        self.assertFalse(target.exists())

    def test_load_refuses_missing_and_non_ed25519(self) -> None:
        with self.assertRaises(Refused) as cm:
            signing.load_private_key(self.tmp / "absent.key")
        self.assertEqual(cm.exception.problems[0].rule, "AN-14")
        self.path.write_bytes(b"not a key")
        os.chmod(self.path, 0o600)
        with self.assertRaises(Refused) as cm:
            signing.load_private_key(self.path)
        self.assertEqual(cm.exception.problems[0].rule, "AN-14")


if __name__ == "__main__":
    unittest.main()
