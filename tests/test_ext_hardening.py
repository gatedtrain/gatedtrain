"""Review fixes of slice 1 (tests/README.md section 11.18, EXT-39 to EXT-47).

Sources: the slice-1 review (findings H1, M1 to M3, L1 to L7) of `ext-launch`, `ext-collect`, `ext-merge-eval`
and `ext-from-nel`; MULTIMODAL-SPEC.md MM-1 to MM-6, MM-30, MM-31, MM-34 to MM-36; design 05 section 11.3
(R13-10: coarsest precision on the path). The Docker reference grammar is distribution/reference's
(`reference.go`: a name component is `[a-z0-9]+(?:(?:[._]|__|[-]*)[a-z0-9]+)*`, a domain starts with an
alphanumeric), so a reference never starts with `-`. The safetensors header cap is the reference
implementation's `MAX_HEADER_SIZE = 100_000_000` (safetensors `lib.rs`). gpu-step.sh's log format and its
hardcoded `WORK=` line are those of `<HOST_WORK>/bin/gpu-step.sh` (read 2026-11-11).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import socket
import struct
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from _ext import (EXT, LABEL, SEEDS, ExtScenario, baseline_score, build_arm, candidate_score, gpu_step_script,
                  guard_log, jcs, launch_label, merge, model_hash, write_canonical)
from _support import (EXIT_OK, EXIT_REFUSED, describe, file_sha256, load_report, parse_errors, parse_run_dir,
                      read_jsonl)
from test_ext_from_nel import convert, nel_runs, scorer_baseline, scorer_candidate
from test_ext_launch import GOLDEN_REQUEST, golden_request, launch

DIGEST = "@sha256:" + "ab" * 32


def no_run_dir(runs_root: Path) -> bool:
    return not (runs_root.exists() and any(p.is_dir() for p in runs_root.iterdir()))


class LaunchCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-hard-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.k = 0

    def sub(self) -> Path:
        self.k += 1
        d = self.tmp / f"case-{self.k}"
        d.mkdir()
        return d

    def launch_refused(self, req: dict, field_suffix: str):
        proc, out = launch(self.sub(), req)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        errs = parse_errors(proc.stderr)
        hits = [e for e in errs if e["rule"] == "EXT-LAUNCH" and e["field"].endswith(field_suffix)]
        self.assertTrue(hits, f"expected EXT-LAUNCH ...{field_suffix}; got {errs}")
        self.assertEqual(proc.stdout, "", "a refused launch prints no command")
        self.assertFalse(out.exists(), "a refused launch writes no record")


class CollectCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-hard-c-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.sc = ExtScenario(cls.tmp / "sc")
        cls.n = 0

    def runs(self) -> Path:
        type(self).n += 1
        return self.tmp / f"runs-{type(self).n}"

    def path(self, stem: str) -> Path:
        type(self).n += 1
        return self.tmp / f"{stem}-{type(self).n}"

    def record(self, **overrides) -> Path:
        return write_canonical(self.path("launch").with_suffix(".json"), self.sc.launch_record(**overrides))

    def refused(self, rule: str, field_prefix: str | None = None, *, reason: str | None = None, **collect_kw):
        """The refusal's hits; with `reason`, one of them names it (review of PR 2, T6)."""
        runs = self.runs()
        collect_kw.setdefault("sources_root", self.tmp)  # MM-5: merged arms' sources are built under the test root
        proc = self.sc.collect(runs, **collect_kw)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        errs = parse_errors(proc.stderr)
        hits = [e for e in errs if e["rule"] == rule and (field_prefix is None or e["field"].startswith(field_prefix))]
        self.assertTrue(hits, f"expected error[{rule}] {field_prefix}...; got {errs}\n{describe(proc)}")
        self.assertNotIn("Traceback", proc.stderr)
        self.assertTrue(no_run_dir(runs), "a refusal creates no run directory (MM-10)")
        if reason is not None:
            self.assertTrue(any(reason in h["message"] for h in hits), hits)
        return hits

    def passes(self, **collect_kw) -> dict:
        collect_kw.setdefault("sources_root", self.tmp)
        proc = self.sc.collect(self.runs(), **collect_kw)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        return load_report(parse_run_dir(proc.stdout))


# ---------------------------------------------------------------------------
# EXT-39 (H1): nothing user-controlled is parsed by docker as an option
# ---------------------------------------------------------------------------

INJECTION_CASES = [
    # (name, mutate, field suffix)
    ("image starts with an option", lambda r: r["runtime"].update(image="--volume=/:/host" + DIGEST), "runtime.image"),
    ("image is a short option", lambda r: r["runtime"].update(image="-v/:/host" + DIGEST), "runtime.image"),
    ("image privileged", lambda r: r["runtime"].update(image="--privileged" + DIGEST), "runtime.image"),
    ("image with upper-case path", lambda r: r["runtime"].update(image="ghcr.io/Example/trl" + DIGEST),
     "runtime.image"),
    ("image with a space", lambda r: r["runtime"].update(image="trl --privileged" + DIGEST), "runtime.image"),
    ("mount host with a colon", lambda r: r["mounts"][0].update(host="/data/x:/etc"), "mounts[0].host"),
    ("mount host with a comma", lambda r: r["mounts"][0].update(host="/data/x,readonly=false"), "mounts[0].host"),
    ("mount container with a colon", lambda r: r["mounts"][0].update(container="/models/base-model:rw"),
     "mounts[0].container"),
    ("mount container with a comma", lambda r: r["mounts"][0].update(container="/m,z"), "mounts[0].container"),
    ("env name is an option", lambda r: r["env"].update({"--privileged": "1"}), "env.--privileged"),
    ("env name with =", lambda r: r["env"].update({"A=B": "1"}), "env.A=B"),
    ("env name with a dot", lambda r: r["env"].update({"A.B": "1"}), "env.A.B"),
    ("env name starts with a digit", lambda r: r["env"].update({"1A": "1"}), "env.1A"),
    ("label is an option", lambda r: r.update(label="--privileged"), "label"),
    ("network is an option", lambda r: r.update(network="--privileged", network_reason="x"), "network"),
    ("network with a space", lambda r: r.update(network="none --privileged", network_reason="x"), "network"),
    ("tmpfs with options", lambda r: r.update(tmpfs=["/tmp:exec"]), "tmpfs"),
    ("tmpfs with a comma", lambda r: r.update(tmpfs=["/tmp,exec"]), "tmpfs"),
]


class TestDockerOptionInjection(LaunchCase):

    def test_launch_refuses(self):
        for name, mutate, suffix in INJECTION_CASES:
            with self.subTest(case=name):
                req = copy.deepcopy(golden_request())
                mutate(req)
                self.launch_refused(req, suffix)

    def test_docker_reference_grammar_accepted(self):
        for image in ("sha256:" + "ab" * 32, "ghcr.io/example/trl" + DIGEST, "localhost:5000/team/trl" + DIGEST,
                      "nvcr.io/nvidia/pytorch:25.01-py3" + DIGEST, "synthetic" + DIGEST, "a/b__c/d-e.f" + DIGEST):
            with self.subTest(image=image):
                req = golden_request()
                req["runtime"]["image"] = image
                proc, out = launch(self.sub(), req)
                self.assertEqual(proc.returncode, EXIT_OK, describe(proc))

    def test_only_fixed_options_before_the_image(self):
        """Between the label and the image every token is a fixed option or the value right after a fixed
        option that takes one; no value starts with `-`."""
        import shlex
        proc, out = launch(self.sub(), GOLDEN_REQUEST)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        tok = shlex.split(proc.stdout)
        script = next(i for i, t in enumerate(tok) if t.endswith("/bin/gpu-step.sh"))
        image = tok.index(golden_request()["runtime"]["image"])
        with_value = {"--gpus", "--network", "--user", "--cap-drop", "--security-opt", "-e", "-v", "--tmpfs"}
        flags = {"--rm", "--runtime=nvidia", "--shm-size=8g"}        # fixed options, no value (EXT-24)
        i = script + 2
        while i < image:
            self.assertIn(tok[i], with_value | flags, tok[i])
            if tok[i] in with_value:
                self.assertFalse(tok[i + 1].startswith("-"), tok[i + 1])
                i += 2
            else:
                i += 1
        self.assertEqual(i, image)


class TestDockerOptionInjectionCollect(CollectCase):

    def test_collect_refuses(self):
        sc = self.sc
        mounts = sc.launch_record()["mounts"]
        cases = {
            "image starts with an option": dict(runtime={"kind": "container", "image": "--volume=/:/host" + DIGEST,
                                                         "venv_lock_sha256": None}),
            "mount host with a colon": dict(mounts=mounts + [{"host": "/data/x:/etc", "container": "/x", "mode": "ro"}]),
            "mount container with a comma": dict(mounts=mounts + [{"host": str(sc.train), "container": "/x,z",
                                                                   "mode": "ro"}]),
            "env name is an option": dict(env_names=["--privileged", "HF_HOME", "HF_HUB_OFFLINE"]),
            "network is an option": dict(network="--privileged", network_reason="x"),
            "tmpfs with options": dict(tmpfs=["/tmp:exec"]),
        }
        for name, kw in cases.items():
            with self.subTest(case=name):
                self.refused("EXT-LAUNCH", "--launch-record", launch=self.record(**kw))
        with self.subTest(case="label is an option"):
            rec = self.sc.launch_record(hashed_label=False, label="--privileged")
            path = write_canonical(self.path("launch").with_suffix(".json"), rec)
            self.refused("EXT-LAUNCH", "--launch-record.label", launch=path)


# ---------------------------------------------------------------------------
# EXT-40 (M2): no Docker socket, no ancestor of one, no socket directory
# ---------------------------------------------------------------------------

class TestSocketMounts(LaunchCase):

    def socket_cases(self) -> dict[str, str]:
        link = self.tmp / "run-link"
        link.symlink_to("/var/run")
        sock_dir = Path(tempfile.mkdtemp(prefix="s", dir="/tmp"))     # AF_UNIX paths are short
        self.addCleanup(shutil.rmtree, sock_dir, True)
        sock = sock_dir / "fixture.sock"
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(str(sock))
        self.addCleanup(s.close)
        plain_socket = sock_dir / "listener"                            # a socket without the .sock suffix
        s2 = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s2.bind(str(plain_socket))
        self.addCleanup(s2.close)
        return {"/": "/", "/var": "/var", "/run": "/run", "/var/run": "/var/run", "/var/run/": "/var/run/",
                "docker socket": "/var/run/docker.sock", "run docker socket": "/run/docker.sock",
                "gatedtrain docker socket": "/run/docker-gatedtrain.sock", "containerd socket": "/run/containerd/containerd.sock",
                "rootless docker socket": "/run/user/1000/docker.sock", "under /run": "/run/user/1000",
                "symlink to /var/run": str(link), "a .sock file": str(sock), "a socket file": str(plain_socket),
                "dot-dot to /run": "/data/../run"}

    def test_launch_refuses(self):
        for name, host in self.socket_cases().items():
            with self.subTest(case=name):
                req = golden_request()
                req["mounts"].insert(0, {"host": host, "container": "/sock", "mode": "ro"})
                self.launch_refused(req, "mounts[0].host")


class TestSocketMountsCollect(CollectCase):

    def test_collect_refuses(self):
        for host in ("/var/run", "/run", "/", "/run/docker-gatedtrain.sock"):
            with self.subTest(host=host):
                mounts = self.sc.launch_record()["mounts"] + [{"host": host, "container": "/s", "mode": "ro"}]
                self.refused("EXT-LAUNCH", "--launch-record.mounts[4].host", launch=self.record(mounts=mounts))


# ---------------------------------------------------------------------------
# EXT-41 (M1, L2): every manifest and input read before the run directory exists
# ---------------------------------------------------------------------------

class TestPreflightReads(CollectCase):

    def test_symlink_in_training_data(self):
        link = self.sc.train / "notes.txt"
        link.symlink_to("/etc/hostname")
        self.addCleanup(link.unlink)
        self.refused("SPEC-REF", "artifacts.train.uri")

    def test_symlink_in_item_set_dir(self):
        link = self.sc.items_dir / "README.md"
        link.symlink_to("/etc/hostname")
        self.addCleanup(link.unlink)
        self.refused("EXT-ITEMS", "artifacts.items.uri")

    def test_item_set_not_utf8(self):
        path = self.sc.items_dir / "items.jsonl"
        original = path.read_bytes()
        self.addCleanup(path.write_bytes, original)
        path.write_bytes(b"\xff\xfe" + original)
        self.refused("EXT-ITEMS", "artifacts.items.uri")

    def test_training_shard_not_utf8(self):
        shard = self.sc.train / "shard-00009.jsonl"
        shard.write_bytes(b"\xff\xfe{}\n")
        self.addCleanup(shard.unlink)
        self.refused("EXT-CONTAMINATION", "artifacts.train.uri")

    def test_still_passes_after_cleanup(self):
        self.assertEqual(self.passes()["status"], "PASS")


class TestFromNelItemsNotUtf8(CollectCase):

    def test_refused(self):
        items = self.tmp / "bad-items"
        shutil.copytree(self.sc.items_dir, items)
        (items / "items.jsonl").write_bytes(b"\xff" + (items / "items.jsonl").read_bytes())
        runs = nel_runs(self.tmp / "nel", self.sc.rows, score_fn=scorer_baseline, model="base")
        out = self.tmp / "arm"
        proc = convert(self.sc, out, runs, checkpoint=self.sc.base, items=items)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertIn("EXT-ITEMS", [e["rule"] for e in parse_errors(proc.stderr)], proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertFalse(out.exists())


# ---------------------------------------------------------------------------
# EXT-42 (M3), EXT-58 (R14-1): a run reached directly and transitively is merged per cut at the finest
# per-path precision, every path listed
# ---------------------------------------------------------------------------

class TestImpactMerge(unittest.TestCase):
    S = "sha256:" + "5" * 64
    SNAP = "sha256:" + "a" * 64

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-impact-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.runs = self.tmp / "runs"

    def mk(self, run, *, ckpts, used_model, prec, ledger_steps=()):
        from gatedtrain.ledger import ORPHAN_SCHEMA, SCHEMA
        d = self.runs / run
        (d / "lineage").mkdir(parents=True)
        (d / "ledger").mkdir()
        pq.write_table(pa.Table.from_pylist([
            {"run_id": run, "stage_id": "rl", "checkpoint_hash": f"c-{run}-{s}", "model_hash": f"m-{run}-{s}",
             "last_step_included": s, "path": "x"} for s in ckpts]), d / "lineage" / "checkpoints.parquet")
        pq.write_table(pa.Table.from_pylist([{"run_id": run, "stage_id": "rl", "snapshot": self.SNAP,
                                              "shard": "shard-0.jsonl", "shard_sha256": self.S}]),
                       d / "lineage" / "shards.parquet")
        pq.write_table(pa.Table.from_pylist([], schema=ORPHAN_SCHEMA), d / "lineage" / "orphaned.parquet")
        rows = [{"run_id": run, "attempt_id": "a0", "step": st, "sample_id": None,
                 "data_ref": {"snapshot": self.SNAP, "shard": "shard-0.jsonl", "row": None, "mixture": None,
                              "component": None},
                 "outcome": "consumed", "precision": prec, "derivation": "observed"} for st in ledger_steps]
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), d / "ledger" / "consumption.parquet")
        (d / "lineage" / "edges.json").write_text(json.dumps({"activities": [
            {"id": f"{run}/rl", "kind": "training_stage",
             "used": [{"role": "model", "hash": "h", "model_hash": used_model},
                      {"role": "data", "hash": self.SNAP, "precision": prec}]}]}), encoding="utf-8")

    def query(self):
        from gatedtrain.impact import impact_query
        return {(r["run_id"], r["last_step_included"]): r for r in impact_query(self.S, runs_root=self.runs)}

    def test_direct_iteration_and_transitive_stage(self):
        """X consumed S at precision `stage`; Y starts from X's model and also consumed S at step 50
        (`iteration`). Y's cut 10 (before its first direct use) is returned through X at `stage`; cut 50 keeps
        its direct `iteration` answer, the finest of its two paths, and lists both (R14-1, EXT-58)."""
        self.mk("X", ckpts=[10], used_model="m-base", prec="stage")
        self.mk("Y", ckpts=[10, 50], used_model="m-X-10", prec="iteration", ledger_steps=[50])
        got = self.query()
        self.assertEqual(set(got), {("X", 10), ("Y", 10), ("Y", 50)})
        y10, y50 = got[("Y", 10)], got[("Y", 50)]
        self.assertEqual((y10["precision"], y10["answer"], y10["via_model_hash"], y10["first_step"]),
                         ("stage", "possibly consumed", "m-X-10", None))
        self.assertEqual((y50["precision"], y50["via_model_hash"], y50["first_step"]), ("iteration", None, 50))
        self.assertEqual(sorted((p["via_model_hash"] or "", p["precision"]) for p in y50["paths"]),
                         [("", "iteration"), ("m-X-10", "stage")])

    def test_direct_stage_and_transitive_iteration(self):
        """X consumed S at step 5 (`iteration`); Y starts from X's model and has S in its own `stage` edge:
        Y's cuts take `iteration`, the finer precision of the path through X, and list both paths (R14-1)."""
        self.mk("X", ckpts=[10], used_model="m-base", prec="iteration", ledger_steps=[5])
        self.mk("Y", ckpts=[10, 50], used_model="m-X-10", prec="stage")
        got = self.query()
        self.assertEqual(got[("X", 10)]["precision"], "iteration")
        for key in (("Y", 10), ("Y", 50)):
            self.assertEqual(got[key]["precision"], "iteration", key)
            self.assertEqual(got[key]["via_model_hash"], "m-X-10", key)
            self.assertEqual(sorted(p["precision"] for p in got[key]["paths"]), ["iteration", "stage"], key)

    def test_chain_of_three(self):
        """Z starts from Y's last model: reached through Y at Y's answer precision; its own direct edge is
        `iteration` too."""
        self.mk("X", ckpts=[10], used_model="m-base", prec="stage")
        self.mk("Y", ckpts=[10, 50], used_model="m-X-10", prec="iteration", ledger_steps=[50])
        self.mk("Z", ckpts=[7], used_model="m-Y-50", prec="iteration", ledger_steps=[3])
        got = self.query()
        self.assertEqual(got[("Z", 7)]["precision"], "iteration")
        self.assertEqual(len(got[("Z", 7)]["paths"]), 2)

    def test_row_answer_not_weakened_by_a_stage_path(self):
        """Review L-e: W consumed S at `stage`; X starts from W's model and consumed S itself at row level at
        step 5. X's cut keeps `row` and the answer `consumed`; the stage path through W is listed (R14-1)."""
        self.mk("W", ckpts=[10], used_model="m-base", prec="stage")
        self.mk("X", ckpts=[10], used_model="m-W-10", prec="row", ledger_steps=[5])
        got = self.query()
        x = got[("X", 10)]
        self.assertEqual((x["precision"], x["answer"], x["first_step"], x["via_model_hash"]),
                         ("row", "consumed", 5, None))
        self.assertEqual(sorted((p["via_model_hash"] or "", p["precision"]) for p in x["paths"]),
                         [("", "row"), ("m-W-10", "stage")])


# ---------------------------------------------------------------------------
# EXT-43 (L1): safetensors headers
# ---------------------------------------------------------------------------

class TestSafetensorsHeader(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-st-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def problem(self, raw: bytes | None = None, *, header: bytes | None = None, data: int = 0, n: int | None = None):
        from gatedtrain.ext.sdk import safetensors_problem
        p = self.tmp / "model.safetensors"
        if raw is None:
            raw = struct.pack("<Q", len(header) if n is None else n) + header + b"\0" * data
        p.write_bytes(raw)
        return safetensors_problem(p)

    def entry(self, offs) -> bytes:
        return json.dumps({"w": {"dtype": "F32", "shape": [2], "data_offsets": offs}}).encode()

    def test_valid(self):
        self.assertIsNone(self.problem(header=self.entry([0, 8]), data=8))

    def test_header_length_cap(self):
        """A header longer than 100 000 000 bytes is refused without reading it (sparse file)."""
        from gatedtrain.ext.sdk import safetensors_problem
        p = self.tmp / "big.safetensors"
        n = 100_000_001
        with p.open("wb") as f:
            f.write(struct.pack("<Q", n))
            f.truncate(8 + n)
        bad = safetensors_problem(p)
        self.assertIsNotNone(bad)
        self.assertIn("100000000", bad.replace("_", "").replace(",", ""))

    def test_deep_nesting(self):
        hdr = b"[" * 200000 + b"]" * 200000
        self.assertIsNotNone(self.problem(header=hdr))

    def test_bool_offsets(self):
        self.assertIsNotNone(self.problem(header=self.entry([False, True]), data=1))

    def test_reversed_offsets(self):
        self.assertIsNotNone(self.problem(header=self.entry([8, 0]), data=0))
        self.assertIsNotNone(self.problem(header=self.entry([8, 4]), data=8))

    def test_float_offsets(self):
        self.assertIsNotNone(self.problem(header=self.entry([0, 8.0]), data=8))


# ---------------------------------------------------------------------------
# EXT-44 (L3): eval_meta values compare as canonical JSON (no bool/int equality)
# ---------------------------------------------------------------------------

class TestMetaCanonicalComparison(CollectCase):

    def cand(self, name: str, **kw) -> Path:
        return self.sc.arm(name, candidate_score, root=self.tmp / "arms", **kw)

    def test_bool_score_range_refused(self):
        self.refused("EXT-ARM", "--eval-candidate", candidate=self.cand("bools", meta={"score_range": [False, True]}))

    def test_bool_in_sampler_refused(self):
        sampler = {"name": "deployment", "temperature": 0.7, "top_p": True, "max_new_tokens": 1024,
                   "enable_thinking": 0}
        self.refused("EXT-ARM", "--eval-candidate", candidate=self.cand("sampler", meta={"sampler": sampler}))

    def test_integral_float_range_accepted(self):
        self.assertEqual(self.passes(candidate=self.cand("floats", meta={"score_range": [0.0, 1.0]}))["status"],
                         "PASS")

    def test_merge_refuses_bool_difference(self):
        a = build_arm(self.tmp / "m" / "a", self.sc.items, baseline_score, model=self.sc.base_model,
                      item_set=self.sc.item_set_hash, seeds=[SEEDS[0]])
        b = build_arm(self.tmp / "m" / "b", self.sc.items, baseline_score, model=self.sc.base_model,
                      item_set=self.sc.item_set_hash, seeds=[SEEDS[1]], meta={"score_range": [False, True]})
        out = self.tmp / "m" / "out"
        proc = merge(self.sc.paths["registration"], out, [a, b], cwd=self.sc.root)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertIn("EXT-MERGE", [e["rule"] for e in parse_errors(proc.stderr)])
        self.assertFalse(out.exists())


# ---------------------------------------------------------------------------
# EXT-45 (L4, review mutant M01): output_ref stays a plain file of the arm, never a reserved file
# ---------------------------------------------------------------------------

RESERVED = ("eval_meta.json", "items.jsonl", "merge.json", "judge_batches.jsonl")


class TestOutputRef(CollectCase):

    def arm_with_ref(self, name: str, rel: str, target: Path | None = None) -> Path:
        arm = self.sc.arm(name, candidate_score, root=self.tmp / "arms")
        lines = read_jsonl(arm / "items.jsonl")
        path = target if target is not None else arm / rel
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture output", encoding="utf-8")
        lines[0]["output_ref"] = {"path": rel, "sha256": file_sha256(path)}
        (arm / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
        return arm

    def test_reserved_names_refused(self):
        for rel in RESERVED:
            with self.subTest(path=rel):
                arm = self.arm_with_ref("res_" + rel.replace("/", "_"), rel)
                self.refused("EXT-ARM", "--eval-candidate", candidate=arm, reason="names a file of the arm itself")
        with self.subTest(path="./eval_meta.json"):                  # T7: another spelling is not canonical
            arm = self.arm_with_ref("res_dot_eval_meta", "./eval_meta.json", self.tmp / "dot-target.txt")
            self.refused("EXT-ARM", "--eval-candidate", candidate=arm, reason="is not in canonical form")

    def test_file_symlink_out_of_the_arm_refused(self):
        outside = self.tmp / "outside.txt"
        outside.write_text("outside the arm", encoding="utf-8")
        arm = self.sc.arm("filelink", candidate_score, root=self.tmp / "arms")
        (arm / "link.txt").symlink_to(outside)
        lines = read_jsonl(arm / "items.jsonl")
        lines[0]["output_ref"] = {"path": "link.txt", "sha256": file_sha256(outside)}
        (arm / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
        self.refused("EXT-ARM", "--eval-candidate", candidate=arm, reason="contains a symbolic link")

    def test_directory_symlink_out_of_the_arm_refused(self):
        outside = self.tmp / "outside-dir"
        outside.mkdir(exist_ok=True)
        (outside / "out.txt").write_text("outside", encoding="utf-8")
        arm = self.sc.arm("dirlink", candidate_score, root=self.tmp / "arms")
        (arm / "sub").symlink_to(outside)
        lines = read_jsonl(arm / "items.jsonl")
        lines[0]["output_ref"] = {"path": "sub/out.txt", "sha256": file_sha256(outside / "out.txt")}
        (arm / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
        self.refused("EXT-ARM", "--eval-candidate", candidate=arm)

    def test_nested_name_accepted(self):
        arm = self.arm_with_ref("nested", "outputs/eval_meta.json")
        self.assertEqual(self.passes(candidate=arm)["status"], "PASS")

    def test_merge_refuses_reserved_output_ref(self):
        a = build_arm(self.tmp / "mr" / "a", self.sc.items, candidate_score, model=self.sc.cand_model,
                      item_set=self.sc.item_set_hash, seeds=[SEEDS[0]])
        b = build_arm(self.tmp / "mr" / "b", self.sc.items, candidate_score, model=self.sc.cand_model,
                      item_set=self.sc.item_set_hash, seeds=[SEEDS[1]])
        lines = read_jsonl(a / "items.jsonl")
        lines[0]["output_ref"] = {"path": "eval_meta.json", "sha256": file_sha256(a / "eval_meta.json")}
        (a / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
        out = self.tmp / "mr" / "out"
        proc = merge(self.sc.paths["registration"], out, [a, b], cwd=self.sc.root)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertIn("EXT-MERGE", [e["rule"] for e in parse_errors(proc.stderr)])
        self.assertFalse(out.exists())


# ---------------------------------------------------------------------------
# EXT-46 (L6): one gpu-step label per launch; the log is the one gpu-step.sh writes; the run finished
# ---------------------------------------------------------------------------

class TestLaunchLabel(LaunchCase):

    def record(self, req: dict, profile: str = "smoke") -> tuple[dict, str]:
        proc, out = launch(self.sub(), req, profile=profile, root=self.tmp / "host")    # one host root (EXT-51)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        return json.loads(out.read_text(encoding="utf-8")), proc.stdout

    def test_label_is_hashed_over_the_record(self):
        rec, cmd = self.record(golden_request())
        base = golden_request()["label"]
        self.assertRegex(rec["label"], rf"^{base}-[0-9a-f]{{12}}$")
        self.assertEqual(rec["label"], launch_label(dict(rec, label=base)))
        self.assertIn(f"/bin/gpu-step.sh {rec['label']} ", cmd)

    def test_label_differs_between_launches(self):
        a, _ = self.record(golden_request())
        req = golden_request()
        req["argv"] = req["argv"] + ["--max-steps", "20"]
        b, _ = self.record(req)
        req = golden_request()
        req["gpu_est_h"] = 0.5
        c, _ = self.record(req)
        self.assertEqual(len({a["label"], b["label"], c["label"]}), 3)

    def test_same_request_same_label(self):
        """A relaunch of the identical request at the same running total is the same launch (its attempts are
        summed together, EXT-46)."""
        a, cmd_a = self.record(golden_request())
        b, cmd_b = self.record(golden_request())
        self.assertEqual((a["label"], cmd_a), (b["label"], cmd_b))

    def test_request_label_room_for_suffix(self):
        req = golden_request()
        req["label"] = "a-" + "b" * 200
        self.launch_refused(req, "label")


class TestGpuSecondsRc(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-gpu-rc-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def seconds(self, lines: list[str], label: str = LABEL):
        from gatedtrain.ext.launch import gpu_seconds
        p = self.tmp / "gpu-guard.log"
        p.write_text("".join(l + "\n" for l in lines), encoding="utf-8")
        return gpu_seconds(p, label)

    def test_final_end_failed_refused(self):
        with self.assertRaises(ValueError) as cm:
            self.seconds([f"2026-11-11T09:00:00Z START {LABEL} attempt=1",
                          f"2026-11-11T09:30:00Z END {LABEL} attempt=1 rc=1"])
        self.assertIn("rc", str(cm.exception))

    def test_failure_then_success_accepted(self):
        self.assertEqual(self.seconds([f"2026-11-11T09:00:00Z START {LABEL} attempt=1",
                                       f"2026-11-11T09:30:00Z END {LABEL} attempt=1 rc=1",
                                       f"2026-11-11T10:00:00Z START {LABEL} attempt=1",
                                       f"2026-11-11T10:10:00Z END {LABEL} attempt=1 rc=0"]), 2400)

    def test_success_then_failure_refused(self):
        with self.assertRaises(ValueError):
            self.seconds([f"2026-11-11T09:00:00Z START {LABEL} attempt=1",
                          f"2026-11-11T09:30:00Z END {LABEL} attempt=1 rc=0",
                          f"2026-11-11T10:00:00Z START {LABEL} attempt=1",
                          f"2026-11-11T10:10:00Z END {LABEL} attempt=1 rc=2"])

    def test_end_without_rc_refused(self):
        with self.assertRaises(ValueError):
            self.seconds([f"2026-11-11T09:00:00Z START {LABEL} attempt=1",
                          f"2026-11-11T09:30:00Z END {LABEL} attempt=1"])

    def test_fixture_log_unchanged(self):
        from gatedtrain.ext.launch import gpu_seconds
        self.assertEqual(gpu_seconds(EXT / "gpu-guard.fixture.log", LABEL), 3902)


class TestGpuCollect(CollectCase):

    def test_unhashed_label_refused(self):
        rec = self.sc.launch_record(hashed_label=False)
        path = write_canonical(self.path("launch").with_suffix(".json"), rec)
        guard_log(self.sc.host_work, rec["label"])
        self.addCleanup(guard_log, self.sc.host_work, self.sc.label)
        self.refused("EXT-LAUNCH", "--launch-record.label", launch=path)

    def test_edited_record_refused(self):
        """The label no longer hashes the record after a field was edited."""
        rec = self.sc.launch_record()
        rec["gpu_est_h"] = 1.0
        path = write_canonical(self.path("launch").with_suffix(".json"), rec)
        self.refused("EXT-LAUNCH", "--launch-record.label", launch=path)

    def test_other_labels_not_summed(self):
        """Lines of the request label without the hash, and of another launch, add nothing."""
        lab = self.sc.label
        guard_log(self.sc.host_work, lab, lines=[
            f"2026-11-11T08:00:00Z START {LABEL} attempt=1", f"2026-11-11T08:20:00Z END {LABEL} attempt=1 rc=0",
            f"2026-11-11T08:30:00Z START {LABEL}-0123456789ab attempt=1",
            f"2026-11-11T08:40:00Z END {LABEL}-0123456789ab attempt=1 rc=0",
            f"2026-11-11T09:00:00Z START {lab} attempt=1", f"2026-11-11T09:40:00Z END {lab} attempt=1 rc=0"])
        self.addCleanup(guard_log, self.sc.host_work, lab)
        self.assertEqual(self.passes()["gpu_seconds"], 2400)

    def test_failed_run_refused(self):
        lab = self.sc.label
        guard_log(self.sc.host_work, lab, lines=[f"2026-11-11T09:00:00Z START {lab} attempt=1",
                                                 f"2026-11-11T09:40:00Z END {lab} attempt=1 rc=1"])
        self.addCleanup(guard_log, self.sc.host_work, lab)
        self.refused("EXT-GPU", "--launch-record")

    def test_gpu_step_logs_elsewhere_refused(self):
        """bin/gpu-step.sh hardcodes its WORK= directory: a record whose host_work is another directory has its
        attempts logged elsewhere."""
        gpu_step_script(self.sc.host_work, work="/srv/gatedtrain/tierb")
        self.addCleanup(gpu_step_script, self.sc.host_work)
        self.refused("EXT-GPU", "--launch-record.host_work")

    def test_gpu_step_missing_refused(self):
        script = self.sc.host_work / "bin" / "gpu-step.sh"
        script.unlink()
        self.addCleanup(gpu_step_script, self.sc.host_work)
        self.refused("EXT-GPU", "--launch-record.host_work")


# ---------------------------------------------------------------------------
# EXT-47 (L7, review mutant M25): merge provenance, base binding of the evaluated model, ext-from-nel scores
# ---------------------------------------------------------------------------

class TestMergeProvenance(CollectCase):

    def merged(self, name: str) -> tuple[Path, list[Path]]:
        srcs = [build_arm(self.tmp / name / f"s{s}", self.sc.items, baseline_score, model=self.sc.base_model,
                          item_set=self.sc.item_set_hash, seeds=[s]) for s in SEEDS]
        out = self.tmp / name / "merged"
        proc = merge(self.sc.paths["registration"], out, srcs, cwd=self.sc.root)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        return out, srcs

    def edit_merge_json(self, arm: Path, fn):
        m = json.loads((arm / "merge.json").read_text(encoding="utf-8"))
        fn(m)
        (arm / "merge.json").write_text(json.dumps(m), encoding="utf-8")

    def test_merged_arm_passes(self):
        out, _ = self.merged("ok")
        self.assertEqual(self.passes(baseline=out)["status"], "PASS")

    def test_source_changed_refused(self):
        out, srcs = self.merged("changed")
        with (srcs[1] / "items.jsonl").open("a", encoding="utf-8") as f:
            f.write("\n")
        self.refused("EXT-ARM", "--eval-baseline", baseline=out,
                     reason="merge.json source sources[1] items.jsonl differs from the recorded hash")

    def test_source_missing_refused(self):
        out, srcs = self.merged("missing")
        shutil.rmtree(srcs[0])
        self.refused("EXT-ARM", "--eval-baseline", baseline=out, reason="is absent; provenance cannot be checked")

    def test_seeds_disagree_refused(self):
        out, _ = self.merged("seeds")
        self.edit_merge_json(out, lambda m: m["sources"][1].update(seeds=[SEEDS[0]]))
        self.refused("EXT-ARM", "--eval-baseline", baseline=out, reason="merge.json sources cover seeds")

    def test_source_meta_other_model_refused(self):
        """A source evaluated another model than the merged arm claims."""
        out, srcs = self.merged("model")
        meta = json.loads((srcs[1] / "eval_meta.json").read_text(encoding="utf-8"))
        meta["checkpoint"]["model_hash"] = "sha256:" + "9" * 64
        write_canonical(srcs[1] / "eval_meta.json", meta)
        self.edit_merge_json(out, lambda m: m["sources"][1].update(eval_meta_sha256=file_sha256(srcs[1] / "eval_meta.json")))
        self.refused("EXT-ARM", "--eval-baseline", baseline=out,
                     reason="eval_meta.json differs from the arm's beyond eval_seeds")

    def test_unknown_schema_refused(self):
        out, _ = self.merged("schema")
        self.edit_merge_json(out, lambda m: m.update(schema="gatedtrain.something"))
        self.refused("EXT-ARM", "--eval-baseline", baseline=out, reason="merge.json schema is 'gatedtrain.something'")

    def test_from_nel_sources_checked(self):
        runs = nel_runs(self.tmp / "nel", self.sc.rows, score_fn=scorer_candidate, model="cand")
        arm = self.tmp / "nel-arm"
        proc = convert(self.sc, arm, runs, checkpoint=self.sc.final_cut)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        m = json.loads((arm / "merge.json").read_text(encoding="utf-8"))
        self.assertEqual([s["service_model"] for s in m["sources"]], ["cand", "cand"])
        self.assertEqual(self.passes(candidate=arm)["status"], "PASS")
        with (runs[1] / "gsm8k" / "results.jsonl").open("a", encoding="utf-8") as f:
            f.write("\n")
        self.refused("EXT-ARM", "--eval-candidate", candidate=arm,
                     reason="merge.json source sources[1] gsm8k/results.jsonl differs from the recorded hash")


class TestEvaluatedBase(CollectCase):

    def test_base_model_hash_checked(self):
        other = self.sc.arm("other_base", candidate_score, root=self.tmp / "arms",
                            meta={"checkpoint": {"model_hash": self.sc.cand_model,
                                                 "base_model_hash": "sha256:" + "4" * 64}})
        self.refused("EXT-ARM", "--eval-candidate", candidate=other)
        same = self.sc.arm("same_base", candidate_score, root=self.tmp / "arms",
                           meta={"checkpoint": {"model_hash": self.sc.cand_model,
                                                "base_model_hash": self.sc.base_model}})
        self.assertEqual(self.passes(candidate=same)["status"], "PASS")


class TestFromNelHardening(CollectCase):

    def refused_nel(self, runs: list[Path], **kw):
        out = self.path("arm")
        proc = convert(self.sc, out, runs, checkpoint=kw.pop("checkpoint", self.sc.base), **kw)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertIn("EXT-NEL", [e["rule"] for e in parse_errors(proc.stderr)], proc.stderr)
        self.assertFalse(out.exists())

    def test_non_numeric_score_refused(self):
        for name, value in (("string", "1"), ("boolean", True), ("object", {"v": 1})):
            with self.subTest(score=name):
                runs = nel_runs(self.path(f"nel-{name}"), self.sc.rows, score_fn=scorer_baseline, model="base")
                results = runs[1] / "gsm8k" / "results.jsonl"
                lines = [json.loads(l) for l in results.read_text(encoding="utf-8").splitlines() if l.strip()]
                for rec in lines:
                    if rec["problem_idx"] == 3:
                        rec["scorer_rewards"]["scorer:numeric_match"] = value
                results.write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
                self.refused_nel(runs)

    def test_service_models_differ_refused(self):
        a = nel_runs(self.path("nel-a"), self.sc.rows, score_fn=scorer_baseline, model="base")
        b = nel_runs(self.path("nel-b"), self.sc.rows, score_fn=scorer_baseline, model="other")
        self.refused_nel([a[0], b[1]])


if __name__ == "__main__":
    unittest.main()
