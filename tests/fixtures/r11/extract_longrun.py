"""Extract a compact per-item fixture from the real long run (read-only source).

Usage: python extract_longrun.py <run directory> <output directory>
"""
import gzip, hashlib, json, sys
from pathlib import Path

src = Path(sys.argv[1])          # <HOST_WORK>/runs/run-20261111T110313Z-7adda9b9
out = Path(sys.argv[2])
key = "scorer:numeric_match"
samples = {}
for arm in ("baseline", "candidate"):
    lines = (src / "eval" / arm / "results.jsonl").read_text(encoding="utf-8").splitlines()
    recs = []
    for ln in lines:
        r = json.loads(ln)
        recs.append({"item": r["problem_idx"], "repeat": r["repeat"], "seed": r["metadata"]["eval_seed"],
                     "score": r["scorer_rewards"][key], "error": None, "sub_metrics": None,
                     "output_ref": None})
    recs.sort(key=lambda d: (d["item"], d["repeat"]))
    data = "".join(json.dumps(d, sort_keys=True, separators=(",", ":")) + "\n" for d in recs).encode()
    with gzip.GzipFile(out / f"{arm}.items.jsonl.gz", "wb", mtime=0) as f:
        f.write(data)
    samples[arm] = [json.loads(lines[0]), json.loads(lines[1])]
(out / "raw_samples.json").write_text(json.dumps(samples, indent=1, sort_keys=True) + "\n", encoding="utf-8")
