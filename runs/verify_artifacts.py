"""Verify downloaded registered artifacts (READ-ONLY on runs_modal/).

Checks: exact 40-file set, JSON/NPZ pairing, filename<->content arm/seed
match, frozen budget constants (400 gens/phase, 41.28M steps/phase,
206.4M total), retention/reacquisition structure, theta shapes/finiteness.
Writes NOTHING into runs_modal/; manifest goes to runs/artifacts_manifest.json.
Exit 0 = all PASS; exit 1 = STOP before scoring.
"""

import hashlib
import json
import os
import sys

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
SRC = f"{ROOT}/src"
RM = f"{ROOT}/runs_modal"
OUT = f"{ROOT}/runs/artifacts_manifest.json"

sys.path.insert(0, SRC)
import train as T
from models import MODULAR_DIM, MONO_DIM

ARMS = ["A1", "A2", "A3", "A4"]
SEEDS = [101, 102, 103, 104, 105]
FAIL = []


def check(name, cond, detail=""):
    if not cond:
        FAIL.append(f"{name} {detail}")
    return bool(cond)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    import numpy as np

    files = sorted(os.listdir(RM))
    want = sorted(f"{a}_{s}{e}" for a in ARMS for s in SEEDS
                  for e in (".json", "_theta.npz"))
    check("exact-40-files", files == want,
          f"got={len(files)} extra={[f for f in files if f not in want][:5]} "
          f"missing={[f for f in want if f not in files][:5]}")

    manifest = {"files": {}, "jobs": {}}
    for a in ARMS:
        for s in SEEDS:
            jp, npz = f"{RM}/{a}_{s}.json", f"{RM}/{a}_{s}_theta.npz"
            tag = f"{a}_{s}"
            if not (os.path.isfile(jp) and os.path.isfile(npz)):
                check(f"{tag}-pair-present", False)
                continue
            manifest["files"][f"{a}_{s}.json"] = {
                "sha256": sha256(jp), "bytes": os.path.getsize(jp)}
            manifest["files"][f"{a}_{s}_theta.npz"] = {
                "sha256": sha256(npz), "bytes": os.path.getsize(npz)}
            try:
                log = json.load(open(jp))
            except Exception as e:
                check(f"{tag}-json-parse", False, repr(e)[:120])
                continue
            ok = check(f"{tag}-arm-seed-match",
                       log.get("arm") == a and log.get("seed") == s,
                       f"{log.get('arm')}_{log.get('seed')}")
            if not ok:
                continue
            for idx, fam in enumerate(["A", "B", "C", "D"]):
                ph = log["phases"][fam]
                check(f"{tag}-{fam}-gens", ph["gens"] == T.GEN_BUDGET, ph["gens"])
                check(f"{tag}-{fam}-checks",
                      len(ph["checks"]) == T.CHECKS_PER_PHASE, len(ph["checks"]))
                check(f"{tag}-{fam}-hist", len(ph["history"]) == T.GEN_BUDGET)
                check(f"{tag}-{fam}-steps",
                      ph["steps"] == (idx + 1) * T.PHASE_STEPS, ph["steps"])
                check(f"{tag}-{fam}-acq-bool",
                      isinstance(ph["acquired"], bool))
                fp = ph["first_pass_gen"]
                check(f"{tag}-{fam}-fp",
                      fp is None or (isinstance(fp, int) and fp % 10 == 0), fp)
                check(f"{tag}-{fam}-acq-consistency",
                      ph["acquired"] == (fp is not None), (ph["acquired"], fp))
            g = log["phases"]["gate"]
            check(f"{tag}-gate-gens", g["gens"] == T.GEN_BUDGET, g["gens"])
            check(f"{tag}-gate-steps", g["steps"] == 5 * T.PHASE_STEPS, g["steps"])
            check(f"{tag}-total-steps",
                  log["total_steps"] == 5 * T.PHASE_STEPS, log["total_steps"])
            for rk in ["after_A", "after_B", "after_C", "after_D", "after_gate"]:
                check(f"{tag}-ret-{rk}", rk in log["retention"])
                for fam, v in log["retention"][rk].items():
                    check(f"{tag}-ret-{rk}-{fam}",
                          v is None or (isinstance(v, dict)
                                        and 0.0 <= v["sr"] <= 1.0
                                        and 0.0 <= v["ci"] <= 1.0), v)
            for fam, r in log.get("reacquisition", {}).items():
                check(f"{tag}-reac-{fam}-keys",
                      set(r) == {"pre", "dropped", "post", "gens",
                                 "scratch_first_pass", "savings"}, set(r))
                check(f"{tag}-reac-{fam}-rates",
                      all(0.0 <= r[k] <= 1.0 for k in ("pre", "dropped", "post")))
                check(f"{tag}-reac-{fam}-gens",
                      isinstance(r["gens"], int) and r["gens"] >= 1, r["gens"])
            try:
                z = np.load(npz)
            except Exception as e:
                check(f"{tag}-npz-load", False, repr(e)[:120])
                continue
            check(f"{tag}-npz-key", list(z.keys()) == ["theta"], list(z.keys()))
            th = z["theta"]
            dim = MONO_DIM if a == "A3" else MODULAR_DIM
            check(f"{tag}-theta-shape", th.shape == (dim,), th.shape)
            check(f"{tag}-theta-finite", bool(np.all(np.isfinite(th))))
            manifest["jobs"][tag] = {
                "acquired": {f: log["phases"][f]["acquired"]
                             for f in ["A", "B", "C", "D"]},
                "reacquired": sorted(log.get("reacquisition", {})),
                "total_steps": log["total_steps"]}
    json.dump(manifest, open(OUT, "w"), indent=1, sort_keys=True)
    print(f"manifest: {OUT}")
    if FAIL:
        print(f"INTEGRITY FAIL ({len(FAIL)}):")
        for f in FAIL[:40]:
            print("  FAIL", f)
        return 1
    print("INTEGRITY PASS: 40/40 artifacts, pairs, budgets, shapes all exact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
