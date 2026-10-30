"""S8 data: 18 fresh train scenarios (ACQ8) + 200 fresh eval (ACQ9).

Same frozen generator (mkdata_conf.conflict_cfg), regional cycle,
band and dynamics. ACQ8: SeedSequence(9700), 18 eps. ACQ9:
SeedSequence(9800), 200 eps. Zero-duplicate checks vs all priors
explicitly incl. S3-eval-9 and S7-200. Counts/hashes only.
"""
import hashlib
import json
import math
import sys

sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/src")
sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/v2conf")

import numpy as np

from mkdata_conf import conflict_cfg

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
SETS = [("s8_acq8", "ACQ8", "S8", 9700, 18),
        ("s8_acq9", "ACQ9", "S9", 9800, 200)]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    regs = ["G1a", "G1b", "G1adv"]
    for name, batt, pre, ss, n in SETS:
        seeds = [int(s.generate_state(1, dtype=np.uint32)[0])
                 for s in np.random.SeedSequence(ss).spawn(n)]
        assert len(set(seeds)) == n
        eps = []
        for i, seed in enumerate(seeds):
            cfg = conflict_cfg(regs[i % 3], np.random.default_rng(seed))
            eps.append({"id": f"{pre}-ACQ-{i:04d}", "battery": batt,
                        "family": "G1", "region": regs[i % 3], "seed": seed,
                        "cfg": cfg})
        out = f"{ROOT}/v2conf_diag/{name}.json"
        with open(out, "w") as f:
            json.dump(eps, f, indent=1)
            f.write("\n")
        print(f"wrote {name}.json n={n} sha256={sha(out)} "
              f"SeedSequence({ss})", flush=True)

    acq8 = json.load(open(f"{ROOT}/v2conf_diag/s8_acq8.json"))
    acq9 = json.load(open(f"{ROOT}/v2conf_diag/s8_acq9.json"))
    priors = []
    for p in ("data/train_meta_v1.json", "data/test_battery_v1.json",
              "v2views/eval_new_v1.json", "v2conf/train_conflict_v1.json",
              "v2conf/eval_conf_v1.json", "v2conf_diag/s7_acq7.json"):
        priors += json.load(open(f"{ROOT}/{p}"))
    key = lambda e: (e["family"], e["seed"])
    cfgk = lambda e: (e["family"], json.dumps(e["cfg"], sort_keys=True))
    pk = {key(e) for e in priors}
    pc = {cfgk(e) for e in priors}
    for name, arr in (("ACQ8", acq8), ("ACQ9", acq9)):
        print(f"{name} vs priors: seed-collisions="
              f"{sum(key(e) in pk for e in arr)} cfg-dups="
              f"{sum(cfgk(e) in pc for e in arr)}", flush=True)
        assert all(key(e) not in pk for e in arr)
        assert all(cfgk(e) not in pc for e in arr)
        ck = [cfgk(e) for e in arr]
        assert len(set(ck)) == len(arr), f"dup inside {name}"
    # Explicit exclusion: S3-eval-9 and S7-200 out of ACQ8-train.
    s3ev = set(json.load(open(f"{ROOT}/v2conf_diag/s3_results.json"))
               ["partition"]["eval"])
    s7ids = {e["id"] for e in json.load(
        open(f"{ROOT}/v2conf_diag/s7_acq7.json"))}
    acq8k = {key(e) for e in acq8}
    evc = {e["id"]: e for e in json.load(
        open(f"{ROOT}/v2conf/eval_conf_v1.json"))}
    s7 = {e["id"]: e for e in json.load(
        open(f"{ROOT}/v2conf_diag/s7_acq7.json"))}
    assert not (acq8k & {key(evc[i]) for i in s3ev}), "S3-eval leak"
    assert not (acq8k & {key(s7[i]) for i in s7ids}), "S7 leak"
    print("exclusions verified: no S3-eval-9 nor S7-200 in ACQ8-train",
          flush=True)
    bad = 0
    for e in acq8 + acq9:
        db = math.dist(e["cfg"]["goal"]["pos"], e["cfg"]["hazard"]["pos"])
        bad += not (12.0 <= db <= 25.0)
    print(f"band violations: {bad}", flush=True)
    assert bad == 0


if __name__ == "__main__":
    main()
