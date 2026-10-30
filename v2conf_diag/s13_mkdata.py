"""S13 data: 12 fresh train scenarios (ACQ15) + 200 fresh eval (ACQ16).
Same frozen generator, regional cycle, band, dynamics. ACQ13:
SeedSequence(9970). ACQ16: SeedSequence(9980). Zero-duplicate
checks vs all priors. Counts/hashes only. Writes its own wall
time for the cumulative budget.
"""
import hashlib
import json
import math
import sys
import time

sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/src")
sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/v2conf")

import numpy as np

from mkdata_conf import conflict_cfg

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
T0 = time.time()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    regs = ["G1a", "G1b", "G1adv"]
    for name, batt, pre, ss, n in (("s13_acq15", "ACQ15", "S15", 9970, 12),
                                   ("s13_acq16", "ACQ16", "S16", 9980, 200)):
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
        print(f"wrote {name}.json n={n} sha256={sha(out)}", flush=True)
    priors = []
    for p in ("data/train_meta_v1.json", "data/test_battery_v1.json",
              "v2views/eval_new_v1.json", "v2conf/train_conflict_v1.json",
              "v2conf/eval_conf_v1.json", "v2conf_diag/s7_acq7.json",
              "v2conf_diag/s8_acq8.json", "v2conf_diag/s8_acq9.json",
              "v2conf_diag/s9_acq10.json", "v2conf_diag/s10_acq11.json",
              "v2conf_diag/s10_acq12.json",
              "v2conf_diag/s11_acq13.json", "v2conf_diag/s11_acq14.json"):
        priors += json.load(open(f"{ROOT}/{p}"))
    key = lambda e: (e["family"], e["seed"])
    cfgk = lambda e: (e["family"], json.dumps(e["cfg"], sort_keys=True))
    pk = {key(e) for e in priors}
    pc = {cfgk(e) for e in priors}
    for name in ("s13_acq15", "s13_acq16"):
        arr = json.load(open(f"{ROOT}/v2conf_diag/{name}.json"))
        print(f"{name}: seed-collisions={sum(key(e) in pk for e in arr)} "
              f"cfg-dups={sum(cfgk(e) in pc for e in arr)}", flush=True)
        assert all(key(e) not in pk for e in arr)
        assert all(cfgk(e) not in pc for e in arr)
        assert len({cfgk(e) for e in arr}) == len(arr)
        bad = 0
        for e in arr:
            db = math.dist(e["cfg"]["goal"]["pos"], e["cfg"]["hazard"]["pos"])
            bad += not (12.0 <= db <= 25.0)
        assert bad == 0
    json.dump({"mkdata_wall_s": round(time.time() - T0, 1)},
              open(f"{ROOT}/v2conf_diag/s13_budget_seed.json", "w"))
    print("band ok; budget seed written", flush=True)


if __name__ == "__main__":
    main()
