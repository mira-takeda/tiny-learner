"""S9 data: 200 fresh eval scenarios (ACQ10). Same frozen generator,
regional cycle, band, dynamics. SeedSequence(9900). Zero-duplicate
checks vs all priors incl. S7/S8 sets. Counts/hashes only.
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


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    seeds = [int(s.generate_state(1, dtype=np.uint32)[0])
             for s in np.random.SeedSequence(9900).spawn(200)]
    assert len(set(seeds)) == 200
    regs = ["G1a", "G1b", "G1adv"]
    eps = []
    for i, seed in enumerate(seeds):
        cfg = conflict_cfg(regs[i % 3], np.random.default_rng(seed))
        eps.append({"id": f"S10-ACQ-{i:04d}", "battery": "ACQ10",
                    "family": "G1", "region": regs[i % 3], "seed": seed,
                    "cfg": cfg})
    out = f"{ROOT}/v2conf_diag/s9_acq10.json"
    with open(out, "w") as f:
        json.dump(eps, f, indent=1)
        f.write("\n")
    print(f"wrote s9_acq10.json n=200 sha256={sha(out)}", flush=True)
    priors = []
    for p in ("data/train_meta_v1.json", "data/test_battery_v1.json",
              "v2views/eval_new_v1.json", "v2conf/train_conflict_v1.json",
              "v2conf/eval_conf_v1.json", "v2conf_diag/s7_acq7.json",
              "v2conf_diag/s8_acq8.json", "v2conf_diag/s8_acq9.json"):
        priors += json.load(open(f"{ROOT}/{p}"))
    key = lambda e: (e["family"], e["seed"])
    cfgk = lambda e: (e["family"], json.dumps(e["cfg"], sort_keys=True))
    pk = {key(e) for e in priors}
    pc = {cfgk(e) for e in priors}
    print(f"vs priors: seed-collisions={sum(key(e) in pk for e in eps)} "
          f"cfg-dups={sum(cfgk(e) in pc for e in eps)}", flush=True)
    assert all(key(e) not in pk for e in eps)
    assert all(cfgk(e) not in pc for e in eps)
    assert len({cfgk(e) for e in eps}) == 200
    bad = 0
    for e in eps:
        db = math.dist(e["cfg"]["goal"]["pos"], e["cfg"]["hazard"]["pos"])
        bad += not (12.0 <= db <= 25.0)
    print(f"band violations: {bad}", flush=True)
    assert bad == 0


if __name__ == "__main__":
    main()
