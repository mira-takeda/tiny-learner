"""S7 data: 200 fresh ACQ scenarios with the frozen v2conf generator.

Same code path (mkdata_conf.conflict_cfg), same regional cycle
(G1a/G1b/G1adv), same conflict band (beacon-haz 12-25) and dynamics
as eval_conf ACQ. New SeedSequence(9600). Provenance + sha256
recorded; zero-duplicate checks vs all prior batteries. Prints
counts/hashes only; no outcomes exist or are inspected here.
"""
import hashlib
import json
import sys

sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/src")
sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/v2conf")

import numpy as np

from mkdata_conf import conflict_cfg

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
SEEDSEQ = 9600
N = 200
OUT = f"{ROOT}/v2conf_diag/s7_acq7.json"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    seeds = [int(s.generate_state(1, dtype=np.uint32)[0])
             for s in np.random.SeedSequence(SEEDSEQ).spawn(N)]
    assert len(set(seeds)) == N, "seed collision inside ACQ7"
    regs = ["G1a", "G1b", "G1adv"]
    eps = []
    for i, seed in enumerate(seeds):
        cfg = conflict_cfg(regs[i % 3], np.random.default_rng(seed))
        eps.append({"id": f"S7-ACQ-{i:04d}", "battery": "ACQ7",
                    "family": "G1", "region": regs[i % 3], "seed": seed,
                    "cfg": cfg})
    with open(OUT, "w") as f:
        json.dump(eps, f, indent=1)
        f.write("\n")
    print(f"wrote {OUT} n={len(eps)} sha256={sha(OUT)}")
    print(f"provenance: mkdata_conf.conflict_cfg "
          f"(sha {sha(f'{ROOT}/v2conf/mkdata_conf.py')[:12]}), "
          f"SeedSequence({SEEDSEQ}), regions cycle i%3")

    # Duplicate checks vs every prior battery + within ACQ7.
    priors = []
    for p in ("data/train_meta_v1.json", "data/test_battery_v1.json",
              "v2views/eval_new_v1.json", "v2conf/train_conflict_v1.json",
              "v2conf/eval_conf_v1.json"):
        priors += json.load(open(f"{ROOT}/{p}"))
    key = lambda e: (e["family"], e["seed"])
    cfgk = lambda e: (e["family"], json.dumps(e["cfg"], sort_keys=True))
    pk = {key(e) for e in priors}
    pc = {cfgk(e) for e in priors}
    print(f"vs priors: seed-collisions={sum(key(e) in pk for e in eps)} "
          f"cfg-dups={sum(cfgk(e) in pc for e in eps)}")
    assert all(key(e) not in pk for e in eps), "seed overlap with priors"
    assert all(cfgk(e) not in pc for e in eps), "cfg overlap with priors"
    ck = [cfgk(e) for e in eps]
    assert len(set(ck)) == N, "dup cfg inside ACQ7"
    acq = [e for e in json.load(open(f"{ROOT}/v2conf/eval_conf_v1.json"))
           if e["battery"] == "ACQ"]
    ak = {key(e) for e in acq}
    print(f"vs ACQ specifically: shared-episodes="
          f"{sum(key(e) in ak for e in eps)}")
    # Geometry sanity (counts only): conflict band respected.
    import math
    bad = 0
    for e in eps:
        g = e["cfg"]["goal"]["pos"]
        h = e["cfg"]["hazard"]["pos"]
        db = math.dist(g, h)
        if not (12.0 <= db <= 25.0):
            bad += 1
    print(f"band violations: {bad}")
    assert bad == 0


if __name__ == "__main__":
    main()
