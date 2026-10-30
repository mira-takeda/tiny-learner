"""Generate frozen v2conf datasets (run ONCE, pre-training).

train_conflict_v1.json: 400 G1 entries (G1a 200/G1b 140/G1adv 60),
same rules as REGIONS except beacon-haz band forced to [12,25] by
deterministic rejection sampling (spawn/speed rules preserved).
SeedSequence(7000).
eval_conf_v1.json: ACQ 200 (same conflict distribution,
SeedSequence(9500)) + H1C/H2C/H3C 200 each via frozen sample_test
(SeedSequence 9501/9502/9503).
Verifies zero seed/cfg duplication vs train/test/eval_new/conflict
itself (same-family). Prints sha256 for the protocol record.
"""

import json
import sys

sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/src")
import numpy as np
from gen import (_agent, _pt, _dist, MIN_SEP, TRIES, REGIONS, sample_test,
                 sha256_file, write_json)

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
OUT = f"{ROOT}/v2conf"


def conflict_cfg(region, rng):
    """Mirror of gen.sample_config 'guarded' branch with the ONLY change:
    beacon-haz band 12-25 instead of >= beacon_haz_min. Same _agent/_pt,
    speed draws, TRIES, inner-50 loop, MIN_SEP rule."""
    r = REGIONS["G1"][region]
    assert r["kind"] == "guarded"
    for _ in range(TRIES):
        agent = _agent(rng)
        beacon = _pt(rng)
        if _dist(agent, beacon) < MIN_SEP:
            continue
        for _ in range(50):
            hp = _pt(rng)
            da = _dist(agent, hp)
            db = _dist(beacon, hp)
            if (r["spawn_lo"] <= da <= r["spawn_hi"]
                    and 12.0 <= db <= 25.0):
                return {"agent": agent.tolist(),
                        "goal": {"pos": beacon.tolist()},
                        "hazard": {"pos": hp.tolist(), "speed": float(
                            rng.uniform(r["speed_lo"], r["speed_hi"]))}}
    raise RuntimeError(f"conflict sampling failed: {region}")


def main():
    train, ev = [], []
    plan = [("G1a", 200), ("G1b", 140), ("G1adv", 60)]
    seeds = [int(s.generate_state(1, dtype=np.uint32)[0])
             for s in np.random.SeedSequence(7000).spawn(400)]
    k = 0
    for region, n in plan:
        for _ in range(n):
            seed = seeds[k]
            cfg = conflict_cfg(region, np.random.default_rng(seed))
            train.append({"id": f"C-G1-{k:04d}", "family": "G1",
                          "region": region, "seed": seed, "cfg": cfg})
            k += 1
    aseeds = [int(s.generate_state(1, dtype=np.uint32)[0])
              for s in np.random.SeedSequence(9500).spawn(200)]
    regs = ["G1a", "G1b", "G1adv"]
    for i, seed in enumerate(aseeds):
        cfg = conflict_cfg(regs[i % 3], np.random.default_rng(seed))
        ev.append({"id": f"A-ACQ-{i:04d}", "battery": "ACQ", "family": "G1",
                   "region": regs[i % 3], "seed": seed, "cfg": cfg})
    for batt, ss in [("H1", 9501), ("H2", 9502), ("H3", 9503)]:
        seeds = [int(s.generate_state(1, dtype=np.uint32)[0])
                 for s in np.random.SeedSequence(ss).spawn(200)]
        for i, seed in enumerate(seeds):
            ev.append({"id": f"N-{batt}-{i:04d}", "battery": batt + "C",
                       "family": batt, "seed": seed,
                       "cfg": sample_test(batt, np.random.default_rng(seed))})
    write_json(f"{OUT}/train_conflict_v1.json", train)
    write_json(f"{OUT}/eval_conf_v1.json", ev)
    # duplicate checks (same family): seeds + full cfgs
    old_t = json.load(open(f"{ROOT}/data/train_meta_v1.json"))
    old_b = json.load(open(f"{ROOT}/data/test_battery_v1.json"))
    old_n = json.load(open(f"{ROOT}/v2views/eval_new_v1.json"))
    key = lambda e: (e["family"], e["seed"])
    cfgk = lambda e: (e["family"], json.dumps(e["cfg"], sort_keys=True))
    prior_keys = {key(e) for e in old_t + old_b + old_n}
    prior_cfg = {cfgk(e) for e in old_t + old_b + old_n}
    for name, arr in (("train_conflict", train), ("eval_conf", ev)):
        print(name, "n=", len(arr),
              "seed-collisions=", sum(key(e) in prior_keys for e in arr),
              "cfg-dups=", sum(cfgk(e) in prior_cfg for e in arr))
    # ACQ vs conflict-train cross-check (no shared episodes)
    trk = {key(e) for e in train}
    trc = {cfgk(e) for e in train}
    acq = [e for e in ev if e["battery"] == "ACQ"]
    print("ACQ-vs-trainconflict seed-collisions=",
          sum(key(e) in trk for e in acq), "cfg-dups=",
          sum(cfgk(e) in trc for e in acq))
    print("train_conflict sha256=", sha256_file(f"{OUT}/train_conflict_v1.json"))
    print("eval_conf sha256=", sha256_file(f"{OUT}/eval_conf_v1.json"))


if __name__ == "__main__":
    main()
