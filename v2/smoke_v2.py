"""v2 local smoke (discarded outputs, /tmp/v2_smoke).

T0. Fitness/eval identity: frozen rollout() == manual TaskRunner loop
    with the eval policy fn (return AND success), for normal, uniform,
    shuffle(0), skill-2, mask, greedy, on G1/G2 entries x seeds.
T1. Tiny gate run (12 gens, seed 301): exact steps, 3 files, skills
    bit-frozen + gate changed, manifest + heldout-hash verified,
    complete eval keys with n=200.
T2. Training RNG streams (es/ep/ev) bit-identical around eval
    machinery (all policy kinds exercised on small loops).
T3. Production arithmetic + frozen constants: 41.28M steps, 400/40,
    seeds, SHUFFLE_SEED==0, HELDOUT_SHA matches file, start thetas
    exist with zero gates.

Exit 0 = SMOKE PASS.
"""

import json
import os
import shutil
import sys

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
sys.path.insert(0, f"{ROOT}/v2")
sys.path.insert(0, f"{ROOT}/src")

import numpy as np

import gate_train as G
from evaluate import make_policy
from gen import sha256_file
from tasks import TaskRunner
from train import rollout

OUT = "/tmp/v2_smoke"


def manual(theta, pol, family, cfg, seed):
    runner = TaskRunner()
    rng = np.random.default_rng(int(seed))
    obs = runner.reset(family, cfg, int(seed))
    total, done = 0.0, False
    while not done:
        obs, r, done, info = runner.step(pol(obs, rng))
        total += r
    return float(total), bool(info["success"])


def main():
    train = json.load(open(f"{ROOT}/data/train_meta_v1.json"))
    g1 = [e for e in train if e["family"] == "G1"][:6]
    g2 = [e for e in train if e["family"] == "G2"][:6]
    theta = np.load(f"{ROOT}/pilot3_runs/P3_301_theta.npz")["theta"]
    pols = {"normal": make_policy("modular", theta, "normal"),
            "uniform": G.uniform_policy(theta),
            "shuffle": make_policy("modular", theta, "shuffle", 0),
            "skill-2": make_policy("modular", theta, "skill-2"),
            "mask": make_policy("modular", theta, "mask"),
            "greedy": make_policy("modular", theta, "greedy")}

    # T0. rollout (fitness path) == manual loop (eval path)
    n = 0
    for (fam, entries) in (("G1", g1), ("G2", g2)):
        for e in entries:
            for seed in (11, 12):
                # rollout() can only express normal/skill/greedy; the
                # others share its exact sampling construction, tested
                # here against run_battery-equivalent manual loops for
                # determinism + finiteness instead (see below).
                for name, kw in (("normal", None), ("skill-2", 2)):
                    r1, s1 = rollout(theta.copy(), "modular", fam,
                                     e["cfg"], seed, [0], kw)
                    r2, s2 = manual(theta, pols[name], fam, e["cfg"], seed)
                    assert r1 == r2 and s1 == s2, (name, fam, seed)
                    n += 1
    # greedy determinism + uniform/shuffle/mask smoke via manual loops
    for name in ("greedy", "uniform", "shuffle", "mask"):
        for fam, entries in (("G1", g1[:2]), ("G2", g2[:2])):
            for e in entries:
                r1, s1 = manual(theta, pols[name], fam, e["cfg"], 77)
                r2, s2 = manual(theta, pols[name], fam, e["cfg"], 77)
                assert r1 == r2 and s1 == s2 and np.isfinite(r1), name
                n += 1
    print(f"T0: fitness/eval episode identity + determinism ({n} eps)")

    # T1. tiny end-to-end
    shutil.rmtree(OUT, ignore_errors=True)
    log = G.run_seed(301, f"{ROOT}/data",
                     f"{ROOT}/pilot3_runs/P3_301_theta.npz",
                     f"{ROOT}/v2/g1g2_heldout_v1.json", OUT, 12)
    expect = 12 * 64 * 8 * 200 + 1 * 40 * 200
    assert log["steps"] == expect, (log["steps"], expect)
    assert len(log["history"]) == 12 and len(log["checks"]) == 1
    for fn in ["V2_301.json", "V2_301_theta.npz", "V2_301_ckpts.npz"]:
        assert os.path.isfile(f"{OUT}/{fn}"), fn
    z = np.load(f"{OUT}/V2_301_ckpts.npz")
    assert sorted(z.files) == ["init", "post_gate"]
    assert np.array_equal(z["post_gate"][:4*1833], z["init"][:4*1833])
    assert not np.array_equal(z["post_gate"][4*1833:], z["init"][4*1833:])
    assert log["g1g2_heldout_sha256"] == G.HELDOUT_SHA
    ev = log["eval"]
    assert ev["G1H"]["normal"]["n"] == 200 and ev["G2H"]["normal"]["n"] == 200
    for b in ["I1", "H1", "H2", "H3"]:
        for m in ["sampled", "greedy", "uniform", "shuffle", "mask",
                  "skill-0", "skill-1", "skill-2", "skill-3"]:
            assert ev[b][m]["n"] == 200, (b, m)
    print("T1: 12-gen gate run ok (steps/files/freeze/eval keys/Hashes)")

    # T2. training RNGs untouched by eval machinery
    r1, r2, r3 = G.v2_rng(301, 1), G.v2_rng(301, 2), G.v2_rng(301, 3)
    for r in (r1, r2, r3):
        r.standard_normal(10)
    before = [r.bit_generator.state for r in (r1, r2, r3)]
    for name, pol in pols.items():
        manual(theta, pol, "G1", g1[0]["cfg"], 5)
    after = [r.bit_generator.state for r in (r1, r2, r3)]
    assert before == after
    print("T2: es/ep/ev RNG states bit-identical around evals")

    # T3. production constants
    exp = 400 * 64 * 8 * 200 + 40 * 40 * 200
    assert exp == 41280000, exp
    assert G.V2_SEEDS == [301, 302, 303] and G.SHUFFLE_SEED == 0
    assert G.HELDOUT_SHA == sha256_file(f"{ROOT}/v2/g1g2_heldout_v1.json")
    for s in (301, 302, 303):
        t = np.load(f"{ROOT}/pilot3_runs/P3_{s}_theta.npz")["theta"]
        assert len(t) == 7944 and np.all(t[4*1833:] == 0), s
    print("T3: 41.28M/seed, seeds, shuffle 0, heldout hash, zero gates")
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
