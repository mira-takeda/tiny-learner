"""pilot2 local smoke (discarded outputs, /tmp/pilot2_smoke).

T1. Harness equivalence: pilot2 all-phases-12-gens == pilot_train
    12-gens bit-exact (theta, ckpts, histories, checks, heldouts,
    retention) on seed 301. Excludes meta fields (design/shas).
T2. Segmentation: C 10+10 with continued streams == C 20 single,
    bit-exact (history/checks/theta/steps). Segments must align with
    the 10-gen check cadence (production 400 % 10 == 0).
T3. Mid-capture machinery: C_MID patched to 10 with C=20 gens ->
    post_C010 ckpt + heldout exist, chain audit passes.
T4. Production step arithmetic: 3x41.28M + 82.56M = 206.4M/seed.

Exit 0 = SMOKE PASS.
"""

import json
import os
import shutil
import sys

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
sys.path.insert(0, f"{ROOT}/pilot2")
sys.path.insert(0, f"{ROOT}/pilot")
sys.path.insert(0, f"{ROOT}/src")

import numpy as np

import pilot2_train as P2
import pilot_train as P1
from train import skill_slice

OUT = "/tmp/pilot2_smoke"


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)

    # T1. equivalence at equal short gens
    pg = {f: 12 for f in ["A", "B", "C", "D"]}
    l2 = P2.run_seed2(301, f"{ROOT}/data", f"{OUT}/p2", pg)
    l1 = P1.run_seed(301, f"{ROOT}/data", f"{OUT}/p1", 12)
    assert l2["total_steps"] == l1["total_steps"]
    assert l2["retention_sampled"] == l1["retention_sampled"]
    assert l2["heldout_after"] == l1["heldout_after"], "heldouts differ"
    for f in ["A", "B", "C", "D"]:
        assert l2["phases"][f]["history"] == l1["phases"][f]["history"], f
        assert l2["phases"][f]["checks"] == l1["phases"][f]["checks"], f
    t2 = np.load(f"{OUT}/p2/P2_301_theta.npz")["theta"]
    t1 = np.load(f"{OUT}/p1/301_theta.npz")["theta"]
    assert np.array_equal(t2, t1)
    z2 = np.load(f"{OUT}/p2/P2_301_ckpts.npz")
    z1 = np.load(f"{OUT}/p1/301_ckpts.npz")
    assert sorted(z2.files) == sorted(z1.files), (z2.files, z1.files)
    for k in z1.files:
        assert np.array_equal(z2[k], z1[k]), k
    print("T1: pilot2 == pilot bit-exact at 12 gens (theta/ckpts/hist/"
          "checks/heldouts/retention)")

    # T2. segmentation 10+10 == 20 (segments MUST align with the
    # 10-gen check cadence; production 400+400 aligns: 400 % 10 == 0)
    train = json.load(open(f"{ROOT}/data/train_meta_v1.json"))
    c_entries = [e for e in train if e["family"] == "C"]
    rng = np.random.default_rng(0)
    from models import SKILL_SHAPES, GATE_DIM, N_SKILLS
    from policies import xavier_init
    th = np.concatenate([xavier_init(rng, SKILL_SHAPES)
                         for _ in range(N_SKILLS)] + [np.zeros(GATE_DIM)])
    sub = skill_slice(2)
    mk = lambda s: np.random.default_rng(s)
    cA, cB = [0], [0]
    fa, _, ha, ca = P1.run_pilot_phase(th.copy(), sub, "C", 2, c_entries,
                                       mk(1), mk(2), mk(3), cA, 20, 2)
    e1, e2, ev = mk(1), mk(2), mk(3)
    f1, _, h1, k1 = P1.run_pilot_phase(th.copy(), sub, "C", 2, c_entries,
                                       e1, e2, ev, cB, 10, 2)
    f2, _, h2, k2 = P1.run_pilot_phase(f1, sub, "C", 2, c_entries,
                                       e1, e2, ev, cB, 10, 2)
    assert np.array_equal(fa, f2) and ha == h1 + h2 and ca == k1 + k2 \
        and cA == cB
    print("T2: C 10+10 segments == single 20 (theta/history/checks/"
          "steps)")

    # T3. mid-capture machinery (patched C_MID=10, C=20)
    P2.C_MID = 10
    try:
        l3 = P2.run_seed2(302, f"{ROOT}/data", f"{OUT}/mid",
                          {"A": 2, "B": 2, "C": 20, "D": 2})
    finally:
        P2.C_MID = 400
    assert "post_C010" in l3["heldout_after"], l3["heldout_after"].keys()
    assert sorted(l3["heldout_after"]["post_C010"]) == ["A", "B", "C"]
    z3 = np.load(f"{OUT}/mid/P2_302_ckpts.npz")
    assert "post_C010" in z3.files
    print("T3: mid-C capture works (ckpt + heldout, chain audit passed)")

    # T4. production arithmetic
    assert P2.phase_steps(400) == 41280000
    assert P2.phase_steps(800) == 82560000
    tot = sum(P2.phase_steps(P2.PHASE_GENS[f]) for f in "ABCD")
    assert tot == 206400000, tot
    assert P2.PHASE_GENS == {"A": 400, "B": 400, "C": 800, "D": 400}
    assert P2.PILOT2_SEEDS == [301, 302, 303]
    print("T4: production steps 3x41.28M + 82.56M = 206.4M/seed")
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
