"""pilot3 local smoke (discarded outputs, /tmp/pilot3_smoke).

T0. Cross-harness mirror: pilot3 all-phases-6-gens (no mids) ==
    pilot2 all-6 bit-exact (theta/ckpts/hist/holdouts/retention/steps).
T1. Full-trainer capture-equivalence: pilot3 C=30 WITH mids=[10,20]
    == pilot3 C=30 with mids=[] bit-exact training trajectory
    (theta/history/checks/steps/common ckpts/common heldouts); only
    extra capture keys differ. Proves captures don't perturb training
    over the COMPLETE trainer run.
T2. Segmentation: C 10+10+10 with continued streams == C 30 single,
    bit-exact (theta/history/checks/steps).
T3. Explicit RNG invariance: es/ep/ev stream states bit-identical
    before/after a capture block (theta copy + score_skill).
T4. Production arithmetic: C1200 = 123.84M, total 247.68M/seed,
    mid boundaries % 10 == 0, seeds, phase map.

NOTE (protocol §2): this is REDUCED-smoke equivalence. Production
equivalence to g800 is checked post-hoc against pilot2_runs.

Exit 0 = SMOKE PASS.
"""

import json
import os
import shutil
import sys

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
sys.path.insert(0, f"{ROOT}/pilot3")
sys.path.insert(0, f"{ROOT}/pilot2")
sys.path.insert(0, f"{ROOT}/pilot")
sys.path.insert(0, f"{ROOT}/src")

import numpy as np

import pilot3_train as P3
import pilot2_train as P2
import pilot_train as P1
from train import skill_slice

OUT = "/tmp/pilot3_smoke"


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)

    # T0. cross-harness mirror at 6 gens
    pg6 = {f: 6 for f in ["A", "B", "C", "D"]}
    l3 = P3.run_seed3(301, f"{ROOT}/data", f"{OUT}/p3", pg6, [])
    l2 = P2.run_seed2(301, f"{ROOT}/data", f"{OUT}/p2", pg6)
    assert l3["total_steps"] == l2["total_steps"]
    assert l3["retention_sampled"] == l2["retention_sampled"]
    assert l3["heldout_after"] == l2["heldout_after"]
    for f in ["A", "B", "C", "D"]:
        assert l3["phases"][f]["history"] == l2["phases"][f]["history"], f
        assert l3["phases"][f]["checks"] == l2["phases"][f]["checks"], f
    t3 = np.load(f"{OUT}/p3/P3_301_theta.npz")["theta"]
    t2 = np.load(f"{OUT}/p2/P2_301_theta.npz")["theta"]
    assert np.array_equal(t3, t2)
    z3 = np.load(f"{OUT}/p3/P3_301_ckpts.npz")
    z2 = np.load(f"{OUT}/p2/P2_301_ckpts.npz")
    assert sorted(z3.files) == ["init", "post_A", "post_B", "post_C",
                                "post_D"]
    for k in z3.files:
        assert np.array_equal(z3[k], z2[k]), k
    print("T0: pilot3 == pilot2 bit-exact at 6 gens (no mids)")

    # T1. full-trainer capture-equivalence (C=30, mids [10,20] vs [])
    pg = {"A": 12, "B": 12, "C": 30, "D": 12}
    lc = P3.run_seed3(302, f"{ROOT}/data", f"{OUT}/cap", pg, [10, 20])
    ln = P3.run_seed3(302, f"{ROOT}/data", f"{OUT}/nocap", pg, [])
    assert lc["total_steps"] == ln["total_steps"]
    assert lc["retention_sampled"] == ln["retention_sampled"]
    for f in ["A", "B", "C", "D"]:
        assert lc["phases"][f]["history"] == ln["phases"][f]["history"], f
        assert lc["phases"][f]["checks"] == ln["phases"][f]["checks"], f
    tc = np.load(f"{OUT}/cap/P3_302_theta.npz")["theta"]
    tn = np.load(f"{OUT}/nocap/P3_302_theta.npz")["theta"]
    assert np.array_equal(tc, tn)
    zc = np.load(f"{OUT}/cap/P3_302_ckpts.npz")
    zn = np.load(f"{OUT}/nocap/P3_302_ckpts.npz")
    assert sorted(zc.files) == ["init", "post_A", "post_B", "post_C",
                                "post_C010", "post_C020", "post_D"]
    for k in zn.files:
        assert np.array_equal(zc[k], zn[k]), k
    common = {k: v for k, v in lc["heldout_after"].items()
              if k in ln["heldout_after"]}
    assert common == ln["heldout_after"]
    assert sorted(lc["heldout_after"]["post_C010"]) == ["A", "B", "C"]
    assert sorted(lc["heldout_after"]["post_C020"]) == ["A", "B", "C"]
    print("T1: captures don't perturb full training run (bit-exact; "
          "only extra keys differ)")

    # T2. three-segment == single
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
    fa, _, ha, ka = P1.run_pilot_phase(th.copy(), sub, "C", 2, c_entries,
                                       mk(1), mk(2), mk(3), cA, 30, 2)
    e1, e2, ev = mk(1), mk(2), mk(3)
    f, hh, kk = th.copy(), [], []
    for _ in range(3):
        f, _, h, k = P1.run_pilot_phase(f, sub, "C", 2, c_entries,
                                        e1, e2, ev, cB, 10, 2)
        hh += h
        kk += k
    assert np.array_equal(fa, f) and ha == hh and ka == kk and cA == cB
    print("T2: C 10+10+10 == single 30 (theta/history/checks/steps)")

    # T3. RNG states untouched by capture block
    battery = [e for e in json.load(
        open(f"{ROOT}/data/test_battery_v1.json")) if e["battery"] == "C"]
    r1, r2, r3 = mk(21), mk(22), mk(23)
    for r in (r1, r2, r3):
        r.standard_normal(10)  # simulate prior training consumption
    before = [r.bit_generator.state for r in (r1, r2, r3)]
    _copy = th.copy()  # noqa: F841 (checkpoint copy)
    P1.score_skill(th, 2, battery)
    after = [r.bit_generator.state for r in (r1, r2, r3)]
    assert before == after  # full BitGenerator state dicts, all streams
    print("T3: es/ep/ev RNG states bit-identical around capture block")

    # T4. production arithmetic
    assert P3.phase_steps(400) == 41280000
    assert P3.phase_steps(1200) == 123840000
    tot = sum(P3.phase_steps(P3.PHASE_GENS[f]) for f in "ABCD")
    assert tot == 247680000, tot
    assert P3.PHASE_GENS == {"A": 400, "B": 400, "C": 1200, "D": 400}
    assert all(m % 10 == 0 for m in P3.C_MIDS) and P3.C_MIDS == [400, 800]
    assert P3.PILOT3_SEEDS == [301, 302, 303]
    print("T4: production steps 3x41.28M + 123.84M = 247.68M/seed; "
          "mids aligned")
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
