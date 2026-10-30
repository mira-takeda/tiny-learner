"""Minimal local smoke test for the v2-pilot harness (discarded outputs).

1. Harness equivalence: pilot run_pilot_phase with fit_skill_only=None ==
   frozen run_es_phase bit-exact (theta/history/checks/steps, 12 gens).
2. One-hot fitness correctness: make_fitness(k) equals a manual mean of
   frozen rollout(skill_only=k) on identical streams; and differs from
   the full-net variant (the manipulation is real).
3. Tiny end-to-end run_seed(301, 12 gens): exact step asserts, 3 output
   files, checkpoint chain bit-consistent, retention drops exactly 0.0,
   heldout keys complete, manifest verified.
Out dir /tmp/pilot_smoke (throwaway). Exit 0 = SMOKE PASS.
"""

import json
import os
import shutil
import sys

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
sys.path.insert(0, f"{ROOT}/pilot")
sys.path.insert(0, f"{ROOT}/src")

import numpy as np

import pilot_train as P
from models import SKILL_SHAPES, GATE_DIM, N_SKILLS
from policies import xavier_init
from train import (POP_PAIRS, HORIZON, CHECK_EPS, run_es_phase, skill_slice,
                   train_eval_rate, rollout)

OUT = "/tmp/pilot_smoke"


def main():
    train = json.load(open(f"{ROOT}/data/train_meta_v1.json"))
    entries = {f: [e for e in train if e["family"] == f]
               for f in ["A", "B", "C", "D"]}
    # NOTE: one sequential stream (like v1/pilot init) so slices differ;
    # a fresh rng(0) per slice would make one-hot == blend trivially.
    rng0 = np.random.default_rng(0)
    th = np.concatenate([xavier_init(rng0, SKILL_SHAPES)
                         for _ in range(N_SKILLS)] + [np.zeros(GATE_DIM)])
    assert not np.array_equal(th[skill_slice(0)], th[skill_slice(1)])
    sub = skill_slice(2)
    mk = lambda s: np.random.default_rng(s)

    # 1. harness equivalence (fitness=None path == frozen)
    c1, c2 = [0], [0]
    rng_ev_frozen = mk(7)

    def frozen_check(full):
        return train_eval_rate(full, "modular", "C", entries["C"],
                               rng_ev_frozen, CHECK_EPS, c1, 2)

    f_full, _, f_h, f_c = run_es_phase(
        th.copy(), sub, "modular", [("C", 8)], {"C": entries["C"]},
        mk(1), mk(2), c1, frozen_check, 12, False)
    p_full, _, p_h, p_c = P.run_pilot_phase(
        th.copy(), sub, "C", 2, entries["C"], mk(1), mk(2), mk(7), c2,
        12, None)
    assert np.array_equal(f_full, p_full) and f_h == p_h and f_c == p_c
    assert c1 == c2 == [12 * 64 * 8 * 200 + 8000], (c1, c2)
    print("equiv 12-gen phase (fitness=None): bit-exact vs frozen")

    # 2. one-hot fitness correctness vs manual frozen rollouts
    base, sub_theta = th.copy(), th[sub].copy()
    for k, fam in [(0, "A"), (2, "C")]:
        cA, cB = [0], [0]
        f = P.make_fitness(base, sub, fam, entries[fam], mk(11), cA, k)
        got = f(sub_theta)
        rng = mk(11)
        manual = []
        for _ in range(8):
            e = entries[fam][int(rng.integers(len(entries[fam])))]
            ret, _ = rollout(base, "modular", fam, e["cfg"],
                             int(rng.integers(2 ** 31)), cB, k)
            manual.append(ret)
        assert got == float(np.mean(manual)), (fam, got, manual)
        assert cA == cB == [8 * HORIZON]
        f_fullnet = P.make_fitness(base, sub, fam, entries[fam], mk(11),
                                   [0], None)
        assert f_fullnet(sub_theta) != got, \
            f"one-hot fitness identical to full-net for {fam}?"
    print("one-hot fitness: matches manual frozen rollout means (A, C); "
          "differs from full-net")

    # 3. tiny end-to-end
    shutil.rmtree(OUT, ignore_errors=True)
    log = P.run_seed(301, f"{ROOT}/data", OUT, 12)
    expect = 12 * (2 * POP_PAIRS) * 8 * HORIZON + 1 * CHECK_EPS * HORIZON
    for fam in ["A", "B", "C", "D"]:
        assert log["phases"][fam]["steps"] == expect, fam
        assert len(log["phases"][fam]["history"]) == 12, fam
        assert len(log["phases"][fam]["checks"]) == 1, fam
    assert log["total_steps"] == 4 * expect, log["total_steps"]
    for fn in ["301.json", "301_theta.npz", "301_ckpts.npz"]:
        assert os.path.isfile(f"{OUT}/{fn}"), fn
    z = np.load(f"{OUT}/301_ckpts.npz")
    assert sorted(z.files) == ["init", "post_A", "post_B", "post_C",
                               "post_D"], z.files
    th_fin = np.load(f"{OUT}/301_theta.npz")["theta"]
    assert np.array_equal(th_fin, z["post_D"])
    for fam, k in [("A", 0), ("B", 1), ("C", 2), ("D", 3)]:
        for later in ["A", "B", "C", "D"][["A", "B", "C", "D"].index(fam):]:
            assert np.array_equal(z[f"post_{later}"][P.skill_slice(k)],
                                  z[f"post_{fam}"][P.skill_slice(k)])
    for fam in ["A", "B", "C", "D"]:
        r = log["retention_sampled"][fam]
        assert r["drop"] == 0.0, (fam, r)
        for post, skills in [("post_A", ["A"]), ("post_B", ["A", "B"]),
                             ("post_C", ["A", "B", "C"]),
                             ("post_D", ["A", "B", "C", "D"])]:
            h = log["heldout_after"][post]
            assert sorted(h) == skills, (post, sorted(h))
            for s in skills:
                assert h[s]["sampled"]["n"] == 200, (post, s)
                assert h[s]["greedy"]["n"] == 200, (post, s)
    print("e2e 12-gens x A-D: steps exact, 3 files, freeze chain "
          "bit-exact, retention drops 0.0, heldout keys complete")
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
