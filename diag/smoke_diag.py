"""Minimal local smoke test for the diag harness (discarded outputs).

1. Equivalence: diag rollout (sampled/TaskRunner) == frozen train.rollout
   bit-exact on fixed episodes; diag run_phase(3 gens) == frozen
   run_es_phase bit-exact theta/history/checks with identical streams.
2. C3Runner: same obs/done/success as frozen for fixed actions; extra >= 0.
3. Tiny end-to-end per path (C0 x12 gens incl. checks; others x2 gens):
   exact step asserts, outputs exist, heldout present.
Out dir /tmp/diag_smoke (throwaway). Exit 0 = SMOKE PASS.
"""

import json
import os
import shutil
import sys

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
sys.path.insert(0, f"{ROOT}/diag")
sys.path.insert(0, f"{ROOT}/src")

import numpy as np

import diag_train as D
from models import SKILL_SHAPES
from policies import xavier_init
from tasks import TaskRunner
from train import (GEN_BUDGET, POP_PAIRS, HORIZON, CHECK_EPS, rollout,
                   run_es_phase, skill_slice, train_eval_rate)

OUT = "/tmp/diag_smoke"


def main():
    train = json.load(open(f"{ROOT}/data/train_meta_v1.json"))
    c_entries = [e for e in train if e["family"] == "C"]
    by_fam = {"C": c_entries}

    # 1a. rollout equivalence
    th = np.concatenate([xavier_init(np.random.default_rng(0), SKILL_SHAPES)
                         for _ in range(4)]
                        + [np.zeros(D.GATE_DIM)])
    for i in range(20):
        e = c_entries[i]
        c1, c2 = [0], [0]
        r1, s1 = rollout(th, "modular", "C", e["cfg"], 1000 + i, c1, 2)
        r2, s2 = D.diag_rollout(th, "modular", "C", e["cfg"], 1000 + i, c2,
                                2, False, TaskRunner)
        assert r1 == r2 and s1 == s2 and c1 == c2 == [HORIZON], i
    print("equiv rollout: 20/20 bit-exact")

    # 1b. 12-gen phase equivalence (covers fitness + one check)
    sub = skill_slice(2)
    mk = lambda s: np.random.default_rng(s)
    c1, c2 = [0], [0]
    rng_ev_frozen = mk(7)

    def frozen_check(full):
        return train_eval_rate(full, "modular", "C", c_entries,
                               rng_ev_frozen, CHECK_EPS, c1, 2)

    f_full, f_g, f_h, f_c = run_es_phase(
        th.copy(), sub, "modular", [("C", 8)], by_fam, mk(1), mk(2), c1,
        frozen_check, 12, False)
    d_full, d_g, d_h, d_c, _, _ = D.run_phase(
        th.copy(), sub, "modular", "C", c_entries, mk(1), mk(2), mk(7), c2,
        2, 8, False, TaskRunner, 12)
    assert np.array_equal(f_full, d_full) and f_h == d_h and f_c == d_c
    assert c1 == c2 == [12 * 64 * 8 * 200 + 8000], (c1, c2)
    print("equiv 12-gen phase: theta/history/checks/steps bit-exact")

    # 2. C3Runner: same trajectory, extra >= 0
    for i in range(5):
        e = c_entries[i * 37]
        rA, rB = TaskRunner(), D.C3Runner()
        oA, oB = rA.reset("C", e["cfg"], 55), rB.reset("C", e["cfg"], 55)
        assert np.array_equal(oA, oB)
        tot_extra = 0.0
        for t in range(60):
            a = (t * 3 + i) % 9
            oA, r1, d1, i1 = rA.step(a)
            oB, r2, d2, i2 = rB.step(a)
            assert np.array_equal(oA, oB) and d1 == d2 and i1 == i2
            assert r2 >= r1 - 1e-9
            tot_extra += r2 - r1
            if d1:
                break
        assert rB._extra >= 0.0
    print("C3Runner: trajectories identical, extra>=0 on 5 eps")

    # 3. tiny end-to-end per path
    shutil.rmtree(OUT, ignore_errors=True)
    for job, gens in [("C0", 12), ("C1", 2), ("C2", 2), ("C3", 2),
                      ("C4", 2), ("B0", 2), ("B1", 2), ("B2", 2)]:
        log = D.run_job(job, 201, f"{ROOT}/data", OUT, gens)
        n_fit = D.JOBS[job]["n_fit"]
        expect = (gens * (2 * POP_PAIRS) * n_fit * HORIZON
                  + (gens // 10) * CHECK_EPS * HORIZON)
        assert log["steps"] == expect, (job, log["steps"], expect)
        assert os.path.isfile(f"{OUT}/{job}_201.json")
        assert os.path.isfile(f"{OUT}/{job}_201_theta.npz")
        assert set(log["heldout"]) == {"sampled", "greedy"}
        ho = {m: round(v["sr"], 3) for m, v in log["heldout"].items()}
        print(f"tiny {job}x{gens}: steps={log['steps']} heldout={ho} OK")
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
