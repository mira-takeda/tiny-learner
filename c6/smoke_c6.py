"""Minimal local smoke test for C6 (discarded outputs).

1. Config: C6 == C4 config exact (one-hot, SAMPLED, n_fit=8, no
   shaping); default gens 800; seeds 501-503 fresh.
2. 800-gen arithmetic: expected steps/checks/snaps counts for the real
   run (no 800-gen execution locally).
3. Tiny end-to-end run_c6(501, 12 gens): exact step asserts, 3 files,
   config echo, blended + pure heldouts with n=200.
Out dir /tmp/c6_smoke (throwaway). Exit 0 = SMOKE PASS.
"""

import json
import os
import shutil
import sys

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
sys.path.insert(0, f"{ROOT}/c6")
sys.path.insert(0, f"{ROOT}/src")

import c6_train as C
import diag_train as D
from train import HORIZON, CHECK_EPS, POP_PAIRS

OUT = "/tmp/c6_smoke"


def main():
    # 1. config + seeds + default budget
    assert C.C6_CONFIG == {"family": "C", "arch": "modular", "n_fit": 8,
                           "greedy": False, "shaping": None,
                           "fit1hot": True}
    assert D.JOBS["C4"] == C.C6_CONFIG, "C6 must equal C4 config exactly"
    assert D.JOBS["C6"] == C.C6_CONFIG
    assert C.C6_GENS == 800 and C.C6_SEEDS == [501, 502, 503]
    print("config: C6 == C4 exact, sampled, 800 gens; seeds 501-503 fresh")

    # 2. 800-gen arithmetic (mirrors diag_train.run_job/run_phase)
    g = 800
    expect_steps = (g * (2 * POP_PAIRS) * 8 * HORIZON
                    + (g // 10) * CHECK_EPS * HORIZON)
    assert expect_steps == 82560000, expect_steps
    assert g // 10 == 80, "80 checks"
    assert [x for x in range(1, g + 1) if x % 50 == 0][-1] == 800
    assert len([x for x in range(1, g + 1) if x % 50 == 0]) == 16, "16 snaps"
    print("800-gen arithmetic: 82,560,000 steps, 80 checks, 16 snaps")

    # 3. tiny end-to-end
    shutil.rmtree(OUT, ignore_errors=True)
    log = C.run_c6(501, f"{ROOT}/data", OUT, 12)
    expect = 12 * 64 * 8 * 200 + 1 * 40 * 200
    assert log["steps"] == expect, (log["steps"], expect)
    assert log["gens"] == 12 and len(log["history"]) == 12
    assert len(log["checks"]) == 1
    assert log["config"] == C.C6_CONFIG
    for fn in ["C6_501.json", "C6_501_theta.npz", "C6_501_snaps.npz"]:
        assert os.path.isfile(f"{OUT}/{fn}"), fn
    for key in ("heldout", "heldout_pure"):
        for m in ("sampled", "greedy"):
            assert log[key][m]["n"] == 200, (key, m)
    print("e2e 12 gens: steps exact, 3 files, blended + pure heldouts ok")
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
