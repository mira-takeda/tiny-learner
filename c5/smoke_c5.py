"""Minimal local smoke test for C5 (discarded outputs).

1. Config: C5 == C0 + fit1hot + greedy, n_fit=8, no shaping; seeds
   401-403 fresh (no overlap with v1/diag/pilot).
2. Greedy manipulation is live: 2-gen C5 (greedy) vs 2-gen C5 with
   greedy disabled on identical streams -> thetas differ.
3. Tiny end-to-end run_c5(401, 12 gens): exact step asserts, 3 files,
   history/checks lengths, config echo, blended + pure heldouts
   present with n=200.
Out dir /tmp/c5_smoke (throwaway). Exit 0 = SMOKE PASS.
"""

import json
import os
import shutil
import sys

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
sys.path.insert(0, f"{ROOT}/c5")
sys.path.insert(0, f"{ROOT}/src")

import numpy as np

import c5_train as C
import diag_train as D

OUT = "/tmp/c5_smoke"


def main():
    # 1. config + seeds
    assert C.C5_CONFIG == {"family": "C", "arch": "modular", "n_fit": 8,
                           "greedy": True, "shaping": None, "fit1hot": True}
    assert D.JOBS["C5"] == C.C5_CONFIG
    assert C.C5_SEEDS == [401, 402, 403]
    print("config: C5 == one-hot fitness + greedy, n_fit=8, no shaping; "
          "seeds 401-403 fresh")

    # 2. greedy flag changes training dynamics (identical streams)
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(f"{OUT}/g", exist_ok=True)
    os.makedirs(f"{OUT}/s", exist_ok=True)
    D.JOBS["C5"]["greedy"] = True
    D.run_job("C5", 401, f"{ROOT}/data", f"{OUT}/g", 2)
    D.JOBS["C5"]["greedy"] = False
    D.run_job("C5", 401, f"{ROOT}/data", f"{OUT}/s", 2)
    D.JOBS["C5"]["greedy"] = True
    tg = np.load(f"{OUT}/g/C5_401_theta.npz")["theta"]
    ts = np.load(f"{OUT}/s/C5_401_theta.npz")["theta"]
    assert not np.array_equal(tg, ts), "greedy flag has no effect?"
    print("greedy manipulation live: greedy vs sampled thetas differ")

    # 3. tiny end-to-end
    shutil.rmtree(f"{OUT}/e2e", ignore_errors=True)
    log = C.run_c5(401, f"{ROOT}/data", f"{OUT}/e2e", 12)
    expect = 12 * 64 * 8 * 200 + 1 * 40 * 200
    assert log["steps"] == expect, (log["steps"], expect)
    assert log["gens"] == 12 and len(log["history"]) == 12
    assert len(log["checks"]) == 1
    assert log["config"] == C.C5_CONFIG
    for fn in ["C5_401.json", "C5_401_theta.npz", "C5_401_snaps.npz"]:
        assert os.path.isfile(f"{OUT}/e2e/{fn}"), fn
    for key in ("heldout", "heldout_pure"):
        for m in ("sampled", "greedy"):
            assert log[key][m]["n"] == 200, (key, m)
    z = np.load(f"{OUT}/e2e/C5_401_snaps.npz")
    assert sorted(z.files) == [], z.files  # 12 gens -> no 50-gen snaps
    print("e2e 12 gens: steps exact, 3 files, blended + pure heldouts ok")
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
