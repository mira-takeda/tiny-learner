"""pilot3 frozen analysis (run AFTER download; read-only on pilot3_runs).

1. Integrity: 9/9 files, config, gens (A/B/D 400, C 1200), steps
   (41.28M x3 + 123.84M), freeze bit-exact, hashes.
2. Production equivalence vs pilot2_runs (pre-registered): A/B/D
   slices of post_D bit-identical; C-slice at g400/g800 bit-identical;
   heldouts post_A/B/C400/C800 identical. Mismatches -> triage.
3. Trend: Delta = C@g1200 - C@g800 pure sampled per seed; plateau if
   |D|<=0.07 in >=2/3; learning if D>+0.07 in >=2/3; else INCONCLUSIVE.
4. Gates: mean phase-end pure sampled >= 0.80 per skill; mean
   retention drop <= 0.10; greedy secondary. GO only if ALL pass.

Usage: python3.14 pilot3/score_pilot3.py [runs_dir]
"""

import hashlib
import json
import os
import sys
from math import sqrt

RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "pilot3_runs")
PILOT2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                      "pilot2_runs")
SEEDS = [301, 302, 303]
T32 = 4.303


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:12]


def main():
    import numpy as np

    files = sorted(os.listdir(RUNS))
    exp = {f"P3_{s}{x}" for s in SEEDS
           for x in (".json", "_theta.npz", "_ckpts.npz")}
    print(f"files={len(files)} missing={sorted(exp - set(files)) or 'none'} "
          f"extra={sorted(set(files) - exp) or 'none'}")
    for f in files:
        print(f"  {sha(os.path.join(RUNS, f))}  {f}")

    logs = {}
    for s in SEEDS:
        L = json.load(open(f"{RUNS}/P3_{s}.json"))
        logs[s] = L
        errs = []
        if L["seed"] != s:
            errs.append("seed")
        if L["es"]["phase_gens"] != {"A": 400, "B": 400, "C": 1200,
                                     "D": 400}:
            errs.append(f"phase_gens {L['es']['phase_gens']}")
        if L["total_steps"] != 247680000:
            errs.append(f"total {L['total_steps']}")
        z = np.load(f"{RUNS}/P3_{s}_ckpts.npz")
        if sorted(z.files) != ["init", "post_A", "post_B", "post_C",
                               "post_C400", "post_C800", "post_D"]:
            errs.append(f"ckpts {sorted(z.files)}")
        th = np.load(f"{RUNS}/P3_{s}_theta.npz")["theta"]
        if not np.array_equal(th, z["post_D"]):
            errs.append("theta!=post_D")
        print(f"{s}: {'OK' if not errs else errs}")

    print("\nproduction equivalence vs pilot2_runs:")
    for s in SEEDS:
        z3 = np.load(f"{RUNS}/P3_{s}_ckpts.npz")
        z2 = np.load(f"{PILOT2}/P2_{s}_ckpts.npz")
        sk = 1833
        abd = all(np.array_equal(z3["post_D"][i*sk:(i+1)*sk],
                                 z2["post_D"][i*sk:(i+1)*sk])
                  for i in (0, 1, 3))
        c4 = np.array_equal(z3["post_C400"][2*sk:3*sk],
                            z2["post_C400"][2*sk:3*sk])
        c8 = np.array_equal(z3["post_C800"][2*sk:3*sk],
                            z2["post_C"][2*sk:3*sk])
        print(f"  {s}: A/B/D identical={abd} C@g400 identical={c4} "
              f"C@g800 identical={c8}")

    print("\ntrend Delta = C@g1200 - C@g800 (pure sampled):")
    ds = []
    for s in SEEDS:
        h = logs[s]["heldout_after"]
        d = (h["post_C"]["C"]["sampled"]["sr"]
             - h["post_C800"]["C"]["sampled"]["sr"])
        ds.append(d)
        print(f"  {s}: {h['post_C800']['C']['sampled']['sr']:.3f} -> "
              f"{h['post_C']['C']['sampled']['sr']:.3f}  D={d:+.3f}")
    ds = np.array(ds)
    if (np.abs(ds) <= 0.07).sum() >= 2:
        trend = "PLATEAU"
    elif (ds > 0.07).sum() >= 2:
        trend = "STILL_LEARNING"
    else:
        trend = "INCONCLUSIVE"
    print(f"TREND: {trend}")

    print("\nphase-end pure sampled (gate 1: mean>=0.80):")
    gate1 = True
    for f in ["A", "B", "C", "D"]:
        v = np.array([logs[s]["heldout_after"][f"post_{f}"][f]["sampled"]["sr"]
                      for s in SEEDS])
        m, sd = v.mean(), v.std(ddof=1)
        ci = T32 * sd / sqrt(3)
        ok = m >= 0.80
        gate1 &= ok
        print(f"  {f}: mean={m:.4f} CI=[{max(0, m-ci):.3f},{min(1, m+ci):.3f}] "
              f"vals={list(np.round(v, 3))} {'PASS' if ok else 'FAIL'}")
    print("\nretention drops (gate 2: mean<=0.10; expect exactly 0.0):")
    gate2 = True
    for f in ["A", "B", "C", "D"]:
        v = np.array([logs[s]["retention_sampled"][f]["drop"] for s in SEEDS])
        ok = v.mean() <= 0.10
        gate2 &= ok
        flag = "NONZERO-TRIAGE" if (v != 0).any() else "exact-0.0"
        print(f"  {f}: mean={v.mean():+.4f} vals={list(v)} "
              f"{'PASS' if ok else 'FAIL'} {flag}")
    print(f"\nVERDICT: {'GO' if gate1 and gate2 else 'NO-GO'} "
          f"(gate1={gate1} gate2={gate2}; criterion unchanged)")


if __name__ == "__main__":
    main()
