"""pilot2 frozen analysis (run AFTER download; read-only on pilot2_runs).

1. Integrity: 9/9 files, config, gens (A/B/D 400, C 800), steps
   (41.28M x3 + 82.56M), freeze bit-exact, hashes.
2. Gates: mean phase-end pure sampled >= 0.80 per skill; mean
   retention drop <= 0.10; greedy secondary.
3. Paired checks vs pilot_runs (pre-registered): A/B/D slices of
   post_D bit-identical; post_C400[C-slice] == pilot post_C[C-slice];
   post_A/post_B heldouts numerically identical. Mismatches go to the
   §4 triage (diagnose eval/randomness/freeze; no auto-bug label).
4. Verdict GO/NO-GO to composition.

Usage: python3.14 pilot2/score_pilot2.py [runs_dir]
"""

import json
import os
import sys

import numpy as np

RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "pilot2_runs")
PILOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                     "pilot_runs")
SEEDS = [301, 302, 303]

T32 = 4.303


def main():
    from math import sqrt
    import hashlib

    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()[:12]

    files = sorted(os.listdir(RUNS))
    exp = {f"P2_{s}{x}" for s in SEEDS
           for x in (".json", "_theta.npz", "_ckpts.npz")}
    print(f"files={len(files)} missing={sorted(exp - set(files)) or 'none'} "
          f"extra={sorted(set(files) - exp) or 'none'}")
    for f in files:
        print(f"  {sha(os.path.join(RUNS, f))}  {f}")

    logs = {}
    for s in SEEDS:
        L = json.load(open(f"{RUNS}/P2_{s}.json"))
        logs[s] = L
        errs = []
        if L["seed"] != s:
            errs.append("seed")
        if L["es"]["phase_gens"] != {"A": 400, "B": 400, "C": 800,
                                     "D": 400}:
            errs.append(f"phase_gens {L['es']['phase_gens']}")
        if L["total_steps"] != 206400000:
            errs.append(f"total {L['total_steps']}")
        z = np.load(f"{RUNS}/P2_{s}_ckpts.npz")
        if sorted(z.files) != ["init", "post_A", "post_B", "post_C",
                               "post_C400", "post_D"]:
            errs.append(f"ckpts {sorted(z.files)}")
        th = np.load(f"{RUNS}/P2_{s}_theta.npz")["theta"]
        if not np.array_equal(th, z["post_D"]):
            errs.append("theta!=post_D")
        print(f"{s}: {'OK' if not errs else errs}")

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
    print("\npaired vs pilot_runs (pre-registered bit-checks):")
    for s in SEEDS:
        z2 = np.load(f"{RUNS}/P2_{s}_ckpts.npz")
        z1 = np.load(f"{PILOT}/{s}_ckpts.npz")
        sk = 1833
        abd = all(np.array_equal(z2["post_D"][i*sk:(i+1)*sk],
                                 z1["post_D"][i*sk:(i+1)*sk])
                  for i in (0, 1, 3))
        c400 = np.array_equal(z2["post_C400"][2*sk:3*sk],
                              z1["post_C"][2*sk:3*sk])
        print(f"  {s}: A/B/D slices identical={abd} "
              f"C@g400==pilot_post_C={c400}")
    print(f"\nVERDICT: {'GO' if gate1 and gate2 else 'NO-GO'} "
          f"(gate1={gate1} gate2={gate2}; criterion unchanged)")


if __name__ == "__main__":
    main()
