"""v2 frozen analysis (run AFTER download; read-only on v2_runs).

1. Integrity: 9/9 files, config, 41.28M steps/seed, ckpts init +
   post_gate, skills bit-frozen, start thetas bit-identical to
   pilot3_runs, heldout hash, hashes.
2. Block (a): absolute I1/each-H (+ G1/G2-heldout informative).
3. Block (b): advantage vs best single skill (per-H max of seed
   means; H-mean reference = mean of the three maxima; +40pp rule;
   feasibility 1.00-ref>=0.40 reported).
4. Block (c): paired gate-vs-control diffs (uniform, shuffle) on
   H-mean, +0.10 rule; per-battery descriptive.
5. Verdicts: Q1 (GO iff I1>=0.70, each H>=0.60, margin holds;
   I1<0.70 -> H UNINTERPRETABLE); Q2 VENTAJA/SIN-VENTAJA per
   control; combined attribution label. Monolith pending always.

Usage: python3.14 v2/score_v2.py [runs_dir]
"""

import hashlib
import json
import os
import sys
from math import sqrt

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..",
                                                          "v2_runs")
PILOT3 = os.path.join(HERE, "..", "pilot3_runs")
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
    exp = {f"V2_{s}{x}" for s in SEEDS
           for x in (".json", "_theta.npz", "_ckpts.npz")}
    print(f"files={len(files)} missing={sorted(exp - set(files)) or 'none'} "
          f"extra={sorted(set(files) - exp) or 'none'}")
    for f in files:
        print(f"  {sha(os.path.join(RUNS, f))}  {f}")

    logs = {}
    for s in SEEDS:
        L = json.load(open(f"{RUNS}/V2_{s}.json"))
        logs[s] = L
        errs = []
        if L["seed"] != s:
            errs.append("seed")
        if L["es"] != {"pop_pairs": 32, "sigma": 0.1, "alpha": 0.05,
                       "fit": "4xG1+4xG2", "gens": 400}:
            errs.append(f"es {L['es']}")
        if L["steps"] != 41280000:
            errs.append(f"steps {L['steps']}")
        if len(L["history"]) != 400 or len(L["checks"]) != 40:
            errs.append("hist/checks")
        if L["shuffle_seed"] != 0:
            errs.append("shuffle_seed")
        z = np.load(f"{RUNS}/V2_{s}_ckpts.npz")
        if sorted(z.files) != ["init", "post_gate"]:
            errs.append(f"ckpts {sorted(z.files)}")
        th = np.load(f"{RUNS}/V2_{s}_theta.npz")["theta"]
        if not np.array_equal(th, z["post_gate"]):
            errs.append("theta!=post_gate")
        if not np.array_equal(z["post_gate"][:4*1833], z["init"][:4*1833]):
            errs.append("SKILLS NOT FROZEN")
        if np.array_equal(z["post_gate"][4*1833:], z["init"][4*1833:]):
            errs.append("gate unchanged?!")
        p3 = np.load(f"{PILOT3}/P3_{s}_theta.npz")["theta"]
        if not np.array_equal(z["init"], p3):
            errs.append("init!=pilot3 post_D")
        print(f"{s}: {'OK' if not errs else errs}")

    def agg(vals):
        v = np.array(vals, dtype=float)
        m, sd = v.mean(), v.std(ddof=1)
        return m, sd, T32 * sd / sqrt(3), list(v)

    print("\n(a) ABSOLUTE (sampled means over seeds):")
    absm = {}
    for b in ["G1H", "G2H", "I1", "H1", "H2", "H3"]:
        key = "normal" if b in ("G1H", "G2H") else "sampled"
        m, sd, ci, v = agg([logs[s]["eval"][b][key]["sr"] for s in SEEDS])
        absm[b] = m
        print(f"  {b}: mean={m:.4f} CI=[{max(0, m-ci):.3f},{min(1, m+ci):.3f}] "
              f"vals={np.round(v, 3)}")

    print("\n(b) VS BEST SINGLE SKILL (+40pp rule):")
    refs = {}
    for b in ["H1", "H2", "H3"]:
        per_skill = [np.mean([logs[s]["eval"][b][f"skill-{k}"]["sr"]
                              for s in SEEDS]) for k in range(4)]
        refs[b] = max(per_skill)
        print(f"  {b}: per-skill seed-means={np.round(per_skill, 3)} "
              f"ref={refs[b]:.4f}")
    ref_mean = float(np.mean([refs[b] for b in ["H1", "H2", "H3"]]))
    hmean = float(np.mean([absm[b] for b in ["H1", "H2", "H3"]]))
    print(f"  H-mean={hmean:.4f} ref-mean={ref_mean:.4f} "
          f"margin={hmean - ref_mean:+.4f} (need >=+0.40) "
          f"feasible={1.00 - ref_mean >= 0.40}")

    print("\n(c) VS CONTROLS (paired gate-control H-mean diffs, +0.10 rule):")
    q2 = {}
    for c in ["uniform", "shuffle"]:
        d = np.mean([[logs[s]["eval"][b]["sampled"]["sr"]
                       - logs[s]["eval"][b][c]["sr"]
                       for b in ["H1", "H2", "H3"]] for s in SEEDS], axis=1)
        m = float(np.mean(d))
        q2[c] = m
        print(f"  {c}: per-seed H-mean diffs={np.round(d, 3)} mean={m:+.4f} "
              f"-> {'VENTAJA' if m >= 0.10 else 'SIN-VENTAJA'}")
        for b in ["H1", "H2", "H3"]:
            dd = [logs[s]["eval"][b]["sampled"]["sr"]
                  - logs[s]["eval"][b][c]["sr"] for s in SEEDS]
            print(f"    {b}: {np.round(dd, 3)}")

    print("\nVERDICTS:")
    i1 = absm["I1"] >= 0.70
    print(f"  I1 {'PASS' if i1 else 'FAIL'}")
    if not i1:
        print("  H: UNINTERPRETABLE (I1 gate-sanity failed)")
        print("  VERDICT: INCONCLUSIVE (criterion unchanged)")
        return
    hpass = all(absm[b] >= 0.60 for b in ["H1", "H2", "H3"])
    marg = (hmean - ref_mean) >= 0.40
    print(f"  each H>=0.60: {'PASS' if hpass else 'FAIL'}")
    print(f"  H-mean>=ref+0.40: {'PASS' if marg else 'FAIL'}")
    go = hpass and marg
    print(f"  Q1: {'PASS' if go else 'FAIL'}")
    att = ("WITH routing advantage" if all(v >= 0.10 for v in q2.values())
           else "WITHOUT attributed routing advantage")
    print(f"  Q2: {att}")
    print(f"  VERDICT: {'GO' if go else 'NO-GO'} (monolith pending always; "
          f"criterion unchanged)")


if __name__ == "__main__":
    main()
