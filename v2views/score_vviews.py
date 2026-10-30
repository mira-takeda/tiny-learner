"""v2views frozen analysis (run AFTER download; read-only on runs).

1. Integrity: 9/9, config, 41.28M/seed, ckpts, skills frozen,
   init==pilot3 bit-exact, battery shas, hashes.
2. NEW battery verdicts: Q1 absolutes + best-single margin (with
   feasibility 1.00-ref>=0.40); Q2 paired vs uniform/shuffle.
3. v2-original reference: v2_runs thetas (UNVIEWED, frozen fns,
   no retraining) scored on the SAME new battery; paired deltas
   views-minus-v2. System improvement reported SEPARATELY from
   gate compliance.
4. OLD batteries + G1/G2-heldout: descriptive diagnostics only.

Usage: python3.14 v2views/score_vviews.py [runs_dir]
"""

import hashlib
import json
import os
import sys
from math import sqrt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "v2"))
RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..",
                                                          "v2views_runs")
V2RUNS = os.path.join(HERE, "..", "v2_runs")
PILOT3 = os.path.join(HERE, "..", "pilot3_runs")
SEEDS = [301, 302, 303]
NEW = ["I1N", "H1N", "H2N", "H3N"]
T32 = 4.303


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:12]


def main():
    import numpy as np
    from evaluate import run_battery, make_policy
    import gate_train as GREF

    files = sorted(os.listdir(RUNS))
    exp = {f"VV_{s}{x}" for s in SEEDS
           for x in (".json", "_theta.npz", "_ckpts.npz")}
    print(f"files={len(files)} missing={sorted(exp - set(files)) or 'none'} "
          f"extra={sorted(set(files) - exp) or 'none'}")
    for f in files:
        print(f"  {sha(os.path.join(RUNS, f))}  {f}")

    logs = {}
    for s in SEEDS:
        L = json.load(open(f"{RUNS}/VV_{s}.json"))
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
        z = np.load(f"{RUNS}/VV_{s}_ckpts.npz")
        if sorted(z.files) != ["init", "post_gate"]:
            errs.append(f"ckpts {sorted(z.files)}")
        th = np.load(f"{RUNS}/VV_{s}_theta.npz")["theta"]
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

    newb = json.load(open(os.path.join(HERE, "eval_new_v1.json")))
    pools = {b: [e for e in newb if e["battery"] == b] for b in NEW}

    print("\n(a) NEW-BATTERY ABSOLUTE (sampled=normal means):")
    absm = {}
    for b in NEW:
        m, sd, ci, v = agg([logs[s]["eval"][b]["normal"]["sr"] for s in SEEDS])
        absm[b] = m
        print(f"  {b}: mean={m:.4f} CI=[{max(0, m-ci):.3f},{min(1, m+ci):.3f}] "
              f"vals={np.round(v, 3)}")

    print("\n(b) VS BEST SINGLE SKILL, new H (+40pp; feasibility kept):")
    refs = {}
    for b in ["H1N", "H2N", "H3N"]:
        per_skill = [np.mean([logs[s]["eval"][b][f"skill-{k}"]["sr"]
                              for s in SEEDS]) for k in range(4)]
        refs[b] = max(per_skill)
        print(f"  {b}: per-skill={np.round(per_skill, 3)} ref={refs[b]:.4f}")
    ref_mean = float(np.mean([refs[b] for b in ["H1N", "H2N", "H3N"]]))
    hmean = float(np.mean([absm[b] for b in ["H1N", "H2N", "H3N"]]))
    print(f"  H-mean={hmean:.4f} ref={ref_mean:.4f} margin={hmean-ref_mean:+.4f} "
          f"(need >=+0.40) feasible={1.00 - ref_mean >= 0.40}")

    print("\n(c) VS CONTROLS new H (paired diffs, +0.10):")
    q2 = {}
    for c in ["uniform", "shuffle"]:
        d = np.mean([[logs[s]["eval"][b]["normal"]["sr"]
                       - logs[s]["eval"][b][c]["sr"]
                       for b in ["H1N", "H2N", "H3N"]] for s in SEEDS], axis=1)
        m = float(np.mean(d))
        q2[c] = m
        print(f"  {c}: per-seed={np.round(d, 3)} mean={m:+.4f} -> "
              f"{'VENTAJA' if m >= 0.10 else 'SIN-VENTAJA'}")

    print("\n(d) v2-ORIGINAL reference on SAME new battery (unviewed, "
          "measurement-only; behavior preserved):")
    ref = {}
    for s in SEEDS:
        th = np.load(f"{V2RUNS}/V2_{s}_theta.npz")["theta"]
        ref[s] = {}
        for b in NEW:
            pool = pools[b]
            r = {}
            for m, pol in [
                    ("normal", make_policy("modular", th, "normal")),
                    ("uniform", GREF.uniform_policy(th)),
                    ("shuffle", make_policy("modular", th, "shuffle", 0))] + [
                    (f"skill-{k}", make_policy("modular", th, f"skill-{k}"))
                    for k in range(4)]:
                rep = run_battery(pol, pool)
                r[m] = rep["success_rate"]
            ref[s][b] = r
        print(f"  {s}: " + " ".join(
            f"{b}={ref[s][b]['normal']:.3f}" for b in NEW))
    print("  paired views-minus-v2 (sampled, per seed):")
    for b in NEW:
        dd = [logs[s]["eval"][b]["normal"]["sr"] - ref[s][b]["normal"]
              for s in SEEDS]
        print(f"    {b}: {np.round(dd, 3)} mean={np.mean(dd):+.4f} "
              f"(improvement, NOT a gate)")

    print("\n(e) OLD batteries + G1/G2 (diagnostic, non-verdict):")
    for b in ["G1H", "G2H"]:
        v = [logs[s]["eval"][b]["normal"]["sr"] for s in SEEDS]
        print(f"  {b}: {np.round(v, 3)}")
    for b in ["I1", "H1", "H2", "H3"]:
        v = [logs[s]["eval"]["OLD_" + b]["normal"]["sr"] for s in SEEDS]
        print(f"  OLD_{b}: {np.round(v, 3)}")

    print("\nVERDICTS (new battery, criterion unchanged):")
    i1 = absm["I1N"] >= 0.70
    print(f"  I1N {'PASS' if i1 else 'FAIL'}")
    if not i1:
        print("  H: UNINTERPRETABLE")
        print("  VERDICT: INCONCLUSIVE")
        return
    hpass = all(absm[b] >= 0.60 for b in ["H1N", "H2N", "H3N"])
    marg = (hmean - ref_mean) >= 0.40
    go = hpass and marg
    print(f"  each H>=0.60: {'PASS' if hpass else 'FAIL'}")
    print(f"  margin: {'PASS' if marg else 'FAIL'}")
    att = ("WITH routing advantage" if all(v >= 0.10 for v in q2.values())
           else "WITHOUT attributed routing advantage")
    print(f"  Q1: {'PASS' if go else 'FAIL'}; Q2: {att}")
    print(f"  VERDICT: {'GO' if go else 'NO-GO'} (monolith pending)")


if __name__ == "__main__":
    main()
