"""v2conf frozen analysis (run AFTER download; read-only on runs).

1. Integrity: 9/9, config (2+2+4 pools), 41.28M/seed, ckpts, skills
   frozen, init==pilot3 bit-exact, dataset shas, hashes.
2. PRIMARY acquisition: ACQ mean>=0.70 else H UNINTERPRETABLE.
3. B contribution (separate): normal-minus-ablated paired drop on
   ACQ per seed/mean/CI; >=0.10 mean => practical contribution (new
   criterion; NOT absolute-necessity proof). Gate weights descriptive.
4. H transfer (previously agreed gates): each>=0.60, H-mean>=ref+0.40
   (feasibility 1.00-ref>=0.40 kept), Q2 +0.10.
5. v2views reference on SAME new batteries (measurement-only);
   paired deltas descriptive (improvement vs gates separated).
6. G1H/G2H + single-skill ACQ: diagnostics.

Usage: python3.14 v2conf/score_vconf.py [runs_dir]
"""

import hashlib
import json
import os
import sys
from math import sqrt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "v2views"))
RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..",
                                                          "v2conf_runs")
VVRUNS = os.path.join(HERE, "..", "v2views_runs")
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
    from evaluate import run_battery
    import views_train as V

    files = sorted(os.listdir(RUNS))
    exp = {f"VC_{s}{x}" for s in SEEDS
           for x in (".json", "_theta.npz", "_ckpts.npz")}
    print(f"files={len(files)} missing={sorted(exp - set(files)) or 'none'} "
          f"extra={sorted(set(files) - exp) or 'none'}")
    for f in files:
        print(f"  {sha(os.path.join(RUNS, f))}  {f}")

    logs = {}
    for s in SEEDS:
        L = json.load(open(f"{RUNS}/VC_{s}.json"))
        logs[s] = L
        errs = []
        if L["seed"] != s:
            errs.append("seed")
        if L["es"] != {"pop_pairs": 32, "sigma": 0.1, "alpha": 0.05,
                       "fit": "2xG1+2xG1conf+4xG2", "gens": 400}:
            errs.append(f"es {L['es']}")
        if L["fitness_pools"] != [["g1orig", 2], ["g1conf", 2], ["g2", 4]]:
            errs.append("pools")
        if L["steps"] != 41280000:
            errs.append(f"steps {L['steps']}")
        if len(L["history"]) != 400 or len(L["checks"]) != 40:
            errs.append("hist/checks")
        z = np.load(f"{RUNS}/VC_{s}_ckpts.npz")
        if sorted(z.files) != ["init", "post_gate"]:
            errs.append(f"ckpts {sorted(z.files)}")
        th = np.load(f"{RUNS}/VC_{s}_theta.npz")["theta"]
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

    print("\nPRIMARY acquisition ACQ (gate: mean>=0.70):")
    m, sd, ci, v = agg([logs[s]["eval"]["ACQ"]["normal"]["sr"] for s in SEEDS])
    acq_pass = m >= 0.70
    print(f"  mean={m:.4f} CI=[{max(0, m-ci):.3f},{min(1, m+ci):.3f}] "
          f"vals={np.round(v, 3)} -> {'PASS' if acq_pass else 'FAIL'}")
    print("  ACQ single-skills (diagnostic: any alone solve conflict?):")
    for k in range(4):
        vv = [logs[s]["eval"]["ACQ"][f"skill-{k}"]["sr"] for s in SEEDS]
        print(f"    skill-{k}: {np.round(vv, 3)}")
    print("  ACQ gate weights (descriptive):")
    for s in SEEDS:
        d = logs[s]["eval"]["ACQ"]["normal"]
        print(f"    {s}: mean={d['w_mean']} bands=" +
              str({k: v["w"] for k, v in d["w_by_band"].items()}))

    print("\nB CONTRIBUTION (paired normal-ablated on ACQ; >=0.10 mean):")
    d = np.array([logs[s]["eval"]["ACQ"]["normal"]["sr"]
                  - logs[s]["eval"]["ACQ"]["ablated-B"]["sr"] for s in SEEDS])
    m, sd, ci, v = agg(d)
    print(f"  per-seed={np.round(v, 3)} mean={m:+.4f} "
          f"CI=[{m-ci:+.3f},{m+ci:+.3f}] -> "
          f"{'PRACTICAL-CONTRIBUTION' if m >= 0.10 else 'BELOW-0.10'} "
          f"(new criterion; not necessity proof)")

    print("\nH TRANSFER (previously agreed gates):")
    absm = {}
    for b in ["H1C", "H2C", "H3C"]:
        m, sd, ci, v = agg([logs[s]["eval"][b]["normal"]["sr"] for s in SEEDS])
        absm[b] = m
        print(f"  {b}: mean={m:.4f} CI=[{max(0, m-ci):.3f},{min(1, m+ci):.3f}] "
              f"vals={np.round(v, 3)}")
    refs = {}
    for b in ["H1C", "H2C", "H3C"]:
        per_skill = [np.mean([logs[s]["eval"][b][f"skill-{k}"]["sr"]
                              for s in SEEDS]) for k in range(4)]
        refs[b] = max(per_skill)
        print(f"  {b} ref: per-skill={np.round(per_skill, 3)} ref={refs[b]:.4f}")
    ref_mean = float(np.mean([refs[b] for b in ["H1C", "H2C", "H3C"]]))
    hmean = float(np.mean([absm[b] for b in ["H1C", "H2C", "H3C"]]))
    print(f"  H-mean={hmean:.4f} ref={ref_mean:.4f} margin={hmean-ref_mean:+.4f} "
          f"(need >=+0.40) feasible={1.00 - ref_mean >= 0.40}")
    q2 = {}
    for c in ["uniform", "shuffle"]:
        dd = np.mean([[logs[s]["eval"][b]["normal"]["sr"]
                        - logs[s]["eval"][b][c]["sr"]
                        for b in ["H1C", "H2C", "H3C"]] for s in SEEDS], axis=1)
        m = float(np.mean(dd))
        q2[c] = m
        print(f"  {c}: per-seed={np.round(dd, 3)} mean={m:+.4f} -> "
              f"{'VENTAJA' if m >= 0.10 else 'SIN-VENTAJA'}")

    print("\nv2views REFERENCE on same new batteries (measurement-only):")
    evc = json.load(open(os.path.join(HERE, "eval_conf_v1.json")))
    for b in ["ACQ", "H1C", "H2C", "H3C"]:
        pool = [e for e in evc if e["battery"] == b]
        dd = []
        for s in SEEDS:
            th = np.load(f"{VVRUNS}/VV_{s}_theta.npz")["theta"]
            r = run_battery(V.policy_fn_for(th, "normal"), pool)["success_rate"]
            dd.append(logs[s]["eval"][b]["normal"]["sr"] - r)
        print(f"  {b}: v2views-ref=" +
              str([round(logs[s]["eval"][b]["normal"]["sr"] - x, 3)
                   for s, x in zip(SEEDS, dd)]) +
              f" paired-conf-minus-ref={np.round(dd, 3)} "
              f"mean={np.mean(dd):+.4f} (improvement, NOT a gate)")

    print("\nG1H/G2H (diagnostic):")
    for b in ["G1H", "G2H"]:
        print(f"  {b}: {[logs[s]['eval'][b]['normal']['sr'] for s in SEEDS]}")

    print("\nVERDICTS (criterion unchanged):")
    if not acq_pass:
        print("  H: UNINTERPRETABLE (acquisition gate failed)")
        print("  VERDICT: PRIMARY-FAIL (+ H uninterpretable)")
        return
    hpass = all(absm[b] >= 0.60 for b in ["H1C", "H2C", "H3C"])
    marg = (hmean - ref_mean) >= 0.40
    att = ("WITH routing advantage" if all(v >= 0.10 for v in q2.values())
           else "WITHOUT attributed routing advantage")
    print(f"  acquisition: PASS; H>=0.60: {'PASS' if hpass else 'FAIL'}; "
          f"margin: {'PASS' if marg else 'FAIL'}; Q2: {att}")
    print(f"  VERDICT: {'GO' if hpass and marg else 'NO-GO'} "
          f"(monolith pending)")


if __name__ == "__main__":
    main()
