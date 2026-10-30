"""Registered verdicts from runs/scores_raw.json (READ-ONLY inputs).

Pre-committed rules (no changes after seeing results):
- FAIL/N/A key: DRIVER acquired flag (compliance correction defines FAIL via
  record-only checks; PREREG 7 keys N/A on FAIL). Heldout >=80% reported
  alongside; disagreements flagged, never re-gated.
- Thresholds judged on MEANS over seeds (point estimates); 95% CI reported
  as mean +- 1.96*SD/sqrt(5) (codebase normal-approx convention, ddof=1).
- I1 gate on A1 mean. H-main PASS iff interpretable + testable +
  each-H>=60% + Hmean-A3>=15pp + per-H A1-best-skill>=40pp.
- Sensitivity: max over (arm in {A2,A3}, skill) of mean-over-seeds drop
  (phase_end - after_gate, log retention) >= 20pp.
- H-iso: INCONCLUSIVE unless sensitivity met; then the success term is the
  preregistered word "measurably": PASS iff the A1-vs-A2 pooled drop
  difference is CI-distinguishable from noise (diff - 1.96*SE > 0, using
  the mandated 95% CIs; skills with data in both arms). No numeric margin
  invented: measurability IS the preregistered bar. Else FAIL.
- H-meta: A1-A4 H-mean diff >= 10pp on non-N/A cells; n=0 -> UNTESTABLE.
  The "adversarial-subset" disjunct is undefined in PREREG -> not evaluable.
- Reacquisition: per event post>=0.80 AND savings>=2x.
- N/A cells excluded, never zero; aggregates report n.
Writes runs/verdict.json; prints the full report.
"""

import json
import math
import sys

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
ARMS = ["A1", "A2", "A3", "A4"]
SEEDS = [101, 102, 103, 104, 105]
SKILLS = ["A", "B", "C", "D"]
HBATT = ["H1", "H2", "H3"]
NEEDS = {"H1": {"B", "C"}, "H2": {"A", "B", "C"}, "H3": {"C", "D"}}


def mean_ci(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, 1.96 * math.sqrt(var / n)


def main():
    raw = json.load(open(f"{ROOT}/runs/scores_raw.json"))
    out = {"rules": "see script header", "per_seed": {}, "na": {},
           "aggregates": {}, "criteria": {}, "hypotheses": {}}
    L = ["=" * 72, "REGISTERED SCORING REPORT (frozen code, sampled policies)",
         "=" * 72]

    # ---- per-seed table + N/A ----
    L.append("\n-- per (arm,seed): final full-net sr by battery (* = N/A) --")
    L.append("arm-seed | " + " ".join(f"{b:>7}" for b in
                                      ["A", "B", "C", "D", "I1", "H1", "H2", "H3"]))
    na_map = {}
    for a in ARMS:
        for s in SEEDS:
            tag = f"{a}_{s}"
            acq = {f for f in SKILLS if raw["logs"][tag]["acquired"][f]}
            blocked = not ({"A", "B"} <= acq) or not ({"A", "C"} <= acq)
            row, nar = {}, {}
            for b in ["A", "B", "C", "D", "I1", "H1", "H2", "H3"]:
                sr = raw["main"][tag][b]["sr"]
                is_na = b in NEEDS and (blocked or not (NEEDS[b] <= acq))
                nar[b] = is_na
                row[b] = sr
            na_map[tag] = nar
            out["per_seed"][tag] = row
            out["na"][tag] = {b: nar[b] for b in HBATT}
            L.append(f"{tag:>7} | " + " ".join(
                f"{row[b]:7.3f}" + ("*" if nar[b] else " ") for b in
                ["A", "B", "C", "D", "I1", "H1", "H2", "H3"]))
    out["na_summary"] = {
        h: sum(1 for t in na_map for b in [h] if na_map[t][b]) for h in HBATT}

    # ---- acquisition: flag + phase-end heldout ----
    L.append("\n-- acquisition per skill: driver flag / phase-end heldout sr --")
    acq_rows = {}
    for a in ARMS:
        for f in SKILLS:
            flags = [raw["logs"][f"{a}_{s}"]["acquired"][f] for s in SEEDS]
            hs = [raw["logs"][f"{a}_{s}"]["phase_end_heldout"][f] for s in SEEDS]
            hv = [x for x in hs if x is not None]
            m, c = mean_ci(hv) if hv else (None, None)
            acq_rows[f"{a}_{f}"] = {"flags": flags, "heldout": hs,
                                    "mean": m, "ci": c}
            L.append(f"{a} {f}: flags={['1' if x else '0' for x in flags]} "
                     f"heldout={[f'{x:.3f}' if x is not None else 'n/e' for x in hs]}"
                     + (f" mean={m:.3f}+-{c:.3f}" if m is not None else ""))
    out["aggregates"]["acquisition"] = acq_rows

    # ---- H aggregates over non-N/A cells ----
    L.append("\n-- H means over NON-N/A cells (n shown) --")
    hmeans = {}
    for a in ARMS:
        hmeans[a] = {}
        for h in HBATT:
            xs = [raw["main"][f"{a}_{s}"][h]["sr"] for s in SEEDS
                  if not na_map[f"{a}_{s}"][h]]
            hmeans[a][h] = {"n": len(xs),
                            **dict(zip(("mean", "ci"),
                                       mean_ci(xs) if xs else (None, None)))}
            v = hmeans[a][h]
            L.append(f"{a} {h}: n={v['n']}" + (f" mean={v['mean']:.3f}+-{v['ci']:.3f}"
                                               if v["n"] else " (ALL N/A)"))
    hmean_all = {}
    for a in ARMS:
        xs = [raw["main"][f"{a}_{s}"][h]["sr"] for s in SEEDS for h in HBATT
              if not na_map[f"{a}_{s}"][h]]
        hmean_all[a] = {"n": len(xs), **dict(zip(("mean", "ci"),
                                                 mean_ci(xs) if xs else (None, None)))}
    out["aggregates"]["H_means"] = hmeans
    out["aggregates"]["H_mean_all"] = hmean_all

    # ---- I1 ----
    L.append("\n-- I1 means (all arms, descriptive except A1 gate) --")
    i1 = {}
    for a in ARMS:
        xs = [raw["main"][f"{a}_{s}"]["I1"]["sr"] for s in SEEDS]
        m, c = mean_ci(xs)
        i1[a] = {"mean": m, "ci": c, "seeds": xs}
        L.append(f"{a}: mean={m:.3f}+-{c:.3f} seeds={[f'{x:.3f}' for x in xs]}")
    out["aggregates"]["I1"] = i1

    # ---- single-skill / shuffle / mask (A1) ----
    L.append("\n-- A1 ablations: mean over seeds (descriptive; H is N/A) --")
    abl = {}
    for h in HBATT:
        for k in range(4):
            xs = [raw["ablation"][f"A1_{s}"][f"skill-{k}@{h}"]["sr"] for s in SEEDS]
            m, c = mean_ci(xs)
            abl[f"skill-{k}@{h}"] = {"mean": m, "ci": c}
            L.append(f"skill-{k}@{h}: {m:.3f}+-{c:.3f}")
    for h in HBATT:
        xs = [raw["ablation"][f"A1_{s}"][f"shuffle@{h}"]["sr"] for s in SEEDS]
        m, c = mean_ci(xs)
        abl[f"shuffle@{h}"] = {"mean": m, "ci": c}
        L.append(f"shuffle@{h}: {m:.3f}+-{c:.3f}")
    for h in ["H1", "H2", "C"]:
        xs = [raw["ablation"][f"A1_{s}"][f"mask@{h}"]["sr"] for s in SEEDS]
        m, c = mean_ci(xs)
        abl[f"mask@{h}"] = {"mean": m, "ci": c}
        L.append(f"mask@{h}: {m:.3f}+-{c:.3f}")
    out["aggregates"]["ablations_A1"] = abl

    # ---- retention drops from logs ----
    L.append("\n-- retention drops (phase_end - after_gate, log srs) --")
    drops = {}
    for a in ARMS:
        drops[a] = {}
        for f in SKILLS:
            ds = []
            for s in SEEDS:
                tag = f"{a}_{s}"
                pre = raw["logs"][tag]["phase_end_heldout"][f]
                post = raw["logs"][tag]["after_gate"][f]
                if pre is not None and post is not None:
                    ds.append(pre - post)
            m, c = mean_ci(ds) if ds else (None, None)
            drops[a][f] = {"n": len(ds), "mean": m, "ci": c, "seeds": ds}
            L.append(f"{a} {f}: n={len(ds)}" + (f" drop={m:+.3f}+-{c:.3f}" if ds else ""))
    out["aggregates"]["drops"] = drops

    # ---- criteria ----
    crit = {}
    # retention A1 <= 10pp
    a1d = [d for f in SKILLS for d in drops["A1"][f]["seeds"]]
    m, c = mean_ci(a1d)
    crit["retention_A1"] = {"mean_drop": m, "ci": c, "bar": 0.10,
                            "met": bool(m <= 0.10)}
    # sensitivity
    sens = {(a, f): drops[a][f]["mean"] for a in ["A2", "A3"] for f in SKILLS
            if drops[a][f]["mean"] is not None}
    smax = max(sens.values())
    crit["sensitivity"] = {"max_mean_drop": smax, "where": max(sens, key=sens.get),
                           "bar": 0.20, "met": bool(smax >= 0.20)}
    # reacquisition
    evs = []
    for a in ARMS:
        for s in SEEDS:
            for fam, r in raw["logs"][f"{a}_{s}"]["reacquisition"].items():
                ok_thr = r["post"] >= 0.80
                ok_sav = (r["savings"] is not None and r["savings"] >= 2.0)
                evs.append({"job": f"{a}_{s}", "fam": fam, **r,
                            "restore_ok": ok_thr, "savings_ok": ok_sav,
                            "pass": bool(ok_thr and ok_sav)})
    crit["reacquisition"] = {"events": evs, "n": len(evs),
                             "all_pass": bool(evs) and all(e["pass"] for e in evs)}
    L.append(f"\nretention A1: drop={m:+.3f}+-{c:.3f} bar<=0.10 met={crit['retention_A1']['met']} (structural)")
    L.append(f"sensitivity: max={smax:+.3f} at {crit['sensitivity']['where']} bar>=0.20 met={crit['sensitivity']['met']}")
    L.append(f"reacquisition: {len(evs)} events, all_pass={crit['reacquisition']['all_pass']}")
    for e in evs:
        L.append(f"  {e['job']} {e['fam']}: pre={e['pre']:.3f} dropped={e['dropped']:.3f} "
                 f"post={e['post']:.3f} gens={e['gens']} scratch={e['scratch_first_pass']} "
                 f"savings={e['savings']} pass={e['pass']}")
    out["criteria"] = crit

    # ---- hypotheses ----
    hyp = {}
    testable_h = any(hmean_all["A1"]["n"] > 0 for _ in [0]) and any(
        hmeans["A1"][h]["n"] > 0 for h in HBATT)
    interp = i1["A1"]["mean"] >= 0.70
    if not testable_h:
        hyp["H-main"] = "INCONCLUSIVE (UNTESTABLE: all A1 H cells N/A per PREREG 7)"
    elif not interp:
        hyp["H-main"] = "INCONCLUSIVE (UNINTERPRETABLE: A1 I1 mean < 0.70)"
    else:
        bars = [hmeans["A1"][h]["mean"] >= 0.60 for h in HBATT]
        d15 = hmean_all["A1"]["mean"] - hmean_all["A3"]["mean"] >= 0.15
        d40 = all(hmeans["A1"][h]["mean"] - max(
            abl[f"skill-{k}@{h}"]["mean"] for k in range(4)) >= 0.40 for h in HBATT)
        hyp["H-main"] = ("PASS" if all(bars) and d15 and d40 else
                         f"FAIL (60%={bars} +15pp={d15} +40pp={d40})")
        hyp["H-main-detail"] = {"each60": bars, "plus15": d15, "plus40": d40}
    if not interp:
        hyp["H-main"] += "; also UNINTERPRETABLE (I1 %.3f < 0.70)" % i1["A1"]["mean"]
    if crit["sensitivity"]["met"]:
        a1 = [d for f in SKILLS for d in drops["A1"][f]["seeds"]]
        a2 = [d for f in SKILLS for d in drops["A2"][f]["seeds"]
              if drops["A1"][f]["n"] > 0]
        m1, _ = mean_ci(a1)
        m2, _ = mean_ci(a2)
        import statistics as _st

        def _se(xs):
            return _st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0
        diff, se = (m2 - m1), math.sqrt(_se(a1) ** 2 + _se(a2) ** 2)
        lo = diff - 1.96 * se
        hyp["H-iso"] = ("PASS (measurable: A2-A1 diff "
                        f"{diff:+.3f}, 95% CI excludes 0)" if lo > 0 else
                        "FAIL (no measurable reduction: A2-A1 diff "
                        f"{diff:+.3f}+-{1.96*se:.3f}, CI includes 0; "
                        "sensitivity held only via non-ablation arm A3/B)")
        hyp["H-iso-detail"] = {"A1": m1, "A2": m2, "diff": diff,
                               "diff_ci": 1.96 * se}
    else:
        hyp["H-iso"] = "INCONCLUSIVE (sensitivity unmet)"
    n1 = hmean_all["A1"]["n"]
    n4 = hmean_all["A4"]["n"]
    if n1 and n4:
        d = hmean_all["A1"]["mean"] - hmean_all["A4"]["mean"]
        hyp["H-meta"] = f"{'PASS' if d >= 0.10 else 'FAIL'} (A1-A4={d:+.3f}, bar 0.10)"
    else:
        hyp["H-meta"] = (f"INCONCLUSIVE (UNTESTABLE: H N/A in A1 n={n1}, A4 n={n4}; "
                         "adversarial-subset disjunct undefined in PREREG)")
    L.append("\n-- VERDICTS --")
    for h, v in hyp.items():
        if not h.endswith("detail"):
            L.append(f"{h}: {v}")
    out["hypotheses"] = hyp

    json.dump(out, open(f"{ROOT}/runs/verdict.json", "w"), indent=1)
    print("\n".join(L))
    print("verdict.json written")


if __name__ == "__main__":
    main()
