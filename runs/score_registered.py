"""Post-hoc registered scoring (READ-ONLY on runs_modal/ and data/).

Uses ONLY frozen src/evaluate.py (make_policy / run_battery / subset) plus
final thetas from runs_modal/. Sampled actions (default policy), matching
the frozen retention_eval semantics; seed_xor=0 (battery seeds as frozen).
No thresholds, seeds, or batteries are modified here; verdicts are computed
in a separate step from runs/scores_raw.json.

Work: 20 thetas x 8 batteries (A,B,C,D,I1,H1,H2,H3) mode normal, plus A1
ablations (4 skills x H1-H3, gate-shuffle x H1-H3, hazard-mask x H1/H2/C).
Saves incrementally (resume-safe): finished keys are skipped on relaunch.
"""

import json
import os
import sys
import time

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
OUT = f"{ROOT}/runs/scores_raw.json"

sys.path.insert(0, f"{ROOT}/src")

ARMS = ["A1", "A2", "A3", "A4"]
SEEDS = [101, 102, 103, 104, 105]
BATTERIES = ["A", "B", "C", "D", "I1", "H1", "H2", "H3"]
H_BATT = ["H1", "H2", "H3"]


def main():
    import numpy as np
    from evaluate import make_policy, run_battery

    t0 = time.time()
    entries = json.load(open(f"{ROOT}/data/test_battery_v1.json"))
    by_batt = {b: [e for e in entries if e["battery"] == b] for b in BATTERIES}
    for b, es in by_batt.items():
        assert len(es) == 200, (b, len(es))

    if os.path.isfile(OUT):
        raw = json.load(open(OUT))
    else:
        raw = {"meta": {"policy": "sampled", "seed_xor": 0,
                        "code": "frozen src/evaluate.py",
                        "batteries": {b: len(by_batt[b]) for b in BATTERIES}},
               "main": {}, "ablation": {}, "logs": {}}

    def rep_summary(rep):
        return {"n": rep["n"], "sr": rep["success_rate"], "ci": rep["ci95"],
                "mean_return": sum(r["return"] for r in rep["episodes"])
                / max(rep["n"], 1)}

    def save():
        json.dump(raw, open(OUT, "w"), indent=1, sort_keys=True)

    n_new = 0
    for a in ARMS:
        for s in SEEDS:
            tag = f"{a}_{s}"
            theta = np.load(f"{ROOT}/runs_modal/{tag}_theta.npz")["theta"]
            arch = "mono" if a == "A3" else "modular"
            raw["main"].setdefault(tag, {})
            for b in BATTERIES:
                if b in raw["main"][tag]:
                    continue
                t1 = time.time()
                pol = make_policy(arch, theta, "normal")
                raw["main"][tag][b] = rep_summary(run_battery(pol, by_batt[b]))
                n_new += 1
                save()
                r = raw["main"][tag][b]
                print(f"[{time.time()-t0:7.0f}s] main {tag} {b}: "
                      f"sr={r['sr']:.3f} ({time.time()-t1:.0f}s)", flush=True)
            if a == "A1":
                raw["ablation"].setdefault(tag, {})
                jobs = ([(f"skill-{k}@{h}", f"skill-{k}", h, 0)
                         for k in range(4) for h in H_BATT]
                        + [(f"shuffle@{h}", "shuffle", h, 0) for h in H_BATT]
                        + [(f"mask@{h}", "mask", h, 0)
                           for h in ["H1", "H2", "C"]])
                for key, mode, h, sseed in jobs:
                    if key in raw["ablation"][tag]:
                        continue
                    t1 = time.time()
                    pol = make_policy(arch, theta, mode, sseed)
                    raw["ablation"][tag][key] = rep_summary(
                        run_battery(pol, by_batt[h]))
                    n_new += 1
                    save()
                    r = raw["ablation"][tag][key]
                    print(f"[{time.time()-t0:7.0f}s] abl {tag} {key}: "
                          f"sr={r['sr']:.3f} ({time.time()-t1:.0f}s)",
                          flush=True)
            if tag not in raw["logs"]:
                log = json.load(open(f"{ROOT}/runs_modal/{tag}.json"))
                raw["logs"][tag] = {
                    "acquired": {f: log["phases"][f]["acquired"]
                                 for f in ["A", "B", "C", "D"]},
                    "first_pass_gen": {f: log["phases"][f]["first_pass_gen"]
                                       for f in ["A", "B", "C", "D"]},
                    "phase_end_heldout": {
                        f: (log["retention"][f"after_{f}"][f]["sr"]
                            if log["retention"][f"after_{f}"][f] else None)
                        for f in ["A", "B", "C", "D"]},
                    "after_gate": {
                        f: (log["retention"]["after_gate"][f]["sr"]
                            if log["retention"]["after_gate"][f] else None)
                        for f in ["A", "B", "C", "D"]},
                    "reacquisition": log.get("reacquisition", {})}
                save()
    print(f"SCORING DONE: {n_new} new batteries in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
