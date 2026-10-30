"""B diagnostic audit, lighter (diag-only, READ-ONLY on runs_modal/ + data/).

1. Baselines (random/still/flee/priv-potential/orbit) on frozen B battery.
2. v1 final one-hot B skills (A1/A2/A4 x seeds) on test-200 AND train-400;
   A3 final whole-net on test-200 (instrumented detail).
3. Return-vs-success scatter dataset (fitness/success alignment test).
4. v1 B curves from logs: checks + fitness-history trends.
Writes diag/audit_b.json. No thresholds, no verdicts, no v1 changes.
"""

import json
import sys
import time

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
sys.path.insert(0, f"{ROOT}/diag")
sys.path.insert(0, f"{ROOT}/src")

import numpy as np

from evaluate import make_policy
from instrument import run_episode, summarize
from oracles import (random_policy, still_policy, b_oracle_action,
                     b_priv_action, b_obs_orbit_action)


def eval_policy(entries, policy_fn, tag):
    eps = []
    for e in entries:
        sd = int(e["seed"]) if "seed" in e else 900000 + int(e["idx"])
        eps.append(run_episode("B", e["cfg"], sd, policy_fn,
                               np.random.default_rng(sd)))
    s = summarize(eps)
    print(f"{tag}: sr={s['sr']:.3f} ret={s['mean_return']:.1f} "
          f"T={s['mean_T']:.0f} gate={s.get('gate_frac_mean', 0):.2f} "
          f"hazmin={s.get('haz_min_mean', 0):.1f}", flush=True)
    return s, eps


def main():
    t0 = time.time()
    out = {}
    entries = json.load(open(f"{ROOT}/data/test_battery_v1.json"))
    testB = [e for e in entries if e["battery"] == "B"]
    train = json.load(open(f"{ROOT}/data/train_meta_v1.json"))
    trainB = [dict(e, idx=i) for i, e in enumerate(train) if e["family"] == "B"]
    print(f"testB={len(testB)} trainB={len(trainB)}", flush=True)

    base, scat = {}, []
    for name, pol in [("random", random_policy), ("still", still_policy),
                      ("flee", b_oracle_action), ("priv", b_priv_action),
                      ("orbit", b_obs_orbit_action)]:
        s, eps = eval_policy(testB, pol, f"base/{name}")
        base[name] = s
        scat += [(e["return"], int(e["success"]), name) for e in eps]
    out["baselines"] = base

    onehot, curves = {}, {}
    for a in ["A1", "A2", "A4"]:
        for s in [101, 102, 103, 104, 105]:
            tag = f"{a}_{s}"
            th = np.load(f"{ROOT}/runs_modal/{tag}_theta.npz")["theta"]
            pol = make_policy("modular", th, "skill-1")
            adap = lambda o, r, g, p=pol: p(o, g)
            st, epst = eval_policy(testB, adap, f"v1/{tag}/test")
            str_, _ = eval_policy(trainB, adap, f"v1/{tag}/train400")
            onehot[tag] = {"test": st, "train": str_}
            scat += [(e["return"], int(e["success"]), tag) for e in epst]
            log = json.load(open(f"{ROOT}/runs_modal/{tag}.json"))
            ch = log["phases"]["B"]["checks"]
            hi = log["phases"]["B"]["history"]
            curves[tag] = {"checks": ch, "hist_first10": float(np.mean(hi[:10])),
                           "hist_last10": float(np.mean(hi[-10:])),
                           "hist_max": float(max(hi))}
    a3 = {}
    for s in [101, 102, 103, 104, 105]:
        tag = f"A3_{s}"
        th = np.load(f"{ROOT}/runs_modal/{tag}_theta.npz")["theta"]
        pol = make_policy("mono", th, "normal")
        adap = lambda o, r, g, p=pol: p(o, g)
        st, epst = eval_policy(testB, adap, f"v1/{tag}/final-net")
        a3[tag] = st
        scat += [(e["return"], int(e["success"]), tag) for e in epst]
        log = json.load(open(f"{ROOT}/runs_modal/{tag}.json"))
        ch = log["phases"]["B"]["checks"]
        hi = log["phases"]["B"]["history"]
        curves[tag] = {"checks": ch, "hist_first10": float(np.mean(hi[:10])),
                       "hist_last10": float(np.mean(hi[-10:])),
                       "hist_max": float(max(hi))}
    out["onehot_final"] = onehot
    out["a3_final_net"] = a3
    out["v1_curves"] = curves
    out["scatter"] = scat
    json.dump(out, open(f"{ROOT}/diag/audit_b.json", "w"), indent=1)
    print(f"AUDIT-B DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
