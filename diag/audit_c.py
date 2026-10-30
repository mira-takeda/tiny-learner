"""C diagnostic audit (diag-only, READ-ONLY on runs_modal/ + data/).

1. Baselines (random/still/scripted priv+obs) on frozen C battery, instrumented.
2. Cross-check: instrumented loop == frozen run_battery sr (same seeds/policy).
3. Beam-search oracle on a 24-ep oz0-stratified sample (achievability bound).
4. v1 final one-hot C skills (A1/A2/A4 x seeds) on test-200 AND train-400.
5. v1 C curves from logs: checks + fitness-history trends.
6. Reward/success (mis)alignment: return vs success/dwell/transport.
Writes diag/audit_c.json. No thresholds, no verdicts, no v1 changes.
"""

import json
import sys
import time

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
sys.path.insert(0, f"{ROOT}/diag")
sys.path.insert(0, f"{ROOT}/src")

import numpy as np

from evaluate import make_policy, run_battery
from instrument import run_episode, summarize
from oracles import (random_policy, still_policy, c_privileged_action,
                     c_obs_action)


def eval_policy(entries, policy_fn, tag, fam="C"):
    eps = []
    for e in entries:
        sd = int(e["seed"]) if "seed" in e else 900000 + int(e["idx"])
        eps.append(run_episode(fam, e["cfg"], sd, policy_fn,
                               np.random.default_rng(sd)))
    s = summarize(eps)
    print(f"{tag}: sr={s['sr']:.3f} ret={s['mean_return']:.1f} "
          f"T={s['mean_T']:.0f} contact={s.get('contact_ever_frac', 0):.2f} "
          f"ozmin={s.get('oz_min_mean', 0):.1f} dwell={s.get('dwell_max_mean', 0):.1f}",
          flush=True)
    return s, eps


def main():
    t0 = time.time()
    out = {}
    entries = json.load(open(f"{ROOT}/data/test_battery_v1.json"))
    testC = [e for e in entries if e["battery"] == "C"]
    train = json.load(open(f"{ROOT}/data/train_meta_v1.json"))
    trainC = [dict(e, idx=i) for i, e in enumerate(train) if e["family"] == "C"]
    print(f"testC={len(testC)} trainC={len(trainC)}", flush=True)

    base = {}
    for name, pol in [("random", random_policy), ("still", still_policy),
                      ("priv", c_privileged_action), ("obs", c_obs_action)]:
        s, _ = eval_policy(testC, pol, f"base/{name}")
        base[name] = s
    out["baselines"] = base

    sub = testC[:20]
    ref = run_battery(lambda o, g: random_policy(o, None, g), sub)
    s20, _ = eval_policy(sub, random_policy, "xcheck/random20")
    assert abs(ref["success_rate"] - s20["sr"]) < 1e-9
    print("XCHECK instrumented==frozen: OK", flush=True)

    import beam as B
    oz0 = [(float(np.linalg.norm(np.array(e["cfg"]["obj"]["pos"])
                 - np.array(e["cfg"]["zone"]["center"]))), e) for e in testC]
    oz0.sort()
    q = len(oz0) // 4
    sample = []
    for i in range(4):
        sample += [e for _, e in oz0[i * q:(i + 1) * q]][:: q // 6][:6]
    print(f"beam sample oz0: {[round(o,0) for o,_ in oz0[::33]]}", flush=True)
    out["beam24"] = B.run_beam_battery("C", sample, width=48)
    print(f"beam24: {out['beam24']}", flush=True)

    onehot, curves = {}, {}
    for a in ["A1", "A2", "A4"]:
        for s in [101, 102, 103, 104, 105]:
            tag = f"{a}_{s}"
            th = np.load(f"{ROOT}/runs_modal/{tag}_theta.npz")["theta"]
            pol = make_policy("modular", th, "skill-2")
            adap = lambda o, r, g, p=pol: p(o, g)
            st, _ = eval_policy(testC, adap, f"v1/{tag}/test")
            str_, _ = eval_policy(trainC, adap, f"v1/{tag}/train400")
            onehot[tag] = {"test": st, "train": str_}
            log = json.load(open(f"{ROOT}/runs_modal/{tag}.json"))
            ch = log["phases"]["C"]["checks"]
            hi = log["phases"]["C"]["history"]
            curves[tag] = {"checks": ch, "hist_first10": float(np.mean(hi[:10])),
                           "hist_last10": float(np.mean(hi[-10:])),
                           "hist_max": float(max(hi))}
    for a in ["A3"]:
        for s in [101, 102, 103, 104, 105]:
            tag = f"{a}_{s}"
            log = json.load(open(f"{ROOT}/runs_modal/{tag}.json"))
            ch = log["phases"]["C"]["checks"]
            hi = log["phases"]["C"]["history"]
            curves[tag] = {"checks": ch, "hist_first10": float(np.mean(hi[:10])),
                           "hist_last10": float(np.mean(hi[-10:])),
                           "hist_max": float(max(hi))}
    out["onehot_final"] = onehot
    out["v1_curves"] = curves
    json.dump(out, open(f"{ROOT}/diag/audit_c.json", "w"), indent=1)
    print(f"AUDIT-C DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
