"""C follow-up diagnostics (diag-only): beam wins by oz0 quartile + return
decomposition of A2/A3 final full-nets on test-200 and train-400.
Appends to diag/audit_c2.json."""

import json
import sys
import time

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
sys.path.insert(0, f"{ROOT}/diag")
sys.path.insert(0, f"{ROOT}/src")

import numpy as np

import beam as B
from evaluate import make_policy
from instrument import run_episode, summarize


def eval_policy(entries, policy_fn, tag, fam="C"):
    eps = []
    for e in entries:
        sd = int(e["seed"]) if "seed" in e else 900000 + int(e["idx"])
        eps.append(run_episode(fam, e["cfg"], sd, policy_fn,
                               np.random.default_rng(sd)))
    s = summarize(eps)
    print(f"{tag}: sr={s['sr']:.3f} ret={s['mean_return']:.1f} "
          f"oz={s['comp_mean']['c_oz']:.1f} ao={s['comp_mean']['c_ao']:.1f} "
          f"bonus={s['comp_mean']['bonus']:.1f} dwell={s['dwell_max_mean']:.1f}",
          flush=True)
    return s


def main():
    t0 = time.time()
    out = {}
    entries = json.load(open(f"{ROOT}/data/test_battery_v1.json"))
    testC = [e for e in entries if e["battery"] == "C"]
    train = json.load(open(f"{ROOT}/data/train_meta_v1.json"))
    trainC = [dict(e, idx=i) for i, e in enumerate(train) if e["family"] == "C"]
    oz = [(float(np.linalg.norm(np.array(e["cfg"]["obj"]["pos"])
               - np.array(e["cfg"]["zone"]["center"]))), e) for e in testC]
    oz.sort()
    q = len(oz) // 4
    byq = {}
    for i in range(4):
        sample = [e for _, e in oz[i * q:(i + 1) * q]][:: q // 6][:6]
        r = B.run_beam_battery("C", sample, width=48)
        lo = oz[i * q][0]
        hi = oz[(i + 1) * q - 1][0]
        byq[f"oz0_{lo:.0f}_{hi:.0f}"] = r
        print(f"beam quartile {lo:.0f}-{hi:.0f}: {r}", flush=True)
    out["beam_by_oz0"] = byq
    full = {}
    for a in ["A2", "A3"]:
        arch = "mono" if a == "A3" else "modular"
        for s in [101, 102, 103, 104, 105]:
            tag = f"{a}_{s}"
            th = np.load(f"{ROOT}/runs_modal/{tag}_theta.npz")["theta"]
            pol = make_policy(arch, th, "normal")
            adap = lambda o, r, g, p=pol: p(o, g)
            st = eval_policy(testC, adap, f"{tag}/test")
            str_ = eval_policy(trainC, adap, f"{tag}/train400")
            full[tag] = {"test": st, "train": str_}
    out["fullnet_final"] = full
    json.dump(out, open(f"{ROOT}/diag/audit_c2.json", "w"), indent=1)
    print(f"AUDIT-C2 DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
