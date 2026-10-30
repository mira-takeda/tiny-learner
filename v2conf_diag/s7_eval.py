"""S6-style closed loop on fresh ACQ7 (no training).

4 arms x 200 eps x 3 thetas: student 32/5000 (primary, post-S6
choice), student 32/500, A-alone greedy viewed, B-alone greedy
viewed. Policy identical to S4/S6 (imported run_ep/greedy_skill).
Asserts s7_freeze (data sha) before running. Budget 300 s,
incremental save; overrun -> INCOMPLETE.
"""
import hashlib
import json
import sys
import time

sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/src")
sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/v2views")
sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/v2conf_diag")

import numpy as np

import s4_closed as S4
from models import GATE_SHAPES, mlp_forward, split_modular

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
SEEDS = (301, 302, 303)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    t0 = time.time()
    fr = json.load(open(f"{ROOT}/v2conf_diag/s7_freeze.json"))
    assert sha(f"{ROOT}/v2conf_diag/s7_acq7.json") == fr["data"]["sha256"], \
        "ACQ7 seal broken"
    eps = json.load(open(f"{ROOT}/v2conf_diag/s7_acq7.json"))
    assert len(eps) == 200 and all(e["family"] == "G1" for e in eps)
    s5 = json.load(open(f"{ROOT}/v2conf_diag/s5_results.json"))
    print(f"freeze ok: {fr['data']['sha256'][:12]}... n=200", flush=True)

    skills = {}
    for s in SEEDS:
        th = np.load(f"{ROOT}/v2conf_runs/VC_{s}_theta.npz")["theta"]
        skills[s] = split_modular(th)[0]
    stud = {(s, ep): np.array(s5["runs"][f"32_{s}"]["ckpts"][str(ep)])
            for s in SEEDS for ep in (500, 5000)}

    out = {"freeze_sha": sha(f"{ROOT}/v2conf_diag/s7_freeze.json"),
           "records": []}
    save = lambda: json.dump(out, open(f"{ROOT}/v2conf_diag/s7_results.json",
                                       "w"))

    def student_fn(gth, sk_all):
        def f(obs):
            k = int(np.argmax(mlp_forward(gth, obs, GATE_SHAPES)))
            return S4.greedy_skill(sk_all, obs, k), k
        return f

    def const_fn(sk_all, k):
        def f(obs):
            return S4.greedy_skill(sk_all, obs, k), k
        return f

    plan = []
    for s in SEEDS:
        plan.append((f"st325000", student_fn(stud[(s, 5000)], skills[s]), s))
        plan.append((f"st32500", student_fn(stud[(s, 500)], skills[s]), s))
        plan.append(("A", const_fn(skills[s], 0), s))
        plan.append(("B", const_fn(skills[s], 1), s))
    for arm, fn, s in plan:
        for e in eps:
            if time.time() - t0 > fr["budget_s"] - 5:
                out["records"].append({"arm": arm, "theta": s,
                                       "id": e["id"], "status": "incomplete"})
                save()
                continue
            rec = S4.run_ep(e, fn)
            rec.update({"arm": arm, "theta": s, "id": e["id"],
                        "status": "complete"})
            # Drop per-step hobs to bound file size (keep aggregates).
            rec.pop("hobs", None)
            out["records"].append(rec)
            save()
        n = sum(1 for r in out["records"] if r["arm"] == arm
                and r["theta"] == s and r["status"] == "complete")
        print(f"{arm} th={s}: {n}/200 complete "
              f"wall={time.time() - t0:.0f}s", flush=True)

    out["wall_s"] = round(time.time() - t0, 1)
    save()
    n_inc = sum(1 for r in out["records"] if r["status"] != "complete")
    print(f"saved s7_results.json wall={out['wall_s']}s "
          f"incomplete={n_inc}", flush=True)


if __name__ == "__main__":
    main()
