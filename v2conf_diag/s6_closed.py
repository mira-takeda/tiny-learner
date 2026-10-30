"""S6: closed-loop eval of the four S5 cells (no training).

Same 9 S3-eval scenarios x 3 thetas as S4. Cell 32/500 is REUSED
from S4 after asserting weight identity (S5 ckpt == S3 model) and
execution identity (policy functions imported from s4_closed).
Only the other 81 episodes run here: (32/5000, 128/500, 128/5000).
Same policy as S4: argmax-4 gate -> greedy viewed-skill action.
No C/D masking, fallback, or planner. Same seeds/world/horizon.
Teacher and baselines are reused from s4_results, never re-run.
Reused diagnostic eval, not independent confirmation. Budget 300 s.
"""
import json
import sys
import time

sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/src")
sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/v2views")
sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/v2conf_diag")

import numpy as np

import s4_closed as S4
import views_train as V
from models import GATE_SHAPES, mlp_forward, split_modular

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
SEEDS = (301, 302, 303)
CELLS = [(32, 5000), (128, 500), (128, 5000)]
T_BUDGET = 300.0


def main():
    t0 = time.time()
    evc = json.load(open(f"{ROOT}/v2conf/eval_conf_v1.json"))
    by_id = {e["id"]: e for e in evc if e["battery"] == "ACQ"}
    s3r = json.load(open(f"{ROOT}/v2conf_diag/s3_results.json"))
    eval_ids = s3r["partition"]["eval"]
    assert len(eval_ids) == 9
    s3m = json.load(open(f"{ROOT}/v2conf_diag/s3_models.json"))
    s5 = json.load(open(f"{ROOT}/v2conf_diag/s5_results.json"))
    s4 = json.load(open(f"{ROOT}/v2conf_diag/s4_results.json"))
    assert s4["eval_ids"] == eval_ids, "S4 scenario mismatch"

    # Weight identity 32/500: S5 ckpt == S3 final model, bit-exact.
    for s in SEEDS:
        ck = np.array(s5["runs"][f"32_{s}"]["ckpts"]["500"])
        ref = np.array(s3m[str(s)]["theta"])
        d = float(np.abs(ck - ref).max())
        print(f"32/500 th={s}: maxabs vs S3 = {d:.1e}", flush=True)
        assert d == 0.0, f"weight identity failed th={s}"
    print("execution identity: policy imported from s4_closed "
          "(run_ep/greedy_skill/cause_of); 32/500 outcomes reused from S4",
          flush=True)

    skills = {}
    for s in SEEDS:
        th = np.load(f"{ROOT}/v2conf_runs/VC_{s}_theta.npz")["theta"]
        skills[s] = split_modular(th)[0]

    out = {"eval_ids": eval_ids,
           "cells": ["32/500", "32/5000", "128/500", "128/5000"],
           "reuse_32_500_from": "s4_results.json student records",
           "records": []}
    save = lambda: json.dump(out, open(f"{ROOT}/v2conf_diag/s6_results.json",
                                       "w"))

    for H, ep in CELLS:
        for s in SEEDS:
            gth = np.array(s5["runs"][f"{H}_{s}"]["ckpts"][str(ep)])
            assert len(gth) == (sum(int(np.prod(x)) for x in GATE_SHAPES)
                                if H == 32 else 14 * H + H + H * 4 + 4)
            sk_all = skills[s]

            def step_fn(obs, _g=gth, _sk=sk_all):
                if H == 32:
                    k = int(np.argmax(mlp_forward(_g, obs, GATE_SHAPES)))
                else:
                    W1 = _g[:14 * H].reshape(14, H)
                    b1 = _g[14 * H:14 * H + H]
                    W2 = _g[14 * H + H:14 * H + H + H * 4].reshape(H, 4)
                    b2 = _g[14 * H + H + H * 4:]
                    k = int(np.argmax(np.tanh(obs @ W1 + b1) @ W2 + b2))
                return S4.greedy_skill(_sk, obs, k), k

            for eid in eval_ids:
                if time.time() - t0 > T_BUDGET - 5:
                    out["records"].append(
                        {"cell": f"{H}/{ep}", "theta": s, "id": eid,
                         "status": "incomplete"})
                    save()
                    continue
                rec = S4.run_ep(by_id[eid], step_fn)
                rec.update({"cell": f"{H}/{ep}", "theta": s, "id": eid,
                            "status": "complete"})
                out["records"].append(rec)
                save()
            n = sum(1 for r in out["records"]
                    if r["cell"] == f"{H}/{ep}" and r["theta"] == s
                    and r["status"] == "complete")
            print(f"cell {H}/{ep} th={s}: {n}/9 complete", flush=True)

    out["wall_s"] = round(time.time() - t0, 1)
    save()
    n_inc = sum(1 for r in out["records"] if r["status"] != "complete")
    print(f"saved s6_results.json wall={out['wall_s']}s "
          f"incomplete={n_inc}", flush=True)


if __name__ == "__main__":
    main()
