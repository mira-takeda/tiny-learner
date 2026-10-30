"""S13-diag: frozen vs tuned-final on ACQ15-train12 (72 eps). Read-only
models; no training, no teacher. Same hard policy + exact return."""
import json
import sys
import time

sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/src")
sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/v2conf_diag")

import numpy as np

import s4_closed as S4
from models import GATE_SHAPES, mlp_forward, split_modular
from tasks import TaskRunner

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
T0 = time.time()
CAP = 60.0


def rollout(gth, sk_all, e):
    r = TaskRunner()
    obs = r.reset(e["family"], e["cfg"], int(e["seed"]))
    total, done, info, n = 0.0, False, {}, 0
    while not done:
        k = int(np.argmax(mlp_forward(gth, obs, GATE_SHAPES)))
        obs, rew, done, info = r.step(S4.greedy_skill(sk_all, obs, k))
        total += rew
        n += 1
    return {"ret": total, "cause": S4.cause_of(r, info), "T": n}


def main():
    tr = json.load(open(f"{ROOT}/v2conf_diag/s13_acq15.json"))
    m0 = json.load(open(f"{ROOT}/v2conf_diag/s10_models.json"))
    m1 = json.load(open(f"{ROOT}/v2conf_diag/s13_models.json"))
    st = json.load(open(f"{ROOT}/v2conf_diag/s13_state.json"))
    assert st["phase"] == "done"
    skills = {s: split_modular(np.load(
        f"{ROOT}/v2conf_runs/VC_{s}_theta.npz")["theta"])[0]
        for s in (301, 302, 303)}
    recs = []
    for s in (301, 302, 303):
        for e in tr:
            for arm, key in (("FROZEN", f"T-STU_{s}"), ("TUNED", f"TUNED_{s}")):
                if time.time() - T0 > CAP - 5:
                    print("INCOMPLETE diag budget", flush=True)
                    json.dump({"records": recs, "history": st["history"]},
                              open(f"{ROOT}/v2conf_diag/s13_diag.json", "w"))
                    return
                src = m0 if arm == "FROZEN" else m1
                r = rollout(np.array(src[key]), skills[s], e)
                recs.append({"theta": s, "id": e["id"], "arm": arm, **r})
    json.dump({"records": recs, "history": st["history"],
               "wall_s": round(time.time() - T0, 1)},
              open(f"{ROOT}/v2conf_diag/s13_diag.json", "w"))
    print(f"diag done n={len(recs)} wall={time.time()-T0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
