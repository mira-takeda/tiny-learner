"""S16: A-only / B-only on ACQ18 x3 thetas (2400 eps). A01 results
reused from s15_eval.json, never re-run. Incremental, 90 s cap."""
import hashlib
import json
import sys
import time

sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/src")
sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/v2conf_diag")

import numpy as np

import s4_closed as S4
from models import split_modular
from tasks import TaskRunner

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
SEEDS = (301, 302, 303)
CAP = 90.0
MARGIN = 8.0


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rollout_const(sk_all, k, e):
    r = TaskRunner()
    obs = r.reset(e["family"], e["cfg"], int(e["seed"]))
    total, done, info, n = 0.0, False, {}, 0
    while not done:
        obs, rew, done, info = r.step(S4.greedy_skill(sk_all, obs, k))
        total += rew
        n += 1
    return {"ret": total, "cause": S4.cause_of(r, info), "T": n}


def main():
    t0 = time.time()
    fr15 = json.load(open(f"{ROOT}/v2conf_diag/s15_freeze.json"))
    assert sha(f"{ROOT}/v2conf_diag/s15_acq18.json") == fr15["data"]["acq18_sha256"]
    for s in SEEDS:
        assert sha(f"{ROOT}/v2conf_runs/VC_{s}_theta.npz") == fr15["skills"][str(s)]
    old = [r for r in json.load(
        open(f"{ROOT}/v2conf_diag/s15_eval.json"))["records"]
        if r["arm"] == "A01"]
    assert len(old) == 1200 and all(r["status"] == "complete" for r in old)
    acq = json.load(open(f"{ROOT}/v2conf_diag/s15_acq18.json"))
    skills = {s: split_modular(np.load(
        f"{ROOT}/v2conf_runs/VC_{s}_theta.npz")["theta"])[0] for s in SEEDS}
    try:
        state = json.load(open(f"{ROOT}/v2conf_diag/s16_state.json"))
    except FileNotFoundError:
        state = {"cumulative_s": 0.0, "evals": {}}
    base = float(state["cumulative_s"])
    spent = lambda: base + time.time() - t0
    ok = lambda: spent() < CAP - MARGIN
    sync = lambda: (state.update(cumulative_s=round(spent(), 1)),
                    json.dump(state, open(
                        f"{ROOT}/v2conf_diag/s16_state.json", "w")))
    for arm, k in (("A-ONLY", 0), ("B-ONLY", 1)):
        for s in SEEDS:
            for e in acq:
                key = f"{arm}_{s}_{e['id']}"
                if key in state["evals"]:
                    continue
                if not ok():
                    sync()
                    print(f"INCOMPLETE {key} wall={spent():.1f}s", flush=True)
                    return
                r = rollout_const(skills[s], k, e)
                state["evals"][key] = {"theta": s, "id": e["id"], "arm": arm,
                                       **r, "status": "complete"}
                if len(state["evals"]) % 50 == 0:
                    sync()
    sync()
    recs = [state["evals"][k] for k in sorted(state["evals"])]
    json.dump({"records": recs, "wall_s": round(spent(), 1),
               "reused_a01_from": "s15_eval.json (1200, not re-run)"},
              open(f"{ROOT}/v2conf_diag/s16_eval.json", "w"))
    print(f"done wall={spent():.1f}s n={len(recs)}", flush=True)


if __name__ == "__main__":
    main()
