"""S15: confirmation eval only. A01 finals vs frozen T-STU on ACQ18.
2400 episodes, incremental, cumulative 300 s. No training."""
import hashlib
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
SEEDS = (301, 302, 303)
CAP = 300.0
MARGIN = 10.0


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rollout(gth, sk_all, e):
    r = TaskRunner()
    obs = r.reset(e["family"], e["cfg"], int(e["seed"]))
    total, done, info, n = 0.0, False, {}, 0
    uses, sw, prev = [0, 0, 0, 0], 0, None
    while not done:
        k = int(np.argmax(mlp_forward(gth, obs, GATE_SHAPES)))
        a = S4.greedy_skill(sk_all, obs, k)
        uses[k] += 1
        sw += (prev is not None and k != prev)
        prev = k
        obs, rew, done, info = r.step(a)
        total += rew
        n += 1
    return {"ret": total, "cause": S4.cause_of(r, info), "T": n,
            "uses": uses, "switches": sw}


def main():
    fr = json.load(open(f"{ROOT}/v2conf_diag/s15_freeze.json"))
    assert sha(f"{ROOT}/v2conf_diag/s15_acq18.json") == fr["data"]["acq18_sha256"]
    assert sha(f"{ROOT}/v2conf_diag/s14_models.json") == fr["models"]["a01_sha256"]
    assert sha(f"{ROOT}/v2conf_diag/s10_models.json") == fr["models"]["tstu_sha256"]
    for s in SEEDS:
        assert sha(f"{ROOT}/v2conf_runs/VC_{s}_theta.npz") == fr["skills"][str(s)]
    m1 = json.load(open(f"{ROOT}/v2conf_diag/s14_models.json"))
    m0 = json.load(open(f"{ROOT}/v2conf_diag/s10_models.json"))
    acq = json.load(open(f"{ROOT}/v2conf_diag/s15_acq18.json"))
    assert len(acq) == 400
    skills = {s: split_modular(np.load(
        f"{ROOT}/v2conf_runs/VC_{s}_theta.npz")["theta"])[0] for s in SEEDS}
    try:
        state = json.load(open(f"{ROOT}/v2conf_diag/s15_state.json"))
    except FileNotFoundError:
        seed = json.load(open(f"{ROOT}/v2conf_diag/s15_budget_seed.json"))
        state = {"cumulative_s": seed["mkdata_wall_s"], "evals": {}}
    t0 = time.time()
    base = float(state["cumulative_s"])
    spent = lambda: base + time.time() - t0
    ok = lambda: spent() < CAP - MARGIN
    sync = lambda: (state.update(cumulative_s=round(spent(), 1)),
                    json.dump(state, open(
                        f"{ROOT}/v2conf_diag/s15_state.json", "w")))
    arms = {"A01": {s: np.array(m1[f"A01_{s}"]) for s in SEEDS},
            "FROZEN": {s: np.array(m0[f"T-STU_{s}"]) for s in SEEDS}}
    for arm in ("A01", "FROZEN"):
        for s in SEEDS:
            for e in acq:
                key = f"{arm}_{s}_{e['id']}"
                if key in state["evals"]:
                    continue
                if not ok():
                    sync()
                    print(f"INCOMPLETE eval {key} wall={spent():.1f}s",
                          flush=True)
                    return
                r = rollout(arms[arm][s], skills[s], e)
                state["evals"][key] = {"theta": s, "id": e["id"], "arm": arm,
                                       **r, "status": "complete"}
                if len(state["evals"]) % 50 == 0:
                    sync()
    for s in SEEDS:
        assert sha(f"{ROOT}/v2conf_runs/VC_{s}_theta.npz") == fr["skills"][str(s)]
    sync()
    recs = [state["evals"][k] for k in sorted(state["evals"])]
    json.dump({"records": recs, "wall_s": round(spent(), 1)},
              open(f"{ROOT}/v2conf_diag/s15_eval.json", "w"))
    n_inc = sum(1 for r in recs if r.get("status") != "complete")
    print(f"{'done' if n_inc == 0 else 'INCOMPLETE'} "
          f"wall={spent():.1f}s incomplete={n_inc} n={len(recs)}", flush=True)


if __name__ == "__main__":
    main()
