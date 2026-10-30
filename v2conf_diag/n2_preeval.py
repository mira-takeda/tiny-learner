"""Pre-N2 control eval: N1b gates (frozen) on sealed heldout
(60 G1 + 60 G2 x3 thetas = 360 eps). Same batteries post-N2 will
use -> paired pre/post. Read-only models; changes NOTHING in N2.
Frozen: n2_freeze.json. Budget 300 s, incremental.
"""
import hashlib
import json
import sys
import time

sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/src")
sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/v2views")

import numpy as np

import s4_closed as S4
from models import GATE_SHAPES, mlp_forward, split_modular
from tasks import TaskRunner

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
SEEDS = (301, 302, 303)
T_BUDGET, T_MARGIN = 300.0, 10.0

sys.path.insert(0, f"{ROOT}/v2conf_diag")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cause_of(runner, info):
    env = runner.env
    if info["success"]:
        return "arrival"
    if env.haz_pos is not None and float(
            np.linalg.norm(env.haz_pos - env.agent_pos)) < 5.0:
        return "contact-death"
    if env.t >= env.MAX_STEPS:
        return "timeout"
    return "other"


def main():
    t0 = time.time()
    spent = lambda: time.time() - t0
    fr = json.load(open(f"{ROOT}/v2conf_diag/n2_freeze.json"))
    assert sha(f"{ROOT}/data/train_meta_v1.json") == fr["data_sha"]
    assert sha(f"{ROOT}/v2conf_diag/n1b_models.json") == fr["init"]["file_sha256"]
    for s in SEEDS:
        assert sha(f"{ROOT}/v2conf_runs/VC_{s}_theta.npz") == fr["skills"][str(s)]
    m = json.load(open(f"{ROOT}/v2conf_diag/n1b_models.json"))
    by_id = {e["id"]: e for e in json.load(
        open(f"{ROOT}/data/train_meta_v1.json"))}
    acq = [by_id[i] for i in fr["heldout"]["G1"] + fr["heldout"]["G2"]]
    assert len(acq) == 120
    skills = {s: split_modular(np.load(
        f"{ROOT}/v2conf_runs/VC_{s}_theta.npz")["theta"])[0] for s in SEEDS}
    try:
        out = json.load(open(f"{ROOT}/v2conf_diag/n2_preeval.json"))
        done = {(r["id"], r["theta"]) for r in out["records"]
                if r["status"] == "complete"}
    except FileNotFoundError:
        out = {"records": []}
        done = set()
    sync = lambda: json.dump(out, open(
        f"{ROOT}/v2conf_diag/n2_preeval.json", "w"))
    for s in SEEDS:
        g = np.array(m[f"N1b_{s}"])
        for e in acq:
            if (e["id"], s) in done:
                continue
            if spent() > T_BUDGET - T_MARGIN:
                out["records"].append(
                    {"id": e["id"], "theta": s, "status": "incomplete"})
                sync()
                print(f"INCOMPLETE cum={spent():.0f}s", flush=True)
                return
            r = TaskRunner()
            obs = r.reset(e["family"], e["cfg"], int(e["seed"]))
            total, done_ep, info, n = 0.0, False, {}, 0
            uses, sw, prev = [0, 0, 0, 0], 0, None
            while not done_ep:
                k = int(np.argmax(mlp_forward(g, obs, GATE_SHAPES)))
                a = S4.greedy_skill(skills[s], obs, k)
                uses[k] += 1
                sw += (prev is not None and k != prev)
                prev = k
                obs, rew, done_ep, info = r.step(a)
                total += rew
                n += 1
            out["records"].append(
                {"id": e["id"], "theta": s, "fam": e["family"],
                 "status": "complete", "cause": cause_of(r, info),
                 "ret": round(total, 3), "T": n, "uses": uses,
                 "switches": sw})
            if len(out["records"]) % 25 == 0:
                sync()
    sync()
    n_inc = sum(1 for r in out["records"] if r["status"] != "complete")
    print(f"pre-eval {'done' if n_inc == 0 else 'INCOMPLETE'} "
          f"cum={spent():.0f}s n={len(out['records'])}", flush=True)


if __name__ == "__main__":
    main()
