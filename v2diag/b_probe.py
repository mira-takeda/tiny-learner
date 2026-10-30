"""B one-hot probe on the 9 ORIGINAL I1 episodes (read-only thetas).

Same 3 entries (T-I1-000{0,1,2}) x V2 thetas 301-303, same per-episode
seeds, sampled policy, 200-step horizon. No teleport, no switching.
Policy: pure skill-1 (pilot3 B skill, frozen in V2 thetas).
Reports per episode: collision/survival, duration, min hazard
separation, beacon arrival (separately). Gate originals kept as ref.
"""
import json, sys
sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/src")
import numpy as np
from models import modular_probs
from tasks import TaskRunner

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
T = {s: np.load(f"{ROOT}/v2_runs/V2_{s}_theta.npz")["theta"] for s in (301, 302, 303)}
BAT = json.load(open(f"{ROOT}/data/test_battery_v1.json"))
E = [e for e in BAT if e["battery"] == "I1"][:3]
GATE = {(x["b"], x["id"], x["theta"]): x for x in json.load(open("/tmp/v2_traces.json")) if x["b"] == "I1"}


def run_b(theta, e):
    r = TaskRunner()
    rng = np.random.default_rng(int(e["seed"]))
    obs = r.reset(e["family"], e["cfg"], int(e["seed"]))
    done, hmin, n, arrived = False, 1e9, 0, False
    while not done:
        p, _ = modular_probs(theta, obs, skill_only=1)
        a = int(rng.choice(len(p), p=p))
        obs, _, done, info = r.step(a)
        n += 1
        h = float(np.linalg.norm(r.env.haz_pos - r.env.agent_pos))
        hmin = min(hmin, h)
        if info["success"]:
            arrived = True
    env = r.env
    # I1 branch: contact -> done without success; arrival -> success;
    # timeout -> done at MAX_STEPS without success.
    if info["success"]:
        out = "arrival"
    elif env.t >= env.MAX_STEPS:
        out = "survived-horizon"
    else:
        out = "collision"
    return {"out": out, "T": n, "hmin": round(hmin, 1)}


print("B ONE-HOT vs GATE-ORIGINAL (same episodes/seeds/policy/horizon):")
for e in E:
    for s in (301, 302, 303):
        b = run_b(T[s], e)
        g = GATE[("I1", e["id"], s)]
        print(f"  {e['id']} th={s} B1HOT: {b['out']} T={b['T']} hmin={b['hmin']} | GATE: ok={int(g['ok'])} {g['cause']} T={g['T']}")
