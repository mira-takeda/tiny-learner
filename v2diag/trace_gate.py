"""V2 gate trace diagnostic (read-only; no training, no artifact writes).

27 episodes: batteries G1H/I1/H1 x first 3 entries (file order, fixed
pre-run, no result selection) x V2 thetas 301-303. Policy = exact
scored policy (make_policy normal, per-episode battery seed).
Per step: obs dists, gate weights, per-skill argmax + sharpness,
blended argmax, sampled action. Outcome cause recomputed from env.
"""
import json, sys
sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/src")
import numpy as np
from models import ModularNet, split_modular
from policies import unpack
from tasks import TaskRunner

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
T = {s: np.load(f"{ROOT}/v2_runs/V2_{s}_theta.npz")["theta"] for s in (301, 302, 303)}
BAT = json.load(open(f"{ROOT}/data/test_battery_v1.json"))
GH = json.load(open(f"{ROOT}/v2/g1g2_heldout_v1.json"))
E = {"G1H": [e for e in GH if e["battery"] == "G1H"][:3],
     "I1": [e for e in BAT if e["battery"] == "I1"][:3],
     "H1": [e for e in BAT if e["battery"] == "H1"][:3]}

print("FIXED ENTRIES (pre-run, first-3 file order):")
for b in ("G1H", "I1", "H1"):
    for e in E[b]:
        print(f"  {b} {e['id']} seed={e['seed']} fam={e['family']}")

def run(theta, e):
    net = ModularNet(theta)
    r = TaskRunner()
    rng = np.random.default_rng(int(e["seed"]))
    obs = r.reset(e["family"], e["cfg"], int(e["seed"]))
    steps = []
    done = False
    while not done:
        p, w = net.probs(obs)
        sk_p = []
        th_sk, _ = split_modular(theta)
        from models import SKILL_SHAPES, mlp_forward
        from policies import softmax
        for k in range(4):
            q = softmax(mlp_forward(th_sk[k], obs, SKILL_SHAPES))
            sk_p.append(q)
        a = int(rng.choice(len(p), p=p))
        steps.append({"w": [round(float(x), 3) for x in w],
                      "sk_argmax": [int(np.argmax(q)) for q in sk_p],
                      "sk_max": [round(float(np.max(q)), 3) for q in sk_p],
                      "blend_argmax": int(np.argmax(p)),
                      "blend_max": round(float(np.max(p)), 3), "a": a,
                      "goal_d": round(float(np.linalg.norm(r.env.goal - r.env.agent_pos)), 1) if r.env.goal is not None else None,
                      "haz_d": round(float(np.linalg.norm(r.env.haz_pos - r.env.agent_pos)), 1) if r.env.haz_pos is not None else None})
        obs, _, done, info = r.step(a)
    # cause
    env = r.env
    if info["success"]:
        cause = "success"
    elif env.haz_pos is not None and float(np.linalg.norm(env.haz_pos - env.agent_pos)) < 5.0:
        cause = "contact-death"
    elif env.t >= env.MAX_STEPS:
        cause = "timeout"
    else:
        cause = "other-fail"
    return steps, bool(info["success"]), cause

OUT = []
for b in ("G1H", "I1", "H1"):
    for e in E[b]:
        for s in (301, 302, 303):
            steps, ok, cause = run(T[s], e)
            n = len(steps)
            W = np.mean([st["w"] for st in steps], axis=0)
            agree = [float(np.mean([st["blend_argmax"] == st["sk_argmax"][k] for st in steps])) for k in range(4)]
            samp_ag = [float(np.mean([st["a"] == st["sk_argmax"][k] for st in steps])) for k in range(4)]
            sharp = [round(float(np.mean([st["sk_max"][k] for st in steps])), 3) for k in range(4)]
            haz = [st["haz_d"] for st in steps if st["haz_d"] is not None]
            print(f"{b} {e['id']} th={s}: T={n} ok={int(ok)} cause={cause} "
                  f"wMean=[{' '.join(f'{x:.2f}' for x in W)}] "
                  f"ctrl=[{' '.join(f'{x:.2f}' for x in agree)}] "
                  f"sampAgr=[{' '.join(f'{x:.2f}' for x in samp_ag)}] "
                  f"sharp={sharp} hazMin={min(haz) if haz else None}")
            OUT.append({"b": b, "id": e["id"], "seed": e["seed"], "theta": s,
                        "T": n, "ok": ok, "cause": cause, "steps": steps})
json.dump(OUT, open("/tmp/v2_traces.json", "w"))
print("saved /tmp/v2_traces.json")
