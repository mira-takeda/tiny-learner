"""v2views gate traces (read-only; no training).

27 episodes fixed pre-run: first 3 entries (file order) of G1H, I1N,
H1N x VV thetas 301-303. Exact v2views policy (viewed blend, sampled),
same per-episode seeds/horizon. Per step: haz dist, gate weights,
per-skill-viewed argmax + sharpness (B-with-view on the same physical
obs, policy unmodified), blended argmax, action. Outcome + cause.
"""
import json, sys
sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/src")
sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/v2views")
import numpy as np
import views_train as V
from models import split_modular, SKILL_SHAPES, mlp_forward
from policies import softmax
from tasks import TaskRunner

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
T = {s: np.load(f"{ROOT}/v2views_runs/VV_{s}_theta.npz")["theta"] for s in (301, 302, 303)}
GH = json.load(open(f"{ROOT}/v2/g1g2_heldout_v1.json"))
NB = json.load(open(f"{ROOT}/v2views/eval_new_v1.json"))
E = {"G1H": [e for e in GH if e["battery"] == "G1H"][:3],
     "I1N": [e for e in NB if e["battery"] == "I1N"][:3],
     "H1N": [e for e in NB if e["battery"] == "H1N"][:3]}

print("FIXED ENTRIES:")
for b in ("G1H", "I1N", "H1N"):
    for e in E[b]:
        print(f"  {b} {e['id']} seed={e['seed']} fam={e['family']}")


def run(theta, e):
    sk_all, _ = split_modular(theta)
    r = TaskRunner()
    rng = np.random.default_rng(int(e["seed"]))
    obs = r.reset(e["family"], e["cfg"], int(e["seed"]))
    done, steps = False, []
    while not done:
        p, w = V.viewed_blend(theta, obs)
        sk_p = [softmax(mlp_forward(sk_all[k], V.view_obs(obs, k), SKILL_SHAPES)) for k in range(4)]
        a = int(rng.choice(len(p), p=p))
        steps.append({"w": [round(float(x), 3) for x in w],
                      "sk_argmax": [int(np.argmax(q)) for q in sk_p],
                      "sk_max": [round(float(np.max(q)), 3) for q in sk_p],
                      "blend_argmax": int(np.argmax(p)), "a": a,
                      "haz_d": round(float(np.linalg.norm(r.env.haz_pos - r.env.agent_pos)), 1)})
        obs, _, done, info = r.step(a)
    env = r.env
    if info["success"]:
        cause = "success"
    elif float(np.linalg.norm(env.haz_pos - env.agent_pos)) < 5.0:
        cause = "contact-death"
    elif env.t >= env.MAX_STEPS:
        cause = "timeout"
    else:
        cause = "other"
    return steps, bool(info["success"]), cause


OUT = []
for b in ("G1H", "I1N", "H1N"):
    for e in E[b]:
        for s in (301, 302, 303):
            steps, ok, cause = run(T[s], e)
            W = np.mean([st["w"] for st in steps], axis=0)
            haz = [st["haz_d"] for st in steps]
            print(f"{b} {e['id']} th={s}: T={len(steps)} ok={int(ok)} {cause} "
                  f"wMean=[{' '.join(f'{x:.2f}' for x in W)}] hazMin={min(haz)}")
            OUT.append({"b": b, "id": e["id"], "theta": s, "T": len(steps),
                        "ok": ok, "cause": cause, "steps": steps})
json.dump(OUT, open("/Users/satoru/Documents/hackathon/tiny-learner/v2views_diag/traces.json", "w"))
print("saved v2views_diag/traces.json")
