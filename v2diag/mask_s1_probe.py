"""S1-mask probe on B one-hot (read-only thetas; no training).

Premise verified pre-run: B trained with S1==[0,0,0] always (400/400
B-train cfgs lack 'goal'; _slot(None)->zeros; S1=obs[2:5]).
Same 9 ORIGINAL I1 episodes x thetas 301-303, B one-hot sampled, same
seeds/horizon. Intervention: mask obs[2:5]=0 before B's forward ONLY.
Beacon still exists (env, reward, success untouched). No other masks.
Compares masked vs unmasked per episode + within-step action-prob
changes holding the physical observation fixed.
"""
import json, sys
sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/src")
import numpy as np
from models import split_modular, SKILL_SHAPES, mlp_forward
from policies import softmax
from tasks import TaskRunner

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
T = {s: np.load(f"{ROOT}/v2_runs/V2_{s}_theta.npz")["theta"] for s in (301, 302, 303)}
BAT = json.load(open(f"{ROOT}/data/test_battery_v1.json"))
E = [e for e in BAT if e["battery"] == "I1"][:3]
SK = {s: split_modular(T[s])[0][1] for s in T}


def b_dist(sk, obs):
    return softmax(mlp_forward(sk, obs, SKILL_SHAPES))


def run(sk, e, masked):
    r = TaskRunner()
    rng = np.random.default_rng(int(e["seed"]))
    obs = r.reset(e["family"], e["cfg"], int(e["seed"]))
    assert np.any(obs[2:5] != 0), "S1 already zero?!"  # beacon active in I1
    done, hmin, n = False, 1e9, 0
    tvs, agr = [], []
    while not done:
        q_full = b_dist(sk, obs)
        mo = obs.copy()
        mo[2:5] = 0.0
        q_mask = b_dist(sk, mo)
        tvs.append(0.5 * float(np.abs(q_full - q_mask).sum()))
        agr.append(int(np.argmax(q_full) == np.argmax(q_mask)))
        q = q_mask if masked else q_full
        a = int(rng.choice(len(q), p=q))
        obs, _, done, info = r.step(a)
        n += 1
        hmin = min(hmin, float(np.linalg.norm(r.env.haz_pos - r.env.agent_pos)))
    env = r.env
    out = "arrival" if info["success"] else ("survived" if env.t >= env.MAX_STEPS else "collision")
    return {"out": out, "T": n, "hmin": round(hmin, 1),
            "tv": round(float(np.mean(tvs)), 3),
            "argmax_agr": round(float(np.mean(agr)), 3), "steps": n}


print("S1-MASK vs UNMASKED B one-hot (same episodes/seeds/horizon):")
for e in E:
    for s in (301, 302, 303):
        u = run(SK[s], e, False)
        m = run(SK[s], e, True)
        print(f"  {e['id']} th={s} UNMASK: {u['out']} T={u['T']} hmin={u['hmin']} "
              f"tv={u['tv']} aAgr={u['argmax_agr']} | "
              f"MASK: {m['out']} T={m['T']} hmin={m['hmin']} "
              f"tv={m['tv']} aAgr={m['argmax_agr']}")
