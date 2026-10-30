"""I1 confirmatory: fixed mask B obs[2:5]=0 on the REMAINING 197
entries (indices 3..199, file order) x thetas 301-303, B one-hot
sampled, same seeds/horizon. Mask fixed from D4, no tuning.
Primary metric: collision. Reports per seed separately; exploratory
first-3 kept apart (not pooled). Read-only thetas.
"""
import json, sys
sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/src")
import numpy as np
from models import split_modular, SKILL_SHAPES, mlp_forward
from policies import softmax
from tasks import TaskRunner

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
SK = {s: split_modular(np.load(f"{ROOT}/v2_runs/V2_{s}_theta.npz")["theta"])[0][1] for s in (301, 302, 303)}
E = [e for e in json.load(open(f"{ROOT}/data/test_battery_v1.json")) if e["battery"] == "I1"]
CONF, EXPL = E[3:], E[:3]
assert len(CONF) == 197 and len(EXPL) == 3


def run(sk, e, masked):
    r = TaskRunner()
    rng = np.random.default_rng(int(e["seed"]))
    obs = r.reset(e["family"], e["cfg"], int(e["seed"]))
    done, n = False, 0
    while not done:
        o = obs.copy()
        if masked:
            o[2:5] = 0.0
        q = softmax(mlp_forward(sk, o, SKILL_SHAPES))
        a = int(rng.choice(len(q), p=q))
        obs, _, done, info = r.step(a)
        n += 1
    env = r.env
    out = "arrival" if info["success"] else ("survived" if env.t >= env.MAX_STEPS else "collision")
    return out, n


for s in (301, 302, 303):
    cu = ca = su = 0
    cm = am = sm = 0
    dT = []
    flips = {"col->noncol": 0, "noncol->col": 0}
    for e in CONF:
        ou, Tu = run(SK[s], e, False)
        om, Tm = run(SK[s], e, True)
        for o, acc in ((ou, None),):
            pass
        cu += ou == "collision"; ca += ou == "arrival"; su += ou == "survived"
        cm += om == "collision"; am += om == "arrival"; sm += om == "survived"
        dT.append(Tm - Tu)
        if ou == "collision" and om != "collision":
            flips["col->noncol"] += 1
        if ou != "collision" and om == "collision":
            flips["noncol->col"] += 1
    dT = np.array(dT)
    print(f"th={s} UNMASK: col={cu} arr={ca} surv={su} | MASK: col={cm} arr={am} surv={sm} | "
          f"dCol={cm - cu:+d} flips={flips} dT_mean={dT.mean():+.1f} dT_med={np.median(dT):+.0f}")
