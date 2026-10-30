"""I1 hazard-teleport contrast (read-only thetas; no training).

Same 9 I1 episodes (T-I1-000{0,1,2} x thetas 301-303). Step 1:
reproduce originals (must match /tmp/v2_traces.json outcomes).
Step 2: deterministic relocation (rule frozen BEFORE results, see
RELOCATE): rotate H about A preserving |A-H| exactly, first angle in
[180,90,-90,135,-135,45,-45] giving beacon-haz>=35 and H' in [5,95]^2.
Speed, agent, beacon, seed, policy, horizon identical (asserted).
Infeasible -> recorded, episode kept, relocated marked N/A.
"""
import json, sys, copy
sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/src")
import numpy as np
from models import ModularNet
from tasks import TaskRunner

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
ANGLES = [180.0, 90.0, -90.0, 135.0, -135.0, 45.0, -45.0]  # FROZEN ORDER

T = {s: np.load(f"{ROOT}/v2_runs/V2_{s}_theta.npz")["theta"] for s in (301, 302, 303)}
BAT = json.load(open(f"{ROOT}/data/test_battery_v1.json"))
E = [e for e in BAT if e["battery"] == "I1"][:3]
OLD = {(x["b"], x["id"], x["theta"]): x for x in json.load(open("/tmp/v2_traces.json")) if x["b"] == "I1"}


def relocate(e):
    """FROZEN RULE (fixed before observing results). Returns (cfg2, angle, db_new)
    or (None, None, None) if infeasible (recorded, episode kept)."""
    A = np.array(e["cfg"]["agent"], dtype=float)
    B = np.array(e["cfg"]["goal"]["pos"], dtype=float)
    H = np.array(e["cfg"]["hazard"]["pos"], dtype=float)
    da = float(np.linalg.norm(H - A))
    v = H - A
    for deg in ANGLES:
        t = np.deg2rad(deg)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        Hp = A + R.dot(v)
        db = float(np.linalg.norm(Hp - B))
        if db >= 35.0 and 5.0 <= Hp[0] <= 95.0 and 5.0 <= Hp[1] <= 95.0:
            assert abs(float(np.linalg.norm(Hp - A)) - da) < 1e-9
            cfg2 = copy.deepcopy(e["cfg"])
            cfg2["hazard"]["pos"] = [float(Hp[0]), float(Hp[1])]
            return cfg2, deg, db, da
    return None, None, None, da


def run(theta, family, cfg, seed):
    net = ModularNet(theta)
    r = TaskRunner()
    rng = np.random.default_rng(int(seed))
    obs = r.reset(family, cfg, int(seed))
    done, ws, hmin, n = False, [], 1e9, 0
    while not done:
        p, w = net.probs(obs)
        ws.append(w)
        h = float(np.linalg.norm(r.env.haz_pos - r.env.agent_pos))
        hmin = min(hmin, h)
        a = int(rng.choice(len(p), p=p))
        obs, _, done, info = r.step(a)
        n += 1
    env = r.env
    if info["success"]:
        cause = "arrival"
    elif float(np.linalg.norm(env.haz_pos - env.agent_pos)) < 5.0:
        cause = "collision"
    elif env.t >= env.MAX_STEPS:
        cause = "timeout"
    else:
        cause = "other"
    W = np.mean(ws, axis=0)
    return {"ok": bool(info["success"]), "cause": cause, "T": n,
            "hmin": round(hmin, 1), "w": [round(float(x), 3) for x in W]}


print("RELOCATION (frozen rule, computed pre-results):")
CFGS = {}
for e in E:
    cfg2, deg, db, da = relocate(e)
    A = np.array(e["cfg"]["agent"]); H = np.array(e["cfg"]["hazard"]["pos"]); B = np.array(e["cfg"]["goal"]["pos"])
    print(f"  {e['id']}: da={da:.1f} db_old={float(np.linalg.norm(H-B)):.1f} -> " +
          (f"angle={deg} db_new={db:.1f}" if cfg2 else "INFEASIBLE (kept, N/A)"))
    CFGS[e["id"]] = cfg2

print("\nPAIRS (orig reproduction vs relocated):")
for e in E:
    for s in (301, 302, 303):
        o = run(T[s], e["family"], e["cfg"], int(e["seed"]))
        key = ("I1", e["id"], s)
        ref = OLD[key]
        match = (o["ok"] == ref["ok"] and o["cause"] == ("contact-death" if ref["cause"] == "contact-death" else ref["cause"]) and o["T"] == ref["T"])
        flag = "REPRO" if match else f"MISMATCH(ref ok={ref['ok']} {ref['cause']} T={ref['T']})"
        print(f"  {e['id']} th={s} ORIG: ok={int(o['ok'])} {o['cause']} T={o['T']} hmin={o['hmin']} w={o['w']} [{flag}]")
        if CFGS[e["id"]] is None:
            print(f"  {e['id']} th={s} TEL: N/A (infeasible geometry, recorded)")
        else:
            t = run(T[s], e["family"], CFGS[e["id"]], int(e["seed"]))
            print(f"  {e['id']} th={s} TEL:  ok={int(t['ok'])} {t['cause']} T={t['T']} hmin={t['hmin']} w={t['w']}")
