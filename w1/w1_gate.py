"""W1 fresh gate (SINGLE OPENING of gate pools).

Arms {W1-final, W1-imit} x {gate_g1, gate_g2} x 3 thetas = 2400 eps
+ W1-final singles spot-check on sing_gate (240 eps, descriptive).
Bar (frozen): W1-final mean >= 0.70 on EACH family (means over 3
thetas). PASS both -> H may open once. Else STOP. Frozen: freeze
+ MANIFEST. Budget 900 s, resume.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "v2views"))
sys.path.insert(0, os.path.join(ROOT, "v2conf_diag"))
sys.path.insert(0, HERE)

import numpy as np

import s4_closed as S4
from gen import sha256_file
from models import GATE_SHAPES, mlp_forward, split_modular
from tasks import TaskRunner
import gen_wide as GW

W1 = HERE
SEEDS = (301, 302, 303)
T_BUDGET, T_MARGIN = 900.0, 10.0
BAR = 0.70


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
    return {"ret": round(total, 3), "cause": cause_of(r, info), "T": n,
            "uses": uses, "switches": sw}


def main():
    t0 = time.time()
    spent = lambda: time.time() - t0
    fr, disk = GW.regen_check()
    pins = fr["pins"]
    assert sha256_file(os.path.join(
        ROOT, "v2conf_diag/s4_closed.py")) == pins["v2conf_diag/s4_closed.py"]
    for s in SEEDS:
        assert sha256_file(os.path.join(
            ROOT, f"pilot3_runs/P3_{s}_theta.npz")) == pins[
                f"pilot3_runs/P3_{s}_theta.npz"], s
    me = json.load(open(os.path.join(W1, "w1_es_models.json")))
    mi = json.load(open(os.path.join(W1, "w1_imit_models.json")))
    arms = {"W1-final": {s: np.array(me[f"W1E_{s}"]) for s in SEEDS},
            "W1-imit": {s: np.array(mi[f"W1I_{s}"]) for s in SEEDS}}
    for a in arms:
        for s in SEEDS:
            assert len(arms[a][s]) == 612
    g1 = disk["pools"]["gate_g1"]
    g2 = disk["pools"]["gate_g2"]
    sg = disk["pools"]["sing_gate"]
    assert len(g1) == 200 and len(g2) == 200 and len(sg) == 80
    assert all(e["family"] == "G1" for e in g1)
    assert all(e["family"] == "G2" for e in g2)
    train_ids = {e["id"] for e in disk["pools"]["demo_g1"]}
    train_ids |= {e["id"] for e in disk["pools"]["espool_g1"]}
    gate_ids = {e["id"] for e in g1 + g2 + sg}
    assert not (train_ids & gate_ids), "gate/train overlap"
    skills = {s: split_modular(np.load(os.path.join(
        ROOT, f"pilot3_runs/P3_{s}_theta.npz"))["theta"])[0] for s in SEEDS}
    out_p = os.path.join(W1, "w1_gate.json")
    try:
        out = json.load(open(out_p))
        done = {(r["arm"], r["id"], r["theta"]) for r in out["records"]
                if r["status"] == "complete"}
        sdone = {(r["id"], r["theta"]) for r in out.get("sing_records", [])
                 if r["status"] == "complete"}
        print(f"resumed: {len(done)} + {len(sdone)} sing kept", flush=True)
    except FileNotFoundError:
        out = {"records": [], "sing_records": []}
        done, sdone = set(), set()
    sync = lambda: json.dump(out, open(out_p, "w"))
    for arm in ("W1-final", "W1-imit"):
        for s in SEEDS:
            for e in g1 + g2:
                if (arm, e["id"], s) in done:
                    continue
                if spent() > T_BUDGET - T_MARGIN:
                    sync()
                    print(f"INCOMPLETE cum={spent():.0f}s", flush=True)
                    return
                r = rollout(arms[arm][s], skills[s], e)
                out["records"].append(
                    {"arm": arm, "id": e["id"], "theta": s,
                     "fam": e["family"], "status": "complete", **r})
                if len(out["records"]) % 25 == 0:
                    sync()
    for s in SEEDS:
        for e in sg:
            if (e["id"], s) in sdone:
                continue
            if spent() > T_BUDGET - T_MARGIN:
                sync()
                print(f"INCOMPLETE cum={spent():.0f}s", flush=True)
                return
            r = rollout(arms["W1-final"][s], skills[s], e)
            out["sing_records"].append(
                {"id": e["id"], "theta": s, "fam": e["family"],
                 "status": "complete", **r})
            if len(out["sing_records"]) % 25 == 0:
                sync()
    recs = [r for r in out["records"] if r["status"] == "complete"]
    assert len(recs) == 2400, len(recs)
    srecs = [r for r in out["sing_records"] if r["status"] == "complete"]
    assert len(srecs) == 240, len(srecs)
    summ = {}
    for arm in ("W1-final", "W1-imit"):
        summ[arm] = {}
        for fam, pool in (("G1", g1), ("G2", g2)):
            pr = []
            for s in SEEDS:
                sub = [r for r in recs if r["arm"] == arm
                       and r["theta"] == s and r["fam"] == fam]
                assert len(sub) == 200, (arm, s, fam, len(sub))
                pr.append(sum(1 for r in sub
                              if r["cause"] == "arrival") / 200)
            summ[arm][fam] = {"per_theta": [round(x, 4) for x in pr],
                              "mean": round(float(np.mean(pr)), 4)}
    verdict = {fam: bool(summ["W1-final"][fam]["mean"] >= BAR)
               for fam in ("G1", "G2")}
    verdict["PASS"] = bool(verdict["G1"] and verdict["G2"])
    sing = {}
    for f in "ABCD":
        sub = [r for r in srecs if r["fam"] == f]
        assert len(sub) == 60, (f, len(sub))
        sing[f] = {"arr": sum(1 for r in sub if r["cause"] == "arrival"),
                   "n": 60}
    out["summary"] = summ
    out["sing_spot"] = sing
    out["verdict"] = verdict
    out["wall_s"] = round(spent(), 1)
    sync()
    print(f"gate G1 W1-final={summ['W1-final']['G1']} "
          f"W1-imit={summ['W1-imit']['G1']}", flush=True)
    print(f"gate G2 W1-final={summ['W1-final']['G2']} "
          f"W1-imit={summ['W1-imit']['G2']}", flush=True)
    print(f"singles spot W1-final: {sing}", flush=True)
    print(f"GATE {'PASS' if verdict['PASS'] else 'FAIL'} "
          f"(bar {BAR} each family) wall={spent():.0f}s", flush=True)


if __name__ == "__main__":
    main()
