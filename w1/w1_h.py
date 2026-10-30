"""W1 H opening (SINGLE OPENING, ONLY if gate PASS).

Reads H1/H2/H3 entries ONLY from test_battery (asserted; I1 and all
other batteries never touched). Arms {W1-final, A/B/C/D-only} x 3H
x 200 x 3 thetas = 9000 eps. Reference bars are DESCRIPTIVE
follow-up (NOT preregistered): each H >= 0.60; H-mean >=
best-single + 0.40. No monolith (pending, declared). Frozen: freeze
+ MANIFEST. Budget 1200 s, resume.
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
T_BUDGET, T_MARGIN = 1200.0, 10.0
T2 = 4.303  # t(2) 95% two-sided (S1 convention, n=3)


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
    fr = GW.check_manifest()
    pins = fr["pins"]
    gate = json.load(open(os.path.join(W1, "w1_gate.json")))
    assert gate["verdict"]["PASS"] is True, "gate did not PASS: H stays sealed"
    assert sha256_file(os.path.join(
        ROOT, "v2conf_diag/s4_closed.py")) == pins["v2conf_diag/s4_closed.py"]
    assert sha256_file(os.path.join(
        ROOT, "data/test_battery_v1.json")) == pins["data/test_battery_v1.json"]
    for s in SEEDS:
        assert sha256_file(os.path.join(
            ROOT, f"pilot3_runs/P3_{s}_theta.npz")) == pins[
                f"pilot3_runs/P3_{s}_theta.npz"], s
    me = json.load(open(os.path.join(W1, "w1_es_models.json")))
    gw = {s: np.array(me[f"W1E_{s}"]) for s in SEEDS}
    for s in SEEDS:
        assert len(gw[s]) == 612
    tb = json.load(open(os.path.join(ROOT, "data/test_battery_v1.json")))
    h = [e for e in tb if e.get("family") in ("H1", "H2", "H3")]
    assert len(h) == 600, len(h)
    for fam in ("H1", "H2", "H3"):
        assert sum(1 for e in h if e["family"] == fam) == 200, fam
    assert all(e.get("family") in ("H1", "H2", "H3") for e in h)
    skills = {s: split_modular(np.load(os.path.join(
        ROOT, f"pilot3_runs/P3_{s}_theta.npz"))["theta"])[0] for s in SEEDS}
    out_p = os.path.join(W1, "w1_h.json")
    try:
        out = json.load(open(out_p))
        done = {(r["arm"], r["id"], r["theta"]) for r in out["records"]
                if r["status"] == "complete"}
        print(f"resumed: {len(done)} kept", flush=True)
    except FileNotFoundError:
        out = {"records": []}
        done = set()
    sync = lambda: json.dump(out, open(out_p, "w"))
    arms = [("W1-final", None), ("A-only", 0), ("B-only", 1),
            ("C-only", 2), ("D-only", 3)]
    for arm, const_k in arms:
        for s in SEEDS:
            for e in h:
                if (arm, e["id"], s) in done:
                    continue
                if spent() > T_BUDGET - T_MARGIN:
                    sync()
                    print(f"INCOMPLETE cum={spent():.0f}s", flush=True)
                    return
                r = TaskRunner()
                obs = r.reset(e["family"], e["cfg"], int(e["seed"]))
                total, done_ep, info, n = 0.0, False, {}, 0
                uses, sw, prev = [0, 0, 0, 0], 0, None
                while not done_ep:
                    k = const_k if const_k is not None else int(
                        np.argmax(mlp_forward(gw[s], obs, GATE_SHAPES)))
                    a = S4.greedy_skill(skills[s], obs, k)
                    uses[k] += 1
                    sw += (prev is not None and k != prev)
                    prev = k
                    obs, rew, done_ep, info = r.step(a)
                    total += rew
                    n += 1
                out["records"].append(
                    {"arm": arm, "id": e["id"], "theta": s,
                     "fam": e["family"], "status": "complete",
                     "cause": cause_of(r, info), "ret": round(total, 3),
                     "T": n, "uses": uses, "switches": sw})
                if len(out["records"]) % 50 == 0:
                    sync()
    recs = [r for r in out["records"] if r["status"] == "complete"]
    assert len(recs) == 9000, len(recs)
    summ = {}
    for arm, _ in arms:
        summ[arm] = {}
        for fam in ("H1", "H2", "H3"):
            pr = []
            for s in SEEDS:
                sub = [r for r in recs if r["arm"] == arm
                       and r["theta"] == s and r["fam"] == fam]
                assert len(sub) == 200, (arm, s, fam, len(sub))
                pr.append(sum(1 for r in sub
                              if r["cause"] == "arrival") / 200)
            m, sd = float(np.mean(pr)), float(np.std(pr, ddof=1))
            summ[arm][fam] = {"per_theta": [round(x, 4) for x in pr],
                              "mean": round(m, 4),
                              "ci95": [round(m - T2 * sd / np.sqrt(3), 4),
                                       round(m + T2 * sd / np.sqrt(3), 4)]}
    ref = {}
    for fam in ("H1", "H2", "H3"):
        ref[fam] = {"w1": summ["W1-final"][fam]["mean"],
                    "bar_060": bool(summ["W1-final"][fam]["mean"] >= 0.60)}
    mh = {a: float(np.mean([summ[a][f]["mean"] for f in ("H1", "H2", "H3")]))
          for a, _ in arms}
    best1 = max(mh[a] for a in ("A-only", "B-only", "C-only", "D-only"))
    ref["H-mean"] = {"w1": round(mh["W1-final"], 4),
                     "best_single": round(best1, 4),
                     "margin": round(mh["W1-final"] - best1, 4),
                     "bar_p40": bool(mh["W1-final"] - best1 >= 0.40)}
    out["summary"] = summ
    out["reference"] = ref
    out["wall_s"] = round(spent(), 1)
    sync()
    for fam in ("H1", "H2", "H3"):
        row = {a: summ[a][fam]["mean"] for a, _ in arms}
        print(f"{fam}: {row} ref060={ref[fam]['bar_060']}", flush=True)
    print(f"H-mean margin vs best-single: {ref['H-mean']} "
          f"(descriptive follow-up refs) wall={spent():.0f}s", flush=True)


if __name__ == "__main__":
    main()
