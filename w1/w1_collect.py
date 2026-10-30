"""W1 demo collection: P-AB planner on 18 wide-G1 (frozen) x3 thetas.

Mirrors n1_collect (G1 part): per real step obs14 pre-action + skill
label + action; planner ref_plan.shoot/snap asserted (H25/K40/5x5);
failures included (N1 protocol). Frozen: w1_freeze.json + MANIFEST.
Budget 1800 s cumulative, incremental resume.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "v2views"))
sys.path.insert(0, os.path.join(ROOT, "v2conf"))
sys.path.insert(0, os.path.join(ROOT, "v2conf_diag"))
sys.path.insert(0, HERE)

import numpy as np

import ref_plan as G1
from gen import sha256_file
from models import split_modular
from tasks import TaskRunner
import gen_wide as GW

W1 = HERE
SEEDS = (301, 302, 303)
T_BUDGET, T_MARGIN = 1800.0, 10.0


def main():
    assert (G1.H, G1.K, G1.NBLOCK, G1.BLEN) == (25, 40, 5, 5)
    t0 = time.time()
    fr, disk = GW.regen_check()
    pins = fr["pins"]
    assert sha256_file(os.path.join(
        ROOT, "v2conf_diag/ref_plan.py")) == pins["v2conf_diag/ref_plan.py"]
    for s in SEEDS:
        assert sha256_file(os.path.join(
            ROOT, f"pilot3_runs/P3_{s}_theta.npz")) == pins[
                f"pilot3_runs/P3_{s}_theta.npz"], s
    demos = disk["pools"]["demo_g1"]
    assert len(demos) == 18
    skills = {}
    for s in SEEDS:
        p3 = np.load(os.path.join(
            ROOT, f"pilot3_runs/P3_{s}_theta.npz"))["theta"]
        assert len(p3) == 7944, s
        skills[s] = split_modular(p3)[0]
    plan = [(e, s) for e in demos for s in SEEDS]
    assert len(plan) == 54
    kids = GW.kid_seeds(fr["seed_block"]["planner"], 54)
    out_p = os.path.join(W1, "w1_demos_g1.json")
    bud_p = os.path.join(W1, "w1_collect_budget.json")
    try:
        out = json.load(open(out_p))
        done = {(r["id"], r["theta"]) for r in out["runs"]
                if r["status"] == "complete"}
        print(f"resumed: {len(done)} kept", flush=True)
    except FileNotFoundError:
        out = {"runs": []}
        done = set()
    try:
        b0 = float(json.load(open(bud_p))["spent_s"])
    except FileNotFoundError:
        b0 = 0.0
    spent = lambda: b0 + time.time() - t0
    sync = lambda: (json.dump(out, open(out_p, "w")),
                    json.dump({"spent_s": round(spent(), 1)},
                              open(bud_p, "w")))
    for xi, (e, s) in enumerate(plan):
        if (e["id"], s) in done:
            continue
        if spent() > T_BUDGET - T_MARGIN:
            out["runs"].append({"id": e["id"], "theta": s,
                                "status": "incomplete",
                                "reason": "budget-before-start"})
            sync()
            continue
        rec = {"id": e["id"], "theta": s, "fam": "G1",
               "planner_seed": kids[xi], "steps": []}
        r = TaskRunner()
        obs = r.reset(e["family"], e["cfg"], int(e["seed"]))
        seq_rng = np.random.default_rng(kids[xi])
        done_ep, info, tot, n, abort = False, {}, 0.0, 0, False
        ep_t0 = time.time()
        while not done_ep:
            if spent() > T_BUDGET - T_MARGIN:
                abort = True
                break
            seqs = seq_rng.integers(0, 2, size=(G1.K, G1.NBLOCK))
            (a, k), _ = G1.shoot(G1.snap(r), r.prev_goal, seqs,
                                 skills[s], False)
            rec["steps"].append({"obs": [float(x) for x in obs],
                                 "k": int(k), "a": int(a)})
            obs, rew, done_ep, info = r.step(a)
            tot += rew
            n += 1
        rec["return"] = round(tot, 4)
        rec["wall_s"] = round(time.time() - ep_t0, 1)
        if abort:
            rec.update({"status": "incomplete",
                        "reason": "budget-mid-episode"})
            del rec["steps"]
        else:
            rec.update({"status": "complete",
                        "success": bool(info["success"]), "T": n})
        out["runs"].append(rec)
        sync()
        st = rec["status"]
        print(f"G1 {e['id']} th={s}: {st} "
              f"{'ok=' + str(rec.get('success')) + ' T=' + str(n) + ' ' if st == 'complete' else ''}"
              f"wall={rec['wall_s']}s cum={spent():.0f}s", flush=True)
    for s in SEEDS:
        assert sha256_file(os.path.join(
            ROOT, f"pilot3_runs/P3_{s}_theta.npz")) == pins[
                f"pilot3_runs/P3_{s}_theta.npz"], s
    n_inc = sum(1 for r in out["runs"] if r["status"] != "complete")
    print(f"W1-collect {'done' if n_inc == 0 else 'INCOMPLETE'} "
          f"cumulative={spent():.0f}s incomplete={n_inc}", flush=True)


if __name__ == "__main__":
    main()
