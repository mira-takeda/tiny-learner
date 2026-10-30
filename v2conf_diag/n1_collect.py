"""N1 demo collection: P-AB planner on 18 fixed G1 + P-AC planner on
18 fixed G2 (train-manifest, permitted), x3 thetas = 108 runs.
Per real step: obs14 pre-action, skill label, action. Frozen:
n1_freeze.json. Budget 1800 s cumulative, incremental, resume.
"""
import json
import sys
import time

sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/src")
sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/v2views")
sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/v2conf_diag")

import numpy as np

import ref_plan as G1
import n0_g2 as G2
from models import split_modular
from tasks import TaskRunner

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
assert (G1.H, G1.K, G1.NBLOCK, G1.BLEN) == (25, 40, 5, 5)
SEEDS = (301, 302, 303)
T_BUDGET, T_MARGIN = 1800.0, 10.0


def sha(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def base():
    try:
        return float(json.load(
            open(f"{ROOT}/v2conf_diag/n1_budget.json"))["spent_s"])
    except FileNotFoundError:
        return 0.0


def main():
    t0 = time.time()
    b0 = base()
    spent = lambda: b0 + time.time() - t0
    fr = json.load(open(f"{ROOT}/v2conf_diag/n1_freeze.json"))
    assert sha(f"{ROOT}/data/train_meta_v1.json") == fr["data_sha"]
    d = json.load(open(f"{ROOT}/data/train_meta_v1.json"))
    by_id = {e["id"]: e for e in d}
    demos = ([(by_id[i], "G1") for i in fr["demos"]["G1"]] +
             [(by_id[i], "G2") for i in fr["demos"]["G2"]])
    assert len(demos) == 36
    skills = {}
    for s in SEEDS:
        th = np.load(f"{ROOT}/v2conf_runs/VC_{s}_theta.npz")["theta"]
        p3 = np.load(f"{ROOT}/pilot3_runs/P3_{s}_theta.npz")["theta"]
        sk, _ = split_modular(th)
        assert all(np.array_equal(a, b)
                   for a, b in zip(sk, split_modular(p3)[0])), s
        skills[s] = sk
    plan = [(e, fam, s) for e, fam in demos for s in SEEDS]
    assert len(plan) == 108
    kids = [int(x.generate_state(1, dtype=np.uint32)[0])
            for x in np.random.SeedSequence(
                fr["demos"]["seq_seed"]).spawn(len(plan))]
    try:
        out = json.load(open(f"{ROOT}/v2conf_diag/n1_demos.json"))
        done = {(r["id"], r["theta"]) for r in out["runs"]
                if r["status"] == "complete"}
        print(f"resumed: {len(done)} kept", flush=True)
    except FileNotFoundError:
        out = {"runs": []}
        done = set()
    sync = lambda: (json.dump(out, open(
        f"{ROOT}/v2conf_diag/n1_demos.json", "w")),
        json.dump({"spent_s": round(spent(), 1)}, open(
            f"{ROOT}/v2conf_diag/n1_budget.json", "w")))
    for xi, (e, fam, s) in enumerate(plan):
        if (e["id"], s) in done:
            continue
        if spent() > T_BUDGET - T_MARGIN:
            out["runs"].append({"id": e["id"], "theta": s, "fam": fam,
                                "status": "incomplete",
                                "reason": "budget-before-start"})
            sync()
            continue
        rec = {"id": e["id"], "theta": s, "fam": fam,
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
            if fam == "G1":
                (a, k), _ = G1.shoot(G1.snap(r), r.prev_goal, seqs,
                                     skills[s], False)
            else:
                st, tr = G2.snap(r)
                (a, k), _ = G2.shoot(st, tr, seqs, skills[s])
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
        print(f"{fam} {e['id']} th={s}: {st} "
              f"{'ok=' + str(rec.get('success')) + ' T=' + str(n) + ' ' if st == 'complete' else ''}"
              f"wall={rec['wall_s']}s cum={spent():.0f}s", flush=True)
    n_inc = sum(1 for r in out["runs"] if r["status"] != "complete")
    print(f"N1-collect {'done' if n_inc == 0 else 'INCOMPLETE'} "
          f"cumulative={spent():.0f}s incomplete={n_inc}", flush=True)


if __name__ == "__main__":
    main()
