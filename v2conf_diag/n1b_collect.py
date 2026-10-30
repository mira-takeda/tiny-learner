"""N1b singles collection: constant-greedy rollouts on fixed
train-manifest A/B/C/D singles (disjoint from heldout, N2 pools,
N1 demos). Per step: obs14 pre-action + trivial family label.
Frozen: n1b_freeze.json (600 train + 200 eval pairs/class,
exact truncation). Cheap; incremental anyway. Budget 600 s.
"""
import json
import sys
import time

sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/src")
sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/v2views")

import numpy as np

import views_train as V
from models import SKILL_SHAPES, mlp_forward, split_modular
from policies import softmax
from tasks import TaskRunner

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
SEEDS = (301, 302, 303)
FAM2K = {"A": 0, "B": 1, "C": 2, "D": 3}
T_BUDGET, T_MARGIN = 600.0, 10.0


def sha(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    t0 = time.time()
    spent = lambda: time.time() - t0
    fr = json.load(open(f"{ROOT}/v2conf_diag/n1b_freeze.json"))
    assert sha(f"{ROOT}/data/train_meta_v1.json") == fr["data_sha"]
    assert sha(f"{ROOT}/v2conf_diag/n1_demos.json") == fr["g1g2_sha"]
    d = json.load(open(f"{ROOT}/v2conf_diag/n1_demos.json"))["runs"]
    assert len(d) == 108 and all(r["status"] == "complete" for r in d)
    by_id = {e["id"]: e for e in
             json.load(open(f"{ROOT}/data/train_meta_v1.json"))}
    skills = {s: split_modular(np.load(
        f"{ROOT}/v2conf_runs/VC_{s}_theta.npz")["theta"])[0] for s in SEEDS}
    try:
        out = json.load(open(f"{ROOT}/v2conf_diag/n1b_singles.json"))
        have = {(r["id"], r["theta"]) for r in out["runs"]}
        print(f"resumed: {len(have)} kept", flush=True)
    except FileNotFoundError:
        out = {"runs": []}
        have = set()
    sync = lambda: json.dump(out, open(
        f"{ROOT}/v2conf_diag/n1b_singles.json", "w"))
    # Collect whole episodes per class until train+eval targets reachable.
    need = {f: fr["singles"]["train_pairs_per_class"] +
            fr["singles"]["eval_pairs_per_class"] for f in "ABCD"}
    def per_theta_min(fam):
        return min(sum(r["T"] for r in out["runs"]
                       if r["fam"] == fam and r["theta"] == s)
                   for s in SEEDS)
    got = {f: per_theta_min(f) for f in "ABCD"}
    for f in "ABCD":
        for eid in fr["singles"]["pools"][f]:
            if got[f] >= need[f]:
                break
            for s in SEEDS:
                if (eid, s) in have:
                    continue
                if spent() > T_BUDGET - T_MARGIN:
                    sync()
                    print(f"INCOMPLETE cum={spent():.0f}s", flush=True)
                    return
                e = by_id[eid]
                k = FAM2K[f]
                r = TaskRunner()
                obs = r.reset(e["family"], e["cfg"], int(e["seed"]))
                steps, done, info, n = [], False, {}, 0
                while not done:
                    a = int(np.argmax(softmax(mlp_forward(
                        skills[s][k], V.view_obs(obs, k), SKILL_SHAPES))))
                    steps.append([float(x) for x in obs])
                    obs, _, done, info = r.step(a)
                    n += 1
                # Count steps once per episode (lengths ~equal across
                # thetas? NO: count per theta separately below).
                out["runs"].append({"id": eid, "theta": s, "fam": f,
                                    "k": k, "T": n,
                                    "success": bool(info["success"]),
                                    "steps": steps})
                have.add((eid, s))
            got[f] = per_theta_min(f)
        sync()
    # Truncate per class+theta to exact targets (train first, eval next).
    final = {"train": {}, "eval": {}}
    for f in "ABCD":
        for s in SEEDS:
            runs = [r for r in out["runs"]
                    if r["fam"] == f and r["theta"] == s]
            flat = [o for r in runs for o in r["steps"]]
            nt = fr["singles"]["train_pairs_per_class"]
            nv = fr["singles"]["eval_pairs_per_class"]
            assert len(flat) >= nt + nv, (f, s, len(flat))
            final["train"][(f, s)] = flat[:nt]
            final["eval"][(f, s)] = flat[nt:nt + nv]
    slim = {"train": {f"{f}_{s}": v for (f, s), v in final["train"].items()},
            "eval": {f"{f}_{s}": v for (f, s), v in final["eval"].items()}}
    json.dump(slim, open(f"{ROOT}/v2conf_diag/n1b_pairs.json", "w"))
    print(f"N1b-collect done cum={spent():.0f}s runs={len(out['runs'])}; "
          f"pairs: train 600x4, eval 200x4 per theta", flush=True)


if __name__ == "__main__":
    main()
