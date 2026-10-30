"""S4 closed-loop eval of final S3 students (no training).

27 episodes: S3 eval partition (9 scenarios) x thetas 301-303.
Student: epoch-500 gate intact; per step argmax over 4 gate outputs
-> greedy action of chosen skill with its view. No C/D masking,
no fallback; C/D occurrences recorded. Current 14 obs only; no
privileged state, no planner. Original world/seeds/horizon.

Baselines on the same episodes: (1) registered P-AB teacher
(s2_30.json, NOT re-queried); (2) train-majority constant skill,
greedy; (3) distance-15 rule, greedy skills; (4) original v2conf
gate, greedy viewed blend. Hard-routing+greedy diagnostic; NOT
the soft+sampled protocol; no verdict changes. Budget 300 s.
"""
import json
import sys
import time

sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/src")
sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/v2views")

import numpy as np

import views_train as V
from env import Arena
from models import GATE_SHAPES, SKILL_SHAPES, mlp_forward, split_modular
from policies import softmax
from tasks import TaskRunner

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
SEEDS = (301, 302, 303)
T_BUDGET = 300.0


def cause_of(runner, info):
    env = runner.env
    if info["success"]:
        return "arrival"
    if float(np.linalg.norm(env.haz_pos - env.agent_pos)) < 5.0:
        return "contact-death"
    if env.t >= env.MAX_STEPS:
        return "timeout"
    return "other"


def greedy_skill(sk_all, obs, k):
    q = softmax(mlp_forward(sk_all[k], V.view_obs(obs, k), SKILL_SHAPES))
    return int(np.argmax(q))


def run_ep(e, step_fn):
    r = TaskRunner()
    obs = r.reset(e["family"], e["cfg"], int(e["seed"]))
    assert r.env.haz_pos is not None
    done, info, n = False, {}, 0
    uses, sw, prev, hobs = [0, 0, 0, 0], 0, None, []
    while not done:
        hobs.append(round(float(obs[7]) * Arena.DMAX, 2))
        a, k = step_fn(obs)
        uses[k] += 1
        sw += (prev is not None and k != prev)
        prev = k
        obs, _, done, info = r.step(a)
        n += 1
    return {"cause": cause_of(r, info), "T": n, "uses": uses,
            "switches": sw, "hobs": hobs}


def main():
    t0 = time.time()
    evc = json.load(open(f"{ROOT}/v2conf/eval_conf_v1.json"))
    by_id = {e["id"]: e for e in evc if e["battery"] == "ACQ"}
    s3 = json.load(open(f"{ROOT}/v2conf_diag/s3_results.json"))
    eval_ids = s3["partition"]["eval"]
    assert len(eval_ids) == 9, eval_ids
    models = json.load(open(f"{ROOT}/v2conf_diag/s3_models.json"))
    s2 = json.load(open(f"{ROOT}/v2conf_diag/s2_30.json"))
    teach = {(r["id"], r["theta"]): r for r in s2["records"]
             if r["mode"] == "AB" and r["status"] == "complete"}

    skills, vct = {}, {}
    for s in SEEDS:
        th = np.load(f"{ROOT}/v2conf_runs/VC_{s}_theta.npz")["theta"]
        skills[s] = split_modular(th)[0]
        vct[s] = th

    out = {"eval_ids": eval_ids, "records": []}

    def student_fn(sk_all, gth):
        def f(obs):
            k = int(np.argmax(mlp_forward(gth, obs, GATE_SHAPES)))
            return greedy_skill(sk_all, obs, k), k
        return f

    def const_fn(sk_all, k):
        def f(obs):
            return greedy_skill(sk_all, obs, k), k
        return f

    def d15_fn(sk_all):
        def f(obs):
            k = 1 if float(obs[7]) * Arena.DMAX < 15 else 0
            return greedy_skill(sk_all, obs, k), k
        return f

    def vgate_fn(th):
        def f(obs):
            p, w = V.viewed_blend(th, obs)
            return int(np.argmax(p)), int(np.argmax(w))
        return f

    for s in SEEDS:
        gth = np.array(models[str(s)]["theta"])
        assert len(gth) == sum(int(np.prod(x)) for x in GATE_SHAPES)
        maj = s3["thetas"][str(s)]["majority_train"]
        arms = {"student": student_fn(skills[s], gth),
                "const": const_fn(skills[s], maj),
                "d15": d15_fn(skills[s]),
                "vgate": vgate_fn(vct[s])}
        for eid in eval_ids:
            e = by_id[eid]
            row = {"theta": s, "id": eid}
            for name, fn in arms.items():
                if time.time() - t0 > T_BUDGET - 5:
                    row[name] = {"status": "incomplete"}
                    continue
                rec = run_ep(e, fn)
                rec["status"] = "complete"
                row[name] = rec
            t = teach[(eid, s)]
            row["teacher"] = {"cause": t["cause"], "T": t["T"],
                              "frac_B": t.get("frac_B"),
                              "switches": t.get("switches"),
                              "source": "registered"}
            out["records"].append(row)
            st = row["student"]
            print(f"th={s} {eid}: student=" +
                  (f"{st['cause']} T={st['T']} uses={st['uses']} "
                   f"sw={st['switches']}" if st.get("status") == "complete"
                   else "incomplete") +
                  f" | teacher={row['teacher']['cause']}", flush=True)

    out["wall_s"] = round(time.time() - t0, 1)
    json.dump(out, open(f"{ROOT}/v2conf_diag/s4_results.json", "w"))
    n_inc = sum(1 for r in out["records"] for a in
                ("student", "const", "d15", "vgate")
                if r[a].get("status") != "complete")
    print(f"saved s4_results.json wall={out['wall_s']}s "
          f"incomplete={n_inc}", flush=True)


if __name__ == "__main__":
    main()
