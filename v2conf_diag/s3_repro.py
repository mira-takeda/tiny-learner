"""S3(b): reproduce the 9 failed AB runs with recorded planner seeds.

Same search code (ref_plan.shoot/model, H/K/blocks/scoring), same
episode, same planner seed -> must reproduce the identical solution.
Verifies cause, T, frac_B, switches, imag_steps vs s2_30.json
(return was not logged originally; imag_steps + per-step equality
is the stronger check). Any mismatch -> STOP. Saves failure traces
+ reproduction cost. No new searches, no other solutions.
"""
import json
import sys
import time

sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/src")
sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/v2views")
sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/v2conf")
sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/v2conf_diag")

import numpy as np

import ref_plan as R
from conf_train import EVCONF_SHA
from models import split_modular
from tasks import TaskRunner

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
assert (R.H, R.K, R.NBLOCK, R.BLEN) == (25, 40, 5, 5)
SEEDS = (301, 302, 303)


def sha(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    t0 = time.time()
    assert sha(f"{ROOT}/v2conf/eval_conf_v1.json") == EVCONF_SHA
    evc = json.load(open(f"{ROOT}/v2conf/eval_conf_v1.json"))
    by_id = {e["id"]: e for e in evc if e["battery"] == "ACQ"}
    s2 = json.load(open(f"{ROOT}/v2conf_diag/s2_30.json"))
    fails = [r for r in s2["records"] if r["mode"] == "AB"
             and r["status"] == "complete" and r["cause"] != "arrival"]
    assert len(fails) == 9, len(fails)
    assert all(r["source"] == "s2" for r in fails)
    print(f"reproducing {len(fails)} failures: "
          f"{[(r['id'], r['theta']) for r in fails]}", flush=True)

    skills = {}
    for s in SEEDS:
        th = np.load(f"{ROOT}/v2conf_runs/VC_{s}_theta.npz")["theta"]
        skills[s] = split_modular(th)[0]

    traces, rows = {}, []
    for r in fails:
        e, s = by_id[r["id"]], r["theta"]
        seq_rng = np.random.default_rng(int(r["planner_seed"]))
        runner = TaskRunner()
        runner.reset(e["family"], e["cfg"], int(e["seed"]))
        n_real, imag, nb, sw, prev = 0, 0, 0, 0, None
        done, info, steps, ret = False, {}, [], 0.0
        ep_t0 = time.time()
        while not done:
            st = R.snap(runner)
            seqs = seq_rng.integers(0, 2, size=(R.K, R.NBLOCK))
            (a, blk), im = R.shoot(st, runner.prev_goal, seqs, skills[s],
                                   False)
            imag += im
            nb += (blk == 1)
            sw += (prev is not None and blk != prev)
            prev = blk
            steps.append({"t": n_real,
                          "apos": [round(float(x), 3) for x in st["apos"]],
                          "avel": [round(float(x), 3) for x in st["avel"]],
                          "haz": [round(float(x), 3) for x in st["haz"]],
                          "a": int(a), "skill": int(blk)})
            _, rw, done, info = runner.step(a)
            ret += rw
            n_real += 1
        env = runner.env
        if info["success"]:
            cause = "arrival"
        elif float(np.linalg.norm(env.haz_pos - env.agent_pos)) < 5.0:
            cause = "contact-death"
        elif env.t >= env.MAX_STEPS:
            cause = "timeout"
        else:
            cause = "other"
        got = {"cause": cause, "T": n_real, "frac_B": round(nb / n_real, 4),
               "switches": sw, "imag_steps": imag}
        exp = {k: r[k] for k in got}
        assert got == exp, f"MISMATCH {r['id']} th={s}: {got} vs {exp}"
        key = f"AB_{e['id']}_{s}"
        traces[key] = {"id": e["id"], "mode": "AB", "theta": s,
                       "planner_seed": r["planner_seed"], "T": n_real,
                       "cause": cause, "return": round(ret, 4),
                       "steps": steps}
        rows.append({"id": e["id"], "theta": s, "match": True,
                     "wall_s": round(time.time() - ep_t0, 1)})
        print(f"  {e['id']} th={s}: MATCH {cause} T={n_real} "
              f"imag={imag}", flush=True)

    out = {"n": len(rows), "all_match": True,
           "repro_wall_s": round(time.time() - t0, 1), "runs": rows}
    json.dump(out, open(f"{ROOT}/v2conf_diag/s3_repro.json", "w"), indent=1)
    json.dump(traces, open(f"{ROOT}/v2conf_diag/s3_repro_traces.json", "w"))
    print(f"9/9 reproduced exactly; repro_wall={out['repro_wall_s']}s; "
          f"saved s3_repro.json + s3_repro_traces.json", flush=True)


if __name__ == "__main__":
    main()
