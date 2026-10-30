"""S2 expansion: privileged reference planner on 30 ACQ (10/region).

Same search as ref_plan (imported, asserted identical: H=25, K=40,
5x5 blocks, return scoring, noiseless model, replan per real step).
30 IDs = file indices 0..29 (first 10/region); the 12 complete
micropilot runs are REUSED (not repeated); 27 new episodes run here:
P-prim x27, then P-AB x27x3 (episode-major, thetas 301-303).
Per-execution planner seeds from SeedSequence(8101) spawn, fixed
order, recorded. Solution traces conserved. TOTAL budget 900 s;
incremental save; interrupted = INCOMPLETE, not FAIL.
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
SEQ_BASE = 8101
T_BUDGET, T_MARGIN = 900.0, 5.0


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
    acq = [e for e in evc if e["battery"] == "ACQ"]
    eps30 = acq[:30]
    assert all(sum(1 for e in eps30 if e["region"] == r) == 10
               for r in ("G1a", "G1b", "G1adv"))
    new_eps = acq[3:30]
    assert len(new_eps) == 27

    skills = {}
    for s in SEEDS:
        th = np.load(f"{ROOT}/v2conf_runs/VC_{s}_theta.npz")["theta"]
        p3 = np.load(f"{ROOT}/pilot3_runs/P3_{s}_theta.npz")["theta"]
        sk, _ = split_modular(th)
        assert all(np.array_equal(a, b)
                   for a, b in zip(sk, split_modular(p3)[0])), s
        skills[s] = sk

    # Reuse micropilot completes (stream 8100, fixed order, not repeated).
    micro = json.load(open(f"{ROOT}/v2conf_diag/refplan_pilot.json"))
    reused = []
    for i, r in enumerate(micro["records"]):
        assert r["status"] == "complete", r
        reused.append({**r, "source": "micropilot",
                       "planner_stream": 8100, "exec_index": i,
                       "trace": None})
    print(f"reused {len(reused)} micropilot completes", flush=True)

    plan = [("prim", e, None) for e in new_eps] + \
           [("AB", e, s) for e in new_eps for s in SEEDS]
    assert len(plan) == 108
    kids = [int(s.generate_state(1, dtype=np.uint32)[0])
            for s in np.random.SeedSequence(SEQ_BASE).spawn(len(plan))]

    out = {"H": R.H, "K": R.K, "blocks": [R.NBLOCK, R.BLEN],
           "seq_base": SEQ_BASE, "budget_s": T_BUDGET,
           "ep30_ids": [e["id"] for e in eps30],
           "records": list(reused)}
    traces = {}

    def save():
        json.dump(out, open(f"{ROOT}/v2conf_diag/s2_30.json", "w"))
        json.dump(traces, open(f"{ROOT}/v2conf_diag/s2_traces.json", "w"))

    for pi, ((mode, e, s), pseed) in enumerate(zip(plan, kids)):
        if time.time() - t0 > T_BUDGET - T_MARGIN:
            out["records"].append(
                {"mode": mode, "id": e["id"], "theta": s, "source": "s2",
                 "planner_seed": pseed, "status": "incomplete",
                 "reason": "budget-exhausted-before-start"})
            continue
        prim = mode == "prim"
        sk_all = None if prim else skills[s]
        seq_rng = np.random.default_rng(pseed)
        runner = TaskRunner()
        runner.reset(e["family"], e["cfg"], int(e["seed"]))
        n_real, imag, nb, sw, prev = 0, 0, 0, 0, None
        done, info, abort, steps = False, {}, False, []
        ep_t0 = time.time()
        while not done:
            if time.time() - t0 > T_BUDGET - T_MARGIN:
                abort = True
                break
            st = R.snap(runner)
            if prim:
                seqs = seq_rng.integers(0, 9, size=(R.K, R.NBLOCK))
            else:
                seqs = seq_rng.integers(0, 2, size=(R.K, R.NBLOCK))
            (a, blk), im = R.shoot(st, runner.prev_goal, seqs, sk_all,
                                   prim)
            imag += im
            if not prim:
                nb += (blk == 1)
                sw += (prev is not None and blk != prev)
                prev = blk
            steps.append({"t": n_real,
                          "apos": [round(float(x), 3) for x in st["apos"]],
                          "avel": [round(float(x), 3) for x in st["avel"]],
                          "haz": [round(float(x), 3) for x in st["haz"]],
                          "a": int(a), "skill": None if prim else int(blk)})
            _, _, done, info = runner.step(a)
            n_real += 1
        rec = {"mode": mode, "id": e["id"], "theta": s, "source": "s2",
               "planner_seed": pseed, "imag_steps": imag,
               "wall_s": round(time.time() - ep_t0, 1)}
        if abort:
            rec.update({"status": "incomplete",
                        "reason": "budget-exhausted-mid-episode",
                        "n_real": n_real})
        else:
            env = runner.env
            if info["success"]:
                cause = "arrival"
            elif float(np.linalg.norm(env.haz_pos - env.agent_pos)) < 5.0:
                cause = "contact-death"
            elif env.t >= env.MAX_STEPS:
                cause = "timeout"
            else:
                cause = "other"
            rec.update({"status": "complete", "cause": cause, "T": n_real})
            if not prim:
                rec.update({"frac_B": round(nb / n_real, 4),
                            "switches": sw})
            if cause == "arrival":
                key = f"{mode}_{e['id']}" + (f"_{s}" if s else "")
                traces[key] = {"id": e["id"], "mode": mode, "theta": s,
                               "planner_seed": pseed, "T": n_real,
                               "steps": steps}
                rec["trace"] = key
        out["records"].append(rec)
        save()
        print(f"[{pi + 1}/108] {mode} {e['id']}" +
              (f" th={s}" if s else "") + f": {rec['status']}" +
              (f" {rec.get('cause')} T={rec.get('T')}"
               if rec["status"] == "complete" else f" ({rec['reason']})") +
              f" imag={imag} wall={rec.get('wall_s', '?')}s", flush=True)

    out["total_wall_s"] = round(time.time() - t0, 1)
    save()
    n_inc = sum(1 for r in out["records"] if r["status"] == "incomplete")
    print(f"saved s2_30.json + s2_traces.json ({len(traces)} traces) "
          f"total_wall={out['total_wall_s']}s incomplete={n_inc}",
          flush=True)


if __name__ == "__main__":
    main()
