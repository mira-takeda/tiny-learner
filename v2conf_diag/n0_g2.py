"""N0: do frozen A+C skills contain a G2 (fetch) solution under a
privileged A/C planner? Diagnostic (existence), NOT a policy.

Reuses S2 search architecture (H=25, K=40, 5x5 blocks, replan per
real step, return scoring, first-max tie-break; asserted identical
to ref_plan). Dynamics mirror is NEW (m_g2: object push/friction
+ G2 reward state machine) because the S2 mirror models G1 only.
G2 has no hazard, so the mirror is EXACT (no noise modification);
validated by full-episode bit-exact twin proof (V1) before use.

Battery: data/train_meta_v1.json G2 (train-manifest, permitted;
NOT I1/H). 30 = first 10/region (G2a/G2b/G2c) in file order.
Arms (same scenarios/seeds): A-only, C-only (greedy, x3 thetas),
P-AC planner (blocks {A,C}, x3 thetas). Skills VC==P3 asserted.
No gate training, no arch change, no heldout. Budget 900 s,
incremental, INCOMPLETE not FAIL.
"""
import hashlib
import json
import sys
import time

sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/src")
sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/v2views")
sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/v2conf_diag")

import numpy as np

import ref_plan as R
import views_train as V
from env import Arena
from models import SKILL_SHAPES, mlp_forward, split_modular
from policies import softmax
from tasks import (DWELL_N, STEP_PENALTY, TOUCH_R, ZONE_R, TaskRunner)

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
assert (R.H, R.K, R.NBLOCK, R.BLEN) == (25, 40, 5, 5)
H, K, NBLOCK, BLEN = R.H, R.K, R.NBLOCK, R.BLEN
SEEDS = (301, 302, 303)
SEQ_BASE = 8200
# Cumulative budget: base read from n0_budget.json (prior processes) +
# this process. Cap 900 -> 1200 (24 fixed runs) -> 1350 (last 7 fixed
# runs; G2c tail episodes run long). Same scenarios/code/seeds.
T_BUDGET, T_MARGIN = 1350.0, 5.0


def budget_base():
    try:
        return float(json.load(
            open(f"{ROOT}/v2conf_diag/n0_budget.json"))["spent_s"])
    except FileNotFoundError:
        return 0.0
MANIFEST_G2_SHA = json.load(
    open(f"{ROOT}/data/MANIFEST.json"))["train_meta_v1.json"]
ACTIONS = Arena().ACTIONS.copy()
VMAX, DAMP = Arena.VMAX, Arena.DAMP
MAX_STEPS = Arena.MAX_STEPS
SKILL_OF_BLK = (0, 2)  # planner block 0->A, 1->C. B/D forbidden.


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def bounce(p, v):
    for i in (0, 1):
        if p[i] < 0.0:
            p[i] = 0.0
            v[i] = -0.5 * v[i]
        elif p[i] > Arena.SIZE:
            p[i] = Arena.SIZE
            v[i] = -0.5 * v[i]
    return p, v


def slot(p, apos):
    if p is None:
        return np.zeros(3)
    d = p - apos
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return np.zeros(3)
    return np.array([d[0] / n, d[1] / n, n / Arena.DMAX])


def m_obs(st):
    o = np.zeros(14)
    o[0:2] = st["avel"] / VMAX
    o[8:11] = slot(st["obj"], st["apos"])
    o[11:14] = slot(st["zone"], st["apos"])
    return o


def m_step(st, a):
    av = (st["avel"] + ACTIONS[int(a)]) * DAMP
    n = float(np.linalg.norm(av))
    if n > VMAX:
        av *= VMAX / n
    ap = st["apos"] + av
    ap, av = bounce(ap.copy(), av.copy())
    ov = st["ovel"] + np.zeros(2)
    op = st["obj"] + np.zeros(2)
    if float(np.linalg.norm(op - ap)) < 6.0:
        ov = ov + av * 0.35
    ov = ov * 0.92
    op = op + ov
    op, ov = bounce(op, ov)
    return {"apos": ap, "avel": av, "obj": op, "ovel": ov,
            "zone": st["zone"], "t": st["t"] + 1}


def m_g2(st, tr, a):
    """One imagined G2 step. tr=(touched,prev_ao,prev_oz,dwell)."""
    ns = m_step(st, a)
    obs = m_obs(ns)
    touched, prev_ao, prev_oz, dwell = tr
    r = STEP_PENALTY
    done, ok = False, False
    if float(np.linalg.norm(ns["apos"] - ns["obj"])) < TOUCH_R:
        touched = True
    if not touched:
        ao = float(np.linalg.norm(ns["apos"] - ns["obj"]))
        r += 0.1 * (prev_ao - ao)
        prev_ao = ao
    else:
        oz = float(np.linalg.norm(ns["obj"] - ns["zone"]))
        r += 0.1 * (prev_oz - oz)
        dwell = dwell + 1 if oz < ZONE_R else 0
        if dwell >= DWELL_N:
            r += 10.0
            done, ok = True, True
    prev_oz = float(np.linalg.norm(ns["obj"] - ns["zone"]))
    if ns["t"] >= MAX_STEPS and not done:
        done = True
    return ns, (touched, prev_ao, prev_oz, dwell), obs, r, done, ok


def snap(runner):
    e = runner.env
    return ({"apos": e.agent_pos.copy(), "avel": e.agent_vel.copy(),
             "obj": e.obj_pos.copy(), "ovel": e.obj_vel.copy(),
             "zone": e.zone.copy(), "t": e.t},
            (runner.touched, runner.prev_ao, runner.prev_oz, runner.dwell))


def greedy_skill(sk_all, obs, k):
    return int(np.argmax(softmax(mlp_forward(
        sk_all[k], V.view_obs(obs, k), SKILL_SHAPES))))


def shoot(st0, tr0, seqs, sk_all):
    best, best_first, imag = -1e18, None, 0
    for s in seqs:
        st, tr, tot, done = dict(st0), tr0, 0.0, False
        first = None
        for t in range(H):
            k = SKILL_OF_BLK[int(s[t // BLEN])]
            a = greedy_skill(sk_all, m_obs(st), k)
            if t == 0:
                first = (a, k)
            st, tr, _, r, done, _ = m_g2(st, tr, a)
            tot += r
            imag += 1
            if done:
                break
        if tot > best:
            best, best_first = tot, first
    return best_first, imag


def main():
    t0 = time.time()
    base = budget_base()
    spent = lambda: base + time.time() - t0
    assert sha(f"{ROOT}/data/train_meta_v1.json") == MANIFEST_G2_SHA
    g2 = [e for e in json.load(open(f"{ROOT}/data/train_meta_v1.json"))
          if e["family"] == "G2"]
    eps30 = []
    for r in ("G2a", "G2b", "G2c"):
        eps30 += [e for e in g2 if e["region"] == r][:10]
    assert len(eps30) == 30
    assert all("wall" not in e["cfg"] for e in eps30)
    print(f"N0 battery: {[e['id'] for e in eps30][:3]}... n=30", flush=True)

    skills = {}
    for s in SEEDS:
        th = np.load(f"{ROOT}/v2conf_runs/VC_{s}_theta.npz")["theta"]
        p3 = np.load(f"{ROOT}/pilot3_runs/P3_{s}_theta.npz")["theta"]
        sk, _ = split_modular(th)
        assert all(np.array_equal(a, b)
                   for a, b in zip(sk, split_modular(p3)[0])), s
        skills[s] = sk
    print("V0 ok: battery hash, VC==P3 skills bit-identical", flush=True)

    # V1: full-episode bit-exact twin (G2 deterministic: no hazard noise).
    n_match = 0
    for ei, e in enumerate(eps30):
        r = TaskRunner()
        o = r.reset(e["family"], e["cfg"], int(e["seed"]))
        st, tr = snap(r)
        ar = np.random.default_rng(5000 + ei)
        while True:
            a = int(ar.integers(9))
            o2, re, d2, i2 = r.step(a)
            st, tr, mo, mr, md, mi = m_g2(st, tr, a)
            assert np.array_equal(mo, o2), (ei, "obs")
            assert (md, mr, mi) == (d2, re, i2["success"]), (ei, "task")
            n_match += 1
            if d2:
                break
    print(f"V1 ok: mirror == env bit-exact over {n_match} steps x30 eps",
          flush=True)

    # V3: greedy identity for k in {A, C} on real G2 obs.
    pool = []
    rr = TaskRunner()
    for e in eps30[:6]:
        o = rr.reset(e["family"], e["cfg"], int(e["seed"]))
        for t in range(20):
            pool.append(o.copy())
            o = rr.env.step(int(np.random.default_rng(t).integers(9)))
    n_probe = 0
    for s in SEEDS:
        for o in pool:
            for k in (0, 2):
                ref = int(np.argmax(softmax(mlp_forward(
                    skills[s][k], V.view_obs(o.copy(), k), SKILL_SHAPES))))
                assert greedy_skill(skills[s], o, k) == ref, (s, k)
                n_probe += 1
    print(f"V3 ok: greedy == frozen skill/view/argmax, {n_probe} probes "
          f"(k=A,C)", flush=True)

    plan = ([("A", e, s) for e in eps30 for s in SEEDS] +
            [("C", e, s) for e in eps30 for s in SEEDS] +
            [("AC", e, s) for e in eps30 for s in SEEDS])
    kids = [int(x.generate_state(1, dtype=np.uint32)[0])
            for x in np.random.SeedSequence(SEQ_BASE).spawn(len(plan))]
    out = {"H": H, "K": K, "blocks": [NBLOCK, BLEN], "seq_base": SEQ_BASE,
           "budget_s": T_BUDGET, "records": []}
    try:
        prev = json.load(open(f"{ROOT}/v2conf_diag/n0_g2.json"))
        # Keep completes only: stale incompletes are re-run and re-added
        # (avoids duplicate keys across resume processes).
        out["records"] = [r for r in prev["records"]
                          if r["status"] == "complete"]
        done_keys = {(r["mode"], r["id"], r["theta"])
                     for r in out["records"]}
        print(f"resumed: {len(done_keys)} complete kept", flush=True)
    except FileNotFoundError:
        done_keys = set()
    sync = lambda: json.dump(out, open(
        f"{ROOT}/v2conf_diag/n0_g2.json", "w"), indent=1)

    for xi, (mode, e, s) in enumerate(plan):
        if (mode, e["id"], s) in done_keys:
            continue
        if spent() > T_BUDGET - T_MARGIN:
            out["records"].append(
                {"mode": mode, "id": e["id"], "theta": s,
                 "status": "incomplete", "reason": "budget-before-start"})
            sync()
            continue
        rec = {"mode": mode, "id": e["id"], "theta": s,
               "planner_seed": kids[xi]}
        runner = TaskRunner()
        obs = runner.reset(e["family"], e["cfg"], int(e["seed"]))
        n_real, imag, tot_ret = 0, 0, 0.0
        uses, sw_steps, prev_k, abort = [0, 0], [], None, False
        max_dwell, touched_end = 0, False
        seq_rng = np.random.default_rng(kids[xi])
        done, info = False, {}
        ep_t0 = time.time()
        while not done:
            if spent() > T_BUDGET - T_MARGIN:
                abort = True
                break
            if mode == "AC":
                st, tr = snap(runner)
                seqs = seq_rng.integers(0, 2, size=(K, NBLOCK))
                (a, k), im = shoot(st, tr, seqs, skills[s])
                imag += im
                uses[k // 2] += 1
                if prev_k is not None and k != prev_k:
                    sw_steps.append(n_real)
                prev_k = k
            else:
                k = 0 if mode == "A" else 2
                a = greedy_skill(skills[s], obs, k)
            obs, rew, done, info = runner.step(a)
            tot_ret += rew
            n_real += 1
            touched_end = runner.touched
            max_dwell = max(max_dwell, runner.dwell)
        rec.update({"imag_steps": imag, "return": round(tot_ret, 4),
                    "wall_s": round(time.time() - ep_t0, 1)})
        if abort:
            rec.update({"status": "incomplete",
                        "reason": "budget-mid-episode", "n_real": n_real})
        else:
            fin_ao = float(np.linalg.norm(
                runner.env.agent_pos - runner.env.obj_pos))
            fin_oz = float(np.linalg.norm(
                runner.env.obj_pos - runner.env.zone))
            rec.update({"status": "complete",
                        "success": bool(info["success"]), "T": n_real,
                        "touched": touched_end, "max_dwell": max_dwell,
                        "final_ao": round(fin_ao, 2),
                        "final_oz": round(fin_oz, 2)})
            if mode == "AC":
                rec.update({"use_A": uses[0], "use_C": uses[1],
                            "switches": len(sw_steps),
                            "switch_steps": sw_steps})
        out["records"].append(rec)
        sync()
        if rec["status"] == "complete":
            extra = f" ok={rec['success']} T={rec['T']}"
            if mode == "AC":
                extra += f" A/C={rec['use_A']}/{rec['use_C']} sw={rec['switches']}"
            print(f"{mode} {e['id']} th={s}: complete{extra} "
                  f"wall={rec['wall_s']}s", flush=True)
        else:
            print(f"{mode} {e['id']} th={s}: {rec['status']} "
                  f"({rec['reason']})", flush=True)
    n_inc = sum(1 for r in out["records"] if r["status"] != "complete")
    json.dump({"spent_s": round(spent(), 1)},
              open(f"{ROOT}/v2conf_diag/n0_budget.json", "w"))
    print(f"N0 {'done' if n_inc == 0 else 'INCOMPLETE'} "
          f"cumulative_wall={spent():.1f}s incomplete={n_inc}", flush=True)


if __name__ == "__main__":
    main()
