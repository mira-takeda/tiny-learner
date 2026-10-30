"""S12: single-intervention diagnostic (no training, no teacher).

Phase 1: reproduce T-STU-S10 on ACQ14 x3 thetas; verify
cause/T/uses/switches vs s11_eval records. Any mismatch halts.
Phase 2: per episode, scan S10 trajectory for first step where
STU-S11 proposes a different EXECUTABLE primitive on the same
obs; re-run with S10 until then, S11 action once, S10 after.
Env noise per step preserved (RNG untouched by policies:
asserted; obs-prefix bit-match asserted). Nulls kept.
Cumulative 300 s, incremental, resume.
"""
import hashlib
import json
import sys
import time

sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/src")
sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/v2views")
sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/v2conf_diag")

import numpy as np

import s4_closed as S4
from models import GATE_SHAPES, mlp_forward, split_modular
from tasks import TaskRunner

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
SEEDS = (301, 302, 303)
CAP = 300.0


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rng_state(rng):
    return json.dumps(rng.bit_generator.state, sort_keys=True,
                      default=int)


def main():
    fr = json.load(open(f"{ROOT}/v2conf_diag/s12_freeze.json"))
    assert sha(f"{ROOT}/v2conf_diag/s11_acq14.json") == fr["data"]["acq14_sha256"]
    assert sha(f"{ROOT}/v2conf_diag/s10_models.json") == fr["models"]["s10_TSTU_sha256"]
    assert sha(f"{ROOT}/v2conf_diag/s11_models.json") == fr["models"]["s11_STU_sha256"]
    acq = json.load(open(f"{ROOT}/v2conf_diag/s11_acq14.json"))
    assert len(acq) == 200
    old = {(r["theta"], r["id"]): r for r in json.load(
        open(f"{ROOT}/v2conf_diag/s11_eval.json"))["records"]
        if r["arm"] == "TSTU10" and r.get("status") == "complete"}
    assert len(old) == 600, len(old)
    m10 = json.load(open(f"{ROOT}/v2conf_diag/s10_models.json"))
    m11 = json.load(open(f"{ROOT}/v2conf_diag/s11_models.json"))
    try:
        state = json.load(open(f"{ROOT}/v2conf_diag/s12_state.json"))
    except FileNotFoundError:
        state = {"cumulative_s": 0.0, "phase": "repro", "records": {}}
    t0 = time.time()
    base = float(state.get("cumulative_s", 0.0))
    spent = lambda: base + time.time() - t0
    ok = lambda: spent() < CAP - 5
    sync = lambda: (state.update(cumulative_s=round(spent(), 1)),
                    json.dump(state, open(
                        f"{ROOT}/v2conf_diag/s12_state.json", "w")))
    skills = {s: split_modular(np.load(
        f"{ROOT}/v2conf_runs/VC_{s}_theta.npz")["theta"])[0] for s in SEEDS}

    def decide(gth, sk_all, obs):
        k = int(np.argmax(mlp_forward(gth, obs, GATE_SHAPES)))
        return k, S4.greedy_skill(sk_all, obs, k)

    # Phase 1: reproduce + verify ALL before any intervention.
    if state["phase"] == "repro":
        for s in SEEDS:
            g10 = np.array(m10[f"T-STU_{s}"])
            for e in acq:
                key = f"{s}_{e['id']}"
                if key in state["records"] and \
                        state["records"][key].get("repro_ok"):
                    continue
                if not ok():
                    state["phase"] = "incomplete"
                    sync()
                    print("REPRO INCOMPLETE", flush=True)
                    return
                r = TaskRunner()
                obs = r.reset(e["family"], e["cfg"], int(e["seed"]))
                done, info, n = False, {}, 0
                uses, sw, prev = [0, 0, 0, 0], 0, None
                while not done:
                    rs0 = rng_state(r.env.rng)
                    k, a = decide(g10, skills[s], obs)
                    assert rng_state(r.env.rng) == rs0, "policy touched RNG"
                    uses[k] += 1
                    sw += (prev is not None and k != prev)
                    prev = k
                    obs, _, done, info = r.step(a)
                    n += 1
                got = {"cause": S4.cause_of(r, info), "T": n, "uses": uses,
                       "switches": sw}
                ref = old[(s, e["id"])]
                assert all(got[k] == ref[k] for k in
                           ("cause", "T", "uses", "switches")), \
                    f"REPRO MISMATCH th={s} {e['id']}: {got} vs " + \
                    str({k: ref[k] for k in got})
                state["records"][key] = {"theta": s, "id": e["id"],
                                         "repro_ok": True, **got}
                sync()
        state["phase"] = "intervene"
        sync()
        print("repro verified 600/600", flush=True)

    # Phase 2: scan + single intervention.
    if state["phase"] == "intervene":
        for s in SEEDS:
            g10 = np.array(m10[f"T-STU_{s}"])
            g11 = np.array(m11[f"STU_{s}"])
            for e in acq:
                key = f"{s}_{e['id']}"
                rec = state["records"].get(key, {})
                if rec.get("status") == "complete":
                    continue
                if not ok():
                    rec.update(status="incomplete")
                    state["records"][key] = rec
                    sync()
                    continue
                # Scan S10 trajectory + S11 proposals on same obs.
                r = TaskRunner()
                obs = r.reset(e["family"], e["cfg"], int(e["seed"]))
                done, info, n = False, {}, 0
                scan_obs, first, first_skill = [], None, None
                while not done:
                    k0, a0 = decide(g10, skills[s], obs)
                    k1, a1 = decide(g11, skills[s], obs)
                    if first is None and a1 != a0:
                        first = {"t": n, "k10": k0, "a10": a0,
                                 "k11": k1, "a11": a1}
                    if first_skill is None and k1 != k0:
                        first_skill = {"t": n, "k10": k0, "k11": k1}
                    scan_obs.append(obs.copy())
                    obs, _, done, info = r.step(a0)
                    n += 1
                if first is None:
                    rec.update(status="complete", intervention=None,
                               first_skill_diff=first_skill,
                               cause=rec["cause"], T=rec["T"])
                    state["records"][key] = rec
                    sync()
                    continue
                # Intervention rollout with obs-prefix verification.
                t = first["t"]
                r2 = TaskRunner()
                obs2 = r2.reset(e["family"], e["cfg"], int(e["seed"]))
                done2, info2, n2 = False, {}, 0
                uses2, sw2, prev2, trace = [0, 0, 0, 0], 0, None, []
                while not done2:
                    if n2 < t:
                        assert np.array_equal(obs2, scan_obs[n2]), \
                            (key, n2, "noise/state diverged pre-t")
                        k, a = decide(g10, skills[s], obs2)
                    elif n2 == t:
                        assert np.array_equal(obs2, scan_obs[n2]), (key, "t")
                        k, a = first["k11"], first["a11"]
                    else:
                        k, a = decide(g10, skills[s], obs2)
                    trace.append({"t": n2, "obs": [float(x) for x in obs2],
                                  "a": int(a), "k": int(k)})
                    uses2[k] += 1
                    sw2 += (prev2 is not None and k != prev2)
                    prev2 = k
                    obs2, _, done2, info2 = r2.step(a)
                    n2 += 1
                rec.update(status="complete", intervention=first,
                           first_skill_diff=first_skill,
                           int_cause=S4.cause_of(r2, info2), int_T=n2,
                           int_uses=uses2, int_switches=sw2, trace=trace)
                state["records"][key] = rec
                sync()
        state["phase"] = "done"
        sync()
        out = {"records": [state["records"][k] for k in sorted(
            state["records"])], "wall_s": round(spent(), 1)}
        json.dump(out, open(f"{ROOT}/v2conf_diag/s12_eval.json", "w"))
        n_inc = sum(1 for r in out["records"] if r.get("status") != "complete")
        print(f"done wall={out['wall_s']}s incomplete={n_inc}", flush=True)


if __name__ == "__main__":
    main()
