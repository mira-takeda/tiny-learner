"""S10 run: collect(A36) -> control(teacher) -> select -> query STU states
-> build datasets -> train T-STU/T-TEA -> eval 3 arms on ACQ12.

Single shared pab_decide() for control rollouts and pointwise queries
(frozen S2 P-AB). Cumulative 900 s budget from prep (seeded), checked
inside planner loops. Incremental saves + resume. Gates: incomplete
collection/queries/validation -> NO training.
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
import s4_closed as S4
from env import Arena
from models import GATE_SHAPES, mlp_forward, split_modular
from policies import layer_shapes, softmax, unpack, xavier_init
from tasks import TaskRunner

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
SEEDS = (301, 302, 303)
EPOCHS, LR, INIT_BASE = 5000, 0.001, 9200
CAP = 900.0
assert (R.H, R.K, R.NBLOCK, R.BLEN) == (25, 40, 5, 5)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Budget:
    def __init__(self, base):
        self.t0 = time.time()
        self.base = base

    def spent(self):
        return self.base + time.time() - self.t0

    def ok(self):
        return self.spent() < CAP - 5


def cause_of(runner, info):
    env = runner.env
    if info["success"]:
        return "arrival"
    if float(np.linalg.norm(env.haz_pos - env.agent_pos)) < 5.0:
        return "contact-death"
    if env.t >= env.MAX_STEPS:
        return "timeout"
    return "other"


def pab_decide(st, prev_goal, seq_rng, sk_all):
    """Frozen S2 P-AB decision. Shared by control rollouts and queries."""
    seqs = seq_rng.integers(0, 2, size=(R.K, R.NBLOCK))
    return R.shoot(st, prev_goal, seqs, sk_all, False)


def main():
    fr = json.load(open(f"{ROOT}/v2conf_diag/s10_freeze.json"))
    assert sha(f"{ROOT}/v2conf_diag/s10_acq11.json") == fr["data"]["acq11_sha256"]
    assert sha(f"{ROOT}/v2conf_diag/s10_acq12.json") == fr["data"]["acq12_sha256"]
    assert sha(f"{ROOT}/v2conf_diag/s8_models36.json") == fr["models36_sha256"]
    acq11 = {e["id"]: e for e in json.load(
        open(f"{ROOT}/v2conf_diag/s10_acq11.json"))}
    assert len(acq11) == 9
    try:
        state = json.load(open(f"{ROOT}/v2conf_diag/s10_state.json"))
    except FileNotFoundError:
        seed = json.load(open(f"{ROOT}/v2conf_diag/s10_budget_seed.json"))
        state = {"cumulative_s": float(seed["mkdata_wall_s"]),
                 "phase": "collect", "collect": {}, "control": {},
                 "queries": {}}
    B = Budget(state["cumulative_s"])
    save_state = lambda: json.dump(
        state, open(f"{ROOT}/v2conf_diag/s10_state.json", "w"))
    sync = lambda: (state.update(cumulative_s=round(B.spent(), 1)),
                    save_state())

    m36 = json.load(open(f"{ROOT}/v2conf_diag/s8_models36.json"))
    skills = {}
    for s in SEEDS:
        th = np.load(f"{ROOT}/v2conf_runs/VC_{s}_theta.npz")["theta"]
        p3 = np.load(f"{ROOT}/pilot3_runs/P3_{s}_theta.npz")["theta"]
        sk, _ = split_modular(th)
        assert all(np.array_equal(a, b)
                   for a, b in zip(sk, split_modular(p3)[0]))
        skills[s] = sk
        assert len(m36[str(s)]) == 612  # collector weight identity (sha above)

    def greedy_a36(gth, sk_all):
        def f(obs):
            k = int(np.argmax(mlp_forward(gth, obs, GATE_SHAPES)))
            return S4.greedy_skill(sk_all, obs, k), k
        return f

    # ---- COLLECT: frozen A36 rollouts + full pre-action snapshots ----
    if state["phase"] == "collect":
        ids = sorted(acq11)
        for s in SEEDS:
            gth = np.array(m36[str(s)])
            fn = greedy_a36(gth, skills[s])
            for eid in ids:
                key = f"{eid}_{s}"
                if key in state["collect"]:
                    continue
                if not B.ok():
                    sync()
                    print("COLLECT INCOMPLETE", flush=True)
                    return
                e = acq11[eid]
                r = TaskRunner()
                obs = r.reset(e["family"], e["cfg"], int(e["seed"]))
                n, done, info, steps = 0, {}, False, []
                uses, sw, prev = [0, 0, 0, 0], 0, None
                while not done:
                    steps.append({"t": n, "obs": [float(x) for x in obs],
                                  "apos": [float(x) for x in r.env.agent_pos],
                                  "avel": [float(x) for x in r.env.agent_vel],
                                  "haz": [float(x) for x in r.env.haz_pos],
                                  "hspeed": float(r.env.haz_speed),
                                  "env_t": int(r.env.t),
                                  "prev_goal": float(runner_prev(r))})
                    a, k = fn(obs)
                    steps[-1].update(a=int(a), skill_exec=int(k))
                    uses[k] += 1
                    sw += (prev is not None and k != prev)
                    prev = k
                    obs, _, done, info = r.step(a)
                    n += 1
                # Execution identity vs S4.run_ep on first scenario/theta.
                if eid == ids[0]:
                    ref = S4.run_ep(e, greedy_a36(gth, skills[s]))
                    assert (ref["cause"], ref["T"], ref["uses"],
                            ref["switches"]) == \
                        (cause_of(r, info), n, uses, sw), (eid, s)
                state["collect"][key] = {"id": eid, "theta": s,
                    "cause": cause_of(r, info), "T": n, "uses": uses,
                    "switches": sw, "steps": steps}
                sync()
                print(f"collect {eid} th={s}: "
                      f"{state['collect'][key]['cause']} T={n}", flush=True)
        state["phase"] = "control"
        sync()

    # ---- CONTROL: 27 frozen teacher runs (full traces count in cost) ----
    if state["phase"] == "control":
        for q in fr["control_runs"]["order"]:
            key = str(q["exec"])
            if key in state["control"]:
                continue
            if not B.ok():
                state["phase"] = "incomplete"
                sync()
                print("CONTROL INCOMPLETE", flush=True)
                return
            e, s = acq11[q["id"]], q["theta"]
            seq_rng = np.random.default_rng(int(q["planner_seed"]))
            runner = TaskRunner()
            obs = runner.reset(e["family"], e["cfg"], int(e["seed"]))
            n, imag, nb, sw, prev = 0, 0, 0, 0, None
            done, info, steps, abort = False, {}, [], False
            while not done:
                if not B.ok():
                    abort = True
                    break
                st = R.snap(runner)
                (a, blk), im = pab_decide(st, runner.prev_goal, seq_rng,
                                          skills[s])
                imag += im
                nb += (blk == 1)
                sw += (prev is not None and blk != prev)
                prev = blk
                steps.append({"t": n, "obs": [float(x) for x in obs],
                              "a": int(a), "skill": int(blk)})
                obs, _, done, info = runner.step(a)
                n += 1
            if abort:
                state["phase"] = "incomplete"
                sync()
                print("CONTROL INCOMPLETE mid-run", flush=True)
                return
            state["control"][key] = {"id": q["id"], "theta": s,
                "planner_seed": q["planner_seed"],
                "cause": cause_of(runner, info), "T": n, "imag_steps": imag,
                "frac_B": round(nb / n, 4), "switches": sw, "steps": steps}
            sync()
            print(f"control {q['exec']}/26 {q['id']} th={s}: "
                  f"{state['control'][key]['cause']} T={n}", flush=True)
        state["phase"] = "query"
        sync()

    # ---- SELECT indices (frozen formula) + QUERY selected STU states ----
    if state["phase"] == "query":
        pairs = [(e, s) for e in sorted(acq11) for s in SEEDS]
        for eid, s in pairs:
            key = f"{eid}_{s}"
            if key in state["queries"]:
                continue
            if not B.ok():
                state["phase"] = "incomplete"
                sync()
                print("QUERY INCOMPLETE", flush=True)
                return
            stu = state["collect"][f"{eid}_{s}"]["steps"]
            exec_idx = next(q["exec"] for q in fr["control_runs"]["order"]
                            if q["id"] == eid and q["theta"] == s)
            tea = state["control"][str(exec_idx)]["steps"]
            Ls, Lt = len(stu), len(tea)
            m = min(Ls, Lt)
            def idxs(L):
                if m == 1:
                    return [0]
                return [min(L - 1, (j * (L - 1)) // (m - 1))
                        for j in range(m)]
            si, ti = idxs(Ls), idxs(Lt)
            assert len(set(si)) == m and len(set(ti)) == m, (eid, s)
            assert si[0] == 0 and ti[0] == 0
            if m > 1:
                assert si[-1] == Ls - 1 and ti[-1] == Lt - 1
            labels, imag_tot, seeds_used = [], 0, []
            for j, t in enumerate(si):
                st = stu[t]
                snap = {"apos": np.array(st["apos"]),
                        "avel": np.array(st["avel"]),
                        "haz": np.array(st["haz"]),
                        "hspeed": st["hspeed"],
                        "goal": np.array(acq11[eid]["cfg"]["goal"]["pos"]),
                        "t": st["env_t"]}
                sd = int(hashlib.sha256(
                    f"S10q|{eid}|{s}|{t}".encode()).hexdigest()[:8], 16)
                seeds_used.append(sd)
                (a, blk), im = pab_decide(
                    snap, st["prev_goal"], np.random.default_rng(sd),
                    skills[s])
                imag_tot += im
                labels.append(int(blk))
                if not B.ok():
                    state["phase"] = "incomplete"
                    sync()
                    print("QUERY INCOMPLETE mid-pair", flush=True)
                    return
            state["queries"][key] = {"id": eid, "theta": s, "Ls": Ls,
                "Lt": Lt, "m": m, "stu_idx": si, "tea_idx": ti,
                "stu_labels": labels, "query_seeds": seeds_used,
                "imag_steps": imag_tot}
            sync()
            print(f"query {eid} th={s}: m={m} labels={len(labels)}",
                  flush=True)
        state["phase"] = "build"
        sync()

    # ---- BUILD datasets (validate control traces by replay) ----
    if state["phase"] == "build":
        ds = json.load(open(f"{ROOT}/v2conf_diag/s3_data.json"))
        new18 = json.load(open(f"{ROOT}/v2conf_diag/s8_new18.json"))
        s3r = json.load(open(f"{ROOT}/v2conf_diag/s3_results.json"))
        tr_ids = set(s3r["partition"]["train"])
        data = {}
        for s in SEEDS:
            d = ds["thetas"][str(s)]
            m = np.array([i in tr_ids for i in d["ids"]])
            Xo = np.array(d["obs"])[m]
            yo = np.array(d["labels"])[m]
            nd = new18["thetas"][str(s)]
            Xn = np.array(nd["obs"])
            yn = np.array(nd["labels"])
            Xb = np.concatenate([Xo, Xn])
            yb = np.concatenate([yo, yn])
            # Control new pairs: replay-validate, take selected steps.
            Xc, yc, Xt, yt = [], [], [], []
            for eid in sorted(acq11):
                exec_idx = next(q["exec"] for q in fr["control_runs"]
                                ["order"] if q["id"] == eid and q["theta"] == s)
                rec = state["control"][str(exec_idx)]
                e = acq11[eid]
                r = TaskRunner()
                obs = r.reset(e["family"], e["cfg"], int(e["seed"]))
                for st in rec["steps"]:
                    assert np.array_equal(
                        np.array(st["obs"]), obs), (eid, s, st["t"])
                    obs, _, done, info = r.step(int(st["a"]))
                assert done and cause_of(r, info) == rec["cause"]
                q = state["queries"][f"{eid}_{s}"]
                for t in q["tea_idx"]:
                    Xc.append(rec["steps"][t]["obs"])
                    yc.append(rec["steps"][t]["skill"])
                stu = state["collect"][f"{eid}_{s}"]["steps"]
                for t, lab in zip(q["stu_idx"], q["stu_labels"]):
                    Xt.append(stu[t]["obs"])
                    yt.append(lab)
            assert len(Xc) == len(yc) == len(Xt) == len(yt), s
            assert len(Xc) == sum(
                state["queries"][f"{eid}_{s}"]["m"] for eid in sorted(acq11))
            data[str(s)] = {"base_n": len(Xb),
                            "new_n": len(Xc),
                            "tea": {"obs": [[float(x) for x in o] for o in Xc],
                                    "labels": [int(y) for y in yc]},
                            "stu": {"obs": [[float(x) for x in o] for o in Xt],
                                    "labels": [int(y) for y in yt]}}
            print(f"build th={s}: base={len(Xb)} new={len(Xc)}", flush=True)
        json.dump(data, open(f"{ROOT}/v2conf_diag/s10_data.json", "w"))
        state["phase"] = "train"
        sync()

    shapes = GATE_SHAPES
    assert shapes == layer_shapes(14, [32], 4)

    def grads(th, X, y):
        n = len(X)
        W1, b1, W2, b2 = unpack(th, shapes)
        Hh = np.tanh(X @ W1 + b1)
        Z = Hh @ W2 + b2
        Z = Z - Z.max(axis=1, keepdims=True)
        E = np.exp(Z)
        P = E / E.sum(axis=1, keepdims=True)
        dZ = P.copy()
        dZ[np.arange(n), y] -= 1.0
        dZ /= n
        dW2 = Hh.T @ dZ
        db2 = dZ.sum(axis=0)
        dZ1 = (dZ @ W2.T) * (1.0 - Hh ** 2)
        return np.concatenate([(X.T @ dZ1).ravel(), dZ1.sum(axis=0),
                               dW2.ravel(), db2])

    def adam(th, X, y):
        m, v = np.zeros_like(th), np.zeros_like(th)
        for t in range(1, EPOCHS + 1):
            if t % 500 == 0 and not B.ok():
                return None
            gg = grads(th, X, y)
            m = 0.9 * m + 0.1 * gg
            v = 0.999 * v + 0.001 * gg ** 2
            th = th - LR * (m / (1 - 0.9 ** t)) / \
                (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        return th

    def init(s):
        rng = np.random.default_rng(INIT_BASE + s)
        W1 = unpack(xavier_init(rng, shapes), shapes)[0]
        return np.concatenate([W1.ravel(), np.zeros(32 + 32 * 4 + 4)])

    # ---- TRAIN both arms ----
    if state["phase"] == "train":
        dd = json.load(open(f"{ROOT}/v2conf_diag/s10_data.json"))
        ds = json.load(open(f"{ROOT}/v2conf_diag/s3_data.json"))
        new18 = json.load(open(f"{ROOT}/v2conf_diag/s8_new18.json"))
        s3r = json.load(open(f"{ROOT}/v2conf_diag/s3_results.json"))
        tr_ids = set(s3r["partition"]["train"])
        try:
            models = json.load(open(f"{ROOT}/v2conf_diag/s10_models.json"))
        except FileNotFoundError:
            models = {}
        for arm in ("T-STU", "T-TEA"):
            for s in SEEDS:
                if f"{arm}_{s}" in models:
                    continue
                d = ds["thetas"][str(s)]
                m = np.array([i in tr_ids for i in d["ids"]])
                Xo = np.array(d["obs"])[m]
                yo = np.array(d["labels"])[m]
                nd = new18["thetas"][str(s)]
                X = np.concatenate([Xo, np.array(nd["obs"]),
                                    np.array(dd[str(s)][
                                        "stu" if arm == "T-STU" else "tea"]
                                        ["obs"])])
                y = np.concatenate([yo, np.array(nd["labels"]),
                                    np.array(dd[str(s)][
                                        "stu" if arm == "T-STU" else "tea"]
                                        ["labels"])])
                th = adam(init(s), X, y)
                if th is None:
                    state["phase"] = "incomplete"
                    sync()
                    print("TRAIN INCOMPLETE", flush=True)
                    return
                models[f"{arm}_{s}"] = [float(x) for x in th]
                json.dump(models, open(
                    f"{ROOT}/v2conf_diag/s10_models.json", "w"))
                print(f"train {arm} th={s} done n={len(X)}", flush=True)
        state["phase"] = "eval"
        sync()

    # ---- EVAL 3 arms on ACQ12 (opens only after training) ----
    if state["phase"] == "eval":
        acq9 = json.load(open(f"{ROOT}/v2conf_diag/s10_acq12.json"))
        assert len(acq9) == 200
        models = json.load(open(f"{ROOT}/v2conf_diag/s10_models.json"))
        assert len(models) == 6
        try:
            out = json.load(open(f"{ROOT}/v2conf_diag/s10_eval.json"))
        except FileNotFoundError:
            out = {"records": []}
        done = {(r["arm"], r["theta"], r["id"]) for r in out["records"]
                if r.get("status") == "complete"}
        save = lambda: json.dump(out, open(
            f"{ROOT}/v2conf_diag/s10_eval.json", "w"))

        def student_fn(gth, sk_all):
            def f(obs):
                k = int(np.argmax(mlp_forward(gth, obs, GATE_SHAPES)))
                return S4.greedy_skill(sk_all, obs, k), k
            return f

        for arm in ("T-STU", "T-TEA", "A36"):
            for s in SEEDS:
                if arm == "A36":
                    gth = np.array(m36[str(s)])
                else:
                    gth = np.array(models[f"{arm}_{s}"])
                fn = student_fn(gth, skills[s])
                for e in acq9:
                    if (arm, s, e["id"]) in done:
                        continue
                    if not B.ok():
                        out["records"].append(
                            {"arm": arm, "theta": s, "id": e["id"],
                             "status": "incomplete"})
                        save()
                        continue
                    rec = S4.run_ep(e, fn)
                    rec.update({"arm": arm, "theta": s, "id": e["id"],
                                "status": "complete"})
                    rec.pop("hobs", None)
                    out["records"].append(rec)
                    save()
        out["wall_s"] = round(B.spent(), 1)
        save()
        state["phase"] = "done"
        sync()
        n_inc = sum(1 for r in out["records"] if r["status"] != "complete")
        print(f"eval done wall={out['wall_s']}s incomplete={n_inc}",
              flush=True)


def runner_prev(r):
    return r.prev_goal


if __name__ == "__main__":
    main()
