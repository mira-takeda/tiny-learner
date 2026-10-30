"""S8 run: teach(54) -> validate -> verify A18 -> train A36 -> eval ACQ9.

Single 600 s budget from teach start (cumulative across resumes),
checked inside the planner per real step. Incremental saves; resume
skips complete teacher execs (same frozen queries, no new ones).
Gates: teach/validate incomplete -> NO training; shadow mismatch
-> halt; ACQ9 opens only after training. Offline training uses
per-step mean (S3/S5 weighting kept).
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
from models import GATE_SHAPES, SKILL_SHAPES, mlp_forward, split_modular
from policies import layer_shapes, softmax, unpack, xavier_init
from tasks import TaskRunner

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
SEEDS = (301, 302, 303)
EPOCHS, LR, INIT_BASE = 5000, 0.001, 9200
CAP = 600.0


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def slot(p, apos):
    if p is None:
        return np.zeros(3)
    d = p - apos
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return np.zeros(3)
    return np.array([d[0] / n, d[1] / n, n / Arena.DMAX])


def replica_obs(apos, avel, goal, haz):
    o = np.zeros(14)
    o[0:2] = avel / Arena.VMAX
    o[2:5] = slot(goal, apos)
    o[5:8] = slot(haz, apos)
    return o


class Budget:
    def __init__(self, state):
        self.t0 = time.time()
        self.state = state

    def spent(self):
        return self.state.get("cumulative_s", 0.0) + time.time() - self.t0

    def ok(self):
        return self.spent() < CAP - 5

    def close(self):
        self.state["cumulative_s"] = round(self.spent(), 1)


def cause_of(runner, info):
    env = runner.env
    if info["success"]:
        return "arrival"
    if float(np.linalg.norm(env.haz_pos - env.agent_pos)) < 5.0:
        return "contact-death"
    if env.t >= env.MAX_STEPS:
        return "timeout"
    return "other"


def main():
    fr = json.load(open(f"{ROOT}/v2conf_diag/s8_freeze.json"))
    assert sha(f"{ROOT}/v2conf_diag/s8_acq8.json") == fr["data"]["acq8_sha256"]
    assert sha(f"{ROOT}/v2conf_diag/s8_acq9.json") == fr["data"]["acq9_sha256"]
    acq8 = {e["id"]: e for e in json.load(
        open(f"{ROOT}/v2conf_diag/s8_acq8.json"))}
    assert len(fr["teacher"]["order"]) == 54
    try:
        state = json.load(open(f"{ROOT}/v2conf_diag/s8_state.json"))
    except FileNotFoundError:
        state = {"cumulative_s": 0.0, "teach": {}, "phase": "teach"}
    B = Budget(state)
    save_state = lambda: json.dump(
        state, open(f"{ROOT}/v2conf_diag/s8_state.json", "w"))

    skills = {}
    for s in SEEDS:
        th = np.load(f"{ROOT}/v2conf_runs/VC_{s}_theta.npz")["theta"]
        p3 = np.load(f"{ROOT}/pilot3_runs/P3_{s}_theta.npz")["theta"]
        sk, _ = split_modular(th)
        assert all(np.array_equal(a, b)
                   for a, b in zip(sk, split_modular(p3)[0]))
        skills[s] = sk

    # ---- TEACH (resume: skip complete execs) ----
    if state["phase"] == "teach":
        for q in fr["teacher"]["order"]:
            key = str(q["exec"])
            if key in state["teach"] and \
                    state["teach"][key].get("status") == "complete":
                continue
            if not B.ok():
                state["teach"][key] = {"status": "incomplete",
                                       "reason": "budget"}
                state["phase"] = "incomplete"
                B.close()
                save_state()
                print("TEACH INCOMPLETE: budget", flush=True)
                return
            e, s = acq8[q["id"]], q["theta"]
            seq_rng = np.random.default_rng(int(q["planner_seed"]))
            runner = TaskRunner()
            runner.reset(e["family"], e["cfg"], int(e["seed"]))
            n, imag, nb, sw, prev = 0, 0, 0, 0, None
            done, info, steps, abort = False, {}, [], False
            while not done:
                if not B.ok():
                    abort = True
                    break
                st = R.snap(runner)
                seqs = seq_rng.integers(0, 2, size=(R.K, R.NBLOCK))
                (a, blk), im = R.shoot(st, runner.prev_goal, seqs,
                                       skills[s], False)
                imag += im
                nb += (blk == 1)
                sw += (prev is not None and blk != prev)
                prev = blk
                steps.append({"t": n,
                              "apos": [round(float(x), 3) for x in st["apos"]],
                              "avel": [round(float(x), 3) for x in st["avel"]],
                              "haz": [round(float(x), 3) for x in st["haz"]],
                              "a": int(a), "skill": int(blk)})
                _, _, done, info = runner.step(a)
                n += 1
            if abort:
                state["teach"][key] = {"status": "incomplete",
                                       "reason": "budget-mid-run"}
                state["phase"] = "incomplete"
                B.close()
                save_state()
                print("TEACH INCOMPLETE: budget mid-run", flush=True)
                return
            state["teach"][key] = {"status": "complete", "id": q["id"],
                "theta": s, "planner_seed": q["planner_seed"],
                "cause": cause_of(runner, info), "T": n,
                "imag_steps": imag, "frac_B": round(nb / n, 4),
                "switches": sw, "steps": steps}
            save_state()
            print(f"teach {q['exec']}/53 {q['id']} th={s}: "
                  f"{state['teach'][key]['cause']} T={n}", flush=True)
        state["phase"] = "validate"
        save_state()

    # ---- VALIDATE new demos (replay; no training on subsets) ----
    if state["phase"] == "validate":
        new_ds = {s: [] for s in SEEDS}
        for q in fr["teacher"]["order"]:
            rec = state["teach"][str(q["exec"])]
            assert rec.get("status") == "complete", "teach incomplete"
            e = acq8[q["id"]]
            goal = np.array(e["cfg"]["goal"]["pos"])
            r = TaskRunner()
            obs = r.reset(e["family"], e["cfg"], int(e["seed"]))
            for st in rec["steps"]:
                live = (r.env.agent_pos, r.env.agent_vel, r.env.haz_pos)
                rc = (np.array(st["apos"]), np.array(st["avel"]),
                      np.array(st["haz"]))
                assert all(np.allclose(a, b, atol=2e-3) for a, b in
                           zip(rc, live)), (q["exec"], st["t"])
                assert np.array_equal(replica_obs(live[0], live[1], goal,
                                                  live[2]), obs)
                assert st["skill"] in (0, 1)
                new_ds[q["theta"]].append((q["id"], obs.copy(), st["skill"]))
                obs, _, done, info = r.step(int(st["a"]))
            assert done and cause_of(r, info) == rec["cause"], q["exec"]
        n_new = sum(len(v) for v in new_ds.values())
        print(f"validate ok: 54 traces replayed, {n_new} new steps",
              flush=True)
        json.dump({"thetas": {str(s): {
            "ids": [i for (i, _, _) in new_ds[s]],
            "labels": [int(y) for (_, _, y) in new_ds[s]],
            "obs": [[float(x) for x in o] for (_, o, _) in new_ds[s]]}
            for s in SEEDS}},
            open(f"{ROOT}/v2conf_diag/s8_new18.json", "w"))
        state["phase"] = "verify"
        save_state()

    # ---- VERIFY A18 reuse (data/partition/config/weights + shadow) ----
    if state["phase"] == "verify":
        ds = json.load(open(f"{ROOT}/v2conf_diag/s3_data.json"))
        s3r = json.load(open(f"{ROOT}/v2conf_diag/s3_results.json"))
        s5 = json.load(open(f"{ROOT}/v2conf_diag/s5_results.json"))
        assert EPOCHS == 5000 and LR == 0.001 and INIT_BASE == 9200
        assert GATE_SHAPES == layer_shapes(14, [32], 4)
        shapes = GATE_SHAPES

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

        tr_ids = set(s3r["partition"]["train"])
        assert len(tr_ids) == 18
        for s in SEEDS:
            d = ds["thetas"][str(s)]
            m = np.array([i in tr_ids for i in d["ids"]])
            X = np.array(d["obs"])[m]
            y = np.array(d["labels"])[m]
            ref = np.array(s5["runs"][f"32_{s}"]["ckpts"]["5000"])
            assert ref.shape == (612,) and np.all(np.isfinite(ref))
            th = adam(init(s), X, y)
            dd = float(np.abs(th - ref).max())
            print(f"shadow th={s}: maxabs vs S5 = {dd:.1e}", flush=True)
            assert dd < 1e-9, f"config drift th={s}"
        print("A18 verified: data/partition/config/weights/shadow",
              flush=True)
        state["phase"] = "train"
        state["_fns"] = True
        save_state()

    # Rebuild closures after possible resume.
    shapes = GATE_SHAPES

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

    # ---- TRAIN A36 ----
    if state["phase"] == "train":
        ds = json.load(open(f"{ROOT}/v2conf_diag/s3_data.json"))
        new = json.load(open(f"{ROOT}/v2conf_diag/s8_new18.json"))
        s3r = json.load(open(f"{ROOT}/v2conf_diag/s3_results.json"))
        tr_ids = set(s3r["partition"]["train"])
        thetas = {}
        for s in SEEDS:
            if not B.ok():
                state["phase"] = "incomplete"
                B.close()
                save_state()
                print("TRAIN INCOMPLETE: budget", flush=True)
                return
            d = ds["thetas"][str(s)]
            m = np.array([i in tr_ids for i in d["ids"]])
            Xo = np.array(d["obs"])[m]
            yo = np.array(d["labels"])[m]
            Xn = np.array(new["thetas"][str(s)]["obs"])
            yn = np.array(new["thetas"][str(s)]["labels"])
            X = np.concatenate([Xo, Xn])
            y = np.concatenate([yo, yn])
            assert len(X) == len(Xo) + len(Xn) and len(Xn) > 0
            print(f"train36 th={s}: orig steps={len(Xo)} "
                  f"B={yo.mean():.3f} | new steps={len(Xn)} "
                  f"B={yn.mean():.3f} | total={len(X)}", flush=True)
            thetas[str(s)] = [float(x) for x in adam(init(s), X, y)]
        json.dump(thetas, open(f"{ROOT}/v2conf_diag/s8_models36.json", "w"))
        state["phase"] = "eval"
        save_state()
        print("train36 done", flush=True)

    # ---- EVAL both arms on ACQ9 (opens only after training) ----
    if state["phase"] == "eval":
        acq9 = json.load(open(f"{ROOT}/v2conf_diag/s8_acq9.json"))
        assert len(acq9) == 200
        s5 = json.load(open(f"{ROOT}/v2conf_diag/s5_results.json"))
        m36 = json.load(open(f"{ROOT}/v2conf_diag/s8_models36.json"))
        out = {"records": []}
        try:
            out = json.load(open(f"{ROOT}/v2conf_diag/s8_eval.json"))
        except FileNotFoundError:
            pass
        done = {(r["arm"], r["theta"], r["id"]) for r in out["records"]
                if r.get("status") == "complete"}
        save = lambda: json.dump(out, open(
            f"{ROOT}/v2conf_diag/s8_eval.json", "w"))

        def student_fn(gth, sk_all):
            def f(obs):
                k = int(np.argmax(mlp_forward(gth, obs, GATE_SHAPES)))
                return S4.greedy_skill(sk_all, obs, k), k
            return f

        for arm, src in (("A18", None), ("A36", None)):
            for s in SEEDS:
                gth = np.array(s5["runs"][f"32_{s}"]["ckpts"]["5000"]) \
                    if arm == "A18" else np.array(m36[str(s)])
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
        B.close()
        save_state()
        n_inc = sum(1 for r in out["records"] if r["status"] != "complete")
        print(f"eval done wall={out['wall_s']}s incomplete={n_inc}",
              flush=True)


if __name__ == "__main__":
    main()
