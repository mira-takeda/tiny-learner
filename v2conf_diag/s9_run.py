"""S9 run: augmented dataset -> train informed -> eval both on ACQ10.

Train-A36 only (1003/1283/1222 pairs/theta, verified vs originals).
Pre-action x/y/t from validated replay + cfg hspeed (provenance
recorded). Informed obs built on a copy; pristine obs asserted
intact; skills always get pristine obs + original views. Original
arm reuses s8_models36 (sha-verified). Single 300 s budget,
incremental saves, resume support.
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
from env import Arena
from models import GATE_SHAPES, mlp_forward
from policies import layer_shapes, unpack, xavier_init
from tasks import TaskRunner

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
SEEDS = (301, 302, 303)
EPOCHS, LR, INIT_BASE = 5000, 0.001, 9200
CAP = 300.0


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


def main():
    fr = json.load(open(f"{ROOT}/v2conf_diag/s9_freeze.json"))
    assert sha(f"{ROOT}/v2conf_diag/s9_acq10.json") == fr["data"]["acq10_sha256"]
    assert sha(f"{ROOT}/v2conf_diag/s8_models36.json") == fr["models36_sha256"]
    try:
        state = json.load(open(f"{ROOT}/v2conf_diag/s9_state.json"))
    except FileNotFoundError:
        state = {"cumulative_s": 0.0, "phase": "build"}
    B = Budget(state)
    save_state = lambda: json.dump(
        state, open(f"{ROOT}/v2conf_diag/s9_state.json", "w"))

    # ---- BUILD augmented dataset (train-A36 only) ----
    if state["phase"] == "build":
        ds = json.load(open(f"{ROOT}/v2conf_diag/s3_data.json"))
        new = json.load(open(f"{ROOT}/v2conf_diag/s8_new18.json"))
        s3r = json.load(open(f"{ROOT}/v2conf_diag/s3_results.json"))
        tr_ids = set(s3r["partition"]["train"])
        evc = {e["id"]: e for e in json.load(
            open(f"{ROOT}/v2conf/eval_conf_v1.json"))}
        acq8 = {e["id"]: e for e in json.load(
            open(f"{ROOT}/v2conf_diag/s8_acq8.json"))}
        tr_all = json.load(open(f"{ROOT}/v2conf_diag/s2_traces.json"))
        tr_all.update(json.load(
            open(f"{ROOT}/v2conf_diag/s3_repro_traces.json")))
        teach = json.load(open(f"{ROOT}/v2conf_diag/s8_state.json"))["teach"]
        teach_by = {(v["id"], v["theta"]): v for v in teach.values()
                    if v.get("status") == "complete"}
        aug, hsprov = {}, {}
        for s in SEEDS:
            d = ds["thetas"][str(s)]
            m = np.array([i in tr_ids for i in d["ids"]])
            rows = [("orig", i, np.array(o), y)
                    for i, o, y in zip(np.array(d["ids"])[m],
                                       np.array(d["obs"])[m],
                                       np.array(d["labels"])[m])]
            nd = new["thetas"][str(s)]
            rows += [("new", i, np.array(o), int(y))
                     for i, o, y in zip(nd["ids"], nd["obs"], nd["labels"])]
            exp_n = fr["data"]["counts_per_theta"][str(s)]
            assert len(rows) == exp_n, (s, len(rows), exp_n)
            # Order/labels verified against originals below during replay.
            out_rows, last = [], None
            by_id = {}
            for src, i, o, y in rows:
                by_id.setdefault((src, i), []).append((o, y))
            for (src, i), lst in sorted(
                    by_id.items(),
                    key=lambda kv: (0 if kv[0][0] == "orig" else 1,
                                    kv[0][1])):
                if src == "orig":
                    e = evc[i]
                    tr = tr_all.get(f"AB_{i}_{s}")
                else:
                    e = acq8[i]
                    tr = teach_by.get((i, s))
                assert tr is not None, (src, i, s)
                assert len(tr["steps"]) == len(lst), (src, i, s)
                goal = np.array(e["cfg"]["goal"]["pos"])
                hs = float(e["cfg"]["hazard"]["speed"])
                hsprov[i] = hs
                r = TaskRunner()
                obs = r.reset(e["family"], e["cfg"], int(e["seed"]))
                assert float(r.env.haz_speed) == hs, i
                for st, (o_exp, y_exp) in zip(tr["steps"], lst):
                    live = (r.env.agent_pos, r.env.agent_vel, r.env.haz_pos)
                    rc = (np.array(st["apos"]), np.array(st["avel"]),
                          np.array(st["haz"]))
                    assert all(np.allclose(a, b, atol=2e-3) for a, b in
                               zip(rc, live)), (i, st["t"])
                    assert np.array_equal(replica_obs(live[0], live[1], goal,
                                                      live[2]), obs)
                    assert np.array_equal(obs, o_exp), (i, st["t"], "repro")
                    assert int(st["skill"]) == int(y_exp), (i, st["t"])
                    inf = o_exp.copy()
                    inf[8] = float(live[0][0]) / 100.0
                    inf[9] = float(live[0][1]) / 100.0
                    inf[10] = hs / 5.0
                    inf[11] = float(r.env.t) / 200.0
                    assert np.array_equal(o_exp, obs), "pristine intact"
                    assert (inf[12:14] == 0).all() and \
                        (o_exp[8:14] == 0).all()
                    out_rows.append((i, o_exp.copy(), inf, int(y_exp)))
                    obs, _, done, info = r.step(int(st["a"]))
            # IDs/labels/order vs originals: exact flat-sequence match.
            exp_flat = [(i, int(y)) for (_, i, _, y) in rows]
            got_flat = [(i, y) for (i, _, _, y) in out_rows]
            assert got_flat == exp_flat, (s, "id/label/order mismatch")
            aug[str(s)] = {"ids": [i for (i, _) in got_flat],
                           "labels": [y for (_, y) in got_flat],
                           "obs_orig": [[float(x) for x in o]
                                        for (_, o, _, _) in out_rows],
                           "obs_inf": [[float(x) for x in o]
                                       for (_, _, o, _) in out_rows]}
            print(f"build th={s}: {len(out_rows)} pairs, "
                  f"order/labels exact", flush=True)
        # Strict order check: rebuild expected flat order and compare.
        json.dump({"thetas": aug, "hspeed_provenance_cfg": hsprov},
                  open(f"{ROOT}/v2conf_diag/s9_data.json", "w"))
        state["phase"] = "train"
        save_state()
        print("build done", flush=True)

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

    # ---- TRAIN informed ----
    if state["phase"] == "train":
        dd = json.load(open(f"{ROOT}/v2conf_diag/s9_data.json"))["thetas"]
        try:
            models = json.load(open(f"{ROOT}/v2conf_diag/s9_models.json"))
        except FileNotFoundError:
            models = {}
        for s in SEEDS:
            if str(s) in models:
                continue
            if not B.ok():
                state["phase"] = "incomplete"
                B.close()
                save_state()
                print("TRAIN INCOMPLETE: budget", flush=True)
                return
            X = np.array(dd[str(s)]["obs_inf"])
            y = np.array(dd[str(s)]["labels"], dtype=int)
            assert len(X) == fr["data"]["counts_per_theta"][str(s)]
            models[str(s)] = [float(x) for x in adam(init(s), X, y)]
            json.dump(models, open(f"{ROOT}/v2conf_diag/s9_models.json", "w"))
            print(f"train informed th={s} done", flush=True)
        state["phase"] = "eval"
        save_state()

    # ---- EVAL both arms on ACQ10 (opens only after training) ----
    if state["phase"] == "eval":
        acq9 = json.load(open(f"{ROOT}/v2conf_diag/s9_acq10.json"))
        assert len(acq9) == 200
        m36 = json.load(open(f"{ROOT}/v2conf_diag/s8_models36.json"))
        mi = json.load(open(f"{ROOT}/v2conf_diag/s9_models.json"))
        assert set(mi) == {"301", "302", "303"}
        try:
            out = json.load(open(f"{ROOT}/v2conf_diag/s9_eval.json"))
        except FileNotFoundError:
            out = {"records": []}
        done = {(r["arm"], r["theta"], r["id"]) for r in out["records"]
                if r.get("status") == "complete"}
        save = lambda: json.dump(out, open(
            f"{ROOT}/v2conf_diag/s9_eval.json", "w"))
        from models import split_modular as _sp
        skills = {s: _sp(np.load(
            f"{ROOT}/v2conf_runs/VC_{s}_theta.npz")["theta"])[0]
            for s in SEEDS}

        def orig_fn(gth, sk_all):
            def f(obs):
                assert (obs[8:14] == 0).all()
                k = int(np.argmax(mlp_forward(gth, obs, GATE_SHAPES)))
                return S4.greedy_skill(sk_all, obs, k), k
            return f

        def inf_fn(gth, sk_all):
            def f(obs, env):
                assert (obs[8:14] == 0).all()
                g = obs.copy()
                g[8] = float(env.agent_pos[0]) / 100.0
                g[9] = float(env.agent_pos[1]) / 100.0
                g[10] = float(env.haz_speed) / 5.0
                g[11] = float(env.t) / 200.0
                # Pristine obs (with zero slots, asserted above) is the
                # object passed to skills; gate reads the copy g only.
                k = int(np.argmax(mlp_forward(gth, g, GATE_SHAPES)))
                return S4.greedy_skill(sk_all, obs, k), k
            return f

        for arm in ("original", "informed"):
            for s in SEEDS:
                gth = np.array(m36[str(s)] if arm == "original"
                               else mi[str(s)])
                sk_all = skills[s]
                fn = orig_fn(gth, sk_all)
                for e in acq9:
                    if (arm, s, e["id"]) in done:
                        continue
                    if not B.ok():
                        out["records"].append(
                            {"arm": arm, "theta": s, "id": e["id"],
                             "status": "incomplete"})
                        save()
                        continue
                    if arm == "original":
                        rec = S4.run_ep(e, fn)
                    else:
                        # Same loop; gate sees augmented copy, skills pristine.
                        r = TaskRunner()
                        obs = r.reset(e["family"], e["cfg"], int(e["seed"]))
                        assert r.env.haz_pos is not None
                        done_, info, n = False, {}, 0
                        uses, sw, prev = [0, 0, 0, 0], 0, None
                        gif = inf_fn(gth, sk_all)
                        while not done_:
                            a, k = gif(obs, r.env)
                            uses[k] += 1
                            sw += (prev is not None and k != prev)
                            prev = k
                            obs, _, done_, info = r.step(a)
                            n += 1
                        env = r.env
                        if info["success"]:
                            cause = "arrival"
                        elif float(np.linalg.norm(
                                env.haz_pos - env.agent_pos)) < 5.0:
                            cause = "contact-death"
                        elif env.t >= env.MAX_STEPS:
                            cause = "timeout"
                        else:
                            cause = "other"
                        rec = {"cause": cause, "T": n, "uses": uses,
                               "switches": sw}
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
