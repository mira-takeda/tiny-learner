"""S3 imitation: can the 14->32->4 gate predict planner A/B choices?

Dataset: 81 AB traces (72 arrivals + 9 reproduced failures, pool of
27 scenarios indices 3..29; micropilot 3 excluded). Exact pre-action
obs obtained by replaying saved actions in the frozen env (episode
seeds); labels = recorded planner skills. Validations: temporal
alignment (obs BEFORE its action), obs-replica bit-exact at every
replayed step, rounded-state agreement, outcome agreement.

Split (shared across thetas): per region file-order first 6 train /
last 3 eval -> 18 train / 9 eval complete scenarios.
Student per theta: same arch, CE A/B, Adam full-batch lr=1e-3,
500 epochs, final model, no tuning. Init: hidden Xavier (seeded per
theta, frozen xavier_init), all biases + output weights zero
(uniform initial probs). Baselines: train-majority constant,
distance-15 on the same obs.

Offline imitation only. The student is NOT run as a controller here.
Cumulative budget: repro (s3_repro.json) + this run <= 600 s.
"""
import json
import sys
import time

sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/src")
sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/v2views")

import numpy as np

import views_train as V  # noqa: F401 (kept: views context, unused here)
from env import Arena
from models import GATE_SHAPES
from policies import layer_shapes, softmax, unpack, xavier_init
from tasks import TaskRunner

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
SEEDS = (301, 302, 303)
INIT_BASE = 9200
EPOCHS, LR = 500, 0.001
TOTAL_BUDGET = 600.0


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


def main():
    t0 = time.time()
    repro = json.load(open(f"{ROOT}/v2conf_diag/s3_repro.json"))
    spent = float(repro["repro_wall_s"])
    evc = json.load(open(f"{ROOT}/v2conf/eval_conf_v1.json"))
    by_id = {e["id"]: e for e in evc if e["battery"] == "ACQ"}
    goal_of = {i: np.array(e["cfg"]["goal"]["pos"]) for i, e in by_id.items()}

    traces = json.load(open(f"{ROOT}/v2conf_diag/s2_traces.json"))
    traces.update(json.load(open(f"{ROOT}/v2conf_diag/s3_repro_traces.json")))
    ab = {k: v for k, v in traces.items() if v["mode"] == "AB"}
    assert len(ab) == 81, len(ab)
    pool = sorted({v["id"] for v in ab.values()})
    assert len(pool) == 27, len(pool)

    # Replay in frozen env -> exact pre-action obs + alignment checks.
    data = {s: [] for s in SEEDS}  # (id, obs, label)
    for key, v in sorted(ab.items()):
        e = by_id[v["id"]]
        r = TaskRunner()
        obs = r.reset(e["family"], e["cfg"], int(e["seed"]))
        assert r.env.haz_pos is not None
        for st in v["steps"]:
            rec = (np.array(st["apos"]), np.array(st["avel"]),
                   np.array(st["haz"]))
            live = (r.env.agent_pos, r.env.agent_vel, r.env.haz_pos)
            assert all(np.allclose(a, b, atol=2e-3) for a, b in
                       zip(rec, live)), (key, st["t"])
            assert np.array_equal(
                replica_obs(live[0], live[1], goal_of[v["id"]], live[2]),
                obs), (key, st["t"], "replica")
            assert st["skill"] in (0, 1)
            data[v["theta"]].append((v["id"], obs.copy(), st["skill"]))
            obs, _, done, info = r.step(int(st["a"]))
        assert done, key
        env = r.env
        cause = ("arrival" if info["success"] else
                 ("contact-death" if float(np.linalg.norm(
                     env.haz_pos - env.agent_pos)) < 5.0 else
                  ("timeout" if env.t >= env.MAX_STEPS else "other")))
        assert cause == v.get("cause", "arrival"), (key, cause)
    n_steps = sum(len(v) for v in data.values())
    print(f"replay ok: 81 traces, {n_steps} (obs,label) steps, "
          f"replica bit-exact, alignment pre-action", flush=True)

    # Shared partition: per region file-order first-6 train / last-3 eval.
    reg = {i: by_id[i]["region"] for i in pool}
    train_ids, eval_ids = set(), set()
    for rg in ("G1a", "G1b", "G1adv"):
        ids = sorted([i for i in pool if reg[i] == rg])
        assert len(ids) == 9
        train_ids.update(ids[:6])
        eval_ids.update(ids[6:])
    assert len(train_ids) == 18 and len(eval_ids) == 9
    part = {"train": sorted(train_ids), "eval": sorted(eval_ids)}
    print(f"partition: train={len(train_ids)} eval={len(eval_ids)} "
          f"scenarios", flush=True)

    def split_xy(s, ids):
        rows = [(o, y) for (i, o, y) in data[s] if i in ids]
        return (np.array([o for o, _ in rows]),
                np.array([y for _, y in rows], dtype=int))

    # Model: flat gate theta, 14->tanh32->4, softmax CE over 4 classes.
    shapes = GATE_SHAPES
    assert shapes == layer_shapes(14, [32], 4)

    def fwd(th, X):
        W1, b1, W2, b2 = unpack(th, shapes)
        H = np.tanh(X @ W1 + b1)
        Z = H @ W2 + b2
        Z = Z - Z.max(axis=1, keepdims=True)
        E = np.exp(Z)
        return H, E / E.sum(axis=1, keepdims=True)

    def grads(th, X, y):
        n = len(X)
        W1, b1, W2, b2 = unpack(th, shapes)
        H, P = fwd(th, X)
        dZ = P.copy()
        dZ[np.arange(n), y] -= 1.0
        dZ /= n
        dW2 = H.T @ dZ
        db2 = dZ.sum(axis=0)
        dH = dZ @ W2.T
        dZ1 = dH * (1.0 - H ** 2)
        dW1 = X.T @ dZ1
        db1 = dZ1.sum(axis=0)
        parts = [dW1, db1, dW2, db2]
        return np.concatenate([p.ravel() for p in parts])

    def ce(th, X, y):
        _, P = fwd(th, X)
        return float(-np.log(P[np.arange(len(X)), y] + 1e-300).mean())

    def init_theta(s):
        rng = np.random.default_rng(INIT_BASE + s)
        h = unpack(xavier_init(rng, layer_shapes(14, [32], 4)), shapes)
        W1 = h[0]
        assert np.std([np.linalg.norm(a - b) for a in W1.T for b in W1.T]) \
            > 0, "hidden symmetry"
        return np.concatenate([W1.ravel(), np.zeros(32 + 32 * 4 + 4)])

    # Smoke: numeric grad check + symmetry breaking + beyond-output-bias.
    Xs, ys = split_xy(301, train_ids)
    Xs5, ys5 = Xs[:5], ys[:5]
    th0 = init_theta(301)
    g = grads(th0, Xs5, ys5)
    num = np.zeros_like(g)
    eps = 1e-6
    for j in range(len(g)):
        tp, tm = th0.copy(), th0.copy()
        tp[j] += eps
        tm[j] -= eps
        num[j] = (ce(tp, Xs5, ys5) - ce(tm, Xs5, ys5)) / (2 * eps)
    rel = float(np.linalg.norm(g - num) / max(1e-12, np.linalg.norm(num)))
    assert rel < 1e-6, rel
    def adam_run(th, X, y, epochs, mask=None):
        m, v = np.zeros_like(th), np.zeros_like(th)
        for t in range(1, epochs + 1):
            gg = grads(th, X, y)
            if mask is not None:
                gg = gg * mask
            m = 0.9 * m + 0.1 * gg
            v = 0.999 * v + 0.001 * gg ** 2
            th = th - LR * (m / (1 - 0.9 ** t)) / \
                (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        return th

    # Smoke: loss decreases + hidden moves (50ep); full beats
    # output-bias-only (500ep, frozen hyperparams).
    th50 = adam_run(th0.copy(), Xs, ys, 50)
    assert ce(th50, Xs, ys) < ce(th0, Xs, ys), "no descent"
    dW1 = unpack(th50 - th0, shapes)[0]
    assert float(np.linalg.norm(dW1)) > 0, "hidden frozen"
    sizes = [int(np.prod(s)) for s in shapes]
    mask_b = np.zeros_like(th0)
    mask_b[-sizes[-1]:] = 1.0  # output biases only
    th_full = adam_run(th0.copy(), Xs, ys, EPOCHS)
    th_b = adam_run(th0.copy(), Xs, ys, EPOCHS, mask_b)
    assert ce(th_full, Xs, ys) < ce(th_b, Xs, ys), "hidden adds nothing"
    print(f"smoke ok: grad-rel-err={rel:.1e}; 50ep CE "
          f"{ce(th0, Xs, ys):.3f}->{ce(th50, Xs, ys):.3f} "
          f"|dW1|={np.linalg.norm(dW1):.4f}; 500ep full "
          f"{ce(th_full, Xs, ys):.3f} < bias-only {ce(th_b, Xs, ys):.3f}",
          flush=True)

    adam = adam_run

    def metrics(th, X, y):
        _, P = fwd(th, X)
        pred = P.argmax(axis=1)
        cm = np.zeros((2, 2), dtype=int)
        for t_, p_ in zip(y, pred):
            if p_ < 2:
                cm[t_, p_] += 1
        cd = int((pred >= 2).sum())
        acc = float((pred == y).mean())
        rec = [float(cm[c, c] / max(1, cm[c].sum())) for c in (0, 1)]
        prec = [float(cm[c, c] / max(1, cm[:, c].sum())) for c in (0, 1)]
        return {"n": len(X), "acc": acc, "cm": cm.tolist(),
                "pred_C/D": cd, "recall": rec, "precision": prec,
                "bal_acc": float(np.mean(rec))}

    results = {"partition": part, "init": f"Xavier({INIT_BASE}+theta) "
                                          f"hidden, output zero",
               "epochs": EPOCHS, "lr": LR, "thetas": {}}
    models = {}
    for s in SEEDS:
        if spent + (time.time() - t0) > TOTAL_BUDGET - 5:
            results["thetas"][s] = {"status": "incomplete",
                                    "reason": "budget"}
            continue
        Xtr, ytr = split_xy(s, train_ids)
        Xev, yev = split_xy(s, eval_ids)
        maj = int(np.bincount(ytr).argmax())
        th = adam(init_theta(s), Xtr, ytr, EPOCHS)
        models[str(s)] = {"theta": [float(x) for x in th],
                          "seed": INIT_BASE + s}
        d15 = lambda X: (X[:, 7] * Arena.DMAX < 15).astype(int)
        rep = {"status": "complete", "n_train": len(Xtr),
               "n_eval": len(Xev),
               "train_fracB": round(float(ytr.mean()), 4),
               "eval_fracB": round(float(yev.mean()), 4),
               "majority_train": maj, "student_train": metrics(th, Xtr, ytr),
               "student_eval": metrics(th, Xev, yev)}
        for tag, X, y in (("train", Xtr, ytr), ("eval", Xev, yev)):
            c = np.full_like(y, maj)
            d = d15(X)
            rep[f"const_{tag}"] = {"acc": round(float((c == y).mean()), 4)}
            rep[f"d15_{tag}"] = {"acc": round(float((d == y).mean()), 4)}
            for nm, p in (("const", c), ("d15", d)):
                r0 = float(((p == 0) & (y == 0)).sum() / max(1, (y == 0).sum()))
                r1 = float(((p == 1) & (y == 1)).sum() / max(1, (y == 1).sum()))
                rep[f"{nm}_{tag}"]["bal_acc"] = round((r0 + r1) / 2, 4)
        results["thetas"][s] = rep
        e = rep["student_eval"]
        print(f"th={s}: train n={rep['n_train']} acc="
              f"{rep['student_train']['acc']:.3f} | eval n={rep['n_eval']} "
              f"acc={e['acc']:.3f} bal={e['bal_acc']:.3f} cm={e['cm']} "
              f"CDpred={e['pred_C/D']} | const={rep['const_eval']['acc']:.3f} "
              f"d15={rep['d15_eval']['acc']:.3f}", flush=True)

    results["compute_s"] = {"repro": spent,
                            "this_run": round(time.time() - t0, 1)}
    json.dump(results, open(f"{ROOT}/v2conf_diag/s3_results.json", "w"),
              indent=1)
    json.dump(models, open(f"{ROOT}/v2conf_diag/s3_models.json", "w"))
    ds = {"partition": part,
          "thetas": {str(s): {"ids": [i for (i, _, _) in data[s]],
                              "labels": [int(y) for (_, _, y) in data[s]],
                              "obs": [[float(x) for x in o]
                                      for (_, o, _) in data[s]]}
                     for s in SEEDS}}
    json.dump(ds, open(f"{ROOT}/v2conf_diag/s3_data.json", "w"))
    print(f"saved s3_results/models/data.json; compute total="
          f"{spent + time.time() - t0:.1f}s of {TOTAL_BUDGET:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
