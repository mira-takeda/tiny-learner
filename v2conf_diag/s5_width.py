"""S5: supervised width x budget contrast on frozen S3 data.

Six trainings (widths 32/128 x thetas 301-303) to 5000 epochs,
full-batch Adam lr=1e-3, CE A/B, same obs/labels/partition as S3.
Epoch-500 cells come from the same run's checkpoint. Width-32
epoch-500 MUST reproduce s3_models (else pause + investigate).
ONE init per width x theta (9200/9300+theta): observed differences
do not identify general capacity effects. Offline only. Budget
300 s, incremental save; overruns -> INCOMPLETE + available ckpts.
"""
import json
import sys
import time

sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/src")

import numpy as np

from policies import layer_shapes, unpack, xavier_init

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
SEEDS = (301, 302, 303)
WIDTHS = (32, 128)
INIT = {32: 9200, 128: 9300}
EPOCHS = 5000
CKPT = 500
LR = 0.001
EVAL_GRID = (50, 100, 250, 500, 1000, 2500, 5000)
T_BUDGET = 300.0
REPRO_TOL = 1e-9


def main():
    t0 = time.time()
    ds = json.load(open(f"{ROOT}/v2conf_diag/s3_data.json"))
    s3r = json.load(open(f"{ROOT}/v2conf_diag/s3_results.json"))
    s3m = json.load(open(f"{ROOT}/v2conf_diag/s3_models.json"))
    train_ids, eval_ids = set(s3r["partition"]["train"]), \
        set(s3r["partition"]["eval"])
    assert len(train_ids) == 18 and len(eval_ids) == 9

    data = {}
    for s in SEEDS:
        d = ds["thetas"][str(s)]
        X = np.array(d["obs"])
        # Precision 1: full 14 gate obs; views applied only to skills
        # during teacher generation, never masking student inputs.
        assert X.shape[1] == 14, X.shape
        both = sum(1 for o in X if np.any(o[2:5] != 0)
                   and np.any(o[5:8] != 0))
        assert both > 0, "inputs look view-masked"
        y = np.array(d["labels"], dtype=int)
        ids = d["ids"]
        mtr = np.array([i in train_ids for i in ids])
        mev = np.array([i in eval_ids for i in ids])
        data[s] = (X[mtr], y[mtr], X[mev], y[mev])
    print(f"inputs verified: full 14-dim obs, S1+S2 jointly active "
          f"(no view masking)", flush=True)

    out = {"widths": list(WIDTHS), "epochs": EPOCHS, "ckpt500": CKPT,
           "init": {str(w): f"Xavier({b}+theta) hidden, rest zero"
                    for w, b in INIT.items()},
           "init_note": "ONE init per width x theta; differences do "
                        "not identify general capacity effects",
           "flops_formula": "per-sample fwd MACs=18H (14H+4H); "
                            "bwd=2x fwd; per-epoch=N*3*2*18H + 12P "
                            "(Adam); total=epochs*(108*H*N + 12P)",
           "runs": {}}
    save = lambda: json.dump(out, open(f"{ROOT}/v2conf_diag/s5_results.json",
                                       "w"))

    def make(H):
        shapes = layer_shapes(14, [H], 4)
        P = sum(int(np.prod(s)) for s in shapes)

        def fwd(th, X):
            W1, b1, W2, b2 = unpack(th, shapes)
            Hh = np.tanh(X @ W1 + b1)
            Z = Hh @ W2 + b2
            Z = Z - Z.max(axis=1, keepdims=True)
            E = np.exp(Z)
            return Hh, E / E.sum(axis=1, keepdims=True)

        def grads(th, X, y):
            n = len(X)
            W1, b1, W2, b2 = unpack(th, shapes)
            Hh, Pp = fwd(th, X)
            dZ = Pp.copy()
            dZ[np.arange(n), y] -= 1.0
            dZ /= n
            dW2 = Hh.T @ dZ
            db2 = dZ.sum(axis=0)
            dZ1 = (dZ @ W2.T) * (1.0 - Hh ** 2)
            return np.concatenate([(X.T @ dZ1).ravel(), dZ1.sum(axis=0),
                                   dW2.ravel(), db2])

        def ce(th, X, y):
            _, Pp = fwd(th, X)
            return float(-np.log(Pp[np.arange(len(X)), y] + 1e-300).mean())

        def init(s):
            rng = np.random.default_rng(INIT[H] + s)
            W1 = unpack(xavier_init(rng, shapes), shapes)[0]
            z = np.zeros(P - W1.size)
            return np.concatenate([W1.ravel(), z])

        return shapes, P, fwd, grads, ce, init

    def metrics(fwd, th, X, y):
        _, Pp = fwd(th, X)
        pred = Pp.argmax(axis=1)
        cm = np.zeros((2, 2), dtype=int)
        for t_, p_ in zip(y, pred):
            if p_ < 2:
                cm[t_, p_] += 1
        rec = [float(cm[c, c] / max(1, cm[c].sum())) for c in (0, 1)]
        prec = [float(cm[c, c] / max(1, cm[:, c].sum())) for c in (0, 1)]
        return {"acc": round(float((pred == y).mean()), 4),
                "bal_acc": round(float(np.mean(rec)), 4),
                "cm": cm.tolist(), "pred_C/D": int((pred >= 2).sum()),
                "recall": [round(x, 4) for x in rec],
                "precision": [round(x, 4) for x in prec]}

    order = [(32, s) for s in SEEDS] + [(128, s) for s in SEEDS]
    for H, s in order:
        key = f"{H}_{s}"
        if time.time() - t0 > T_BUDGET - 5:
            out["runs"][key] = {"status": "incomplete",
                                "reason": "budget-before-start"}
            save()
            continue
        shapes, P, fwd, grads, ce, init = make(H)
        Xtr, ytr, Xev, yev = data[s]
        th = init(s)
        m, v = np.zeros_like(th), np.zeros_like(th)
        curves, ckpts, abort = {"train_ce": []}, {}, False
        for t in range(1, EPOCHS + 1):
            if time.time() - t0 > T_BUDGET - 5:
                abort = True
                break
            gg = grads(th, Xtr, ytr)
            m = 0.9 * m + 0.1 * gg
            v = 0.999 * v + 0.001 * gg ** 2
            th = th - LR * (m / (1 - 0.9 ** t)) / \
                (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
            if t % 10 == 0 or t <= 10:
                curves["train_ce"].append([t, round(ce(th, Xtr, ytr), 5)])
            if t in EVAL_GRID:
                curves.setdefault("eval", []).append(
                    [t, round(ce(th, Xev, yev), 5)])
            if t in (CKPT, EPOCHS):
                ckpts[t] = [float(x) for x in th]
        run = {"status": "complete" if not abort else "incomplete",
               "H": H, "theta": s, "n_train": len(Xtr),
               "n_eval": len(Xev),
               "updates": EPOCHS if not abort else None,
               "sample_epochs": (EPOCHS * len(Xtr)) if not abort else None,
               "flops_approx": EPOCHS * (108 * H * len(Xtr) + 12 * P),
               "curves": curves, "cells": {}, "ckpts": ckpts}
        for ep, vec in ckpts.items():
            thv = np.array(vec)
            run["cells"][str(ep)] = {
                "train_ce": round(ce(thv, Xtr, ytr), 5),
                "eval_ce": round(ce(thv, Xev, yev), 5),
                "train": metrics(fwd, thv, Xtr, ytr),
                "eval": metrics(fwd, thv, Xev, yev)}
        # Repro check width-32 @500 BEFORE interpreting anything.
        if H == 32 and CKPT in ckpts:
            ref = np.array(s3m[str(s)]["theta"])
            d = float(np.abs(np.array(ckpts[CKPT]) - ref).max())
            run["repro_maxabs"] = d
            print(f"width32 th={s} ep500 repro maxabs={d:.2e}",
                  flush=True)
            assert d < REPRO_TOL, f"S3 REPRO FAILED th={s}: {d}"
        out["runs"][key] = run
        if H == 32:
            json.dump({k: v for k, v in out["runs"].items()
                       if k.startswith("32_")},
                      open(f"{ROOT}/v2conf_diag/s5_models32.json", "w"))
        save()
        st = run["status"]
        c = run["cells"].get(str(CKPT), {})
        print(f"[{key}] {st} ep500 eval_acc="
              f"{(c.get('eval') or {}).get('acc', '?')} "
              f"wall={time.time() - t0:.0f}s", flush=True)
        if abort:
            continue

    out["wall_s"] = round(time.time() - t0, 1)
    save()
    n_inc = sum(1 for v in out["runs"].values()
                if v.get("status") != "complete")
    print(f"saved s5_results.json wall={out['wall_s']}s "
          f"incomplete={n_inc}", flush=True)


if __name__ == "__main__":
    main()
