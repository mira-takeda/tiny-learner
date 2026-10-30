"""N1 imitation: 14->32->4 gate per theta on planner demos (G1 A/B +
G2 A/C labels; D absent, documented). S3 recipe: Xavier hidden,
zero output, Adam full-batch lr=0.001, 5000 epochs, final ckpt.
Frozen: n1_freeze.json. Offline only; no control eval here.
"""
import json
import sys
import time

sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/src")

import numpy as np

from models import GATE_SHAPES
from policies import mlp_forward
from policies import unpack as unpack_shapes

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
SEEDS = (301, 302, 303)


def unpack(th):
    m = unpack_shapes(th, GATE_SHAPES)
    return [m[0], m[2]], [m[1], m[3]]


def ce_grads(th, X, y):
    Ws, bs = unpack(th)
    h = X @ Ws[0] + bs[0]
    a = np.tanh(h)
    z = a @ Ws[1] + bs[1]
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    n = len(X)
    loss = float(-np.log(p[np.arange(n), y] + 1e-12).mean())
    dz = p.copy()
    dz[np.arange(n), y] -= 1
    dz /= n
    dW1 = a.T @ dz
    db1 = dz.sum(0)
    da = dz @ Ws[1].T * (1 - a ** 2)
    dW0 = X.T @ da
    db0 = da.sum(0)
    g = np.concatenate([dW0.ravel(), db0, dW1.ravel(), db1])
    return loss, g, p


def main():
    t0 = time.time()
    fr = json.load(open(f"{ROOT}/v2conf_diag/n1_freeze.json"))
    demos = json.load(open(f"{ROOT}/v2conf_diag/n1_demos.json"))["runs"]
    assert all(r["status"] == "complete" for r in demos), "demos incomplete"
    assert len(demos) == 108
    part = fr["partition"]
    data = {}
    for s in SEEDS:
        for split in ("train", "eval"):
            Xs, ys = [], []
            for r in demos:
                if r["theta"] == s and r["id"] in part[split]:
                    for st in r["steps"]:
                        Xs.append(st["obs"])
                        ys.append(st["k"])
            data[(s, split)] = (np.array(Xs), np.array(ys, dtype=int))
            assert len(ys) > 0
    print("pairs:", {k: len(v[1]) for k, v in data.items()}, flush=True)
    print("label balance:",
          {k: np.bincount(v[1], minlength=4).tolist()
           for k, v in data.items()}, flush=True)
    models, results = {}, {}
    for s in SEEDS:
        Xtr, ytr = data[(s, "train")]
        Xev, yev = data[(s, "eval")]
        rng = np.random.default_rng(fr["train"]["seeds"][str(s)])
        th = np.zeros(612)
        Ws, bs = unpack(th)
        # Xavier hidden, zero output (S3 recipe); set via slices.
        off = 0
        lim = np.sqrt(6.0 / (14 + 32))
        Ws[0][:] = rng.uniform(-lim, lim, Ws[0].shape)
        th = np.concatenate([Ws[0].ravel(), bs[0], Ws[1].ravel(), bs[1]])
        th0 = th.copy()
        m = np.zeros_like(th)
        v = np.zeros_like(th)
        lr, b1, b2, eps = 0.001, 0.9, 0.999, 1e-8
        for ep in range(1, 5001):
            loss, g, _ = ce_grads(th, Xtr, ytr)
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            mh = m / (1 - b1 ** ep)
            vh = v / (1 - b2 ** ep)
            th = th - lr * mh / (np.sqrt(vh) + eps)
        assert np.any(th[: 14 * 32] != th0[: 14 * 32]), "hidden unmoved"
        _, _, ptr = ce_grads(th, Xtr, ytr)
        lev, _, pev = ce_grads(th, Xev, yev)
        ltr = float(-np.log(ptr[np.arange(len(ytr)), ytr] + 1e-12).mean())
        atr = float((ptr.argmax(1) == ytr).mean())
        aev = float((pev.argmax(1) == yev).mean())
        cm = np.zeros((4, 4), dtype=int)
        for t, p in zip(yev, pev.argmax(1)):
            cm[int(t), int(p)] += 1
        maj = int(np.bincount(ytr, minlength=4).argmax())
        const = float((np.full_like(yev, maj) == yev).mean())
        models[f"N1_{s}"] = [float(x) for x in th]
        results[str(s)] = {"train_ce": round(ltr, 4), "train_acc": round(atr, 4),
                            "eval_ce": round(lev, 4), "eval_acc": round(aev, 4),
                            "confusion_eval": cm.tolist(),
                            "const_eval": round(const, 4)}
        print(f"th={s}: train CE={ltr:.4f} acc={atr:.4f} | eval CE={lev:.4f} "
              f"acc={aev:.4f} const={const:.4f}", flush=True)
    json.dump(models, open(f"{ROOT}/v2conf_diag/n1_models.json", "w"))
    json.dump(results, open(f"{ROOT}/v2conf_diag/n1_results.json", "w"),
              indent=1)
    print(f"N1-train done wall={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
