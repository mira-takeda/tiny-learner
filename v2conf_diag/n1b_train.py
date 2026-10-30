"""N1b (N1-corrected): same recipe/seeds/from-scratch, dataset = 108
G1/G2 demos (intact) + singles A/B/C/D trivial labels (600 train
+ 200 eval pairs/class/theta, exact). Corrects the N1 omission
(singles planned in original protocol; D got CE negative pressure).
Frozen: n1b_freeze.json. Regression assert: D-mass>0 on singles-eval.
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
    fr = json.load(open(f"{ROOT}/v2conf_diag/n1b_freeze.json"))
    fr1 = json.load(open(f"{ROOT}/v2conf_diag/n1_freeze.json"))
    import hashlib
    assert hashlib.sha256(open(f"{ROOT}/v2conf_diag/n1_demos.json","rb").read()).hexdigest() == fr["g1g2_sha"]
    demos = json.load(open(f"{ROOT}/v2conf_diag/n1_demos.json"))["runs"]
    assert all(r["status"] == "complete" for r in demos), "demos incomplete"
    assert len(demos) == 108
    part = fr1["partition"]
    prs = json.load(open(f"{ROOT}/v2conf_diag/n1b_pairs.json"))
    for split, n in (("train", 600), ("eval", 200)):
        for f in "ABCD":
            for s in SEEDS:
                assert len(prs[split][f"{f}_{s}"]) == n, (split, f, s)
    print("singles balance exact: 600/200 per class/theta/split", flush=True)
    data, sdata = {}, {}
    for s in SEEDS:
        for split in ("train", "eval"):
            Xs, ys = [], []
            for r in demos:
                if r["theta"] == s and r["id"] in part[split]:
                    for st in r["steps"]:
                        Xs.append(st["obs"])
                        ys.append(st["k"])
            for i, f in enumerate("ABCD"):
                for o in prs[split][f"{f}_{s}"]:
                    Xs.append(o)
                    ys.append(i)
            data[(s, split)] = (np.array(Xs), np.array(ys, dtype=int))
            assert len(ys) > 0
            Xs2, ys2 = [], []
            for i, f in enumerate("ABCD"):
                for o in prs[split][f"{f}_{s}"]:
                    Xs2.append(o)
                    ys2.append(i)
            sdata[(s, split)] = (np.array(Xs2), np.array(ys2, dtype=int))
    print("pairs:", {k: len(v[1]) for k, v in data.items()}, flush=True)
    print("label balance:",
          {k: np.bincount(v[1], minlength=4).tolist()
           for k, v in data.items()}, flush=True)
    models, results = {}, {}
    FAM2K = {"A": 0, "B": 1, "C": 2, "D": 3}
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
        rec = {}
        for c in range(4):
            m = yev == c
            rec[c] = float((pev.argmax(1)[m] == c).mean()) if m.sum() else None
        mprob = {str(c): [round(float(x), 4) for x in
                           pev[yev == c].mean(0)] for c in range(4)}
        Xs_ev, ys_ev = sdata[(s, "eval")]
        pse = np.exp(mlp_forward(th, Xs_ev, GATE_SHAPES) -
                     mlp_forward(th, Xs_ev, GATE_SHAPES).max(
                         1, keepdims=True))
        pse /= pse.sum(1, keepdims=True)
        sel = (pse.argmax(1) == ys_ev)
        sel_rate = {f: round(float(sel[ys_ev == i].mean()), 4)
                    for f, i in FAM2K.items()}
        dmass = float((pse.argmax(1) == 3).sum())
        assert dmass > 0, f"D never predicted on singles-eval th={s}: BUG"
        srec = {f: round(float((pse.argmax(1)[ys_ev == i] == i).mean()), 4)
                for f, i in FAM2K.items()}
        maj = int(np.bincount(ytr, minlength=4).argmax())
        const = float((np.full_like(yev, maj) == yev).mean())
        models[f"N1b_{s}"] = [float(x) for x in th]
        results[str(s)] = {"train_ce": round(ltr, 4), "train_acc": round(atr, 4),
                            "eval_ce": round(lev, 4), "eval_acc": round(aev, 4),
                            "confusion_eval": cm.tolist(),
                            "const_eval": round(const, 4),
                            "recall_eval": rec, "meanprob_eval": mprob,
                            "singles_sel_rate": sel_rate,
                            "singles_recall": srec,
                            "singles_D_mass": dmass}
        print(f"th={s}: train CE={ltr:.4f} acc={atr:.4f} | eval CE={lev:.4f} "
              f"acc={aev:.4f} const={const:.4f} | singles sel={sel_rate} "
              f"D-mass={dmass:.0f}", flush=True)
    json.dump(models, open(f"{ROOT}/v2conf_diag/n1b_models.json", "w"))
    json.dump(results, open(f"{ROOT}/v2conf_diag/n1b_results.json", "w"),
              indent=1)
    print(f"N1b-train done wall={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
