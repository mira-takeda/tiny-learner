"""W1 imitation: 14->32->4 gate per theta (N1b recipe verbatim).

Data = 54 new wide-G1 demos + 54 reused G2 demos (n1_demos, pinned)
+ reused singles pairs (n1b_pairs, pinned). S3 recipe: Xavier hidden,
zero output, Adam full-batch lr=0.001, CE, 5000 epochs, final ckpt.
Frozen: w1_freeze.json + MANIFEST. Offline only (no env imports).
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import numpy as np

from gen import sha256_file
from models import GATE_SHAPES
from policies import mlp_forward
from policies import unpack as unpack_shapes
import gen_wide as GW

W1 = HERE
SEEDS = (301, 302, 303)
T_BUDGET = 900.0
FAM2K = {"A": 0, "B": 1, "C": 2, "D": 3}


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
    assert "tasks" not in sys.modules and "env" not in sys.modules
    fr, disk = GW.regen_check()
    pins = fr["pins"]
    assert sha256_file(os.path.join(
        ROOT, "v2conf_diag/n1_demos.json")) == pins["v2conf_diag/n1_demos.json"]
    assert sha256_file(os.path.join(
        ROOT, "v2conf_diag/n1b_pairs.json")) == pins["v2conf_diag/n1b_pairs.json"]
    wd = json.load(open(os.path.join(W1, "w1_demos_g1.json")))["runs"]
    assert len(wd) == 54 and all(r["status"] == "complete" for r in wd)
    g2d = [r for r in json.load(open(os.path.join(
        ROOT, "v2conf_diag/n1_demos.json")))["runs"] if r["fam"] == "G2"]
    assert len(g2d) == 54 and all(r["status"] == "complete" for r in g2d)
    prs = json.load(open(os.path.join(ROOT, "v2conf_diag/n1b_pairs.json")))
    for split, n in (("train", 600), ("eval", 200)):
        for f in "ABCD":
            for s in SEEDS:
                assert len(prs[split][f"{f}_{s}"]) == n, (split, f, s)
    print("reuse verified: 54 G2 demos + singles 600/200 exact", flush=True)
    # Partitions: wide-G1 by frozen index rule; G2 by N1 lists.
    demo_ids = [e["id"] for e in disk["pools"]["demo_g1"]]
    assert len(demo_ids) == 18
    pw = fr["partition_wide_g1"]
    wtr = {demo_ids[i] for i in pw["train_idx"]}
    wev = {demo_ids[i] for i in pw["eval_idx"]}
    assert len(wtr) == 12 and len(wev) == 6 and not (wtr & wev)
    pg = fr["partition_g2_reuse"]
    gtr, gev = set(pg["train"]), set(pg["eval"])
    assert len(gtr) == 12 and len(gev) == 6
    data, sdata = {}, {}
    for s in SEEDS:
        for split, wset, gset in (("train", wtr, gtr), ("eval", wev, gev)):
            Xs, ys = [], []
            for r in wd:
                if r["theta"] == s and r["id"] in wset:
                    for st in r["steps"]:
                        Xs.append(st["obs"])
                        ys.append(st["k"])
            for r in g2d:
                if r["theta"] == s and r["id"] in gset:
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
    inits = fr["imit"]["init_seeds"]
    assert len(inits) == 3
    models, results = {}, {}
    for si, s in enumerate(SEEDS):
        Xtr, ytr = data[(s, "train")]
        Xev, yev = data[(s, "eval")]
        rng = np.random.default_rng(inits[si])
        th = np.zeros(612)
        Ws, bs = unpack(th)
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
        Xs_ev, ys_ev = sdata[(s, "eval")]
        z = mlp_forward(th, Xs_ev, GATE_SHAPES)
        pse = np.exp(z - z.max(1, keepdims=True))
        pse /= pse.sum(1, keepdims=True)
        sel_rate = {f: round(float((pse.argmax(1)[ys_ev == i] == i).mean()), 4)
                    for f, i in FAM2K.items()}
        dmass = float((pse.argmax(1) == 3).sum())
        assert dmass > 0, f"D never predicted on singles-eval th={s}: BUG"
        maj = int(np.bincount(ytr, minlength=4).argmax())
        const = float((np.full_like(yev, maj) == yev).mean())
        models[f"W1I_{s}"] = [float(x) for x in th]
        results[str(s)] = {"train_ce": round(ltr, 4),
                            "train_acc": round(atr, 4),
                            "eval_ce": round(lev, 4),
                            "eval_acc": round(aev, 4),
                            "confusion_eval": cm.tolist(),
                            "const_eval": round(const, 4),
                            "singles_sel_rate": sel_rate,
                            "singles_D_mass": dmass}
        print(f"th={s}: train CE={ltr:.4f} acc={atr:.4f} | eval CE={lev:.4f} "
              f"acc={aev:.4f} const={const:.4f} | singles sel={sel_rate} "
              f"D-mass={dmass:.0f}", flush=True)
    json.dump(models, open(os.path.join(W1, "w1_imit_models.json"), "w"))
    json.dump(results, open(os.path.join(W1, "w1_imit_results.json"), "w"),
              indent=1)
    wall = time.time() - t0
    assert wall < T_BUDGET, wall
    print(f"W1-imit done wall={wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
