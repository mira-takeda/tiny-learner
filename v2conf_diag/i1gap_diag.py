"""I1-vs-G1 gap diagnostic (OFFLINE/READ-ONLY).

Compares configs (analytic geometry + analytic t0 obs slots) across:
G1-train-seen (N1 demos + N2 pools), G1-heldouts 1/2/3, full train_meta
G1 reference, and I1. Frozen N3b gate evaluated forward-only on t0
obs (no episodes, no learning). I1 errors joined from EXISTING
n5_eval records (no new I1 runs). No H reads. No training.
"""
import json
import sys

sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/src")

import numpy as np

from models import GATE_SHAPES, mlp_forward

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
DMAX = 100.0 * np.sqrt(2.0)


def geom(e):
    a = np.array(e["cfg"]["agent"], float)
    g = np.array(e["cfg"]["goal"]["pos"], float)
    h = np.array(e["cfg"]["hazard"]["pos"], float)
    vg, vh = g - a, h - a
    dag, dah, dgh = (float(np.linalg.norm(v)) for v in
                     (vg, vh, g - h))
    cosang = float(np.dot(vg, vh) / max(dag * dah, 1e-12))
    ang = float(np.arccos(np.clip(cosang, -1, 1)))
    return {"d_ag": dag, "d_ah": dah, "d_gh": dgh, "ang": ang,
            "bear_g": float(np.arctan2(vg[1], vg[0])),
            "bear_h": float(np.arctan2(vh[1], vh[0])),
            "ax": a[0], "ay": a[1], "hspeed": float(
                e["cfg"]["hazard"].get("speed", 4.5))}


def t0obs(gm):
    o = np.zeros(14)
    o[2] = np.cos(gm["bear_g"])
    o[3] = np.sin(gm["bear_g"])
    o[4] = gm["d_ag"] / DMAX
    o[5] = np.cos(gm["bear_h"])
    o[6] = np.sin(gm["bear_h"])
    o[7] = gm["d_ah"] / DMAX
    return o


def main():
    meta = json.load(open(f"{ROOT}/data/train_meta_v1.json"))
    ctrl = json.load(open(f"{ROOT}/data/train_control_v1.json"))
    tb = json.load(open(f"{ROOT}/data/test_battery_v1.json"))
    n1 = json.load(open(f"{ROOT}/v2conf_diag/n1_freeze.json"))
    n2 = json.load(open(f"{ROOT}/v2conf_diag/n2_freeze.json"))
    n3 = json.load(open(f"{ROOT}/v2conf_diag/n3_freeze.json"))
    n4 = json.load(open(f"{ROOT}/v2conf_diag/n4_freeze.json"))
    by_id = {e["id"]: e for e in meta + ctrl}
    i1 = [e for e in tb if e.get("family") == "I1"]
    assert len(i1) == 200 and all(e["id"] not in by_id for e in i1)
    groups = {
        "train_seen": [by_id[i] for i in
                       n1["demos"]["G1"] + n2["pools"]["G1"]],
        "heldout1": [by_id[i] for i in n2["heldout"]["G1"]],
        "heldout2": [by_id[i] for i in n3["heldout2"]["G1"]],
        "heldout3": [by_id[i] for i in n4["heldout3"]["G1"]],
        "meta400": [e for e in meta if e.get("family") == "G1"],
        "I1": i1,
    }
    feats = ["d_ag", "d_ah", "d_gh", "ang", "hspeed",
             "ax", "ay", "bear_g", "bear_h"]
    G = {k: [geom(e) for e in v] for k, v in groups.items()}
    O = {k: np.array([t0obs(g) for g in v]) for k, v in G.items()}
    rep = {}
    # 1. ranges per group
    rep["ranges"] = {k: {f: [round(float(min(g[f] for g in v)), 2),
                                 round(float(max(g[f] for g in v)), 2)]
                          for f in feats} for k, v in G.items()}
    # 2. coverage: I1 (and heldouts) outside train_seen per-feature box
    lo = {f: min(g[f] for g in G["train_seen"]) for f in feats}
    hi = {f: max(g[f] for g in G["train_seen"]) for f in feats}
    cov = {}
    for k in ("heldout1", "heldout2", "heldout3", "I1"):
        oos_f = {}
        for f in feats:
            oos_f[f] = round(float(np.mean(
                [(g[f] < lo[f] or g[f] > hi[f]) for g in G[k]])), 4)
        joint = float(np.mean([
            any(g[f] < lo[f] or g[f] > hi[f] for f in feats)
            for g in G[k]]))
        cov[k] = {"per_feature_oos": oos_f, "joint_oos": round(joint, 4)}
    rep["coverage_vs_train_seen"] = cov
    # 3. standardized mean shifts (I1 vs train_seen) + NN distance calib
    M = np.array([[g[f] for f in feats] for g in G["train_seen"]])
    mu, sd = M.mean(0), M.std(0) + 1e-12
    shifts = {}
    for k in ("heldout1", "heldout2", "heldout3", "I1", "meta400"):
        Mk = np.array([[g[f] for f in feats] for g in G[k]])
        shifts[k] = {f: round(float((Mk[:, j].mean() - mu[j]) / sd[j]), 3)
                     for j, f in enumerate(feats)}
    rep["std_mean_shift_vs_train"] = shifts
    Z = (M - mu) / sd
    nnd = {}
    for k in ("heldout1", "heldout2", "heldout3", "I1"):
        Mk = (np.array([[g[f] for f in feats] for g in G[k]]) - mu) / sd
        d = np.sqrt(((Mk[:, None, :] - Z[None, :, :]) ** 2).sum(-1)).min(1)
        nnd[k] = {"mean": round(float(d.mean()), 3),
                  "p90": round(float(np.quantile(d, 0.9)), 3)}
    rep["nn_dist_to_train"] = nnd
    # 4. frozen N3b gate forward-only on t0 obs
    m3 = json.load(open(f"{ROOT}/v2conf_diag/n3b_models.json"))
    sel = {}
    for s in (301, 302, 303):
        th = np.array(m3[f"N3b_{s}"])
        sel[str(s)] = {}
        for k in G:
            p = mlp_forward(th, O[k], GATE_SHAPES)
            a = p.argmax(1)
            sel[str(s)][k] = [int((a == c).sum()) for c in range(4)]
    rep["n3b_t0_selection_counts"] = sel
    # 5. join existing I1 outcomes (n5) + heldout outcomes with geometry
    n5 = {(r["theta"], r["id"]): r for r in json.load(
        open(f"{ROOT}/v2conf_diag/n5_eval.json"))["records"]
        if r["arm"] == "N3b"}
    assert len(n5) == 600
    idi1 = {e["id"]: geom(e) for e in i1}
    err = {"d_gh": [], "ang": [], "d_ah": [], "d_ag": []}
    okk = {"d_gh": [], "ang": [], "d_ah": [], "d_ag": []}
    for (s, i), r in n5.items():
        d = err if r["cause"] != "arrival" else okk
        for f in d:
            d[f].append(idi1[i][f])
    rep["i1_fail_vs_ok_geometry"] = {
        f: {"fail_mean": round(float(np.mean(err[f])), 2),
            "ok_mean": round(float(np.mean(okk[f])), 2),
            "fail_n": len(err[f]), "ok_n": len(okk[f])} for f in err}
    # fail rate by d_gh quartile (bins from train_seen)
    qs = np.quantile([g["d_gh"] for g in G["train_seen"]], [0, .25, .5, .75, 1])
    bins = []
    for b in range(4):
        ids = [i for i in idi1
               if qs[b] <= idi1[i]["d_gh"] <= qs[b + 1] + (b == 3)]
        outs = [n5[(s, i)]["cause"] != "arrival"
                for i in ids for s in (301, 302, 303)]
        bins.append({"d_gh_range": [round(float(qs[b]), 1),
                                       round(float(qs[b + 1]), 1)],
                     "n_i1": len(ids),
                     "fail_rate": round(float(np.mean(outs)), 3)})
    rep["i1_fail_by_train_dgh_quartile"] = bins
    # same for angle
    qa = np.quantile([g["ang"] for g in G["train_seen"]], [0, .25, .5, .75, 1])
    binsa = []
    for b in range(4):
        ids = [i for i in idi1
               if qa[b] <= idi1[i]["ang"] <= qa[b + 1] + (b == 3)]
        outs = [n5[(s, i)]["cause"] != "arrival"
                for i in ids for s in (301, 302, 303)]
        binsa.append({"ang_range": [round(float(qa[b]), 2),
                                      round(float(qa[b + 1]), 2)],
                      "n_i1": len(ids),
                      "fail_rate": round(float(np.mean(outs)), 3)
                      if outs else None})
    rep["i1_fail_by_train_ang_quartile"] = binsa
    json.dump(rep, open(f"{ROOT}/v2conf_diag/i1gap.json", "w"), indent=1)
    print("=== ranges (train_seen vs I1) ===")
    for f in feats:
        print(f"  {f}: train={rep['ranges']['train_seen'][f]} "
              f"I1={rep['ranges']['I1'][f]}")
    print("=== joint OOS vs train_seen ===",
          {k: v["joint_oos"] for k, v in cov.items()})
    print("=== per-feature I1 OOS ===", cov["I1"]["per_feature_oos"])
    print("=== std shifts I1 ===", shifts["I1"])
    print("=== NN dist ===", nnd)
    print("=== N3b t0 selection (A/B/C/D counts) ===")
    for s in sel:
        print("  th=" + s, {k: v for k, v in sel[s].items()})
    print("=== I1 fail-vs-ok geometry ===", rep["i1_fail_vs_ok_geometry"])
    print("=== fail by d_gh quartile ===", bins)
    print("=== fail by ang quartile ===", binsa)


if __name__ == "__main__":
    main()
