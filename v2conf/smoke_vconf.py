"""v2conf local smoke (discarded outputs, /tmp/vconf_smoke).

T0. Conflict datasets valid: counts, regions, db band 12-25 on all
    600 conflict cfgs, speeds/spawns per REGIONS rules, family G1,
    file hashes match frozen constants.
T1. Fitness pool composition: instrumented 2-gen run records cfg
    identity per fitness call -> exactly 2xG1-orig + 2xG1-conf +
    4xG2 per eval, in fixed order.
T2. Ablation correctness: weights bit-identical to normal blend; B
    term uniform; other skills identical; p differs only via B.
T3. Tiny 12-gen run: exact steps, 3 files, skills frozen + gate
    changed, ACQ keys (normal+weights/greedy/ablated/skills), H/G1G2
    keys, hashes. ACQ weight-eval == run_battery on subset.
T4. No obs mutation; training RNGs bit-identical around evals.
T5. Production constants: 41.28M, seeds, POOL_PLAN, starts.

Exit 0 = SMOKE PASS.
"""

import json
import os
import shutil
import sys

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
sys.path.insert(0, f"{ROOT}/v2conf")
sys.path.insert(0, f"{ROOT}/v2views")
sys.path.insert(0, f"{ROOT}/src")

import numpy as np

import conf_train as C
import views_train as V
from evaluate import run_battery
from gen import sha256_file, REGIONS
from models import split_modular

OUT = "/tmp/vconf_smoke"


def main():
    # T0. dataset validity
    tc = json.load(open(f"{ROOT}/v2conf/train_conflict_v1.json"))
    ev = json.load(open(f"{ROOT}/v2conf/eval_conf_v1.json"))
    assert len(tc) == 400 and len([e for e in ev if e["battery"] == "ACQ"]) == 200
    from collections import Counter
    assert Counter(e["region"] for e in tc) == {"G1a": 200, "G1b": 140, "G1adv": 60}
    for e in tc + [x for x in ev if x["battery"] == "ACQ"]:
        a = np.array(e["cfg"]["agent"])
        b = np.array(e["cfg"]["goal"]["pos"])
        h = np.array(e["cfg"]["hazard"]["pos"])
        r = REGIONS["G1"][e["region"]]
        da, db = float(np.linalg.norm(h - a)), float(np.linalg.norm(h - b))
        assert 12.0 <= db <= 25.0, (e["id"], db)
        assert r["spawn_lo"] <= da <= r["spawn_hi"], (e["id"], da)
        sp = e["cfg"]["hazard"]["speed"]
        assert r["speed_lo"] <= sp <= r["speed_hi"], (e["id"], sp)
        assert e["family"] == "G1"
    assert C.TCONF_SHA == sha256_file(f"{ROOT}/v2conf/train_conflict_v1.json")
    assert C.EVCONF_SHA == sha256_file(f"{ROOT}/v2conf/eval_conf_v1.json")
    for b, ss in [("H1C", 0), ("H2C", 0), ("H3C", 0)]:
        assert len([e for e in ev if e["battery"] == b]) == 200, b
    print("T0: conflict sets valid (600 band-checked, hashes match)")

    # T1. pool composition via cfg-identity instrumentation
    train = json.load(open(f"{ROOT}/data/train_meta_v1.json"))
    pools = {"g1orig": [e for e in train if e["family"] == "G1"],
             "g1conf": tc, "g2": [e for e in train if e["family"] == "G2"]}
    ids = {k: {id(e["cfg"]) for e in v} for k, v in pools.items()}
    seen = []
    real = V.viewed_rollout

    def spy(theta, family, cfg, seed, counter, *a, **k):
        seen.append(next(k for k in ids if id(cfg) in ids[k]))
        return real(theta, family, cfg, seed, counter, *a, **k)

    V.viewed_rollout = spy
    try:
        C.run_gate_phase(np.load(f"{ROOT}/pilot3_runs/P3_301_theta.npz")["theta"],
                         pools, V.vv_rng(301, 1), V.vv_rng(301, 2),
                         V.vv_rng(301, 3), [0], 2)
    finally:
        V.viewed_rollout = real
    per_eval = [seen[i:i + 8] for i in range(0, len(seen), 8)]
    want = ["g1orig"] * 2 + ["g1conf"] * 2 + ["g2"] * 4
    assert len(seen) == 2 * 64 * 8, len(seen)
    assert all(p == want for p in per_eval), per_eval[0]
    print(f"T1: fitness pools exactly 2+2+4 in order ({len(per_eval)} evals)")

    # T2. ablation correctness
    theta = np.load(f"{ROOT}/pilot3_runs/P3_301_theta.npz")["theta"]
    rng = np.random.default_rng(0)
    for _ in range(200):
        o = rng.normal(size=14)
        p1, w1 = V.viewed_blend(theta, o)
        p2, w2 = C.ablated_blend(theta, o)
        assert np.array_equal(w1, w2)
        skills, _ = split_modular(theta)
        from models import SKILL_SHAPES, mlp_forward
        from policies import softmax
        manual = sum(w1[k] * (np.full(9, 1 / 9) if k == 1 else softmax(
            mlp_forward(skills[k], V.view_obs(o, k), SKILL_SHAPES)))
            for k in range(4))
        assert np.allclose(p2, manual, atol=1e-12)
    # ablated differs from normal iff w_B > 0
    o = rng.normal(size=14)
    _, w = V.viewed_blend(theta, o)
    p1, _ = V.viewed_blend(theta, o)
    p2, _ = C.ablated_blend(theta, o)
    assert (not np.array_equal(p1, p2)) == (w[1] > 0)
    print("T2: ablation = normal with ONLY B->uniform (200 obs)")

    # T3. tiny end-to-end
    shutil.rmtree(OUT, ignore_errors=True)
    log = C.run_seed(301, f"{ROOT}/data",
                     f"{ROOT}/pilot3_runs/P3_301_theta.npz",
                     f"{ROOT}/v2conf/train_conflict_v1.json",
                     f"{ROOT}/v2conf/eval_conf_v1.json",
                     f"{ROOT}/v2/g1g2_heldout_v1.json", OUT, 12)
    expect = 12 * 64 * 8 * 200 + 1 * 40 * 200
    assert log["steps"] == expect, (log["steps"], expect)
    assert log["fitness_pools"] == [["g1orig", 2], ["g1conf", 2], ["g2", 4]]
    assert len(log["history"]) == 12 and len(log["checks"]) == 1
    for fn in ["VC_301.json", "VC_301_theta.npz", "VC_301_ckpts.npz"]:
        assert os.path.isfile(f"{OUT}/{fn}"), fn
    z = np.load(f"{OUT}/VC_301_ckpts.npz")
    assert sorted(z.files) == ["init", "post_gate"]
    assert np.array_equal(z["post_gate"][:4*1833], z["init"][:4*1833])
    assert not np.array_equal(z["post_gate"][4*1833:], z["init"][4*1833:])
    evv = log["eval"]
    assert evv["ACQ"]["normal"]["n"] == 200
    assert set(evv["ACQ"]["normal"]) >= {"w_mean", "w_by_band"}
    for m in ["greedy", "ablated-B", "skill-0", "skill-1", "skill-2", "skill-3"]:
        assert evv["ACQ"][m]["n"] == 200, m
    for b in ["H1C", "H2C", "H3C"]:
        for m in ["normal", "greedy", "uniform", "shuffle", "mask",
                  "skill-0", "skill-1", "skill-2", "skill-3"]:
            assert evv[b][m]["n"] == 200, (b, m)
    # weight-eval == run_battery on subset
    sub = [e for e in ev if e["battery"] == "ACQ"][:20]
    th = np.load(f"{OUT}/VC_301_theta.npz")["theta"]
    r1 = C.acq_eval_with_weights(th, sub)
    r2 = run_battery(V.policy_fn_for(th, "normal"), sub)
    assert r1["sr"] == r2["success_rate"] and r1["n"] == 20
    print("T3: 12-gen run ok (steps/files/freeze/eval/weights==run_battery)")

    # T4. no mutation + RNG intact
    o = rng.normal(size=14)
    before = o.copy()
    V.viewed_blend(theta, o)
    C.ablated_blend(theta, o)
    for m in ["normal", "greedy", "uniform", "shuffle", "mask",
              "skill-0", "skill-1", "skill-2", "skill-3", "ablated-B"]:
        C.policy_fn_for(theta, m)(o, rng)
    assert np.array_equal(o, before)
    r1, r2, r3 = V.vv_rng(301, 1), V.vv_rng(301, 2), V.vv_rng(301, 3)
    for r in (r1, r2, r3):
        r.standard_normal(10)
    s0 = [r.bit_generator.state for r in (r1, r2, r3)]
    C.acq_eval_with_weights(theta, sub)
    s1 = [r.bit_generator.state for r in (r1, r2, r3)]
    assert s0 == s1
    print("T4: no obs mutation; RNGs bit-identical around evals")

    # T5. constants
    assert 400 * 64 * 8 * 200 + 40 * 40 * 200 == 41280000
    assert C.VC_SEEDS == [301, 302, 303] and C.POOL_PLAN == (("g1orig", 2), ("g1conf", 2), ("g2", 4))
    assert sum(n for _, n in C.POOL_PLAN) == 8
    for s in (301, 302, 303):
        t = np.load(f"{ROOT}/pilot3_runs/P3_{s}_theta.npz")["theta"]
        assert len(t) == 7944 and np.all(t[4*1833:] == 0), s
    print("T5: 41.28M, seeds, pools 2+2+4, starts")
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
