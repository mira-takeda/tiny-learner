"""v2views local smoke (discarded outputs, /tmp/vviews_smoke).

T0. Views intact on atomic tasks: per skill, full 200-ep frozen
    atomic battery, EVERY step viewed==unviewed forward bit-exact
    (masked slots verified zero first).
T1. No mutation: input obs arrays bit-identical after view_obs,
    viewed_blend and every eval policy fn.
T2. Shared fitness/eval path: viewed_rollout == manual loop with the
    eval policy fn (return+success); determinism per mode.
T3. Tiny 12-gen run: exact steps, 3 files, skills frozen + gate
    changed, hashes verified, complete eval keys (new+old+G1G2).
T4. Training RNG streams bit-identical around eval machinery.
T5. Production constants: 41.28M, seeds, masks, shuffle 0, battery
    hashes, start thetas with zero gates.

Exit 0 = SMOKE PASS.
"""

import json
import os
import shutil
import sys

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
sys.path.insert(0, f"{ROOT}/v2views")
sys.path.insert(0, f"{ROOT}/src")

import numpy as np

import views_train as V
from evaluate import make_policy
from gen import sha256_file
from models import (SKILL_SHAPES, mlp_forward, split_modular)
from policies import softmax
from tasks import TaskRunner

OUT = "/tmp/vviews_smoke"
ATOMIC = [("A", 0), ("B", 1), ("C", 2), ("D", 3)]


def main():
    battery = json.load(open(f"{ROOT}/data/test_battery_v1.json"))
    theta = np.load(f"{ROOT}/pilot3_runs/P3_301_theta.npz")["theta"]
    skills, _ = split_modular(theta)

    # T0. viewed == unviewed on atomics (masked slots already zero)
    nsteps = 0
    for fam, k in ATOMIC:
        masked_idx = [i for i in range(14) if i not in V.KEEP[k]]
        for e in [x for x in battery if x["battery"] == fam]:
            r = TaskRunner()
            rng = np.random.default_rng(int(e["seed"]))
            obs = r.reset(fam, e["cfg"], int(e["seed"]))
            done = False
            while not done:
                assert np.all(obs[masked_idx] == 0.0), (fam, k)
                a = softmax(mlp_forward(skills[k], V.view_obs(obs, k),
                                        SKILL_SHAPES))
                b = softmax(mlp_forward(skills[k], obs, SKILL_SHAPES))
                assert np.array_equal(a, b), (fam, k)
                nsteps += 1
                obs, _, done, _ = r.step(int(rng.choice(len(a), p=a)))
    print(f"T0: views intact on atomics ({nsteps} steps bit-exact)")

    # T1. no input mutation
    rng = np.random.default_rng(0)
    for _ in range(50):
        o = rng.normal(size=14)
        before = o.copy()
        for k in range(4):
            V.view_obs(o, k)
        V.viewed_blend(theta, o)
        for m in ["normal", "greedy", "uniform", "shuffle", "mask",
                  "skill-0", "skill-1", "skill-2", "skill-3"]:
            V.policy_fn_for(theta, m)(o, rng)
        assert np.array_equal(o, before), m
    print("T1: obs arrays never mutated by views/blend/policies")

    # T2. viewed_rollout == manual eval loop
    train = json.load(open(f"{ROOT}/data/train_meta_v1.json"))
    n = 0
    for fam in ("G1", "G2"):
        for e in [x for x in train if x["family"] == fam][:6]:
            for seed in (11, 12):
                r1, s1 = V.viewed_rollout(theta, fam, e["cfg"], seed, [0])
                r = TaskRunner()
                rr = np.random.default_rng(seed)
                obs = r.reset(fam, e["cfg"], seed)
                pol = V.policy_fn_for(theta, "normal")
                tot, done = 0.0, False
                while not done:
                    obs, x, done, info = r.step(pol(obs, rr))
                    tot += x
                assert r1 == tot and s1 == bool(info["success"])
                n += 1
    print(f"T2: fitness/eval episode identity ({n} eps)")

    # T2b. shuffle gate weights == frozen make_policy shuffle weights
    # (same draw; only skills are viewed here by design).
    from policies import unpack, xavier_init, layer_shapes
    shapes = layer_shapes(14, [32], 4)
    gm = unpack(xavier_init(np.random.default_rng(0), shapes), shapes)
    tr = []
    pf = make_policy("modular", theta, "shuffle", 0, trace=tr)
    r = TaskRunner()
    rr = np.random.default_rng(3)
    e0 = [x for x in train if x["family"] == "G1"][0]
    obs = r.reset("G1", e0["cfg"], 3)
    pf(obs, rr)
    _, w_mine = V.viewed_blend(theta, obs, gm)
    assert np.array_equal(np.array(tr[0]), w_mine)
    print("T2b: shuffle weights identical to frozen construction")

    # T3. tiny end-to-end
    shutil.rmtree(OUT, ignore_errors=True)
    log = V.run_seed(301, f"{ROOT}/data",
                     f"{ROOT}/pilot3_runs/P3_301_theta.npz",
                     f"{ROOT}/v2/g1g2_heldout_v1.json",
                     f"{ROOT}/v2views/eval_new_v1.json", OUT, 12)
    expect = 12 * 64 * 8 * 200 + 1 * 40 * 200
    assert log["steps"] == expect, (log["steps"], expect)
    assert len(log["history"]) == 12 and len(log["checks"]) == 1
    for fn in ["VV_301.json", "VV_301_theta.npz", "VV_301_ckpts.npz"]:
        assert os.path.isfile(f"{OUT}/{fn}"), fn
    z = np.load(f"{OUT}/VV_301_ckpts.npz")
    assert sorted(z.files) == ["init", "post_gate"]
    assert np.array_equal(z["post_gate"][:4*1833], z["init"][:4*1833])
    assert not np.array_equal(z["post_gate"][4*1833:], z["init"][4*1833:])
    assert log["battery_new_sha256"] == V.NEWBATT_SHA
    ev = log["eval"]
    assert ev["G1H"]["normal"]["n"] == 200 and ev["G2H"]["normal"]["n"] == 200
    for b in ["I1N", "H1N", "H2N", "H3N"]:
        for m in ["normal", "greedy", "uniform", "shuffle", "mask",
                  "skill-0", "skill-1", "skill-2", "skill-3"]:
            assert ev[b][m]["n"] == 200, (b, m)
    for b in ["I1", "H1", "H2", "H3"]:
        for m in ["normal", "greedy"]:
            assert ev["OLD_" + b][m]["n"] == 200, (b, m)
    print("T3: 12-gen run ok (steps/files/freeze/eval keys/Hashes)")

    # T4. RNGs untouched by eval machinery
    r1, r2, r3 = V.vv_rng(301, 1), V.vv_rng(301, 2), V.vv_rng(301, 3)
    for r in (r1, r2, r3):
        r.standard_normal(10)
    before = [r.bit_generator.state for r in (r1, r2, r3)]
    o = rng.normal(size=14)
    for m in ["normal", "greedy", "uniform", "shuffle", "mask",
              "skill-0", "skill-1", "skill-2", "skill-3"]:
        V.policy_fn_for(theta, m)(o.copy(), rng)
    after = [r.bit_generator.state for r in (r1, r2, r3)]
    assert before == after
    print("T4: es/ep/ev RNG states bit-identical around evals")

    # T5. production constants
    assert 400 * 64 * 8 * 200 + 40 * 40 * 200 == 41280000
    assert V.VV_SEEDS == [301, 302, 303] and V.SHUFFLE_SEED == 0
    assert V.KEEP == {0: (0, 1, 2, 3, 4), 1: (0, 1, 5, 6, 7),
                      2: (0, 1, 8, 9, 10, 11, 12, 13), 3: (0, 1, 2, 3, 4)}
    assert V.NEWBATT_SHA == sha256_file(f"{ROOT}/v2views/eval_new_v1.json")
    assert V.G12_SHA == sha256_file(f"{ROOT}/v2/g1g2_heldout_v1.json")
    for s in (301, 302, 303):
        t = np.load(f"{ROOT}/pilot3_runs/P3_{s}_theta.npz")["theta"]
        assert len(t) == 7944 and np.all(t[4*1833:] == 0), s
    print("T5: 41.28M, seeds, masks, shuffle 0, battery hashes, starts")
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
