"""Toy-only self-tests. NEVER loads registered train/test configs.

Covers: param counts/matching, pack roundtrip, gate math, mask invariance,
TaskRunner on hand-made toy configs, ES plumbing on LineWorld, a 2-generation
toy skill-phase, budget arithmetic, seed/curriculum maps, eval determinism.
Exit 0 = all pass.
"""

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from env import LineWorld
from es import train_es
from models import (MODULAR_DIM, MONO_DIM, SKILL_DIM, GATE_DIM, N_SKILLS,
                    SKILL_SHAPES, GATE_SHAPES, MONO_SHAPES,
                    verify_param_match, split_modular, modular_probs,
                    mono_probs)
from policies import xavier_init, count_params
from tasks import TaskRunner
from train import SEEDS, ARM_MANIFEST, POP_PAIRS, FIT_EPS, GEN_BUDGET

PASS = []


def check(name, cond):
    PASS.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


TOY_CFGS = {
    "A": {"agent": [50, 50], "goal": {"pos": [70, 60]}},
    "B": {"agent": [50, 50], "hazard": {"pos": [20, 20], "speed": 3.5}},
    "C": {"agent": [30, 30], "obj": {"pos": [40, 40]},
          "zone": {"center": [50, 92]}},
    "D": {"agent": [50, 20], "goal": {"pos": [50, 65]},
          "wall": {"y": 50, "gap_x": 50, "gap_w": 16}},
    "G1": {"agent": [50, 50], "goal": {"pos": [70, 60]},
           "hazard": {"pos": [20, 20], "speed": 3.2}},
    "G2": {"agent": [30, 30], "obj": {"pos": [40, 40]},
           "zone": {"center": [50, 92]}},
    "H1": {"agent": [30, 30], "obj": {"pos": [40, 40]},
           "zone": {"center": [50, 92]},
           "hazard": {"pos": [80, 80], "speed": 4.5}},
}


def main():
    rng = np.random.default_rng(0)
    m = verify_param_match()
    print(f"modular={m['modular']} monolith={m['monolith']} rel={m['rel_diff']:.4f}")
    check("skill-dim-1833", SKILL_DIM == 1833)
    check("gate-dim-612", GATE_DIM == 612)
    check("modular-dim-7944", MODULAR_DIM == 7944)
    check("mono-dim-8043", MONO_DIM == 8043)

    theta = np.concatenate(
        [xavier_init(rng, SKILL_SHAPES) for _ in range(N_SKILLS)]
        + [xavier_init(rng, GATE_SHAPES)])
    skills, gate = split_modular(theta)
    check("pack-roundtrip",
          np.array_equal(np.concatenate(skills + [gate]), theta))

    from policies import mlp_forward, softmax
    obs = rng.normal(size=14)
    p, w = modular_probs(theta, obs)
    check("blend-sums-to-1", abs(p.sum() - 1.0) < 1e-9 and abs(w.sum() - 1.0) < 1e-9)
    p0, w0 = modular_probs(theta, obs, skill_only=0)
    solo0 = softmax(mlp_forward(skills[0], obs, SKILL_SHAPES))
    check("skill-only-is-onehot",
          bool(np.array_equal(w0, [1.0, 0.0, 0.0, 0.0])
               and np.allclose(p0, solo0)))
    pm = mono_probs(xavier_init(rng, MONO_SHAPES), obs)
    check("mono-sums-to-1", abs(pm.sum() - 1.0) < 1e-9)

    o1, o2 = obs.copy(), obs.copy()
    o1[5:8] = rng.normal(size=3)
    o2[5:8] = rng.normal(size=3)
    pa, _ = modular_probs(theta, o1, mask_hazard=True)
    pb, _ = modular_probs(theta, o2, mask_hazard=True)
    check("mask-invariance", np.array_equal(pa, pb))

    for fam, cfg in TOY_CFGS.items():
        tr = TaskRunner()
        o = tr.reset(fam, cfg, 123)
        assert o.shape == (14,) and np.all(np.isfinite(o))
        r2 = np.random.default_rng(9)
        tot, done, n = 0.0, False, 0
        while not done and n < 250:
            o, r, done, info = tr.step(int(r2.integers(9)))
            tot += r
            n += 1
        check(f"toy-{fam}-terminates-finite",
              done and np.isfinite(tot) and isinstance(info["success"], bool))

    # determinism: same theta + seed -> identical episode
    def run_once():
        tr = TaskRunner()
        r2 = np.random.default_rng(5)
        o = tr.reset("A", TOY_CFGS["A"], 5)
        seq = []
        done = False
        while not done:
            a = int(r2.integers(9))
            o, r, done, info = tr.step(a)
            seq.append((float(r), o.tobytes()))
        return seq, info["success"]

    s1, ok1 = run_once()
    s2, ok2 = run_once()
    check("eval-deterministic", ok1 == ok2 and [x[0] for x in s1] == [x[0] for x in s2])

    # ES plumbing on LineWorld
    from policies import layer_shapes
    sh = layer_shapes(2, [8, 8], 3)
    t0 = xavier_init(np.random.default_rng(1), sh)

    def fit(th):
        from policies import act_sample
        env = LineWorld()
        tot = 0.0
        for s in range(2):
            r2 = np.random.default_rng(s)
            o = env.reset()
            done = False
            while not done:
                o, r, done, _ = env.step(act_sample(r2, th, o, sh))
                tot += r
        return tot / 2

    res = train_es(t0, fit, np.random.default_rng(2), pop_pairs=4,
                   generations=3)
    check("es-plumbing", res["generations"] == 3 and len(res["history"]) == 3)

    # toy modular-phase plumbing: 2 gens, tiny pop, synthetic in-memory entries
    toy_entries = [{"family": "A", "cfg": TOY_CFGS["A"]} for _ in range(4)]
    base = np.concatenate(
        [xavier_init(np.random.default_rng(3), SKILL_SHAPES)
         for _ in range(N_SKILLS)]
        + [np.zeros(GATE_DIM)])
    sub = slice(0, SKILL_DIM)
    frng = np.random.default_rng(5)

    def toy_fit(sub_th):
        full = base.copy()
        full[sub] = sub_th
        rets = []
        for _ in range(2):
            e = toy_entries[int(frng.integers(len(toy_entries)))]
            tr = TaskRunner()
            rr = np.random.default_rng(int(frng.integers(2**31)))
            o = tr.reset("A", e["cfg"], 11)
            tot, done = 0.0, False
            while not done:
                pp, _ = modular_probs(full, o)
                o, r, done, _ = tr.step(int(rr.choice(len(pp), p=pp)))
                tot += r
            rets.append(tot)
        return float(np.mean(rets))

    res = train_es(base[sub].copy(), toy_fit, np.random.default_rng(4),
                   pop_pairs=4, generations=2)
    check("toy-phase-plumbing",
          res["generations"] == 2 and len(res["history"]) == 2)

    # budget arithmetic + maps (pure, no data files)
    cap = GEN_BUDGET * (2 * POP_PAIRS) * FIT_EPS * 200
    check("budget-fixed-41M-max", cap == 400 * 64 * 8 * 200)
    check("seeds-frozen", SEEDS == [101, 102, 103, 104, 105])
    check("curriculum-map", ARM_MANIFEST == {"A1": "train_meta_v1.json",
                                            "A2": "train_meta_v1.json",
                                            "A3": "train_meta_v1.json",
                                            "A4": "train_control_v1.json"})

    # bound nets are bit-identical to unbound forward (speed-only change)
    from models import ModularNet, MonoNet
    from models import mono_probs as _mp
    bt = np.concatenate(
        [xavier_init(np.random.default_rng(11), SKILL_SHAPES)
         for _ in range(N_SKILLS)]
        + [xavier_init(np.random.default_rng(11), GATE_SHAPES)])
    mt = xavier_init(np.random.default_rng(12), MONO_SHAPES)
    bo = np.random.default_rng(13).normal(size=(5, 14))
    bmod, bmono = ModularNet(bt), MonoNet(mt)
    same = all(np.array_equal(bmod.probs(o)[0], modular_probs(bt, o)[0])
               and np.array_equal(bmod.probs(o)[1], modular_probs(bt, o)[1])
               for o in bo)
    same &= all(np.array_equal(bmono.probs(o), _mp(mt, o)) for o in bo)
    check("bound-bit-identical", same)

    # fixed budget: passing checks NEVER shorten a comparative phase
    from train import run_es_phase
    ft = np.concatenate(
        [xavier_init(np.random.default_rng(14), SKILL_SHAPES)
         for _ in range(N_SKILLS)]
        + [np.zeros(GATE_DIM)])
    fentries = {"A": [{"family": "A", "cfg": TOY_CFGS["A"]} for _ in range(4)]}
    _, fgens, _, fchecks = run_es_phase(
        ft, slice(0, SKILL_DIM), "modular", [("A", 2)], fentries,
        np.random.default_rng(15), np.random.default_rng(16), [0],
        check_fn=lambda full: 1.0, generations=25)
    check("fixed-budget-ignores-passing-checks",
          fgens == 25 and fchecks == [1.0, 1.0])

    # absorbing padding: early termination cannot alter budget or outcome
    from train import rollout, HORIZON, PHASE_STEPS
    adj = {"agent": [50, 50], "goal": {"pos": [52, 50]}}  # arrival on step 1
    pth = np.concatenate(
        [xavier_init(np.random.default_rng(21), SKILL_SHAPES)
         for _ in range(N_SKILLS)]
        + [xavier_init(np.random.default_rng(21), GATE_SHAPES)])
    ok = True
    for es in (3, 17, 99):
        c = [0]
        ret, suc = rollout(pth, "modular", "A", adj, es, c)
        tr = TaskRunner()
        rr = np.random.default_rng(es)
        o = tr.reset("A", adj, es)
        net = ModularNet(pth)
        tot, done, n = 0.0, False, 0
        while not done:
            pp, _ = net.probs(o)
            o, r, done, info = tr.step(int(rr.choice(len(pp), p=pp)))
            tot += r
            n += 1
        ok &= (n == 1 and tot == ret and info["success"] == suc
               and suc and c[0] == HORIZON == 200)
    check("padded-exact-200-early-term", bool(ok))
    check("phase-steps-constant",
          PHASE_STEPS == 400 * 64 * 8 * 200 + 40 * 40 * 200 == 41280000)

    print("SELFTEST", "PASS" if all(PASS) else "FAIL")
    return 0 if all(PASS) else 1


if __name__ == "__main__":
    sys.exit(main())
