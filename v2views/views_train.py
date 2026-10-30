"""v2views gate trainer (v2views-only; v2/pilots/src untouched).

Paired with v2 (seeds 301-303, same streams/init/curriculum/budget),
ONE factor changed: per-skill observation views. Each skill sees its
training slots + velocity, rest zeros (training absence encoding);
gate sees the FULL 14-dim obs. Views apply at EVERY skill forward
(fitness, checks, eval, blend controls) via ONE shared code path.
v2-original behavior is NOT reproduced here (frozen in v2_runs/).

Masks (0-based indices kept): A [0:5]; B [0:2]+[5:8]; C [0:2]+[8:14];
D [0:2]+[2:5]. Gate: full obs. Views NEVER mutate the input array.
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

from es import train_es
from evaluate import run_battery
from gen import sha256_file
from models import (MODULAR_DIM, SKILL_DIM, GATE_DIM, N_SKILLS, N_ACTIONS,
                    SKILL_SHAPES, GATE_SHAPES, split_modular, mlp_forward,
                    _forward_mats)
from policies import softmax, xavier_init, unpack, layer_shapes
from tasks import TaskRunner
from train import (GEN_BUDGET, POP_PAIRS, SIGMA, ALPHA, HORIZON, CHECK_EPS,
                   GATE_SLICE)

PREFIX = "VV_"
VV_SEEDS = [301, 302, 303]
SHUFFLE_SEED = 0
G12_SHA = ("f684fc34561835830eaf0ba000a75f074e3a4d8289b39c651d48c7"
           "3f6726c997")
NEWBATT_SHA = ("2fd9a68c0d26f884a23eea2aad8e77a6cd27ec1d71cc00bfa41b0"
               "63c8cce51e9")
NEW_BATTERIES = ["I1N", "H1N", "H2N", "H3N"]
OLD_BATTERIES = ["I1", "H1", "H2", "H3"]

# Exact per-index keep masks (protocol §1). Built as new arrays; the
# input obs is never mutated (asserted in smoke T1).
KEEP = {0: (0, 1, 2, 3, 4),
        1: (0, 1, 5, 6, 7),
        2: (0, 1, 8, 9, 10, 11, 12, 13),
        3: (0, 1, 2, 3, 4)}


def view_obs(obs, k):
    out = np.zeros(14)
    out[list(KEEP[k])] = obs[list(KEEP[k])]
    return out


def viewed_blend(theta, obs, gate_mats=None):
    """ONE shared path for fitness and eval. Gate on FULL obs; skills
    on their views. Returns (p, w)."""
    skills, gate = split_modular(theta)
    # Flat gate slice -> mlp_forward; unpacked mats (shuffle control) ->
    # _forward_mats, mirroring frozen ModularNet.probs exactly.
    if gate_mats is None:
        w = softmax(mlp_forward(gate, obs, GATE_SHAPES))
    else:
        w = softmax(_forward_mats(gate_mats, obs))
    p = np.zeros(N_ACTIONS)
    for k in range(N_SKILLS):
        if w[k] > 0:
            q = softmax(mlp_forward(skills[k], view_obs(obs, k),
                                    SKILL_SHAPES))
            p += w[k] * q
    return p, w


def viewed_rollout(theta, family, cfg, ep_seed, counter, gate_mats=None,
                   greedy=False):
    """Frozen rollout() mechanics with the viewed blend (same loop,
    rng use, counter, asserts; only p construction differs)."""
    runner = TaskRunner()
    rng = np.random.default_rng(ep_seed)
    obs = runner.reset(family, cfg, int(ep_seed))
    total, done, steps = 0.0, False, 0
    while not done:
        p, _ = viewed_blend(theta, obs, gate_mats)
        a = int(np.argmax(p)) if greedy else int(rng.choice(len(p), p=p))
        obs, r, done, info = runner.step(a)
        total += r
        steps += 1
    assert steps <= HORIZON, (steps, HORIZON)
    counter[0] += HORIZON
    return total, bool(info["success"])


def policy_fn_for(theta, mode):
    """Eval policies sharing viewed_blend (fitness/eval identity)."""
    gm = None
    if mode == "shuffle":
        shapes = layer_shapes(14, [32], N_SKILLS)
        gm = unpack(xavier_init(np.random.default_rng(SHUFFLE_SEED),
                                shapes), shapes)
    greedy = mode == "greedy"
    skill_only = int(mode.split("-")[1]) if mode.startswith("skill-") else None

    def policy_fn(obs, rng):
        base = obs.copy()
        if mode == "mask":
            base[5:8] = 0.0
        if skill_only is not None:
            skills, _ = split_modular(theta)
            p = softmax(mlp_forward(skills[skill_only],
                                    view_obs(base, skill_only),
                                    SKILL_SHAPES))
        elif mode == "uniform":
            skills, _ = split_modular(theta)
            p = np.zeros(N_ACTIONS)
            for k in range(N_SKILLS):
                p += 0.25 * softmax(mlp_forward(
                    skills[k], view_obs(base, k), SKILL_SHAPES))
        elif mode == "shuffle":
            p, _ = viewed_blend(theta, base, gm)
        else:
            p, _ = viewed_blend(theta, base)
        return int(np.argmax(p)) if greedy else int(rng.choice(len(p), p=p))

    return policy_fn


def vv_rng(seed, purpose):
    return np.random.default_rng([seed, 4, purpose])


def run_gate_phase(theta0_full, entries_g1, entries_g2, rng_es, rng_ep,
                   rng_ev, counter, generations):
    base = theta0_full.copy()
    sub = GATE_SLICE

    def fitness(sub_theta):
        full = base.copy()
        full[sub] = sub_theta
        rets = []
        for entries in (entries_g1, entries_g2):
            for _ in range(4):
                e = entries[int(rng_ep.integers(len(entries)))]
                ret, _ = viewed_rollout(full, e["family"], e["cfg"],
                                        int(rng_ep.integers(2 ** 31)),
                                        counter)
                rets.append(ret)
        return float(np.mean(rets))

    theta_sub = base[sub].copy()
    history, checks = [], []

    def cb(g, th, best):
        history.append(best)
        if (g + 1) % 10 == 0:
            full = base.copy()
            full[sub] = th
            rates = []
            for entries in (entries_g1, entries_g2):
                ok = 0
                for _ in range(20):
                    e = entries[int(rng_ev.integers(len(entries)))]
                    _, s = viewed_rollout(
                        full, e["family"], e["cfg"],
                        int(rng_ev.integers(2 ** 31)), counter)
                    ok += s
                rates.append(ok / 20)
            checks.append(sum(rates) / 2)
        return False

    res = train_es(theta_sub, fitness, rng_es, pop_pairs=POP_PAIRS,
                   sigma=SIGMA, alpha=ALPHA, generations=generations,
                   progress_cb=cb)
    full = base.copy()
    full[sub] = res["theta"]
    return full, res["generations"], history, checks


def summarize(rep):
    return {"n": rep["n"], "sr": rep["success_rate"], "ci": rep["ci95"]}


def run_seed(seed, data_dir, start_path, g12_path, newbatt_path, out_dir,
             generations=GEN_BUDGET):
    assert seed in VV_SEEDS
    manifest = json.load(open(f"{data_dir}/MANIFEST.json"))
    for fn, h in manifest.items():
        assert sha256_file(f"{data_dir}/{fn}") == h, f"dataset altered: {fn}"
    assert sha256_file(g12_path) == G12_SHA, "g1g2 heldout altered"
    assert sha256_file(newbatt_path) == NEWBATT_SHA, "new battery altered"
    train = json.load(open(f"{data_dir}/train_meta_v1.json"))
    g1tr = [e for e in train if e["family"] == "G1"]
    g2tr = [e for e in train if e["family"] == "G2"]
    battery = json.load(open(f"{data_dir}/test_battery_v1.json"))
    newb = json.load(open(newbatt_path))
    gh = json.load(open(g12_path))

    theta0 = np.load(start_path)["theta"]
    assert len(theta0) == MODULAR_DIM
    assert np.all(theta0[GATE_SLICE] == 0), "gate must start at zeros"
    start_sha = sha256_file(start_path)

    counter = [0]
    t0 = time.time()
    theta, gens, hist, checks = run_gate_phase(
        theta0, g1tr, g2tr, vv_rng(seed, 1), vv_rng(seed, 2),
        vv_rng(seed, 3), counter, generations)
    wall = time.time() - t0
    expect = (generations * (2 * POP_PAIRS) * 8 * HORIZON
              + (generations // 10) * CHECK_EPS * HORIZON)
    assert counter[0] == expect, (counter[0], expect)
    assert np.array_equal(theta[:N_SKILLS * SKILL_DIM],
                          theta0[:N_SKILLS * SKILL_DIM]), "freeze violated"

    ev = {}
    for batt, pool, modes in (
            ("G1H", [e for e in gh if e["battery"] == "G1H"], ["normal"]),
            ("G2H", [e for e in gh if e["battery"] == "G2H"], ["normal"])):
        assert len(pool) == 200
        ev[batt] = {m: summarize(run_battery(policy_fn_for(theta, m), pool))
                    for m in modes}
    for b in NEW_BATTERIES:
        pool = [e for e in newb if e["battery"] == b]
        assert len(pool) == 200, b
        ev[b] = {m: summarize(run_battery(policy_fn_for(theta, m), pool))
                 for m in ["normal", "greedy", "uniform", "shuffle", "mask",
                           "skill-0", "skill-1", "skill-2", "skill-3"]}
    for b in OLD_BATTERIES:
        pool = [e for e in battery if e["battery"] == b]
        ev["OLD_" + b] = {
            m: summarize(run_battery(policy_fn_for(theta, m), pool))
            for m in ["normal", "greedy"]}

    log = {"seed": seed, "start": os.path.basename(start_path),
           "start_sha256": start_sha,
           "design": "v2views: gate training with per-skill views",
           "views": {str(k): list(v) for k, v in KEEP.items()},
           "train_entries": "train_meta_v1.json",
           "battery_new": os.path.basename(newbatt_path),
           "battery_new_sha256": NEWBATT_SHA,
           "g1g2_heldout_sha256": G12_SHA, "shuffle_seed": SHUFFLE_SEED,
           "es": {"pop_pairs": POP_PAIRS, "sigma": SIGMA, "alpha": ALPHA,
                  "fit": "4xG1+4xG2", "gens": gens},
           "history": hist, "checks": checks, "steps": counter[0],
           "wall_s": wall, "eval": ev,
           "code_sha256": sha256_file(os.path.join(HERE, "views_train.py"))}
    os.makedirs(out_dir, exist_ok=True)
    np.savez(f"{out_dir}/{PREFIX}{seed}_theta.npz", theta=theta)
    np.savez(f"{out_dir}/{PREFIX}{seed}_ckpts.npz", init=theta0,
             post_gate=theta.copy())
    json.dump(log, open(f"{out_dir}/{PREFIX}{seed}.json", "w"), indent=1,
              default=float)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, choices=VV_SEEDS, required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--start", default="")
    ap.add_argument("--g12", default="v2/g1g2_heldout_v1.json")
    ap.add_argument("--newbatt", default="v2views/eval_new_v1.json")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--gens", type=int, default=GEN_BUDGET)
    args = ap.parse_args()
    start = args.start or f"pilot3_runs/P3_{args.seed}_theta.npz"
    log = run_seed(args.seed, args.data, start, args.g12, args.newbatt,
                   args.out, args.gens)
    print(json.dumps({k: v for k, v in log.items()
                      if k not in ("history", "checks", "eval")}, indent=1,
                     default=float))


if __name__ == "__main__":
    main()
