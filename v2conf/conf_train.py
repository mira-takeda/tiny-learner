"""v2conf gate trainer: acquisition under conflict (v2conf-only).

Paired with v2views (seeds 301-303, same streams/init/curriculum-base),
ONE factor changed: fitness includes G1-conflict episodes (beacon-haz
12-25 band). Per generation: 2xG1-orig + 2xG1-conflict + 4xG2 (fixed
order, 8 eps: budget identical). Skills frozen, v2views views, gate
from zeros, ES v1, 400 gens. Checks record-only identical to v2
(20 G1-orig + 20 G2; conflict acquisition measured post-hoc).

Evals (final theta only): ACQ (normal, greedy, B-ablation, single
skills, gate-weight logging), H1C/H2C/H3C (verdict set), G1H/G2H
(informative). B-ablation: ONLY B's action distribution -> uniform;
gate weights and other skills intact.
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "v2views"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

import views_train as V
from evaluate import run_battery
from gen import sha256_file
from models import (MODULAR_DIM, SKILL_DIM, N_SKILLS, N_ACTIONS,
                    SKILL_SHAPES, split_modular, mlp_forward)
from policies import softmax
from tasks import TaskRunner
from train import (GEN_BUDGET, POP_PAIRS, SIGMA, ALPHA, HORIZON, CHECK_EPS,
                   GATE_SLICE)

PREFIX = "VC_"
VC_SEEDS = [301, 302, 303]
SHUFFLE_SEED = 0
TCONF_SHA = ("37c2524a2afbef40b67695032564fcbc8d1e5633f47d573a98e627"
             "70d27932e2")
EVCONF_SHA = ("feba0472c4d217b865e8c42ae81f2fab48f9d747819dba5faca2ed"
              "a98185c553")
G12_SHA = V.G12_SHA
# Fitness pool composition per eval, FIXED ORDER (protocol).
POOL_PLAN = (("g1orig", 2), ("g1conf", 2), ("g2", 4))


def ablated_blend(theta, obs):
    """Viewed blend with ONLY B's distribution replaced by uniform.
    Gate weights and other skills identical to viewed_blend."""
    skills, gate = split_modular(theta)
    w = softmax(mlp_forward(gate, obs, V.GATE_SHAPES))
    p = np.zeros(N_ACTIONS)
    for k in range(N_SKILLS):
        q = (np.full(N_ACTIONS, 1.0 / N_ACTIONS) if k == 1
             else softmax(mlp_forward(skills[k], V.view_obs(obs, k),
                                      SKILL_SHAPES)))
        p += w[k] * q
    return p, w


def policy_fn_for(theta, mode):
    if mode == "ablated-B":
        def policy_fn(obs, rng):
            p, _ = ablated_blend(theta, obs.copy())
            return int(rng.choice(len(p), p=p))

        return policy_fn
    return V.policy_fn_for(theta, mode)


def run_gate_phase(theta0_full, pools, rng_es, rng_ep, rng_ev, counter,
                   generations):
    base = theta0_full.copy()
    sub = GATE_SLICE

    def fitness(sub_theta):
        full = base.copy()
        full[sub] = sub_theta
        rets = []
        for key, n in POOL_PLAN:
            entries = pools[key]
            for _ in range(n):
                e = entries[int(rng_ep.integers(len(entries)))]
                ret, _ = V.viewed_rollout(full, e["family"], e["cfg"],
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
            for key in ("g1orig", "g2"):
                entries = pools[key]
                ok = 0
                for _ in range(20):
                    e = entries[int(rng_ev.integers(len(entries)))]
                    _, s = V.viewed_rollout(
                        full, e["family"], e["cfg"],
                        int(rng_ev.integers(2 ** 31)), counter)
                    ok += s
                rates.append(ok / 20)
            checks.append(sum(rates) / 2)
        return False

    from es import train_es
    res = train_es(theta_sub, fitness, rng_es, pop_pairs=POP_PAIRS,
                   sigma=SIGMA, alpha=ALPHA, generations=generations,
                   progress_cb=cb)
    full = base.copy()
    full[sub] = res["theta"]
    return full, res["generations"], history, checks


def summarize(rep):
    return {"n": rep["n"], "sr": rep["success_rate"], "ci": rep["ci95"]}


def acq_eval_with_weights(theta, pool):
    """ACQ normal eval mirroring run_battery exactly + gate-weight log
    (mean and by hazard-distance band). Same rng/loop/trajectory."""
    ok, ws, bands = 0, [], {"far>30": [], "mid10-30": [], "near<10": []}
    for e in pool:
        runner = TaskRunner()
        rng = np.random.default_rng(int(e["seed"]))
        obs = runner.reset(e["family"], e["cfg"], int(e["seed"]))
        done = False
        while not done:
            p, w = V.viewed_blend(theta, obs)
            ws.append(w)
            h = float(np.linalg.norm(runner.env.haz_pos - runner.env.agent_pos))
            k = "far>30" if h > 30 else ("mid10-30" if h >= 10 else "near<10")
            bands[k].append(w)
            a = int(rng.choice(len(p), p=p))
            obs, _, done, info = runner.step(a)
        ok += bool(info["success"])
    n = len(pool)
    sr = ok / n
    ci = 1.96 * float(np.sqrt(sr * (1 - sr) / n)) if 0 < sr < 1 else 0.0
    W = np.mean(ws, axis=0)
    out = {"n": n, "sr": sr, "ci": ci,
           "w_mean": [round(float(x), 4) for x in W], "w_by_band": {}}
    for k, v in bands.items():
        m = np.mean(v, axis=0) if v else np.zeros(4)
        out["w_by_band"][k] = {"n_steps": len(v),
                               "w": [round(float(x), 4) for x in m]}
    return out


def run_seed(seed, data_dir, start_path, tconf_path, evconf_path, g12_path,
             out_dir, generations=GEN_BUDGET):
    assert seed in VC_SEEDS
    manifest = json.load(open(f"{data_dir}/MANIFEST.json"))
    for fn, h in manifest.items():
        assert sha256_file(f"{data_dir}/{fn}") == h, f"dataset altered: {fn}"
    assert sha256_file(tconf_path) == TCONF_SHA, "conflict train altered"
    assert sha256_file(evconf_path) == EVCONF_SHA, "eval conf altered"
    assert sha256_file(g12_path) == G12_SHA, "g1g2 heldout altered"
    train = json.load(open(f"{data_dir}/train_meta_v1.json"))
    pools = {"g1orig": [e for e in train if e["family"] == "G1"],
             "g1conf": json.load(open(tconf_path)),
             "g2": [e for e in train if e["family"] == "G2"]}
    assert len(pools["g1orig"]) == 400 and len(pools["g2"]) == 400
    assert len(pools["g1conf"]) == 400
    evc = json.load(open(evconf_path))
    acq = [e for e in evc if e["battery"] == "ACQ"]
    assert len(acq) == 200
    gh = json.load(open(g12_path))

    theta0 = np.load(start_path)["theta"]
    assert len(theta0) == MODULAR_DIM
    assert np.all(theta0[GATE_SLICE] == 0), "gate must start at zeros"
    start_sha = sha256_file(start_path)

    counter = [0]
    t0 = time.time()
    theta, gens, hist, checks = run_gate_phase(
        theta0, pools, V.vv_rng(seed, 1), V.vv_rng(seed, 2),
        V.vv_rng(seed, 3), counter, generations)
    wall = time.time() - t0
    expect = (generations * (2 * POP_PAIRS) * 8 * HORIZON
              + (generations // 10) * CHECK_EPS * HORIZON)
    assert counter[0] == expect, (counter[0], expect)
    assert np.array_equal(theta[:N_SKILLS * SKILL_DIM],
                          theta0[:N_SKILLS * SKILL_DIM]), "freeze violated"

    ev = {"ACQ": {"normal": acq_eval_with_weights(theta, acq)}}
    ev["ACQ"]["greedy"] = summarize(run_battery(policy_fn_for(theta, "greedy"), acq))
    ev["ACQ"]["ablated-B"] = summarize(run_battery(policy_fn_for(theta, "ablated-B"), acq))
    for k in range(N_SKILLS):
        ev["ACQ"][f"skill-{k}"] = summarize(run_battery(
            policy_fn_for(theta, f"skill-{k}"), acq))
    for b in ["H1C", "H2C", "H3C"]:
        pool = [e for e in evc if e["battery"] == b]
        assert len(pool) == 200, b
        ev[b] = {m: summarize(run_battery(policy_fn_for(theta, m), pool))
                 for m in ["normal", "greedy", "uniform", "shuffle", "mask",
                           "skill-0", "skill-1", "skill-2", "skill-3"]}
    for b, tag in (("G1H", "G1H"), ("G2H", "G2H")):
        pool = [e for e in gh if e["battery"] == tag]
        ev[b] = {"normal": summarize(run_battery(policy_fn_for(theta, "normal"), pool))}

    log = {"seed": seed, "start": os.path.basename(start_path),
           "start_sha256": start_sha,
           "design": "v2conf: gate training with conflict curriculum + views",
           "fitness_pools": [["g1orig", 2], ["g1conf", 2], ["g2", 4]],
           "train_conflict_sha256": TCONF_SHA,
           "eval_conf_sha256": EVCONF_SHA,
           "g1g2_heldout_sha256": G12_SHA, "shuffle_seed": SHUFFLE_SEED,
           "es": {"pop_pairs": POP_PAIRS, "sigma": SIGMA, "alpha": ALPHA,
                  "fit": "2xG1+2xG1conf+4xG2", "gens": gens},
           "history": hist, "checks": checks, "steps": counter[0],
           "wall_s": wall, "eval": ev,
           "code_sha256": sha256_file(os.path.join(HERE, "conf_train.py"))}
    os.makedirs(out_dir, exist_ok=True)
    np.savez(f"{out_dir}/{PREFIX}{seed}_theta.npz", theta=theta)
    np.savez(f"{out_dir}/{PREFIX}{seed}_ckpts.npz", init=theta0,
             post_gate=theta.copy())
    json.dump(log, open(f"{out_dir}/{PREFIX}{seed}.json", "w"), indent=1,
              default=float)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, choices=VC_SEEDS, required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--start", default="")
    ap.add_argument("--tconf", default="v2conf/train_conflict_v1.json")
    ap.add_argument("--evconf", default="v2conf/eval_conf_v1.json")
    ap.add_argument("--g12", default="v2/g1g2_heldout_v1.json")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--gens", type=int, default=GEN_BUDGET)
    args = ap.parse_args()
    start = args.start or f"pilot3_runs/P3_{args.seed}_theta.npz"
    log = run_seed(args.seed, args.data, start, args.tconf, args.evconf,
                   args.g12, args.out, args.gens)
    print(json.dumps({k: v for k, v in log.items()
                      if k not in ("history", "checks", "eval")}, indent=1,
                     default=float))


if __name__ == "__main__":
    main()
