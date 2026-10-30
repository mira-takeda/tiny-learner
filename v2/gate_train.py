"""v2 gate trainer (v2-only; pilot/pilot2/pilot3/src untouched).

Compositional pilot: start from pilot3 post_D thetas (seeds 301-303,
no selection), skills FROZEN bit-exact, train ONLY the gate slice
(612 params) with v1 gate mechanics: ES, fitness = mean over 4xG1 +
4xG2 train episodes (frozen train_meta_v1), 400 gens, checks
record-only mean(20 G1 + 20 G2). Fresh streams [seed,4,purpose].

Post-training eval (final theta ONLY, no checkpoint selection, no
tuning): frozen G1/G2-heldout (gate acquisition, informative),
I1/H1/H2/H3 (primary sampled + greedy secondary), controls (uniform,
shuffle seed 0, single skills, hazard mask). All deterministic.
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
from evaluate import run_battery, make_policy
from gen import sha256_file
from models import (MODULAR_DIM, SKILL_DIM, GATE_DIM, N_SKILLS,
                    modular_probs)
from tasks import TaskRunner
from train import (GEN_BUDGET, POP_PAIRS, SIGMA, ALPHA, HORIZON, CHECK_EPS,
                   PHASE_STEPS, GATE_SLICE, rollout, train_eval_rate)

PREFIX = "V2_"
V2_SEEDS = [301, 302, 303]
SHUFFLE_SEED = 0  # frozen control draw
HELDOUT_SHA = ("f684fc34561835830eaf0ba000a75f074e3a4d8289b39c651d48c7"
               "3f6726c997")
EVAL_BATTERIES = ["I1", "H1", "H2", "H3"]


def v2_rng(seed, purpose):
    return np.random.default_rng([seed, 4, purpose])


def uniform_policy(theta):
    """Full-gate blend with zeroed gate (= uniform 1/4 mix), sampled."""
    z = np.zeros(GATE_DIM)

    def policy_fn(obs, rng):
        p, _ = modular_probs(theta, obs, gate_theta=z)
        return int(rng.choice(len(p), p=p))

    return policy_fn


def run_gate_phase(theta0_full, entries_g1, entries_g2, rng_es, rng_ep,
                   rng_ev, counter, generations):
    """v1 gate_phase mechanics (frozen arms): fitness 4xG1+4xG2,
    full-net sampled rollouts; checks mean(20+20)."""
    base = theta0_full.copy()
    sub = GATE_SLICE

    def fitness(sub_theta):
        full = base.copy()
        full[sub] = sub_theta
        rets = []
        for entries in (entries_g1, entries_g2):
            for _ in range(4):
                e = entries[int(rng_ep.integers(len(entries)))]
                ret, _ = rollout(full, "modular", e["family"], e["cfg"],
                                 int(rng_ep.integers(2 ** 31)), counter)
                rets.append(ret)
        return float(np.mean(rets))

    theta_sub = base[sub].copy()
    history, checks = [], []

    def cb(g, th, best):
        history.append(best)
        if (g + 1) % 10 == 0:
            full = base.copy()
            full[sub] = th
            g1 = train_eval_rate(full, "modular", "G1", entries_g1,
                                 rng_ev, 20, counter)
            g2 = train_eval_rate(full, "modular", "G2", entries_g2,
                                 rng_ev, 20, counter)
            checks.append((g1 + g2) / 2)
        return False

    res = train_es(theta_sub, fitness, rng_es, pop_pairs=POP_PAIRS,
                   sigma=SIGMA, alpha=ALPHA, generations=generations,
                   progress_cb=cb)
    full = base.copy()
    full[sub] = res["theta"]
    return full, res["generations"], history, checks


def summarize(rep):
    return {"n": rep["n"], "sr": rep["success_rate"], "ci": rep["ci95"]}


def run_seed(seed, data_dir, start_path, heldout_path, out_dir,
             generations=GEN_BUDGET):
    assert seed in V2_SEEDS
    manifest = json.load(open(f"{data_dir}/MANIFEST.json"))
    for fn, h in manifest.items():
        assert sha256_file(f"{data_dir}/{fn}") == h, f"dataset altered: {fn}"
    assert sha256_file(heldout_path) == HELDOUT_SHA, "heldout altered"
    train = json.load(open(f"{data_dir}/train_meta_v1.json"))
    g1tr = [e for e in train if e["family"] == "G1"]
    g2tr = [e for e in train if e["family"] == "G2"]
    assert len(g1tr) == 400 and len(g2tr) == 400
    battery = json.load(open(f"{data_dir}/test_battery_v1.json"))
    bat = {b: [e for e in battery if e["battery"] == b]
           for b in EVAL_BATTERIES}
    for b in bat:
        assert len(bat[b]) == 200, b
    gh = json.load(open(heldout_path))
    g1h = [e for e in gh if e["battery"] == "G1H"]
    g2h = [e for e in gh if e["battery"] == "G2H"]
    assert len(g1h) == 200 and len(g2h) == 200

    theta0 = np.load(start_path)["theta"]
    assert len(theta0) == MODULAR_DIM
    assert np.all(theta0[GATE_SLICE] == 0), "gate must start at zeros"
    start_sha = sha256_file(start_path)

    counter = [0]
    t0 = time.time()
    theta, gens, hist, checks = run_gate_phase(
        theta0, g1tr, g2tr, v2_rng(seed, 1), v2_rng(seed, 2),
        v2_rng(seed, 3), counter, generations)
    wall = time.time() - t0
    expect = (generations * (2 * POP_PAIRS) * 8 * HORIZON
              + (generations // 10) * CHECK_EPS * HORIZON)
    assert counter[0] == expect, (counter[0], expect)
    # Skills bit-frozen (protocol §2: any change invalidates the run).
    assert np.array_equal(theta[:N_SKILLS * SKILL_DIM],
                          theta0[:N_SKILLS * SKILL_DIM]), "skill freeze violated"

    ev = {}
    ev["G1H"] = {"normal": summarize(run_battery(
        make_policy("modular", theta, "normal"), g1h))}
    ev["G2H"] = {"normal": summarize(run_battery(
        make_policy("modular", theta, "normal"), g2h))}
    for b in EVAL_BATTERIES:
        d = {"sampled": summarize(run_battery(
            make_policy("modular", theta, "normal"), bat[b])),
            "greedy": summarize(run_battery(
                make_policy("modular", theta, "greedy"), bat[b])),
            "uniform": summarize(run_battery(
                uniform_policy(theta), bat[b])),
            "shuffle": summarize(run_battery(
                make_policy("modular", theta, "shuffle", SHUFFLE_SEED),
                bat[b])),
            "mask": summarize(run_battery(
                make_policy("modular", theta, "mask"), bat[b]))}
        for k in range(N_SKILLS):
            d[f"skill-{k}"] = summarize(run_battery(
                make_policy("modular", theta, f"skill-{k}"), bat[b]))
        ev[b] = d

    log = {"seed": seed, "start": os.path.basename(start_path),
           "start_sha256": start_sha,
           "design": "v2 compositional pilot: gate-only training",
           "train_entries": "train_meta_v1.json",
           "battery": "test_battery_v1.json",
           "g1g2_heldout": os.path.basename(heldout_path),
           "g1g2_heldout_sha256": HELDOUT_SHA,
           "shuffle_seed": SHUFFLE_SEED,
           "es": {"pop_pairs": POP_PAIRS, "sigma": SIGMA, "alpha": ALPHA,
                  "fit": "4xG1+4xG2", "gens": gens},
           "history": hist, "checks": checks, "steps": counter[0],
           "wall_s": wall, "eval": ev,
           "code_sha256": sha256_file(os.path.join(HERE, "gate_train.py"))}
    os.makedirs(out_dir, exist_ok=True)
    np.savez(f"{out_dir}/{PREFIX}{seed}_theta.npz", theta=theta)
    np.savez(f"{out_dir}/{PREFIX}{seed}_ckpts.npz", init=theta0,
             post_gate=theta.copy())
    json.dump(log, open(f"{out_dir}/{PREFIX}{seed}.json", "w"), indent=1,
              default=float)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, choices=V2_SEEDS, required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--start", default="")
    ap.add_argument("--heldout", default="v2/g1g2_heldout_v1.json")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--gens", type=int, default=GEN_BUDGET)
    args = ap.parse_args()
    start = args.start or f"pilot3_runs/P3_{args.seed}_theta.npz"
    log = run_seed(args.seed, args.data, start, args.heldout, args.out,
                   args.gens)
    print(json.dumps({k: v for k, v in log.items()
                      if k not in ("history", "checks", "eval")}, indent=1,
                     default=float))


if __name__ == "__main__":
    main()
