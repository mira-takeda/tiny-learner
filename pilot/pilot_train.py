"""v2-pilot sequential trainer (pilot-only; v1 CLOSED, src/ untouched).

Modular-frozen A->B->C->D, v1 A1 mechanics with ONE change: fitness is
evaluated one-hot on the phase's own skill (C4/B2 finding). Everything
else mirrors frozen src/train.py: env/obs/actions/rewards, ES hypers
(32 pairs, sigma 0.1, alpha 0.05), 400 gens/phase, fitness 8 eps,
sampled one-hot checks every 10 gens (record-only), fresh Xavier slice
per phase, freeze by construction, exact step asserts.

Per phase end: pure-skill heldout (200 eps, frozen batteries) sampled
(primary, v1-comparable) + greedy (secondary descriptive) for every
skill trained so far. Checkpoints: init, post_A/B/C/D. No gate phase,
no I1/H batteries. Seeds 301-303 (fresh streams).
"""

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src")))

import numpy as np

from es import train_es
from evaluate import run_battery, make_policy
from gen import sha256_file
from models import SKILL_SHAPES, GATE_DIM, N_SKILLS
from policies import xavier_init
from tasks import TaskRunner
from train import (GEN_BUDGET, POP_PAIRS, SIGMA, ALPHA, HORIZON, CHECK_EPS,
                   PHASE_STEPS, ACQ, rollout, train_eval_rate,
                   first_pass_gen, skill_slice)

PILOT_SEEDS = [301, 302, 303]
PHASES = [("A", 0), ("B", 1), ("C", 2), ("D", 3)]
TRAIN_MANIFEST = "train_meta_v1.json"  # same as v1 A1
BATTERY_FILE = "test_battery_v1.json"
MANIFEST_FILE = "MANIFEST.json"
PURPOSES = {"init": 0, "es": 1, "ep": 2, "ev": 3}


def pilot_rng(seed, phase_idx, purpose):
    return np.random.default_rng([seed, phase_idx, PURPOSES[purpose]])


def make_fitness(base, sub, fam, entries, rng_ep, counter,
                 fit_skill_only):
    """ES fitness closure. The ONLY diff vs frozen run_es_phase is the
    skill_only argument (one-hot when fit_skill_only=k)."""

    def fitness(sub_theta):
        full = base.copy()
        full[sub] = sub_theta
        rets = []
        for _ in range(8):
            e = entries[int(rng_ep.integers(len(entries)))]
            ret, _ = rollout(full, "modular", fam, e["cfg"],
                             int(rng_ep.integers(2 ** 31)), counter,
                             fit_skill_only)
            rets.append(ret)
        return float(np.mean(rets))

    return fitness


def run_pilot_phase(theta0_full, sub, fam, k, entries, rng_es, rng_ep,
                    rng_ev, counter, generations, fit_skill_only):
    """Mirror of frozen run_es_phase; the ONLY diff is skill_only in
    fitness (fit_skill_only=k). With fit_skill_only=None it is bit-exact
    vs frozen run_es_phase (proven in smoke_pilot.py)."""
    base = theta0_full.copy()
    fitness = make_fitness(base, sub, fam, entries, rng_ep, counter,
                           fit_skill_only)
    theta_sub = base[sub].copy()
    history, checks = [], []

    def cb(g, th, best):
        history.append(best)
        if (g + 1) % 10 == 0:
            full = base.copy()
            full[sub] = th
            checks.append(train_eval_rate(full, "modular", fam, entries,
                                          rng_ev, CHECK_EPS, counter, k))
        return False

    res = train_es(theta_sub, fitness, rng_es, pop_pairs=POP_PAIRS,
                   sigma=SIGMA, alpha=ALPHA, generations=generations,
                   progress_cb=cb)
    full = base.copy()
    full[sub] = res["theta"]
    return full, res["generations"], history, checks


def greedy_skill_policy(theta, k):
    """Pure-skill greedy policy (descriptive secondary; the argmax is the
    only diff vs frozen retention_eval sampled construction)."""
    from models import modular_probs

    def policy_fn(obs, rng):
        p, _ = modular_probs(theta, obs, skill_only=k)
        return int(np.argmax(p))

    return policy_fn


def score_skill(theta, k, battery_entries):
    out = {}
    pol_s = make_policy("modular", theta, f"skill-{k}")
    rep = run_battery(pol_s, battery_entries)
    out["sampled"] = {"n": rep["n"], "sr": rep["success_rate"],
                      "ci": rep["ci95"]}
    rep = run_battery(greedy_skill_policy(theta, k), battery_entries)
    out["greedy"] = {"n": rep["n"], "sr": rep["success_rate"],
                     "ci": rep["ci95"]}
    return out


def run_seed(seed, data_dir, out_dir, generations=GEN_BUDGET):
    assert seed in PILOT_SEEDS
    manifest = json.load(open(f"{data_dir}/{MANIFEST_FILE}"))
    for fn, h in manifest.items():
        assert sha256_file(f"{data_dir}/{fn}") == h, f"dataset altered: {fn}"
    train_entries = json.load(open(f"{data_dir}/{TRAIN_MANIFEST}"))
    battery = json.load(open(f"{data_dir}/{BATTERY_FILE}"))
    entries_by_fam = {f: [e for e in train_entries if e["family"] == f]
                      for f in ["A", "B", "C", "D"]}
    for f in entries_by_fam:
        assert len(entries_by_fam[f]) == 400, f
    bat_by_fam = {f: [e for e in battery if e["battery"] == f]
                  for f in ["A", "B", "C", "D"]}
    for f in bat_by_fam:
        assert len(bat_by_fam[f]) == 200, f

    # v1-identical init: 4 Xavier slices, one stream, + zero gate.
    parts = [xavier_init(pilot_rng(seed, 9, "init"), SKILL_SHAPES)
             for _ in range(N_SKILLS)]
    parts.append(np.zeros(GATE_DIM))
    theta = np.concatenate(parts)
    ckpts = {"init": theta.copy()}

    counter = [0]
    t_all = time.time()
    log = {"seed": seed,
           "design": "v2-pilot modular-frozen A->B->C->D, one-hot fitness",
           "change_vs_v1": "fitness skill_only=k per phase (C4/B2); "
                           "all else v1-A1 mechanics",
           "train_entries": TRAIN_MANIFEST, "battery": BATTERY_FILE,
           "es": {"pop_pairs": POP_PAIRS, "sigma": SIGMA, "alpha": ALPHA,
                  "n_fit": 8, "gens_per_phase": generations},
           "phases": {}, "heldout_after": {}}
    expect_per_phase = (generations * (2 * POP_PAIRS) * 8 * HORIZON
                        + (generations // 10) * CHECK_EPS * HORIZON)

    for idx, (fam, k) in enumerate(PHASES):
        # v1 skill_phase re-inits the phase slice fresh (frozen arms).
        sub = skill_slice(k)
        theta[sub] = xavier_init(pilot_rng(seed, idx, "init"), SKILL_SHAPES)
        # Frozen slices must be untouched by later phases (checked each
        # phase BEFORE training it, then asserted again at the end).
        for pfam, pk in PHASES[:idx]:
            assert np.array_equal(theta[skill_slice(pk)],
                                  ckpts[f"post_{pfam}"][skill_slice(pk)]), \
                f"frozen slice {pfam} altered before phase {fam}"
        c0 = counter[0]
        t0 = time.time()
        theta, gens, hist, checks = run_pilot_phase(
            theta, sub, fam, k, entries_by_fam[fam],
            pilot_rng(seed, idx, "es"), pilot_rng(seed, idx, "ep"),
            pilot_rng(seed, idx, "ev"), counter, generations, k)
        wall = time.time() - t0
        assert counter[0] - c0 == expect_per_phase, (counter[0] - c0,
                                                    expect_per_phase)
        ckpts[f"post_{fam}"] = theta.copy()
        fp = first_pass_gen(checks)
        log["phases"][fam] = {"gens": gens, "history": hist, "checks": checks,
                              "first_pass_gen": fp,
                              "acquired_flag": bool(fp is not None),
                              "steps": counter[0] - c0, "wall_s": wall}
        held = {}
        for sfam, sk in PHASES[:idx + 1]:
            held[sfam] = score_skill(theta, sk, bat_by_fam[sfam])
        log["heldout_after"][f"post_{fam}"] = held

    # Final freeze audit across the checkpoint chain.
    for i, (fam, k) in enumerate(PHASES):
        for later in [p for p, _ in PHASES[i:]]:
            assert np.array_equal(ckpts[f"post_{later}"][skill_slice(k)],
                                  ckpts[f"post_{fam}"][skill_slice(k)]), \
                (fam, later)
    # Retention table: own phase-end -> post_D (expect exactly 0.0).
    ret = {}
    for fam, _ in PHASES:
        pre = log["heldout_after"][f"post_{fam}"][fam]["sampled"]["sr"]
        cur = log["heldout_after"]["post_D"][fam]["sampled"]["sr"]
        ret[fam] = {"phase_end": pre, "post_D": cur, "drop": pre - cur}
    log["retention_sampled"] = ret
    log["total_steps"] = counter[0]
    log["wall_s"] = time.time() - t_all
    log["code_sha256"] = sha256_file(os.path.join(HERE, "pilot_train.py"))

    os.makedirs(out_dir, exist_ok=True)
    np.savez(f"{out_dir}/{seed}_theta.npz", theta=theta)
    np.savez(f"{out_dir}/{seed}_ckpts.npz", **ckpts)
    json.dump(log, open(f"{out_dir}/{seed}.json", "w"), indent=1,
              default=float)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, choices=PILOT_SEEDS, required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--gens", type=int, default=GEN_BUDGET)
    args = ap.parse_args()
    log = run_seed(args.seed, args.data, args.out, args.gens)
    print(json.dumps({k: v for k, v in log.items()
                      if k not in ("phases", "heldout_after")},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
