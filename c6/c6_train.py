"""C6 trainer: C4 exact + 800 generations (budget test, single variable).

Isolated family-C, modular arch, one-hot fitness, SAMPLED training
(like C4, not greedy), n_fit=8, no shaping, same arch/env/reward/obs.
Only change vs C4: 800 gens instead of 400. Fresh seeds 501-503.

Reuses diag_train.run_job UNCHANGED (snaps every 50 -> 16 snaps to
g800; checks every 10 -> 80 checks; step assert adapts). Measurement-
only addition (training untouched): pure-skill heldout recorded
in-JSON as heldout_pure (frozen GO metric, see README).
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "diag"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

import diag_train as D
from evaluate import run_battery, make_policy
from models import modular_probs

C6_CONFIG = {"family": "C", "arch": "modular", "n_fit": 8, "greedy": False,
             "shaping": None, "fit1hot": True}
C6_SEEDS = [501, 502, 503]
C6_GENS = 800
assert not set(C6_SEEDS) & {101, 102, 103, 104, 105, 201, 202, 203,
                            301, 302, 303, 401, 402, 403}

D.JOBS["C6"] = dict(C6_CONFIG)
D.JOB_INDEX = {j: i for i, j in enumerate(sorted(D.JOBS))}
D.DIAG_SEEDS = list(C6_SEEDS)


def greedy_skill2_policy(theta):
    def policy_fn(obs, rng):
        p, _ = modular_probs(theta, obs, skill_only=2)
        return int(np.argmax(p))

    return policy_fn


def run_c6(seed, data_dir, out_dir, generations=C6_GENS):
    assert seed in C6_SEEDS
    log = D.run_job("C6", seed, data_dir, out_dir, generations)
    assert log["config"] == C6_CONFIG, log["config"]
    theta = np.load(f"{out_dir}/C6_{seed}_theta.npz")["theta"]
    battery = [e for e in json.load(
        open(f"{data_dir}/test_battery_v1.json")) if e["battery"] == "C"]
    assert len(battery) == 200
    pure = {}
    rep = run_battery(make_policy("modular", theta, "skill-2"), battery)
    pure["sampled"] = {"n": rep["n"], "sr": rep["success_rate"],
                       "ci": rep["ci95"]}
    rep = run_battery(greedy_skill2_policy(theta), battery)
    pure["greedy"] = {"n": rep["n"], "sr": rep["success_rate"],
                      "ci": rep["ci95"]}
    log["heldout_pure"] = pure
    json.dump(log, open(f"{out_dir}/C6_{seed}.json", "w"), indent=1)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, choices=C6_SEEDS, required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--gens", type=int, default=C6_GENS)
    args = ap.parse_args()
    log = run_c6(args.seed, args.data, args.out, args.gens)
    print(json.dumps({k: v for k, v in log.items()
                      if k not in ("history", "checks")}, indent=1))


if __name__ == "__main__":
    main()
