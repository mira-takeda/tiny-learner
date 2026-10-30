"""C5 trainer: C4 (one-hot fitness) + C2 (greedy training) interaction.

Isolated family-C, modular arch, 400 gens, n_fit=8, no shaping, same
budget as C4/C2. Reuses diag_train.run_job UNCHANGED (same code path as
the diag campaign) with job C5 = C0 + fit1hot + greedy, on fresh seeds
401-403 (fresh RNG streams, no overlap with v1/diag/pilot).

Measurement-only addition (training untouched): pure-skill heldout
(sampled primary + greedy secondary, frozen batteries/code) is recorded
in-JSON as heldout_pure, because the diag blended heldout understates
modular skills (C4 precedent: blended sampled 0.165 vs pure 0.735).
The frozen GO criterion gates on heldout_pure.sampled (see README).
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

C5_CONFIG = {"family": "C", "arch": "modular", "n_fit": 8, "greedy": True,
             "shaping": None, "fit1hot": True}
C5_SEEDS = [401, 402, 403]
assert not set(C5_SEEDS) & {101, 102, 103, 104, 105, 201, 202, 203,
                            301, 302, 303}

D.JOBS["C5"] = dict(C5_CONFIG)
D.JOB_INDEX = {j: i for i, j in enumerate(sorted(D.JOBS))}
D.DIAG_SEEDS = list(C5_SEEDS)


def greedy_skill2_policy(theta):
    def policy_fn(obs, rng):
        p, _ = modular_probs(theta, obs, skill_only=2)
        return int(np.argmax(p))

    return policy_fn


def run_c5(seed, data_dir, out_dir, generations=400):
    assert seed in C5_SEEDS
    log = D.run_job("C5", seed, data_dir, out_dir, generations)
    assert log["config"] == C5_CONFIG, log["config"]
    # Pure-skill heldout AFTER training (measurement only; frozen code,
    # frozen battery, deterministic per-episode seeds like v1 retention).
    theta = np.load(f"{out_dir}/C5_{seed}_theta.npz")["theta"]
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
    json.dump(log, open(f"{out_dir}/C5_{seed}.json", "w"), indent=1)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, choices=C5_SEEDS, required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--gens", type=int, default=400)
    args = ap.parse_args()
    log = run_c5(args.seed, args.data, args.out, args.gens)
    print(json.dumps({k: v for k, v in log.items()
                      if k not in ("history", "checks")}, indent=1))


if __name__ == "__main__":
    main()
