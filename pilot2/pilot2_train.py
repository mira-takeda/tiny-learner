"""pilot2 sequential trainer (pilot2-only; pilot/ and src/ untouched).

Paired with v2-pilot (seeds 301-303, same RNG streams -> identical
init), ONE intervention: phase C at 800 gens (segments 400+400 through
the pilot runner with continued streams: bit-exact vs a single 800 run,
proven in smoke). A/B/D at 400. All else pilot-identical: one-hot
fitness, frozen ES, checks, freeze, budgets, pure-skill heldouts.

Checkpoints: init, post_A, post_B, post_C400 (C@g400 + heldout),
post_C (C@g800 = C phase end), post_D (= final theta).
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "pilot"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

import pilot_train as P
from gen import sha256_file
from models import SKILL_SHAPES, GATE_DIM, N_SKILLS
from policies import xavier_init
from train import (GEN_BUDGET, POP_PAIRS, SIGMA, ALPHA, HORIZON, CHECK_EPS,
                   skill_slice, first_pass_gen)

PREFIX = "P2_"
PILOT2_SEEDS = [301, 302, 303]
PHASE_GENS = {"A": 400, "B": 400, "C": 800, "D": 400}
C_MID = 400  # mid-C capture generation (protocol key post_C400)


def phase_steps(gens):
    return (gens * (2 * POP_PAIRS) * 8 * HORIZON
            + (gens // 10) * CHECK_EPS * HORIZON)


def run_seed2(seed, data_dir, out_dir, phase_gens=None):
    assert seed in PILOT2_SEEDS
    pg = dict(PHASE_GENS) if phase_gens is None else dict(phase_gens)
    manifest = json.load(open(f"{data_dir}/{P.MANIFEST_FILE}"))
    for fn, h in manifest.items():
        assert sha256_file(f"{data_dir}/{fn}") == h, f"dataset altered: {fn}"
    train_entries = json.load(open(f"{data_dir}/{P.TRAIN_MANIFEST}"))
    battery = json.load(open(f"{data_dir}/{P.BATTERY_FILE}"))
    entries_by_fam = {f: [e for e in train_entries if e["family"] == f]
                      for f in ["A", "B", "C", "D"]}
    for f in entries_by_fam:
        assert len(entries_by_fam[f]) == 400, f
    bat_by_fam = {f: [e for e in battery if e["battery"] == f]
                  for f in ["A", "B", "C", "D"]}
    for f in bat_by_fam:
        assert len(bat_by_fam[f]) == 200, f

    parts = [xavier_init(P.pilot_rng(seed, 9, "init"), SKILL_SHAPES)
             for _ in range(N_SKILLS)]
    parts.append(np.zeros(GATE_DIM))
    theta = np.concatenate(parts)
    ckpts = {"init": theta.copy()}

    import time
    counter = [0]
    t_all = time.time()
    log = {"seed": seed, "paired_with": "v2-pilot",
           "design": "pilot2 sequential A->B->C(800)->D, one-hot fitness",
           "change_vs_pilot": "C 800 gens only; all else identical",
           "train_entries": P.TRAIN_MANIFEST, "battery": P.BATTERY_FILE,
           "es": {"pop_pairs": POP_PAIRS, "sigma": SIGMA, "alpha": ALPHA,
                  "n_fit": 8, "phase_gens": pg},
           "phases": {}, "heldout_after": {}}

    for idx, (fam, k) in enumerate(P.PHASES):
        sub = skill_slice(k)
        theta[sub] = xavier_init(P.pilot_rng(seed, idx, "init"),
                                 SKILL_SHAPES)
        for pfam, pk in P.PHASES[:idx]:
            assert np.array_equal(theta[skill_slice(pk)],
                                  ckpts[f"post_{pfam}"][skill_slice(pk)]), \
                f"frozen slice {pfam} altered before phase {fam}"
        c0 = counter[0]
        t0 = time.time()
        gens = pg[fam]
        rng_es, rng_ep = P.pilot_rng(seed, idx, "es"), P.pilot_rng(seed, idx, "ep")
        rng_ev = P.pilot_rng(seed, idx, "ev")
        if fam == "C" and gens > C_MID:
            # Segmentation is bit-exact vs a single run ONLY when the
            # boundary aligns with the 10-gen check cadence.
            assert C_MID % 10 == 0, C_MID
            f1, _, h1, c1 = P.run_pilot_phase(
                theta, sub, fam, k, entries_by_fam[fam], rng_es, rng_ep,
                rng_ev, counter, C_MID, k)
            ckpts[f"post_C{C_MID:03d}"] = f1.copy()
            log["heldout_after"][f"post_C{C_MID:03d}"] = {
                sf: P.score_skill(f1, sk, bat_by_fam[sf])
                for sf, sk in P.PHASES[:idx + 1]}
            theta, _, h2, c2 = P.run_pilot_phase(
                f1, sub, fam, k, entries_by_fam[fam], rng_es, rng_ep,
                rng_ev, counter, gens - C_MID, k)
            hist, checks = h1 + h2, c1 + c2
        else:
            theta, _, hist, checks = P.run_pilot_phase(
                theta, sub, fam, k, entries_by_fam[fam], rng_es, rng_ep,
                rng_ev, counter, gens, k)
        wall = time.time() - t0
        assert counter[0] - c0 == phase_steps(gens), (counter[0] - c0,
                                                     phase_steps(gens))
        ckpts[f"post_{fam}"] = theta.copy()
        fp = first_pass_gen(checks)
        log["phases"][fam] = {"gens": gens, "history": hist, "checks": checks,
                              "first_pass_gen": fp,
                              "acquired_flag": bool(fp is not None),
                              "steps": counter[0] - c0, "wall_s": wall}
        log["heldout_after"][f"post_{fam}"] = {
            sf: P.score_skill(theta, sk, bat_by_fam[sf])
            for sf, sk in P.PHASES[:idx + 1]}

    for i, (fam, k) in enumerate(P.PHASES):
        for later in [p for p, _ in P.PHASES[i:]]:
            assert np.array_equal(ckpts[f"post_{later}"][skill_slice(k)],
                                  ckpts[f"post_{fam}"][skill_slice(k)]), \
                (fam, later)
    ret = {}
    for fam, _ in P.PHASES:
        pre = log["heldout_after"][f"post_{fam}"][fam]["sampled"]["sr"]
        cur = log["heldout_after"]["post_D"][fam]["sampled"]["sr"]
        ret[fam] = {"phase_end": pre, "post_D": cur, "drop": pre - cur}
    log["retention_sampled"] = ret
    log["total_steps"] = counter[0]
    log["wall_s"] = time.time() - t_all
    log["code_sha256"] = sha256_file(os.path.join(HERE, "pilot2_train.py"))
    log["pilot_code_sha256"] = sha256_file(
        os.path.join(ROOT, "pilot", "pilot_train.py"))

    os.makedirs(out_dir, exist_ok=True)
    np.savez(f"{out_dir}/{PREFIX}{seed}_theta.npz", theta=theta)
    np.savez(f"{out_dir}/{PREFIX}{seed}_ckpts.npz", **ckpts)
    json.dump(log, open(f"{out_dir}/{PREFIX}{seed}.json", "w"), indent=1,
              default=float)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, choices=PILOT2_SEEDS, required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--gens", type=int, default=0,
                    help="override: all phases to G gens (smoke only)")
    args = ap.parse_args()
    pg = None
    if args.gens:
        pg = {f: args.gens for f in ["A", "B", "C", "D"]}
    log = run_seed2(args.seed, args.data, args.out, pg)
    print(json.dumps({k: v for k, v in log.items()
                      if k not in ("phases", "heldout_after")},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
