"""Registered sequential training driver. DO NOT RUN until explicitly authorized.

Arms: A1 modular-frozen+meta, A2 modular-plastic+meta, A3 monolith+meta,
      A4 modular-frozen+control. Protocol: PREREG 2/4/5/7.
Fixed GEN_BUDGET per phase for all arms/seeds; acquisition checks are
record-only (compliance correction, see RUNLOG). Gate phase likewise fixed.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from es import train_es
from evaluate import assert_no_net, run_battery
from gen import sha256_file
from models import (MODULAR_DIM, MONO_DIM, SKILL_DIM, GATE_DIM, N_SKILLS,
                    SKILL_SHAPES, GATE_SHAPES, MONO_SHAPES,
                    verify_param_match, modular_probs, ModularNet, MonoNet)
from policies import xavier_init, softmax
from tasks import TaskRunner

SEEDS = [101, 102, 103, 104, 105]
ARMS = ["A1", "A2", "A3", "A4"]
ARM_MANIFEST = {"A1": "train_meta_v1.json", "A2": "train_meta_v1.json",
                "A3": "train_meta_v1.json", "A4": "train_control_v1.json"}
BATTERY_FILE = "test_battery_v1.json"
MANIFEST_FILE = "MANIFEST.json"
# Compliance correction (pre-run): fixed equal generation budget per phase for
# all arms/seeds. Acquisition checks are measured and reported but NEVER shorten
# the budget. FAIL = unacquired at GEN_BUDGET (the 800-gen extension is removed:
# it cannot coexist with equal actual budgets). See RUNLOG.
GEN_BUDGET = 400
POP_PAIRS, SIGMA, ALPHA = 32, 0.1, 0.05
FIT_EPS = 8
ACQ = 0.80
PURPOSES = {"init": 0, "es": 1, "ep": 2, "ev": 3}


def rng_for(seed, arm, phase, purpose):
    return np.random.default_rng(
        [seed, ARMS.index(arm), phase, PURPOSES[purpose]])


def skill_slice(k):
    return slice(k * SKILL_DIM, (k + 1) * SKILL_DIM)


GATE_SLICE = slice(N_SKILLS * SKILL_DIM, MODULAR_DIM)


def rollout(theta_full, arch, family, cfg, ep_seed, counter, skill_only=None):
    runner = TaskRunner()
    rng = np.random.default_rng(ep_seed)
    obs = runner.reset(family, cfg, int(ep_seed))
    if arch == "modular":
        net = ModularNet(theta_full)
    else:
        net = MonoNet(theta_full)
    total, done, steps = 0.0, False, 0
    while not done:
        if arch == "modular":
            p, _ = net.probs(obs, skill_only=skill_only)
        else:
            p = net.probs(obs)
        obs, r, done, info = runner.step(int(rng.choice(len(p), p=p)))
        total += r
        steps += 1
    counter[0] += steps
    return total, bool(info["success"])


def subset(entries, family):
    return [e for e in entries if e["family"] == family]


def train_eval_rate(theta_full, arch, family, entries, rng, n_eps=40,
                    counter=None, skill_only=None):
    c = counter if counter is not None else [0]
    ok = 0
    for _ in range(n_eps):
        e = entries[int(rng.integers(len(entries)))]
        _, s = rollout(theta_full, arch, family, e["cfg"],
                       int(rng.integers(2**31)), c, skill_only)
        ok += s
    return ok / n_eps


def first_pass_gen(checks, every=10):
    for i in range(1, len(checks)):
        if checks[i - 1] >= ACQ and checks[i] >= ACQ:
            return (i + 1) * every
    return None


def run_es_phase(theta0_full, sub, arch, fit_families, entries_by_fam,
                 rng_es, rng_ep, counter, check_fn=None, generations=None,
                 early_stop=False):
    """ES on a parameter subspace. Always runs the full generation budget
    unless early_stop=True (reacquisition measurement only)."""
    budget = GEN_BUDGET if generations is None else generations
    base = theta0_full.copy()

    def fitness(sub_theta):
        full = base.copy()
        full[sub] = sub_theta
        rets = []
        for fam, n_eps in fit_families:
            pool = entries_by_fam[fam]
            for _ in range(n_eps):
                e = pool[int(rng_ep.integers(len(pool)))]
                ret, _ = rollout(full, arch, fam, e["cfg"],
                                 int(rng_ep.integers(2**31)), counter)
                rets.append(ret)
        return float(np.mean(rets))

    theta_sub = base[sub].copy()
    history, checks = [], []

    def cb(g, th, best):
        history.append(best)
        if check_fn is not None and (g + 1) % 10 == 0:
            full = base.copy()
            full[sub] = th
            checks.append(check_fn(full))
            if early_stop and first_pass_gen(checks) is not None:
                return True
        return False

    res = train_es(theta_sub, fitness, rng_es, pop_pairs=POP_PAIRS,
                   sigma=SIGMA, alpha=ALPHA, generations=budget,
                   progress_cb=cb)
    full = base.copy()
    full[sub] = res["theta"]
    return full, res["generations"], history, checks


def skill_phase(arm, arch, family, k, entries_by_fam, theta_base, seed,
                phase_idx, counter, plastic_full):
    rng_es = rng_for(seed, arm, phase_idx, "es")
    rng_ep = rng_for(seed, arm, phase_idx, "ep")
    rng_ev = rng_for(seed, arm, phase_idx, "ev")
    if plastic_full:
        sub = slice(0, len(theta_base))
    else:
        sub = skill_slice(k)
        theta_base = theta_base.copy()
        fresh = xavier_init(rng_for(seed, arm, phase_idx, "init"), SKILL_SHAPES)
        theta_base[sub] = fresh

    def check_fn(full):
        return train_eval_rate(full, arch, family, entries_by_fam[family],
                               rng_ev, skill_only=(k if not plastic_full else None),
                               counter=counter)

    # Identical check schedule for every arm (record-only; never shortens).
    full, gens, hist, checks = run_es_phase(
        theta_base, sub, arch, [(family, FIT_EPS)], entries_by_fam,
        rng_es, rng_ep, counter, check_fn)
    fp = first_pass_gen(checks)
    return {"theta": full, "gens": gens, "history": hist, "checks": checks,
            "first_pass_gen": fp, "acquired": bool(fp is not None),
            "steps": counter[0]}


def gate_phase(arm, arch, entries_by_fam, theta_base, seed, phase_idx,
               counter, plastic_full):
    rng_es = rng_for(seed, arm, phase_idx, "es")
    rng_ep = rng_for(seed, arm, phase_idx, "ep")
    rng_ev = rng_for(seed, arm, phase_idx, "ev")
    sub = slice(0, len(theta_base)) if plastic_full else GATE_SLICE

    def check_fn(full):
        g1 = train_eval_rate(full, arch, "G1", entries_by_fam["G1"],
                             rng_ev, n_eps=20, counter=counter)
        g2 = train_eval_rate(full, arch, "G2", entries_by_fam["G2"],
                             rng_ev, n_eps=20, counter=counter)
        return (g1 + g2) / 2

    full, gens, hist, checks = run_es_phase(
        theta_base, sub, arch, [("G1", 4), ("G2", 4)], entries_by_fam,
        rng_es, rng_ep, counter, check_fn=check_fn)
    return {"theta": full, "gens": gens, "history": hist, "checks": checks,
            "steps": counter[0]}


def retention_eval(arch, theta, battery, acquired):
    """Per-skill (one-hot) retention for modular; whole-net for monolith."""
    out = {}
    for k, fam in enumerate(["A", "B", "C", "D"]):
        if fam not in acquired:
            out[fam] = None
            continue
        entries = [e for e in battery if e["battery"] == fam]
        if arch == "modular":
            def policy_fn(obs, rng, k=k):
                p, _ = modular_probs(theta, obs, skill_only=k)
                return int(rng.choice(len(p), p=p))
        else:
            from models import mono_probs

            def policy_fn(obs, rng):
                p = mono_probs(theta, obs)
                return int(rng.choice(len(p), p=p))
        rep = run_battery(policy_fn, entries)
        out[fam] = {"sr": rep["success_rate"], "ci": rep["ci95"]}
    return out


def run_arm_seed(arm, seed, data_dir, out_dir):
    assert_no_net()
    verify_param_match()
    manifest = json.load(open(f"{data_dir}/{MANIFEST_FILE}"))
    for fn, h in manifest.items():
        assert sha256_file(f"{data_dir}/{fn}") == h, f"dataset altered: {fn}"
    train_entries = json.load(open(f"{data_dir}/{ARM_MANIFEST[arm]}"))
    battery = json.load(open(f"{data_dir}/{BATTERY_FILE}"))
    entries_by_fam = {f: subset(train_entries, f)
                      for f in ["A", "B", "C", "D", "G1", "G2"]}
    arch = "mono" if arm == "A3" else "modular"
    plastic = arm in ("A2", "A3")
    dim = MONO_DIM if arch == "mono" else MODULAR_DIM
    if arch == "mono":
        theta = xavier_init(rng_for(seed, arm, 9, "init"), MONO_SHAPES)
    else:
        parts = [xavier_init(rng_for(seed, arm, 9, "init"), SKILL_SHAPES)
                 for _ in range(N_SKILLS)]
        parts.append(np.zeros(GATE_DIM))
        theta = np.concatenate(parts)
    counter = [0]
    log = {"arm": arm, "seed": seed, "phases": {}, "retention": {},
           "reacquisition": {}}
    acquired = []
    for idx, (fam, k) in enumerate(zip(["A", "B", "C", "D"], range(4))):
        res = skill_phase(arm, arch, fam, k, entries_by_fam, theta, seed,
                          idx, counter, plastic)
        theta = res["theta"]
        if res["acquired"]:
            acquired.append(fam)
        log["phases"][fam] = {k2: v for k2, v in res.items() if k2 != "theta"}
        log["retention"][f"after_{fam}"] = retention_eval(arch, theta, battery,
                                                          acquired if not plastic
                                                          else ["A", "B", "C", "D"])
    gate = gate_phase(arm, arch, entries_by_fam, theta, seed, 4, counter, plastic)
    theta = gate["theta"]
    log["phases"]["gate"] = {k2: v for k2, v in gate.items() if k2 != "theta"}
    post = retention_eval(arch, theta, battery,
                          acquired if not plastic else ["A", "B", "C", "D"])
    log["retention"]["after_gate"] = post
    if not plastic:
        for fam in acquired:
            before = log["retention"]["after_D"][fam]["sr"]
            assert post[fam]["sr"] == before, f"freeze violated: {fam}"
    # reacquisition on drops > 10pp: 10% of the fixed phase budget, with
    # early stop (a relearning-speed measurement, not a comparative budget).
    for fam in (acquired if not plastic else ["A", "B", "C", "D"]):
        pre = log["retention"][f"after_{fam}"][fam]
        cur = post[fam]
        if pre is None or cur is None:
            continue
        if pre["sr"] - cur["sr"] > 0.10:
            c2 = [0]
            k = ["A", "B", "C", "D"].index(fam)
            sub = slice(0, len(theta)) if plastic else skill_slice(k)
            rng_ev2 = rng_for(seed, arm, 6, "ev")

            def recheck(full, k=k):
                return train_eval_rate(
                    full, arch, fam, entries_by_fam[fam], rng_ev2,
                    skill_only=(None if plastic else k), counter=c2)

            full, gens, _, _ = run_es_phase(
                theta, sub, arch, [(fam, FIT_EPS)], entries_by_fam,
                rng_for(seed, arm, 6, "es"), rng_for(seed, arm, 6, "ep"), c2,
                check_fn=recheck, generations=GEN_BUDGET // 10,
                early_stop=True)
            theta = full
            rec = retention_eval(arch, theta, battery, [fam])[fam]
            scratch_fp = log["phases"][fam]["first_pass_gen"]
            log["reacquisition"][fam] = {
                "pre": pre["sr"], "dropped": cur["sr"], "post": rec["sr"],
                "gens": gens, "scratch_first_pass": scratch_fp,
                "savings": (scratch_fp / gens) if (scratch_fp and gens) else None}
    log["total_steps"] = counter[0]
    os.makedirs(out_dir, exist_ok=True)
    np.savez(f"{out_dir}/{arm}_{seed}_theta.npz", theta=theta)
    json.dump(log, open(f"{out_dir}/{arm}_{seed}.json", "w"), indent=1,
              default=float)
    return log


def plan():
    info = {"seeds": SEEDS, "arms": ARMS, "manifest": ARM_MANIFEST,
            "pop_pairs": POP_PAIRS, "sigma": SIGMA, "alpha": ALPHA,
            "fit_eps": FIT_EPS, "gen_budget_per_phase": GEN_BUDGET,
            "early_stop_comparative": False, "acq": ACQ,
            "params": verify_param_match()}
    print(json.dumps(info, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--seed", type=int, choices=SEEDS)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()
    if args.plan:
        plan()
        return
    if not args.arm or not args.seed:
        ap.error("--arm and --seed required")
    log = run_arm_seed(args.arm, args.seed, args.data, args.out)
    print(json.dumps({k: v for k, v in log.items() if k != "phases"},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
