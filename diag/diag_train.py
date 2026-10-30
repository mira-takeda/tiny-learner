"""Isolated diagnostic training jobs (diag-only; v1 CLOSED, src/ untouched).

Jobs x seeds 201-203 (fresh streams, no overlap with v1/batteries):
  C0: C isolated, exact v1 C-phase mechanics (modular slice; fitness on the
      FULL gated net with zero-gate uniform blend exactly as v1; co-slices
      Xavier-fresh, gate zeros; checks one-hot; fitness 8 eps)
  C1: C0 + fitness 32 eps (tests fitness-noise hypothesis)
  C4: C0 + ONE-HOT fitness (aligned train/eval; direct test of the v1
      fitness/checks mismatch hypothesis — main campaign)
  C2: C0 + greedy rollouts/checks (diag-local rollout; argmax for sampling)
  C3: C0 + delivery-staged shaping (pickup +0.5 once; +1.0 per dwell step
      1-4; frozen +10 at 5-dwell unchanged; obs/actions/geometry/success
      definition unchanged; heldout scored with FROZEN reward)
  B0: B modular-slice isolated, v1 mechanics (reproduces A1-condition)
  B1: B mono A3-arch full-net isolated, fresh init (capacity-vs-init test)
  B2: B0 + one-hot fitness (same alignment test for B — main campaign)

Fresh Xavier init per job; fixed 400 gens; checks/10 (40 eps, record-only);
final heldout 200 eps sampled AND greedy (frozen); exact step asserts;
snapshots every 50 gens + final. ES hypers identical to v1 (32 pairs,
sigma 0.1, alpha 0.05). diag rollout mirrors train.rollout exactly except
the documented one-line differences (equivalence-tested in smoke).
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src")))

import numpy as np

from env import Arena
from es import train_es
from evaluate import make_policy, run_battery
from models import (MODULAR_DIM, MONO_DIM, SKILL_DIM, GATE_DIM, N_SKILLS,
                    SKILL_SHAPES, GATE_SHAPES, MONO_SHAPES)
from policies import xavier_init
from tasks import TaskRunner
from train import (GEN_BUDGET, POP_PAIRS, SIGMA, ALPHA, HORIZON,
                   CHECKS_PER_PHASE, CHECK_EPS, PHASE_STEPS, rollout,
                   train_eval_rate, first_pass_gen, skill_slice,
                   rng_for as _unused)

DIAG_SEEDS = [201, 202, 203]
JOBS = {
    "C0": {"family": "C", "arch": "modular", "n_fit": 8, "greedy": False,
           "shaping": None, "fit1hot": False},
    "C1": {"family": "C", "arch": "modular", "n_fit": 32, "greedy": False,
           "shaping": None, "fit1hot": False},
    "C2": {"family": "C", "arch": "modular", "n_fit": 8, "greedy": True,
           "shaping": None, "fit1hot": False},
    "C3": {"family": "C", "arch": "modular", "n_fit": 8, "greedy": False,
           "shaping": "delivery_stage", "fit1hot": False},
    "C4": {"family": "C", "arch": "modular", "n_fit": 8, "greedy": False,
           "shaping": None, "fit1hot": True},
    "B0": {"family": "B", "arch": "modular", "n_fit": 8, "greedy": False,
           "shaping": None, "fit1hot": False},
    "B1": {"family": "B", "arch": "mono", "n_fit": 8, "greedy": False,
           "shaping": None, "fit1hot": False},
    "B2": {"family": "B", "arch": "modular", "n_fit": 8, "greedy": False,
           "shaping": None, "fit1hot": True},
}
JOB_INDEX = {j: i for i, j in enumerate(sorted(JOBS))}
FAM_K = {"B": 1, "C": 2}
assert HORIZON == Arena.MAX_STEPS


def diag_rng(seed, job, stream):
    return np.random.default_rng([seed, JOB_INDEX[job], stream])


class C3Runner(TaskRunner):
    """Frozen C task + delivery-staged shaping (diag-only manipulation).

    +0.5 once on first contact (pickup staging); +1.0 per dwell step while
    1 <= dwell < 5 (delivery ramp); frozen +10 at 5-dwell unchanged.
    Success definition, obs, actions, geometry untouched.
    """

    def reset(self, family, cfg, seed):
        obs = super().reset(family, cfg, seed)
        self._contacted = False
        self._extra = 0.0
        return obs

    def step(self, a):
        obs, r, done, info = super().step(a)
        e = self.env
        if self.family == "C" and not done:
            ao = float(np.linalg.norm(e.obj_pos - e.agent_pos))
            if ao < 6.0 and not self._contacted:
                r += 0.5
                self._extra += 0.5
                self._contacted = True
            if 1 <= self.dwell < 5:
                r += 1.0
                self._extra += 1.0
        return obs, float(r), done, info


def diag_rollout(theta_full, arch, family, cfg, ep_seed, counter, skill_only,
                 greedy=False, runner_cls=TaskRunner, xcounter=None):
    """Mirror of frozen train.rollout; diffs: argmax iff greedy, C3Runner
    iff shaping (extra reward tracked in xcounter). Accounting identical."""
    from models import ModularNet, MonoNet
    runner = runner_cls()
    rng = np.random.default_rng(ep_seed)
    obs = runner.reset(family, cfg, int(ep_seed))
    net = ModularNet(theta_full) if arch == "modular" else MonoNet(theta_full)
    total, done, steps = 0.0, False, 0
    while not done:
        if arch == "modular":
            p, _ = net.probs(obs, skill_only=skill_only)
        else:
            p = net.probs(obs)
        a = int(np.argmax(p)) if greedy else int(rng.choice(len(p), p=p))
        obs, r, done, info = runner.step(a)
        total += r
        steps += 1
    assert steps <= HORIZON, (steps, HORIZON)
    counter[0] += HORIZON
    if xcounter is not None and isinstance(runner, C3Runner):
        xcounter[0] += runner._extra
    return total, bool(info["success"])


def diag_check_rate(theta_full, arch, family, entries, rng, counter,
                    skill_only, greedy=False, runner_cls=TaskRunner):
    c, ok = counter, 0
    for _ in range(CHECK_EPS):
        e = entries[int(rng.integers(len(entries)))]
        _, s = diag_rollout(theta_full, arch, family, e["cfg"],
                            int(rng.integers(2 ** 31)), c, skill_only,
                            greedy, runner_cls)
        ok += s
    return ok / CHECK_EPS


def run_phase(theta0_full, sub, arch, family, entries, rng_es, rng_ep,
              rng_ev, counter, skill_only, n_fit, greedy, runner_cls,
              generations, fit_skill_only=None):
    """Mirrors frozen run_es_phase exactly when fit_skill_only=None (v1:
    fitness on the FULL gated net, checks one-hot for frozen modular).
    fit_skill_only=k gives one-hot fitness (C4/B2 alignment test)."""
    base = theta0_full.copy()
    xcounter = [0]

    def fitness(sub_theta):
        full = base.copy()
        full[sub] = sub_theta
        rets = []
        for _ in range(n_fit):
            e = entries[int(rng_ep.integers(len(entries)))]
            ret, _ = diag_rollout(full, arch, family, e["cfg"],
                                  int(rng_ep.integers(2 ** 31)), counter,
                                  fit_skill_only, greedy, runner_cls,
                                  xcounter)
            rets.append(ret)
        return float(np.mean(rets))

    theta_sub = base[sub].copy()
    history, checks, snaps = [], [], {}

    def cb(g, th, best):
        history.append(best)
        if (g + 1) % 50 == 0:
            full = base.copy()
            full[sub] = th
            snaps[f"g{g + 1:03d}"] = full.copy()
        if (g + 1) % 10 == 0:
            full = base.copy()
            full[sub] = th
            checks.append(diag_check_rate(full, arch, family, entries,
                                          rng_ev, counter, skill_only,
                                          greedy, runner_cls))
        return False

    res = train_es(theta_sub, fitness, rng_es, pop_pairs=POP_PAIRS,
                   sigma=SIGMA, alpha=ALPHA, generations=generations,
                   progress_cb=cb)
    full = base.copy()
    full[sub] = res["theta"]
    return full, res["generations"], history, checks, snaps, xcounter[0]


def run_job(job, seed, data_dir, out_dir, generations=GEN_BUDGET):
    assert job in JOBS and seed in DIAG_SEEDS
    cfg = JOBS[job]
    fam, arch = cfg["family"], cfg["arch"]
    train_entries = [e for e in json.load(
        open(f"{data_dir}/train_meta_v1.json")) if e["family"] == fam]
    battery = [e for e in json.load(
        open(f"{data_dir}/test_battery_v1.json")) if e["battery"] == fam]
    assert len(battery) == 200 and len(train_entries) == 400
    rng_init = diag_rng(seed, job, 0)
    if arch == "mono":
        theta0 = xavier_init(rng_init, MONO_SHAPES)
        sub = slice(0, len(theta0))
        skill_only = None
    else:
        # v1-identical init structure: 4 Xavier slices in index order (the
        # trained slice occupies its v1 stream position), gate zeros. The
        # ISOLATION delta vs v1 is that co-slices stay Xavier-fresh (v1 had
        # trained past-phase slices there); no sequential context by design.
        parts = [xavier_init(rng_init, SKILL_SHAPES)
                 for _ in range(N_SKILLS)]
        parts.append(np.zeros(GATE_DIM))
        theta0 = np.concatenate(parts)
        sub = skill_slice(FAM_K[fam])
        skill_only = FAM_K[fam]
    runner_cls = C3Runner if cfg["shaping"] else TaskRunner
    counter = [0]
    t0 = time.time()
    fit_so = skill_only if cfg["fit1hot"] else None
    theta, gens, hist, checks, snaps, xtra = run_phase(
        theta0, sub, arch, fam, train_entries, diag_rng(seed, job, 1),
        diag_rng(seed, job, 2), diag_rng(seed, job, 3), counter, skill_only,
        cfg["n_fit"], cfg["greedy"], runner_cls, generations, fit_so)
    wall = time.time() - t0
    expect = (generations * (2 * POP_PAIRS) * cfg["n_fit"] * HORIZON
              + (generations // 10) * CHECK_EPS * HORIZON)
    assert counter[0] == expect, (counter[0], expect)
    fp = first_pass_gen(checks)
    held = {}
    for mode in ("sampled", "greedy"):
        pol = make_policy(arch, theta, "normal" if mode == "sampled" else "greedy")
        rep = run_battery(pol, battery)
        held[mode] = {"n": rep["n"], "sr": rep["success_rate"], "ci": rep["ci95"]}
    os.makedirs(out_dir, exist_ok=True)
    np.savez(f"{out_dir}/{job}_{seed}_theta.npz", theta=theta)
    np.savez(f"{out_dir}/{job}_{seed}_snaps.npz", **snaps)
    log = {"job": job, "seed": seed, "config": cfg, "gens": gens,
           "history": hist, "checks": checks, "first_pass_gen": fp,
           "acquired": bool(fp is not None), "steps": counter[0],
           "shaping_extra": xtra, "wall_s": wall, "heldout": held,
           "train_entries": "train_meta_v1.json", "battery": "test_battery_v1.json"}
    json.dump(log, open(f"{out_dir}/{job}_{seed}.json", "w"), indent=1)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", choices=sorted(JOBS), required=True)
    ap.add_argument("--seed", type=int, choices=DIAG_SEEDS, required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--gens", type=int, default=GEN_BUDGET)
    args = ap.parse_args()
    log = run_job(args.job, args.seed, args.data, args.out, args.gens)
    print(json.dumps({k: v for k, v in log.items()
                      if k not in ("history", "checks")}, indent=1))


if __name__ == "__main__":
    main()
