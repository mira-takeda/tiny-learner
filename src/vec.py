"""Batched episode runner over VecArena/VecTaskRunner.

Backend "scalar": per-env policy calls (bit-exact vs scalar rollout).
Backend "batched": batched policy forward (fast; characterized divergence).
Per-env RNG streams match scalar rollout exactly in both backends.
"""

import numpy as np

from models import (ModularNet, MonoNet, batched_modular_probs,
                    batched_mono_probs, N_SKILLS)
from tasks import VecTaskRunner


def run_batch(theta, arch, family, items, backend="scalar", skill_only=None,
              gate_mats=None, mask_hazard=False, max_steps=200):
    """items: [(cfg, seed)...]. Returns (returns[N], success[N])."""
    n = len(items)
    cfgs = [c for c, _ in items]
    seeds = [s for _, s in items]
    runner = VecTaskRunner(n)
    obs = runner.reset(family, cfgs, seeds)
    arngs = [np.random.default_rng(s) for s in seeds]
    if arch == "modular":
        net = ModularNet(theta)
    else:
        net = MonoNet(theta)
    for _ in range(max_steps):
        if not runner.active.any():
            break
        acts = np.zeros(n, dtype=int)
        if backend == "scalar":
            for i in np.flatnonzero(runner.active):
                if arch == "modular":
                    p, _ = net.probs(obs[i], skill_only=skill_only,
                                     gate_mats=gate_mats,
                                     mask_hazard=mask_hazard)
                else:
                    p = net.probs(obs[i], mask_hazard=mask_hazard)
                acts[i] = int(arngs[i].choice(len(p), p=p))
        else:
            if arch == "modular":
                P, _ = batched_modular_probs(
                    net.skill_mats, net.gate_mats, obs, skill_only=skill_only,
                    gate_mats_override=gate_mats, mask_hazard=mask_hazard)
            else:
                P = batched_mono_probs(net.mats, obs, mask_hazard=mask_hazard)
            for i in np.flatnonzero(runner.active):
                acts[i] = int(arngs[i].choice(len(P[i]), p=P[i]))
        obs, _, _, _ = runner.step(acts)
    return runner.rets.copy(), runner.success.copy()
