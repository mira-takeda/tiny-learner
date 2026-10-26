"""Diagnostic helpers (toy-only). Exact replica of selftest's vec check.

NOT imported by the registered driver (train.py). Used to discriminate:
value-dependent divergence vs in-process context dependence vs flakes.
"""

import numpy as np

from models import (N_SKILLS, SKILL_SHAPES, GATE_SHAPES, MONO_SHAPES,
                    ModularNet, MonoNet)
from policies import xavier_init
from selftest import TOY_CFGS
from tasks import TaskRunner, VecTaskRunner
from train import rollout
from vec import run_batch

FAMS = ("A", "B", "H1")
EP_SEEDS = (200, 201, 202, 203)


def build_thetas():
    vth = np.concatenate(
        [xavier_init(np.random.default_rng(31), SKILL_SHAPES)
         for _ in range(N_SKILLS)]
        + [xavier_init(np.random.default_rng(31), GATE_SHAPES)])
    vmono = xavier_init(np.random.default_rng(32), MONO_SHAPES)
    return {"modular": vth, "mono": vmono}


def exact_vec_pass():
    """One exact replica of selftest's vec-exact-vs-scalar check."""
    thetas = build_thetas()
    for fam in FAMS:
        for arch in ("modular", "mono"):
            th = thetas[arch]
            items = [(TOY_CFGS[fam], s) for s in EP_SEEDS]
            vr, vs = run_batch(th, arch, fam, items, backend="scalar")
            for k, (c, s) in enumerate(items):
                r0, s0 = rollout(th, arch, fam, c, s, [0])
                if not (r0 == vr[k] and s0 == vs[k]):
                    return False, {"family": fam, "arch": arch, "ep": k,
                                   "seed": s, "ret_scalar": float(r0),
                                   "ret_vec": float(vr[k]),
                                   "suc_scalar": bool(s0),
                                   "suc_vec": bool(vs[k]),
                                   "ret_abs_diff": float(abs(r0 - vr[k]))}
    return True, None


def lockstep_first_div(family, arch, cfg, seed, theta):
    """Same actions to both runners; first differing step/quantity/magnitude."""
    s_tr, v_tr = TaskRunner(), VecTaskRunner(1)
    r2 = np.random.default_rng(seed)
    net = ModularNet(theta) if arch == "modular" else MonoNet(theta)
    o1 = s_tr.reset(family, cfg, seed)
    v_tr.reset(family, [cfg], [seed])
    t = 0
    while t < 200:
        if arch == "modular":
            p, _ = net.probs(o1)
        else:
            p = net.probs(o1)
        a = int(r2.choice(len(p), p=p))
        o1, r1, done1, i1 = s_tr.step(a)
        oo, rr, dd, ss = v_tr.step(np.array([a]))
        t += 1
        cands = [("obs", o1, oo[0]),
                 ("agent_pos", s_tr.env.agent_pos, v_tr.env.agent_pos[0]),
                 ("agent_vel", s_tr.env.agent_vel, v_tr.env.agent_vel[0]),
                 ("reward", np.array(r1), np.array(float(rr[0])))]
        if s_tr.env.haz_pos is not None:
            cands.append(("haz_pos", s_tr.env.haz_pos, v_tr.env.haz_pos[0]))
        if s_tr.env.obj_pos is not None:
            cands.append(("obj_pos", s_tr.env.obj_pos, v_tr.env.obj_pos[0]))
            cands.append(("obj_vel", s_tr.env.obj_vel, v_tr.env.obj_vel[0]))
        for name, x, y in cands:
            if not np.array_equal(x, y):
                return {"step": t, "quantity": name,
                        "max_abs_diff": float(np.abs(x - y).max())}
        if done1 != bool(dd[0]) or i1["success"] != bool(ss[0]):
            return {"step": t, "quantity": "done/success", "max_abs_diff": 1.0}
        if done1:
            break
    return None
