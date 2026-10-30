"""Instrumented episode runner for audits (diag-only, frozen TaskRunner inside).

Recomputes the frozen reward decomposition from consecutive states and
asserts it matches the frozen scalar (validates the audit, not the env).
Per-episode summary covers return + components, pickup/transport/delivery,
success/timeout, geometry, and action histograms.
"""

import sys

import numpy as np

sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/src")
from tasks import TaskRunner  # noqa: E402  (frozen)


def run_episode(family, cfg, seed, policy_fn, rng, trace=False):
    r = TaskRunner()
    obs = r.reset(family, cfg, int(seed))
    e = r.env
    steps = []
    ao0 = float(np.linalg.norm(e.obj_pos - e.agent_pos)) if e.obj_pos is not None else None
    oz0 = float(np.linalg.norm(e.obj_pos - e.zone)) if e.obj_pos is not None else None
    prev_oz, prev_ao = oz0, ao0
    hist = [e.agent_pos.copy()]
    dwell_max, contact_steps = 0, 0
    oz_min, ao_min = oz0, ao0
    haz_min = None
    gate_steps = 0
    comp = {"penalty": 0.0, "bonus": 0.0, "other": 0.0, "c_oz": 0.0,
            "c_ao": 0.0, "b_gate": 0.0, "b_margin": 0.0, "b_death": 0.0}
    acts = [0] * 9
    total, done, info = 0.0, False, {"success": False}
    while not done:
        a = policy_fn(obs, r, rng)
        acts[int(a)] += 1
        obs, rew, done, info = r.step(int(a))
        total += rew
        if family == "C":
            comp["penalty"] += -0.01
            oz = float(np.linalg.norm(e.obj_pos - e.zone))
            ao = float(np.linalg.norm(e.obj_pos - e.agent_pos))
            comp["c_oz"] += 0.1 * (prev_oz - oz)
            comp["c_ao"] += 0.02 * (prev_ao - ao)
            prev_oz, prev_ao = oz, ao
            oz_min = min(oz_min, oz)
            ao_min = min(ao_min, ao)
            if ao < 6.0:
                contact_steps += 1
            dwell_max = max(dwell_max, r.dwell)
            if info["success"]:
                comp["bonus"] += 10.0
        elif family == "B":
            dh = float(np.linalg.norm(e.agent_pos - e.haz_pos))
            haz_min = dh if haz_min is None else min(haz_min, dh)
            if rew == -10.0:
                comp["b_death"] += -10.0  # death replaces the step reward
            else:
                comp["penalty"] += -0.01
                hist.append(e.agent_pos.copy())
                ref = hist[0] if len(hist) < 40 else hist[-40]
                if float(np.linalg.norm(e.agent_pos - ref)) >= 6.0:
                    comp["b_gate"] += 0.05
                    gate_steps += 1
                comp["b_margin"] += min(dh, 30.0) / 30.0 * 0.05
        steps.append(1)
        if trace:
            pass
    t = len(steps)
    if family == "C":
        expect = comp["penalty"] + comp["c_oz"] + comp["c_ao"] + comp["bonus"]
        assert abs(expect - total) < 1e-6, (expect, total)
    elif family == "B":
        expect = (comp["penalty"] + comp["b_gate"] + comp["b_margin"]
                  + comp["b_death"] + comp["bonus"])
        assert abs(expect - total) < 1e-6, (expect, total)
    return {"success": bool(info["success"]), "T": t,
            "timeout": bool(t >= 200 and not info["success"]),
            "return": float(total), "comp": {k: float(v) for k, v in comp.items()},
            "ao0": ao0, "oz0": oz0, "ao_min": ao_min, "oz_min": oz_min,
            "contact_ever": bool(contact_steps > 0),
            "contact_steps": contact_steps, "dwell_max": dwell_max,
            "transport": (oz0 - oz_min) if oz0 is not None else None,
            "gate_frac": gate_steps / t, "haz_min": haz_min,
            "actions": acts}


def summarize(eps):
    n = len(eps)
    out = {"n": n, "sr": sum(e["success"] for e in eps) / n,
           "mean_return": sum(e["return"] for e in eps) / n,
           "mean_T": sum(e["T"] for e in eps) / n,
           "timeout_frac": sum(e["timeout"] for e in eps) / n}
    for k in ("contact_ever",):
        out[k + "_frac"] = sum(e[k] for e in eps) / n
    for k in ("oz_min", "dwell_max", "transport", "gate_frac", "haz_min",
              "contact_steps", "ao0", "oz0"):
        xs = [e[k] for e in eps if e[k] is not None]
        if xs:
            out[k + "_mean"] = float(np.mean(xs))
            out[k + "_min"] = float(np.min(xs))
            out[k + "_max"] = float(np.max(xs))
    comps = {}
    for k in eps[0]["comp"]:
        comps[k] = float(np.mean([e["comp"][k] for e in eps]))
    out["comp_mean"] = comps
    ah = [0] * 9
    for e in eps:
        for i, c in enumerate(e["actions"]):
            ah[i] += c
    tot = max(sum(ah), 1)
    out["action_frac"] = [round(c / tot, 3) for c in ah]
    return out
