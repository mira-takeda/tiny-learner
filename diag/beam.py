"""Beam-search oracle over the frozen simulator (diag-only).

Episodes are deterministic given the action sequence (C: no stochasticity;
B: hazard noise from a seeded rng, saved/restored with the state), so beam
search with the TRUE model reports the best success observed by THIS
searcher (width 48, heuristic-guided). It is NOT exhaustive and proves NO
optimality bound: treat its rate as a strong-controller reference, never as
an environment ceiling. Used on small stratified samples (expensive).

State save/restore covers Arena + TaskRunner trackers + env rng state.
"""

import sys

import numpy as np

sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/src")
from tasks import TaskRunner  # noqa: E402  (frozen)


def _get_state(r):
    e = r.env
    return {"ap": e.agent_pos.copy(), "av": e.agent_vel.copy(),
            "hp": None if e.haz_pos is None else e.haz_pos.copy(),
            "op": None if e.obj_pos is None else e.obj_pos.copy(),
            "ov": e.obj_vel.copy(), "t": e.t,
            "rng": e.rng.bit_generator.state,
            "pgoal": r.prev_goal, "poz": r.prev_oz, "pao": r.prev_ao,
            "dwell": r.dwell, "hist": [h.copy() for h in r.hist],
            "touched": r.touched}


def _set_state(r, s):
    e = r.env
    e.agent_pos = s["ap"].copy()
    e.agent_vel = s["av"].copy()
    e.haz_pos = None if s["hp"] is None else s["hp"].copy()
    e.obj_pos = None if s["op"] is None else s["op"].copy()
    e.obj_vel = s["ov"].copy()
    e.t = s["t"]
    e.rng.bit_generator.state = dict(s["rng"])
    r.prev_goal, r.prev_oz, r.prev_ao = s["pgoal"], s["poz"], s["pao"]
    r.dwell = s["dwell"]
    r.hist = [h.copy() for h in s["hist"]]
    r.touched = s["touched"]


def _heur_c(r, ret):
    e = r.env
    oz = float(np.linalg.norm(e.obj_pos - e.zone))
    ao = float(np.linalg.norm(e.obj_pos - e.agent_pos))
    return (r.dwell * 100.0 - oz * 2.0 - ao * 0.2 + ret * 0.5 - e.t * 0.01)


def _heur_b(r, ret):
    e = r.env
    dh = float(np.linalg.norm(e.agent_pos - e.haz_pos))
    ap = e.agent_pos
    wall = min(ap[0], ap[1], 100.0 - ap[0], 100.0 - ap[1])
    return (ret * 1.0 + dh * 0.3 + wall * 0.8 + e.t * 0.02)


def beam_episode(family, cfg, seed, width=64, max_steps=200):
    """Return (success, actions, return) via beam search. Deterministic."""
    heur = _heur_c if family == "C" else _heur_b
    r = TaskRunner()
    r.reset(family, cfg, int(seed))
    s0 = _get_state(r)
    beam = [(0.0, s0, [])]  # (heuristic, state, actions)
    best_ret = -1e18
    for depth in range(max_steps):
        cands = []
        for _, s, acts in beam:
            _set_state(r, s)
            for a in range(9):
                _set_state(r, s)
                _, rew, done, info = r.step(a)
                ret = 0.0  # recompute not needed; heuristic uses live state
                h = heur(r, 0.0)
                ns = _get_state(r)
                nacts = acts + [a]
                if info["success"]:
                    return True, nacts, None
                if done:
                    continue
                cands.append((h, ns, nacts))
        if not cands:
            return False, [], None
        cands.sort(key=lambda x: -x[0])
        beam = cands[:width]
    return False, [], None


def run_beam_battery(family, entries, width=64):
    """entries: list of {cfg, seed}. Returns {sr, n, wins}. Sequential."""
    wins = 0
    for i, e in enumerate(entries):
        ok, _, _ = beam_episode(family, e["cfg"], int(e["seed"]), width)
        wins += bool(ok)
        print(f"  beam {family} {i+1}/{len(entries)}: {'WIN' if ok else '---'}",
              flush=True)
    return {"sr": wins / len(entries), "n": len(entries), "wins": wins}
