"""Scripted oracle controllers for the C/B diagnostic audits (diag-only).

Privileged oracles use exact simulator state (solvability upper bound);
obs-oracles use ONLY the frozen 14-dim obs (obs-sufficiency test).
Deterministic; no learning; frozen code untouched (TaskRunner used as-is).

Obs layout (frozen): [0:2] agent_vel/5, [2:5] goal slot, [5:8] hazard slot,
[8:11] obj slot, [11:14] zone slot; slot = [ux, uy, dist/DMAX].
Actions 0-7: compass accel impulses (k=0 -> +x, CCW); 8: coast.
"""

import numpy as np

DIRS = np.array([(np.cos(k * np.pi / 4.0), np.sin(k * np.pi / 4.0))
                 for k in range(8)])
DMAX = 100.0 * np.sqrt(2.0)
VMAX = 5.0


def dir_to_action(vec, coast_thresh=0.15):
    n = float(np.linalg.norm(vec))
    if n < coast_thresh:
        return 8
    return int(np.argmax(DIRS @ (vec / n)))


def slot_rel(slot):
    return np.array([slot[0], slot[1]]) * (slot[2] * DMAX)


def random_policy(obs, runner, rng):
    return int(rng.integers(9))


def still_policy(obs, runner, rng):
    return 8


# ---------------- C transport ----------------

def _capped(vec, cap, av):
    sp = float(np.linalg.norm(av))
    if sp > cap:
        return -av  # brake first
    return vec


def _tap_drive(runner, target_rel, av, ao_rel, dao, cool_len=7):
    cd = getattr(runner, "_tap_cd", 0)
    if cd > 0:  # cooling: back off first, then brake/coast
        runner._tap_cd = cd - 1
        if cd > cool_len - 2:
            return dir_to_action(-ao_rel)
        spd = float(np.linalg.norm(av))
        return 8 if spd < 0.4 else dir_to_action(-av)
    if dao < 6.0:
        runner._tap_cd = cool_len  # tap now, back off next
    return dir_to_action(target_rel)


def c_privileged_action(obs, runner, rng=None):
    e = runner.env
    ap, op, zn = e.agent_pos, e.obj_pos, e.zone
    ao = op - ap
    oz = zn - op
    doz = float(np.linalg.norm(oz))
    dao = float(np.linalg.norm(ao))
    ov = e.obj_vel
    av = e.agent_vel
    if doz < 8.0:  # dwell: break contact, brake self, hold clear
        if dao < 7.0:
            return dir_to_action(-ao)  # back off first
        if float(np.linalg.norm(ov)) >= 2.5 and dao < 12.0:
            return dir_to_action(-ov)  # fast obj escaping: intercept-oppose
        if float(np.linalg.norm(av)) < 0.5:
            return 8
        return dir_to_action(-av)
    if doz < 35.0:  # slow creep: keep obj slow enough to dwell on arrival
        cap = 1.0 if doz < 15.0 else 2.0
        u = oz / max(doz, 1e-9)
        behind = op - u * 5.0
        tgt = behind - ap if float(np.linalg.norm(ap - behind)) > 3.0 else zn - ap
        return dir_to_action(_capped(tgt, cap, av))
    if dao > 5.5:
        return dir_to_action(ao)  # far: full-speed approach/push
    u = oz / max(doz, 1e-9)
    behind = op - u * 5.0
    if float(np.linalg.norm(ap - behind)) > 3.0:
        return dir_to_action(behind - ap)
    return dir_to_action(zn - ap)


def c_obs_action(obs, runner, rng=None):
    obj_rel = slot_rel(obs[8:11])
    zone_rel = slot_rel(obs[11:14])
    oz_vec = zone_rel - obj_rel
    doz = float(np.linalg.norm(oz_vec))
    dao = float(np.linalg.norm(obj_rel))
    av = obs[0:2] * VMAX
    if doz < 8.0:  # dwell without obj velocity: break contact, hold
        if dao < 7.0:
            return dir_to_action(-obj_rel)
        if float(np.linalg.norm(av)) < 0.5:
            return 8
        return dir_to_action(-av)
    if doz < 35.0:
        cap = 1.0 if doz < 15.0 else 2.0
        u = oz_vec / max(doz, 1e-9)
        behind_rel = obj_rel - u * 5.0
        tgt = behind_rel if float(np.linalg.norm(behind_rel)) > 3.0 else zone_rel
        return dir_to_action(_capped(tgt, cap, av))
    if dao > 5.5:
        return dir_to_action(obj_rel)
    u = oz_vec / max(doz, 1e-9)
    behind_rel = obj_rel - u * 5.0
    if float(np.linalg.norm(behind_rel)) > 3.0:
        return dir_to_action(behind_rel)
    return dir_to_action(zone_rel)


# ---------------- B avoidance ----------------

def b_oracle_action(obs, runner, rng=None):
    haz_rel = slot_rel(obs[5:8])
    n = float(np.linalg.norm(haz_rel))
    if n < 1e-9:
        return 8
    return dir_to_action(-haz_rel / n)  # pure flee (obs-only)


def b_priv_action(obs, runner, rng=None):
    e = runner.env
    flee = e.agent_pos - e.haz_pos
    n = float(np.linalg.norm(flee))
    if n < 1e-9:
        return 8
    acc = flee / n
    ap = e.agent_pos  # wall repulsion + center pull (privileged bounds)
    for i in (0, 1):
        if ap[i] < 25.0:
            u = np.zeros(2)
            u[i] = (25.0 - ap[i]) / 25.0
            acc = acc + 1.5 * u
        if ap[i] > 75.0:
            u = np.zeros(2)
            u[i] = -(ap[i] - 75.0) / 25.0
            acc = acc + 1.5 * u
    acc = acc + 0.2 * ((np.array([50.0, 50.0]) - ap) / 70.0)
    return dir_to_action(acc)


def b_obs_orbit_action(obs, runner, rng=None):
    haz_rel = slot_rel(obs[5:8])
    n = float(np.linalg.norm(haz_rel))
    if n < 1e-9:
        return 8
    flee = -haz_rel / n
    perp = np.array([-flee[1], flee[0]])
    return dir_to_action(flee + 0.5 * perp)  # flee + orbit bias
