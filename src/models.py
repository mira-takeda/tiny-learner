"""Modular skills + gate, monolithic baseline. Flat ES-optimized parameters."""

import numpy as np

from policies import (layer_shapes, count_params, xavier_init, mlp_forward,
                      softmax, act_sample, act_greedy)

OBS_DIM = 14
N_ACTIONS = 9
N_SKILLS = 4

SKILL_SHAPES = layer_shapes(OBS_DIM, [32, 32], N_ACTIONS)
GATE_SHAPES = layer_shapes(OBS_DIM, [32], N_SKILLS)
SKILL_DIM = count_params(SKILL_SHAPES)
GATE_DIM = count_params(GATE_SHAPES)
MODULAR_DIM = N_SKILLS * SKILL_DIM + GATE_DIM

MONO_WIDTH = 78
MONO_SHAPES = layer_shapes(OBS_DIM, [MONO_WIDTH, MONO_WIDTH], N_ACTIONS)
MONO_DIM = count_params(MONO_SHAPES)


def verify_param_match(tol=0.05):
    rel = abs(MONO_DIM - MODULAR_DIM) / MODULAR_DIM
    assert rel <= tol, (MONO_DIM, MODULAR_DIM, rel)
    return {"modular": MODULAR_DIM, "monolith": MONO_DIM, "rel_diff": rel}


def split_modular(theta):
    skills = [theta[k * SKILL_DIM:(k + 1) * SKILL_DIM] for k in range(N_SKILLS)]
    return skills, theta[N_SKILLS * SKILL_DIM:]


def modular_probs(theta, obs, skill_only=None, gate_theta=None, mask_hazard=False):
    if mask_hazard:
        obs = obs.copy()
        obs[5:8] = 0.0
    skills, gate = split_modular(theta)
    if gate_theta is not None:
        gate = gate_theta
    if skill_only is None:
        w = softmax(mlp_forward(gate, obs, GATE_SHAPES))
    else:
        w = np.zeros(N_SKILLS)
        w[skill_only] = 1.0
    p = np.zeros(N_ACTIONS)
    for k in range(N_SKILLS):
        if w[k] > 0:
            p += w[k] * softmax(mlp_forward(skills[k], obs, SKILL_SHAPES))
    return p, w


def modular_act(theta, obs, rng, deterministic=False, **kw):
    p, w = modular_probs(theta, obs, **kw)
    if deterministic:
        return int(np.argmax(p)), w
    return int(rng.choice(len(p), p=p)), w


def mono_probs(theta, obs, mask_hazard=False):
    if mask_hazard:
        obs = obs.copy()
        obs[5:8] = 0.0
    return softmax(mlp_forward(theta, obs, MONO_SHAPES))


def mono_act(theta, obs, rng, deterministic=False, **kw):
    p = mono_probs(theta, obs, **kw)
    if deterministic:
        return int(np.argmax(p))
    return int(rng.choice(len(p), p=p))
