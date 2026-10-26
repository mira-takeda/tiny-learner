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


# Bound networks: identical math to *_probs, weights unpacked once per theta
# instead of once per step. Bit-identical outputs (covered by selftest).


def _forward_mats(mats, x):
    h = x
    for i in range(0, len(mats) - 2, 2):
        h = np.tanh(h @ mats[i] + mats[i + 1])
    return h @ mats[-2] + mats[-1]


class ModularNet:
    def __init__(self, theta):
        from policies import unpack
        skills, gate = split_modular(theta)
        self.skill_mats = [unpack(s, SKILL_SHAPES) for s in skills]
        self.gate_mats = unpack(gate, GATE_SHAPES)

    def probs(self, obs, skill_only=None, gate_mats=None, mask_hazard=False):
        if mask_hazard:
            obs = obs.copy()
            obs[5:8] = 0.0
        if skill_only is None:
            w = softmax(_forward_mats(gate_mats or self.gate_mats, obs))
        else:
            w = np.zeros(N_SKILLS)
            w[skill_only] = 1.0
        p = np.zeros(N_ACTIONS)
        for k in range(N_SKILLS):
            if w[k] > 0:
                p += w[k] * softmax(_forward_mats(self.skill_mats[k], obs))
        return p, w


class MonoNet:
    def __init__(self, theta):
        from policies import unpack
        self.mats = unpack(theta, MONO_SHAPES)

    def probs(self, obs, mask_hazard=False):
        if mask_hazard:
            obs = obs.copy()
            obs[5:8] = 0.0
        return softmax(_forward_mats(self.mats, obs))


def _forward_mats_batch(mats, X):
    H = X
    for i in range(0, len(mats) - 2, 2):
        H = np.tanh(H @ mats[i] + mats[i + 1])
    return H @ mats[-2] + mats[-1]


def _softmax_batch(L):
    Z = L - L.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def batched_modular_probs(skill_mats_list, gate_mats, X, skill_only=None,
                          gate_mats_override=None, mask_hazard=False):
    """Batched ModularNet.probs. Row n must equal the scalar call on X[n]."""
    if mask_hazard:
        X = X.copy()
        X[:, 5:8] = 0.0
    if skill_only is None:
        W = _softmax_batch(_forward_mats_batch(
            gate_mats_override or gate_mats, X))
    else:
        W = np.zeros((X.shape[0], N_SKILLS))
        W[:, skill_only] = 1.0
    P0 = _softmax_batch(_forward_mats_batch(skill_mats_list[0], X))
    P1 = _softmax_batch(_forward_mats_batch(skill_mats_list[1], X))
    P2 = _softmax_batch(_forward_mats_batch(skill_mats_list[2], X))
    P3 = _softmax_batch(_forward_mats_batch(skill_mats_list[3], X))
    P = (W[:, 0:1] * P0 + W[:, 1:2] * P1 + W[:, 2:3] * P2 + W[:, 3:4] * P3)
    return P, W


def batched_mono_probs(mats, X, mask_hazard=False):
    if mask_hazard:
        X = X.copy()
        X[:, 5:8] = 0.0
    return _softmax_batch(_forward_mats_batch(mats, X))
