"""Flat-parameter MLP policies. Numpy only."""

import numpy as np


def layer_shapes(in_dim, hidden, out_dim):
    dims = [in_dim] + list(hidden) + [out_dim]
    shapes = []
    for a, b in zip(dims[:-1], dims[1:]):
        shapes.append((a, b))  # W
        shapes.append((b,))  # b
    return shapes


def count_params(shapes):
    return sum(int(np.prod(s)) for s in shapes)


def xavier_init(rng, shapes):
    parts = []
    for s in shapes:
        if len(s) == 2:
            lim = np.sqrt(6.0 / (s[0] + s[1]))
            parts.append(rng.uniform(-lim, lim, s))
        else:
            parts.append(np.zeros(s))
    return np.concatenate([p.ravel() for p in parts])


def unpack(theta, shapes):
    out, k = [], 0
    for s in shapes:
        n = int(np.prod(s))
        out.append(theta[k : k + n].reshape(s))
        k += n
    return out


def mlp_forward(theta, x, shapes):
    mats = unpack(theta, shapes)
    h = x
    for i in range(0, len(mats) - 2, 2):
        h = np.tanh(h @ mats[i] + mats[i + 1])
    return h @ mats[-2] + mats[-1]


def softmax(logits):
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def act_sample(rng, theta, obs, shapes):
    p = softmax(mlp_forward(theta, obs, shapes))
    return int(rng.choice(len(p), p=p))


def act_greedy(theta, obs, shapes):
    return int(np.argmax(mlp_forward(theta, obs, shapes)))
