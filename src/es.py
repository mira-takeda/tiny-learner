"""Evolution Strategies (antithetic, centered-rank shaping). Numpy only."""

import numpy as np


def train_es(
    theta0,
    fitness_fn,
    rng,
    pop_pairs=32,
    sigma=0.1,
    alpha=0.05,
    generations=400,
    progress_cb=None,
):
    theta = theta0.copy()
    n = theta.size
    hist = []
    g = 0
    for g in range(generations):
        eps = rng.standard_normal((pop_pairs, n))
        fits = np.empty(2 * pop_pairs)
        for i in range(pop_pairs):
            fits[2 * i] = fitness_fn(theta + sigma * eps[i])
            fits[2 * i + 1] = fitness_fn(theta - sigma * eps[i])
        order = np.argsort(fits)
        utils = np.empty_like(fits)
        utils[order] = np.arange(2 * pop_pairs) / (2 * pop_pairs - 1) - 0.5
        grad = np.zeros(n)
        for i in range(pop_pairs):
            grad += (utils[2 * i] - utils[2 * i + 1]) * eps[i]
        grad /= pop_pairs
        theta = theta + (alpha / sigma) * grad
        hist.append(float(fits.max()))
        if progress_cb is not None and progress_cb(g, theta, hist[-1]):
            break
    return {"theta": theta, "history": hist, "generations": g + 1}
