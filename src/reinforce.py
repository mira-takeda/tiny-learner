"""Discrete REINFORCE fallback (PREREG 6). Numpy only. Analytic backprop."""

import numpy as np

from policies import unpack, softmax


def _forward(theta, x, shapes):
    mats = unpack(theta, shapes)
    acts, pres = [x], []
    h = x
    for i in range(0, len(mats) - 2, 2):
        z = h @ mats[i] + mats[i + 1]
        pres.append(z)
        h = np.tanh(z)
        acts.append(h)
    logits = h @ mats[-2] + mats[-1]
    return acts, pres, mats, logits


def _backward(acts, pres, mats, dlog):
    n_hidden = (len(mats) - 2) // 2
    grads = [None] * len(mats)
    grads[-2] = np.outer(acts[-1], dlog)
    grads[-1] = dlog.copy()
    dh = mats[-2] @ dlog
    for j in range(n_hidden - 1, -1, -1):
        dz = dh * (1.0 - np.tanh(pres[j]) ** 2)
        grads[2 * j] = np.outer(acts[j], dz)
        grads[2 * j + 1] = dz.copy()
        dh = mats[2 * j] @ dz
    return np.concatenate([g.ravel() for g in grads])


def logp_grad(theta, obs, action, shapes):
    acts, pres, mats, logits = _forward(theta, obs, shapes)
    p = softmax(logits)
    dlog = -p.copy()
    dlog[action] += 1.0
    return _backward(acts, pres, mats, dlog)


def entropy_grad(theta, obs, shapes):
    acts, pres, mats, logits = _forward(theta, obs, shapes)
    p = softmax(logits)
    logp = np.log(p + 1e-12)
    h = float(-np.sum(p * logp))
    return _backward(acts, pres, mats, p * (-logp - h))


class Adam:
    def __init__(self, n, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.m = np.zeros(n)
        self.v = np.zeros(n)
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.t = 0

    def update(self, theta, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad * grad
        mh = self.m / (1 - self.b1**self.t)
        vh = self.v / (1 - self.b2**self.t)
        return theta + self.lr * mh / (np.sqrt(vh) + self.eps)


def train_re(
    theta0,
    shapes,
    rollout_fn,
    rng,
    episode_cap=14400,
    batch_eps=8,
    gamma=0.99,
    lr=0.01,
    entropy=0.01,
    baseline_rate=0.05,
    progress_cb=None,
):
    theta = theta0.copy()
    opt = Adam(theta.size, lr=lr)
    baseline, ep = 0.0, 0
    while ep < episode_cap:
        batch_grad = np.zeros_like(theta)
        batch_ret = []
        for _ in range(batch_eps):
            if ep >= episode_cap:
                break
            ep_rng = np.random.default_rng(int(rng.integers(2**31)))
            obs, acts, rews = rollout_fn(theta, ep_rng)
            ep += 1
            advs = np.empty(len(rews))
            g = 0.0
            for t in range(len(rews) - 1, -1, -1):
                g = rews[t] + gamma * g
                advs[t] = g - baseline
            for t in range(len(rews)):
                batch_grad += advs[t] * logp_grad(theta, obs[t], acts[t], shapes)
                batch_grad += entropy * entropy_grad(theta, obs[t], shapes)
            batch_ret.append(float(sum(rews)))
        if not batch_ret:
            break
        batch_grad /= len(batch_ret)
        theta = opt.update(theta, batch_grad)
        mean_ret = float(np.mean(batch_ret))
        baseline += baseline_rate * (mean_ret - baseline)
        if progress_cb is not None and progress_cb(ep, theta, mean_ret):
            break
    return {"theta": theta, "episodes": ep, "mean_ret": mean_ret}
