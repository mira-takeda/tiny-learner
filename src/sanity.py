"""Optimizer sanity check (PREREG 6). LineWorld ONLY. Never touches registered tasks.

Usage:
  python3.14 sanity.py [--optimizer es|re] [--tag TEXT] [--smoke]

--smoke runs a short unregistered debug run. The formal run is the first clean
full-bar run of final code; label it with --tag FORMAL.
"""

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from env import LineWorld, Arena, random_arena_config
from policies import layer_shapes, count_params, xavier_init, act_sample
from es import train_es
from reinforce import train_re

SANITY_SEED = 777
INIT_SEED = 1234
TRAIN_SEED = 4242
BAR = 0.90
GEN_CAP = 150
EP_CAP = 14400
WALL_CAP_S = 20 * 60


def eval_seeds(n):
    return [
        int(s.generate_state(1, dtype=np.uint32)[0])
        for s in np.random.SeedSequence(SANITY_SEED).spawn(n)
    ]


def rollout(theta, shapes, ep_seed):
    env = LineWorld()
    rng = np.random.default_rng(ep_seed)
    obs = env.reset()
    O, A, R = [], [], []
    done = False
    while not done:
        a = act_sample(rng, theta, obs, shapes)
        O.append(obs)
        A.append(a)
        obs, r, done, info = env.step(a)
        R.append(r)
    return (O, A, R, info["success"])


def success_rate(theta, shapes, n_eps):
    ok = sum(rollout(theta, shapes, s)[3] for s in eval_seeds(n_eps))
    return ok / n_eps


def throughput():
    out = {}
    for name, make, steps in (("lineworld", LineWorld, 20000), ("arena", None, 20000)):
        rng = np.random.default_rng(7)
        t0 = time.time()
        if make is not None:
            env = make()
            obs = env.reset()
            for _ in range(steps):
                obs, _, done, _ = env.step(int(rng.integers(3)))
                if done:
                    obs = env.reset()
        else:
            env = Arena()
            cfg = random_arena_config(rng)
            obs = env.reset(cfg, 7)
            for i in range(steps):
                obs = env.step(int(rng.integers(9)))
                if env.t >= env.MAX_STEPS:
                    obs = env.reset(random_arena_config(rng), 7 + i)
        out[name] = steps / (time.time() - t0)
    return out


def run_es(shapes, smoke):
    rng = np.random.default_rng(TRAIN_SEED)
    theta0 = xavier_init(np.random.default_rng(INIT_SEED), shapes)
    fit_rng = np.random.default_rng(TRAIN_SEED + 1)

    def fitness(th):
        return float(
            np.mean(
                [
                    sum(rollout(th, shapes, int(fit_rng.integers(2**31)))[2])
                    for _ in range(2)
                ]
            )
        )

    n_eval = 20 if smoke else 100
    state = {"passes": 0}

    def cb(g, th, best):
        if (g + 1) % 5 == 0 or smoke:
            sr = success_rate(th, shapes, n_eval)
            print(f"  [es] gen={g + 1} best_fit={best:.2f} sr={sr:.2f}", flush=True)
            state["passes"] = state["passes"] + 1 if sr >= BAR else 0
            if state["passes"] >= 2:
                return True
        return False

    t0 = time.time()
    res = train_es(
        theta0,
        fitness,
        rng,
        pop_pairs=24,
        sigma=0.1,
        alpha=0.1,
        generations=12 if smoke else GEN_CAP,
        progress_cb=cb,
    )
    wall = time.time() - t0
    sr = success_rate(res["theta"], shapes, n_eval)
    return {"generations": res["generations"], "sr": sr, "wall_s": wall}


def run_re(shapes, smoke):
    rng = np.random.default_rng(TRAIN_SEED)
    theta0 = xavier_init(np.random.default_rng(INIT_SEED), shapes)
    n_eval = 20 if smoke else 100
    state = {"passes": 0}

    def cb(ep, th, mean_ret):
        if ep % 200 == 0 or smoke:
            sr = success_rate(th, shapes, n_eval)
            print(f"  [re] ep={ep} mean_ret={mean_ret:.2f} sr={sr:.2f}", flush=True)
            state["passes"] = state["passes"] + 1 if sr >= BAR else 0
            if state["passes"] >= 2:
                return True
        return False

    def rollout_fn(th, ep_rng):
        O, A, R, _ = rollout(th, shapes, int(ep_rng.integers(2**31)))
        return (O, A, R)

    t0 = time.time()
    res = train_re(
        theta0,
        shapes,
        rollout_fn,
        rng,
        episode_cap=400 if smoke else EP_CAP,
        progress_cb=cb,
    )
    wall = time.time() - t0
    sr = success_rate(res["theta"], shapes, n_eval)
    return {"episodes": res["episodes"], "sr": sr, "wall_s": wall}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", choices=["es", "re"], default="es")
    ap.add_argument("--tag", default="UNREGISTERED")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    shapes = layer_shapes(2, [16, 16], 3)
    print(f"sanity [{args.tag}] optimizer={args.optimizer} smoke={args.smoke}")
    print(f"policy params: {count_params(shapes)}")
    tp = throughput()
    print(f"throughput steps/s: lineworld={tp['lineworld']:.0f} arena={tp['arena']:.0f}")

    t0 = time.time()
    if args.optimizer == "es":
        r = run_es(shapes, args.smoke)
        used = f"gens={r['generations']}/{GEN_CAP}"
    else:
        r = run_re(shapes, args.smoke)
        used = f"eps={r['episodes']}/{EP_CAP}"
    wall = time.time() - t0
    verdict = "PASS" if (r["sr"] >= BAR and wall <= WALL_CAP_S) else "FAIL"
    print(f"RESULT [{args.tag}] {verdict} sr={r['sr']:.2f} (bar {BAR}) "
          f"{used} wall={wall:.1f}s (cap {WALL_CAP_S}s)")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
