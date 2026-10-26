"""Registered battery runner + ablation modes.

Modes (modular): normal | skill-K | shuffle | mask | greedy
Modes (mono): normal | mask | greedy
Refuses to run if network modules are loaded.
"""

import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from models import (N_SKILLS, GATE_DIM, modular_probs, mono_probs,
                    split_modular)
from policies import xavier_init, layer_shapes
from tasks import TaskRunner

FORBIDDEN_MODS = ("socket", "urllib", "requests", "ssl", "http", "httpx")


def assert_no_net():
    loaded = [m for m in FORBIDDEN_MODS if m in sys.modules]
    if loaded:
        raise RuntimeError(f"network modules loaded during eval: {loaded}")


def run_battery(policy_fn, entries, seed_xor=0):
    assert_no_net()
    results = []
    for e in entries:
        runner = TaskRunner()
        rng = np.random.default_rng(int(e["seed"]) ^ seed_xor)
        obs = runner.reset(e["family"], e["cfg"], int(e["seed"]))
        total, done = 0.0, False
        while not done:
            obs, r, done, info = runner.step(policy_fn(obs, rng))
            total += r
        results.append({"id": e["id"], "success": bool(info["success"]),
                        "return": float(total)})
    n = len(results)
    p = sum(r["success"] for r in results) / n
    ci = 1.96 * float(np.sqrt(p * (1 - p) / n)) if n else 0.0
    return {"n": n, "success_rate": p, "ci95": ci, "episodes": results}


def subset(path, battery):
    return [e for e in json.load(open(path)) if e["battery"] == battery]


def make_policy(arch, theta, mode="normal", shuffle_seed=0, trace=None):
    from models import ModularNet, MonoNet
    from policies import unpack
    net = ModularNet(theta) if arch == "modular" else MonoNet(theta)
    gate_mats = None
    if mode == "shuffle":
        gate_mats = unpack(
            xavier_init(np.random.default_rng(shuffle_seed),
                        layer_shapes(14, [32], N_SKILLS)),
            layer_shapes(14, [32], N_SKILLS))
    skill_only = None
    if mode.startswith("skill-"):
        skill_only = int(mode.split("-")[1])
        assert 0 <= skill_only < N_SKILLS
    mask = mode == "mask"
    greedy = mode == "greedy"

    def policy_fn(obs, rng):
        if arch == "modular":
            p, w = net.probs(obs, skill_only=skill_only,
                             gate_mats=gate_mats, mask_hazard=mask)
        else:
            p = net.probs(obs, mask_hazard=mask)
            w = None
        if trace is not None and w is not None:
            trace.append([float(x) for x in w])
        if greedy:
            return int(np.argmax(p))
        return int(rng.choice(len(p), p=p))

    return policy_fn


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", required=True)
    ap.add_argument("--subset", default="")
    ap.add_argument("--policy", choices=["random", "modular", "mono"],
                    default="random")
    ap.add_argument("--params", default="")
    ap.add_argument("--mode", default="normal")
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    entries = json.load(open(args.battery))
    if args.subset:
        entries = [e for e in entries if e["battery"] == args.subset]
    if args.limit:
        entries = entries[: args.limit]
    if args.policy == "random":
        def policy_fn(obs, rng):
            return int(rng.integers(9))
    else:
        theta = np.load(args.params)["theta"]
        policy_fn = make_policy(args.policy, theta, args.mode,
                                args.shuffle_seed)
    rep = run_battery(policy_fn, entries)
    print(f"n={rep['n']} sr={rep['success_rate']:.3f} ci95={rep['ci95']:.3f}")
    if args.out:
        json.dump(rep, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
