"""Registered battery runner (implemented now; learner evals run after training).

policy_fn(obs, rng) -> action int. Refuses to run if network modules are loaded.
"""

import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

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


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", required=True)
    ap.add_argument("--policy", choices=["random"], default="random")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    entries = json.load(open(args.battery))
    if args.limit:
        entries = entries[: args.limit]
    n_actions = 9

    def policy_fn(obs, rng):
        return int(rng.integers(n_actions))

    rep = run_battery(policy_fn, entries)
    print(f"battery={args.battery} n={rep['n']} sr={rep['success_rate']:.3f} "
          f"ci95={rep['ci95']:.3f}")
    if args.out:
        json.dump(rep, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
