"""Minimal reproduction: A01-final vs A-only vs B-only on ACQ18.

REAL LOCAL TINY EXECUTION on CPU (numpy, deterministic scenario seeds).
Not a recorded sample: every number below is computed by running the
frozen Tiny environment in this process. See MANIFEST.json for hashes.

Usage (from repo root):
  python3.14 -m venv .venv
  .venv/bin/pip install -r repro_s16/requirements.txt
  .venv/bin/python repro_s16/repro.py
"""
import hashlib
import json
import os
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "v2views"))

import numpy as np  # noqa: E402

import views_train as V  # noqa: E402
from models import GATE_SHAPES, SKILL_SHAPES, mlp_forward, split_modular  # noqa: E402
from policies import softmax  # noqa: E402
from tasks import TaskRunner  # noqa: E402

SEEDS = (301, 302, 303)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def greedy_skill(sk_all, obs, k):
    q = softmax(mlp_forward(sk_all[k], V.view_obs(obs, k), SKILL_SHAPES))
    return int(np.argmax(q))


def rollout_gate(gth, sk_all, e):
    r = TaskRunner()
    obs = r.reset(e["family"], e["cfg"], int(e["seed"]))
    total, done, info, n = 0.0, False, {}, 0
    while not done:
        k = int(np.argmax(mlp_forward(gth, obs, GATE_SHAPES)))
        obs, rew, done, info = r.step(greedy_skill(sk_all, obs, k))
        total += rew
        n += 1
    return total, cause_of(r, info)


def rollout_const(sk_all, k, e):
    r = TaskRunner()
    obs = r.reset(e["family"], e["cfg"], int(e["seed"]))
    total, done, info = 0.0, False, {}
    while not done:
        obs, rew, done, info = r.step(greedy_skill(sk_all, obs, k))
        total += rew
    return total, cause_of(r, info)


def cause_of(runner, info):
    env = runner.env
    if info["success"]:
        return "arrival"
    if float(np.linalg.norm(env.haz_pos - env.agent_pos)) < 5.0:
        return "contact-death"
    if env.t >= env.MAX_STEPS:
        return "timeout"
    return "other"


def main():
    t0 = time.time()
    man = json.load(open(os.path.join(HERE, "MANIFEST.json")))
    for rel, want in man["files"].items():
        got = sha(os.path.join(REPO, rel))
        assert got == want, f"hash mismatch {rel}"
    print("hashes ok:", len(man["files"]), "files")
    acq = json.load(open(os.path.join(REPO, "v2conf_diag/s15_acq18.json")))
    a01 = json.load(open(os.path.join(REPO, "v2conf_diag/s14_models.json")))
    skills = {s: split_modular(np.load(os.path.join(
        REPO, f"v2conf_runs/VC_{s}_theta.npz"))["theta"])[0] for s in SEEDS}
    res = {}
    for s in SEEDS:
        g = np.array(a01[f"A01_{s}"])
        for arm in ("A01", "A-ONLY", "B-ONLY"):
            arr = c = t = 0
            rets = []
            for e in acq:
                if arm == "A01":
                    ret, cause = rollout_gate(g, skills[s], e)
                else:
                    ret, cause = rollout_const(
                        skills[s], 0 if arm == "A-ONLY" else 1, e)
                rets.append(ret)
                arr += cause == "arrival"
                c += cause == "contact-death"
                t += cause == "timeout"
            res[(s, arm)] = (arr, c, t, float(np.mean(rets)))
    print(f"{'th':>4} {'arm':<7} {'arr':>4} {'col':>4} {'tmo':>4} {'ret':>7}")
    for s in SEEDS:
        for arm in ("A01", "A-ONLY", "B-ONLY"):
            a, c, t, r = res[(s, arm)]
            print(f"{s:>4} {arm:<7} {a:>4} {c:>4} {t:>4} {r:>7.3f}")
    print("mean arrivals over 1200:")
    for arm in ("A01", "A-ONLY", "B-ONLY"):
        m = sum(res[(s, arm)][0] for s in SEEDS) / 1200
        print(f"  {arm:<7} {m:.4f}")
    json.dump({f"{s}_{a}": {"arr": v[0], "col": v[1], "tmo": v[2],
                             "ret": v[3]} for (s, a), v in res.items()},
              open(os.path.join(HERE, "repro_out.json"), "w"), indent=1)
    print(f"REAL LOCAL EXECUTION wall={time.time()-t0:.1f}s "
          f"episodes={3*400*3} (deterministic, CPU-only)")


if __name__ == "__main__":
    main()
