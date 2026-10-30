"""N4: N2 recipe from N3b inits (only init+seed+heldout differ).

Train: manifest pools, 25 gens x 16 cands x 8 eps (2G1+2G2+ABCD).
Eval: manifest-heldout 60 G1 + 60 G2 (sealed pre-train), N2-final
vs N1-frozen. Frozen: n4_freeze.json. Cumulative 1800 s.
"""
import hashlib
import json
import sys
import time

sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/src")
sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/v2conf_diag")

import numpy as np

import s4_closed as S4
from models import GATE_SHAPES, mlp_forward, split_modular
from tasks import TaskRunner

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
SEEDS = (301, 302, 303)
CAP = 1800.0
MARGIN = 10.0


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tied_utils(fits):
    """Centered ranks with averaged tied ordinals. All-tied -> all equal."""
    m = len(fits)
    order = np.argsort(np.asarray(fits), kind="mergesort")
    rbar = np.empty(m)
    i = 0
    while i < m:
        j = i
        while j + 1 < m and fits[order[j + 1]] == fits[order[i]]:
            j += 1
        rbar[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return (rbar - (m - 1) / 2.0) / (m - 1)


def check_ties():
    u = tied_utils([1.0] * 16)
    assert np.all(u == u[0]), "all-tied must share weight"
    assert float(np.max(np.abs(u))) == 0.0, "all-tied centered at 0"
    u = tied_utils([3.0, 1.0, 3.0, 2.0] + [0.0] * 12)
    assert u[0] == u[1 + 1], "tied pair must share weight"
    u = tied_utils([float(i) for i in range(16)])
    assert np.allclose(u, np.arange(16) / 15.0 - 0.5), "no-tie classic"
    # zero-update proof on dummy eps
    eps = np.random.default_rng(0).standard_normal((8, 4))
    uu = tied_utils([5.0] * 16)
    g = sum((uu[2 * p] - uu[2 * p + 1]) * eps[p] for p in range(8)) / 8
    assert np.all(g == 0.0), "all-tied must give zero grad"
    print("ties ok: equal->equal weight, all-tied->zero update", flush=True)


def cause_of(runner, info):
    env = runner.env
    if info["success"]:
        return "arrival"
    if env.haz_pos is not None and float(
            np.linalg.norm(env.haz_pos - env.agent_pos)) < 5.0:
        return "contact-death"
    if env.t >= env.MAX_STEPS:
        return "timeout"
    return "other"


def rollout(gth, sk_all, e):
    r = TaskRunner()
    obs = r.reset(e["family"], e["cfg"], int(e["seed"]))
    total, done, info, n = 0.0, False, {}, 0
    uses, sw, prev = [0, 0, 0, 0], 0, None
    while not done:
        k = int(np.argmax(mlp_forward(gth, obs, GATE_SHAPES)))
        a = S4.greedy_skill(sk_all, obs, k)
        uses[k] += 1
        sw += (prev is not None and k != prev)
        prev = k
        obs, rew, done, info = r.step(a)
        total += rew
        n += 1
    return {"ret": total, "cause": cause_of(r, info), "T": n,
            "uses": uses, "switches": sw}


def main():
    check_ties()
    fr = json.load(open(f"{ROOT}/v2conf_diag/n4_freeze.json"))
    assert sha(f"{ROOT}/data/train_meta_v1.json") == fr["data_sha"]
    assert sha(f"{ROOT}/v2conf_diag/n3b_models.json") == fr["init"]["file_sha256"]
    es = fr["es"]
    assert (es["pop_pairs"], es["candidates"], es["sigma"], es["alpha"],
            es["generations"]) == (8, 16, 0.1, 0.01, 25)
    assert fr["es"]["seed"] == 4545
    for s in SEEDS:
        assert sha(f"{ROOT}/v2conf_runs/VC_{s}_theta.npz") == fr["skills"][str(s)]
    m1 = json.load(open(f"{ROOT}/v2conf_diag/n3b_models.json"))
    for s in SEEDS:
        assert len(m1[f"N3b_{s}"]) == 612
    by_id = {e["id"]: e for e in json.load(open(f"{ROOT}/data/train_meta_v1.json"))}
    assert len(fr["calendar"]) == 25 and all(len(w) == 8 for w in fr["calendar"])
    skills = {s: split_modular(np.load(
        f"{ROOT}/v2conf_runs/VC_{s}_theta.npz")["theta"])[0] for s in SEEDS}
    try:
        state = json.load(open(f"{ROOT}/v2conf_diag/n4_state.json"))
    except FileNotFoundError:
        seed = {"cumulative_s": 0.0}
        state = {"cumulative_s": seed["cumulative_s"], "phase": "train",
                 "theta": {str(s): m1[f"N3b_{s}"] for s in SEEDS},
                 "gen_done": {str(s): 0 for s in SEEDS},
                 "fits": {}, "history": {str(s): [] for s in SEEDS},
                 "ckpts": {str(s): [] for s in SEEDS},
                 "upd_norms": {str(s): [] for s in SEEDS},
                 "evals": {}}
    t0 = time.time()
    base = float(state["cumulative_s"])
    spent = lambda: base + time.time() - t0
    ok = lambda: spent() < CAP - MARGIN
    sync = lambda: (state.update(cumulative_s=round(spent(), 1)),
                    json.dump(state, open(
                        f"{ROOT}/v2conf_diag/n4_state.json", "w")))

    def stop_incomplete(msg):
        state["phase"] = "incomplete"
        sync()
        print(f"INCOMPLETE {msg} wall={spent():.1f}s", flush=True)

    # Phase 1: ES finetune.
    if state["phase"] == "train":
        for ti, s in enumerate(SEEDS):
            th = np.array(state["theta"][str(s)])
            for g in range(25):
                if g < state["gen_done"][str(s)]:
                    continue
                kid = ti * 100 + g
                rng = np.random.default_rng(
                    np.random.SeedSequence(fr["es"]["seed"]).spawn(kid + 1)[kid])
                eps = rng.standard_normal((8, 612))
                window = [by_id[i] for i in fr["calendar"][g]]
                means = []
                for i in range(16):
                    key = f"{s}_{g}_{i}"
                    if key in state["fits"]:
                        means.append(state["fits"][key]["mean"])
                        continue
                    if not ok():
                        sync()
                        return stop_incomplete(f"train th={s} gen={g} cand={i}")
                    cand = th + es["sigma"] * eps[i // 2] * (1 if i % 2 == 0 else -1)
                    rets = []
                    for e in window:
                        if not ok():
                            sync()
                            return stop_incomplete(
                                f"train th={s} gen={g} cand={i} ep={len(rets)}")
                        rets.append(rollout(cand, skills[s], e)["ret"])
                    state["fits"][key] = {"mean": float(np.mean(rets)),
                                          "rets": rets}
                    means.append(state["fits"][key]["mean"])
                    sync()
                u = tied_utils(means)
                grad = sum((u[2 * p] - u[2 * p + 1]) * eps[p]
                           for p in range(8)) / 8
                upd = (es["alpha"] / es["sigma"]) * grad
                th = th + upd
                state["theta"][str(s)] = [float(x) for x in th]
                state["gen_done"][str(s)] = g + 1
                state["history"][str(s)].append(float(np.max(means)))
                state["ckpts"][str(s)].append([float(x) for x in th])
                state["upd_norms"][str(s)].append(float(np.linalg.norm(upd)))
                sync()
                print(f"train th={s} gen={g} best={np.max(means):.3f} "
                      f"wall={spent():.1f}s", flush=True)
        state["phase"] = "eval"
        sync()
        json.dump({f"N4_{s}": state["theta"][str(s)] for s in SEEDS},
                  open(f"{ROOT}/v2conf_diag/n4_models.json", "w"))
        print(f"train done wall={spent():.1f}s", flush=True)

    if state["phase"] != "eval":
        return
    if any(state["gen_done"][str(s)] != 25 for s in SEEDS):
        return stop_incomplete("eval gate: a theta lacks 25 gens")
    # Phase 2: heldout opens here only (sealed pre-train).
    ctrl = {e["id"]: e for e in json.load(open(f"{ROOT}/data/train_control_v1.json"))}
    assert sha(f"{ROOT}/data/train_control_v1.json") == fr["heldout3_sha"]
    acq = [ctrl[i] for i in fr["heldout3"]["G1"] + fr["heldout3"]["G2"]]
    assert len(acq) == 120
    arms = {"N4": {s: np.array(state["theta"][str(s)]) for s in SEEDS},
            "N3b": {s: np.array(m1[f"N3b_{s}"]) for s in SEEDS}}
    for arm in ("N4", "N3b"):
        for s in SEEDS:
            for e in acq:
                key = f"{arm}_{s}_{e['id']}"
                if key in state["evals"]:
                    continue
                if not ok():
                    sync()
                    return stop_incomplete(f"eval {key}")
                r = rollout(arms[arm][s], skills[s], e)
                state["evals"][key] = {"theta": s, "id": e["id"], "arm": arm,
                                       **r, "status": "complete"}
                if len(state["evals"]) % 25 == 0:
                    sync()
    for s in SEEDS:
        assert sha(f"{ROOT}/v2conf_runs/VC_{s}_theta.npz") == fr["skills"][str(s)]
    state["phase"] = "done"
    sync()
    recs = [state["evals"][k] for k in sorted(state["evals"])]
    json.dump({"records": recs, "wall_s": round(spent(), 1)},
              open(f"{ROOT}/v2conf_diag/n4_eval.json", "w"))
    n_inc = sum(1 for r in recs if r.get("status") != "complete")
    print(f"{'done' if n_inc == 0 else 'INCOMPLETE'} "
          f"wall={spent():.1f}s incomplete={n_inc}", flush=True)


if __name__ == "__main__":
    main()
