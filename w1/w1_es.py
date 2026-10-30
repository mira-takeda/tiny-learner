"""W1 hard-ES alpha=0.01 from W1-imit inits (N2 recipe verbatim).

25 gens x 16 cands x 8 eps (2 wide-G1 + 2 G2 + ABCD, frozen
calendar). Hard policy, exact-return fitness, tied rank-shaping.
Gate pools structurally unreachable (calendar IDs asserted).
Frozen: w1_freeze.json + MANIFEST. Budget 2400 s, resume.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "v2views"))
sys.path.insert(0, os.path.join(ROOT, "v2conf_diag"))
sys.path.insert(0, HERE)

import numpy as np

import s4_closed as S4
from gen import sha256_file
from models import GATE_SHAPES, mlp_forward, split_modular
from tasks import TaskRunner
import gen_wide as GW

W1 = HERE
SEEDS = (301, 302, 303)
CAP = 2400.0
MARGIN = 10.0


def tied_utils(fits):
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
    assert np.all(u == u[0])
    assert float(np.max(np.abs(u))) == 0.0
    u = tied_utils([3.0, 1.0, 3.0, 2.0] + [0.0] * 12)
    assert u[0] == u[1 + 1]
    u = tied_utils([float(i) for i in range(16)])
    assert np.allclose(u, np.arange(16) / 15.0 - 0.5)
    eps = np.random.default_rng(0).standard_normal((8, 4))
    uu = tied_utils([5.0] * 16)
    g = sum((uu[2 * p] - uu[2 * p + 1]) * eps[p] for p in range(8)) / 8
    assert np.all(g == 0.0)
    print("ties ok", flush=True)


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
    fr, disk = GW.regen_check()
    pins = fr["pins"]
    assert sha256_file(os.path.join(
        ROOT, "v2conf_diag/s4_closed.py")) == pins["v2conf_diag/s4_closed.py"]
    assert sha256_file(os.path.join(
        ROOT, "data/train_meta_v1.json")) == pins["data/train_meta_v1.json"]
    es = fr["es"]
    assert (es["pop_pairs"], es["candidates"], es["sigma"], es["alpha"],
            es["generations"]) == (8, 16, 0.1, 0.01, 25)
    assert es["seed"] == 21107
    for s in SEEDS:
        assert sha256_file(os.path.join(
            ROOT, f"pilot3_runs/P3_{s}_theta.npz")) == pins[
                f"pilot3_runs/P3_{s}_theta.npz"], s
    m1 = json.load(open(os.path.join(W1, "w1_imit_models.json")))
    for s in SEEDS:
        assert len(m1[f"W1I_{s}"]) == 612
        assert np.all(np.isfinite(m1[f"W1I_{s}"])), s
    by_id = {e["id"]: e for e in
             json.load(open(os.path.join(ROOT, "data/train_meta_v1.json")))}
    for e in disk["pools"]["espool_g1"]:
        assert e["id"] not in by_id
        by_id[e["id"]] = e
    cal = disk["calendar"]
    assert len(cal) == 25 and all(len(w) == 8 for w in cal)
    esp = {e["id"] for e in disk["pools"]["espool_g1"]}
    g2p = set(fr["es_pool_g2"])
    sgp = {i for v in fr["es_pool_singles"].values() for i in v}
    for w in cal:  # structural seal: only ES-pool IDs, never gate IDs
        assert w[0] in esp and w[1] in esp
        assert w[2] in g2p and w[3] in g2p
        assert w[4] in sgp and w[5] in sgp and w[6] in sgp and w[7] in sgp
        fams = [by_id[i]["family"] for i in w]
        assert fams[0] == "G1" and fams[1] == "G1" and fams[2] == "G2" \
            and fams[3] == "G2" and fams[4:] == ["A", "B", "C", "D"], fams
    skills = {s: split_modular(np.load(os.path.join(
        ROOT, f"pilot3_runs/P3_{s}_theta.npz"))["theta"])[0] for s in SEEDS}
    st_p = os.path.join(W1, "w1_es_state.json")
    try:
        state = json.load(open(st_p))
    except FileNotFoundError:
        state = {"cumulative_s": 0.0, "phase": "train",
                 "theta": {str(s): m1[f"W1I_{s}"] for s in SEEDS},
                 "gen_done": {str(s): 0 for s in SEEDS},
                 "fits": {}, "history": {str(s): [] for s in SEEDS},
                 "ckpts": {str(s): [] for s in SEEDS},
                 "upd_norms": {str(s): [] for s in SEEDS}}
    t0 = time.time()
    base = float(state["cumulative_s"])
    spent = lambda: base + time.time() - t0
    ok = lambda: spent() < CAP - MARGIN
    sync = lambda: (state.update(cumulative_s=round(spent(), 1)),
                    json.dump(state, open(st_p, "w")))

    def stop_incomplete(msg):
        state["phase"] = "incomplete"
        sync()
        print(f"INCOMPLETE {msg} wall={spent():.1f}s", flush=True)

    if state["phase"] == "train":
        for ti, s in enumerate(SEEDS):
            th = np.array(state["theta"][str(s)])
            for g in range(25):
                if g < state["gen_done"][str(s)]:
                    continue
                kid = ti * 100 + g
                rng = np.random.default_rng(
                    np.random.SeedSequence(es["seed"]).spawn(kid + 1)[kid])
                eps = rng.standard_normal((8, 612))
                window = [by_id[i] for i in cal[g]]
                means = []
                for i in range(16):
                    key = f"{s}_{g}_{i}"
                    if key in state["fits"]:
                        means.append(state["fits"][key]["mean"])
                        continue
                    if not ok():
                        sync()
                        return stop_incomplete(f"train th={s} gen={g} cand={i}")
                    cand = th + es["sigma"] * eps[i // 2] * (
                        1 if i % 2 == 0 else -1)
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
        state["phase"] = "done"
        sync()
        json.dump({f"W1E_{s}": state["theta"][str(s)] for s in SEEDS},
                  open(os.path.join(W1, "w1_es_models.json"), "w"))
        print(f"train done wall={spent():.1f}s", flush=True)
    if any(state["gen_done"][str(s)] != 25 for s in SEEDS):
        return stop_incomplete("a theta lacks 25 gens")
    for s in SEEDS:
        assert sha256_file(os.path.join(
            ROOT, f"pilot3_runs/P3_{s}_theta.npz")) == pins[
                f"pilot3_runs/P3_{s}_theta.npz"], s
    print(f"W1-ES done wall={spent():.1f}s", flush=True)


if __name__ == "__main__":
    main()
