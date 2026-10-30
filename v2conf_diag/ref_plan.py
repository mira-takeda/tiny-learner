"""Privileged reference planner probe (read-only; no training).

REFERENCE WITH PRIVILEGED INFO (exact state + simulator). NOT a Tiny
capability. Success = a solution was found; limited-search failure
does NOT prove impossibility.

Fixed pre-run (no sweep/tuning): H=25, K=40, 5 blocks x 5 steps.
P-prim: one sustained primitive per block. P-AB: one skill {A,B} per
block, greedy action recomputed from each imagined obs (same
skill/view/argmax composition as the frozen implementation, verified
by observation in V3). Replan after every real step. Scoring =
exact G1 task return (fitness currency), first-max tie-break.
Planning model = env mirror with hazard noise as an explicit arg;
production uses zeros (declared modification). Sequence-sampling RNG:
SeedSequence(8100), fixed order (P-prim 3 eps, then P-AB 3x3).

Micropilot (fixed IDs): A-ACQ-0000 (G1a), A-ACQ-0001 (G1b),
A-ACQ-0002 (G1adv). P-prim on 3; P-AB x thetas 301-303 if budget
remains. TOTAL wall budget 300 s including 10 s calibration.
Interrupted episodes are recorded INCOMPLETE, not failures.

Validations: V0 battery/thetas integrity; V1 clone-with-identical-
noise == env bit-exact; V2 noiseless model == env except the
declared noise modification (proved via noise-injected equality);
V3 skill/view/argmax identity by observation vs frozen composition.
"""
import hashlib
import json
import sys
import time

sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/src")
sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/v2views")
sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/v2conf")

import numpy as np

import views_train as V
from conf_train import EVCONF_SHA
from env import Arena
from models import SKILL_SHAPES, mlp_forward, split_modular
from policies import softmax
from tasks import TaskRunner

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
EVC_PATH = f"{ROOT}/v2conf/eval_conf_v1.json"
SEEDS = (301, 302, 303)

# Fixed search parameters (no sweep).
H, K, NBLOCK, BLEN = 25, 40, 5, 5
SEQ_SEED = 8100
PILOT_IDS = ["A-ACQ-0000", "A-ACQ-0001", "A-ACQ-0002"]
T_BUDGET = 300.0
T_MARGIN = 5.0
CALIB_S = 10.0

ACTIONS = Arena().ACTIONS.copy()
VMAX, DAMP = Arena.VMAX, Arena.DAMP
STEP_PEN = -0.01
ARRIVE_R, CONTACT_R = 6.0, 5.0
MAX_STEPS = Arena.MAX_STEPS


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---- planning model (env mirror, noise explicit) ----
def m_obs(st):
    o = np.zeros(14)
    o[0:2] = st["avel"] / VMAX
    for sl, p in ((2, st["goal"]), (5, st["haz"]), (8, None), (11, None)):
        if p is None:
            continue
        d = p - st["apos"]
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            continue
        o[sl:sl + 3] = (d[0] / n, d[1] / n, n / Arena.DMAX)
    return o


def m_step(st, a, noise):
    ap = st["apos"] + np.zeros(2)
    av = (st["avel"] + ACTIONS[int(a)]) * DAMP
    n = float(np.linalg.norm(av))
    if n > VMAX:
        av *= VMAX / n  # in-place like env (1-ulp differs from av*...)
    ap = ap + av
    for i in (0, 1):
        if ap[i] < 0.0:
            ap[i] = 0.0
            av[i] = -0.5 * av[i]
        elif ap[i] > Arena.SIZE:
            ap[i] = Arena.SIZE
            av[i] = -0.5 * av[i]
    d = ap - st["haz"]
    n = float(np.linalg.norm(d))
    hv = np.zeros(2) if n < 1e-9 else d / n * st["hspeed"]
    hv = hv + noise
    hp = st["haz"] + hv
    for i in (0, 1):
        hp[i] = min(max(hp[i], 0.0), Arena.SIZE)
    return {"apos": ap, "avel": av, "haz": hp, "hspeed": st["hspeed"],
            "goal": st["goal"], "t": st["t"] + 1}


def m_g1(st, a, prev_goal, noise):
    """One imagined G1 step. Returns (next_state, obs, ret, done, ok)."""
    ns = m_step(st, a, noise)
    obs = m_obs(ns)
    if float(np.linalg.norm(ns["apos"] - ns["haz"])) < CONTACT_R:
        return ns, obs, -10.0, True, False
    cur = float(np.linalg.norm(ns["apos"] - ns["goal"]))
    r = STEP_PEN + 0.1 * (prev_goal - cur)
    if cur < ARRIVE_R:
        return ns, obs, r + 10.0, True, True
    if ns["t"] >= MAX_STEPS:
        return ns, obs, r, True, False
    return ns, obs, r, False, False


def snap(runner):
    e = runner.env
    return {"apos": e.agent_pos.copy(), "avel": e.agent_vel.copy(),
            "haz": e.haz_pos.copy(), "hspeed": e.haz_speed,
            "goal": e.goal.copy(), "t": e.t}


def greedy_skill(sk_all, obs, k):
    return int(np.argmax(softmax(mlp_forward(
        sk_all[k], V.view_obs(obs, k), SKILL_SHAPES))))


def shoot(st0, prev_goal, seqs, sk_all, prim):
    """Random shooting over block sequences. Returns (first_move, imagined)."""
    best, best_first, imag = -1e18, None, 0
    for s in seqs:
        st, pg, tot, done = dict(st0), prev_goal, 0.0, False
        first = None
        for t in range(H):
            blk = s[t // BLEN]
            if prim:
                a = int(blk)
            else:
                a = greedy_skill(sk_all, m_obs(st), int(blk))
            if t == 0:
                first = (a, int(blk))
            st, _, r, done, _ = m_g1(st, a, pg, np.zeros(2))
            pg = float(np.linalg.norm(st["apos"] - st["goal"]))
            tot += r
            imag += 1
            if done:
                break
        if tot > best:
            best, best_first = tot, first
    return best_first, imag


def main():
    t0 = time.time()
    out = {"H": H, "K": K, "blocks": [NBLOCK, BLEN], "seq_seed": SEQ_SEED,
           "pilot_ids": PILOT_IDS, "budget_s": T_BUDGET, "records": []}

    # V0: battery + thetas integrity.
    assert sha256_file(EVC_PATH) == EVCONF_SHA, "eval battery altered"
    evc = json.load(open(EVC_PATH))
    pilot = [next(e for e in evc if e["id"] == i) for i in PILOT_IDS]
    assert [e["region"] for e in pilot] == ["G1a", "G1b", "G1adv"]
    assert all(e["family"] == "G1" for e in pilot)
    skills = {}
    for s in SEEDS:
        th = np.load(f"{ROOT}/v2conf_runs/VC_{s}_theta.npz")["theta"]
        p3 = np.load(f"{ROOT}/pilot3_runs/P3_{s}_theta.npz")["theta"]
        sk, _ = split_modular(th)
        assert all(np.array_equal(a, b)
                   for a, b in zip(sk, split_modular(p3)[0])), s
        skills[s] = sk
    print(f"V0 ok: battery match, pilot={[ (e['id'],e['region'],e['seed']) for e in pilot ]}",
          flush=True)

    # V1/V2 twin-env proof (exact RNG fork): model+injected draws vs
    # real env. Both rngs start at the episode seed; the twin draws the
    # same noise the real env consumes internally each step. Matched
    # termination is agreement, not mismatch.
    matched = 0
    for ei, e in enumerate(pilot):
        r1, r2 = TaskRunner(), TaskRunner()
        r1.reset(e["family"], e["cfg"], int(e["seed"]))
        r2.reset(e["family"], e["cfg"], int(e["seed"]))
        st = snap(r2)
        pg = float(np.linalg.norm(st["apos"] - st["goal"]))
        for t in range(50):
            a = int(np.random.default_rng(1000 + ei * 100 + t).integers(9))
            draws = r1.env.rng.normal(0.0, 0.5, 2)
            o2, re, d2, i2 = r2.step(a)
            st, mo, mr, md, mi = m_g1(st, a, pg, draws)
            assert np.array_equal(mo, o2), (ei, t, "obs")
            assert (md, mr, mi) == (d2, re, i2["success"]), (ei, t, "task")
            matched += 1
            if d2:
                break
            pg = float(np.linalg.norm(st["apos"] - st["goal"]))
    print(f"V1/V2 model+injected-noise == env bit-exact "
          f"(obs/done/return) over {matched} steps x3 entries", flush=True)
    print("V2 interpretation: noiseless production model differs ONLY by "
          "the declared noise=0 modification", flush=True)

    # V3: skill/view/argmax identity by observation vs frozen composition.
    obs_pool = []
    rr = TaskRunner()
    for e in pilot:
        o = rr.reset(e["family"], e["cfg"], int(e["seed"]))
        for t in range(20):
            obs_pool.append(o.copy())
            o = rr.env.step(int(np.random.default_rng(t).integers(9)))
    n_probe = 0
    for s in SEEDS:
        sk_all = skills[s]
        for o in obs_pool:
            for k in (0, 1):
                ref = int(np.argmax(softmax(mlp_forward(
                    sk_all[k], V.view_obs(o.copy(), k), SKILL_SHAPES))))
                got = greedy_skill(sk_all, o, k)
                assert got == ref, (s, k)
                n_probe += 1
    print(f"V3 ok: greedy == frozen skill/view/argmax on {n_probe} probes",
          flush=True)

    # Calibration: 10 s total (5 s model-only, 5 s model+MLP).
    def calib(with_mlp, secs):
        st = {"apos": np.array([50., 50.]), "avel": np.zeros(2),
              "haz": np.array([60., 60.]), "hspeed": 3.2,
              "goal": np.array([80., 80.]), "t": 0}
        sk = skills[301]
        n, end = 0, time.time() + secs
        while time.time() < end:
            if with_mlp:
                a = greedy_skill(sk, m_obs(st), n % 2)
            else:
                a = n % 9
            st, _, _, done, _ = m_g1(
                st, a, float(np.linalg.norm(st["apos"] - st["goal"])),
                np.zeros(2))
            n += 1
            if done or st["t"] >= MAX_STEPS:
                st["t"] = 0
        return n / secs
    r_env = calib(False, CALIB_S / 2)
    r_mlp = calib(True, CALIB_S / 2)
    out["calib_imag_steps_per_s"] = {"model_only": r_env,
                                     "model_mlp": r_mlp}
    print(f"CALIB: model-only={r_env:.0f} imag-steps/s, "
          f"model+MLP={r_mlp:.0f} imag-steps/s", flush=True)

    # Micropilot.
    seq_rng = np.random.default_rng(SEQ_SEED)
    plan = [("prim", e, None) for e in pilot] + \
           [("AB", e, s) for e in pilot for s in SEEDS]
    for mode, e, s in plan:
        if time.time() - t0 > T_BUDGET - T_MARGIN:
            out["records"].append({"mode": mode, "id": e["id"], "theta": s,
                                   "status": "incomplete",
                                   "reason": "budget-exhausted-before-start"})
            continue
        prim = mode == "prim"
        sk_all = None if prim else skills[s]
        runner = TaskRunner()
        obs = runner.reset(e["family"], e["cfg"], int(e["seed"]))
        assert runner.env.haz_pos is not None
        n_real, imag, nb, sw, prev = 0, 0, 0, 0, None
        done, info, abort = False, {}, False
        ep_t0 = time.time()
        while not done:
            if time.time() - t0 > T_BUDGET - T_MARGIN:
                abort = True
                break
            st = snap(runner)
            pg = runner.prev_goal
            if prim:
                seqs = seq_rng.integers(0, 9, size=(K, NBLOCK))
            else:
                seqs = seq_rng.integers(0, 2, size=(K, NBLOCK))
            (a, blk), im = shoot(st, pg, seqs, sk_all, prim)
            imag += im
            if not prim:
                nb += (blk == 1)
                sw += (prev is not None and blk != prev)
                prev = blk
            obs, _, done, info = runner.step(a)
            n_real += 1
        rec = {"mode": mode, "id": e["id"], "theta": s,
               "imag_steps": imag, "wall_s": round(time.time() - ep_t0, 1)}
        if abort:
            rec.update({"status": "incomplete",
                        "reason": "budget-exhausted-mid-episode",
                        "n_real": n_real})
        else:
            env = runner.env
            if info["success"]:
                cause = "arrival"
            elif float(np.linalg.norm(env.haz_pos - env.agent_pos)) < 5.0:
                cause = "contact-death"
            elif env.t >= env.MAX_STEPS:
                cause = "timeout"
            else:
                cause = "other"
            rec.update({"status": "complete", "cause": cause, "T": n_real})
            if not prim:
                rec.update({"frac_B": round(nb / n_real, 4),
                            "switches": sw})
        out["records"].append(rec)
        print(f"{mode} {e['id']}" + (f" th={s}" if s else "") +
              f": {rec['status']}" +
              (f" {rec.get('cause')} T={rec.get('T')}" if rec["status"] == "complete"
               else f" ({rec['reason']})") +
              f" imag={imag} wall={rec['wall_s']}s", flush=True)

    out["total_wall_s"] = round(time.time() - t0, 1)
    json.dump(out, open(f"{ROOT}/v2conf_diag/refplan_pilot.json", "w"),
              indent=1)
    print(f"saved v2conf_diag/refplan_pilot.json "
          f"total_wall={out['total_wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
