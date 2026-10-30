"""Fixed-switching probe on v2conf ACQ (read-only; no training).

Policy (fixed pre-run, no sweep/hysteresis/tuning):
  h_obs = obs[7] * DMAX (S2 slot of the gate-visible 14-dim obs).
  If h_obs < 15 -> skill B (idx 1) with its v2views view; else skill A
  (idx 0) with its view. Gate weights unused. No privileged info in
  the policy: the switching signal comes only from the observation
  the learned gate also sees.

Encoding check (pre-run verification, env used ONLY to verify, never
by the policy): env._slot returns [ux, uy, dist/DMAX] with
DMAX = 100*sqrt(2); _obs places the hazard slot at [5:8]. Asserted
here by comparing h_obs vs the true norm on sampled steps.

Battery: v2conf/eval_conf_v1.json ACQ (200 eps, file order, family G1).
World, reward and success condition untouched (frozen TaskRunner).
Sampled actions, per-episode seeds, 200-step horizon: same mechanics
as run_battery. Thetas VC 301-303 (skills asserted bit-equal to
pilot3 post_D).

Reference: >=0.70 sufficient performance for THIS probe (same bar as
ACQ acquisition). v2conf PRIMARY-FAIL verdict is NOT modified.
"""
import hashlib
import json
import sys
from math import sqrt

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
SEEDS = (301, 302, 303)
THRESH = 15.0  # fixed pre-run; no sweep, hysteresis, or post-hoc tuning
T32 = 4.303  # t(0.975, df=2): 95% CI over 3 seeds (score_vconf convention)
EVC_PATH = f"{ROOT}/v2conf/eval_conf_v1.json"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    got = sha256_file(EVC_PATH)
    print(f"eval_conf sha256={got[:12]}... match={got == EVCONF_SHA}")
    assert got == EVCONF_SHA, "eval battery altered"

    evc = json.load(open(EVC_PATH))
    acq = [e for e in evc if e["battery"] == "ACQ"]
    assert len(acq) == 200, len(acq)
    assert all(e["family"] == "G1" for e in acq)
    print(f"ACQ n={len(acq)} family=G1 rule: obs[7]*DMAX<{THRESH:g}->B-view "
          f"else A-view (sampled)")

    # Encoding/scale verification: S2 slot == dist/DMAX at obs[5:8].
    assert abs(Arena.DMAX - 100.0 * sqrt(2.0)) < 1e-9, Arena.DMAX
    probe = TaskRunner()
    worst = 0.0
    for e in acq[:10]:
        obs = probe.reset(e["family"], e["cfg"], int(e["seed"]))
        for _ in range(20):
            h_obs = float(obs[7]) * Arena.DMAX
            h_true = float(np.linalg.norm(probe.env.haz_pos
                                           - probe.env.agent_pos))
            worst = max(worst, abs(h_obs - h_true))
            assert abs(h_obs - h_true) < 1e-6, (h_obs, h_true)
            obs = probe.env.step(0)
    print(f"S2 encoding verified: obs[7]*DMAX==true dist "
          f"(DMAX={Arena.DMAX:.4f}, worst_err={worst:.2e} over 200 steps)")

    skills = {}
    for s in SEEDS:
        th = np.load(f"{ROOT}/v2conf_runs/VC_{s}_theta.npz")["theta"]
        p3 = np.load(f"{ROOT}/pilot3_runs/P3_{s}_theta.npz")["theta"]
        sk, _ = split_modular(th)
        p3sk, _ = split_modular(p3)
        assert all(np.array_equal(a, b) for a, b in zip(sk, p3sk)), \
            f"VC skills != pilot3 post_D for {s}"
        skills[s] = sk
    print("skills VC == pilot3 post_D bit-exact (gate unused by probe)")

    def run(sk_all, e):
        r = TaskRunner()
        rng = np.random.default_rng(int(e["seed"]))
        obs = r.reset(e["family"], e["cfg"], int(e["seed"]))
        assert r.env.haz_pos is not None
        done, n, nb, sw, prev = False, 0, 0, 0, None
        while not done:
            h = float(obs[7]) * Arena.DMAX  # gate-visible obs only
            k = 1 if h < THRESH else 0
            nb += (k == 1)
            sw += (prev is not None and k != prev)
            prev = k
            q = softmax(mlp_forward(sk_all[k], V.view_obs(obs, k),
                                    SKILL_SHAPES))
            obs, _, done, info = r.step(int(rng.choice(len(q), p=q)))
            n += 1
        env = r.env
        if info["success"]:
            cause = "arrival"
        elif float(np.linalg.norm(env.haz_pos - env.agent_pos)) < 5.0:
            cause = "contact-death"
        elif env.t >= env.MAX_STEPS:
            cause = "timeout"
        else:
            cause = "other"
        return {"ok": bool(info["success"]), "cause": cause, "T": n,
                "frac_B": round(nb / n, 4), "switches": sw}

    recs, per_seed = [], {}
    for s in SEEDS:
        outs = [run(skills[s], e) for e in acq]
        n_arr = sum(o["cause"] == "arrival" for o in outs)
        n_col = sum(o["cause"] == "contact-death" for o in outs)
        n_tmo = sum(o["cause"] == "timeout" for o in outs)
        n_oth = len(outs) - n_arr - n_col - n_tmo
        sr = n_arr / len(outs)
        Tmean = float(np.mean([o["T"] for o in outs]))
        fb = float(np.mean([o["frac_B"] for o in outs]))
        swm = float(np.mean([o["switches"] for o in outs]))
        per_seed[s] = {"sr": sr, "arrival": n_arr, "collision": n_col,
                       "timeout": n_tmo, "other": n_oth, "T_mean": Tmean,
                       "frac_B": fb, "switches_mean": swm}
        print(f"th={s}: sr={sr:.3f} ({n_arr}/200) "
              f"collision={n_col} timeout={n_tmo} other={n_oth} "
              f"T_mean={Tmean:.1f} frac_B={fb:.3f} switches={swm:.1f}")
        for e, o in zip(acq, outs):
            recs.append({"theta": s, "id": e["id"], **o})

    v = np.array([per_seed[s]["sr"] for s in SEEDS])
    m, sd = v.mean(), v.std(ddof=1)
    ci = T32 * sd / sqrt(3)
    print(f"\nPROBE mean sr={m:.4f} 95%CI=[{max(0, m - ci):.3f},"
          f"{min(1, m + ci):.3f}] vals={np.round(v, 3)} "
          f"(t(2)=4.303; ref>=0.70: "
          f"{'REACHED' if m >= 0.70 else 'NOT reached'}; "
          f"v2conf PRIMARY-FAIL unchanged)")

    print("\nComparison vs logged VC ACQ refs (sampled, frozen JSONs):")
    for s in SEEDS:
        L = json.load(open(f"{ROOT}/v2conf_runs/VC_{s}.json"))["eval"]["ACQ"]
        p = per_seed[s]
        print(f"  th={s}: probe={p['sr']:.3f} gate-normal={L['normal']['sr']:.3f} "
              f"A-alone={L['skill-0']['sr']:.3f} B-alone={L['skill-1']['sr']:.3f}")

    summary = {"rule": "h_obs=obs[7]*DMAX; B-view if h_obs<15 else A-view",
              "threshold": THRESH, "battery": "ACQ", "n": 200,
              "seeds": list(SEEDS), "thetas": "VC_301-303",
              "reference": 0.70, "reference_reached": bool(m >= 0.70),
              "mean_sr": m, "ci95": ci, "per_seed": per_seed}
    json.dump(summary, open(f"{ROOT}/v2conf_diag/switch_summary.json", "w"),
              indent=1)
    json.dump(recs, open(f"{ROOT}/v2conf_diag/switch_records.json", "w"))
    print(f"saved v2conf_diag/switch_summary.json + "
          f"switch_records.json ({len(recs)} records)")


if __name__ == "__main__":
    main()
