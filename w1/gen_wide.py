"""W1 wide-G1 generator + pools + coverage (FROZEN pre-pools).

PREREG_W1.md sections 2-3. Mirror of gen.sample_config 'guarded'
with ONLY two bound changes, both from registered task constants
(see PREREG_W1 derivation table). No I1-derived numbers anywhere.

Modes:
  --selftest : mechanics on THROWAWAY seeds (ss 21999, discarded).
  --freeze   : write w1_freeze.json (specs+shas) + MANIFEST.json.
  --pools    : assert MANIFEST; generate pools; write w1_pools.json
               + w1_coverage.json (descriptive only).
regen_check(): rebuild pools from frozen specs; used by every
  later stage to prove pool artifacts == deterministic specs.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

from gen import (_agent, _dist, _pt, MIN_SEP, TRIES, REGIONS,
                 sample_config, sha256_file, write_json)

SEEDS = (301, 302, 303)
W1 = os.path.join(ROOT, "w1")

# --- PREREG_W1 section 2: every number traces to registered task
# constants (gen.py REGIONS / MIN_SEP). No I1 values here.
W_AGZ_LO, W_AGZ_HI = 35.0, 70.0  # union of REGIONS B spawn bands:
# B1 [50,70] B2 [40,60] B3 [40,55] Badv [35,45] (registered pursuit)
W_BGH_MIN = MIN_SEP  # distinct-entity separation; NO upper bound
W_SPD_LO, W_SPD_HI = 3.0, 3.5  # union of REGIONS G1 speeds
W_INNER = 50  # same inner loop as sample_config 'guarded'

# PREREG_W1 section 3: W1 seed block (no prior-stage reuse).
SS_DEMO, SS_ESPOOL, SS_GATE_G1 = 21100, 21101, 21102
SS_GATE_G2, SS_SING_GATE = 21103, 21104
SS_PLAN, SS_INIT, SS_ES = 21105, 21106, 21107
SS_WASTE = 21999  # throwaway selftest only; never pooled

CODE_FILES = ["gen_wide.py", "w1_collect.py", "w1_imitate.py",
              "w1_es.py", "w1_gate.py", "w1_h.py"]

# Prior datasets for disjointness (explicit frozen list).
PRIOR_FILES = [
    "data/train_meta_v1.json", "data/train_control_v1.json",
    "data/test_battery_v1.json", "v2conf/train_conflict_v1.json",
    "v2conf/eval_conf_v1.json", "v2views/eval_new_v1.json",
    "v2/g1g2_heldout_v1.json",
    "v2conf_diag/s7_acq7.json", "v2conf_diag/s8_acq8.json",
    "v2conf_diag/s8_acq9.json", "v2conf_diag/s9_acq10.json",
    "v2conf_diag/s10_acq11.json", "v2conf_diag/s10_acq12.json",
    "v2conf_diag/s11_acq13.json", "v2conf_diag/s11_acq14.json",
    "v2conf_diag/s13_acq15.json", "v2conf_diag/s13_acq16.json",
    "v2conf_diag/s14_acq17.json", "v2conf_diag/s15_acq18.json",
]


def wide_g1_cfg(rng):
    """Guarded-approach scenario over the general valid space."""
    for _ in range(TRIES):
        agent = _agent(rng)
        beacon = _pt(rng)
        if _dist(agent, beacon) < MIN_SEP:
            continue
        for _ in range(W_INNER):
            hp = _pt(rng)
            da = _dist(agent, hp)
            db = _dist(beacon, hp)
            if (W_AGZ_LO <= da <= W_AGZ_HI and db >= W_BGH_MIN):
                return {"agent": agent.tolist(),
                        "goal": {"pos": beacon.tolist()},
                        "hazard": {"pos": hp.tolist(), "speed": float(
                            rng.uniform(W_SPD_LO, W_SPD_HI))}}
    raise RuntimeError("wide-G1 sampling failed")


def kid_seeds(master, n):
    return [int(s.generate_state(1, dtype=np.uint32)[0])
            for s in np.random.SeedSequence(master).spawn(n)]


def geom(e):
    a = np.array(e["cfg"]["agent"], float)
    g = np.array(e["cfg"]["goal"]["pos"], float)
    h = np.array(e["cfg"]["hazard"]["pos"], float)
    vg, vh = g - a, h - a
    dag = float(np.linalg.norm(vg))
    dah = float(np.linalg.norm(vh))
    dgh = float(np.linalg.norm(g - h))
    cosang = float(np.dot(vg, vh) / max(dag * dah, 1e-12))
    return {"d_ag": dag, "d_ah": dah, "d_gh": dgh,
            "ang": float(np.arccos(np.clip(cosang, -1, 1))),
            "hspeed": float(e["cfg"]["hazard"]["speed"])}


def check_wide(e):
    c = e["cfg"]
    assert MIN_SEP <= _dist(c["agent"], c["goal"]["pos"])
    da = _dist(c["agent"], c["hazard"]["pos"])
    db = _dist(c["goal"]["pos"], c["hazard"]["pos"])
    assert W_AGZ_LO <= da <= W_AGZ_HI, da
    assert db >= W_BGH_MIN, db
    assert W_SPD_LO <= c["hazard"]["speed"] <= W_SPD_HI


def selftest():
    # Determinism: same seed -> identical cfg.
    a = wide_g1_cfg(np.random.default_rng(12345))
    b = wide_g1_cfg(np.random.default_rng(12345))
    assert a == b, "generator not deterministic"
    # Validity on 2000 throwaway samples (ss 21999, discarded).
    angs, dghs, dahs, dags = [], [], [], []
    for s in kid_seeds(SS_WASTE, 2000):
        e = {"cfg": wide_g1_cfg(np.random.default_rng(s))}
        check_wide(e)
        g = geom(e)
        angs.append(g["ang"])
        dghs.append(g["d_gh"])
        dahs.append(g["d_ah"])
        dags.append(g["d_ag"])
    angs, dghs = np.array(angs), np.array(dghs)
    print(f"selftest 2000 throwaway: all constraints hold; "
          f"d_gh [{dghs.min():.1f},{dghs.max():.1f}] "
          f"ang [{angs.min():.2f},{angs.max():.2f}] "
          f"d_ah [{np.min(dahs):.1f},{np.max(dahs):.1f}] "
          f"d_ag [{np.min(dags):.1f},{np.max(dags):.1f}]")
    # Full angular bins reachable (mechanics, not a pool assert).
    h, _ = np.histogram(angs, bins=6, range=(0, np.pi))
    assert all(h > 0), h
    print("selftest T1 (bounds/determinism/angular-reach): PASS")


def build_pools():
    """Deterministic pools from frozen specs. Returns dict of lists."""
    pools = {}
    # W1G1-DEMO / ESPOOL / GATE via wide generator.
    for key, ss, n, pre, width in (
            ("demo_g1", SS_DEMO, 18, "W1G1-D", 2),
            ("espool_g1", SS_ESPOOL, 12, "W1G1-E", 2),
            ("gate_g1", SS_GATE_G1, 200, "W1G1-G", 3)):
        arr = []
        for i, s in enumerate(kid_seeds(ss, n)):
            cfg = wide_g1_cfg(np.random.default_rng(s))
            e = {"id": f"{pre}{i:0{width}d}", "family": "G1",
                 "region": "wide", "seed": s, "cfg": cfg}
            check_wide(e)
            arr.append(e)
        pools[key] = arr
    # W1G2-GATE via FROZEN sample_config, region cycle (ACQ pattern).
    regs = ["G2a", "G2b", "G2c"]
    arr = []
    for i, s in enumerate(kid_seeds(SS_GATE_G2, 200)):
        cfg = sample_config("G2", regs[i % 3], np.random.default_rng(s))
        arr.append({"id": f"W1G2-G{i:03d}", "family": "G2",
                    "region": regs[i % 3], "seed": s, "cfg": cfg})
    pools["gate_g2"] = arr
    # W1SING-GATE via FROZEN sample_config, 20/family cycled.
    arr = []
    fam_regs = {"A": ["A1", "A2", "A3", "Aadv"],
                "B": ["B1", "B2", "B3", "Badv"],
                "C": ["C1", "C2", "C3", "Cadv"],
                "D": ["D1", "D2", "D3", "Dadv"]}
    plan = [(f, r) for f in "ABCD" for r in fam_regs[f] for _ in range(5)]
    assert len(plan) == 80
    for i, ((f, r), s) in enumerate(zip(plan, kid_seeds(SS_SING_GATE, 80))):
        cfg = sample_config(f, r, np.random.default_rng(s))
        arr.append({"id": f"W1S1-{f}{i:02d}", "family": f,
                    "region": r, "seed": s, "cfg": cfg})
    pools["sing_gate"] = arr
    return pools


def disjoint_check(pools):
    key = lambda e: (e["family"], e["seed"])
    cfgk = lambda e: (e["family"], json.dumps(e["cfg"], sort_keys=True))
    seen_k, seen_c = set(), set()
    for name, arr in pools.items():
        for e in arr:
            assert key(e) not in seen_k, ("W1 seed dup", name, e["id"])
            assert cfgk(e) not in seen_c, ("W1 cfg dup", name, e["id"])
            seen_k.add(key(e))
            seen_c.add(cfgk(e))
    n_prior = 0
    for rel in PRIOR_FILES:
        for e in json.load(open(os.path.join(ROOT, rel))):
            n_prior += 1
            assert key(e) not in seen_k, ("prior seed collision", rel, e["id"])
            assert cfgk(e) not in seen_c, ("prior cfg dup", rel, e["id"])
    return n_prior


def describe(pools):
    rep = {}
    for name in ("demo_g1", "espool_g1", "gate_g1"):
        G = [geom(e) for e in pools[name]]
        block = {}
        for f in ("d_ag", "d_ah", "d_gh", "ang", "hspeed"):
            v = np.array([g[f] for g in G])
            q = np.quantile(v, [0, .25, .5, .75, 1])
            block[f] = {"min": round(float(v.min()), 2),
                        "q25": round(float(q[1]), 2),
                        "med": round(float(q[2]), 2),
                        "q75": round(float(q[3]), 2),
                        "max": round(float(v.max()), 2)}
        h, edges = np.histogram([g["ang"] for g in G], bins=6,
                                range=(0, np.pi))
        block["ang_sextiles"] = {
            "edges": [round(float(x), 3) for x in edges],
            "counts": [int(x) for x in h]}
        hd, _ = np.histogram([g["d_gh"] for g in G], bins=8)
        block["d_gh_hist8"] = [int(x) for x in hd]
        rep[name] = {"n": len(G), **block}
    return rep


def do_freeze():
    for c in CODE_FILES:
        assert os.path.isfile(os.path.join(W1, c)), c
    n1 = json.load(open(os.path.join(ROOT, "v2conf_diag/n1_freeze.json")))
    n2 = json.load(open(os.path.join(ROOT, "v2conf_diag/n2_freeze.json")))
    g2tr = [i for i in n1["partition"]["train"] if i.startswith("M-G2-")]
    g2ev = [i for i in n1["partition"]["eval"] if i.startswith("M-G2-")]
    assert (len(g2tr), len(g2ev)) == (12, 6)
    fr = {
        "w1": "redesigned follow-up post-I1 (PREREG_W1.md); NOT blind prereg continuation",
        "thetas": list(SEEDS),
        "arch": {"gate": "14->32->4", "gate_dim": 612,
                 "skills": "pilot3 frozen x4", "total_dim": 7944},
        "generator": {"file": "w1/gen_wide.py",
                      "agz_band": [W_AGZ_LO, W_AGZ_HI],
                      "bgh_min": W_BGH_MIN, "bgh_max": None,
                      "speed": [W_SPD_LO, W_SPD_HI],
                      "tries": [TRIES, W_INNER],
                      "derivation": "PREREG_W1.md section 2 table"},
        "seed_block": {"demo_g1": SS_DEMO, "espool_g1": SS_ESPOOL,
                       "gate_g1": SS_GATE_G1, "gate_g2": SS_GATE_G2,
                       "sing_gate": SS_SING_GATE, "planner": SS_PLAN,
                       "imit_init": SS_INIT, "es": SS_ES,
                       "waste": SS_WASTE},
        "pools_spec": {"demo_g1": 18, "espool_g1": 12,
                       "gate_g1": 200, "gate_g2": 200, "sing_gate": 80},
        "partition_wide_g1": {"rule": "generation order: first 12 train, last 6 eval",
                              "train_idx": list(range(12)),
                              "eval_idx": list(range(12, 18))},
        "partition_g2_reuse": {"train": g2tr, "eval": g2ev,
                               "source": "n1_freeze.json (transcription verified here)"},
        "calendar_rule": "N2 rule: gen g: G1 idx (2g,2g+1)%12, G2 same, singles g%8; window 2G1+2G2+ABCD",
        "es_pool_g2": list(n2["pools"]["G2"]),
        "es_pool_singles": {f: list(n2["pools"][f]) for f in "ABCD"},
        "planner": {"source": "ref_plan.shoot/snap",
                    "H": 25, "K": 40, "blocks": [5, 5],
                    "seq_seed": SS_PLAN, "runs": 54,
                    "plan_order": "demo-major x thetas (N1 pattern)"},
        "imit": {"recipe": "N1b verbatim (S3: Xavier hidden, zero out, Adam 1e-3 full-batch, CE, 5000ep, final ckpt)",
                 "init_seeds": kid_seeds(SS_INIT, 3),
                 "d_mass_assert": True, "offline_only": True},
        "es": {"pop_pairs": 8, "candidates": 16, "sigma": 0.1,
               "alpha": 0.01, "generations": 25, "seed": SS_ES,
               "policy": "greedy argmax-4 gate -> greedy viewed skill",
               "fitness": "mean exact task return over 8 window eps"},
        "gate_bar": "W1-final mean >= 0.70 on gate_g1 AND gate_g2 (means over 3 thetas); else STOP",
        "h_ref": "each H >= 0.60; H-mean >= best-single + 0.40 (descriptive follow-up bars, NOT preregistered)",
        "budgets": {"collect": 1800, "imitate": 900, "es": 2400,
                    "gate": 900, "h": 1200},
        "forbidden": ["read/execute I1 ever", "open H without gate PASS",
                      "open gate/H twice", "post-result recipe/seed/pool change",
                      "checkpoint selection", "Modal"],
        "pins": {},
    }
    assert len(fr["es_pool_g2"]) == 12
    assert all(len(v) == 8 for v in fr["es_pool_singles"].values())
    pins = fr["pins"]
    for c in CODE_FILES:
        pins[c] = sha256_file(os.path.join(W1, c))
    pins["PREREG_W1.md"] = sha256_file(os.path.join(W1, "PREREG_W1.md"))
    for rel in ["data/train_meta_v1.json", "data/test_battery_v1.json",
                "v2conf_diag/n1_freeze.json", "v2conf_diag/n2_freeze.json",
                "v2conf_diag/n1_demos.json", "v2conf_diag/n1b_pairs.json",
                "v2conf_diag/ref_plan.py", "v2conf_diag/s4_closed.py",
                "src/tasks.py", "src/env.py", "src/models.py",
                "src/policies.py", "src/gen.py", "v2views/views_train.py"]:
        pins[rel] = sha256_file(os.path.join(ROOT, rel))
    for s in SEEDS:
        pins[f"pilot3_runs/P3_{s}_theta.npz"] = sha256_file(
            os.path.join(ROOT, f"pilot3_runs/P3_{s}_theta.npz"))
        pins[f"v2conf_runs/VC_{s}_theta.npz"] = sha256_file(
            os.path.join(ROOT, f"v2conf_runs/VC_{s}_theta.npz"))
    assert pins["data/train_meta_v1.json"] == \
        "7ebd5b9bd50b49dd8692080dfa294941cde652588da187914b04800f470aae85"
    # VC==P3 bit-exact re-verification (skills provenance).
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from models import split_modular
    for s in SEEDS:
        vc = np.load(os.path.join(
            ROOT, f"v2conf_runs/VC_{s}_theta.npz"))["theta"]
        p3 = np.load(os.path.join(
            ROOT, f"pilot3_runs/P3_{s}_theta.npz"))["theta"]
        assert len(p3) == 7944 and len(vc) == 7944, s
        assert all(np.array_equal(a, b) for a, b in
                   zip(split_modular(vc)[0], split_modular(p3)[0])), s
    fr["vc_eq_p3"] = "bit-exact 4/4 skills x3 thetas (re-verified at freeze)"
    write_json(os.path.join(W1, "w1_freeze.json"), fr)
    man = {c: pins[c] for c in CODE_FILES}
    man["PREREG_W1.md"] = pins["PREREG_W1.md"]
    man["w1_freeze.json"] = sha256_file(os.path.join(W1, "w1_freeze.json"))
    write_json(os.path.join(W1, "MANIFEST.json"), man)
    print("freeze written: w1_freeze.json + MANIFEST.json")
    for k, v in man.items():
        print(f"  {k} {v[:12]}")


def check_manifest():
    man = json.load(open(os.path.join(W1, "MANIFEST.json")))
    for rel, h in man.items():
        got = sha256_file(os.path.join(W1, rel))
        assert got == h, (rel, "FROZEN FILE ALTERED")
    return json.load(open(os.path.join(W1, "w1_freeze.json")))


def regen_check():
    """Rebuild pools from frozen specs; assert == artifacts on disk."""
    fr = check_manifest()
    pools = build_pools()
    disk = json.load(open(os.path.join(W1, "w1_pools.json")))
    for k in pools:
        assert pools[k] == disk["pools"][k], (k, "POOL MISMATCH")
    cal = build_calendar(fr, pools)
    assert cal == disk["calendar"], "CALENDAR MISMATCH"
    return fr, disk


def build_calendar(fr, pools):
    g1 = [e["id"] for e in pools["espool_g1"]]
    g2 = list(fr["es_pool_g2"])
    sg = fr["es_pool_singles"]
    assert len(g1) == 12 and len(g2) == 12
    cal = []
    for g in range(25):
        w = [g1[(2 * g) % 12], g1[(2 * g + 1) % 12],
             g2[(2 * g) % 12], g2[(2 * g + 1) % 12]]
        for f in "ABCD":
            w.append(sg[f][g % 8])
        cal.append(w)
    return cal


def do_pools():
    fr = check_manifest()
    for c in CODE_FILES:
        assert os.path.isfile(os.path.join(W1, c))
    pools = build_pools()
    n_prior = disjoint_check(pools)
    disk = {"pools": pools, "calendar": build_calendar(fr, pools)}
    write_json(os.path.join(W1, "w1_pools.json"), disk)
    cov = describe(pools)
    write_json(os.path.join(W1, "w1_coverage.json"), cov)
    print(f"pools written: " +
          ", ".join(f"{k}={len(v)}" for k, v in pools.items()))
    print(f"disjointness: 0 collisions vs {n_prior} prior entries "
          f"({len(PRIOR_FILES)} files) + internal 0")
    print("w1_pools.json sha256=" +
          sha256_file(os.path.join(W1, "w1_pools.json")))
    print("w1_coverage.json sha256=" +
          sha256_file(os.path.join(W1, "w1_coverage.json")))
    for name in ("demo_g1", "espool_g1", "gate_g1"):
        b = cov[name]
        print(f"{name}: d_gh [{b['d_gh']['min']},{b['d_gh']['max']}] "
              f"med={b['d_gh']['med']} | ang [{b['ang']['min']},"
              f"{b['ang']['max']}] med={b['ang']['med']} | "
              f"sextiles={b['ang_sextiles']['counts']}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "--selftest":
        selftest()
    elif mode == "--freeze":
        do_freeze()
    elif mode == "--pools":
        do_pools()
    else:
        raise SystemExit("usage: gen_wide.py --selftest|--freeze|--pools")
