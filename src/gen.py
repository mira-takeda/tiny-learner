"""Scenario space gen-v1, curriculum expansion, manifests. Numpy + stdlib only.

Generator schema (PREREG 3). All sampling is seeded and deterministic.
Train config streams: SeedSequence(6000 + arm). Test batteries: SeedSequence(2000 + idx).
"""

import hashlib
import json

import numpy as np

ARENA = 100.0
MIN_SEP = 10.0
PHASES = ["A", "B", "C", "D", "G1", "G2"]
TEST_BATTERIES = ["A", "B", "C", "D", "I1", "H1", "H2", "H3"]
TRAIN_PER_PHASE = 400
TEST_PER_BATTERY = 200

# Regions: the shared scenario space. Meta orders/emphasizes; control stratifies.
# Train G1 additionally constrains beacon-hazard >= 35 so I1's 12-25 range is
# held out (PREREG is silent on train beacon-hazard geometry; logged gap-fill).
REGIONS = {
    "A": {
        "A1": {"kind": "beacon_dist", "lo": 15.0, "hi": 40.0},
        "A2": {"kind": "beacon_dist", "lo": 40.0, "hi": 70.0},
        "A3": {"kind": "beacon_dist", "lo": 70.0, "hi": 110.0},
        "Aadv": {"kind": "beacon_near_wall", "wall_margin": 12.0, "lo": 50.0, "hi": 110.0},
    },
    "B": {
        "B1": {"kind": "pursuit", "speed_lo": 3.0, "speed_hi": 3.3,
               "spawn_lo": 50.0, "spawn_hi": 70.0},
        "B2": {"kind": "pursuit", "speed_lo": 3.3, "speed_hi": 3.7,
               "spawn_lo": 40.0, "spawn_hi": 60.0},
        "B3": {"kind": "pursuit", "speed_lo": 3.7, "speed_hi": 4.0,
               "spawn_lo": 40.0, "spawn_hi": 55.0},
        "Badv": {"kind": "pursuit", "speed_lo": 4.0, "speed_hi": 4.0,
                 "spawn_lo": 35.0, "spawn_hi": 45.0},
    },
    "C": {
        "C1": {"kind": "obj_zone_dist", "lo": 15.0, "hi": 45.0},
        "C2": {"kind": "obj_zone_dist", "lo": 45.0, "hi": 75.0},
        "C3": {"kind": "obj_zone_dist", "lo": 75.0, "hi": 105.0},
        "Cadv": {"kind": "obj_zone_dist", "lo": 90.0, "hi": 110.0},
    },
    "D": {
        "D1": {"kind": "gap", "gap_lo": 18.0, "gap_hi": 20.0,
               "wy_lo": 40.0, "wy_hi": 60.0},
        "D2": {"kind": "gap", "gap_lo": 16.0, "gap_hi": 18.0,
               "wy_lo": 40.0, "wy_hi": 60.0},
        "D3": {"kind": "gap", "gap_lo": 14.0, "gap_hi": 16.0,
               "wy_lo": 40.0, "wy_hi": 60.0},
        "Dadv": {"kind": "gap", "gap_lo": 14.0, "gap_hi": 14.0,
                 "wy_lo": 56.0, "wy_hi": 60.0},
    },
    "G1": {
        "G1a": {"kind": "guarded", "speed_lo": 3.0, "speed_hi": 3.2,
                "spawn_lo": 55.0, "spawn_hi": 70.0, "beacon_haz_min": 35.0},
        "G1b": {"kind": "guarded", "speed_lo": 3.2, "speed_hi": 3.5,
                "spawn_lo": 50.0, "spawn_hi": 60.0, "beacon_haz_min": 35.0},
        "G1adv": {"kind": "guarded", "speed_lo": 3.5, "speed_hi": 3.5,
                  "spawn_lo": 50.0, "spawn_hi": 52.0, "beacon_haz_min": 35.0},
    },
    "G2": {
        "G2a": {"kind": "fetch", "lo": 15.0, "hi": 45.0},
        "G2b": {"kind": "fetch", "lo": 45.0, "hi": 75.0},
        "G2c": {"kind": "fetch", "lo": 75.0, "hi": 105.0},
    },
}

TRIES = 600


def _pt(rng, lo=5.0, hi=95.0):
    return np.array([rng.uniform(lo, hi), rng.uniform(lo, hi)])


def _agent(rng):
    return np.array([rng.uniform(20, 80), rng.uniform(20, 80)])


def _edge_midpoint(rng):
    pts = [(50.0, 8.0), (50.0, 92.0), (8.0, 50.0), (92.0, 50.0)]
    return np.array(pts[int(rng.integers(4))])


def _dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def sample_config(family, region_id, rng):
    r = REGIONS[family][region_id]
    kind = r["kind"]
    if kind == "beacon_dist":
        for _ in range(TRIES):
            agent = _agent(rng)
            p = _pt(rng)
            if r["lo"] <= _dist(agent, p) <= r["hi"]:
                return {"agent": agent.tolist(), "goal": {"pos": p.tolist()}}
    elif kind == "beacon_near_wall":
        for _ in range(TRIES):
            agent = _agent(rng)
            p = _pt(rng, 2.0, 98.0)
            near_wall = min(p[0], p[1], 100 - p[0], 100 - p[1]) <= r["wall_margin"]
            if near_wall and r["lo"] <= _dist(agent, p) <= r["hi"]:
                return {"agent": agent.tolist(), "goal": {"pos": p.tolist()}}
    elif kind == "pursuit":
        for _ in range(TRIES):
            agent = _agent(rng)
            p = _pt(rng)
            d = _dist(agent, p)
            if r["spawn_lo"] <= d <= r["spawn_hi"]:
                return {"agent": agent.tolist(), "hazard": {
                    "pos": p.tolist(),
                    "speed": float(rng.uniform(r["speed_lo"], r["speed_hi"]))}}
    elif kind in ("obj_zone_dist", "fetch"):
        for _ in range(TRIES):
            agent = _agent(rng)
            zone = _edge_midpoint(rng)
            obj = _pt(rng)
            if (r["lo"] <= _dist(obj, zone) <= r["hi"]
                    and _dist(agent, obj) >= MIN_SEP):
                return {"agent": agent.tolist(), "obj": {"pos": obj.tolist()},
                        "zone": {"center": zone.tolist()}}
    elif kind == "gap":
        wy = float(rng.uniform(r["wy_lo"], r["wy_hi"]))
        gap_x = float(rng.uniform(15, 85))
        gap_w = float(rng.uniform(r["gap_lo"], r["gap_hi"]))
        agent = np.array([rng.uniform(20, 80), rng.uniform(10, wy - 8)])
        goal = np.array([gap_x, min(wy + 15.0, 92.0)])
        return {"agent": agent.tolist(), "goal": {"pos": goal.tolist()},
                "wall": {"y": wy, "gap_x": gap_x, "gap_w": gap_w}}
    elif kind == "guarded":
        for _ in range(TRIES):
            agent = _agent(rng)
            beacon = _pt(rng)
            if _dist(agent, beacon) < MIN_SEP:
                continue
            for _ in range(50):
                hp = _pt(rng)
                da = _dist(agent, hp)
                db = _dist(beacon, hp)
                if (r["spawn_lo"] <= da <= r["spawn_hi"]
                        and db >= r["beacon_haz_min"]):
                    return {"agent": agent.tolist(),
                            "goal": {"pos": beacon.tolist()},
                            "hazard": {"pos": hp.tolist(), "speed": float(
                                rng.uniform(r["speed_lo"], r["speed_hi"]))}}
    raise RuntimeError(f"sampling failed: {family}/{region_id}")


def sample_test(battery, rng):
    if battery == "A":
        for _ in range(TRIES):
            agent = _agent(rng)
            p = _pt(rng)
            if 15.0 <= _dist(agent, p) <= 110.0:
                return {"agent": agent.tolist(), "goal": {"pos": p.tolist()}}
    elif battery == "B":
        for _ in range(TRIES):
            agent = _agent(rng)
            p = _pt(rng)
            if 35.0 <= _dist(agent, p) <= 70.0:
                return {"agent": agent.tolist(), "hazard": {
                    "pos": p.tolist(), "speed": float(rng.uniform(3.0, 4.0))}}
    elif battery == "C":
        for _ in range(TRIES):
            agent = _agent(rng)
            zone = _edge_midpoint(rng)
            obj = _pt(rng)
            if (15.0 <= _dist(obj, zone) <= 110.0
                    and _dist(agent, obj) >= MIN_SEP):
                return {"agent": agent.tolist(), "obj": {"pos": obj.tolist()},
                        "zone": {"center": zone.tolist()}}
    elif battery == "D":
        wy = float(rng.uniform(40, 60))
        gap_x = float(rng.uniform(15, 85))
        gap_w = float(rng.uniform(14, 20))
        agent = np.array([rng.uniform(20, 80), rng.uniform(10, wy - 8)])
        return {"agent": agent.tolist(),
                "goal": {"pos": [gap_x, min(wy + 15.0, 92.0)]},
                "wall": {"y": wy, "gap_x": gap_x, "gap_w": gap_w}}
    elif battery == "I1":
        for _ in range(TRIES):
            agent = _agent(rng)
            beacon = _pt(rng)
            if _dist(agent, beacon) < MIN_SEP:
                continue
            for _ in range(80):
                hp = _pt(rng)
                da = _dist(agent, hp)
                db = _dist(beacon, hp)
                if 50.0 <= da <= 70.0 and 12.0 <= db <= 25.0:
                    return {"agent": agent.tolist(),
                            "goal": {"pos": beacon.tolist()},
                            "hazard": {"pos": hp.tolist(), "speed": float(
                                rng.uniform(3.0, 3.5))}}
    elif battery == "H1":
        for _ in range(TRIES):
            agent = _agent(rng)
            zone = _edge_midpoint(rng)
            obj = _pt(rng)
            if not (15.0 <= _dist(obj, zone) <= 105.0
                    and _dist(agent, obj) >= MIN_SEP):
                continue
            for _ in range(50):
                hp = _pt(rng)
                if 40.0 <= _dist(agent, hp) <= 70.0:
                    return {"agent": agent.tolist(), "obj": {"pos": obj.tolist()},
                            "zone": {"center": zone.tolist()},
                            "hazard": {"pos": hp.tolist(), "speed": 4.5}}
    elif battery == "H2":
        for _ in range(TRIES):
            agent = _agent(rng)
            zone = _edge_midpoint(rng)
            obj = _pt(rng)
            if not (15.0 <= _dist(obj, zone) <= 105.0
                    and _dist(agent, obj) >= MIN_SEP):
                continue
            for _ in range(80):
                hp = _pt(rng)
                if (15.0 <= _dist(obj, hp) <= 25.0
                        and _dist(agent, hp) >= MIN_SEP):
                    return {"agent": agent.tolist(), "obj": {"pos": obj.tolist()},
                            "zone": {"center": zone.tolist()},
                            "hazard": {"pos": hp.tolist(), "speed": 4.5}}
    elif battery == "H3":
        wy = float(rng.uniform(40, 60))
        gap_x = float(rng.uniform(15, 85))
        gap_w = float(rng.uniform(10, 12))
        up = bool(rng.integers(2))
        edges = [(50.0, 92.0), (92.0, 50.0), (8.0, 50.0)] if up else \
            [(50.0, 8.0), (92.0, 50.0), (8.0, 50.0)]
        for _ in range(TRIES):
            zone = np.array(edges[int(rng.integers(len(edges)))])
            ok_zone = (zone[1] > wy + 5) if up else (zone[1] < wy - 5)
            if not ok_zone:
                continue
            oy = rng.uniform(wy + 8, 90) if up else rng.uniform(10, wy - 8)
            obj = np.array([rng.uniform(15, 85), oy])
            ay = rng.uniform(wy + 8, 88) if up else rng.uniform(12, wy - 8)
            agent = np.array([rng.uniform(20, 80), ay])
            if _dist(agent, obj) >= MIN_SEP:
                return {"agent": agent.tolist(), "obj": {"pos": obj.tolist()},
                        "zone": {"center": zone.tolist()},
                        "wall": {"y": wy, "gap_x": gap_x, "gap_w": gap_w}}
    raise RuntimeError(f"test sampling failed: {battery}")


def expand_meta(spec):
    """Compact Meta stage spec -> ordered [(phase, region)] slots."""
    slots = []
    for phase in PHASES:
        for stage in spec["stages"][phase]:
            if stage["region"] not in REGIONS[phase]:
                raise ValueError(f"unknown region {stage['region']}")
            slots.extend([(phase, stage["region"])] * stage["n"])
    return slots


def expand_control(seed=31337):
    """Stratified-uniform schedule over the same regions. Fixed order seed."""
    rng = np.random.default_rng(seed)
    slots = []
    for phase in PHASES:
        regs = sorted(REGIONS[phase])
        per, rem = divmod(TRAIN_PER_PHASE, len(regs))
        bag = []
        for i, reg in enumerate(regs):
            bag.extend([(phase, reg)] * (per + (1 if i < rem else 0)))
        order = rng.permutation(len(bag))
        slots.extend([bag[i] for i in order])
    return slots


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sig(entry):
    """Canonical signature for leakage comparison (6dp rounding)."""
    def rnd(o):
        if isinstance(o, float):
            return round(o, 6)
        if isinstance(o, dict):
            return {k: rnd(v) for k, v in sorted(o.items())}
        if isinstance(o, list):
            return [rnd(v) for v in o]
        return o

    return json.dumps({"family": entry["family"], "cfg": rnd(entry["cfg"])},
                      sort_keys=True)
