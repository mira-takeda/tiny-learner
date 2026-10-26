"""Leakage + structural audits. Exit 0 = PASS, 1 = FAIL."""

import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from gen import (PHASES, REGIONS, TEST_BATTERIES, TRAIN_PER_PHASE,
                 TEST_PER_BATTERY, sha256_file, sig)

DATA = os.path.join(__file__.rsplit("/", 1)[0], "..", "data")
SRC = os.path.join(__file__.rsplit("/", 1)[0])


def feat(cfg):
    def xy(d, key, sub):
        e = d.get(key)
        if e is None:
            return [-1.0, -1.0]
        p = e[sub]
        return [p[0] / 100.0, p[1] / 100.0]

    v = []
    v += [cfg["agent"][0] / 100.0, cfg["agent"][1] / 100.0]
    v += xy(cfg, "goal", "pos")
    v += xy(cfg, "hazard", "pos")
    hz = cfg.get("hazard")
    v += [hz["speed"] / 5.0 if hz else -1.0]
    v += xy(cfg, "obj", "pos")
    v += xy(cfg, "zone", "center")
    w = cfg.get("wall")
    v += [w["y"] / 100.0, w["gap_x"] / 100.0, w["gap_w"] / 20.0] if w else [-1.0] * 3
    return np.array(v)


class Audit:
    def __init__(self):
        self.checks = []

    def check(self, name, ok, detail=""):
        self.checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")

    @property
    def verdict(self):
        return all(c["ok"] for c in self.checks)


def main():
    au = Audit()
    train_m = json.load(open(f"{DATA}/train_meta_v1.json"))
    train_c = json.load(open(f"{DATA}/train_control_v1.json"))
    battery = json.load(open(f"{DATA}/test_battery_v1.json"))

    # counts + equal budgets
    for name, tr in (("meta", train_m), ("control", train_c)):
        au.check(f"train-{name}-total", len(tr) == 6 * TRAIN_PER_PHASE, len(tr))
        for ph in PHASES:
            n = sum(1 for e in tr if e["family"] == ph)
            au.check(f"train-{name}-{ph}", n == TRAIN_PER_PHASE, n)
    au.check("budget-equal-configs", len(train_m) == len(train_c),
             f"{len(train_m)} vs {len(train_c)}")
    for b in TEST_BATTERIES:
        n = sum(1 for e in battery if e["battery"] == b)
        au.check(f"test-{b}", n == TEST_PER_BATTERY, n)

    # battery seeds follow SeedSequence(2000+idx) scheme
    for i, b in enumerate(TEST_BATTERIES):
        expect = [int(s.generate_state(1, dtype=np.uint32)[0])
                  for s in np.random.SeedSequence(2000 + i).spawn(TEST_PER_BATTERY)]
        got = [e["seed"] for e in battery if e["battery"] == b]
        au.check(f"seeds-{b}", got == expect)

    # exact-match leakage (hard gate)
    train_sigs = {sig(e) for e in train_m} | {sig(e) for e in train_c}
    hits = [e["id"] for e in battery if sig(e) in train_sigs]
    au.check("leakage-exact-zero", not hits, f"hits={hits[:5]}")

    # near-duplicate distances (report; PREREG sets no numeric bar)
    TR = np.array([feat(e["cfg"]) for e in train_m + train_c])
    mins = []
    for e in battery:
        d = np.sqrt(((TR - feat(e["cfg"])) ** 2).sum(axis=1)).min()
        mins.append(float(d))
    mins = np.array(mins)
    au.check("distance-report", True,
             f"min={mins.min():.4f} p5={np.percentile(mins,5):.4f} "
             f"median={np.median(mins):.4f}")

    # held-out subrange compliance
    def sub(cfg, key, sub):
        return cfg[key][sub]

    h1 = [e for e in battery if e["battery"] == "H1"]
    au.check("H1-speed-4.5", all(sub(e["cfg"], "hazard", "speed") == 4.5 for e in h1))
    au.check("H1-spawn-range", all(
        40.0 <= float(np.linalg.norm(np.array(e["cfg"]["agent"])
                      - np.array(sub(e["cfg"], "hazard", "pos")))) <= 70.0 for e in h1))
    g1speeds = [sub(e["cfg"], "hazard", "speed") for e in train_m + train_c
                if e["family"] == "G1"]
    au.check("G1-train-max-speed-below-4.5", max(g1speeds) < 4.5, max(g1speeds))
    h2 = [e["id"] for e in battery if e["battery"] == "H2"]
    au.check("H2-proximity-15-25", all(
        15.0 <= float(np.linalg.norm(
            np.array(e["cfg"]["obj"]["pos"]) - np.array(e["cfg"]["hazard"]["pos"])))
        <= 25.0 for e in battery if e["battery"] == "H2"), len(h2))
    h3 = [e for e in battery if e["battery"] == "H3"]
    au.check("H3-gap-10-12", all(
        10.0 <= e["cfg"]["wall"]["gap_w"] <= 12.0 for e in h3))
    dgaps = [e["cfg"]["wall"]["gap_w"] for e in train_m + train_c if e["family"] == "D"]
    au.check("D-train-gap-min-14", min(dgaps) >= 14.0, min(dgaps))
    i1 = [e for e in battery if e["battery"] == "I1"]
    au.check("I1-beacon-haz-12-25", all(
        12.0 <= float(np.linalg.norm(
            np.array(e["cfg"]["goal"]["pos"]) - np.array(e["cfg"]["hazard"]["pos"])))
        <= 25.0 for e in i1))
    g1bh = [float(np.linalg.norm(np.array(e["cfg"]["goal"]["pos"])
                                 - np.array(e["cfg"]["hazard"]["pos"])))
            for e in train_m + train_c if e["family"] == "G1"]
    au.check("G1-train-beacon-haz-min-35", min(g1bh) >= 35.0, f"{min(g1bh):.2f}")

    # curriculum specs reference region IDs only (no sampled values)
    for path in ("curriculum_meta_v1.json", "curriculum_control_v1.json"):
        spec = json.load(open(f"{DATA}/{path}"))
        ok = True
        for ph in PHASES:
            for st in spec["stages"][ph]:
                ok &= set(st) == {"region", "n"} and st["region"] in REGIONS[ph]
        au.check(f"curriculum-schema-{path}", ok)

    # scenario export matches code
    scen = json.load(open(f"{DATA}/scenario_space_v1.json"))
    au.check("scenario-matches-code", scen["regions"] == REGIONS)

    # static import audit: no network in src
    pat = re.compile(r"^\s*(import|from)\s+(socket|urllib|requests|ssl|http|httpx)\b")
    bad = []
    for fn in sorted(os.listdir(SRC)):
        if fn.endswith(".py"):
            for i, line in enumerate(open(f"{SRC}/{fn}"), 1):
                if pat.match(line):
                    bad.append(f"{fn}:{i}")
    au.check("no-network-imports", not bad, bad[:3])

    print("AUDIT", "PASS" if au.verdict else "FAIL")
    json.dump(au.checks, open(f"{DATA}/audit_report.json", "w"), indent=1)
    return 0 if au.verdict else 1


if __name__ == "__main__":
    sys.exit(main())
