"""Ordered dataset pipeline. Run in two stages to preserve blinding order:

  stage curricula : export scenario, validate + expand curricula, sample train
                    manifests. Snapshot + record hashes BEFORE stage data.
  stage data      : sample frozen test battery + manifest.

Usage: python3.14 mkdata.py --stage curricula|data
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from gen import (PHASES, REGIONS, TEST_BATTERIES, TRAIN_PER_PHASE,
                 TEST_PER_BATTERY, expand_meta, expand_control, sample_config,
                 sample_test, write_json, sha256_file)

ROOT = os.path.join(__file__.rsplit("/", 1)[0], "..")
DATA = os.path.join(ROOT, "data")


def stage_curricula():
    os.makedirs(DATA, exist_ok=True)
    write_json(f"{DATA}/scenario_space_v1.json",
               {"version": "gen-v1", "regions": REGIONS,
                "constants": {"arena": 100.0, "min_sep": 10.0,
                              "train_per_phase": TRAIN_PER_PHASE}})
    meta_spec = json.load(open(f"{DATA}/curriculum_meta_v1.json"))
    for ph in PHASES:
        total = sum(st["n"] for st in meta_spec["stages"][ph])
        assert total == TRAIN_PER_PHASE, (ph, total)
    slots = {"meta": expand_meta(meta_spec), "control": expand_control()}
    write_json(f"{DATA}/curriculum_control_v1.json",
               {"version": "curriculum-control-v1",
                "schedule": "stratified-uniform, order seed 31337",
                "stages": {ph: [{"region": r, "n": sum(
                    1 for p, q in slots["control"] if p == ph and q == r)}
                    for r in sorted(REGIONS[ph])] for ph in PHASES}})
    for i, arm in enumerate(("meta", "control")):
        seeds = [int(s.generate_state(1, dtype=np.uint32)[0])
                 for s in np.random.SeedSequence(6000 + i).spawn(len(slots[arm]))]
        entries = []
        for k, ((phase, region), seed) in enumerate(zip(slots[arm], seeds)):
            cfg = sample_config(phase, region, np.random.default_rng(seed))
            entries.append({"id": f"{'M' if arm == 'meta' else 'C'}-{phase}-{k:04d}",
                            "family": phase, "region": region, "seed": seed,
                            "cfg": cfg})
        write_json(f"{DATA}/train_{arm}_v1.json", entries)
    for fn in ("scenario_space_v1.json", "curriculum_meta_v1.json",
               "curriculum_control_v1.json", "train_meta_v1.json",
               "train_control_v1.json"):
        print(f"{fn} {sha256_file(f'{DATA}/{fn}')}")


def stage_data():
    entries = []
    for i, batt in enumerate(TEST_BATTERIES):
        seeds = [int(s.generate_state(1, dtype=np.uint32)[0])
                 for s in np.random.SeedSequence(2000 + i).spawn(TEST_PER_BATTERY)]
        for k, seed in enumerate(seeds):
            cfg = sample_test(batt, np.random.default_rng(seed))
            entries.append({"id": f"T-{batt}-{k:04d}", "battery": batt,
                            "family": batt, "seed": seed, "cfg": cfg})
    write_json(f"{DATA}/test_battery_v1.json", entries)
    manifest = {}
    for fn in ("scenario_space_v1.json", "curriculum_meta_v1.json",
               "curriculum_control_v1.json", "train_meta_v1.json",
               "train_control_v1.json", "test_battery_v1.json"):
        manifest[fn] = sha256_file(f"{DATA}/{fn}")
    write_json(f"{DATA}/MANIFEST.json", manifest)
    for fn, h in manifest.items():
        print(f"{fn} {h}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["curricula", "data"], required=True)
    args = ap.parse_args()
    if args.stage == "curricula":
        stage_curricula()
    else:
        stage_data()


if __name__ == "__main__":
    main()
