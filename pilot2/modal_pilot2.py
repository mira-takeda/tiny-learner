"""Modal launcher for pilot2 (pilot2-only).

Operational wrapper only: bakes frozen src/data + pilot snapshot
(reused read-only) + pilot2 snapshot into the image (copy=True) and
launches the 3 paired seeds (301-303, sequential A->B->C(800)->D) as
ONE batch call via experimental_spawn_map (validated in client 1.6.2).
Each seed runs the pilot2 trainer in a CLEAN SUBPROCESS (still
required: frozen run_battery asserts no-net and the Modal worker
preloads net modules). Results go to a SEPARATE volume
(tiny-learner-pilot2); pilot/, pilot_runs/, C6 and all other records
untouched.

DO NOT INVOKE without explicit user GO.
"""

import modal

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
APP = "tiny-learner-pilot2"
VOLUME = "tiny-learner-pilot2"

PILOT2_SEEDS = [301, 302, 303]

image = (modal.Image.debian_slim(python_version="3.14")
         .pip_install("numpy==2.5.4")
         .add_local_dir(f"{ROOT}/src", remote_path="/root/tl/src", copy=True)
         .add_local_dir(f"{ROOT}/data", remote_path="/root/tl/data", copy=True)
         .add_local_dir(f"{ROOT}/pilot", remote_path="/root/tl/pilot",
                        copy=True)
         .add_local_dir(f"{ROOT}/pilot2", remote_path="/root/tl/pilot2",
                        copy=True))
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)

app = modal.App(APP)
BASE = dict(image=image, cpu=1.0, memory=1024)


@app.function(**BASE, volumes={"/vol": vol}, timeout=86400)
def run_pilot2(seed):
    """DO NOT INVOKE without explicit user GO. Runs one pilot2 seed via
    the pilot2 trainer in a clean subprocess (see module docstring)."""
    import json
    import os
    import subprocess
    import sys

    sys.path.insert(0, "/root/tl/pilot2")
    import pilot2_train

    assert seed in pilot2_train.PILOT2_SEEDS
    out_json = f"/vol/pilot2/runs/P2_{seed}.json"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    cmd = [sys.executable, "/root/tl/pilot2/pilot2_train.py",
           "--seed", str(seed),
           "--data", "/root/tl/data", "--out", "/vol/pilot2/runs"]
    subprocess.run(cmd, check=True, env=env, timeout=86000)
    vol.commit()
    log = json.load(open(out_json))
    return {"seed": seed, "steps": log["total_steps"],
            "phase_end": {f: log["heldout_after"][f"post_{f}"][f]["sampled"]["sr"]
                          for f in ["A", "B", "C", "D"]}}


@app.local_entrypoint()
def main(only=""):
    seeds = [int(s) for s in only.split(",") if s] if only else list(PILOT2_SEEDS)
    assert all(s in PILOT2_SEEDS for s in seeds), seeds
    print(f"creating ONE pilot2 batch call with {len(seeds)} input(s) "
          f"via experimental_spawn_map: {seeds}")
    fc = run_pilot2.experimental_spawn_map(seeds)
    print(f"pilot2 batch call created: {fc.object_id} "
          f"({len(seeds)} inputs running detached; results land in "
          f"/vol/pilot2/runs on volume {VOLUME})")
