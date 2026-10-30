"""Modal launcher for the v2-pilot (pilot-only).

Operational wrapper only: bakes frozen src/data + pilot snapshot into the
image (copy=True) and launches the 3 pilot seeds (301-303, sequential
A->B->C->D modular-frozen with one-hot fitness) as ONE batch call via
experimental_spawn_map. Each seed runs the pilot trainer in a CLEAN
SUBPROCESS (still required: frozen run_battery asserts no-net and the
Modal worker preloads net modules). Results go to a SEPARATE volume
(tiny-learner-pilot); runs_modal/, v1 records, and diag/ untouched.

DO NOT INVOKE without explicit user GO.
"""

import modal

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
APP = "tiny-learner-pilot"
VOLUME = "tiny-learner-pilot"

PILOT_SEEDS = [301, 302, 303]

image = (modal.Image.debian_slim(python_version="3.14")
         .pip_install("numpy==2.5.4")
         .add_local_dir(f"{ROOT}/src", remote_path="/root/tl/src", copy=True)
         .add_local_dir(f"{ROOT}/data", remote_path="/root/tl/data", copy=True)
         .add_local_dir(f"{ROOT}/pilot", remote_path="/root/tl/pilot",
                        copy=True))
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)

app = modal.App(APP)
BASE = dict(image=image, cpu=1.0, memory=1024)


@app.function(**BASE, volumes={"/vol": vol}, timeout=86400)
def run_pilot(seed):
    """DO NOT INVOKE without explicit user GO. Runs one pilot seed via
    the pilot trainer in a clean subprocess (see module docstring)."""
    import json
    import os
    import subprocess
    import sys

    sys.path.insert(0, "/root/tl/pilot")
    import pilot_train

    assert seed in pilot_train.PILOT_SEEDS
    out_json = f"/vol/pilot/runs/{seed}.json"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    cmd = [sys.executable, "/root/tl/pilot/pilot_train.py",
           "--seed", str(seed),
           "--data", "/root/tl/data", "--out", "/vol/pilot/runs"]
    subprocess.run(cmd, check=True, env=env, timeout=86000)
    vol.commit()
    log = json.load(open(out_json))
    return {"seed": seed, "steps": log["total_steps"],
            "phase_end": {f: log["heldout_after"][f"post_{f}"][f]["sampled"]["sr"]
                          for f in ["A", "B", "C", "D"]}}


@app.local_entrypoint()
def main(only=""):
    seeds = [int(s) for s in only.split(",") if s] if only else list(PILOT_SEEDS)
    assert all(s in PILOT_SEEDS for s in seeds), seeds
    print(f"creating ONE v2-pilot batch call with {len(seeds)} input(s) "
          f"via experimental_spawn_map: {seeds}")
    fc = run_pilot.experimental_spawn_map(seeds)
    print(f"v2-pilot batch call created: {fc.object_id} "
          f"({len(seeds)} inputs running detached; results land in "
          f"/vol/pilot/runs on volume {VOLUME})")
